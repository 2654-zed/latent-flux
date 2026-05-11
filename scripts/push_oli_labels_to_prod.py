"""Push local oli_labels HIGH+LOW rows up to production via railway ssh.

Production's `oli_labels` table is empty because the Correction #20 backfill
populated only the local cache. The patched `approval_drain_monitor.py` (added
2026-05-10) queries `oli_labels` at runtime to decide which deployer contracts
should not promote drain detections. Until prod has the suppression set in its
own `oli_labels`, the runtime gate is a no-op on prod even after redeploy.

This script pushes the HIGH+LOW severity rows up. `self-confirming` rows are
excluded by design — those are correctly-flagged adversarial deployers whose
drains should remain detected. `none` rows are excluded — they're cache misses
with no runtime effect.

Idempotent: uses `INSERT OR REPLACE` keyed on `(address, chain_id)`. Safe to
re-run.

Precondition: railway CLI linked to blockchain@stellar-embrace.

Usage:
    python scripts/push_oli_labels_to_prod.py --verify-only
    python scripts/push_oli_labels_to_prod.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"


def _resolve_railway() -> tuple[str, list[str]]:
    railway = (
        shutil.which("railway.exe")
        or shutil.which("railway.cmd")
        or shutil.which("railway")
    )
    if not railway:
        raise SystemExit("railway CLI not found in PATH")
    if os.name == "nt" and railway.lower().endswith((".cmd", ".bat")):
        return railway, ["cmd.exe", "/c", railway]
    return railway, [railway]


def _check_linked() -> None:
    _, prefix = _resolve_railway()
    r = subprocess.run(prefix + ["status"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "blockchain" not in r.stdout or "stellar-embrace" not in r.stdout:
        raise SystemExit(
            "railway CLI not linked to blockchain@stellar-embrace. Run:\n"
            "  railway link --project blockchain && railway service stellar-embrace"
        )


def _local_rows() -> list[dict]:
    if not DB_PATH.exists():
        raise SystemExit(f"local DB not found at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM oli_labels WHERE severity IN ('HIGH', 'LOW') ORDER BY address"
        )
    ]
    conn.close()
    return rows


def _build_apply_script(rows: list[dict]) -> str:
    # Tuples are (address, chain_id, tags_json, tag_count, primary_entity,
    # primary_tag_name, severity, fetched_at).
    payload = [
        (
            r["address"],
            r["chain_id"],
            r["tags_json"],
            r["tag_count"],
            r["primary_entity"],
            r["primary_tag_name"],
            r["severity"],
            r["fetched_at"],
        )
        for r in rows
    ]
    rows_json = json.dumps(payload)
    # Use json.loads inside the remote script to avoid string-quoting issues.
    return f'''import sqlite3, json
DB="/app/surveillance/data/surveillance.db"
ROWS=json.loads({rows_json!r})
conn=sqlite3.connect(DB,timeout=30);c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS oli_labels (address TEXT NOT NULL, chain_id INTEGER NOT NULL, tags_json TEXT, tag_count INTEGER, primary_entity TEXT, primary_tag_name TEXT, severity TEXT, fetched_at TEXT, PRIMARY KEY (address, chain_id))")
c.execute("CREATE INDEX IF NOT EXISTS idx_oli_severity ON oli_labels(severity)")
c.execute("CREATE INDEX IF NOT EXISTS idx_oli_primary_entity ON oli_labels(primary_entity)")
n=0
for r in ROWS:
 c.execute("INSERT OR REPLACE INTO oli_labels (address, chain_id, tags_json, tag_count, primary_entity, primary_tag_name, severity, fetched_at) VALUES (?,?,?,?,?,?,?,?)", r)
 n+=1
conn.commit()
hi=c.execute("SELECT COUNT(*) FROM oli_labels WHERE severity='HIGH'").fetchone()[0]
lo=c.execute("SELECT COUNT(*) FROM oli_labels WHERE severity='LOW'").fetchone()[0]
tot=c.execute("SELECT COUNT(*) FROM oli_labels").fetchone()[0]
conn.close()
print(f"upserted={{n}} prod_total={{tot}} HIGH={{hi}} LOW={{lo}}")
'''


VERIFY_SCRIPT = '''import sqlite3
DB="/app/surveillance/data/surveillance.db"
try:
 c=sqlite3.connect(DB,timeout=30).cursor()
 tot=c.execute("SELECT COUNT(*) FROM oli_labels").fetchone()[0]
 hi=c.execute("SELECT COUNT(*) FROM oli_labels WHERE severity='HIGH'").fetchone()[0]
 lo=c.execute("SELECT COUNT(*) FROM oli_labels WHERE severity='LOW'").fetchone()[0]
 sc=c.execute("SELECT COUNT(*) FROM oli_labels WHERE severity='self-confirming'").fetchone()[0]
 nn=c.execute("SELECT COUNT(*) FROM oli_labels WHERE severity='none' OR severity IS NULL").fetchone()[0]
 print(f"prod oli_labels total={tot} HIGH={hi} LOW={lo} self-confirming={sc} none={nn}")
except sqlite3.OperationalError as e:
 print(f"prod oli_labels: table missing ({e})")
'''


def _run_remote(script: str) -> tuple[int, str, str]:
    _, prefix = _resolve_railway()
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    bootstrap = f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\'))"'
    if len(bootstrap) > 8000:
        raise SystemExit(
            f"bootstrap is {len(bootstrap)} chars, exceeds Windows cmd.exe limit. "
            f"Reduce payload."
        )
    cmd = prefix + ["ssh", bootstrap]
    r = subprocess.run(cmd, capture_output=True, text=False, timeout=120)
    return (
        r.returncode,
        r.stdout.decode("utf-8", errors="replace"),
        r.stderr.decode("utf-8", errors="replace"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--verify-only", action="store_true",
        help="Probe prod oli_labels state; do not write",
    )
    args = ap.parse_args()
    _check_linked()
    if args.verify_only:
        print("[push-oli] verify-only")
        rc, out, err = _run_remote(VERIFY_SCRIPT)
        print(out)
        if err.strip():
            print("--- stderr ---")
            print(err[-2000:])
        return rc
    rows = _local_rows()
    print(f"[push-oli] local HIGH+LOW rows: {len(rows)}")
    if not rows:
        print("[push-oli] nothing to push")
        return 0
    script = _build_apply_script(rows)
    print(f"[push-oli] remote script: {len(script)} bytes")
    rc, out, err = _run_remote(script)
    print(out)
    if err.strip():
        print("--- stderr ---")
        print(err[-2000:])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
