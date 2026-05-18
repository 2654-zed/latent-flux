"""Add 0xa27bba42 + its deployer 0x7bf3269c to the watchlist as HIGH-priority
pre-discharge surveillance targets.

Why: 2026-05-17 fresh-suspected review surfaced 0xa27bba42 as the largest
fresh pre-discharge candidate in the corpus:
  - Deployed 2026-05-17T17:17Z on Base
  - Bytecode contains SELFDESTRUCT (Tier B "deferred threat ELEVATED")
  - 264 approvals from 264 victims in the first 7 hours
  - 122 in hour 1 alone — bot-rush at deploy signature
  - No mainnet identity (L2-native, different from 0x80b12bd0 cover)
  - No traceable funder yet (auto_funder_tracer empty)
  - Unique bytecode (no sibling contracts)

Q-002 (approval-spike detector) filters by watchlisted operators:
    JOIN watchlist w ON w.address = c.deployer_address AND w.active = 1
So adding 0x7bf3269c to watchlist makes 0xa27bba42 surveilled by Q-002
automatically (4x daily in production per the ANALYSIS_JOBS schedule).

Writes both locally and to production via railway ssh.

CLI:
    python scripts/add_a27bba42_watchlist.py             # local only
    python scripts/add_a27bba42_watchlist.py --prod      # local + production
"""
from __future__ import annotations
import argparse, sqlite3, base64, subprocess, shutil
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEPLOYER = "0x7bf3269c608f09bd8c1eaff34f2f210d467bd8f2"
CONTRACT = "0xa27bba42e0e1d3db503cf0d3be39f23db64781a3"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

DEPLOYER_NAME = "pristine_solo_operator_pre_discharge_a27bba42"
DEPLOYER_REASON = (
    "Pristine solo operator. Deployed 0xa27bba42 on Base 2026-05-17T17:17Z "
    "with SELFDESTRUCT in bytecode (Tier B deferred ELEVATED). Absorbed 264 "
    "victims in first 7 hours (122 in hour 1 = bot-rush). L2-native, no "
    "mainnet identity, no traceable funder. Q-002 surveils for discharge. "
    "Surfaced 2026-05-18 fresh-suspected review (post-2026-05-15 sync)."
)
CONTRACT_NAME = "pre_discharge_bait_a27bba42"
CONTRACT_REASON = (
    "Bait contract by 0x7bf3269c. SELFDESTRUCT (Tier B deferred ELEVATED). "
    "264 victims approved in 7 hours after deploy. No drain yet. Code hash "
    "f6c6aa16...; unique to this contract. Monitor via Q-002 daily."
)


def insert(conn: sqlite3.Connection, address: str, kind: str, name: str, reason: str) -> str:
    """Insert one watchlist row. Returns 'inserted' | 'already_exists' | error str."""
    existing = conn.execute(
        "SELECT id, entity_name, priority, active FROM watchlist WHERE address=?",
        (address,)
    ).fetchone()
    if existing:
        return f"already_exists id={existing[0]} name={existing[1]} priority={existing[2]} active={existing[3]}"
    conn.execute(
        """INSERT INTO watchlist
           (address, address_type, entity_name, watch_reason, priority,
            added_date, last_seen_date, hit_count, active)
           VALUES (?, ?, ?, ?, 'HIGH', ?, NULL, 0, 1)""",
        (address, kind, name, reason, NOW),
    )
    return "inserted"


def apply_local() -> tuple[str, str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        r1 = insert(conn, DEPLOYER, "deployer", DEPLOYER_NAME, DEPLOYER_REASON)
        r2 = insert(conn, CONTRACT, "contract", CONTRACT_NAME, CONTRACT_REASON)
        conn.commit()
    finally:
        conn.close()
    return r1, r2


def apply_prod() -> tuple[int, str, str]:
    """Apply via railway ssh using a compact inline insert script."""
    inner = (
        "import sqlite3, sys\n"
        "DB = '/app/surveillance/data/surveillance.db'\n"
        f"NOW = {NOW!r}\n"
        "conn = sqlite3.connect(DB)\n"
        "def ins(addr, kind, name, reason):\n"
        "    cur = conn.execute('SELECT id, entity_name, priority FROM watchlist WHERE address=?', (addr,))\n"
        "    r = cur.fetchone()\n"
        "    if r:\n"
        "        print(f'EXISTS {addr} id={r[0]} name={r[1]}')\n"
        "        return\n"
        "    conn.execute(\"INSERT INTO watchlist (address, address_type, entity_name, watch_reason, priority, added_date, last_seen_date, hit_count, active) VALUES (?, ?, ?, ?, 'HIGH', ?, NULL, 0, 1)\", (addr, kind, name, reason, NOW))\n"
        "    print(f'INSERTED {addr} {name}')\n"
        f"ins({DEPLOYER!r}, 'deployer', {DEPLOYER_NAME!r}, {DEPLOYER_REASON!r})\n"
        f"ins({CONTRACT!r}, 'contract', {CONTRACT_NAME!r}, {CONTRACT_REASON!r})\n"
        "conn.commit()\n"
        "conn.close()\n"
        "print('DONE')\n"
    )
    b64 = base64.b64encode(inner.encode()).decode()
    bootstrap = (
        "python3 -c \"import base64; exec(base64.b64decode('" + b64 + "'))\""
    )
    exe = shutil.which("railway.cmd") or shutil.which("railway")
    if not exe:
        return (-1, "", "railway CLI not found in PATH")
    cmd = ["cmd.exe", "/c", exe, "ssh", bootstrap]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prod", action="store_true",
                    help="also apply to production via railway ssh")
    args = ap.parse_args()

    print(f"=== LOCAL apply ({DB_PATH}) ===")
    r1, r2 = apply_local()
    print(f"  deployer {DEPLOYER}: {r1}")
    print(f"  contract {CONTRACT}: {r2}")

    if args.prod:
        print()
        print("=== PROD apply (via railway ssh) ===")
        rc, out, err = apply_prod()
        print(f"  rc={rc}")
        print(f"  stdout: {out.strip()}")
        if err.strip():
            print(f"  stderr: {err.strip()[-500:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
