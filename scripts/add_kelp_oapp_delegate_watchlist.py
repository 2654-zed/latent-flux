"""Add Kelp OApp delegate 0x1f7A03b70C5448DFd0a2C5a7865169253c2C769b to watchlist.

This is the single EOA that modified the LayerZero channel configuration from
2-of-2 to 1-of-1 before the 2026-04-18 exploit. Watchlisted as a
configuration-authority anchor.

Per the 2026-05-18 LayerZero Labs incident report (kelpdao-incident-report.pdf),
this EOA's setConfig call is the load-bearing pre-exploit action that converted
the channel from "compromise of both DVNs required" to "compromise of one DVN's
RPC layer suffices." A single key controlled the security policy of a $292M
bridge.

Watchlist this address as kelp_oapp_delegate_config_authority. Subsequent
activity on this EOA — particularly setConfig calls on other channels or
ownership transfers — is high-signal.

CLI:
    python scripts/add_kelp_oapp_delegate_watchlist.py             # local only
    python scripts/add_kelp_oapp_delegate_watchlist.py --prod      # local + production
"""
from __future__ import annotations
import argparse, sqlite3, base64, subprocess, shutil
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
ADDRESS = "0x1f7a03b70c5448dfd0a2c5a7865169253c2c769b"  # lowercase per watchlist convention
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
NAME = "kelp_oapp_delegate_config_authority"
REASON = (
    "Kelp rsETH OApp delegate (EOA). Modified the LayerZero channel security "
    "configuration from 2-of-2 to 1-of-1 before the 2026-04-18 exploit ($292M, "
    "116,500 rsETH). Single EOA controlled the security policy of a $292M bridge. "
    "Per LayerZero Labs incident report 2026-05-18: this address's setConfig "
    "call was the load-bearing pre-exploit action. UNC4899 / TraderTraitor (DPRK) "
    "compromised LayerZero's RPC infrastructure (FLATROOF/ROOFDECK malware on "
    "macOS); the 1-of-1 config meant compromise of one DVN's RPC layer sufficed. "
    "See reports/extraction_event_008_kelp.md + docs/INDEX.md Section 2 KelpDAO "
    "LayerZero exploit subsection."
)


def insert(conn: sqlite3.Connection) -> str:
    existing = conn.execute(
        "SELECT id, entity_name, priority, active FROM watchlist WHERE LOWER(address)=?",
        (ADDRESS,)
    ).fetchone()
    if existing:
        return f"already_exists id={existing[0]} name={existing[1]} priority={existing[2]} active={existing[3]}"
    conn.execute(
        """INSERT INTO watchlist
           (address, address_type, entity_name, watch_reason, priority,
            added_date, last_seen_date, hit_count, active)
           VALUES (?, 'eoa', ?, ?, 'HIGH', ?, NULL, 0, 1)""",
        (ADDRESS, NAME, REASON, NOW),
    )
    return "inserted"


def apply_local() -> str:
    conn = sqlite3.connect(DB_PATH)
    try:
        result = insert(conn)
        conn.commit()
    finally:
        conn.close()
    return result


def apply_prod() -> tuple[int, str, str]:
    inner = (
        "import sqlite3\n"
        "DB = '/app/surveillance/data/surveillance.db'\n"
        f"NOW = {NOW!r}\n"
        f"ADDR = {ADDRESS!r}\n"
        f"NAME = {NAME!r}\n"
        f"REASON = {REASON!r}\n"
        "conn = sqlite3.connect(DB)\n"
        "cur = conn.execute('SELECT id, entity_name, priority FROM watchlist WHERE LOWER(address)=?', (ADDR,))\n"
        "r = cur.fetchone()\n"
        "if r:\n"
        "    print(f'EXISTS {ADDR} id={r[0]} name={r[1]}')\n"
        "else:\n"
        "    conn.execute(\"INSERT INTO watchlist (address, address_type, entity_name, watch_reason, priority, added_date, last_seen_date, hit_count, active) VALUES (?, 'eoa', ?, ?, 'HIGH', ?, NULL, 0, 1)\", (ADDR, NAME, REASON, NOW))\n"
        "    conn.commit()\n"
        "    print(f'INSERTED {ADDR} {NAME}')\n"
        "conn.close()\n"
        "print('DONE')\n"
    )
    b64 = base64.b64encode(inner.encode()).decode()
    bootstrap = "python3 -c \"import base64; exec(base64.b64decode('" + b64 + "'))\""
    exe = shutil.which("railway.cmd") or shutil.which("railway")
    if not exe:
        return (-1, "", "railway CLI not found")
    cmd = ["cmd.exe", "/c", exe, "ssh", bootstrap]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prod", action="store_true")
    args = ap.parse_args()

    print(f"=== LOCAL apply ({DB_PATH}) ===")
    print(f"  {ADDRESS}: {apply_local()}")

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
