"""Add 0xc0ee427bee1d (self-deploying trap operator, active 2026-05-21) to
watchlist as HIGH-priority.

Why: surfaced 2026-05-21 during recent-activity review. Self-deploying — the
EOA acts as both deployer and drain_caller. Deployed 0x7937b4c76ec2 on
Arbitrum at 09:41:18 UTC, drained 4 victims in a single tx at 12:57:29 UTC
(3h 16m post-deploy), then 1 more victim at 13:53:18 UTC. No mainnet
history, no funder trace, behavioral_score 0 — pristine pure-L2 operator.

Fits the established self_deploying_trap_operator archetype seen in
0xacc79e7b9f8d (case CASE_SELF_DEPLOYING_TRAP_OPERATOR_0xACC79E7B) and
0x73c0c56bbf16. Adding the deployer (which is also the drainer) to the
watchlist makes the contract automatically surveilled by Q-002.

Q-002's join `JOIN watchlist w ON w.address = c.deployer_address` will pick
up the contract via the deployer entry. The contract itself is also added as
a separate row so the deployed bait is on the explicit watchlist.

CLI:
    python scripts/add_0xc0ee427b_watchlist.py            # local only
    python scripts/add_0xc0ee427b_watchlist.py --prod     # local + production
"""
from __future__ import annotations
import argparse, sqlite3, base64, subprocess, shutil
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEPLOYER = "0xc0ee427bee1d1f67861612c11fdf5f9b6b49cd66"
CONTRACT = "0x7937b4c76ec2649b8fcf032655762a7b1bffb366"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

DEPLOYER_NAME = "self_deploying_trap_operator_c0ee427b"
DEPLOYER_REASON = (
    "Self-deploying trap operator on Arbitrum. EOA is both deployer and "
    "drain_caller. Deployed 0x7937b4c76ec2 at 2026-05-21T09:41:18Z; first "
    "drain 12:57:29Z (4 victims in single tx 0x63385b14); second 13:53:18Z "
    "(1 victim). 3h 16m deploy-to-drain. No mainnet history, no funder, "
    "behavioral_score 0. Fits self_deploying_trap_operator archetype "
    "(0xacc79e7b9f8d, 0x73c0c56bbf16). Add to enable Q-002 daily surveillance."
)
CONTRACT_NAME = "self_deploy_bait_c0ee427b_7937b4"
CONTRACT_REASON = (
    "Bait deployed by 0xc0ee427bee1d. Arbitrum, confirmed-tier after first "
    "drain. 5 drains across 2 tx in 4-hour window post-deploy. Same EOA "
    "deploys and sweeps. Monitor for additional victims approving "
    "post-discharge (the 0x752c5a95 harvester pattern — accumulation "
    "continues after first sweep)."
)


def insert(conn: sqlite3.Connection, address: str, kind: str, name: str, reason: str) -> str:
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
    inner = (
        "import sqlite3\n"
        "DB = '/app/surveillance/data/surveillance.db'\n"
        f"NOW = {NOW!r}\n"
        "conn = sqlite3.connect(DB)\n"
        "def ins(addr, kind, name, reason):\n"
        "    cur = conn.execute('SELECT id, entity_name FROM watchlist WHERE address=?', (addr,))\n"
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
    bootstrap = "python3 -c \"import base64; exec(base64.b64decode('" + b64 + "'))\""
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
