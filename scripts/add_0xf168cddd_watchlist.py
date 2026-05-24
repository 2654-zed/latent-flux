"""Add 0xf168cddd9093 (self-deploying single-contract mass-drainer) and
its drain target 0x7c5517e212b0 to the watchlist as HIGH priority.

Discovery: 2026-05-24 post-Phase-C audit health check. Contract
0x7c5517e212b0 was deployed 2026-05-23T10:13Z on Base by 0xf168cddd9093,
which then drained 154/154 victims (100% of approval pool) across 4
txs at 11:05Z (150 victims mass-sweep), 13:22Z (2), 16:59Z (1), and
2026-05-24T08:40Z (1 — actively trolling).

Matches the Self-Deploying Single-Contract Mass-Drain lexicon archetype:
deployer wallet is the drain_caller, single-purpose drainer (24 corpus
tx_events total, drained only this one contract).

CLI:
    python scripts/add_0xf168cddd_watchlist.py             # local only
    python scripts/add_0xf168cddd_watchlist.py --prod      # local + prod
"""
from __future__ import annotations
import argparse, sqlite3, base64, subprocess, shutil
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEPLOYER = "0xf168cddd9093b6257f2562cc6ab4fcc7132e76bd"
CONTRACT = "0x7c5517e212b00f1344f000edf59b6952a1821281"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

DEPLOYER_NAME = "self_deploying_trap_operator_f168cddd"
DEPLOYER_REASON = (
    "Self-deploying single-contract mass-drainer. Deployed 0x7c5517e212b0 "
    "on Base 2026-05-23T10:13Z, drained 150 victims mass-sweep at 11:05Z, "
    "then trickle-drained over 22 hours (4 total txs, 154 victims = 100%). "
    "Itself is the drain_caller. Latest drain 2026-05-24T08:40Z. Single-"
    "purpose in corpus (24 tx_events total, drained only this contract). "
    "Same archetype as 0xacc79e7b (slow-bleed) and 0x7bf3269c (a27bba42)."
)
CONTRACT_NAME = "discharged_trap_f168cddd_0x7c5517e2"
CONTRACT_REASON = (
    "Trap contract deployed by 0xf168cddd9093 on Base 2026-05-23T10:13Z. "
    "100% of approval pool drained (154/154) across 4 txs spanning 22 "
    "hours. ERC-20 facade (approve+transfer selectors dominate) but the "
    "deployer/drainer is the same wallet — classic self-deployer trap."
)


def insert(conn, address, kind, name, reason):
    existing = conn.execute(
        "SELECT id, entity_name, priority, active FROM watchlist WHERE address=?", (address,)
    ).fetchone()
    if existing:
        return f"already_exists id={existing[0]} name={existing[1]} priority={existing[2]}"
    conn.execute(
        "INSERT INTO watchlist (address, address_type, entity_name, watch_reason, priority, "
        "added_date, last_seen_date, hit_count, active) VALUES (?, ?, ?, ?, 'HIGH', ?, NULL, 0, 1)",
        (address, kind, name, reason, NOW),
    )
    return "inserted"


def apply_local():
    conn = sqlite3.connect(DB_PATH)
    try:
        r1 = insert(conn, DEPLOYER, "deployer", DEPLOYER_NAME, DEPLOYER_REASON)
        r2 = insert(conn, CONTRACT, "contract", CONTRACT_NAME, CONTRACT_REASON)
        conn.commit()
    finally:
        conn.close()
    return r1, r2


def apply_prod():
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
        "conn.commit(); conn.close(); print('DONE')\n"
    )
    b64 = base64.b64encode(inner.encode()).decode()
    bootstrap = "python3 -c \"import base64; exec(base64.b64decode('" + b64 + "'))\""
    exe = shutil.which("railway.cmd") or shutil.which("railway")
    if not exe:
        return (-1, "", "railway CLI not found")
    cmd = ["cmd.exe", "/c", exe, "ssh", bootstrap]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true")
    args = ap.parse_args()
    print(f"=== LOCAL ({DB_PATH}) ===")
    r1, r2 = apply_local()
    print(f"  deployer {DEPLOYER}: {r1}")
    print(f"  contract {CONTRACT}: {r2}")
    if args.prod:
        print("\n=== PROD (railway ssh) ===")
        rc, out, err = apply_prod()
        print(f"  rc={rc}\n  stdout: {out.strip()}")
        if err.strip(): print(f"  stderr: {err.strip()[-300:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
