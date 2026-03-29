"""
Layer 3 — Approval Drain Monitor

Tracks the deferred exploitation pattern:
1. Victim calls approve() on a trap contract (looks clean, 0% revert)
2. Later, the operator calls transferFrom() to drain the approved tokens
3. The drain may come from a different address than the deployer

This monitor:
- Tracks all approve() calls on suspected/self-test contracts
- Builds a watchlist of (victim, approved_contract) pairs
- Monitors for transferFrom() calls that match pending approvals
- Alerts when drains begin (the approval trap fires)

Runs as a periodic scan in the heartbeat loop (no API calls).

Usage:
    python -m surveillance.approval_drain_monitor --scan
    python -m surveillance.approval_drain_monitor --watchlist
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_tables(conn: sqlite3.Connection):
    """Create approval monitoring tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            victim_address TEXT NOT NULL,
            contract_address TEXT NOT NULL,
            approve_tx_hash TEXT,
            approve_timestamp TEXT,
            approve_block INTEGER,
            contract_tier TEXT,
            is_self_test_trap INTEGER DEFAULT 0,
            deployer_address TEXT,
            drain_detected INTEGER DEFAULT 0,
            drain_tx_hash TEXT,
            drain_timestamp TEXT,
            drain_caller TEXT,
            logged_at TEXT,
            UNIQUE(victim_address, contract_address)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approval_victim ON approval_watchlist(victim_address)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approval_contract ON approval_watchlist(contract_address)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approval_drain ON approval_watchlist(drain_detected)
    """)
    conn.commit()


def scan_approvals(conn: sqlite3.Connection) -> dict:
    """
    Scan for approve() calls on suspected contracts.
    Build/update the approval watchlist.
    """
    ensure_tables(conn)
    now = _now()

    # Find approve() calls on suspected/confirmed contracts not yet in watchlist
    new_approvals = conn.execute("""
        SELECT te.interacting_address as victim,
               te.contract_address,
               te.tx_hash,
               te.timestamp,
               te.block_number,
               c.confidence_tier,
               c.deployer_address,
               CASE WHEN st.contract_address IS NOT NULL THEN 1 ELSE 0 END as is_self_test
        FROM transaction_events te
        JOIN contracts c ON c.contract_address = te.contract_address
        LEFT JOIN self_test_traps st ON st.contract_address = te.contract_address
        LEFT JOIN approval_watchlist aw ON aw.victim_address = te.interacting_address
            AND aw.contract_address = te.contract_address
        WHERE te.function_selector = '095ea7b3'
        AND c.confidence_tier IN ('suspected', 'confirmed')
        AND te.interacting_address != c.deployer_address
        AND aw.id IS NULL
    """).fetchall()

    added = 0
    for a in new_approvals:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO approval_watchlist
                (victim_address, contract_address, approve_tx_hash, approve_timestamp,
                 approve_block, contract_tier, is_self_test_trap, deployer_address, logged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (a["victim"], a["contract_address"], a["tx_hash"],
                  a["timestamp"], a["block_number"], a["confidence_tier"],
                  a["is_self_test"], a["deployer_address"], now))
            added += 1
        except Exception:
            pass

    conn.commit()
    return {"new_approvals_tracked": added}


def check_drains(conn: sqlite3.Connection) -> dict:
    """
    Check if any watched approvals have been drained.

    Looks for:
    1. transferFrom() calls on watched contracts
    2. Any token transfer FROM a victim TO the deployer or unknown collector
    3. Contract interactions by the deployer AFTER approvals came in
    """
    ensure_tables(conn)
    now = _now()
    drains_found = 0

    # Method 1: transferFrom() on watched contracts
    pending = conn.execute("""
        SELECT aw.victim_address, aw.contract_address, aw.deployer_address,
               aw.approve_timestamp
        FROM approval_watchlist aw
        WHERE aw.drain_detected = 0
    """).fetchall()

    for p in pending:
        # Check for transferFrom on this contract after the approval
        drain = conn.execute("""
            SELECT te.tx_hash, te.timestamp, te.interacting_address as caller
            FROM transaction_events te
            WHERE te.contract_address = ?
            AND te.function_selector = '23b872dd'
            AND te.timestamp > ?
            LIMIT 1
        """, (p["contract_address"], p["approve_timestamp"])).fetchone()

        if drain:
            conn.execute("""
                UPDATE approval_watchlist
                SET drain_detected = 1, drain_tx_hash = ?, drain_timestamp = ?,
                    drain_caller = ?
                WHERE victim_address = ? AND contract_address = ?
            """, (drain["tx_hash"], drain["timestamp"], drain["caller"],
                  p["victim_address"], p["contract_address"]))
            drains_found += 1

    # Method 2: Deployer interacting with contract after external approvals
    # (might use a custom drain function, not standard transferFrom)
    deployer_drains = conn.execute("""
        SELECT aw.contract_address, aw.deployer_address,
               te.tx_hash, te.timestamp, te.function_selector
        FROM approval_watchlist aw
        JOIN transaction_events te ON te.contract_address = aw.contract_address
            AND te.interacting_address = aw.deployer_address
        WHERE aw.drain_detected = 0
        AND te.timestamp > aw.approve_timestamp
        AND te.function_selector NOT IN ('095ea7b3', 'a9059cbb')
        GROUP BY aw.contract_address, te.tx_hash
    """).fetchall()

    for d in deployer_drains:
        conn.execute("""
            UPDATE approval_watchlist
            SET drain_detected = 1, drain_tx_hash = ?, drain_timestamp = ?,
                drain_caller = ?
            WHERE contract_address = ? AND drain_detected = 0
        """, (d["tx_hash"], d["timestamp"], d["deployer_address"],
              d["contract_address"]))
        drains_found += 1

    conn.commit()
    return {"drains_detected": drains_found}


def get_summary(conn: sqlite3.Connection) -> dict:
    """Get current approval watchlist statistics."""
    ensure_tables(conn)

    total = conn.execute("SELECT COUNT(*) FROM approval_watchlist").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=0").fetchone()[0]
    drained = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
    self_test = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE is_self_test_trap=1").fetchone()[0]

    unique_victims = conn.execute("SELECT COUNT(DISTINCT victim_address) FROM approval_watchlist").fetchone()[0]
    unique_contracts = conn.execute("SELECT COUNT(DISTINCT contract_address) FROM approval_watchlist").fetchone()[0]

    return {
        "total_tracked": total,
        "pending_drain": pending,
        "drain_detected": drained,
        "on_self_test_traps": self_test,
        "unique_victims": unique_victims,
        "unique_contracts": unique_contracts,
    }


def print_watchlist(conn: sqlite3.Connection):
    """Print the current watchlist status."""
    ensure_tables(conn)
    summary = get_summary(conn)

    print(f"[approval_drain] Watchlist: {summary['total_tracked']} tracked | {summary['pending_drain']} pending | {summary['drain_detected']} drained")
    print(f"  Unique victims: {summary['unique_victims']} | Contracts: {summary['unique_contracts']} | On self-test traps: {summary['on_self_test_traps']}")

    # Top contracts by pending approvals
    print()
    print("Top contracts by pending approvals:")
    for r in conn.execute("""
        SELECT contract_address, deployer_address, COUNT(*) as pending,
            is_self_test_trap,
            MIN(approve_timestamp) as first_approve,
            MAX(approve_timestamp) as last_approve
        FROM approval_watchlist WHERE drain_detected=0
        GROUP BY contract_address ORDER BY pending DESC LIMIT 15
    """):
        st = " [SELF-TEST]" if r["is_self_test_trap"] else ""
        print(f"  {r['contract_address'][:18]}... | {r['pending']:>4} pending | deployer={r['deployer_address'][:12]}... | {r['first_approve'][:10]} to {r['last_approve'][:10]}{st}")

    # Any drains detected?
    drained = conn.execute("""
        SELECT contract_address, drain_caller, drain_timestamp, COUNT(*) as victims_drained
        FROM approval_watchlist WHERE drain_detected=1
        GROUP BY contract_address ORDER BY victims_drained DESC LIMIT 10
    """).fetchall()
    if drained:
        print()
        print("DRAINS DETECTED:")
        for r in drained:
            print(f"  {r['contract_address'][:18]}... | {r['victims_drained']} victims drained | caller={r['drain_caller'][:14]}... | {r['drain_timestamp'][:16]}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approval drain monitor")
    parser.add_argument("--scan", action="store_true", help="Scan for new approvals + check drains")
    parser.add_argument("--watchlist", action="store_true", help="Show current watchlist")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    if args.scan:
        r1 = scan_approvals(conn)
        print(f"[approval_drain] New approvals tracked: {r1['new_approvals_tracked']}")
        r2 = check_drains(conn)
        print(f"[approval_drain] Drains detected: {r2['drains_detected']}")
        print()
        print_watchlist(conn)
    elif args.watchlist:
        print_watchlist(conn)
    else:
        parser.print_help()

    conn.close()
