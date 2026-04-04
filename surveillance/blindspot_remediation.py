"""
Layer 3 -- Blindspot Remediation
Fixes 5 and 6: SLOAD categorization + permit monitoring.
Fixes 3 and 4: approval value + balance sampling (RPC required, generates scripts).

Run: python -m surveillance.blindspot_remediation
"""

import json
import logging
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("surveillance.blindspot")

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def ensure_tables(conn: sqlite3.Connection):
    """Create tables for all blindspot fixes."""
    conn.executescript("""
        -- Fix 3: Approval values
        CREATE TABLE IF NOT EXISTS approval_values (
            approver TEXT NOT NULL,
            contract_address TEXT NOT NULL,
            token_address TEXT,
            chain TEXT,
            allowance_raw TEXT,
            allowance_is_max INTEGER DEFAULT 0,
            balance_raw TEXT,
            value_at_risk_wei TEXT,
            checked_at TEXT,
            PRIMARY KEY (approver, contract_address)
        );

        -- Fix 4: Contract balances
        CREATE TABLE IF NOT EXISTS contract_balances (
            contract_address TEXT NOT NULL,
            chain TEXT,
            balance_wei TEXT,
            balance_eth REAL,
            checked_at TEXT,
            PRIMARY KEY (contract_address, chain)
        );

        -- Fix 5: SLOAD pattern categorization
        CREATE TABLE IF NOT EXISTS sload_patterns (
            contract_address TEXT PRIMARY KEY,
            chain TEXT,
            deployer_address TEXT,
            confidence_tier TEXT,
            pattern_type TEXT,
            org_id TEXT,
            interaction_count INTEGER DEFAULT 0,
            last_spike_check TEXT,
            categorized_at TEXT
        );

        -- Fix 6: Permit events
        CREATE TABLE IF NOT EXISTS permit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address TEXT NOT NULL,
            chain TEXT,
            caller TEXT,
            timestamp TEXT,
            tx_hash TEXT,
            function_selector TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_permit_contract ON permit_events(contract_address);
        CREATE INDEX IF NOT EXISTS idx_sload_pattern ON sload_patterns(pattern_type);
        CREATE INDEX IF NOT EXISTS idx_sload_org ON sload_patterns(org_id);
        CREATE INDEX IF NOT EXISTS idx_approval_val_contract ON approval_values(contract_address);
    """)


# =========================================================================
# Fix 5: SLOAD-Gated Contract Categorization (zero RPC)
# =========================================================================

def categorize_sload_contracts(conn: sqlite3.Connection) -> dict:
    """Categorize 3,066 SLOAD-gated contracts by pattern type."""
    ensure_tables(conn)
    now = datetime.now(timezone.utc).isoformat()

    contracts = conn.execute("""
        SELECT c.contract_address, c.chain, c.deployer_address, c.confidence_tier,
               c.bytecode_pattern_notes
        FROM contracts c
        WHERE c.bytecode_pattern_notes LIKE '%SLOAD%'
        AND c.confidence_tier IN ('suspected', 'confirmed')
        AND c.contract_address NOT IN (SELECT contract_address FROM sload_patterns)
    """).fetchall()

    counts = Counter()

    for c in contracts:
        notes = (c["bytecode_pattern_notes"] or "").upper()
        addr = c["contract_address"]

        # Categorize by the opcodes surrounding SLOAD
        if "CALLER" in notes and "SLOAD" in notes and "EQ" in notes:
            pattern = "owner_check"  # CALLER -> SLOAD -> EQ: only owner
        elif "CALLER" in notes and "SLOAD" in notes and ("JUMPI" in notes or "REVERT" in notes):
            pattern = "blacklist"    # address-keyed mapping check
        elif "SLOAD" in notes and ("LT" in notes or "GT" in notes):
            pattern = "threshold"    # value comparison (balance/counter)
        elif "SLOAD" in notes and "ISZERO" in notes:
            pattern = "toggle"       # on/off switch
        elif "SHA3" in notes and "SLOAD" in notes:
            pattern = "mapping_lookup"  # mapping(address => X) read
        elif "SLOAD" in notes and "DIV" in notes:
            pattern = "fee_calc"     # storage-based fee calculation
        else:
            pattern = "other"

        # Org attribution
        org = None
        trail_row = conn.execute(
            "SELECT funding_trail FROM deployers WHERE deployer_address = ?",
            (c["deployer_address"],)
        ).fetchone()
        if trail_row and trail_row["funding_trail"]:
            try:
                org = json.loads(trail_row["funding_trail"]).get("org_link")
            except (json.JSONDecodeError, TypeError):
                pass
        if not org:
            ec = conn.execute(
                "SELECT org_id FROM entity_classification WHERE address = ?",
                (c["deployer_address"],)
            ).fetchone()
            if ec:
                org = ec["org_id"]

        # Interaction count
        tx_count = conn.execute(
            "SELECT COUNT(*) FROM transaction_events WHERE contract_address = ?",
            (addr,)
        ).fetchone()[0]

        conn.execute("""
            INSERT OR IGNORE INTO sload_patterns
            (contract_address, chain, deployer_address, confidence_tier,
             pattern_type, org_id, interaction_count, categorized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (addr, c["chain"], c["deployer_address"], c["confidence_tier"],
              pattern, org, tx_count, now))

        counts[pattern] += 1

    conn.commit()
    return {"categorized": len(contracts), "by_type": dict(counts)}


def detect_sload_spikes(conn: sqlite3.Connection) -> dict:
    """Detect interaction spikes on SLOAD-gated contracts (possible state change)."""
    now = datetime.now(timezone.utc).isoformat()

    # Find SLOAD contracts with sudden activity increases
    spikes = conn.execute("""
        SELECT sp.contract_address, sp.pattern_type, sp.org_id,
               sp.interaction_count as old_count,
               (SELECT COUNT(*) FROM transaction_events
                WHERE contract_address = sp.contract_address) as new_count
        FROM sload_patterns sp
        WHERE sp.interaction_count > 0
    """).fetchall()

    spike_alerts = []
    for s in spikes:
        old = s["old_count"]
        new = s["new_count"]
        if new > old * 3 and new - old >= 5:  # 3x spike and at least 5 new
            spike_alerts.append({
                "contract": s["contract_address"],
                "pattern": s["pattern_type"],
                "org": s["org_id"],
                "old_count": old,
                "new_count": new,
            })
            # Update stored count
            conn.execute("""
                UPDATE sload_patterns SET interaction_count = ?, last_spike_check = ?
                WHERE contract_address = ?
            """, (new, now, s["contract_address"]))

    conn.commit()
    return {"spikes": len(spike_alerts), "details": spike_alerts}


# =========================================================================
# Fix 6: Permit Selector Monitoring (zero RPC)
# =========================================================================

PERMIT_SELECTORS = {
    "d505accf": "permit_eip2612",
    "8fcbaf0c": "permit_dai",
    "2b67b570": "permit2_uniswap",
    "30f28b7a": "permitTransferFrom",
}


def scan_permit_events(conn: sqlite3.Connection) -> dict:
    """Scan transaction_events for permit-related selectors."""
    ensure_tables(conn)

    found = 0
    for selector, name in PERMIT_SELECTORS.items():
        events = conn.execute("""
            SELECT te.contract_address, te.interacting_address, te.timestamp,
                   te.tx_hash, c.chain
            FROM transaction_events te
            JOIN contracts c ON te.contract_address = c.contract_address
            WHERE te.function_selector = ?
            AND te.tx_hash NOT IN (SELECT tx_hash FROM permit_events WHERE tx_hash IS NOT NULL)
        """, (selector,)).fetchall()

        for e in events:
            conn.execute("""
                INSERT OR IGNORE INTO permit_events
                (contract_address, chain, caller, timestamp, tx_hash, function_selector)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (e["contract_address"], e["chain"], e["interacting_address"],
                  e["timestamp"], e["tx_hash"], selector))
            found += 1

    conn.commit()
    return {"permit_events_found": found}


# =========================================================================
# Main: run all zero-RPC fixes
# =========================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    db = os.environ.get("DB_PATH", str(DB_PATH))
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row

    ensure_tables(conn)

    print("=" * 60)
    print("BLINDSPOT REMEDIATION")
    print("=" * 60)

    # Fix 5: SLOAD categorization
    print("\n## Fix 5: SLOAD Pattern Categorization")
    r5 = categorize_sload_contracts(conn)
    print(f"Categorized: {r5['categorized']}")
    for pattern, count in sorted(r5["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    # SLOAD spike detection
    print("\n## Fix 5b: SLOAD Spike Detection")
    r5b = detect_sload_spikes(conn)
    print(f"Spikes detected: {r5b['spikes']}")
    for s in r5b["details"][:5]:
        print(f"  {s['contract'][:18]}... {s['pattern']} {s['old_count']}->{s['new_count']} org={s['org']}")

    # Fix 6: Permit monitoring
    print("\n## Fix 6: Permit Event Scan")
    r6 = scan_permit_events(conn)
    print(f"Permit events found: {r6['permit_events_found']}")

    # SLOAD org attribution
    print("\n## Fix 5c: SLOAD by Org")
    org_sload = conn.execute("""
        SELECT COALESCE(org_id, 'unattributed') as org, pattern_type, COUNT(*) as cnt
        FROM sload_patterns
        WHERE org_id IS NOT NULL AND org_id != ''
        GROUP BY org, pattern_type ORDER BY org, cnt DESC
    """).fetchall()
    for r in org_sload:
        print(f"  {r['org']}: {r['pattern_type']} = {r['cnt']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    sload_total = conn.execute("SELECT COUNT(*) FROM sload_patterns").fetchone()[0]
    permit_total = conn.execute("SELECT COUNT(*) FROM permit_events").fetchone()[0]
    tl_with_ts = conn.execute(
        "SELECT COUNT(*) FROM timelock_countdowns WHERE activation_timestamp IS NOT NULL"
    ).fetchone()[0]
    proxy_checked = 0
    try:
        proxy_checked = conn.execute(
            "SELECT COUNT(DISTINCT contract_address) FROM proxy_implementations"
        ).fetchone()[0]
    except Exception:
        pass

    print(f"  Fix 1 (Timelock extraction): {tl_with_ts} timestamps extracted (needs RPC backfill for rest)")
    print(f"  Fix 2 (Proxy watcher): {proxy_checked} contracts checked (runs daily on Railway)")
    print(f"  Fix 3 (Approval values): tables created, needs RPC sampling script")
    print(f"  Fix 4 (Contract balances): tables created, needs RPC sampling script")
    print(f"  Fix 5 (SLOAD categorization): {sload_total} contracts categorized")
    print(f"  Fix 6 (Permit monitoring): {permit_total} permit events found")

    conn.close()


if __name__ == "__main__":
    main()
