"""Document EXTRACTION_001 — first live observation of full org_001 pipeline cycle."""
import sqlite3
import json
from datetime import datetime, timezone

DB = "surveillance/data/surveillance.db"
conn = sqlite3.connect(DB)
now = datetime.now(timezone.utc).isoformat()

# 1. Create tables
conn.executescript("""
CREATE TABLE IF NOT EXISTS extraction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    documented_at TEXT NOT NULL,
    summary TEXT NOT NULL,
    raw_transactions TEXT,
    total_usd_moved REAL,
    nodes_active INTEGER,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS stats_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TEXT NOT NULL,
    event_ref TEXT,
    stats_json TEXT NOT NULL
);
""")

# 2. Raw transaction evidence
raw_txs = json.dumps({
    "cashout_to_defi_router": [
        {"to": "0x51c728", "amount": 4575.15, "asset": "USDC", "ts": "2026-03-20T04:04:51"},
        {"to": "0x51c728", "amount": 1675.75, "asset": "USDC", "ts": "2026-03-20T04:04:51"},
        {"to": "0x51c728", "amount": 789.30, "asset": "USDC", "ts": "2026-03-20T04:04:51"},
    ],
    "cashout_to_lp_staging": [
        {"to": "0x96daa0", "amount": 1918.96, "asset": "USDC", "ts": "2026-03-20T04:04:50"},
    ],
    "cashout_to_exit_ramp": [
        {"to": "0x01989c", "amount": 4144.99, "asset": "USDC", "ts": "2026-03-20T04:04:50"},
    ],
    "lp_staging_weth_back": [
        {"to": "cashout", "weth": 0.8918, "ts": "2026-03-20T04:04:50"},
        {"to": "cashout", "weth": 1.2285, "ts": "2026-03-20T04:04:48"},
        {"to": "cashout", "weth": 0.1616, "ts": "2026-03-20T04:04:48"},
        {"to": "cashout", "weth": 0.1457, "ts": "2026-03-20T04:04:48"},
    ],
    "treasury_branch_to_exit_ramp": [
        {"to": "0x01989c", "amount": 1108.83, "asset": "USDC", "ts": "2026-03-20T04:04:51",
         "note": "NEW direct link: treasury branch -> exit ramp"},
    ],
    "exit_ramp_weth_back_to_cashout": [
        {"to": "cashout", "weth": 1.9263, "ts": "2026-03-20T04:04:50",
         "note": "Exit ramp now TWO-WAY: sends WETH back to cashout"},
    ],
    "exit_ramp_wbtc_dispersal": [
        {"to": "0x0e4831", "wbtc": 0.0353, "ts": "2026-03-20T04:04:51"},
        {"to": "0x5969ef", "wbtc": 0.0388, "ts": "2026-03-20T04:04:51"},
        {"to": "0x360e68 (treasury_branch)", "wbtc": 0.0157, "ts": "2026-03-20T04:04:51"},
    ],
    "defi_router_scatter": [
        {"to": "0x7fcdc3", "amount": 268.79, "asset": "USDC", "ts": "2026-03-20T04:04:53"},
        {"to": "0x389938", "amount": 751.01, "asset": "USDT", "ts": "2026-03-20T04:04:53"},
    ],
    "laundry_activity": [
        {"to": "0x5a17cb", "wbtc": 0.00115, "ts": "2026-03-20T03:50:41"},
        {"to": "0x6985cb", "amount": 144.80, "asset": "USDC", "ts": "2026-03-20T03:50:40"},
        {"to": "0x0e4831", "wbtc": 0.00169, "ts": "2026-03-20T03:50:03"},
    ],
})

summary = (
    "Full org_001 extraction cycle observed live across 6 simultaneous wallet nodes. "
    "Cashout processing fresh batch: $4,575 + $1,675 + $789 USDC to DeFi Router, "
    "$1,918 to LP Staging, $4,144 to CEX Exit Ramp. Exit ramp confirmed two-way: "
    "receiving USDC/USDT/WBTC inflows AND sending WBTC out + 1.93 WETH back to cashout. "
    "LP Staging cycling USDC->WETH ($4,910 in, 2.43 WETH out). "
    "Treasury Branch feeding exit ramp directly ($1,108 USDC) - new direct connection. "
    "DeFi Router distributing to scattered addresses. "
    "Laundry active at 03:50 UTC. Pipeline fully automated."
)

notes = (
    "First live observation of full extraction cycle. "
    "Exit ramp behavior changed - now two-way node (sends WBTC/WETH, not just receives). "
    "Treasury Branch -> Exit Ramp direct connection is new infrastructure link. "
    "All 6 nodes active within 20-minute window. "
    "Batch total: $13,101 direct cashout + $4,910 LP staging + $6,336 exit ramp inflows "
    "+ $1,108 treasury branch direct = $25,455+ observed. "
    "Additional WETH/WBTC legs not USD-denominated."
)

conn.execute(
    """INSERT INTO extraction_events
       (event_id, event_type, observed_at, documented_at, summary,
        raw_transactions, total_usd_moved, nodes_active, notes)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    ("EXTRACTION_001", "full_pipeline_cycle", "2026-03-20T04:04:00+00:00",
     now, summary, raw_txs, 25455.0, 6, notes),
)
print("EXTRACTION_001 recorded.")

# 3. Stats snapshot
stats = {
    "contracts": dict(conn.execute(
        "SELECT confidence_tier, COUNT(*) FROM contracts GROUP BY confidence_tier"
    ).fetchall()),
    "deployers": conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0],
    "tx_events": conn.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0],
    "bots": conn.execute("SELECT COUNT(*) FROM bot_candidates").fetchone()[0],
    "chains": dict(conn.execute(
        "SELECT chain, COUNT(*) FROM contracts GROUP BY chain"
    ).fetchall()),
    "cross_chain": conn.execute(
        "SELECT COUNT(*) FROM (SELECT deployer_address FROM contracts "
        "GROUP BY deployer_address HAVING COUNT(DISTINCT chain)>1)"
    ).fetchone()[0],
    "live_exposures_open": conn.execute(
        "SELECT COUNT(*) FROM live_exposures WHERE status='open'"
    ).fetchone()[0],
}
conn.execute(
    "INSERT INTO stats_snapshots (snapshot_at, event_ref, stats_json) VALUES (?, ?, ?)",
    (now, "EXTRACTION_001", json.dumps(stats)),
)
print("Stats snapshot saved.")

# 4. Alert
payload = json.dumps({
    "event_id": "EXTRACTION_001",
    "cashout_batch_usdc": 13104.15,
    "exit_ramp_new_behavior": "two-way",
    "treasury_branch_direct_to_exit": 1108.83,
    "nodes": ["cashout", "exit_ramp", "lp_staging", "defi_router", "laundry", "treasury_branch"],
})
conn.execute(
    "INSERT INTO alerts (alert_type, address, tx_hash, timestamp, false_positive) "
    "VALUES (?, ?, ?, ?, ?)",
    ("LIVE_EXTRACTION_OBSERVED", "0xc6962004f452be9203591991d15f6b388e09e8d0",
     payload, "2026-03-20T04:04:51+00:00", 0),
)
print("Alert LIVE_EXTRACTION_OBSERVED recorded.")

conn.commit()

# Verify
print()
e = conn.execute("SELECT event_id, observed_at, total_usd_moved, nodes_active FROM extraction_events").fetchone()
print(f"Event: {e[0]} | observed={e[1]} | usd=${e[2]:,.0f} | nodes={e[3]}")
print(f"Stats snapshots: {conn.execute('SELECT COUNT(*) FROM stats_snapshots').fetchone()[0]}")
print(f"Extraction alerts: {conn.execute('SELECT COUNT(*) FROM alerts WHERE alert_type=\"LIVE_EXTRACTION_OBSERVED\"').fetchone()[0]}")

conn.close()
print("\nAll evidence documented.")
