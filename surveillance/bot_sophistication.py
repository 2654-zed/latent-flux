"""
Layer 3 — Bot Sophistication Classifier

Classifies bots by simulation capability using 4 indirect signals:
1. Revert rate (low = likely simulates before submitting)
2. Selector diversity (high = analyzes contract interface)
3. First-interaction delay (slow = takes time to evaluate)
4. Contract selectivity (skips many = has filtering criteria)

Composite score: 0 (blind) to 1.0 (sophisticated simulator)

Usage:
    python -m surveillance.bot_sophistication --classify-all
    python -m surveillance.bot_sophistication --check 0xADDRESS
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_bot(conn, bot_address: str) -> dict:
    """Compute sophistication score for a single bot."""

    # Get interaction stats
    stats = conn.execute("""
        SELECT COUNT(*) as total_hits,
            COUNT(DISTINCT contract_address) as contracts,
            COUNT(DISTINCT function_selector) as selectors,
            SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts,
            MIN(timestamp) as first_seen, MAX(timestamp) as last_seen
        FROM transaction_events WHERE interacting_address=?
    """, (bot_address,)).fetchone()

    if not stats or not stats["total_hits"] or stats["total_hits"] < 3:
        return {"bot_address": bot_address, "score": None, "reason": "insufficient_data"}

    total = stats["total_hits"]
    contracts = stats["contracts"] or 1
    selectors = stats["selectors"] or 1
    reverts = stats["reverts"] or 0

    # Signal 1: Revert rate (0% = 1.0, 100% = 0.0)
    revert_rate = reverts / total
    revert_score = max(0, 1.0 - (revert_rate * 1.25))  # 80%+ reverts = 0

    # Signal 2: Selector diversity (selectors per contract)
    sel_per_contract = selectors / contracts
    if sel_per_contract >= 3.0:
        selector_score = 1.0
    elif sel_per_contract >= 2.0:
        selector_score = 0.7
    elif sel_per_contract >= 1.5:
        selector_score = 0.4
    else:
        selector_score = 0.1  # single selector = blind

    # Signal 3: First-interaction delay after deployment
    delay_data = conn.execute("""
        SELECT AVG(delay_sec) as avg_delay FROM (
            SELECT (julianday(MIN(te.timestamp)) - julianday(c.detection_timestamp)) * 86400 as delay_sec
            FROM transaction_events te
            JOIN contracts c ON c.contract_address = te.contract_address
            WHERE te.interacting_address=?
            GROUP BY te.contract_address
        ) WHERE delay_sec >= 0
    """, (bot_address,)).fetchone()

    avg_delay = delay_data["avg_delay"] if delay_data and delay_data["avg_delay"] else 0
    # Very fast (< 60s) = blind, moderate (60-600s) = maybe simulating, slow (>600s) = likely simulating or just slow
    if avg_delay < 30:
        delay_score = 0.1
    elif avg_delay < 120:
        delay_score = 0.3
    elif avg_delay < 600:
        delay_score = 0.6
    elif avg_delay < 3600:
        delay_score = 0.8
    else:
        delay_score = 0.5  # Very slow could be manual or scheduled, not necessarily simulating

    # Signal 4: Contract selectivity
    # What fraction of available suspected contracts did this bot interact with?
    total_suspected = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier IN ('suspected', 'confirmed')"
    ).fetchone()[0] or 1
    selectivity = contracts / total_suspected
    if selectivity < 0.001:  # hits <0.1% of available = very selective
        selectivity_score = 1.0
    elif selectivity < 0.01:
        selectivity_score = 0.7
    elif selectivity < 0.05:
        selectivity_score = 0.4
    else:
        selectivity_score = 0.1  # hits everything = blind

    # Composite score (weighted)
    weights = {
        "revert": 0.40,      # strongest signal
        "selector": 0.20,
        "delay": 0.15,
        "selectivity": 0.25,
    }
    composite = (
        revert_score * weights["revert"]
        + selector_score * weights["selector"]
        + delay_score * weights["delay"]
        + selectivity_score * weights["selectivity"]
    )

    # Classification
    if composite >= 0.7:
        sophistication = "SIMULATOR"
    elif composite >= 0.45:
        sophistication = "HYBRID"
    elif composite >= 0.25:
        sophistication = "REACTIVE"
    else:
        sophistication = "BLIND"

    return {
        "bot_address": bot_address,
        "sophistication": sophistication,
        "composite_score": round(composite, 3),
        "revert_score": round(revert_score, 3),
        "selector_score": round(selector_score, 3),
        "delay_score": round(delay_score, 3),
        "selectivity_score": round(selectivity_score, 3),
        "revert_rate": round(revert_rate * 100, 1),
        "selectors_per_contract": round(sel_per_contract, 2),
        "avg_delay_seconds": round(avg_delay, 0),
        "contracts_hit": contracts,
        "total_hits": total,
        "first_seen": stats["first_seen"],
        "last_seen": stats["last_seen"],
    }


def classify_all():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_sophistication (
            bot_address TEXT PRIMARY KEY,
            sophistication TEXT,
            composite_score REAL,
            revert_score REAL,
            selector_score REAL,
            delay_score REAL,
            selectivity_score REAL,
            revert_rate REAL,
            selectors_per_contract REAL,
            avg_delay_seconds REAL,
            contracts_hit INTEGER,
            total_hits INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            classified_at TEXT
        )
    """)

    bots = conn.execute("SELECT address FROM bot_candidates").fetchall()
    classified = 0
    results = {"SIMULATOR": 0, "HYBRID": 0, "REACTIVE": 0, "BLIND": 0, "insufficient": 0}

    for bot in bots:
        r = classify_bot(conn, bot["address"])

        if r.get("score") is None and r.get("composite_score") is None:
            results["insufficient"] += 1
            continue

        conn.execute("""
            INSERT OR REPLACE INTO bot_sophistication
            (bot_address, sophistication, composite_score, revert_score, selector_score,
             delay_score, selectivity_score, revert_rate, selectors_per_contract,
             avg_delay_seconds, contracts_hit, total_hits, first_seen, last_seen, classified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["bot_address"], r["sophistication"], r["composite_score"],
            r["revert_score"], r["selector_score"], r["delay_score"],
            r["selectivity_score"], r["revert_rate"], r["selectors_per_contract"],
            r["avg_delay_seconds"], r["contracts_hit"], r["total_hits"],
            r["first_seen"], r["last_seen"], _now()
        ))
        results[r["sophistication"]] += 1
        classified += 1

    conn.commit()

    print(f"[bot_sophistication] Classified {classified} bots ({results['insufficient']} insufficient data)")
    print()
    print(f"  {'Category':12} | {'Count':>5} | Description")
    print(f"  {'-'*60}")
    print(f"  {'SIMULATOR':12} | {results['SIMULATOR']:>5} | Low revert, high selectivity, analyzes before acting")
    print(f"  {'HYBRID':12} | {results['HYBRID']:>5} | Mixed signals — some analysis, some blind firing")
    print(f"  {'REACTIVE':12} | {results['REACTIVE']:>5} | Minimal analysis, reacts to surface patterns")
    print(f"  {'BLIND':12} | {results['BLIND']:>5} | No filtering, fires on everything, high revert")

    # Show top simulators
    print()
    print("Top SIMULATOR bots:")
    for r in conn.execute("""
        SELECT * FROM bot_sophistication WHERE sophistication='SIMULATOR'
        ORDER BY composite_score DESC LIMIT 10
    """):
        print(f"  {r['bot_address'][:18]}... | score={r['composite_score']:.3f} | {r['revert_rate']:.0f}% rev | {r['contracts_hit']}c | {r['total_hits']} hits | delay={r['avg_delay_seconds']:.0f}s")

    # Show top BLIND bots
    print()
    print("Top BLIND bots:")
    for r in conn.execute("""
        SELECT * FROM bot_sophistication WHERE sophistication='BLIND'
        ORDER BY total_hits DESC LIMIT 10
    """):
        print(f"  {r['bot_address'][:18]}... | score={r['composite_score']:.3f} | {r['revert_rate']:.0f}% rev | {r['contracts_hit']}c | {r['total_hits']} hits | delay={r['avg_delay_seconds']:.0f}s")

    # Cross-reference: do simulators survive traps better?
    print()
    print("Sophistication vs trap survival:")
    for r in conn.execute("""
        SELECT sophistication, COUNT(*) as bots,
            ROUND(AVG(revert_rate), 1) as avg_rev,
            ROUND(AVG(total_hits), 0) as avg_hits,
            ROUND(AVG(contracts_hit), 1) as avg_contracts
        FROM bot_sophistication GROUP BY sophistication ORDER BY avg_rev
    """):
        print(f"  {r['sophistication']:12} | {r['bots']:>4} bots | {r['avg_rev']:>5.1f}% avg rev | {r['avg_hits']:>6.0f} avg hits | {r['avg_contracts']:.1f} avg contracts")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot sophistication classifier")
    parser.add_argument("--classify-all", action="store_true")
    parser.add_argument("--check", type=str, help="Check single address")
    args = parser.parse_args()

    if args.classify_all:
        classify_all()
    elif args.check:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        r = classify_bot(conn, args.check.lower())
        print(json.dumps(r, indent=2))
        conn.close()
    else:
        parser.print_help()
