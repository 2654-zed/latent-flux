"""
Layer 3 — Bot Strategy Fingerprinting & Bait Pattern Detection

Maps bot strategies to trap architectures. Predicts what new traps will
target based on which bot strategies are active in the ecosystem.

Usage:
    python -m surveillance.strategy_fingerprint --classify-all
    python -m surveillance.strategy_fingerprint --profile-baits
    python -m surveillance.strategy_fingerprint --lifecycle
    python -m surveillance.strategy_fingerprint --predict-new
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from surveillance import db

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------
# Strategy classification rules
# ---------------------------------------------------------------

# Selectors that identify strategy types
CALLBACK_SELECTORS = {
    "fa461e33", "10d1e85c", "84800812", "23a69e75", "920f5c84",
    "2a217007", "61461954",
}
ROUTER_SELECTORS = {
    "3593564c", "7ff36ab5", "18cbafe5", "38ed1739",
}
APPROVE_SELECTORS = {"095ea7b3", "a22cb465"}
TRANSFER_SELECTORS = {"a9059cbb", "23b872dd"}
FALLBACK_SELECTORS = {"00000000", None, "NULL"}

# L2-optimized selectors: short numeric patterns used by gas-optimized bots
def _is_l2_optimized(sel: str) -> bool:
    if not sel or len(sel) < 6:
        return False
    try:
        val = int(sel, 16)
        return val < 0x100000  # < 1M = likely gas-optimized short selector
    except ValueError:
        return False


# Proprietary framework selectors from known_selectors analysis
FRAMEWORK_PAIR_SELECTORS = {"b2460c48", "6bfd6286", "b42ebb63", "a28dd96e", "b7f49f3f"}
SPRAY_PROBE_SELECTORS = {"a5cda515", "51936ec4", "0db84129", "675675f1", "b18a95c3"}


def classify_bot_strategy(selectors: list[dict], revert_count: int,
                          contract_count: int) -> tuple[str, str]:
    """
    Classify a bot into a strategy category.

    Args:
        selectors: list of {selector, call_count} from bot_candidate_selectors
        revert_count: total reverts from bot_candidates
        contract_count: distinct contracts interacted with

    Returns:
        (strategy_type, trap_vulnerability)
    """
    sel_set = {s["selector"] for s in selectors if s.get("selector")}
    total_calls = sum(s.get("call_count", 0) for s in selectors)

    # Check for callback arbitrage
    if sel_set & CALLBACK_SELECTORS:
        return "CALLBACK_ARBITRAGE", "CALLBACK_TRAP"

    # Check for framework pair (cluster_002 pattern)
    if sel_set & FRAMEWORK_PAIR_SELECTORS:
        return "PROPRIETARY_FRAMEWORK", "TX_ORIGIN_TRAP"

    # Check for L2-optimized scanning
    l2_count = sum(1 for s in selectors if _is_l2_optimized(s.get("selector", "")))
    if l2_count > len(selectors) * 0.5 and len(selectors) >= 3:
        return "L2_OPTIMIZED_SCANNER", "GENERIC_REVERT_TRAP"

    # Check for spray probe (many selectors, broad scanning)
    if sel_set & SPRAY_PROBE_SELECTORS:
        return "SPRAY_PROBER", "ANY_TRAP"

    # Check for router users
    if sel_set & ROUTER_SELECTORS:
        return "ROUTER_USER", "FEE_SKIM_TRAP"

    # Check for blind scanner (high revert, few selectors)
    if revert_count > 100 and len(selectors) <= 3:
        return "BLIND_SCANNER", "REVERT_DRAIN"

    # Fallback-only probers
    if sel_set <= FALLBACK_SELECTORS or (len(sel_set) == 1 and None in sel_set):
        return "FALLBACK_PROBER", "TX_ORIGIN_TRAP"

    # Single-selector grinders
    if len(selectors) == 1 and total_calls > 50:
        return "SINGLE_FUNCTION_GRINDER", "DEDICATED_COUNTER_TRAP"

    return "UNKNOWN", "UNKNOWN"


# ---------------------------------------------------------------
# Phase 1: Classify all bots
# ---------------------------------------------------------------

def classify_all_bots():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    # Ensure table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_address TEXT NOT NULL UNIQUE,
            strategy_type TEXT,
            primary_selectors TEXT,
            contracts_targeted INTEGER,
            avg_revert_rate REAL,
            total_interactions INTEGER,
            active_hours TEXT,
            avg_gas_per_tx REAL,
            first_seen TEXT,
            last_seen TEXT,
            trap_vulnerability TEXT,
            classified_at TEXT
        )
    """)

    bots = conn.execute("SELECT * FROM bot_candidates").fetchall()
    classified = 0

    for bot in bots:
        addr = bot["address"]

        # Get selectors
        sels = conn.execute(
            "SELECT function_selector as selector, call_count FROM bot_candidate_selectors WHERE bot_address=?",
            (addr,)
        ).fetchall()
        sel_list = [{"selector": s["selector"], "call_count": s["call_count"]} for s in sels]

        # Get interaction stats
        stats = conn.execute("""
            SELECT COUNT(DISTINCT contract_address) as contracts,
                   COUNT(*) as total_hits,
                   ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/NULLIF(COUNT(*),0)*100, 1) as rev_pct,
                   ROUND(AVG(gas_price_gwei), 2) as avg_gas,
                   MIN(timestamp) as first_seen, MAX(timestamp) as last_seen
            FROM transaction_events WHERE interacting_address=?
        """, (addr,)).fetchone()

        # Active hours
        hours = conn.execute("""
            SELECT CAST(substr(timestamp, 12, 2) AS INTEGER) as hour, COUNT(*) as cnt
            FROM transaction_events WHERE interacting_address=?
            GROUP BY hour ORDER BY cnt DESC LIMIT 5
        """, (addr,)).fetchall()
        active_hours = [{"hour": h["hour"], "count": h["cnt"]} for h in hours]

        # Classify
        strategy, vulnerability = classify_bot_strategy(
            sel_list, bot["total_revert_count"],
            stats["contracts"] if stats["contracts"] else 0
        )

        primary_sels = sorted(sel_list, key=lambda x: -x["call_count"])[:5]

        conn.execute("""
            INSERT OR REPLACE INTO bot_strategies
            (bot_address, strategy_type, primary_selectors, contracts_targeted,
             avg_revert_rate, total_interactions, active_hours, avg_gas_per_tx,
             first_seen, last_seen, trap_vulnerability, classified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            addr, strategy, json.dumps(primary_sels),
            stats["contracts"] or 0, stats["rev_pct"] or 0,
            stats["total_hits"] or 0, json.dumps(active_hours),
            stats["avg_gas"], stats["first_seen"], stats["last_seen"],
            vulnerability, _now()
        ))
        classified += 1

    conn.commit()

    # Print summary
    print(f"[strategy_fingerprint] Classified {classified} bots")
    for r in conn.execute("""
        SELECT strategy_type, COUNT(*) as bots,
            ROUND(AVG(avg_revert_rate), 1) as avg_rev,
            ROUND(AVG(total_interactions), 0) as avg_interactions,
            ROUND(AVG(contracts_targeted), 1) as avg_contracts
        FROM bot_strategies GROUP BY strategy_type ORDER BY bots DESC
    """):
        print(f"  {r['strategy_type']:25} | {r['bots']:>4} bots | {r['avg_rev']:>5.1f}% rev | {r['avg_interactions']:>6.0f} avg hits | {r['avg_contracts']:.1f} avg contracts")

    conn.close()


# ---------------------------------------------------------------
# Phase 2: Profile baits
# ---------------------------------------------------------------

def profile_baits():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bait_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address TEXT NOT NULL UNIQUE,
            bait_type TEXT,
            target_strategy TEXT,
            bait_selectors TEXT,
            trap_mechanism TEXT,
            bot_victims INTEGER,
            retail_victims INTEGER,
            bot_revert_rate REAL,
            retail_revert_rate REAL,
            effectiveness REAL,
            bytecode_family TEXT,
            first_seen TEXT,
            assessed_at TEXT,
            notes TEXT
        )
    """)

    # Get all contracts with bot interactions
    contracts = conn.execute("""
        SELECT te.contract_address,
            COUNT(DISTINCT CASE WHEN bs.bot_address IS NOT NULL THEN te.interacting_address END) as bot_callers,
            COUNT(DISTINCT CASE WHEN bs.bot_address IS NULL THEN te.interacting_address END) as retail_callers,
            COUNT(*) as total_hits,
            SUM(CASE WHEN bs.bot_address IS NOT NULL AND te.is_reverted=1 THEN 1 ELSE 0 END) as bot_reverts,
            SUM(CASE WHEN bs.bot_address IS NOT NULL THEN 1 ELSE 0 END) as bot_hits,
            SUM(CASE WHEN bs.bot_address IS NULL AND te.is_reverted=1 THEN 1 ELSE 0 END) as retail_reverts,
            SUM(CASE WHEN bs.bot_address IS NULL THEN 1 ELSE 0 END) as retail_hits,
            GROUP_CONCAT(DISTINCT te.function_selector) as all_sels,
            MIN(te.timestamp) as first_seen
        FROM transaction_events te
        LEFT JOIN bot_strategies bs ON te.interacting_address = bs.bot_address
        GROUP BY te.contract_address
        HAVING bot_callers >= 1 AND total_hits >= 5
    """).fetchall()

    profiled = 0
    for c in contracts:
        addr = c["contract_address"]
        bot_rev_rate = round(c["bot_reverts"] / max(c["bot_hits"], 1) * 100, 1)
        retail_rev_rate = round(c["retail_reverts"] / max(c["retail_hits"], 1) * 100, 1)

        # Get bytecode info
        contract_info = conn.execute(
            "SELECT bytecode_pattern_notes, confidence_tier FROM contracts WHERE contract_address=?",
            (addr,)
        ).fetchone()
        pattern_notes = contract_info["bytecode_pattern_notes"] if contract_info else ""

        # Get family
        fam = conn.execute(
            "SELECT family_id FROM bytecode_family_members WHERE contract_address=?",
            (addr,)
        ).fetchone()
        family_id = fam["family_id"] if fam else None

        # Determine bait type and target strategy from bytecode patterns
        bait_type = "UNKNOWN"
        target_strategy = "UNKNOWN"
        trap_mechanism = "UNKNOWN"

        if pattern_notes:
            if "callback" in pattern_notes.lower():
                bait_type = "CALLBACK_OPPORTUNITY"
                target_strategy = "CALLBACK_ARBITRAGE"
                trap_mechanism = "CALLBACK_STEAL"
            elif "tx.origin" in pattern_notes.lower():
                bait_type = "NEW_TOKEN"
                target_strategy = "BLIND_SCANNER"
                trap_mechanism = "BUY_NOT_SELL"
            elif "fee" in pattern_notes.lower():
                bait_type = "ROUTER_POOL"
                target_strategy = "ROUTER_USER"
                trap_mechanism = "FEE_SKIM"
            elif "delegatecall" in pattern_notes.lower():
                bait_type = "UPGRADEABLE_TOKEN"
                target_strategy = "L2_OPTIMIZED_SCANNER"
                trap_mechanism = "STATE_CHANGE"

        # If no bytecode pattern, infer from bot selectors used
        if bait_type == "UNKNOWN" and c["all_sels"]:
            sels = set(c["all_sels"].split(",")) if c["all_sels"] else set()
            if sels & CALLBACK_SELECTORS:
                bait_type = "CALLBACK_OPPORTUNITY"
                target_strategy = "CALLBACK_ARBITRAGE"
            elif any(_is_l2_optimized(s) for s in sels if s):
                bait_type = "L2_SCANNABLE_CONTRACT"
                target_strategy = "L2_OPTIMIZED_SCANNER"

        # Determine if it's effective bait
        if bot_rev_rate > 50:
            trap_mechanism = trap_mechanism if trap_mechanism != "UNKNOWN" else "HIGH_REVERT_TRAP"
        elif bot_rev_rate < 10 and retail_rev_rate > 30:
            trap_mechanism = "CAMOUFLAGED_SELECTIVE"

        # Calculate effectiveness: bot victims / total bots running this strategy
        strategy_bots = conn.execute(
            "SELECT COUNT(*) FROM bot_strategies WHERE strategy_type=?",
            (target_strategy,)
        ).fetchone()[0]
        effectiveness = round(c["bot_callers"] / max(strategy_bots, 1), 3)

        conn.execute("""
            INSERT OR REPLACE INTO bait_profiles
            (contract_address, bait_type, target_strategy, bait_selectors,
             trap_mechanism, bot_victims, retail_victims, bot_revert_rate,
             retail_revert_rate, effectiveness, bytecode_family, first_seen, assessed_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            addr, bait_type, target_strategy, c["all_sels"],
            trap_mechanism, c["bot_callers"], c["retail_callers"],
            bot_rev_rate, retail_rev_rate, effectiveness,
            family_id, c["first_seen"], _now(), pattern_notes or None
        ))
        profiled += 1

    conn.commit()

    print(f"[strategy_fingerprint] Profiled {profiled} bait contracts")
    for r in conn.execute("""
        SELECT bait_type, target_strategy, COUNT(*) as traps,
            SUM(bot_victims) as total_bot_victims,
            ROUND(AVG(bot_revert_rate), 1) as avg_bot_rev,
            ROUND(AVG(retail_revert_rate), 1) as avg_retail_rev
        FROM bait_profiles GROUP BY bait_type, target_strategy ORDER BY traps DESC
    """):
        print(f"  {r['bait_type']:25} -> {r['target_strategy']:25} | {r['traps']:>4} traps | {r['total_bot_victims']:>5} bot victims | bot_rev={r['avg_bot_rev']:.1f}% | retail_rev={r['avg_retail_rev']:.1f}%")

    conn.close()


# ---------------------------------------------------------------
# Phase 3: Predict new bait at deployment time
# ---------------------------------------------------------------

def predict_bait_for_contract(conn, contract_address: str) -> dict:
    """
    Given a contract, predict whether it's bait based on:
    1. Bytecode patterns (from classifier)
    2. Historical bait profile match rate for similar contracts
    3. Deployer history
    """
    c = conn.execute(
        "SELECT * FROM contracts WHERE contract_address=?",
        (contract_address,)
    ).fetchone()
    if not c:
        return {"is_bait": False, "confidence": 0, "reason": "not_found"}

    pattern = c["bytecode_pattern_notes"] or ""
    confidence = 0.0
    bait_type = "UNKNOWN"
    target_strategy = "UNKNOWN"

    # Rule 1: Bytecode pattern match
    if "callback" in pattern.lower():
        bait_type = "CALLBACK_OPPORTUNITY"
        target_strategy = "CALLBACK_ARBITRAGE"
        # What % of callback-pattern contracts in bait_profiles had high bot revert?
        stats = conn.execute("""
            SELECT COUNT(*) as total,
                SUM(CASE WHEN bot_revert_rate > 50 THEN 1 ELSE 0 END) as traps
            FROM bait_profiles WHERE bait_type='CALLBACK_OPPORTUNITY'
        """).fetchone()
        if stats["total"] > 0:
            confidence = stats["traps"] / stats["total"]
        else:
            confidence = 0.7  # prior: most callback-pattern contracts are traps
    elif "tx.origin" in pattern.lower():
        bait_type = "NEW_TOKEN"
        target_strategy = "BLIND_SCANNER"
        confidence = 0.9  # tx.origin in transfer context is almost always a trap
    elif "fee" in pattern.lower():
        bait_type = "ROUTER_POOL"
        target_strategy = "ROUTER_USER"
        confidence = 0.6
    elif "delegatecall" in pattern.lower():
        bait_type = "UPGRADEABLE_TOKEN"
        target_strategy = "L2_OPTIMIZED_SCANNER"
        confidence = 0.5

    # Rule 2: Deployer history
    deployer = c["deployer_address"]
    deployer_traps = conn.execute("""
        SELECT COUNT(*) FROM bait_profiles bp
        JOIN contracts c ON c.contract_address = bp.contract_address
        WHERE c.deployer_address=? AND bp.bot_revert_rate > 50
    """, (deployer,)).fetchone()[0]
    if deployer_traps > 0:
        confidence = min(confidence + 0.2, 1.0)

    # Rule 3: Bytecode family history
    fam = conn.execute(
        "SELECT family_id FROM bytecode_family_members WHERE contract_address=?",
        (contract_address,)
    ).fetchone()
    if fam:
        fam_stats = conn.execute("""
            SELECT COUNT(*) as total,
                SUM(CASE WHEN bp.bot_revert_rate > 50 THEN 1 ELSE 0 END) as traps
            FROM bytecode_family_members bfm
            JOIN bait_profiles bp ON bp.contract_address = bfm.contract_address
            WHERE bfm.family_id=?
        """, (fam["family_id"],)).fetchone()
        if fam_stats["total"] > 0:
            family_trap_rate = fam_stats["traps"] / fam_stats["total"]
            confidence = max(confidence, family_trap_rate)

    return {
        "contract_address": contract_address,
        "is_bait": confidence >= 0.5,
        "confidence": round(confidence, 2),
        "bait_type": bait_type,
        "target_strategy": target_strategy,
        "deployer_prior_traps": deployer_traps,
        "bytecode_family": fam["family_id"] if fam else None,
    }


def predict_new():
    """Run prediction on all suspected contracts not yet profiled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    unprofiled = conn.execute("""
        SELECT c.contract_address FROM contracts c
        LEFT JOIN bait_profiles bp ON bp.contract_address = c.contract_address
        WHERE bp.contract_address IS NULL
        AND c.confidence_tier = 'suspected'
        AND c.bytecode_pattern_notes IS NOT NULL
        AND c.bytecode_pattern_notes != ''
    """).fetchall()

    predictions = []
    for row in unprofiled:
        pred = predict_bait_for_contract(conn, row["contract_address"])
        if pred["is_bait"]:
            predictions.append(pred)

    predictions.sort(key=lambda x: -x["confidence"])
    print(f"[strategy_fingerprint] {len(predictions)} contracts predicted as bait (of {len(unprofiled)} unprofiled suspected)")
    for p in predictions[:20]:
        print(f"  {p['contract_address'][:18]}... | conf={p['confidence']:.2f} | {p['bait_type']:25} -> {p['target_strategy']:25} | deployer_traps={p['deployer_prior_traps']}")

    conn.close()
    return predictions


# ---------------------------------------------------------------
# Phase 4: Strategy lifecycle
# ---------------------------------------------------------------

def compute_lifecycle():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_lifecycle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_type TEXT NOT NULL,
            date TEXT NOT NULL,
            active_bots INTEGER,
            avg_bot_revert_rate REAL,
            trap_count INTEGER,
            saturation_index REAL,
            lifecycle_stage TEXT,
            avg_bot_profitability TEXT,
            notes TEXT,
            UNIQUE(strategy_type, date)
        )
    """)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    strategies = conn.execute("""
        SELECT strategy_type, COUNT(*) as bots,
            ROUND(AVG(avg_revert_rate), 1) as avg_rev,
            SUM(total_interactions) as total_hits
        FROM bot_strategies
        WHERE strategy_type != 'UNKNOWN'
        GROUP BY strategy_type
    """).fetchall()

    for s in strategies:
        st = s["strategy_type"]

        # Count traps targeting this strategy
        trap_count = conn.execute(
            "SELECT COUNT(*) FROM bait_profiles WHERE target_strategy=? AND bot_revert_rate > 30",
            (st,)
        ).fetchone()[0]

        saturation = round(trap_count / max(s["bots"], 1), 3)

        # Lifecycle stage
        if saturation < 0.1:
            stage = "EARLY"
        elif saturation < 1.0:
            stage = "ARMS_RACE"
        elif saturation < 3.0:
            stage = "WEAPONIZED"
        else:
            stage = "DEAD"

        # Profitability estimate
        avg_rev = s["avg_rev"] or 0
        if avg_rev < 20:
            profit = "PROFITABLE"
        elif avg_rev < 50:
            profit = "MARGINAL"
        elif avg_rev < 80:
            profit = "NEGATIVE"
        else:
            profit = "DEAD"

        conn.execute("""
            INSERT OR REPLACE INTO strategy_lifecycle
            (strategy_type, date, active_bots, avg_bot_revert_rate, trap_count,
             saturation_index, lifecycle_stage, avg_bot_profitability, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (st, today, s["bots"], avg_rev, trap_count, saturation, stage, profit, None))

    conn.commit()

    print(f"[strategy_fingerprint] Lifecycle computed for {len(strategies)} strategies")
    print(f"{'Strategy':30} | {'Bots':>5} | {'Traps':>5} | {'Sat.':>6} | {'Stage':12} | {'Profit':10} | {'Avg Rev':>7}")
    print("-" * 100)
    for r in conn.execute("SELECT * FROM strategy_lifecycle WHERE date=? ORDER BY saturation_index DESC", (today,)):
        print(f"  {r['strategy_type']:28} | {r['active_bots']:>5} | {r['trap_count']:>5} | {r['saturation_index']:>6.3f} | {r['lifecycle_stage']:12} | {r['avg_bot_profitability']:10} | {r['avg_bot_revert_rate']:>6.1f}%")

    conn.close()


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot strategy fingerprinting & bait detection")
    parser.add_argument("--classify-all", action="store_true", help="Phase 1: Classify all bots")
    parser.add_argument("--profile-baits", action="store_true", help="Phase 2: Profile bait contracts")
    parser.add_argument("--predict-new", action="store_true", help="Phase 3: Predict bait on unprofiled contracts")
    parser.add_argument("--lifecycle", action="store_true", help="Phase 4: Compute strategy lifecycle")
    parser.add_argument("--all", action="store_true", help="Run all phases in sequence")
    args = parser.parse_args()

    if args.all or args.classify_all:
        classify_all_bots()
    if args.all or args.profile_baits:
        profile_baits()
    if args.all or args.predict_new:
        predict_new()
    if args.all or args.lifecycle:
        compute_lifecycle()

    if not any([args.classify_all, args.profile_baits, args.predict_new, args.lifecycle, args.all]):
        parser.print_help()
