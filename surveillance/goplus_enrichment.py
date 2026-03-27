"""
Layer 3 — GoPlus Security API Enrichment

Enriches our contract and address data with GoPlus security assessments.
Used as a validation layer — compares our detection against an independent source.

Free tier, no API key required. Rate limited to ~30 req/min.

Endpoints used:
    - Token Security: honeypot detection, trade simulation
    - Address Security: malicious address labels
    - Approval Security: risky approvals

Usage:
    python -m surveillance.goplus_enrichment --check-suspected
    python -m surveillance.goplus_enrichment --check-address 0x...
    python -m surveillance.goplus_enrichment --benchmark
"""

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

# GoPlus chain IDs
CHAIN_IDS = {
    "base": "8453",
    "arbitrum": "42161",
    "ethereum": "1",
}

# Rate limiting
RATE_LIMIT_DELAY = 2.1  # seconds between requests (~30/min free tier)

BASE_URL = "https://api.gopluslabs.io/api/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------
# API calls
# ---------------------------------------------------------------

async def check_token_security(session: aiohttp.ClientSession,
                               contract_address: str, chain: str) -> dict:
    """Check token security via GoPlus. Returns honeypot status, trade risks."""
    chain_id = CHAIN_IDS.get(chain, "8453")
    url = f"{BASE_URL}/token_security/{chain_id}?contract_addresses={contract_address}"
    try:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
            result = data.get("result", {})
            # GoPlus returns results keyed by lowercase address
            token_data = result.get(contract_address.lower(), {})
            return token_data
    except Exception as e:
        return {"error": str(e)}


async def check_address_security(session: aiohttp.ClientSession,
                                 address: str, chain: str = "base") -> dict:
    """Check address for malicious activity flags."""
    chain_id = CHAIN_IDS.get(chain, "8453")
    url = f"{BASE_URL}/address_security/{address}?chain_id={chain_id}"
    try:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
            return data.get("result", {})
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------
# Enrichment logic
# ---------------------------------------------------------------

def parse_token_result(raw: dict) -> dict:
    """Parse GoPlus token security response into our assessment."""
    if not raw or "error" in raw:
        return {"goplus_status": "error", "detail": raw.get("error", "empty response")}

    assessment = {
        "goplus_status": "checked",
        "is_honeypot": raw.get("is_honeypot", "unknown"),
        "honeypot_with_same_creator": raw.get("honeypot_with_same_creator", "0"),
        "is_open_source": raw.get("is_open_source", "unknown"),
        "is_proxy": raw.get("is_proxy", "unknown"),
        "can_take_back_ownership": raw.get("can_take_back_ownership", "unknown"),
        "cannot_buy": raw.get("cannot_buy", "unknown"),
        "cannot_sell_all": raw.get("cannot_sell_all", "unknown"),
        "slippage_modifiable": raw.get("slippage_modifiable", "unknown"),
        "personal_slippage_modifiable": raw.get("personal_slippage_modifiable", "unknown"),
        "trading_cooldown": raw.get("trading_cooldown", "unknown"),
        "is_anti_whale": raw.get("is_anti_whale", "unknown"),
        "transfer_pausable": raw.get("transfer_pausable", "unknown"),
        "is_blacklisted": raw.get("is_blacklisted", "unknown"),
        "is_whitelisted": raw.get("is_whitelisted", "unknown"),
        "is_true_token": raw.get("is_true_token", "unknown"),
        "external_call": raw.get("external_call", "unknown"),
        "selfdestruct": raw.get("selfdestruct", "unknown"),
        "owner_change_balance": raw.get("owner_change_balance", "unknown"),
        "buy_tax": raw.get("buy_tax", "unknown"),
        "sell_tax": raw.get("sell_tax", "unknown"),
        "holder_count": raw.get("holder_count", "0"),
        "total_supply": raw.get("total_supply", "unknown"),
        "token_name": raw.get("token_name", ""),
        "token_symbol": raw.get("token_symbol", ""),
    }

    # Compute risk score
    risks = 0
    if assessment["is_honeypot"] == "1":
        risks += 3
    if assessment["cannot_sell_all"] == "1":
        risks += 3
    if assessment["cannot_buy"] == "1":
        risks += 2
    if assessment["owner_change_balance"] == "1":
        risks += 2
    if assessment["slippage_modifiable"] == "1":
        risks += 1
    if assessment["transfer_pausable"] == "1":
        risks += 1
    if assessment["is_blacklisted"] == "1":
        risks += 1
    if assessment["external_call"] == "1":
        risks += 1
    if assessment["selfdestruct"] == "1":
        risks += 1

    try:
        buy_tax = float(assessment["buy_tax"]) if assessment["buy_tax"] not in ("unknown", "", None) else 0
        sell_tax = float(assessment["sell_tax"]) if assessment["sell_tax"] not in ("unknown", "", None) else 0
        if sell_tax > 0.5:
            risks += 3
        elif sell_tax > 0.1:
            risks += 1
        assessment["buy_tax_pct"] = round(buy_tax * 100, 1)
        assessment["sell_tax_pct"] = round(sell_tax * 100, 1)
    except (ValueError, TypeError):
        assessment["buy_tax_pct"] = None
        assessment["sell_tax_pct"] = None

    assessment["risk_score"] = risks
    assessment["risk_level"] = (
        "CRITICAL" if risks >= 5
        else "HIGH" if risks >= 3
        else "MEDIUM" if risks >= 1
        else "LOW"
    )

    return assessment


def parse_address_result(raw: dict) -> dict:
    """Parse GoPlus address security response."""
    if not raw or "error" in raw:
        return {"goplus_status": "error"}

    flags = []
    for key in ["cybercrime", "money_laundering", "financial_crime",
                 "phishing_activities", "stealing_attack", "blackmail_activities",
                 "fake_kyc", "sanctioned", "malicious_mining_activities",
                 "mixer", "honeypot_related_address", "fake_token",
                 "blacklist_doubt", "gas_abuse", "darkweb_transactions"]:
        if raw.get(key) == "1":
            flags.append(key)

    malicious_contracts = int(raw.get("number_of_malicious_contracts_created", 0))

    return {
        "goplus_status": "checked",
        "flags": flags,
        "flag_count": len(flags),
        "malicious_contracts_created": malicious_contracts,
        "is_contract": raw.get("contract_address") == "1",
    }


# ---------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goplus_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            check_type TEXT NOT NULL,
            chain TEXT,
            goplus_status TEXT,
            risk_level TEXT,
            risk_score INTEGER,
            is_honeypot TEXT,
            flags TEXT,
            our_tier TEXT,
            match TEXT,
            raw_result TEXT,
            checked_at TEXT,
            UNIQUE(address, check_type)
        )
    """)
    conn.commit()


def store_result(conn: sqlite3.Connection, address: str, check_type: str,
                 chain: str, parsed: dict, our_tier: str = None):
    """Store enrichment result and compute match vs our classification."""
    # Determine if GoPlus agrees with us
    match = "unknown"
    if check_type == "token" and our_tier:
        goplus_says_bad = parsed.get("risk_level") in ("HIGH", "CRITICAL")
        we_say_bad = our_tier in ("suspected", "confirmed")
        if goplus_says_bad and we_say_bad:
            match = "BOTH_FLAGGED"
        elif goplus_says_bad and not we_say_bad:
            match = "GOPLUS_ONLY"
        elif not goplus_says_bad and we_say_bad:
            match = "L3_ONLY"  # We detected it, they didn't
        else:
            match = "BOTH_CLEAN"

    conn.execute("""
        INSERT OR REPLACE INTO goplus_results
        (address, check_type, chain, goplus_status, risk_level, risk_score,
         is_honeypot, flags, our_tier, match, raw_result, checked_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        address.lower(), check_type, chain,
        parsed.get("goplus_status"),
        parsed.get("risk_level"),
        parsed.get("risk_score"),
        parsed.get("is_honeypot"),
        json.dumps(parsed.get("flags", [])),
        our_tier, match,
        json.dumps(parsed),
        _now(),
    ))
    conn.commit()


# ---------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------

async def check_suspected_contracts(limit: int = 100):
    """Check our suspected contracts against GoPlus token security."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)

    # Get suspected contracts not yet checked
    unchecked = conn.execute("""
        SELECT c.contract_address, c.chain, c.confidence_tier, c.bytecode_pattern_notes
        FROM contracts c
        LEFT JOIN goplus_results g ON c.contract_address = g.address AND g.check_type = 'token'
        WHERE c.confidence_tier IN ('suspected', 'confirmed')
        AND g.address IS NULL
        ORDER BY c.detection_timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()

    print(f"[goplus] Checking {len(unchecked)} suspected contracts...")
    checked = 0
    results_summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "error": 0, "empty": 0}

    async with aiohttp.ClientSession() as session:
        for row in unchecked:
            raw = await check_token_security(session, row["contract_address"], row["chain"])

            if raw and "error" not in raw and raw:
                parsed = parse_token_result(raw)
                store_result(conn, row["contract_address"], "token",
                             row["chain"], parsed, row["confidence_tier"])
                results_summary[parsed.get("risk_level", "error")] += 1
                checked += 1

                if parsed.get("risk_level") in ("HIGH", "CRITICAL"):
                    print(f"  GOPLUS CONFIRMS: {row['contract_address'][:18]}... | {parsed['risk_level']} | honeypot={parsed.get('is_honeypot')} | sell_tax={parsed.get('sell_tax_pct')}%")
            else:
                # Empty result = GoPlus doesn't know this contract
                store_result(conn, row["contract_address"], "token",
                             row["chain"], {"goplus_status": "not_found", "risk_level": "UNKNOWN"},
                             row["confidence_tier"])
                results_summary["empty"] += 1
                checked += 1

            await asyncio.sleep(RATE_LIMIT_DELAY)

    # Summary
    print(f"[goplus] Checked {checked} contracts:")
    for level, count in sorted(results_summary.items()):
        if count > 0:
            print(f"  {level}: {count}")

    # Benchmark: how many did we catch that GoPlus missed?
    l3_only = conn.execute("""
        SELECT COUNT(*) FROM goplus_results WHERE match = 'L3_ONLY'
    """).fetchone()[0]
    both = conn.execute("""
        SELECT COUNT(*) FROM goplus_results WHERE match = 'BOTH_FLAGGED'
    """).fetchone()[0]
    goplus_only = conn.execute("""
        SELECT COUNT(*) FROM goplus_results WHERE match = 'GOPLUS_ONLY'
    """).fetchone()[0]
    total_checked = conn.execute("SELECT COUNT(*) FROM goplus_results WHERE check_type='token'").fetchone()[0]

    print(f"\n[goplus] Detection comparison (total: {total_checked}):")
    print(f"  BOTH_FLAGGED:  {both} (we agree)")
    print(f"  L3_ONLY:       {l3_only} (we caught, they missed)")
    print(f"  GOPLUS_ONLY:   {goplus_only} (they caught, we missed)")

    if l3_only + both > 0:
        advantage = l3_only / (l3_only + both) * 100
        print(f"  Detection advantage: {advantage:.0f}% of our flags are invisible to GoPlus")

    conn.close()


async def check_addresses(addresses: list[str], chain: str = "base"):
    """Check a list of addresses against GoPlus address security."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)

    print(f"[goplus] Checking {len(addresses)} addresses...")
    async with aiohttp.ClientSession() as session:
        for addr in addresses:
            raw = await check_address_security(session, addr, chain)
            parsed = parse_address_result(raw)
            store_result(conn, addr, "address", chain, parsed)

            if parsed.get("flag_count", 0) > 0:
                print(f"  FLAGGED: {addr[:18]}... | flags={parsed['flags']}")
            await asyncio.sleep(RATE_LIMIT_DELAY)

    conn.close()


async def benchmark():
    """Run a detection benchmark: compare our classifications vs GoPlus."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)

    # Check top 50 suspected contracts with most activity
    top = conn.execute("""
        SELECT c.contract_address, c.chain, c.confidence_tier, c.bytecode_pattern_notes,
            COUNT(te.tx_hash) as hits
        FROM contracts c
        JOIN transaction_events te ON te.contract_address = c.contract_address
        LEFT JOIN goplus_results g ON c.contract_address = g.address AND g.check_type = 'token'
        WHERE c.confidence_tier = 'suspected'
        AND g.address IS NULL
        AND c.bytecode_pattern_notes IS NOT NULL AND c.bytecode_pattern_notes != ''
        GROUP BY c.contract_address
        ORDER BY hits DESC
        LIMIT 50
    """).fetchall()

    print(f"[goplus] Benchmarking top {len(top)} active suspected contracts...")

    async with aiohttp.ClientSession() as session:
        for row in top:
            raw = await check_token_security(session, row["contract_address"], row["chain"])
            if raw:
                parsed = parse_token_result(raw)
                store_result(conn, row["contract_address"], "token",
                             row["chain"], parsed, "suspected")
                status = f"GOPLUS: {parsed.get('risk_level', '?')}"
                honeypot = f"honeypot={parsed.get('is_honeypot', '?')}"
                print(f"  {row['contract_address'][:18]}... | {row['hits']:>5} hits | L3=suspected | {status} | {honeypot} | {row['bytecode_pattern_notes'][:30]}")
            else:
                store_result(conn, row["contract_address"], "token",
                             row["chain"], {"goplus_status": "not_found", "risk_level": "UNKNOWN"},
                             "suspected")
                print(f"  {row['contract_address'][:18]}... | {row['hits']:>5} hits | L3=suspected | GOPLUS: not found")

            await asyncio.sleep(RATE_LIMIT_DELAY)

    # Print final comparison
    print("\n=== BENCHMARK RESULTS ===")
    for r in conn.execute("""
        SELECT match, COUNT(*) as c FROM goplus_results
        WHERE check_type='token' GROUP BY match ORDER BY c DESC
    """):
        print(f"  {r['match']}: {r['c']}")

    conn.close()


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoPlus Security API enrichment")
    parser.add_argument("--check-suspected", action="store_true", help="Check suspected contracts")
    parser.add_argument("--check-address", type=str, help="Check a single address")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark our detection vs GoPlus")
    parser.add_argument("--limit", type=int, default=50, help="Max contracts to check")
    args = parser.parse_args()

    if args.check_suspected:
        asyncio.run(check_suspected_contracts(args.limit))
    elif args.check_address:
        asyncio.run(check_addresses([args.check_address]))
    elif args.benchmark:
        asyncio.run(benchmark())
    else:
        parser.print_help()
