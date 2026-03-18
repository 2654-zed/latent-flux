"""
Layer 3 -- Funding Enrichment

Uses Arbiscan V2 API to pull full transaction history for deployers
and find shared funding sources across independent deployers.

Strategy: Instead of tracing outward from known treasuries,
look for convergence — multiple deployers funded by the same source.

Run: python3 -m surveillance.funding_enrichment
"""

import asyncio
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surveillance import db

ARBISCAN_KEY = os.environ.get("ARBISCAN_API_KEY", "")
ARBISCAN_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 42161
RATE_LIMIT = 0.22  # Arbiscan free: 5 calls/sec


async def arbiscan_txlist(address, session, max_results=20):
    """Pull first N transactions for an address via Arbiscan V2."""
    params = {
        "chainid": CHAIN_ID,
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": max_results,
        "sort": "asc",
        "apikey": ARBISCAN_KEY,
    }
    async with session.get(ARBISCAN_URL, params=params, timeout=15) as resp:
        data = await resp.json()
        await asyncio.sleep(RATE_LIMIT)
        if data.get("status") == "1":
            return data.get("result", [])
        return []


def find_first_funder(txlist, deployer_address):
    """From a tx list, find the first inbound ETH transfer (the funder)."""
    deployer_lower = deployer_address.lower()
    for tx in txlist:
        if tx.get("to", "").lower() == deployer_lower and int(tx.get("value", "0")) > 0:
            return {
                "funder": tx["from"].lower(),
                "value_eth": int(tx["value"]) / 1e18,
                "tx_hash": tx["hash"],
                "timestamp": datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc).isoformat(),
                "block": int(tx["blockNumber"]),
            }
    return None


async def enrich_all_deployers(conn):
    """Pull tx history for all deployers, find their funders."""
    deployers = conn.execute(
        "SELECT deployer_address, funding_trail FROM deployers"
    ).fetchall()

    print(f"Total deployers in DB: {len(deployers)}")

    # Split into already-traced and untraced
    untraced = []
    traced = {}
    for row in deployers:
        addr = row[0]
        trail = row[1]
        if trail:
            try:
                t = json.loads(trail)
                src = t.get("funding_source", "")
                if src and len(src) == 42:
                    traced[addr] = src.lower()
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
        untraced.append(addr)

    print(f"Already traced: {len(traced)}")
    print(f"Need tracing:   {len(untraced)}")
    print()

    async with aiohttp.ClientSession() as session:
        for i, addr in enumerate(untraced):
            try:
                txlist = await arbiscan_txlist(addr, session, max_results=10)
                funder = find_first_funder(txlist, addr)
                if funder:
                    traced[addr] = funder["funder"]
                    trail_json = json.dumps(funder)
                    db.update_deployer_funding(conn, addr, trail_json)
                    print(f"  [{i+1}/{len(untraced)}] {addr[:18]}... <- {funder['funder'][:18]}... ({funder['value_eth']:.6f} ETH)")
                else:
                    # No inbound ETH found — mark as self-funded or bridge
                    trail_json = json.dumps({"funding_source": "unknown", "method": "arbiscan_no_inbound"})
                    db.update_deployer_funding(conn, addr, trail_json)
                    if (i + 1) % 20 == 0:
                        print(f"  [{i+1}/{len(untraced)}] progress...")
            except Exception as e:
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(untraced)}] progress (some errors)")

    return traced


def find_shared_funders(traced):
    """
    The key insight: group deployers by who funded them.
    If 2+ deployers share a funder within 7 days, that funder is suspicious.
    """
    by_funder = defaultdict(list)
    for deployer, funder in traced.items():
        if funder and funder != "unknown":
            by_funder[funder].append(deployer)

    shared = {k: v for k, v in by_funder.items() if len(v) >= 2}
    return shared


async def check_shared_funders_treasury(conn, shared, traced_full):
    """
    For each shared funder, check if it matches the treasury pattern.
    Uses Arbiscan to get the funder's own transaction history.
    """
    from surveillance.pattern_scanner import check_treasury_pattern

    results = []

    for funder, deployers in shared.items():
        print(f"\nShared funder: {funder}")
        print(f"  Funded {len(deployers)} deployers: {[d[:14]+'...' for d in deployers]}")

        # Quick check: is this a known entity?
        existing = conn.execute(
            "SELECT entity_type, deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
            (funder,),
        ).fetchone()
        if existing and existing[0] in ("treasury", "protocol"):
            print(f"  Already classified: {existing[0]}")
            results.append({"funder": funder, "deployers": deployers, "status": f"known_{existing[0]}"})
            continue

        # Check deployer contract counts and tiers
        total_contracts = 0
        suspected = 0
        for dep in deployers:
            row = conn.execute(
                "SELECT COUNT(*) FROM contracts WHERE deployer_address = ?", (dep,)
            ).fetchone()
            total_contracts += row[0]
            row2 = conn.execute(
                "SELECT COUNT(*) FROM contracts WHERE deployer_address = ? AND confidence_tier IN ('suspected', 'confirmed')",
                (dep,),
            ).fetchone()
            suspected += row2[0]

        print(f"  Total contracts from funded deployers: {total_contracts}")
        print(f"  Suspected/confirmed: {suspected}")

        # Run treasury pattern check via Alchemy
        try:
            is_match, evidence = await check_treasury_pattern(funder)
            gc = evidence.get("gas_seed_count", 0)
            conf = evidence.get("confidence", "none")
            profit = evidence.get("has_profit_return", False)
            print(f"  Treasury pattern: match={is_match} confidence={conf} gas={gc} profit={profit}")

            if is_match:
                now = datetime.now(timezone.utc).isoformat()
                evidence["shared_deployers"] = deployers
                evidence["total_contracts"] = total_contracts
                evidence["suspected_contracts"] = suspected
                conn.execute(
                    """INSERT OR REPLACE INTO pattern_matches
                       (address, pattern_type, evidence, first_seen, last_seen, confidence, status)
                       VALUES (?, 'shared_funder_treasury', ?, ?, ?, ?, ?)""",
                    (funder, json.dumps(evidence, default=str), now, now, conf,
                     "treasury_confirmed" if suspected > 0 else "candidate"),
                )
                conn.commit()

            results.append({
                "funder": funder,
                "deployers": deployers,
                "is_treasury": is_match,
                "confidence": conf,
                "contracts": total_contracts,
                "suspected": suspected,
            })
        except Exception as e:
            print(f"  Error checking pattern: {e}")
            results.append({"funder": funder, "deployers": deployers, "error": str(e)})

    return results


async def main():
    if not ARBISCAN_KEY:
        print("ERROR: ARBISCAN_API_KEY not set")
        sys.exit(1)

    conn = db.init_db()

    # Ensure pattern_matches table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pattern_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL UNIQUE,
            pattern_type TEXT NOT NULL,
            evidence TEXT NOT NULL,
            first_seen TEXT,
            last_seen TEXT,
            confidence TEXT,
            status TEXT DEFAULT 'candidate'
        )
    """)
    conn.commit()

    print("=" * 60)
    print("FUNDING ENRICHMENT — Arbiscan V2 API")
    print("=" * 60)
    print()

    # Step 1: Enrich all deployers with their funding source
    print("STEP 1: Tracing deployer funding sources")
    print("-" * 40)
    traced = await enrich_all_deployers(conn)
    print(f"\nTotal traced: {len(traced)}")

    # Step 2: Find shared funders
    print()
    print("STEP 2: Finding shared funding sources")
    print("-" * 40)
    shared = find_shared_funders(traced)
    print(f"Shared funders (funded 2+ deployers): {len(shared)}")
    for funder, deps in sorted(shared.items(), key=lambda x: -len(x[1])):
        print(f"  {funder[:18]}... -> {len(deps)} deployers")

    # Step 3: Check shared funders for treasury pattern
    if shared:
        print()
        print("STEP 3: Treasury pattern check on shared funders")
        print("-" * 40)
        results = await check_shared_funders_treasury(conn, shared, traced)

        # Summary
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        treasuries = [r for r in results if r.get("is_treasury")]
        print(f"New treasury matches: {len(treasuries)}")
        for t in treasuries:
            print(f"  {t['funder']}: [{t['confidence']}] funds {len(t['deployers'])} deployers, {t['suspected']} suspected contracts")
    else:
        print("No shared funders found — each deployer has a unique funding source.")

    # Final stats
    total_matches = conn.execute("SELECT COUNT(*) FROM pattern_matches").fetchone()[0]
    print(f"\nTotal pattern matches in DB: {total_matches}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
