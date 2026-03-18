"""
Layer 3 -- Pattern Scanner

Identifies treasury-pattern addresses on Arbitrum by checking for
the gas-seed + capital-allocate + profit-return signature.

Cannot scan the entire chain (no enumerate endpoint). Instead:
1. Seeds from known addresses in our database (deployer funders,
   counterparties from operator traces)
2. Validates each candidate against the treasury pattern
3. Cross-references against our contracts table

Run: python3 -m surveillance.pattern_scanner
"""

import asyncio
import json
import os
import sqlite3
import sys
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

HTTP_URL = os.environ.get("ARB_WSS_URL", "").replace("wss://", "https://")
USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"


async def get_transfers(addr, direction, cats, contract_addrs=None, max_count=100):
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "category": cats, "maxCount": hex(max_count),
        "order": "desc", "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = addr
    else:
        params["toAddress"] = addr
    if contract_addrs:
        params["contractAddresses"] = contract_addrs

    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "alchemy_getAssetTransfers", "params": [params]
        }, timeout=30) as r:
            data = (await r.json()).get("result", {})
            return data.get("transfers", [])


async def check_treasury_pattern(addr):
    """
    Check if an address exhibits the treasury pattern:
    1. Sent ETH to 3+ addresses (0.005-0.1 each) = gas seeding
    2. Sent USDC to those same addresses = capital allocation
    3. Received USDC back from at least one = profit return

    Returns (is_match, evidence_dict)
    """
    evidence = {
        "address": addr,
        "gas_seeds": [],
        "usdc_allocations": [],
        "usdc_returns": [],
        "funded_addresses": set(),
        "usdc_recipients": set(),
        "usdc_returners": set(),
    }

    # 1. ETH outbound (recent 200 txs)
    eth_out = await get_transfers(addr, "from", ["external"], max_count=200)
    for t in eth_out:
        val = float(t.get("value") or 0)
        to = (t.get("to") or "").lower()
        if 0.005 <= val <= 0.1 and to:
            evidence["gas_seeds"].append({
                "to": to, "eth": val,
                "ts": t.get("metadata", {}).get("blockTimestamp", ""),
            })
            evidence["funded_addresses"].add(to)

    if len(evidence["funded_addresses"]) < 3:
        return False, evidence

    # 2. USDC outbound
    usdc_out = await get_transfers(addr, "from", ["erc20"], [USDC], max_count=200)
    for t in usdc_out:
        val = float(t.get("value") or 0)
        to = (t.get("to") or "").lower()
        if val > 0 and to:
            evidence["usdc_allocations"].append({
                "to": to, "usdc": val,
                "ts": t.get("metadata", {}).get("blockTimestamp", ""),
            })
            evidence["usdc_recipients"].add(to)

    # Check overlap: gas-seeded addresses that also got USDC
    overlap = evidence["funded_addresses"] & evidence["usdc_recipients"]
    if not overlap:
        return False, evidence

    # 3. USDC inbound — profit returns
    usdc_in = await get_transfers(addr, "to", ["erc20"], [USDC], max_count=200)
    for t in usdc_in:
        val = float(t.get("value") or 0)
        fr = (t.get("from") or "").lower()
        if val > 0 and fr:
            evidence["usdc_returns"].append({
                "from": fr, "usdc": val,
                "ts": t.get("metadata", {}).get("blockTimestamp", ""),
            })
            evidence["usdc_returners"].add(fr)

    # Check: did any USDC come back from an address we funded?
    return_overlap = evidence["usdc_recipients"] & evidence["usdc_returners"]
    if not return_overlap:
        return False, evidence

    # Check profit: any returner sent back more than they received?
    sent_to = {}
    for a in evidence["usdc_allocations"]:
        to = a["to"]
        sent_to[to] = sent_to.get(to, 0) + a["usdc"]
    recv_from = {}
    for a in evidence["usdc_returns"]:
        fr = a["from"]
        recv_from[fr] = recv_from.get(fr, 0) + a["usdc"]

    has_profit = False
    for addr_funded in return_overlap:
        sent = sent_to.get(addr_funded, 0)
        received = recv_from.get(addr_funded, 0)
        if received > sent:
            has_profit = True
            evidence["profit_from"] = {
                "address": addr_funded,
                "sent": sent,
                "received": received,
                "profit": received - sent,
            }
            break

    # Convert sets to lists for JSON
    evidence["funded_addresses"] = list(evidence["funded_addresses"])
    evidence["usdc_recipients"] = list(evidence["usdc_recipients"])
    evidence["usdc_returners"] = list(evidence["usdc_returners"])

    # Determine confidence
    gas_count = len(evidence["funded_addresses"])
    usdc_overlap = len(overlap)
    confidence = "low"
    if gas_count >= 5 and usdc_overlap >= 3 and has_profit:
        confidence = "high"
    elif gas_count >= 3 and usdc_overlap >= 2 and has_profit:
        confidence = "medium"
    elif gas_count >= 3 and usdc_overlap >= 1:
        confidence = "low"

    evidence["confidence"] = confidence
    evidence["gas_seed_count"] = gas_count
    evidence["usdc_overlap_count"] = len(overlap)
    evidence["has_profit_return"] = has_profit

    is_match = gas_count >= 3 and len(overlap) >= 1
    return is_match, evidence


def check_vanity_prefixes(addresses):
    """Check if 2+ addresses share a 4+ char prefix (after 0x)."""
    prefixes = {}
    for addr in addresses:
        if not addr or len(addr) < 10:
            continue
        prefix4 = addr[2:6].lower()  # first 4 hex chars after 0x
        prefix6 = addr[2:8].lower()  # first 6 hex chars after 0x
        prefixes.setdefault(prefix4, []).append(addr)

    vanity_groups = {}
    for prefix, addrs in prefixes.items():
        if len(addrs) >= 2:
            vanity_groups[prefix] = addrs

    return vanity_groups


async def scan_from_deployer_funders(conn):
    """
    Phase 1 seed: check funding sources of our known deployers.
    Any deployer's funder could be a treasury.
    """
    print("=" * 60)
    print("PHASE 1: Scanning deployer funding sources")
    print("=" * 60)

    # Get all deployers with funding_trail
    rows = conn.execute(
        "SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL"
    ).fetchall()

    funder_addresses = set()
    for row in rows:
        try:
            trail = json.loads(row[1])
            src = trail.get("funding_source", "")
            if src and len(src) == 42:
                funder_addresses.add(src.lower())
        except (json.JSONDecodeError, TypeError):
            pass

    # Also get unique USDC sources from the operator trace
    # (addresses that sent USDC to the confirmed operator)
    operator = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"
    op_usdc_in = await get_transfers(operator, "to", ["erc20"], [USDC], max_count=50)
    for t in op_usdc_in:
        fr = (t.get("from") or "").lower()
        if fr and len(fr) == 42 and fr != operator:
            funder_addresses.add(fr)

    # Add the known treasury for validation
    funder_addresses.add("0xf186cb00e49e18491db5783ff04fae3818102ff7")

    print(f"Candidate addresses to check: {len(funder_addresses)}")
    print()

    matches = []
    for i, addr in enumerate(funder_addresses):
        print(f"[{i+1}/{len(funder_addresses)}] Checking {addr[:18]}...", end=" ", flush=True)
        try:
            is_match, evidence = await check_treasury_pattern(addr)
            if is_match:
                conf = evidence.get("confidence", "low")
                gas = evidence.get("gas_seed_count", 0)
                profit = evidence.get("has_profit_return", False)
                print(f"MATCH [{conf}] gas_seeds={gas} profit={profit}")

                # Store in DB
                conn.execute(
                    """INSERT OR REPLACE INTO pattern_matches
                       (address, pattern_type, evidence, first_seen, last_seen, confidence, status)
                       VALUES (?, 'treasury_pattern', ?, ?, ?, ?, 'candidate')""",
                    (addr, json.dumps(evidence, default=str),
                     datetime.now(timezone.utc).isoformat(),
                     datetime.now(timezone.utc).isoformat(),
                     conf),
                )
                conn.commit()
                matches.append((addr, evidence))
            else:
                gas = len(evidence.get("funded_addresses", []))
                print(f"no match (gas_seeds={gas})")
        except Exception as e:
            print(f"error: {e}")
        await asyncio.sleep(0.2)

    return matches


async def validate_matches(conn, matches):
    """Phase 2: Check if treasury candidates funded contract deployers."""
    print()
    print("=" * 60)
    print("PHASE 2: Validating treasury candidates")
    print("=" * 60)

    our_deployers = set()
    rows = conn.execute("SELECT deployer_address FROM deployers").fetchall()
    for r in rows:
        our_deployers.add(r[0].lower())

    our_contracts = set()
    rows = conn.execute("SELECT contract_address FROM contracts").fetchall()
    for r in rows:
        our_contracts.add(r[0].lower())

    for addr, evidence in matches:
        print(f"\nValidating: {addr[:18]}...")
        funded = set(evidence.get("funded_addresses", []) + evidence.get("usdc_recipients", []))

        # Check overlap with our deployers
        deployer_overlap = funded & our_deployers
        if deployer_overlap:
            print(f"  DEPLOYER OVERLAP: {len(deployer_overlap)} addresses")
            for d in deployer_overlap:
                contracts = conn.execute(
                    "SELECT COUNT(*) FROM contracts WHERE deployer_address = ?", (d,)
                ).fetchone()[0]
                tier = conn.execute(
                    "SELECT confidence_tier, COUNT(*) FROM contracts WHERE deployer_address = ? GROUP BY confidence_tier",
                    (d,),
                ).fetchall()
                print(f"    {d}: {contracts} contracts, tiers={dict(tier)}")

        # Check vanity prefixes
        vanity = check_vanity_prefixes(list(funded))
        if vanity:
            print(f"  VANITY GROUPS: {len(vanity)}")
            for prefix, addrs in vanity.items():
                print(f"    0x{prefix}...: {len(addrs)} addresses")
                for a in addrs[:5]:
                    print(f"      {a}")


async def main():
    conn = db.init_db()

    # Ensure pattern_matches table exists
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

    matches = await scan_from_deployer_funders(conn)

    if matches:
        await validate_matches(conn, matches)
    else:
        print("\nNo treasury-pattern matches found in seed addresses.")

    # Summary
    print()
    print("=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    total = conn.execute("SELECT COUNT(*) FROM pattern_matches").fetchone()[0]
    high = conn.execute("SELECT COUNT(*) FROM pattern_matches WHERE confidence = 'high'").fetchone()[0]
    med = conn.execute("SELECT COUNT(*) FROM pattern_matches WHERE confidence = 'medium'").fetchone()[0]
    print(f"Total pattern matches: {total}")
    print(f"  High confidence: {high}")
    print(f"  Medium confidence: {med}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
