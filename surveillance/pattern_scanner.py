"""
Layer 3 -- Pattern Scanner (v2)

Systematic detection of treasury-pattern addresses on Arbitrum.

Strategy: We can't enumerate the chain, so we mine seed addresses from:
1. Our deployment monitor's deployer table — trace each deployer's funder
2. Our operator's USDC counterparties
3. Bot candidates that are also deployers
4. Any address that appears in multiple roles

Three passes:
  Pass 1: Treasury detection (gas-seed + capital-allocate + profit-return)
  Pass 2: Operator validation (funded address deployed contracts?)
  Pass 3: Vanity address check (shared prefixes = same entity)

Run: python3 -m surveillance.pattern_scanner
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

HTTP_URL = os.environ.get("ARB_WSS_URL", "").replace("wss://", "https://")
USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

RATE_LIMIT = 0.2  # 200ms between Alchemy calls


async def _alchemy(params):
    """Single alchemy_getAssetTransfers call with rate limiting."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [params]}
    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json=payload, timeout=45) as r:
            data = (await r.json()).get("result", {})
            await asyncio.sleep(RATE_LIMIT)
            return data.get("transfers", []), data.get("pageKey")


async def get_transfers(addr, direction, cats, contract_addrs=None, max_results=200):
    """Paginated transfer fetch."""
    all_t = []
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "category": cats, "maxCount": hex(min(max_results, 1000)),
        "order": "desc", "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = addr
    else:
        params["toAddress"] = addr
    if contract_addrs:
        params["contractAddresses"] = contract_addrs

    transfers, pk = await _alchemy(params)
    all_t.extend(transfers)
    while pk and len(all_t) < max_results:
        params["pageKey"] = pk
        transfers, pk = await _alchemy(params)
        all_t.extend(transfers)

    return all_t[:max_results]


# ---------------------------------------------------------------
# SEED GENERATION — find addresses worth checking
# ---------------------------------------------------------------

async def generate_seeds(conn) -> set:
    """Build the set of candidate addresses to check for treasury pattern."""
    seeds = set()

    # Source 1: All deployer funding sources from our DB
    rows = conn.execute(
        "SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            trail = json.loads(row[1])
            src = trail.get("funding_source", "")
            if src and len(src) == 42:
                seeds.add(src.lower())
        except (json.JSONDecodeError, TypeError):
            pass

    # Source 2: For deployers WITHOUT funding_trail, trace their funder live
    untraced = conn.execute(
        "SELECT deployer_address FROM deployers WHERE funding_trail IS NULL LIMIT 50"
    ).fetchall()
    print(f"Tracing funding for {len(untraced)} untraced deployers...")
    for i, row in enumerate(untraced):
        addr = row[0]
        try:
            inbound = await get_transfers(addr, "to", ["external", "internal"], max_results=5)
            for t in inbound:
                val = float(t.get("value") or 0)
                if val > 0:
                    funder = (t.get("from") or "").lower()
                    if funder and len(funder) == 42:
                        seeds.add(funder)
                        # Store the funding trail
                        trail = json.dumps({
                            "funding_source": funder,
                            "value_eth": val,
                            "method": "pattern_scanner_trace",
                        })
                        db.update_deployer_funding(conn, addr, trail)
                    break
        except Exception as e:
            pass
        if (i + 1) % 10 == 0:
            print(f"  traced {i+1}/{len(untraced)}")

    # Source 3: USDC counterparties of confirmed operators
    operators = conn.execute(
        "SELECT deployer_address FROM deployers WHERE entity_type = 'operator'"
    ).fetchall()
    for row in operators:
        op = row[0]
        usdc_in = await get_transfers(op, "to", ["erc20"], [USDC], max_results=50)
        for t in usdc_in:
            fr = (t.get("from") or "").lower()
            if fr and len(fr) == 42 and fr != op:
                seeds.add(fr)

    # Source 4: Bot candidates that are deployers
    rows = conn.execute(
        "SELECT address FROM bot_candidates WHERE is_deployer = 1"
    ).fetchall()
    for row in rows:
        seeds.add(row[0].lower())

    # Remove addresses we've already classified
    known = set()
    for row in conn.execute("SELECT deployer_address, entity_type FROM deployers WHERE entity_type IS NOT NULL").fetchall():
        if row[1] in ("treasury", "operator", "cashout", "protocol"):
            known.add(row[0].lower())
    seeds -= known

    return seeds


# ---------------------------------------------------------------
# PASS 1 — Treasury detection
# ---------------------------------------------------------------

async def check_treasury_pattern(addr) -> tuple[bool, dict]:
    """
    Check if an address exhibits the treasury pattern:
    1. Sent ETH to 3+ addresses (0.003-0.1 each) = gas seeding
    2. Sent USDC to any of those same addresses = capital allocation
    3. Received USDC back from at least one in larger amount = profit return
    """
    evidence = {
        "address": addr,
        "funded_addresses": [],
        "usdc_recipients": [],
        "usdc_returners": [],
        "gas_seed_count": 0,
        "usdc_overlap_count": 0,
        "has_profit_return": False,
        "confidence": "none",
    }

    # Step 1: ETH outbound — find gas seeds
    eth_out = await get_transfers(addr, "from", ["external"], max_results=200)
    funded = set()
    for t in eth_out:
        val = float(t.get("value") or 0)
        to = (t.get("to") or "").lower()
        if 0.003 <= val <= 0.1 and to:
            funded.add(to)

    evidence["funded_addresses"] = list(funded)
    evidence["gas_seed_count"] = len(funded)

    if len(funded) < 3:
        return False, evidence

    # Step 2: USDC outbound — capital allocation
    usdc_out = await get_transfers(addr, "from", ["erc20"], [USDC], max_results=200)
    usdc_recip = set()
    sent_to = defaultdict(float)
    for t in usdc_out:
        val = float(t.get("value") or 0)
        to = (t.get("to") or "").lower()
        if val > 0 and to:
            usdc_recip.add(to)
            sent_to[to] += val

    evidence["usdc_recipients"] = list(usdc_recip)
    overlap = funded & usdc_recip
    evidence["usdc_overlap_count"] = len(overlap)

    if not overlap:
        return False, evidence

    # Step 3: USDC inbound — profit returns
    usdc_in = await get_transfers(addr, "to", ["erc20"], [USDC], max_results=200)
    returners = set()
    recv_from = defaultdict(float)
    for t in usdc_in:
        val = float(t.get("value") or 0)
        fr = (t.get("from") or "").lower()
        if val > 0 and fr:
            returners.add(fr)
            recv_from[fr] += val

    evidence["usdc_returners"] = list(returners)
    return_overlap = usdc_recip & returners

    # Check for profit (received > sent from any funded address)
    for addr_funded in return_overlap:
        sent = sent_to.get(addr_funded, 0)
        received = recv_from.get(addr_funded, 0)
        if received > sent:
            evidence["has_profit_return"] = True
            evidence["profit_example"] = {
                "address": addr_funded,
                "sent": round(sent, 2),
                "received": round(received, 2),
                "profit": round(received - sent, 2),
            }
            break

    # Confidence scoring
    gc = evidence["gas_seed_count"]
    oc = evidence["usdc_overlap_count"]
    profit = evidence["has_profit_return"]
    if gc >= 5 and oc >= 3 and profit:
        evidence["confidence"] = "high"
    elif gc >= 3 and oc >= 2 and profit:
        evidence["confidence"] = "medium"
    elif gc >= 3 and oc >= 1:
        evidence["confidence"] = "low"

    is_match = gc >= 3 and oc >= 1
    return is_match, evidence


# ---------------------------------------------------------------
# PASS 2 — Operator validation
# ---------------------------------------------------------------

async def validate_operators(conn, treasury_addr, funded_addresses):
    """Check if any funded address deployed contracts in our DB."""
    our_deployers = {r[0].lower() for r in conn.execute("SELECT deployer_address FROM deployers").fetchall()}
    our_contracts = {r[0].lower() for r in conn.execute("SELECT contract_address FROM contracts").fetchall()}

    results = {
        "deployer_matches": [],
        "contract_matches": [],
    }

    funded_set = set(a.lower() for a in funded_addresses)
    deployer_overlap = funded_set & our_deployers

    for dep in deployer_overlap:
        contracts = conn.execute(
            "SELECT contract_address, confidence_tier FROM contracts WHERE deployer_address = ?",
            (dep,),
        ).fetchall()
        tiers = defaultdict(int)
        for c in contracts:
            tiers[c[1]] += 1
        results["deployer_matches"].append({
            "deployer": dep,
            "contracts": len(contracts),
            "tiers": dict(tiers),
        })

    return results


# ---------------------------------------------------------------
# PASS 3 — Vanity address check
# ---------------------------------------------------------------

def check_vanity(addresses):
    """Find addresses sharing 4+ char prefixes."""
    by_prefix = defaultdict(list)
    for addr in addresses:
        if not addr or len(addr) < 10:
            continue
        p = addr[2:6].lower()
        by_prefix[p].append(addr)

    return {k: v for k, v in by_prefix.items() if len(v) >= 2}


# ---------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------

async def run_scan():
    conn = db.init_db()

    # Ensure table exists
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

    # Generate seeds
    print("=" * 60)
    print("GENERATING SEED ADDRESSES")
    print("=" * 60)
    seeds = await generate_seeds(conn)
    print(f"\nTotal seed addresses: {len(seeds)}")

    # PASS 1: Treasury detection
    print()
    print("=" * 60)
    print("PASS 1: TREASURY DETECTION")
    print("=" * 60)

    matches = []
    batch_size = 10
    seed_list = list(seeds)

    for batch_start in range(0, len(seed_list), batch_size):
        batch = seed_list[batch_start:batch_start + batch_size]
        for addr in batch:
            short = addr[:18] + "..."
            try:
                is_match, evidence = await check_treasury_pattern(addr)
                gc = evidence["gas_seed_count"]
                if is_match:
                    conf = evidence["confidence"]
                    print(f"  MATCH [{conf}]: {short} gas={gc} overlap={evidence['usdc_overlap_count']} profit={evidence['has_profit_return']}")
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """INSERT OR REPLACE INTO pattern_matches
                           (address, pattern_type, evidence, first_seen, last_seen, confidence, status)
                           VALUES (?, 'treasury_candidate', ?, ?, ?, ?, 'candidate')""",
                        (addr, json.dumps(evidence, default=str), now, now, conf),
                    )
                    conn.commit()
                    matches.append((addr, evidence))
                elif gc > 0:
                    print(f"  partial: {short} gas={gc} (need 3+)")
                # else: silent skip
            except Exception as e:
                print(f"  error: {short} {e}")

        processed = min(batch_start + batch_size, len(seed_list))
        print(f"  [{processed}/{len(seed_list)}] processed")

    print(f"\nPass 1 complete: {len(matches)} treasury candidates found")

    # PASS 2: Operator validation
    if matches:
        print()
        print("=" * 60)
        print("PASS 2: OPERATOR VALIDATION")
        print("=" * 60)

        for addr, evidence in matches:
            funded = evidence.get("funded_addresses", []) + evidence.get("usdc_recipients", [])
            results = await validate_operators(conn, addr, funded)

            if results["deployer_matches"]:
                print(f"\n  {addr[:18]}... -> DEPLOYER OVERLAP")
                for dm in results["deployer_matches"]:
                    print(f"    {dm['deployer']}: {dm['contracts']} contracts {dm['tiers']}")

                # Upgrade to confirmed
                conn.execute(
                    "UPDATE pattern_matches SET status = 'treasury_confirmed' WHERE address = ?",
                    (addr,),
                )
                conn.commit()

                # Add operators to deployers table
                for dm in results["deployer_matches"]:
                    dep = dm["deployer"]
                    existing = conn.execute(
                        "SELECT entity_type FROM deployers WHERE deployer_address = ?", (dep,)
                    ).fetchone()
                    if existing and existing[0] in (None, "unknown"):
                        conn.execute(
                            "UPDATE deployers SET entity_type = 'operator_candidate' WHERE deployer_address = ?",
                            (dep,),
                        )
                conn.commit()
            else:
                print(f"  {addr[:18]}... -> no deployer overlap")

    # PASS 3: Vanity check
    if matches:
        print()
        print("=" * 60)
        print("PASS 3: VANITY ADDRESS CHECK")
        print("=" * 60)

        for addr, evidence in matches:
            funded = list(set(
                evidence.get("funded_addresses", []) + evidence.get("usdc_recipients", [])
            ))
            vanity = check_vanity(funded)
            if vanity:
                print(f"\n  {addr[:18]}... -> VANITY GROUPS DETECTED")
                for prefix, addrs in vanity.items():
                    print(f"    0x{prefix}...: {len(addrs)} addresses")
                    for a in addrs[:5]:
                        print(f"      {a}")

                # Update evidence
                evidence["vanity_groups"] = {k: v for k, v in vanity.items()}
                conn.execute(
                    "UPDATE pattern_matches SET evidence = ? WHERE address = ?",
                    (json.dumps(evidence, default=str), addr),
                )
                conn.commit()
            else:
                print(f"  {addr[:18]}... -> no vanity groups")

    # SUMMARY
    print()
    print("=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    total = conn.execute("SELECT COUNT(*) FROM pattern_matches").fetchone()[0]
    confirmed = conn.execute("SELECT COUNT(*) FROM pattern_matches WHERE status = 'treasury_confirmed'").fetchone()[0]
    candidates = conn.execute("SELECT COUNT(*) FROM pattern_matches WHERE status = 'candidate'").fetchone()[0]
    print(f"Total matches:     {total}")
    print(f"  Confirmed:       {confirmed}")
    print(f"  Candidates:      {candidates}")

    # Show all matches
    rows = conn.execute(
        "SELECT address, confidence, status FROM pattern_matches ORDER BY confidence DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r[0]}: [{r[1]}] {r[2]}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(run_scan())
