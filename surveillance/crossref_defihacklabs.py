"""Cross-reference DeFiHackLabs known attackers/contracts against our live operation."""
import asyncio
import json
import os
import sys
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

HTTP_URL = os.environ["ARB_WSS_URL"].replace("wss://", "https://")
TREASURY = "0xf186cb00e49e18491db5783ff04fae3818102ff7"
CASHOUT = "0xc6962004f452be9203591991d15f6b388e09e8d0"
OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"


async def get_funder(addr):
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "toAddress": addr, "category": ["external", "internal"],
        "maxCount": "0x5", "order": "asc", "withMetadata": True,
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "alchemy_getAssetTransfers", "params": [params]
        }, timeout=30) as r:
            transfers = (await r.json()).get("result", {}).get("transfers", [])
            for t in transfers:
                val = float(t.get("value") or 0)
                if val > 0:
                    return t.get("from", "").lower(), val
    return None, 0


async def get_usdc_counterparties(addr):
    cps = set()
    for direction, key in [("from", "fromAddress"), ("to", "toAddress")]:
        params = {
            "fromBlock": "0x0", "toBlock": "latest",
            key: addr, "category": ["erc20"],
            "maxCount": "0x64", "order": "desc", "withMetadata": True,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(HTTP_URL, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "alchemy_getAssetTransfers", "params": [params]
            }, timeout=30) as r:
                transfers = (await r.json()).get("result", {}).get("transfers", [])
                for t in transfers:
                    cp = (t.get("to") if direction == "from" else t.get("from")) or ""
                    cps.add(cp.lower())
    return cps


async def main():
    conn = db.init_db()

    all_deployers = {r[0].lower() for r in conn.execute("SELECT deployer_address FROM deployers").fetchall()}
    key_addrs = {TREASURY.lower(): "TREASURY", CASHOUT.lower(): "CASHOUT", OPERATOR.lower(): "OPERATOR"}

    attackers = conn.execute(
        "SELECT deployer_address, deployment_pattern_notes FROM deployers WHERE entity_type = 'known_attacker'"
    ).fetchall()
    ground_truth = conn.execute(
        "SELECT deployer_address, deployment_pattern_notes FROM deployers WHERE entity_type = 'ground_truth'"
    ).fetchall()

    # Get operator USDC counterparties
    print("Loading operator USDC counterparties...")
    op_usdc = await get_usdc_counterparties(OPERATOR)
    print(f"Operator USDC counterparties: {len(op_usdc)}")

    our_contracts = {r[0].lower() for r in conn.execute("SELECT contract_address FROM contracts").fetchall()}

    print()
    print("=" * 60)
    print("CROSS-REFERENCE: DeFiHackLabs vs Live Operation")
    print("=" * 60)
    print(f"Known attackers:        {len(attackers)}")
    print(f"Ground truth contracts: {len(ground_truth)}")
    print(f"Our deployers:          {len(all_deployers)}")
    print(f"Our contracts:          {len(our_contracts)}")
    print()

    total_matches = 0

    # CHECK 1: Attacker funding sources
    print("--- CHECK 1: Attacker funding sources vs treasury/cashout/deployers ---")
    for row in attackers:
        addr = row[0]
        notes = row[1][:50] if row[1] else "?"
        funder, val = await get_funder(addr)
        await asyncio.sleep(0.2)

        if not funder:
            continue

        tags = []
        if funder in key_addrs:
            tags.append(key_addrs[funder])
        if funder in all_deployers:
            tags.append("IN_DEPLOYERS")
        if funder in op_usdc:
            tags.append("OPERATOR_USDC_CP")

        if tags:
            total_matches += 1
            print(f"  *** MATCH: {addr}")
            print(f"      funder: {funder}")
            print(f"      tags:   {tags}")
            print(f"      ETH:    {val:.6f}")
        else:
            print(f"  {addr[:18]}... funder={funder[:18]}... - no match")

    # CHECK 2: Ground truth contracts vs operator USDC flows
    print()
    print("--- CHECK 2: Ground truth contracts in operator USDC flows ---")
    gt_addrs = {r[0].lower() for r in ground_truth}
    overlap = gt_addrs & op_usdc
    if overlap:
        for addr in overlap:
            total_matches += 1
            print(f"  *** MATCH: {addr} is an operator USDC counterparty")
    else:
        print("  No overlap")

    # CHECK 3: Ground truth in our contracts table
    print()
    print("--- CHECK 3: Ground truth contracts in our contracts table ---")
    ct_overlap = gt_addrs & our_contracts
    if ct_overlap:
        for addr in ct_overlap:
            total_matches += 1
            row = conn.execute(
                "SELECT confidence_tier, deployer_address FROM contracts WHERE contract_address = ?", (addr,)
            ).fetchone()
            print(f"  *** MATCH: {addr} tier={row[0]} deployer={row[1]}")
    else:
        print("  No overlap")

    # CHECK 4: Known attackers that are also deployers in our DB
    print()
    print("--- CHECK 4: Known attackers that deployed contracts in our DB ---")
    attacker_addrs = {r[0].lower() for r in attackers}
    direct = attacker_addrs & (all_deployers - attacker_addrs)
    # Check if any attacker deployed contracts
    for addr in attacker_addrs:
        contracts = conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE deployer_address = ?", (addr,)
        ).fetchone()[0]
        if contracts > 0:
            total_matches += 1
            print(f"  *** MATCH: {addr} deployed {contracts} contracts")
    if not any(conn.execute("SELECT COUNT(*) FROM contracts WHERE deployer_address = ?", (a,)).fetchone()[0] > 0 for a in attacker_addrs):
        print("  No known attackers deployed contracts in our DB")

    # CHECK 5: Shared funders
    print()
    print("--- CHECK 5: Shared funders (same address funded both attacker + our deployer) ---")
    our_funders = {}
    rows = conn.execute(
        "SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            trail = json.loads(row[1])
            src = trail.get("funding_source", "")
            if src and len(src) == 42:
                our_funders.setdefault(src.lower(), []).append(row[0])
        except Exception:
            pass

    attacker_funder_map = {}
    for row in attackers:
        funder, _ = await get_funder(row[0])
        await asyncio.sleep(0.2)
        if funder:
            attacker_funder_map[row[0]] = funder

    for attacker_addr, attacker_funder in attacker_funder_map.items():
        if attacker_funder in our_funders:
            total_matches += 1
            deployers = our_funders[attacker_funder]
            print(f"  *** SHARED FUNDER: {attacker_funder}")
            print(f"      funded attacker: {attacker_addr}")
            print(f"      also funded: {deployers}")

    if not any(f in our_funders for f in attacker_funder_map.values()):
        print("  No shared funders")

    # VERDICT
    print()
    print("=" * 60)
    if total_matches > 0:
        print(f"CROSS-REFERENCES FOUND: {total_matches}")
        print("Historical exploits ARE connected to our live operation.")
    else:
        print("CROSS-REFERENCES FOUND: 0")
        print("No connection between DeFiHackLabs exploits and our operation.")
        print("The trap operation is independent of known exploit actors.")
    print("=" * 60)

    conn.close()


asyncio.run(main())
