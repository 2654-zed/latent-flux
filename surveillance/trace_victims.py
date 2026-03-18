"""Retroactive victim analysis for pre-surveillance trap contracts."""
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

HTTP_URL = os.environ["ARB_WSS_URL"].replace("wss://", "https://")
OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"

USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_E = "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"

TARGETS = [
    ("0x226d4254c4497b30605ec5784ba290de5bbbc180", "$4,095 returned Mar 7"),
    ("0x195fdd00224702fc43f36ece4dfd1b28af4f4f51", "$1,886 returned Mar 2"),
    ("0xa84a17a9d2c62999ca015ef5e4871a20e1147b32", "$1,878 returned Mar 2"),
    ("0xd3091c2ac3d117df4ba96a7f8c44d01ce4554998", "$1,838 returned Mar 2"),
    ("0x22fe25bee06d20d08d350e7a944e9e5bb0ced998", "$1,185 returned Mar 1"),
    ("0x511cebfafd4cbd364d643b1b7edfa5d6dd831349", "$1,127 returned Mar 2"),
]


async def get_transfers(addr, direction, page_key=None):
    params = {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "category": ["erc20"],
        "contractAddresses": [USDC, USDC_E],
        "maxCount": "0x64",
        "order": "asc",
        "withMetadata": True,
    }
    if direction == "to":
        params["toAddress"] = addr
    else:
        params["fromAddress"] = addr
    if page_key:
        params["pageKey"] = page_key

    payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [params]}
    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json=payload, timeout=30) as resp:
            data = await resp.json()
            result = data.get("result", {})
            return result.get("transfers", []), result.get("pageKey")


async def analyze_contract(addr, note):
    print(f"{'='*60}")
    print(f"CONTRACT: {addr}")
    print(f"Context:  {note}")
    print()

    # Get all USDC flowing INTO the contract
    inbound = []
    transfers, pk = await get_transfers(addr, "to")
    inbound.extend(transfers)
    while pk:
        transfers, pk = await get_transfers(addr, "to", pk)
        inbound.extend(transfers)
        await asyncio.sleep(0.15)

    # Get all USDC flowing OUT of the contract
    outbound = []
    transfers, pk = await get_transfers(addr, "from")
    outbound.extend(transfers)
    while pk:
        transfers, pk = await get_transfers(addr, "from", pk)
        outbound.extend(transfers)
        await asyncio.sleep(0.15)

    # Build per-address ledger: who put in, who got out
    ledger = {}  # addr -> {deposited, withdrawn}
    for t in inbound:
        fr = t.get("from", "").lower()
        val = float(t.get("value", 0))
        ts = t.get("metadata", {}).get("blockTimestamp", "")
        if fr not in ledger:
            ledger[fr] = {"deposited": 0, "withdrawn": 0, "dep_txs": [], "wd_txs": []}
        ledger[fr]["deposited"] += val
        ledger[fr]["dep_txs"].append({"value": val, "ts": ts, "tx": t.get("hash", "")})

    for t in outbound:
        to = t.get("to", "").lower()
        val = float(t.get("value", 0))
        ts = t.get("metadata", {}).get("blockTimestamp", "")
        if to not in ledger:
            ledger[to] = {"deposited": 0, "withdrawn": 0, "dep_txs": [], "wd_txs": []}
        ledger[to]["withdrawn"] += val
        ledger[to]["wd_txs"].append({"value": val, "ts": ts, "tx": t.get("hash", "")})

    total_in = sum(float(t.get("value", 0)) for t in inbound)
    total_out = sum(float(t.get("value", 0)) for t in outbound)

    print(f"Total USDC in:  ${total_in:,.2f} ({len(inbound)} transfers)")
    print(f"Total USDC out: ${total_out:,.2f} ({len(outbound)} transfers)")
    print(f"Unique addresses: {len(ledger)}")
    print()

    # Identify the operator
    op_data = ledger.get(OPERATOR, {"deposited": 0, "withdrawn": 0})
    print(f"OPERATOR ({OPERATOR[:18]}...):")
    print(f"  Deposited: ${op_data['deposited']:,.2f}")
    print(f"  Withdrawn: ${op_data['withdrawn']:,.2f}")
    print(f"  Net:       ${op_data['withdrawn'] - op_data['deposited']:,.2f}")
    print()

    # Find victims: deposited but withdrew less than deposited (or nothing)
    victims = []
    for address, data in ledger.items():
        if address == OPERATOR:
            continue
        if address in (USDC.lower(), USDC_E.lower(), "0x0000000000000000000000000000000000000000"):
            continue
        net = data["withdrawn"] - data["deposited"]
        if data["deposited"] > 0 and net < -0.01:  # lost more than dust
            loss = abs(net)
            victims.append({
                "address": address,
                "deposited": data["deposited"],
                "withdrawn": data["withdrawn"],
                "loss": loss,
                "dep_txs": data["dep_txs"],
            })

    # Find addresses that only received (beneficiaries besides operator)
    beneficiaries = []
    for address, data in ledger.items():
        if address == OPERATOR:
            continue
        if address in (USDC.lower(), USDC_E.lower(), "0x0000000000000000000000000000000000000000"):
            continue
        if data["deposited"] == 0 and data["withdrawn"] > 0.01:
            beneficiaries.append({
                "address": address,
                "withdrawn": data["withdrawn"],
            })

    if victims:
        victims.sort(key=lambda x: -x["loss"])
        print(f"VICTIMS IDENTIFIED: {len(victims)}")
        for v in victims:
            print(f"  {v['address']}")
            print(f"    deposited: ${v['deposited']:,.2f}")
            print(f"    withdrawn: ${v['withdrawn']:,.2f}")
            print(f"    LOSS:      ${v['loss']:,.2f}")
            if v["dep_txs"]:
                first = v["dep_txs"][0]
                print(f"    first_deposit: {first['ts'][:19]} tx={first['tx'][:18]}...")
    else:
        print("NO VICTIMS FOUND (all depositors got their funds back)")

    if beneficiaries:
        print()
        print(f"  OTHER BENEFICIARIES (received without depositing): {len(beneficiaries)}")
        for b in beneficiaries:
            print(f"    {b['address']}: ${b['withdrawn']:,.2f}")

    print()
    return victims


async def main():
    all_victims = []
    for addr, note in TARGETS:
        victims = await analyze_contract(addr, note)
        all_victims.extend(victims)
        await asyncio.sleep(0.3)

    print("=" * 60)
    print(f"TOTAL VICTIMS ACROSS ALL CONTRACTS: {len(all_victims)}")
    total_loss = sum(v["loss"] for v in all_victims)
    print(f"TOTAL ESTIMATED LOSSES: ${total_loss:,.2f}")
    print()

    # Deduplicate by address
    by_addr = {}
    for v in all_victims:
        a = v["address"]
        if a not in by_addr:
            by_addr[a] = {"total_loss": 0, "contracts": 0}
        by_addr[a]["total_loss"] += v["loss"]
        by_addr[a]["contracts"] += 1

    if by_addr:
        print("UNIQUE VICTIMS:")
        for addr, info in sorted(by_addr.items(), key=lambda x: -x[1]["total_loss"]):
            print(f"  {addr}: ${info['total_loss']:,.2f} across {info['contracts']} contracts")


asyncio.run(main())
