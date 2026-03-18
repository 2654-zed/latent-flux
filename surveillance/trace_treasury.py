"""Trace the treasury address 0xf186cb... and cashout wallet 0xc69620..."""
import asyncio
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

TREASURY = "0xf186cb00e49e18491db5783ff04fae3818102ff7"
CASHOUT = "0xc6962004f452be9203591991d15f6b388e09e8d0"
OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"

LP_POOLS = {
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "LP_POOL_1",
    "0x01989c93890aed05cbcda4e62eec1b2eb4c55b1b": "LP_POOL_2",
}


async def get_all_transfers(addr, direction, cats, label=""):
    all_t = []
    params = {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "category": cats,
        "maxCount": "0x3e8",
        "order": "asc",
        "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = addr
    else:
        params["toAddress"] = addr

    payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [params]}
    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json=payload, timeout=60) as resp:
            data = await resp.json()
            result = data.get("result", {})
            all_t.extend(result.get("transfers", []))
            pk = result.get("pageKey")

        while pk:
            params["pageKey"] = pk
            payload["params"] = [params]
            async with s.post(HTTP_URL, json=payload, timeout=60) as resp:
                data = await resp.json()
                result = data.get("result", {})
                all_t.extend(result.get("transfers", []))
                pk = result.get("pageKey")
            await asyncio.sleep(0.15)

    return all_t


async def analyze_address(addr, name):
    print(f"{'='*60}")
    print(f"{name}: {addr}")
    print()

    out_eth = await get_all_transfers(addr, "from", ["external"])
    in_eth = await get_all_transfers(addr, "to", ["external", "internal"])
    out_erc20 = await get_all_transfers(addr, "from", ["erc20"])
    in_erc20 = await get_all_transfers(addr, "to", ["erc20"])

    all_t = (
        [{**t, "_d": "OUT", "_c": "ETH"} for t in out_eth]
        + [{**t, "_d": "IN", "_c": "ETH"} for t in in_eth]
        + [{**t, "_d": "OUT", "_c": "ERC20"} for t in out_erc20]
        + [{**t, "_d": "IN", "_c": "ERC20"} for t in in_erc20]
    )
    all_t.sort(key=lambda t: int(t.get("blockNum", "0x0"), 16))

    # Financial summary
    eth_in = sum(float(t.get("value", 0)) for t in all_t if t["_d"] == "IN" and t["_c"] == "ETH")
    eth_out = sum(float(t.get("value", 0)) for t in all_t if t["_d"] == "OUT" and t["_c"] == "ETH")

    usdc_in = usdc_out = 0.0
    weth_in = weth_out = 0.0
    for t in all_t:
        asset = (t.get("asset") or "").upper()
        val = float(t.get("value", 0))
        if "USDC" in asset:
            if t["_d"] == "IN":
                usdc_in += val
            else:
                usdc_out += val
        if "WETH" in asset:
            if t["_d"] == "IN":
                weth_in += val
            else:
                weth_out += val

    print(f"Total transfers: {len(all_t)}")
    if all_t:
        first = all_t[0].get("metadata", {}).get("blockTimestamp", "?")
        last = all_t[-1].get("metadata", {}).get("blockTimestamp", "?")
        print(f"First activity: {first}")
        print(f"Last activity:  {last}")
    print()
    print(f"ETH in:   {eth_in:.6f} | ETH out:  {eth_out:.6f}")
    print(f"USDC in:  ${usdc_in:,.2f} | USDC out: ${usdc_out:,.2f}")
    print(f"WETH in:  {weth_in:.6f} | WETH out: {weth_out:.6f}")
    print()

    # Counterparty analysis
    counterparties = {}
    for t in all_t:
        if t["_d"] == "OUT":
            cp = t.get("to", "").lower()
        else:
            cp = t.get("from", "").lower()
        if not cp or cp == addr.lower():
            continue
        asset = (t.get("asset") or "ETH").upper()
        val = float(t.get("value", 0))
        key = cp
        if key not in counterparties:
            counterparties[key] = {"in": 0, "out": 0, "txs": 0, "assets": set()}
        if t["_d"] == "OUT":
            counterparties[key]["out"] += val if "USDC" in asset else 0
        else:
            counterparties[key]["in"] += val if "USDC" in asset else 0
        counterparties[key]["txs"] += 1
        counterparties[key]["assets"].add(asset)

    # Check key relationships
    print("KEY RELATIONSHIPS:")

    # Did this address interact with the operator?
    if OPERATOR in counterparties:
        op = counterparties[OPERATOR]
        print(f"  -> OPERATOR {OPERATOR[:18]}...")
        print(f"     USDC sent: ${op['out']:,.2f} | USDC received: ${op['in']:,.2f} | txs: {op['txs']}")

    # Did it interact with the cashout wallet?
    if CASHOUT.lower() in counterparties:
        co = counterparties[CASHOUT.lower()]
        print(f"  -> CASHOUT {CASHOUT[:18]}...")
        print(f"     USDC sent: ${co['out']:,.2f} | USDC received: ${co['in']:,.2f} | txs: {co['txs']}")

    # Did it interact with LP pools?
    for pool, pool_name in LP_POOLS.items():
        if pool.lower() in counterparties:
            p = counterparties[pool.lower()]
            print(f"  -> {pool_name} {pool[:18]}...")
            print(f"     USDC sent: ${p['out']:,.2f} | USDC received: ${p['in']:,.2f} | txs: {p['txs']}")

    # How many OTHER addresses funded?
    funded_addrs = []
    for cp, data in counterparties.items():
        if data["out"] > 0:
            funded_addrs.append((cp, data["out"], data["txs"]))
    funded_addrs.sort(key=lambda x: -x[1])

    print()
    print(f"ADDRESSES FUNDED (USDC sent to): {len(funded_addrs)}")
    for cp_addr, amount, txs in funded_addrs[:15]:
        tag = ""
        if cp_addr == OPERATOR:
            tag = " [OPERATOR]"
        elif cp_addr == CASHOUT.lower():
            tag = " [CASHOUT]"
        elif cp_addr in [p.lower() for p in LP_POOLS]:
            tag = " [LP_POOL]"
        print(f"  {cp_addr}: ${amount:,.2f} ({txs} txs){tag}")

    print()
    return counterparties


async def main():
    print("TREASURY TRACE")
    print("Tracing 0xf186cb... (funded the operator with $4,205)")
    print()

    treasury_cps = await analyze_address(TREASURY, "TREASURY")

    print()
    print("CASHOUT FULL HISTORY")
    print("Tracing 0xc69620... (full LP + cashout history)")
    print()

    cashout_cps = await analyze_address(CASHOUT, "CASHOUT_WALLET")

    # Cross-reference check
    print("=" * 60)
    print("CROSS-REFERENCE ANALYSIS")
    print()

    # Did treasury and cashout ever interact directly?
    if CASHOUT.lower() in treasury_cps:
        print("DIRECT LINK: Treasury <-> Cashout wallet have transacted directly")
        d = treasury_cps[CASHOUT.lower()]
        print(f"  USDC flow: sent ${d['out']:,.2f} | received ${d['in']:,.2f}")
    else:
        print("No direct transactions between Treasury and Cashout wallet")

    # Shared LP pool usage
    shared_pools = []
    for pool in LP_POOLS:
        in_treasury = pool.lower() in treasury_cps
        in_cashout = pool.lower() in cashout_cps
        if in_treasury and in_cashout:
            shared_pools.append(pool)

    if shared_pools:
        print(f"SHARED LP POOLS: {len(shared_pools)}")
        for p in shared_pools:
            print(f"  {p} used by BOTH treasury and cashout")
    else:
        print("No shared LP pool usage between treasury and cashout")

    # How many addresses did treasury fund?
    treasury_funded = [
        cp for cp, data in treasury_cps.items()
        if data["out"] > 0 and cp != OPERATOR
    ]
    print(f"Treasury funded {len(treasury_funded) + 1} addresses total (including operator)")


asyncio.run(main())
