"""Trace new laundry pipeline candidates 0xc6f780... and 0xcda53b..."""
import asyncio
import os
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

HTTP_URL = os.environ["ARB_WSS_URL"].replace("wss://", "https://")

TARGETS = [
    "0xc6f780497a95e246eb9449f5e4770916dcd6396a",
    "0xcda53b1f66614552f834ceef361a8d12a0b8dad8",
]

KNOWN = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": "TREASURY",
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "OPERATOR",
    "0xc6962004f452be9203591991d15f6b388e09e8d0": "CASHOUT",
    "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70": "ETH_WRAPPER",
    "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "LAUNDRY_WBTC",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "LP_POOL_1",
    "0x01989c93890aed05cbcda4e62eec1b2eb4c55b1b": "LP_POOL_2",
}


async def get_transfers(addr, direction, cats, max_r=200):
    all_t = []
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "category": cats, "maxCount": hex(min(max_r, 1000)),
        "order": "asc", "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = addr
    else:
        params["toAddress"] = addr

    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "alchemy_getAssetTransfers", "params": [params]
        }, timeout=30) as r:
            data = (await r.json()).get("result", {})
            all_t.extend(data.get("transfers", []))
            pk = data.get("pageKey")
        while pk and len(all_t) < max_r:
            params["pageKey"] = pk
            async with s.post(HTTP_URL, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "alchemy_getAssetTransfers", "params": [params]
            }, timeout=30) as r:
                data = (await r.json()).get("result", {})
                all_t.extend(data.get("transfers", []))
                pk = data.get("pageKey")
            await asyncio.sleep(0.15)
    return all_t[:max_r]


async def analyze(addr):
    print("=" * 60)
    print(f"TARGET: {addr}")
    print("=" * 60)

    out_eth = await get_transfers(addr, "from", ["external"])
    in_eth = await get_transfers(addr, "to", ["external", "internal"])
    out_erc = await get_transfers(addr, "from", ["erc20"])
    in_erc = await get_transfers(addr, "to", ["erc20"])

    all_t = (
        [dict(**t, _d="OUT", _c="ETH") for t in out_eth]
        + [dict(**t, _d="IN", _c="ETH") for t in in_eth]
        + [dict(**t, _d="OUT", _c="ERC20") for t in out_erc]
        + [dict(**t, _d="IN", _c="ERC20") for t in in_erc]
    )
    all_t.sort(key=lambda t: int(t.get("blockNum", "0x0"), 16))

    print(f"Total transfers: {len(all_t)}")
    if all_t:
        first = all_t[0].get("metadata", {}).get("blockTimestamp", "?")
        last = all_t[-1].get("metadata", {}).get("blockTimestamp", "?")
        print(f"First active: {first}")
        print(f"Last active:  {last}")
    print()

    # Financial summary
    eth_in = eth_out = weth_in = weth_out = usdc_in = usdc_out = 0.0
    for t in all_t:
        asset = (t.get("asset") or "").upper()
        val = float(t.get("value") or 0)
        if t["_c"] == "ETH":
            if t["_d"] == "IN": eth_in += val
            else: eth_out += val
        elif "WETH" in asset:
            if t["_d"] == "IN": weth_in += val
            else: weth_out += val
        elif "USDC" in asset:
            if t["_d"] == "IN": usdc_in += val
            else: usdc_out += val

    print(f"ETH:  in={eth_in:.6f} out={eth_out:.6f}")
    print(f"WETH: in={weth_in:.6f} out={weth_out:.6f}")
    print(f"USDC: in=${usdc_in:,.2f} out=${usdc_out:,.2f}")
    print()

    # First funder
    for t in all_t:
        if t["_d"] == "IN" and float(t.get("value") or 0) > 0:
            fr = (t.get("from") or "").lower()
            val = float(t.get("value") or 0)
            asset = t.get("asset") or "ETH"
            tag = f" [{KNOWN[fr]}]" if fr in KNOWN else ""
            print(f"FIRST FUNDER: {fr}{tag}")
            print(f"  {val:.6f} {asset}")
            break

    # All counterparties
    cps = defaultdict(lambda: {"in_eth": 0, "out_eth": 0, "in_erc": 0, "out_erc": 0, "txs": 0})
    for t in all_t:
        cp = (t.get("to") if t["_d"] == "OUT" else t.get("from")) or ""
        cp = cp.lower()
        if not cp or cp == addr.lower():
            continue
        val = float(t.get("value") or 0)
        cps[cp]["txs"] += 1
        if t["_d"] == "OUT":
            if t["_c"] == "ETH":
                cps[cp]["out_eth"] += val
            else:
                cps[cp]["out_erc"] += val
        else:
            if t["_c"] == "ETH":
                cps[cp]["in_eth"] += val
            else:
                cps[cp]["in_erc"] += val

    # Check known entity connections
    print()
    print("KNOWN ENTITY CONNECTIONS:")
    connected_to_org = False
    for entity, name in KNOWN.items():
        if entity.lower() in cps:
            d = cps[entity.lower()]
            connected_to_org = True
            print(f"  {name} ({entity[:18]}...):")
            print(f"    ETH: in={d['in_eth']:.6f} out={d['out_eth']:.6f}")
            print(f"    ERC: in={d['in_erc']:.6f} out={d['out_erc']:.6f}")
            print(f"    txs: {d['txs']}")

    if not connected_to_org:
        print("  No direct connections to known entities")

    # Top counterparties
    print()
    print("TOP COUNTERPARTIES:")
    by_volume = sorted(cps.items(),
                       key=lambda x: -(x[1]["in_eth"] + x[1]["out_eth"] + x[1]["in_erc"] + x[1]["out_erc"]))
    for cp_addr, d in by_volume[:10]:
        tag = f" [{KNOWN[cp_addr]}]" if cp_addr in KNOWN else ""
        total = d["in_eth"] + d["out_eth"] + d["in_erc"] + d["out_erc"]
        print(f"  {cp_addr}: {total:.4f} total ({d['txs']} txs){tag}")

    print()
    return connected_to_org


async def main():
    any_connected = False
    for addr in TARGETS:
        connected = await analyze(addr)
        if connected:
            any_connected = True

    print("=" * 60)
    if any_connected:
        print("VERDICT: Connected to organization. Upgrade to laundry_confirmed.")
    else:
        print("VERDICT: No org connection found. Remain laundry_candidate.")
    print("=" * 60)


asyncio.run(main())
