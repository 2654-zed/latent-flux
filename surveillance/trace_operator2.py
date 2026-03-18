"""Operator analysis for 0x90b5f4... and 0x511ceb..."""
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
USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_E = "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"
KNOWN_OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"
TREASURY = "0xf186cb00e49e18491db5783ff04fae3818102ff7"
CASHOUT = "0xc6962004f452be9203591991d15f6b388e09e8d0"


async def all_transfers(addr, direction, cats):
    result = []
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "category": cats, "maxCount": "0x3e8",
        "order": "asc", "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = addr
    else:
        params["toAddress"] = addr

    async with aiohttp.ClientSession() as s:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [params]}
        async with s.post(HTTP_URL, json=payload, timeout=60) as r:
            d = (await r.json()).get("result", {})
            result.extend(d.get("transfers", []))
            pk = d.get("pageKey")
        while pk:
            params["pageKey"] = pk
            payload["params"] = [params]
            async with s.post(HTTP_URL, json=payload, timeout=60) as r:
                d = (await r.json()).get("result", {})
                result.extend(d.get("transfers", []))
                pk = d.get("pageKey")
            await asyncio.sleep(0.1)
    return result


async def analyze_operator(addr, label):
    print("=" * 60)
    print(f"{label}: {addr}")
    print("=" * 60)

    out_eth = await all_transfers(addr, "from", ["external"])
    in_eth = await all_transfers(addr, "to", ["external", "internal"])
    out_erc = await all_transfers(addr, "from", ["erc20"])
    in_erc = await all_transfers(addr, "to", ["erc20"])

    all_t = (
        [dict(**t, _d="OUT", _c="ETH") for t in out_eth]
        + [dict(**t, _d="IN", _c="ETH") for t in in_eth]
        + [dict(**t, _d="OUT", _c="ERC20") for t in out_erc]
        + [dict(**t, _d="IN", _c="ERC20") for t in in_erc]
    )
    all_t.sort(key=lambda t: int(t.get("blockNum", "0x0"), 16))

    print(f"Total transfers: {len(all_t)}")
    if all_t:
        print(f"First: {all_t[0].get('metadata',{}).get('blockTimestamp','?')}")
        print(f"Last:  {all_t[-1].get('metadata',{}).get('blockTimestamp','?')}")
    print()

    usdc_in = usdc_out = eth_in = eth_out = 0.0
    for t in all_t:
        asset = (t.get("asset") or "").upper()
        val = float(t.get("value", 0))
        if "USDC" in asset:
            if t["_d"] == "IN": usdc_in += val
            else: usdc_out += val
        elif t["_c"] == "ETH":
            if t["_d"] == "IN": eth_in += val
            else: eth_out += val

    print(f"USDC in:  ${usdc_in:,.2f}  |  USDC out: ${usdc_out:,.2f}")
    print(f"USDC net: ${usdc_in - usdc_out:,.2f}")
    print(f"ETH in:   {eth_in:.6f}  |  ETH out:  {eth_out:.6f}")
    nonce = len(out_eth)
    print(f"Outbound ETH txs (approx nonce): {nonce}")
    print()

    # Contract deployments (ETH out with 0 value = likely deploy)
    deploys = [t for t in all_t if t["_d"] == "OUT" and t["_c"] == "ETH" and float(t.get("value", 0)) == 0]
    print(f"Possible contract deployments: {len(deploys)}")
    print()

    # USDC recipients
    usdc_recipients = {}
    for t in all_t:
        if t["_d"] == "OUT" and "USDC" in (t.get("asset") or "").upper():
            to = t.get("to", "").lower()
            val = float(t.get("value", 0))
            ts = t.get("metadata", {}).get("blockTimestamp", "")
            if to not in usdc_recipients:
                usdc_recipients[to] = {"total": 0, "count": 0, "first": ts, "last": ts}
            usdc_recipients[to]["total"] += val
            usdc_recipients[to]["count"] += 1
            if ts < usdc_recipients[to]["first"]: usdc_recipients[to]["first"] = ts
            if ts > usdc_recipients[to]["last"]: usdc_recipients[to]["last"] = ts

    print(f"USDC RECIPIENTS ({len(usdc_recipients)} addresses):")
    for a, info in sorted(usdc_recipients.items(), key=lambda x: -x[1]["total"]):
        tag = ""
        if a == KNOWN_OPERATOR: tag = " [KNOWN_OPERATOR]"
        elif a == TREASURY: tag = " [TREASURY]"
        elif a == CASHOUT.lower(): tag = " [CASHOUT]"
        print(f"  {a}")
        print(f"    ${info['total']:,.2f} ({info['count']} txs) {info['first'][:10]} to {info['last'][:10]}{tag}")
    print()

    # USDC sources
    usdc_sources = {}
    for t in all_t:
        if t["_d"] == "IN" and "USDC" in (t.get("asset") or "").upper():
            fr = t.get("from", "").lower()
            val = float(t.get("value", 0))
            ts = t.get("metadata", {}).get("blockTimestamp", "")
            if fr not in usdc_sources:
                usdc_sources[fr] = {"total": 0, "count": 0, "first": ts}
            usdc_sources[fr]["total"] += val
            usdc_sources[fr]["count"] += 1

    print(f"USDC SOURCES ({len(usdc_sources)} addresses):")
    for a, info in sorted(usdc_sources.items(), key=lambda x: -x[1]["total"])[:15]:
        tag = ""
        if a == TREASURY: tag = " [TREASURY]"
        elif a == KNOWN_OPERATOR: tag = " [KNOWN_OPERATOR]"
        elif a == CASHOUT.lower(): tag = " [CASHOUT]"
        print(f"  {a}: ${info['total']:,.2f} ({info['count']} txs){tag}")
    print()

    # Connections to known entities
    known = {
        KNOWN_OPERATOR: "KNOWN_OPERATOR",
        TREASURY: "TREASURY",
        CASHOUT.lower(): "CASHOUT",
    }
    print("CONNECTIONS TO KNOWN ENTITIES:")
    for entity_addr, entity_name in known.items():
        sent = usdc_recipients.get(entity_addr, {}).get("total", 0)
        recv = usdc_sources.get(entity_addr, {}).get("total", 0)
        if sent > 0 or recv > 0:
            print(f"  {entity_name} ({entity_addr[:18]}...)")
            print(f"    USDC sent to:   ${sent:,.2f}")
            print(f"    USDC recv from: ${recv:,.2f}")
    print()

    # Daily timeline
    daily = {}
    for t in all_t:
        ts = t.get("metadata", {}).get("blockTimestamp", "")[:10]
        if not ts: continue
        if ts not in daily:
            daily[ts] = {"txs": 0, "usdc_out": 0, "usdc_in": 0}
        daily[ts]["txs"] += 1
        val = float(t.get("value", 0))
        asset = (t.get("asset") or "").upper()
        if t["_d"] == "OUT" and "USDC" in asset: daily[ts]["usdc_out"] += val
        elif t["_d"] == "IN" and "USDC" in asset: daily[ts]["usdc_in"] += val

    print("DAILY TIMELINE:")
    for date in sorted(daily.keys()):
        d = daily[date]
        bar = "#" * min(d["txs"], 40)
        notes = ""
        if d["usdc_out"] > 0: notes += f" out=${d['usdc_out']:,.0f}"
        if d["usdc_in"] > 0: notes += f" in=${d['usdc_in']:,.0f}"
        print(f"  {date}  {bar:40s} {d['txs']:>4} txs{notes}")
    print()


async def main():
    await analyze_operator(
        "0x90b5f4d2aeca515d368bb73e4b3cd4155dcb42d3",
        "SUSPECTED OPERATOR 2"
    )
    await analyze_operator(
        "0x511cebfafd4cbd364d643b1b7edfa5d6dd831349",
        "SUSPECTED OPERATOR 3"
    )

asyncio.run(main())
