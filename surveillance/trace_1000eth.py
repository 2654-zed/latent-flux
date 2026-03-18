"""Investigate 0xe93d2a... — the 1,000 ETH recipient from treasury."""
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
TARGET = "0xe93d2a52f549b9726f2914ab4c2ff0f25c6e7f86"
TREASURY = "0xf186cb00e49e18491db5783ff04fae3818102ff7"
CASHOUT = "0xc6962004f452be9203591991d15f6b388e09e8d0"
OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"
LP_POOLS = {"0x51c72848c68a965f66fa7a88855f9f7784502a7f", "0x01989c93890aed05cbcda4e62eec1b2eb4c55b1b"}


async def get_transfers(addr, direction, cats, pk=None):
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "category": cats, "maxCount": "0x3e8",
        "order": "asc", "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = addr
    else:
        params["toAddress"] = addr
    if pk:
        params["pageKey"] = pk
    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "alchemy_getAssetTransfers", "params": [params]
        }, timeout=60) as r:
            d = (await r.json()).get("result", {})
            return d.get("transfers", []), d.get("pageKey")


async def count_all(addr, direction, cats):
    """Count total transfers without storing all of them."""
    total = 0
    first_ts = last_ts = None
    t, pk = await get_transfers(addr, direction, cats)
    total += len(t)
    if t:
        first_ts = t[0].get("metadata", {}).get("blockTimestamp")
        last_ts = t[-1].get("metadata", {}).get("blockTimestamp")
    while pk:
        t, pk = await get_transfers(addr, direction, cats, pk)
        total += len(t)
        if t:
            last_ts = t[-1].get("metadata", {}).get("blockTimestamp")
        await asyncio.sleep(0.1)
        # Safety: if this is a massive protocol, stop early
        if total > 5000:
            return total, first_ts, last_ts, True  # truncated
    return total, first_ts, last_ts, False


async def main():
    print(f"TARGET: {TARGET}")
    print(f"(Received 1,000 ETH from treasury {TREASURY[:18]}...)")
    print()

    # Step 1: Count transfers to determine protocol vs operator
    print("=== TRANSFER COUNTS ===")
    for direction, cats, label in [
        ("from", ["external"], "ETH out"),
        ("to", ["external", "internal"], "ETH in"),
        ("from", ["erc20"], "ERC20 out"),
        ("to", ["erc20"], "ERC20 in"),
    ]:
        count, first, last, trunc = await count_all(TARGET, direction, cats)
        trunc_note = " (TRUNCATED >5000)" if trunc else ""
        print(f"  {label:12s}: {count:>6}{trunc_note}")
        if first:
            print(f"               first: {first}")
        if last:
            print(f"               last:  {last}")
    print()

    # Step 2: Get first page of each direction for analysis
    print("=== SAMPLE TRANSFERS (first page each) ===")

    # ETH outbound — check for contract deployments (0 value = deploy)
    out_eth, _ = await get_transfers(TARGET, "from", ["external"])
    deploys = 0
    total_eth_out = 0.0
    for t in out_eth:
        val = float(t.get("value") or 0)
        total_eth_out += val
        if val == 0:
            deploys += 1

    print(f"ETH outbound (first page): {len(out_eth)} transfers")
    print(f"  Total ETH out (page 1): {total_eth_out:.6f}")
    print(f"  Possible deploys (0 value): {deploys}")
    print()

    # ETH inbound
    in_eth, _ = await get_transfers(TARGET, "to", ["external", "internal"])
    total_eth_in = 0.0
    sources = {}
    for t in in_eth:
        val = float(t.get("value") or 0)
        total_eth_in += val
        fr = (t.get("from") or "").lower()
        if fr not in sources:
            sources[fr] = {"total": 0, "count": 0, "first": t.get("metadata", {}).get("blockTimestamp", "")}
        sources[fr]["total"] += val
        sources[fr]["count"] += 1

    print(f"ETH inbound (first page): {len(in_eth)} transfers")
    print(f"  Total ETH in (page 1): {total_eth_in:.6f}")
    print(f"  Unique sources: {len(sources)}")
    print()

    print("  Top ETH sources:")
    for addr, info in sorted(sources.items(), key=lambda x: -x[1]["total"])[:10]:
        tag = ""
        if addr == TREASURY: tag = " [TREASURY]"
        elif addr == CASHOUT.lower(): tag = " [CASHOUT]"
        elif addr == OPERATOR: tag = " [OPERATOR]"
        elif addr in {p.lower() for p in LP_POOLS}: tag = " [LP_POOL]"
        print(f"    {addr}: {info['total']:.4f} ETH ({info['count']} txs) first={info['first'][:10]}{tag}")
    print()

    # ERC20 outbound
    out_erc, _ = await get_transfers(TARGET, "from", ["erc20"])
    erc_out_by_asset = {}
    erc_out_recipients = {}
    for t in out_erc:
        asset = (t.get("asset") or "?").upper()
        val = float(t.get("value") or 0)
        erc_out_by_asset[asset] = erc_out_by_asset.get(asset, 0) + val
        to = (t.get("to") or "").lower()
        if to not in erc_out_recipients:
            erc_out_recipients[to] = 0
        erc_out_recipients[to] += val

    print(f"ERC20 outbound (first page): {len(out_erc)} transfers")
    for asset, total in sorted(erc_out_by_asset.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {total:,.2f}")
    print()

    # Check LP pool interactions
    print("=== LP POOL CONNECTIONS ===")
    for pool in LP_POOLS:
        found_in = pool.lower() in sources
        found_out = pool.lower() in erc_out_recipients
        if found_in or found_out:
            print(f"  {pool[:18]}...: {'IN' if found_in else ''} {'OUT' if found_out else ''}")
        else:
            print(f"  {pool[:18]}...: NO CONNECTION")
    print()

    # Check connections to known entities
    print("=== KNOWN ENTITY CONNECTIONS ===")
    all_counterparties = set(sources.keys()) | set(erc_out_recipients.keys())
    for entity, name in [
        (TREASURY, "TREASURY"),
        (CASHOUT.lower(), "CASHOUT"),
        (OPERATOR, "OPERATOR"),
    ]:
        if entity in all_counterparties:
            in_val = sources.get(entity, {}).get("total", 0)
            out_val = erc_out_recipients.get(entity, 0)
            print(f"  {name}: ETH in={in_val:.4f}, ERC20 out={out_val:,.2f}")
        else:
            print(f"  {name}: NO CONNECTION")
    print()

    # Check the address similarity — 0xe93d2a vs 0xe93d64 (operator)
    print("=== ADDRESS SIMILARITY CHECK ===")
    print(f"  Target:   {TARGET}")
    print(f"  Operator: {OPERATOR}")
    shared = sum(1 for a, b in zip(TARGET, OPERATOR) if a == b)
    print(f"  Matching chars: {shared}/42")
    if TARGET[:6] == OPERATOR[:6]:
        print(f"  *** FIRST 6 CHARS MATCH: {TARGET[:6]} ***")
        print(f"  This could be a vanity address from the same generator")


asyncio.run(main())
