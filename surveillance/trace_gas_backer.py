"""Trace 0x391e7c... — primary backer of the gas station."""
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
ARBISCAN_URL = "https://api.etherscan.io/v2/api"
KEY = os.environ["ARBISCAN_API_KEY"]

TARGET = "0x391e7c679d29bd940d63be94ad22a25d25b5a604"
TREASURY = "0xf186cb00e49e18491db5783ff04fae3818102ff7"
GAS_STATION = "0x8c826f795466e39acbff1bb4eeeb759609377ba1"
OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"
CASHOUT = "0xc6962004f452be9203591991d15f6b388e09e8d0"

KNOWN = {
    TREASURY.lower(): "TREASURY",
    OPERATOR.lower(): "OPERATOR",
    CASHOUT.lower(): "CASHOUT",
    GAS_STATION.lower(): "GAS_STATION",
}


async def arbiscan_txs(addr, session, pages=10):
    all_txs = []
    for page in range(1, pages + 1):
        params = {
            "chainid": 42161, "module": "account", "action": "txlist",
            "address": addr, "startblock": 0, "endblock": 99999999,
            "page": page, "offset": 1000, "sort": "desc", "apikey": KEY,
        }
        async with session.get(ARBISCAN_URL, params=params, timeout=30) as r:
            d = await r.json()
            txs = d.get("result", []) if d.get("status") == "1" else []
            all_txs.extend(txs)
            if len(txs) < 1000:
                break
            await asyncio.sleep(0.25)
    return all_txs


async def alchemy_transfers(addr, direction, cats, max_r=50):
    params = {
        "fromBlock": "0x0", "toBlock": "latest",
        "category": cats, "maxCount": hex(max_r),
        "order": "desc", "withMetadata": True,
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
            return (await r.json()).get("result", {}).get("transfers", [])


async def main():
    print(f"TARGET: {TARGET}")
    print(f"(Primary backer of gas station {GAS_STATION[:18]}... with 336 ETH)")
    print()

    # Arbiscan history
    async with aiohttp.ClientSession() as s:
        txs = await arbiscan_txs(TARGET, s, pages=10)

    print(f"Total txs (up to 10K): {len(txs)}")
    if txs:
        latest = datetime.fromtimestamp(int(txs[0]["timeStamp"]), tz=timezone.utc)
        earliest = datetime.fromtimestamp(int(txs[-1]["timeStamp"]), tz=timezone.utc)
        print(f"Range: {earliest.date()} to {latest.date()}")
    print()

    target_lower = TARGET.lower()
    eth_recipients = defaultdict(lambda: {"total": 0, "count": 0})
    eth_sources = defaultdict(lambda: {"total": 0, "count": 0})

    for tx in txs:
        fr = tx.get("from", "").lower()
        to = tx.get("to", "").lower()
        val = int(tx.get("value", "0")) / 1e18
        if fr == target_lower and to and val > 0:
            eth_recipients[to]["total"] += val
            eth_recipients[to]["count"] += 1
        elif to == target_lower and val > 0:
            eth_sources[fr]["total"] += val
            eth_sources[fr]["count"] += 1

    total_out = sum(d["total"] for d in eth_recipients.values())
    total_in = sum(d["total"] for d in eth_sources.values())

    print(f"ETH out: {total_out:.4f} to {len(eth_recipients)} addresses")
    print(f"ETH in:  {total_in:.4f} from {len(eth_sources)} addresses")
    print()

    # Connection check
    print("=== CONNECTION TO KNOWN ENTITIES ===")
    for addr, name in KNOWN.items():
        sent = eth_recipients.get(addr, {}).get("total", 0)
        sent_c = eth_recipients.get(addr, {}).get("count", 0)
        recv = eth_sources.get(addr, {}).get("total", 0)
        recv_c = eth_sources.get(addr, {}).get("count", 0)
        if sent > 0 or recv > 0:
            print(f"  {name}:")
            print(f"    ETH sent to:   {sent:.6f} ({sent_c} txs)")
            print(f"    ETH recv from: {recv:.6f} ({recv_c} txs)")
        else:
            print(f"  {name}: NO CONNECTION")
    print()

    # Top recipients
    print("TOP ETH RECIPIENTS:")
    for addr, info in sorted(eth_recipients.items(), key=lambda x: -x[1]["total"])[:15]:
        tag = f" [{KNOWN[addr]}]" if addr in KNOWN else ""
        print(f"  {addr}: {info['total']:.4f} ETH ({info['count']} txs){tag}")
    print()

    # Sources
    print("ETH SOURCES (who funded this address):")
    for addr, info in sorted(eth_sources.items(), key=lambda x: -x[1]["total"])[:10]:
        tag = f" [{KNOWN[addr]}]" if addr in KNOWN else ""
        print(f"  {addr}: {info['total']:.4f} ETH ({info['count']} txs){tag}")
    print()

    # USDC check via Alchemy
    print("=== USDC CONNECTIONS ===")
    usdc_out = await alchemy_transfers(TARGET, "from", ["erc20"], max_r=50)
    usdc_in = await alchemy_transfers(TARGET, "to", ["erc20"], max_r=50)

    usdc_out_total = sum(float(t.get("value") or 0) for t in usdc_out if "USDC" in (t.get("asset") or "").upper())
    usdc_in_total = sum(float(t.get("value") or 0) for t in usdc_in if "USDC" in (t.get("asset") or "").upper())
    print(f"USDC out: ${usdc_out_total:,.2f}")
    print(f"USDC in:  ${usdc_in_total:,.2f}")

    # Check for any USDC connection to known entities
    for t in usdc_out + usdc_in:
        counterparty = (t.get("to") or t.get("from") or "").lower()
        if counterparty in KNOWN:
            val = float(t.get("value") or 0)
            asset = t.get("asset", "?")
            direction = "to" if t.get("to", "").lower() in KNOWN else "from"
            print(f"  USDC {direction} {KNOWN[counterparty]}: {val} {asset}")
    print()

    # Daily activity
    daily = defaultdict(lambda: {"txs": 0, "eth_out": 0})
    for tx in txs:
        ts = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        daily[day]["txs"] += 1
        if tx.get("from", "").lower() == target_lower:
            daily[day]["eth_out"] += int(tx.get("value", "0")) / 1e18

    print("DAILY ACTIVITY (last 7 days):")
    for day in sorted(daily.keys())[-7:]:
        d = daily[day]
        bar = "#" * min(d["txs"] // 10, 40)
        print(f"  {day}  {bar:40s} {d['txs']:>5} txs  {d['eth_out']:.2f} ETH out")

    # VERDICT
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    treasury_connected = TREASURY.lower() in eth_recipients or TREASURY.lower() in eth_sources
    if treasury_connected:
        print("CONNECTION TO TREASURY: YES")
        print("This suggests shared organizational infrastructure.")
    else:
        print("CONNECTION TO TREASURY: NO")
        print("0x391e7c... and 0xf186cb... appear to be independent.")
        print("The gas station is shared public infrastructure.")


asyncio.run(main())
