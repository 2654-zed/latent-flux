"""One-shot script: full archive history for the confirmed operator."""
import asyncio
import os
import sys
from pathlib import Path

import aiohttp

# Load env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

HTTP_URL = os.environ["ARB_WSS_URL"].replace("wss://", "https://")
OPERATOR = "0xe93d64f3fbc352131e79fc5578cbe44b66697f86"


async def get_transfers(direction, category, page_key=None):
    params = {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "category": category,
        "maxCount": "0x3e8",
        "order": "asc",
        "withMetadata": True,
    }
    if direction == "from":
        params["fromAddress"] = OPERATOR
    else:
        params["toAddress"] = OPERATOR
    if page_key:
        params["pageKey"] = page_key

    payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [params]}
    async with aiohttp.ClientSession() as s:
        async with s.post(HTTP_URL, json=payload, timeout=30) as resp:
            data = await resp.json()
            result = data.get("result", {})
            return result.get("transfers", []), result.get("pageKey")


async def main():
    all_transfers = []

    for direction, cats, cat_label in [
        ("from", ["external"], "ETH"),
        ("to", ["external", "internal"], "ETH"),
        ("from", ["erc20"], "ERC20"),
        ("to", ["erc20"], "ERC20"),
    ]:
        transfers, pk = await get_transfers(direction, cats)
        dir_label = "OUT" if direction == "from" else "IN"
        all_transfers.extend([{**t, "_dir": dir_label, "_cat": cat_label} for t in transfers])
        while pk:
            transfers, pk = await get_transfers(direction, cats, pk)
            all_transfers.extend([{**t, "_dir": dir_label, "_cat": cat_label} for t in transfers])
            await asyncio.sleep(0.1)

    all_transfers.sort(key=lambda t: int(t.get("blockNum", "0x0"), 16))
    print(f"Total transfers: {len(all_transfers)}")
    print()

    eth_in = sum(float(t.get("value", 0)) for t in all_transfers if t["_dir"] == "IN" and t["_cat"] == "ETH")
    eth_out = sum(float(t.get("value", 0)) for t in all_transfers if t["_dir"] == "OUT" and t["_cat"] == "ETH")

    usdc_in = usdc_out = 0.0
    for t in all_transfers:
        asset = (t.get("asset") or "").upper()
        val = float(t.get("value", 0))
        if "USDC" in asset:
            if t["_dir"] == "IN":
                usdc_in += val
            else:
                usdc_out += val

    print("=== FINANCIAL SUMMARY ===")
    print(f"ETH in:    {eth_in:.6f}")
    print(f"ETH out:   {eth_out:.6f}")
    print(f"USDC in:   ${usdc_in:,.2f}")
    print(f"USDC out:  ${usdc_out:,.2f}")
    print(f"USDC net:  ${usdc_in - usdc_out:,.2f}")
    print()

    if all_transfers:
        first_ts = all_transfers[0].get("metadata", {}).get("blockTimestamp", "?")
        last_ts = all_transfers[-1].get("metadata", {}).get("blockTimestamp", "?")
        print(f"First activity: {first_ts}")
        print(f"Last activity:  {last_ts}")
    print()

    counterparties = set()
    for t in all_transfers:
        cp = t.get("to" if t["_dir"] == "OUT" else "from", "").lower()
        if cp and cp != OPERATOR:
            counterparties.add(cp)
    print(f"Unique counterparties: {len(counterparties)}")
    print()

    # USDC recipients
    usdc_recipients = {}
    for t in all_transfers:
        if t["_dir"] == "OUT" and "USDC" in (t.get("asset") or "").upper():
            to = t.get("to", "").lower()
            val = float(t.get("value", 0))
            ts = t.get("metadata", {}).get("blockTimestamp", "")
            if to not in usdc_recipients:
                usdc_recipients[to] = {"total": 0, "count": 0, "first": ts, "last": ts}
            usdc_recipients[to]["total"] += val
            usdc_recipients[to]["count"] += 1
            if ts < usdc_recipients[to]["first"]:
                usdc_recipients[to]["first"] = ts
            if ts > usdc_recipients[to]["last"]:
                usdc_recipients[to]["last"] = ts

    print(f"=== USDC RECIPIENTS ({len(usdc_recipients)} addresses) ===")
    for addr, info in sorted(usdc_recipients.items(), key=lambda x: -x[1]["total"]):
        print(f"  {addr}")
        print(f"    total: ${info['total']:,.2f} ({info['count']} txs)")
        print(f"    period: {info['first'][:10]} to {info['last'][:10]}")

    # USDC sources
    usdc_senders = {}
    for t in all_transfers:
        if t["_dir"] == "IN" and "USDC" in (t.get("asset") or "").upper():
            fr = t.get("from", "").lower()
            val = float(t.get("value", 0))
            ts = t.get("metadata", {}).get("blockTimestamp", "")
            if fr not in usdc_senders:
                usdc_senders[fr] = {"total": 0, "count": 0, "first": ts}
            usdc_senders[fr]["total"] += val
            usdc_senders[fr]["count"] += 1

    print()
    print(f"=== USDC SOURCES ({len(usdc_senders)} addresses) ===")
    for addr, info in sorted(usdc_senders.items(), key=lambda x: -x[1]["total"]):
        print(f"  {addr}: ${info['total']:,.2f} ({info['count']} txs) first={info['first'][:10]}")

    # ETH funding
    eth_senders = {}
    for t in all_transfers:
        if t["_dir"] == "IN" and t["_cat"] == "ETH":
            fr = t.get("from", "").lower()
            val = float(t.get("value", 0))
            ts = t.get("metadata", {}).get("blockTimestamp", "")
            if fr not in eth_senders:
                eth_senders[fr] = {"total": 0, "count": 0, "first": ts}
            eth_senders[fr]["total"] += val
            eth_senders[fr]["count"] += 1

    print()
    print(f"=== ETH FUNDING SOURCES ({len(eth_senders)} addresses) ===")
    for addr, info in sorted(eth_senders.items(), key=lambda x: -x[1]["total"]):
        print(f"  {addr}: {info['total']:.6f} ETH ({info['count']} txs) first={info['first'][:10]}")

    # Daily timeline
    daily = {}
    for t in all_transfers:
        ts = t.get("metadata", {}).get("blockTimestamp", "")[:10]
        if not ts:
            continue
        if ts not in daily:
            daily[ts] = {"txs": 0, "usdc_out": 0, "usdc_in": 0, "contracts_deployed": 0}
        daily[ts]["txs"] += 1
        val = float(t.get("value", 0))
        asset = (t.get("asset") or "").upper()
        if t["_dir"] == "OUT" and "USDC" in asset:
            daily[ts]["usdc_out"] += val
        elif t["_dir"] == "IN" and "USDC" in asset:
            daily[ts]["usdc_in"] += val
        # Contract creation = ETH out to null/new address with 0 value
        if t["_dir"] == "OUT" and t["_cat"] == "ETH" and val == 0:
            daily[ts]["contracts_deployed"] += 1

    print()
    print("=== DAILY TIMELINE ===")
    for date in sorted(daily.keys()):
        d = daily[date]
        bar = "#" * min(d["txs"], 50)
        notes = ""
        if d["usdc_out"] > 0:
            notes += f" out=${d['usdc_out']:,.0f}"
        if d["usdc_in"] > 0:
            notes += f" in=${d['usdc_in']:,.0f}"
        if d["contracts_deployed"] > 0:
            notes += f" deploys={d['contracts_deployed']}"
        print(f"  {date}  {bar:50s} {d['txs']:>4} txs{notes}")


asyncio.run(main())
