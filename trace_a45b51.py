"""Trace 0xa45b51 — the biggest recipient in Bot_A's operator network."""
import asyncio
import aiohttp
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

ARB = os.environ["ARB_WSS_URL"].replace("wss://", "https://")
TARGET = "0xa45b5130f36cdca45667738e2a258ab09f4a5f7f"
BOT_A = "0x84792c2a999c0c12a0a0e00fbadb090cb2f84ad6"
FUNDER = "0x9ebed688a923060d3d713cb27d356ced4fa005c5"
FUNDER2 = "0x260790b15bbae85c85987cbb0a533e3c2ce9ca53"

NETWORK = {BOT_A.lower(), FUNDER.lower(), FUNDER2.lower(), TARGET.lower()}


async def xfers(session, addr, direction="from", cats=None, count=50):
    params = {"fromBlock": "0x0", "toBlock": "latest",
              "category": cats or ["external"],
              "maxCount": hex(count), "order": "desc", "withMetadata": True}
    params["fromAddress" if direction == "from" else "toAddress"] = addr
    async with session.post(ARB, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "alchemy_getAssetTransfers", "params": [params]
    }, timeout=aiohttp.ClientTimeout(total=30)) as r:
        return (await r.json()).get("result", {}).get("transfers", [])


async def get_info(session, addr):
    async with session.post(ARB, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getTransactionCount", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        n = int((await r.json()).get("result", "0x0"), 16)
    async with session.post(ARB, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getBalance", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        b = int((await r.json()).get("result", "0x0"), 16) / 1e18
    async with session.post(ARB, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getCode", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        code = (await r.json()).get("result", "0x")
        cs = len(code) // 2 - 1 if len(code) > 2 else 0
    return n, b, cs


async def main():
    async with aiohttp.ClientSession() as s:
        # Basic profile
        n, b, cs = await get_info(s, TARGET)
        print(f"=== TARGET: {TARGET} ===")
        tp = f"CONTRACT ({cs}B)" if cs > 0 else "EOA"
        print(f"  Type: {tp}")
        print(f"  Nonce: {n:,}")
        print(f"  Balance: {b:.6f} ETH")
        await asyncio.sleep(0.2)

        # ETH inflows
        print()
        print("=== ETH INFLOWS ===")
        eth_in = await xfers(s, TARGET, "to", ["external"], 30)
        await asyncio.sleep(0.3)
        total_in = 0
        in_sources = {}
        for t in eth_in:
            fr = (t.get("from") or "").lower()
            val = float(t.get("value") or 0)
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            total_in += val
            if fr not in in_sources:
                in_sources[fr] = {"total": 0, "count": 0, "last": ts}
            in_sources[fr]["total"] += val
            in_sources[fr]["count"] += 1

        print(f"  Total: {total_in:.4f} ETH from {len(in_sources)} sources")
        for addr, info in sorted(in_sources.items(), key=lambda x: -x[1]["total"]):
            net_label = ""
            if addr in NETWORK:
                labels = {BOT_A.lower(): "BOT_A", FUNDER.lower(): "PRIMARY_FUNDER",
                          FUNDER2.lower(): "SECONDARY_FUNDER"}
                net_label = f" [{labels.get(addr, 'NETWORK')}]"
            print(f"  <- {addr}{net_label} | {info['total']:.4f} ETH ({info['count']} txs)")

        # ETH outflows
        print()
        print("=== ETH OUTFLOWS ===")
        eth_out = await xfers(s, TARGET, "from", ["external"], 30)
        await asyncio.sleep(0.3)
        total_out = 0
        out_dests = {}
        for t in eth_out:
            to = (t.get("to") or "").lower()
            val = float(t.get("value") or 0)
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            total_out += val
            if to not in out_dests:
                out_dests[to] = {"total": 0, "count": 0, "last": ts}
            out_dests[to]["total"] += val
            out_dests[to]["count"] += 1

        print(f"  Total: {total_out:.4f} ETH to {len(out_dests)} recipients")
        for addr, info in sorted(out_dests.items(), key=lambda x: -x[1]["total"]):
            net_label = ""
            if addr in NETWORK:
                labels = {BOT_A.lower(): "BOT_A", FUNDER.lower(): "PRIMARY_FUNDER",
                          FUNDER2.lower(): "SECONDARY_FUNDER"}
                net_label = f" [{labels.get(addr, 'NETWORK')}]"
            print(f"  -> {addr}{net_label} | {info['total']:.4f} ETH ({info['count']} txs)")

        # ERC-20 inflows
        print()
        print("=== ERC-20 INFLOWS ===")
        erc_in = await xfers(s, TARGET, "to", ["erc20"], 50)
        await asyncio.sleep(0.3)
        tokens_in = {}
        erc_senders = set()
        for t in erc_in:
            asset = (t.get("asset") or "?")[:20]
            val = float(t.get("value") or 0)
            tokens_in[asset] = tokens_in.get(asset, 0) + val
            erc_senders.add((t.get("from") or "").lower())
        print(f"  {len(erc_in)} transfers from {len(erc_senders)} senders")
        for asset, total in sorted(tokens_in.items(), key=lambda x: -x[1])[:10]:
            print(f"    {asset}: {total:,.4f}")

        # ERC-20 outflows
        print()
        print("=== ERC-20 OUTFLOWS ===")
        erc_out = await xfers(s, TARGET, "from", ["erc20"], 50)
        await asyncio.sleep(0.3)
        tokens_out = {}
        erc_recipients = set()
        for t in erc_out:
            asset = (t.get("asset") or "?")[:20]
            val = float(t.get("value") or 0)
            tokens_out[asset] = tokens_out.get(asset, 0) + val
            erc_recipients.add((t.get("to") or "").lower())
        print(f"  {len(erc_out)} transfers to {len(erc_recipients)} recipients")
        for asset, total in sorted(tokens_out.items(), key=lambda x: -x[1])[:10]:
            print(f"    {asset}: {total:,.4f}")

        # Internal transactions (contract calls with value)
        print()
        print("=== INTERNAL TRANSACTIONS ===")
        int_in = await xfers(s, TARGET, "to", ["internal"], 30)
        await asyncio.sleep(0.3)
        int_out = await xfers(s, TARGET, "from", ["internal"], 30)
        await asyncio.sleep(0.3)
        int_in_total = sum(float(t.get("value") or 0) for t in int_in)
        int_out_total = sum(float(t.get("value") or 0) for t in int_out)
        print(f"  Internal in:  {int_in_total:.6f} ETH ({len(int_in)} transfers)")
        print(f"  Internal out: {int_out_total:.6f} ETH ({len(int_out)} transfers)")
        if int_in:
            print(f"  Recent internal inflows:")
            for t in int_in[:5]:
                fr = (t.get("from") or "?")[:18]
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
                print(f"    <- {fr}... {val:.6f} ETH at {ts}")
        if int_out:
            print(f"  Recent internal outflows:")
            for t in int_out[:5]:
                to = (t.get("to") or "?")[:18]
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
                print(f"    -> {to}... {val:.6f} ETH at {ts}")

        # P&L Summary
        print()
        print("=" * 60)
        print("P&L SUMMARY")
        print("=" * 60)
        eth_net = total_in - total_out + b
        usdc_in = tokens_in.get("USDC", 0) + tokens_in.get("USDC.e", 0)
        usdc_out = tokens_out.get("USDC", 0) + tokens_out.get("USDC.e", 0)
        usdt_in = tokens_in.get("USDT", 0)
        usdt_out = tokens_out.get("USDT", 0)
        weth_in = tokens_in.get("WETH", 0)
        weth_out = tokens_out.get("WETH", 0)

        print(f"  ETH:  in={total_in:.4f}  out={total_out:.4f}  bal={b:.4f}")
        print(f"  USDC: in={usdc_in:,.2f}  out={usdc_out:,.2f}  net={usdc_in-usdc_out:+,.2f}")
        print(f"  USDT: in={usdt_in:,.2f}  out={usdt_out:,.2f}  net={usdt_in-usdt_out:+,.2f}")
        print(f"  WETH: in={weth_in:.4f}  out={weth_out:.4f}  net={weth_in-weth_out:+.4f}")
        print(f"  Internal: in={int_in_total:.4f}  out={int_out_total:.4f}")
        print()

        # Is this profitable?
        eth_price = 2100
        total_revenue = (usdc_in + usdt_in + weth_in * eth_price + int_in_total * eth_price)
        total_cost = total_in * eth_price  # ETH received = gas budget
        gas_burned = (total_in - total_out - b) * eth_price
        print(f"  Stablecoin revenue: ${usdc_in + usdt_in:,.2f}")
        print(f"  WETH revenue: ${weth_in * eth_price:,.2f}")
        print(f"  Internal ETH revenue: ${int_in_total * eth_price:,.2f}")
        print(f"  Gas burned (est): ${gas_burned:,.2f}")
        print(f"  Stablecoin sent out: ${usdc_out + usdt_out:,.2f}")

        # Compare to Bot_A
        print()
        print("=== COMPARISON WITH BOT_A ===")
        print(f"  Bot_A:  nonce=375,502 | funded=2.15 ETH | ERC-20 revenue=$0 | LOSS")
        print(f"  Target: nonce={n:,} | funded={total_in:.2f} ETH | USDC={usdc_in:,.0f} USDT={usdt_in:,.0f}")
        if usdc_in + usdt_in > 0:
            print(f"  *** TARGET HAS STABLECOIN REVENUE — THIS MAY BE THE PROFITABLE BOT ***")
        elif int_in_total > 0.1:
            print(f"  *** TARGET HAS INTERNAL TX REVENUE — PROFITS FLOW THROUGH CONTRACTS ***")
        else:
            print(f"  Target also shows no visible revenue — same pattern as Bot_A")


asyncio.run(main())
