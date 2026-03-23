"""Trace Bot_A's funder network — who else do they fund, are any profitable?"""
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
FUNDER = "0x9ebed688a923060d3d713cb27d356ced4fa005c5"
FUNDER2 = "0x260790b15bbae85c85987cbb0a533e3c2ce9ca53"
BOT_A = "0x84792c2a999c0c12a0a0e00fbadb090cb2f84ad6"


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


async def info(session, addr):
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
    return n, b


async def main():
    async with aiohttp.ClientSession() as s:
        # 1. Profile primary funder
        n, b = await info(s, FUNDER)
        print(f"=== PRIMARY FUNDER: {FUNDER} ===")
        print(f"  Nonce: {n:,} | Balance: {b:.6f} ETH")
        await asyncio.sleep(0.2)

        # Who funds the funder?
        print()
        print("Who funds the funder?")
        fin = await xfers(s, FUNDER, "to", ["external"], 20)
        await asyncio.sleep(0.3)
        sources = {}
        for t in fin:
            fr = (t.get("from") or "").lower()
            val = float(t.get("value") or 0)
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            if fr not in sources:
                sources[fr] = {"total": 0, "count": 0, "last": ts}
            sources[fr]["total"] += val
            sources[fr]["count"] += 1
        for addr, info_ in sorted(sources.items(), key=lambda x: -x[1]["total"]):
            print(f"  <- {addr} | {info_['total']:.4f} ETH ({info_['count']} txs)")

        # Where does funder send? (all recipients)
        print()
        print("Funder outflows (all recipients):")
        fout = await xfers(s, FUNDER, "from", ["external"], 50)
        await asyncio.sleep(0.3)
        recipients = {}
        for t in fout:
            to = (t.get("to") or "").lower()
            val = float(t.get("value") or 0)
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            if to not in recipients:
                recipients[to] = {"total": 0, "count": 0, "first": ts, "last": ts}
            recipients[to]["total"] += val
            recipients[to]["count"] += 1
            if ts < recipients[to]["first"]:
                recipients[to]["first"] = ts

        print(f"  {len(fout)} outflows to {len(recipients)} unique addresses")
        print()

        # Profile each recipient
        for addr, ri in sorted(recipients.items(), key=lambda x: -x[1]["total"]):
            is_bot_a = (addr == BOT_A.lower())
            label = " *** BOT_A ***" if is_bot_a else ""

            rn, rb = await info(s, addr)
            await asyncio.sleep(0.15)

            # Check ERC-20 outflows (revenue signal)
            erc = await xfers(s, addr, "from", ["erc20"], 5)
            await asyncio.sleep(0.15)
            tokens = {}
            for t in erc:
                a = (t.get("asset") or "?")[:20]
                v = float(t.get("value") or 0)
                tokens[a] = tokens.get(a, 0) + v

            erc_str = str(tokens) if tokens else "NONE"

            print(f"  {addr}{label}")
            print(f"    Funded: {ri['total']:.4f} ETH ({ri['count']} txs) | nonce={rn:,} | bal={rb:.6f}")
            print(f"    Period: {ri['first']} to {ri['last']}")
            print(f"    ERC-20 out: {erc_str}")
            if rn > 100000:
                print(f"    *** HIGH NONCE: {rn:,} — active bot ***")
            if tokens:
                print(f"    *** HAS TOKEN REVENUE ***")
            print()

        # 2. Profile secondary funder
        print()
        print(f"=== SECONDARY FUNDER: {FUNDER2} ===")
        n2, b2 = await info(s, FUNDER2)
        print(f"  Nonce: {n2:,} | Balance: {b2:.6f} ETH")
        await asyncio.sleep(0.2)

        f2out = await xfers(s, FUNDER2, "from", ["external"], 20)
        await asyncio.sleep(0.3)
        r2 = {}
        for t in f2out:
            to = (t.get("to") or "").lower()
            val = float(t.get("value") or 0)
            r2[to] = r2.get(to, 0) + val

        print(f"  Outflows to {len(r2)} addresses:")
        for addr, total in sorted(r2.items(), key=lambda x: -x[1]):
            ba = " [BOT_A]" if addr == BOT_A.lower() else ""
            pf = " [PRIMARY_FUNDER]" if addr == FUNDER.lower() else ""
            print(f"    -> {addr}{ba}{pf} {total:.4f} ETH")

        # Does Bot_A send anything back to either funder?
        print()
        print("=== DOES BOT_A SEND ANYTHING BACK? ===")
        bot_out = await xfers(s, BOT_A, "from", ["external", "erc20"], 20)
        await asyncio.sleep(0.3)
        for t in bot_out:
            to = (t.get("to") or "").lower()
            val = t.get("value", 0)
            asset = t.get("asset", "ETH")
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            label = ""
            if to == FUNDER.lower():
                label = " [PRIMARY_FUNDER]"
            elif to == FUNDER2.lower():
                label = " [SECONDARY_FUNDER]"
            print(f"  -> {to[:18]}...{label} {val} {asset} at {ts}")

        if not bot_out:
            print("  Bot_A has ZERO outflows (never sends anything to anyone)")


asyncio.run(main())
