"""Deep investigation: 0xd4624228 — the infrastructure parasite."""
import asyncio
import aiohttp
import os
import sqlite3
import json
from pathlib import Path
from collections import defaultdict

env_path = Path(__file__).resolve().parent / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

BASE = os.environ["BASE_WSS_URL"].replace("wss://", "https://")

CONTRACT = "0xd4624228cce5baa0814c9e7f666a8a2c83b6f159"
DEPLOYER = "0xe8e0c4883d7196a7de87a6489f6da58212dbe813"

ORG_WALLETS = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": "org_001:treasury",
    "0xc6962004f452be9203591991d15f6b388e09e8d0": "org_001:cashout",
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": "org_001:gas_station",
    "0x238d7170f309a55b87a144a341bd6105897082ca": "org_002:treasury_sr",
    "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473": "org_002:treasury_jr",
}


async def xfers(session, addr, direction="to", cats=None, count=50):
    params = {"fromBlock": "0x0", "toBlock": "latest",
              "category": cats or ["external"],
              "maxCount": hex(count), "order": "desc", "withMetadata": True}
    params["toAddress" if direction == "to" else "fromAddress"] = addr
    async with session.post(BASE, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "alchemy_getAssetTransfers", "params": [params]
    }, timeout=aiohttp.ClientTimeout(total=30)) as r:
        return (await r.json()).get("result", {}).get("transfers", [])


async def info(session, addr):
    async with session.post(BASE, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getTransactionCount", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        n = int((await r.json()).get("result", "0x0"), 16)
    async with session.post(BASE, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getBalance", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        b = int((await r.json()).get("result", "0x0"), 16) / 1e18
    async with session.post(BASE, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getCode", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        code = (await r.json()).get("result", "0x")
        cs = len(code) // 2 - 1 if len(code) > 2 else 0
    return n, b, cs


async def main():
    async with aiohttp.ClientSession() as s:
        # 1. DEPLOYER PROFILE
        print("=" * 60)
        print("DEPLOYER: " + DEPLOYER)
        print("=" * 60)
        n, b, cs = await info(s, DEPLOYER)
        print(f"  Type: {'CONTRACT' if cs > 0 else 'EOA'} | Nonce: {n:,} | Balance: {b:.6f} ETH")
        await asyncio.sleep(0.2)

        # Deployer funding
        print()
        print("Deployer ETH funding:")
        eth_in = await xfers(s, DEPLOYER, "to", ["external"], 20)
        await asyncio.sleep(0.3)
        for t in eth_in[:5]:
            fr = t.get("from", "?")
            val = float(t.get("value") or 0)
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            org = ORG_WALLETS.get(fr.lower(), "")
            tag = f" [{org}]" if org else ""
            print(f"  <- {fr}{tag} {val:.6f} ETH at {ts}")

        # Deployer ERC-20 activity
        print()
        print("Deployer ERC-20 outflows:")
        erc_out = await xfers(s, DEPLOYER, "from", ["erc20"], 20)
        await asyncio.sleep(0.3)
        tokens = {}
        for t in erc_out:
            asset = (t.get("asset") or "?")[:20]
            val = float(t.get("value") or 0)
            tokens[asset] = tokens.get(asset, 0) + val
        for asset, total in sorted(tokens.items(), key=lambda x: -x[1])[:5]:
            print(f"  {asset}: {total:,.4f}")
        if not tokens:
            print("  None")

        print()
        print("Deployer ERC-20 inflows:")
        erc_in = await xfers(s, DEPLOYER, "to", ["erc20"], 20)
        await asyncio.sleep(0.3)
        tokens_in = {}
        for t in erc_in:
            asset = (t.get("asset") or "?")[:20]
            val = float(t.get("value") or 0)
            tokens_in[asset] = tokens_in.get(asset, 0) + val
        for asset, total in sorted(tokens_in.items(), key=lambda x: -x[1])[:5]:
            print(f"  {asset}: {total:,.4f}")

        # 2. CONTRACT TOKEN FLOWS
        print()
        print("=" * 60)
        print("CONTRACT: " + CONTRACT)
        print("=" * 60)
        cn, cb, ccs = await info(s, CONTRACT)
        print(f"  Code: {ccs}B | Nonce: {cn} | Balance: {cb:.6f} ETH")
        await asyncio.sleep(0.2)

        print()
        print("Contract ERC-20 inflows (what tokens flow in):")
        c_erc_in = await xfers(s, CONTRACT, "to", ["erc20"], 50)
        await asyncio.sleep(0.3)
        c_tokens_in = defaultdict(lambda: {"total": 0, "senders": set()})
        for t in c_erc_in:
            asset = (t.get("asset") or "?")[:20]
            val = float(t.get("value") or 0)
            fr = (t.get("from") or "").lower()
            c_tokens_in[asset]["total"] += val
            c_tokens_in[asset]["senders"].add(fr)
        for asset, info_ in sorted(c_tokens_in.items(), key=lambda x: -x[1]["total"])[:5]:
            print(f"  {asset}: {info_['total']:,.4f} from {len(info_['senders'])} senders")

        print()
        print("Contract ERC-20 outflows (where tokens go):")
        c_erc_out = await xfers(s, CONTRACT, "from", ["erc20"], 50)
        await asyncio.sleep(0.3)
        c_tokens_out = defaultdict(lambda: {"total": 0, "recipients": set()})
        for t in c_erc_out:
            asset = (t.get("asset") or "?")[:20]
            val = float(t.get("value") or 0)
            to = (t.get("to") or "").lower()
            c_tokens_out[asset]["total"] += val
            c_tokens_out[asset]["recipients"].add(to)
        for asset, info_ in sorted(c_tokens_out.items(), key=lambda x: -x[1]["total"])[:5]:
            print(f"  {asset}: {info_['total']:,.4f} to {len(info_['recipients'])} recipients")

        # Net token flows — the extraction signal
        print()
        print("NET TOKEN FLOW (in - out):")
        all_assets = set(c_tokens_in.keys()) | set(c_tokens_out.keys())
        for asset in all_assets:
            in_val = c_tokens_in.get(asset, {"total": 0})["total"]
            out_val = c_tokens_out.get(asset, {"total": 0})["total"]
            net = in_val - out_val
            if abs(net) > 0.001:
                direction = "ACCUMULATING" if net > 0 else "DISTRIBUTING"
                print(f"  {asset}: in={in_val:,.4f} out={out_val:,.4f} net={net:+,.4f} [{direction}]")

        # 3. WHERE DOES THE DEPLOYER SEND EXTRACTED VALUE?
        print()
        print("=" * 60)
        print("DEPLOYER OUTBOUND TRANSFERS (where profits go)")
        print("=" * 60)
        dep_out_all = await xfers(s, DEPLOYER, "from", ["external", "erc20"], 30)
        await asyncio.sleep(0.3)
        for t in dep_out_all[:10]:
            to = (t.get("to") or "?")
            val = t.get("value", 0)
            asset = (t.get("asset") or "ETH")[:15]
            ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
            cat = t.get("category", "?")
            org = ORG_WALLETS.get(to.lower(), "")
            tag = f" [{org}]" if org else ""
            print(f"  -> {to[:18]}...{tag} {val} {asset} [{cat}] at {ts}")

        # 4. THE FUNDER — trace one hop back
        if eth_in:
            funder = (eth_in[0].get("from") or "").lower()
            print()
            print("=" * 60)
            print(f"FUNDER: {funder}")
            print("=" * 60)
            fn, fb, fcs = await info(s, funder)
            print(f"  Type: {'CONTRACT' if fcs > 0 else 'EOA'} | Nonce: {fn:,} | Balance: {fb:.6f} ETH")
            await asyncio.sleep(0.2)

            # Who else does the funder fund?
            funder_out = await xfers(s, funder, "from", ["external"], 30)
            await asyncio.sleep(0.3)
            fund_recipients = defaultdict(float)
            for t in funder_out:
                to = (t.get("to") or "").lower()
                val = float(t.get("value") or 0)
                fund_recipients[to] += val

            print(f"  Funds {len(fund_recipients)} addresses:")
            for addr, total in sorted(fund_recipients.items(), key=lambda x: -x[1])[:10]:
                is_deployer = " [THIS DEPLOYER]" if addr == DEPLOYER.lower() else ""
                org = ORG_WALLETS.get(addr, "")
                org_tag = f" [{org}]" if org else ""
                print(f"    -> {addr}{is_deployer}{org_tag} {total:.4f} ETH")

            # Does the funder fund any OTHER deployers in our DB?
            conn = sqlite3.connect("surveillance/data/surveillance.db")
            print()
            print("  Funder's recipients that are KNOWN DEPLOYERS:")
            for addr in fund_recipients:
                dep_check = conn.execute("SELECT total_contracts_deployed, entity_type FROM deployers WHERE deployer_address = ?", (addr,)).fetchone()
                if dep_check and dep_check[0] > 0:
                    print(f"    {addr[:18]}... {dep_check[0]} contracts, type={dep_check[1]}")
            conn.close()

        # 5. Check Base chain for the contract's liquidity position
        print()
        print("=" * 60)
        print("LIQUIDITY ANALYSIS")
        print("=" * 60)
        # Check internal transactions (LP pool interactions)
        int_in = await xfers(s, CONTRACT, "to", ["internal"], 20)
        await asyncio.sleep(0.3)
        int_out = await xfers(s, CONTRACT, "from", ["internal"], 20)
        await asyncio.sleep(0.3)

        int_in_total = sum(float(t.get("value") or 0) for t in int_in)
        int_out_total = sum(float(t.get("value") or 0) for t in int_out)
        print(f"  Internal ETH in: {int_in_total:.6f} ({len(int_in)} transfers)")
        print(f"  Internal ETH out: {int_out_total:.6f} ({len(int_out)} transfers)")

        if int_in:
            print("  Recent internal inflows:")
            for t in int_in[:5]:
                fr = (t.get("from") or "?")[:18]
                val = float(t.get("value") or 0)
                print(f"    <- {fr}... {val:.6f} ETH")

        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Contract: {CONTRACT}")
        print(f"  Mechanism: Obfuscated fee-on-transfer (SHA3->SLOAD->JUMPI->MUL)")
        print(f"  Traffic source: Uniswap Universal Router execute() — 2,276 victims")
        print(f"  Revert rate: 0.3% (near-invisible)")
        print(f"  Deployer nonce: {n:,}")
        print(f"  Deployer balance: {b:.6f} ETH")


asyncio.run(main())
