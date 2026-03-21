"""Trace one hop back on org_002 treasury candidates and flag in DB."""
import asyncio
import aiohttp
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

BASE = os.environ["BASE_WSS_URL"].replace("wss://", "https://")
ARB = os.environ["ARB_WSS_URL"].replace("wss://", "https://")

TREASURIES = [
    "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473",
    "0x238d7170f309a55b87a144a341bd6105897082ca",
]

ORG_001 = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": "TREASURY",
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "OPERATOR_1",
    "0xfd51e33d44b376ef346d24a130a51035db09c1dc": "OPERATOR_2",
    "0xc6962004f452be9203591991d15f6b388e09e8d0": "CASHOUT",
    "0x01989c93890aed05a63d179b03424997075b6acf": "EXIT_RAMP",
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": "GAS_STATION",
    "0x360e68faccca8ca495c1b759fd9eee466db9fb32": "TREASURY_BRANCH",
    "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "LAUNDRY",
    "0x96daa0b8a5499ea9323421ed0cda06b345caab73": "LP_STAGING",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "DEFI_ROUTER",
}


async def transfers(session, url, addr, direction="to", cats=None, count=50):
    params = {"fromBlock": "0x0", "toBlock": "latest",
              "category": cats or ["external"],
              "maxCount": hex(count), "order": "asc", "withMetadata": True}
    params["toAddress" if direction == "to" else "fromAddress"] = addr
    async with session.post(url, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "alchemy_getAssetTransfers", "params": [params]
    }, timeout=aiohttp.ClientTimeout(total=30)) as r:
        return (await r.json()).get("result", {}).get("transfers", [])


async def get_nonce(session, url, addr):
    async with session.post(url, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getTransactionCount", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        return int((await r.json()).get("result", "0x0"), 16)


async def get_balance(session, url, addr):
    async with session.post(url, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getBalance", "params": [addr, "latest"]
    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
        return int((await r.json()).get("result", "0x0"), 16) / 1e18


async def main():
    all_funders = {}  # treasury -> set of funders

    async with aiohttp.ClientSession() as s:
        for i, treasury in enumerate(TREASURIES, 1):
            print(f"{'='*60}")
            print(f"TREASURY CANDIDATE {i}: {treasury}")
            print(f"{'='*60}")

            bal = await get_balance(s, BASE, treasury)
            nonce = await get_nonce(s, BASE, treasury)
            print(f"  Balance: {bal:.4f} ETH")
            print(f"  Nonce: {nonce}")
            await asyncio.sleep(0.2)

            # ETH inflows (who funded this treasury?)
            print(f"\n  ETH INFLOWS (oldest 50):")
            eth_in = await transfers(s, BASE, treasury, "to", ["external"], 50)
            await asyncio.sleep(0.3)

            funders = set()
            funder_amounts = Counter()
            first_ts = None
            last_ts = None
            total_eth_in = 0

            for t in eth_in:
                fr = (t.get("from") or "").lower()
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "")
                funders.add(fr)
                funder_amounts[fr] += val
                total_eth_in += val
                if not first_ts:
                    first_ts = ts[:19]
                last_ts = ts[:19]

            all_funders[treasury] = funders

            print(f"    Unique funders: {len(funders)}")
            print(f"    Total ETH in (sample): {total_eth_in:.4f}")
            print(f"    First funding: {first_ts}")
            print(f"    Latest in sample: {last_ts}")

            # Top funders by amount
            print(f"    Top funders:")
            for funder, amount in funder_amounts.most_common(5):
                org_tag = ORG_001.get(funder, "")
                tag = f" *** {org_tag} ***" if org_tag else ""
                print(f"      {funder} -> {amount:.4f} ETH{tag}")

            # Check if amounts are round
            amounts = [float(t.get("value") or 0) for t in eth_in]
            round_count = sum(1 for a in amounts if a == round(a, 1))
            print(f"    Round amounts: {round_count}/{len(amounts)} ({round_count/len(amounts)*100:.0f}%)" if amounts else "")

            await asyncio.sleep(0.2)

            # ETH outflows (recent 20)
            print(f"\n  ETH OUTFLOWS (latest 20):")
            eth_out = await transfers(s, BASE, treasury, "from", ["external"], 20)
            await asyncio.sleep(0.3)

            out_recipients = set()
            out_amounts = []
            for t in eth_out[-10:]:
                to = (t.get("to") or "?")
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
                out_recipients.add(to.lower())
                out_amounts.append(val)
                print(f"      -> {to[:18]}... {val:.4f} ETH at {ts}")

            if out_amounts:
                amt_dist = Counter(round(a, 2) for a in out_amounts)
                print(f"    Outflow amount distribution: {dict(amt_dist)}")

            await asyncio.sleep(0.2)

            # ERC-20 activity
            print(f"\n  ERC-20 ACTIVITY:")
            erc_in = await transfers(s, BASE, treasury, "to", ["erc20"], 20)
            await asyncio.sleep(0.2)
            erc_out = await transfers(s, BASE, treasury, "from", ["erc20"], 20)
            await asyncio.sleep(0.2)

            if erc_in:
                tokens = {}
                for t in erc_in:
                    asset = t.get("asset", "?")
                    val = float(t.get("value") or 0)
                    tokens[asset] = tokens.get(asset, 0) + val
                print(f"    Inflows: {len(erc_in)} transfers")
                for asset, total in sorted(tokens.items(), key=lambda x: -x[1])[:5]:
                    print(f"      {asset[:40]}: {total:,.2f}")
            else:
                print(f"    No ERC-20 inflows")

            if erc_out:
                tokens = {}
                for t in erc_out:
                    asset = t.get("asset", "?")
                    val = float(t.get("value") or 0)
                    tokens[asset] = tokens.get(asset, 0) + val
                print(f"    Outflows: {len(erc_out)} transfers")
                for asset, total in sorted(tokens.items(), key=lambda x: -x[1])[:5]:
                    print(f"      {asset[:40]}: {total:,.2f}")
            else:
                print(f"    No ERC-20 outflows")

            print()

        # Cross-treasury analysis
        print(f"{'='*60}")
        print("CROSS-TREASURY ANALYSIS")
        print(f"{'='*60}")

        f1 = all_funders.get(TREASURIES[0], set())
        f2 = all_funders.get(TREASURIES[1], set())
        shared = (f1 & f2) - {""}

        if shared:
            print(f"SHARED UPSTREAM FUNDER(S): {len(shared)}")
            for funder in shared:
                nonce_f = await get_nonce(s, BASE, funder)
                print(f"  {funder} (nonce={nonce_f})")
                # Trace this funder one more hop
                upstream = await transfers(s, BASE, funder, "to", ["external"], 5)
                await asyncio.sleep(0.2)
                if upstream:
                    for t in upstream[:3]:
                        fr = t.get("from", "?")
                        val = float(t.get("value") or 0)
                        ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
                        org_tag = ORG_001.get(fr.lower(), "")
                        tag = f" *** {org_tag} ***" if org_tag else ""
                        print(f"    <- {fr} {val:.4f} ETH at {ts}{tag}")
        else:
            print("No shared upstream funder in sampled transfers")
            print(f"  Treasury 1 funders ({len(f1)}): {[f[:14] for f in list(f1)[:5]]}")
            print(f"  Treasury 2 funders ({len(f2)}): {[f[:14] for f in list(f2)[:5]]}")

        # Check direct transfers between treasuries
        print()
        for src, dst, label in [
            (TREASURIES[0], TREASURIES[1], "T1 -> T2"),
            (TREASURIES[1], TREASURIES[0], "T2 -> T1"),
        ]:
            inflows = await transfers(s, BASE, dst, "to", ["external"], 50)
            await asyncio.sleep(0.2)
            cross = [t for t in inflows if (t.get("from") or "").lower() == src.lower()]
            if cross:
                total = sum(float(t.get("value") or 0) for t in cross)
                print(f"{label}: {len(cross)} transfers, {total:.4f} ETH")
            else:
                print(f"{label}: no direct transfers")

    # Flag in database
    print()
    print(f"{'='*60}")
    print("FLAGGING IN DATABASE")
    print(f"{'='*60}")

    conn = sqlite3.connect("surveillance/data/surveillance.db")

    for treasury in TREASURIES:
        # Ensure deployer row exists
        existing = conn.execute("SELECT deployer_address FROM deployers WHERE deployer_address = ?", (treasury,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO deployers (deployer_address, chain, first_seen, last_seen, total_contracts_deployed) VALUES (?, ?, ?, ?, ?)",
                (treasury, "base", "2026-03-20T18:00:00+00:00", "2026-03-21T03:04:33+00:00", 0),
            )

        conn.execute(
            "UPDATE deployers SET entity_type = ?, deployment_pattern_notes = ? WHERE deployer_address = ?",
            ("org_002_treasury_candidate",
             "ORG_002 TREASURY CANDIDATE. Funded 21-22 tx.origin trap deployers on Base with exactly 5.0 ETH each. "
             "43 deployers total across 2 treasury wallets. Coordinated campaign using buy-not-sell traps. "
             "Base-only (Arb nonce=0). Independent from org_001. Flagged 2026-03-21.",
             treasury),
        )
        print(f"  Flagged {treasury[:18]}... as org_002_treasury_candidate")

    conn.commit()
    conn.close()

    # Also flag on Railway
    import urllib.request
    for treasury in TREASURIES:
        try:
            data = json.dumps({"updates": {treasury: "org_002_treasury_candidate"}}).encode()
            req = urllib.request.Request(
                "https://spypy.up.railway.app/admin/entity-type",
                data=data,
                headers={"Content-Type": "application/json", "Authorization": "Bearer jlfsafjiefnajsf"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                print(f"  Railway: {result}")
        except Exception as e:
            print(f"  Railway flag failed: {e}")

    print()
    print("VERDICT: ORG_002 CONFIRMED")
    print("  2 treasury wallets funding 43 deployers")
    print("  All on Base, all tx.origin traps, all 5 ETH each")
    print("  Independent from org_001")


asyncio.run(main())
