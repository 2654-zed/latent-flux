"""Trace funding sources for all 43 tx.origin campaign deployers on Base."""
import asyncio
import aiohttp
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

BASE_HTTP = os.environ["BASE_WSS_URL"].replace("wss://", "https://")
ARB_HTTP = os.environ["ARB_WSS_URL"].replace("wss://", "https://")

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

DB = "surveillance/data/surveillance.db"


def get_deployers():
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT DISTINCT deployer_address FROM contracts
        WHERE detection_timestamp > '2026-03-20T18:00'
          AND bytecode_pattern_notes LIKE '%tx.origin%'
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


async def get_funding(session, url, addr):
    async with session.post(url, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [{"fromBlock": "0x0", "toBlock": "latest",
                     "toAddress": addr, "category": ["external"],
                     "maxCount": "0x5", "order": "asc",
                     "withMetadata": True}]
    }, timeout=aiohttp.ClientTimeout(total=20)) as r:
        data = (await r.json()).get("result", {})
        return data.get("transfers", [])


async def main():
    deployers = get_deployers()
    print(f"Tracing funding for {len(deployers)} tx.origin deployers on Base")
    print()

    funder_map = {}       # deployer -> funder address
    funder_counts = Counter()  # funder -> how many deployers it funded
    funding_times = []     # (deployer, funder, timestamp, eth_amount)
    no_funding = []

    async with aiohttp.ClientSession() as s:
        # Batch in groups of 5 to avoid rate limits
        for i in range(0, len(deployers), 5):
            batch = deployers[i:i+5]
            tasks = [get_funding(s, BASE_HTTP, addr) for addr in batch]
            results = await asyncio.gather(*tasks)

            for addr, transfers in zip(batch, results):
                if transfers:
                    first = transfers[0]
                    funder = (first.get("from") or "").lower()
                    val = float(first.get("value") or 0)
                    ts = (first.get("metadata") or {}).get("blockTimestamp", "")

                    funder_map[addr] = funder
                    funder_counts[funder] += 1
                    funding_times.append((addr, funder, ts, val))
                else:
                    no_funding.append(addr)

            await asyncio.sleep(0.3)

    # Report
    print(f"=== FUNDING RESULTS ===")
    print(f"Funded: {len(funder_map)}")
    print(f"No funding found: {len(no_funding)}")
    print()

    print(f"=== UNIQUE FUNDERS ===")
    for funder, count in funder_counts.most_common():
        org_tag = ORG_001.get(funder, "")
        tag = f"  *** {org_tag} ***" if org_tag else ""
        print(f"  {funder} -> funded {count} of {len(deployers)} deployers{tag}")

    # Check if any funder funded >1 deployer
    shared = {f: c for f, c in funder_counts.items() if c > 1}
    print()
    if shared:
        print(f"=== SHARED FUNDERS ({len(shared)}) ===")
        for funder, count in sorted(shared.items(), key=lambda x: -x[1]):
            org_tag = ORG_001.get(funder, "")
            tag = f"  *** {org_tag} ***" if org_tag else ""
            print(f"  {funder} funded {count} deployers{tag}")
            funded_addrs = [addr for addr, f in funder_map.items() if f == funder]
            for addr in funded_addrs[:5]:
                ts_entry = next((t for t in funding_times if t[0] == addr), None)
                ts_str = ts_entry[2][:19] if ts_entry else "?"
                eth = ts_entry[3] if ts_entry else 0
                print(f"    -> {addr[:18]}... {eth:.6f} ETH at {ts_str}")
            if len(funded_addrs) > 5:
                print(f"    ... and {len(funded_addrs) - 5} more")
    else:
        print("NO shared funders found — each deployer funded independently")

    # Timing analysis
    print()
    print(f"=== FUNDING TIMING ===")
    if funding_times:
        times_sorted = sorted(funding_times, key=lambda x: x[2])
        first_ts = times_sorted[0][2][:19]
        last_ts = times_sorted[-1][2][:19]
        print(f"Earliest funding: {first_ts}")
        print(f"Latest funding:   {last_ts}")

        # Parse timestamps and check window
        from datetime import datetime
        parsed = []
        for _, _, ts, _ in times_sorted:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
                parsed.append(dt)
            except:
                pass

        if len(parsed) >= 2:
            span = (parsed[-1] - parsed[0]).total_seconds()
            print(f"Funding window: {span:.0f} seconds ({span/3600:.1f} hours)")
            if span < 3600:
                print("  -> ALL FUNDED WITHIN 1 HOUR: coordinated deployment")
            elif span < 86400:
                print("  -> All funded within 24 hours")

        # ETH amounts
        amounts = [t[3] for t in funding_times]
        print(f"ETH amounts: min={min(amounts):.6f} max={max(amounts):.6f} avg={sum(amounts)/len(amounts):.6f}")
        amount_counts = Counter(round(a, 6) for a in amounts)
        if len(amount_counts) <= 3:
            print(f"  -> UNIFORM AMOUNTS: {dict(amount_counts)}")

    # Check shared funders against org_001 on Arbitrum too
    if shared:
        print()
        print("=== CHECKING SHARED FUNDERS ON ARBITRUM ===")
        async with aiohttp.ClientSession() as s:
            for funder in shared:
                # Check nonce on both chains
                for chain_name, url in [("Base", BASE_HTTP), ("Arb", ARB_HTTP)]:
                    async with s.post(url, json={
                        "jsonrpc": "2.0", "id": 1,
                        "method": "eth_getTransactionCount",
                        "params": [funder, "latest"]
                    }, timeout=aiohttp.ClientTimeout(total=10)) as r:
                        nonce = int((await r.json()).get("result", "0x0"), 16)
                    print(f"  {funder[:18]}... {chain_name} nonce={nonce}")
                    await asyncio.sleep(0.1)

                # Check if funder received ETH from org_001 on Arb
                arb_funding = await get_funding(s, ARB_HTTP, funder)
                await asyncio.sleep(0.2)
                if arb_funding:
                    for t in arb_funding[:3]:
                        fr = (t.get("from") or "").lower()
                        org_tag = ORG_001.get(fr, "")
                        tag = f"  *** {org_tag} ***" if org_tag else ""
                        val = float(t.get("value") or 0)
                        ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
                        print(f"    Arb funder: {fr[:18]}...{tag} {val:.6f} ETH at {ts}")

    # Verdict
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if shared and max(funder_counts.values()) >= 10:
        top_funder = funder_counts.most_common(1)[0]
        print(f"COORDINATED CAMPAIGN CONFIRMED")
        print(f"Treasury: {top_funder[0]}")
        print(f"Funded {top_funder[1]} of {len(deployers)} deployers")
        org_link = ORG_001.get(top_funder[0], "")
        if org_link:
            print(f"*** LINKS TO ORG_001: {org_link} ***")
        else:
            print("No direct org_001 link at funding layer")
    elif shared:
        print(f"PARTIAL COORDINATION: {len(shared)} shared funders")
    else:
        print("NO SHARED FUNDERS — either:")
        print("  A) Each deployer funded via separate disposable wallets (sophisticated opsec)")
        print("  B) Truly independent deployers using same bytecode template")
        print("  Check: are funders themselves funded by a common source? (2-hop trace)")


asyncio.run(main())
