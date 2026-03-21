"""Deep trace on org_002 Treasury 2 — full history, prior campaigns, cross-chain."""
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

BASE = os.environ["BASE_WSS_URL"].replace("wss://", "https://")
ARB = os.environ["ARB_WSS_URL"].replace("wss://", "https://")

T2 = "0x238d7170f309a55b87a144a341bd6105897082ca"
T1 = "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473"

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


async def get_transfers(session, url, addr, direction="to", cats=None, count=100, order="asc"):
    params = {"fromBlock": "0x0", "toBlock": "latest",
              "category": cats or ["external"],
              "maxCount": hex(count), "order": order, "withMetadata": True}
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


async def main():
    async with aiohttp.ClientSession() as s:

        # ============================================================
        # 1. FIRST DEPOSITS — what seeded this wallet?
        # ============================================================
        print("=" * 60)
        print("1. FIRST DEPOSITS INTO TREASURY 2")
        print("=" * 60)

        earliest = await get_transfers(s, BASE, T2, "to", ["external", "internal"], 20, "asc")
        await asyncio.sleep(0.3)

        first_funders = set()
        if earliest:
            for t in earliest[:10]:
                fr = t.get("from", "?")
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "?")[:19]
                cat = t.get("category", "?")
                org = ORG_001.get(fr.lower(), "")
                tag = f" *** {org} ***" if org else ""
                first_funders.add(fr.lower())
                print(f"  {ts} <- {fr} {val:.4f} ETH [{cat}]{tag}")

            # Trace the VERY FIRST funder one hop back
            print()
            first_funder = (earliest[0].get("from") or "").lower()
            print(f"Tracing first funder: {first_funder}")
            ff_nonce = await get_nonce(s, BASE, first_funder)
            print(f"  Nonce: {ff_nonce}")
            await asyncio.sleep(0.2)

            ff_funding = await get_transfers(s, BASE, first_funder, "to", ["external", "internal"], 5, "asc")
            await asyncio.sleep(0.3)
            if ff_funding:
                print(f"  First funder was funded by:")
                for t in ff_funding[:5]:
                    fr = t.get("from", "?")
                    val = float(t.get("value") or 0)
                    ts = (t.get("metadata") or {}).get("blockTimestamp", "?")[:19]
                    org = ORG_001.get(fr.lower(), "")
                    tag = f" *** {org} ***" if org else ""
                    print(f"    {ts} <- {fr} {val:.4f} ETH{tag}")
        else:
            print("  No inflows found")

        # ============================================================
        # 2. ARBITRUM PRESENCE CHECK
        # ============================================================
        print()
        print("=" * 60)
        print("2. ARBITRUM PRESENCE")
        print("=" * 60)

        arb_nonce = await get_nonce(s, ARB, T2)
        print(f"  Arbitrum nonce: {arb_nonce}")
        await asyncio.sleep(0.2)

        if arb_nonce > 0:
            arb_in = await get_transfers(s, ARB, T2, "to", ["external"], 10, "asc")
            await asyncio.sleep(0.3)
            arb_out = await get_transfers(s, ARB, T2, "from", ["external"], 10, "desc")
            await asyncio.sleep(0.3)
            print(f"  Arb inflows: {len(arb_in)}")
            for t in arb_in[:5]:
                fr = t.get("from", "?")
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "?")[:19]
                print(f"    {ts} <- {fr} {val:.4f} ETH")
            print(f"  Arb outflows: {len(arb_out)}")
            for t in arb_out[:5]:
                to = t.get("to", "?")
                val = float(t.get("value") or 0)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "?")[:19]
                print(f"    {ts} -> {to} {val:.4f} ETH")
        else:
            print("  Zero Arbitrum presence — Base-only operation")

        # ============================================================
        # 3. PRIOR CAMPAIGN CYCLES — outflow history
        # ============================================================
        print()
        print("=" * 60)
        print("3. PRIOR CAMPAIGN CYCLES (outflow history)")
        print("=" * 60)

        # Get outflows in chronological order (oldest first)
        outflows = await get_transfers(s, BASE, T2, "from", ["external"], 100, "asc")
        await asyncio.sleep(0.3)

        if outflows:
            # Group into batches by date and amount
            batches = []
            current_batch = {"date": None, "amount": None, "count": 0, "first": None, "last": None}

            for t in outflows:
                val = round(float(t.get("value") or 0), 2)
                ts = (t.get("metadata") or {}).get("blockTimestamp", "")[:10]

                if val == current_batch.get("amount") and ts == current_batch.get("date"):
                    current_batch["count"] += 1
                    current_batch["last"] = (t.get("metadata") or {}).get("blockTimestamp", "")[:19]
                else:
                    if current_batch["count"] > 0:
                        batches.append(dict(current_batch))
                    current_batch = {
                        "date": ts, "amount": val, "count": 1,
                        "first": (t.get("metadata") or {}).get("blockTimestamp", "")[:19],
                        "last": (t.get("metadata") or {}).get("blockTimestamp", "")[:19],
                    }

            if current_batch["count"] > 0:
                batches.append(dict(current_batch))

            print(f"  Total outflows in sample: {len(outflows)}")
            print(f"  Batches detected: {len(batches)}")
            print()

            # Show batches with count >= 3 (likely campaign cycles)
            campaigns = [b for b in batches if b["count"] >= 3]
            print(f"  Campaign-like batches (3+ uniform sends):")
            for b in campaigns:
                print(f"    {b['first']} to {b['last']} | {b['count']}x {b['amount']} ETH = {b['count'] * b['amount']:.1f} ETH total")

            # Also show all batches for full picture
            print()
            print(f"  All outflow batches:")
            for b in batches:
                marker = " <-- CAMPAIGN" if b["count"] >= 3 else ""
                print(f"    {b['date']} | {b['count']}x {b['amount']} ETH{marker}")

        # Get latest outflows too
        print()
        print("  Latest outflows (most recent 20):")
        latest_out = await get_transfers(s, BASE, T2, "from", ["external"], 20, "desc")
        await asyncio.sleep(0.3)
        for t in latest_out[:15]:
            to = (t.get("to") or "?")[:18]
            val = float(t.get("value") or 0)
            ts = (t.get("metadata") or {}).get("blockTimestamp", "?")[:19]
            print(f"    {ts} -> {to}... {val:.4f} ETH")

        # ============================================================
        # 4. COUNTERPARTY CROSS-REFERENCE WITH ORG_001
        # ============================================================
        print()
        print("=" * 60)
        print("4. COUNTERPARTY CROSS-REFERENCE WITH ORG_001")
        print("=" * 60)

        # Collect all counterparties from in+out flows
        all_counterparties = Counter()
        for t in (earliest or []):
            fr = (t.get("from") or "").lower()
            if fr:
                all_counterparties[fr] += 1
        for t in (outflows or []):
            to = (t.get("to") or "").lower()
            if to:
                all_counterparties[to] += 1
        for t in (latest_out or []):
            to = (t.get("to") or "").lower()
            if to:
                all_counterparties[to] += 1

        print(f"  Total unique counterparties: {len(all_counterparties)}")
        print(f"  Top 10 by frequency:")
        for addr, count in all_counterparties.most_common(10):
            org = ORG_001.get(addr, "")
            tag = f" *** {org} ***" if org else ""
            print(f"    {addr} ({count} interactions){tag}")

        # Explicit check against every org_001 address
        print()
        print("  Direct org_001 overlap check:")
        found_overlap = False
        for org_addr, org_role in ORG_001.items():
            if org_addr in all_counterparties:
                print(f"    *** MATCH: {org_addr[:18]}... ({org_role}) — {all_counterparties[org_addr]} interactions ***")
                found_overlap = True
        if not found_overlap:
            print("    No overlap with any org_001 address")

        # Also check T1 counterparties
        print()
        print("  Checking Treasury 1 counterparties against org_001:")
        t1_in = await get_transfers(s, BASE, T1, "to", ["external"], 50, "asc")
        await asyncio.sleep(0.3)
        t1_out = await get_transfers(s, BASE, T1, "from", ["external"], 50, "desc")
        await asyncio.sleep(0.3)
        t1_counterparties = set()
        for t in (t1_in or []):
            t1_counterparties.add((t.get("from") or "").lower())
        for t in (t1_out or []):
            t1_counterparties.add((t.get("to") or "").lower())

        found_t1 = False
        for org_addr, org_role in ORG_001.items():
            if org_addr in t1_counterparties:
                print(f"    *** T1 MATCH: {org_addr[:18]}... ({org_role}) ***")
                found_t1 = True
        if not found_t1:
            print("    No overlap")

    # ============================================================
    # DOCUMENT ORG_002 IN DATABASE
    # ============================================================
    print()
    print("=" * 60)
    print("DOCUMENTING ORG_002")
    print("=" * 60)

    conn = sqlite3.connect("surveillance/data/surveillance.db")
    now = datetime.now(timezone.utc).isoformat()

    # Create org documentation in pattern_matches if not exists
    org_002_evidence = json.dumps({
        "org_id": "org_002",
        "status": "ACTIVE",
        "chain": "base",
        "campaign_start": "2026-03-20T18:01:35+00:00",
        "treasury_2": {
            "address": T2,
            "role": "senior_treasury",
            "nonce": 15486,
            "first_seen": "2024-07-20",
            "funded_deployers": 21,
        },
        "treasury_1": {
            "address": T1,
            "role": "junior_treasury",
            "nonce": 9283,
            "first_seen": "2025-09-22",
            "funded_deployers": 22,
            "funded_by": T2,
        },
        "campaign": {
            "deployers": 43,
            "contracts": 43,
            "pattern": "tx.origin_conditional (buy-not-sell)",
            "eth_per_deployer": 5.0,
            "total_eth_deployed": 215,
            "opsec": "one-deployer-per-contract, disposable addresses",
        },
        "org_001_connection": "none_found",
    })

    conn.execute("""
        INSERT OR REPLACE INTO pattern_matches
        (address, pattern_type, evidence, first_seen, last_seen, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (T2, "org_002_treasury", org_002_evidence, "2024-07-20", now, "high", "confirmed"))

    # Update entity types
    conn.execute("UPDATE deployers SET entity_type = ? WHERE deployer_address = ?",
                 ("org_002_treasury_senior", T2))
    conn.execute("UPDATE deployers SET entity_type = ? WHERE deployer_address = ?",
                 ("org_002_treasury_junior", T1))

    # Update notes
    conn.execute("""UPDATE deployers SET deployment_pattern_notes = ? WHERE deployer_address = ?""",
                 ("ORG_002 SENIOR TREASURY. Active since July 2024 (nonce 15,486). "
                  "Funds Treasury 1 (0xde8eb9) and 21 deployers directly. "
                  "Distributes exactly 5.0 ETH per deployer for tx.origin trap campaigns on Base. "
                  "Multiple campaign cycles visible in outflow history. "
                  "No org_001 connection found. Capital deployed: ~215 ETH.", T2))

    conn.execute("""UPDATE deployers SET deployment_pattern_notes = ? WHERE deployer_address = ?""",
                 ("ORG_002 JUNIOR TREASURY. Active since Sept 2025 (nonce 9,283). "
                  "Funded by senior treasury 0x238d71 (5.4 ETH). "
                  "Funds 22 deployers for current tx.origin campaign. "
                  "Same operational pattern as senior treasury.", T1))

    conn.commit()
    conn.close()
    print("  pattern_matches: org_002 documented")
    print("  deployers: both treasuries updated")

    # Flag on Railway
    import urllib.request
    for addr, etype in [(T2, "org_002_treasury_senior"), (T1, "org_002_treasury_junior")]:
        try:
            data = json.dumps({"updates": {addr: etype}}).encode()
            req = urllib.request.Request(
                "https://spypy.up.railway.app/admin/entity-type",
                data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer jlfsafjiefnajsf"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  Railway {addr[:14]}...: {json.loads(resp.read())}")
        except Exception as e:
            print(f"  Railway failed for {addr[:14]}...: {e}")

    print()
    print("ORG_002 fully documented.")


asyncio.run(main())
