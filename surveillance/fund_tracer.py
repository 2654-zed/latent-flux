"""
Post-Extraction Fund Flow Tracer for Layer 3 Surveillance.

Traces where money goes AFTER extraction from org exit ramps.
Works from two data sources:
1. extraction_events.raw_transactions (manually documented flows)
2. Real-time monitoring via org_transfer_events table (filled by event_monitors.py)

Usage:
    python -m surveillance.fund_tracer --report
    python -m surveillance.fund_tracer --wallet 0x...
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
CASES_DIR = Path(__file__).resolve().parent / "data" / "cases"

# ──────────────────────────────────────────────────────
# KNOWN ADDRESS REGISTRY
# ──────────────────────────────────────────────────────

CEX_HOT_WALLETS = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance",
    "0xf977814e90da44bfa03b6295a0616a897441acec": "Binance",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": "Coinbase",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "Bybit",
}

DEX_ROUTERS = {
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap V3",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "Uniswap V3 Router2",
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "Uniswap Universal",
    "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24": "PancakeSwap V3",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch V5",
    "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": "SushiSwap V2",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "SushiSwap",
}

MIXERS = {
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b": "Tornado Cash",
    "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9": "Railgun",
}

DEX_POOLS = {
    "0x0e4831319a50228b9e450861297ab92dee15b44f": "WBTC/USDC Pool (org_001 exit swap)",
    "0x5969efdde3cf5c0d9a88ae51e47d721096a97203": "WBTC/USDT Pool (org_001 exit swap)",
    "0x5a17cbf5f866bde11c28861a2742764fac0eba4b": "USDC/WBTC Pool (org_001 laundry swap)",
    "0x6985cb98ce393fce8d6272127f39013f61e36166": "WBTC/USDC Pool (org_001 exit swap)",
    "0x7fcdc35463e3770c2fb992716cd070b63540b947": "USDC/WETH Pool (org_001 defi router swap)",
}

BRIDGE_CONTRACTS = {
    "0x0000000000000000000000000000000000000064": "Arbitrum ArbSys",
    "0x4200000000000000000000000000000000000010": "Base L2StandardBridge",
    "0x53bf833a5d6c4dda888f69c22c88c9f356a41614": "Stargate (Arb)",
    "0x27a16dc786820b16e5c9028b75b99f6f604b5d26": "Stargate (Base)",
}

# Org wallets — for INTERNAL classification
ORG_WALLETS = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": "org_001:treasury",
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "org_001:operator",
    "0xfd51e33d44b376ef346d24a130a51035db09c1dc": "org_001:operator_2",
    "0xc6962004f452be9203591991d15f6b388e09e8d0": "org_001:cashout",
    "0x01989c93890aed05a63d179b03424997075b6acf": "org_001:exit_cex",
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": "org_001:gas_station",
    "0x360e68faccca8ca495c1b759fd9eee466db9fb32": "org_001:treasury_branch",
    "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "org_001:laundry",
    "0x96daa0b8a5499ea9323421ed0cda06b345caab73": "org_001:lp_staging",
    "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70": "org_001:lp_companion",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "org_001:defi_exit_channel",
    "0x238d7170f309a55b87a144a341bd6105897082ca": "org_002:treasury_senior",
    "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473": "org_002:treasury_junior",
}


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def classify_destination(address):
    """Classify a destination address."""
    addr = address.lower().strip()
    if addr in ORG_WALLETS:
        return {"type": "INTERNAL", "label": ORG_WALLETS[addr]}
    if addr in CEX_HOT_WALLETS:
        return {"type": "CEX_DEPOSIT", "label": CEX_HOT_WALLETS[addr]}
    if addr in DEX_ROUTERS:
        return {"type": "DEX_SWAP", "label": DEX_ROUTERS[addr]}
    if addr in BRIDGE_CONTRACTS:
        return {"type": "BRIDGE", "label": BRIDGE_CONTRACTS[addr]}
    if addr in MIXERS:
        return {"type": "MIXER", "label": MIXERS[addr]}
    if addr in DEX_POOLS:
        return {"type": "DEX_POOL", "label": DEX_POOLS[addr]}
    return {"type": "UNKNOWN", "label": addr[:18] + "..."}


def extract_flows_from_extraction_events(conn):
    """Parse the manually documented extraction_events for fund flows."""
    flows = []
    for r in conn.execute("SELECT event_id, observed_at, total_usd_moved, raw_transactions FROM extraction_events"):
        try:
            raw = json.loads(r["raw_transactions"]) if r["raw_transactions"] else {}
        except (json.JSONDecodeError, TypeError):
            continue

        event_id = r["event_id"]
        ts = r["observed_at"]

        for flow_type, transfers in raw.items():
            if not isinstance(transfers, list):
                continue
            for tx in transfers:
                if not isinstance(tx, dict):
                    continue
                to_addr = tx.get("to", "").lower().rstrip(".")
                # Handle label references
                label_map = {"cashout": "0xc6962004f452be9203591991d15f6b388e09e8d0"}
                if to_addr in label_map:
                    to_addr = label_map[to_addr]
                # Normalize short/truncated addresses against all known registries
                if len(to_addr) < 42:
                    prefix = to_addr.replace("...", "").replace(" ", "")
                    for registry in [ORG_WALLETS, CEX_HOT_WALLETS, DEX_ROUTERS, DEX_POOLS, MIXERS, BRIDGE_CONTRACTS]:
                        for full_addr in registry:
                            if full_addr.startswith(prefix):
                                to_addr = full_addr
                                break
                        if len(to_addr) == 42:
                            break

                amount = tx.get("amount") or tx.get("weth") or tx.get("wbtc") or tx.get("amount_weth") or tx.get("amount_wbtc") or 0
                token = "USDC" if "usdc" in flow_type.lower() or tx.get("asset") == "USDC" else \
                        "WETH" if "weth" in flow_type.lower() or tx.get("weth") else \
                        "WBTC" if "wbtc" in flow_type.lower() or tx.get("wbtc") else \
                        tx.get("asset", "ETH")

                # Determine source from flow_type
                if "cashout_to" in flow_type:
                    from_addr = "0xc6962004f452be9203591991d15f6b388e09e8d0"
                    from_role = "cashout"
                elif "lp_staging" in flow_type and "to" not in flow_type:
                    from_addr = "0x96daa0b8a5499ea9323421ed0cda06b345caab73"
                    from_role = "lp_staging"
                elif "exit_ramp" in flow_type:
                    from_addr = "0x01989c93890aed05a63d179b03424997075b6acf"
                    from_role = "exit_cex"
                elif "treasury_branch" in flow_type:
                    from_addr = "0x360e68faccca8ca495c1b759fd9eee466db9fb32"
                    from_role = "treasury_branch"
                elif "laundry" in flow_type:
                    from_addr = "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb"
                    from_role = "laundry"
                elif "defi_router" in flow_type:
                    from_addr = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"
                    from_role = "defi_exit_channel"
                else:
                    from_addr = "unknown"
                    from_role = flow_type

                dest = classify_destination(to_addr)

                flows.append({
                    "event_id": event_id,
                    "org_id": "org_001",
                    "from_address": from_addr,
                    "from_role": from_role,
                    "to_address": to_addr,
                    "destination_type": dest["type"],
                    "destination_label": dest["label"],
                    "amount": amount,
                    "token": token,
                    "timestamp": tx.get("ts", ts),
                    "note": tx.get("note", ""),
                })

    return flows


def extract_flows_from_org_transfers(conn):
    """Pull from org_transfer_events table (real-time captured data)."""
    flows = []
    try:
        for r in conn.execute("""
            SELECT from_address, to_address, value_eth, token, timestamp, tx_hash, org_id
            FROM org_transfer_events ORDER BY timestamp
        """):
            dest = classify_destination(r["to_address"])
            from_role = ORG_WALLETS.get(r["from_address"].lower(), "unknown")
            flows.append({
                "event_id": "realtime",
                "org_id": r["org_id"] or "unknown",
                "from_address": r["from_address"],
                "from_role": from_role,
                "to_address": r["to_address"],
                "destination_type": dest["type"],
                "destination_label": dest["label"],
                "amount": r["value_eth"],
                "token": r["token"] or "ETH",
                "timestamp": r["timestamp"],
            })
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet
    return flows


def generate_flow_report(org_id=None):
    """Generate the fund flow analysis report."""
    conn = get_connection()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Collect flows from all sources
    flows = extract_flows_from_extraction_events(conn)
    flows += extract_flows_from_org_transfers(conn)

    if org_id:
        flows = [f for f in flows if f["org_id"] == org_id]

    # Aggregate
    by_type = defaultdict(list)
    by_org = defaultdict(list)
    for f in flows:
        by_type[f["destination_type"]].append(f)
        by_org[f["org_id"]].append(f)

    type_implications = {
        "CEX_DEPOSIT": "Cashout to fiat — identifiable with exchange cooperation",
        "DEX_SWAP": "Token conversion via router — laundering through legitimate DeFi",
        "DEX_POOL": "Direct LP pool swap — WBTC/USDC/USDT conversion, fully traceable",
        "BRIDGE": "Cross-chain movement — expanding the trace surface",
        "MIXER": "Deliberate obfuscation — trail goes cold here",
        "INTERNAL": "Org-internal transfer — funds still in the operation",
        "UNKNOWN": "Unclassified destination — needs further investigation",
    }

    md = f"""# POST-EXTRACTION FUND FLOW ANALYSIS
**Generated:** {now}
**Total flows traced:** {len(flows)}
**Data sources:** extraction_events ({sum(1 for f in flows if f['event_id'] != 'realtime')} manual), org_transfer_events ({sum(1 for f in flows if f['event_id'] == 'realtime')} realtime)
**Organizations:** {', '.join(sorted(by_org.keys())) if by_org else 'None'}

---

## Destination Breakdown

| Destination Type | Count | % | Implication |
|---|---|---|---|
"""
    for dtype in ["CEX_DEPOSIT", "MIXER", "DEX_SWAP", "DEX_POOL", "BRIDGE", "INTERNAL", "UNKNOWN"]:
        type_flows = by_type.get(dtype, [])
        if not type_flows:
            continue
        pct = round(len(type_flows) / len(flows) * 100, 1) if flows else 0
        impl = type_implications.get(dtype, "")
        md += f"| {dtype} | {len(type_flows)} | {pct}% | {impl} |\n"

    # Per-org detail
    for oid, org_flows in sorted(by_org.items()):
        md += f"\n---\n\n## {oid.upper()}: Fund Flow Detail\n\n"

        # Group by from_role -> destination
        role_flows = defaultdict(lambda: defaultdict(list))
        for f in org_flows:
            role_flows[f["from_role"]][f["destination_type"]].append(f)

        for role, dests in sorted(role_flows.items()):
            md += f"### From: {role}\n"
            md += "| Destination | Type | Count | Tokens | Sample Amount | Notes |\n"
            md += "|---|---|---|---|---|---|\n"
            for dtype, dest_flows in sorted(dests.items()):
                tokens = set(f["token"] for f in dest_flows)
                labels = set(f["destination_label"] for f in dest_flows if f["destination_label"])
                sample_amt = dest_flows[0].get("amount", "N/A")
                notes = set(f.get("note", "") for f in dest_flows if f.get("note"))
                label_str = ", ".join(list(labels)[:3])
                md += f"| {label_str} | {dtype} | {len(dest_flows)} | {', '.join(tokens)} | {sample_amt} | {'; '.join(notes) if notes else ''} |\n"
            md += "\n"

    # Timeline
    md += "---\n\n## Transfer Timeline\n\n"
    md += "| Time | From (Role) | To | Type | Amount | Token |\n"
    md += "|---|---|---|---|---|---|\n"
    for f in sorted(flows, key=lambda x: str(x.get("timestamp", "")))[:50]:
        ts = str(f.get("timestamp", "N/A"))[:19]
        from_short = f["from_address"][:12] + "..."
        to_short = f["to_address"][:12] + "..." if len(f["to_address"]) > 14 else f["to_address"]
        md += f"| {ts} | {from_short} ({f['from_role']}) | {to_short} ({f['destination_label']}) | {f['destination_type']} | {f.get('amount', 'N/A')} | {f['token']} |\n"

    # Kill chain
    md += "\n---\n\n## Updated Kill Chain\n\n```\n"
    md += "DEPLOYMENT -> TRAP -> VICTIM -> EXTRACTION -> EXIT RAMP\n"
    md += "                                                 |\n"
    for dtype in ["INTERNAL", "CEX_DEPOSIT", "DEX_SWAP", "BRIDGE", "MIXER", "UNKNOWN"]:
        if by_type.get(dtype):
            count = len(by_type[dtype])
            labels = set(f["destination_label"][:20] for f in by_type[dtype])
            md += f"                                                 +-> {dtype} ({count}x) [{', '.join(list(labels)[:3])}]\n"
    md += "```\n"

    # Data gap notice
    md += "\n---\n\n## Data Gaps & Recommendations\n\n"
    md += "**Current limitation:** The surveillance system tracks contract interactions (caller->contract) "
    md += "but does NOT capture wallet-to-wallet transfers. The flows in this report come from:\n"
    md += "- Manually documented extraction_events (2 events with detailed JSON)\n"
    md += "- Future: org_transfer_events table (filled by event_monitors.py org wallet hook)\n\n"
    md += "**To close this gap:** Add an org wallet outbound transfer monitor to event_monitors.py "
    md += "that captures all outbound ETH and ERC-20 transfers from addresses in the ORG_WALLETS registry.\n"

    conn.close()

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = CASES_DIR / f"FUND_FLOW_TRACE_{ts}.md"
    filepath.write_text(md, encoding="utf-8")
    print(f"Fund flow report: {filepath}")
    print(f"  Flows traced: {len(flows)}")
    print(f"  Destinations: {dict(Counter(f['destination_type'] for f in flows))}" if flows else "  No flows")
    return filepath


# Need Counter import
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace post-extraction fund flows")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--org", default=None)
    args = parser.parse_args()
    generate_flow_report(org_id=args.org)
