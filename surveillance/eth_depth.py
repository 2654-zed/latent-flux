"""
Layer 3 — Ethereum Mainnet Depth Tracing

On-demand only. No active monitoring. Uses Ethereum mainnet to trace
funding sources, cross-chain activity, and upstream wallets for addresses
already identified on L2 (Arbitrum/Base).

Usage:
    # From admin endpoint: POST /admin/eth-trace
    # From CLI: python -m surveillance.eth_depth --address 0x...

Costs: 1 Alchemy CU per alchemy_getAssetTransfers call.
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

from surveillance import db

logger = logging.getLogger("surveillance.eth_depth")

# Ethereum mainnet known addresses
ETH_CEX_DEPOSITS = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60": "binance_hot_1",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance_hot_2",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "binance_hot_3",
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": "binance_hot_4",
    # Coinbase
    "0x503828976d22510aad0201ac7ec88293211d23da": "coinbase_hot_1",
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "coinbase_commerce",
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "coinbase_hot_2",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "okx_hot_1",
    "0x236f9f97e0e62388479bf9e5ba4889e46b0273c3": "okx_hot_2",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "kraken_hot_1",
    "0x53d284357ec70ce289d6d64134dfac8e511c8a3d": "kraken_hot_2",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "bybit_hot_1",
    # Gate.io
    "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "gate_hot_1",
    # KuCoin
    "0xd6216fc19db775df9774a6e33526131da7d19a2c": "kucoin_hot_1",
    "0xeb2629a2734e272bcc07bda959863f316f4bd4cf": "kucoin_hot_2",
}

ETH_BRIDGES = {
    "0x4dbd4fc535ac27206064b68ffcf827b0a60bab3f": "arbitrum_delayed_inbox",
    "0x72ce9c846789fdb6fc1f34ac4ad25dd9ef7031ef": "arbitrum_gateway_router",
    "0xa3a7b6f88361f48403514059f1f16c8e78d60eec": "arbitrum_erc20_gateway",
    "0x3154cf16ccdb4c6d922629664174b904d80f2c35": "base_portal",
    "0x49048044d57e1c92a77f79988d21fa8faf74e97e": "base_l1_bridge",
    "0x3ee18b2214aff97000d974cf647e7c347e8fa585": "wormhole_bridge",
    "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf": "polygon_bridge",
    "0x737901bea3eeb88459df9ef1be8ff3ae1b42a2ba": "zksync_era_diamond",
    "0x57891966931eb4bb6fb81430e6ce0a03aabde063": "stargate_stg_staking",
}

ETH_MIXERS = {
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b": "tornado_100eth",
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936": "tornado_10eth",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf": "tornado_router",
}

ETH_KNOWN_LABELS = {**ETH_CEX_DEPOSITS, **ETH_BRIDGES, **ETH_MIXERS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _alchemy_transfers(http_url: str, params: dict) -> list[dict]:
    """Call alchemy_getAssetTransfers via HTTP JSON-RPC."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [params],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(http_url, json=payload, timeout=30) as resp:
            data = await resp.json()
            return data.get("result", {}).get("transfers", [])


async def _rpc_call(http_url: str, method: str, params: list) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    async with aiohttp.ClientSession() as session:
        async with session.post(http_url, json=payload, timeout=30) as resp:
            return await resp.json()


# ---------------------------------------------------------------
# Core: trace an address on Ethereum mainnet
# ---------------------------------------------------------------

async def trace_eth_mainnet(http_url: str, address: str,
                            conn: sqlite3.Connection,
                            max_hops: int = 2) -> dict:
    """
    Trace an address on Ethereum mainnet. Finds:
    - Inbound funding (who sent ETH to this address on L1)
    - Outbound flows (where did they send ETH on L1)
    - Bridge interactions (did they bridge to/from L2)
    - CEX connections (exchange deposits/withdrawals)

    Costs: 2 + (2 * max_hops) Alchemy CU max.
    """
    address = address.lower()
    result = {
        "address": address,
        "chain": "ethereum",
        "traced_at": _now(),
        "inbound": [],
        "outbound": [],
        "bridge_activity": [],
        "cex_connections": [],
        "funding_hops": [],
        "labels_found": [],
    }

    # 1. Inbound ETH transfers (first funding)
    inbound = await _alchemy_transfers(http_url, {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "toAddress": address,
        "category": ["external", "internal"],
        "maxCount": "0x14",  # 20 transfers
        "order": "asc",
        "withMetadata": True,
    })

    for t in inbound:
        sender = t.get("from", "").lower()
        entry = {
            "from": sender,
            "value_eth": float(t.get("value", 0)),
            "tx_hash": t.get("hash"),
            "block": int(t.get("blockNum", "0x0"), 16) if t.get("blockNum") else None,
            "timestamp": t.get("metadata", {}).get("blockTimestamp", ""),
            "label": ETH_KNOWN_LABELS.get(sender, None),
        }
        result["inbound"].append(entry)
        if entry["label"]:
            result["labels_found"].append({"address": sender, "label": entry["label"], "direction": "inbound"})

    # 2. Outbound ETH transfers
    outbound = await _alchemy_transfers(http_url, {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "fromAddress": address,
        "category": ["external", "internal"],
        "maxCount": "0x14",
        "order": "desc",
        "withMetadata": True,
    })

    for t in outbound:
        recipient = t.get("to", "").lower()
        entry = {
            "to": recipient,
            "value_eth": float(t.get("value", 0)),
            "tx_hash": t.get("hash"),
            "block": int(t.get("blockNum", "0x0"), 16) if t.get("blockNum") else None,
            "timestamp": t.get("metadata", {}).get("blockTimestamp", ""),
            "label": ETH_KNOWN_LABELS.get(recipient, None),
        }
        result["outbound"].append(entry)
        if entry["label"]:
            result["labels_found"].append({"address": recipient, "label": entry["label"], "direction": "outbound"})

        # Flag bridge interactions
        if recipient in ETH_BRIDGES:
            result["bridge_activity"].append({
                "bridge": ETH_BRIDGES[recipient],
                "value_eth": entry["value_eth"],
                "tx_hash": entry["tx_hash"],
                "timestamp": entry["timestamp"],
                "direction": "l1_to_l2",
            })

        # Flag CEX connections
        if recipient in ETH_CEX_DEPOSITS:
            result["cex_connections"].append({
                "exchange": ETH_CEX_DEPOSITS[recipient],
                "value_eth": entry["value_eth"],
                "tx_hash": entry["tx_hash"],
                "timestamp": entry["timestamp"],
                "direction": "deposit",
            })

    # Also check inbound for CEX withdrawals
    for entry in result["inbound"]:
        sender = entry["from"]
        if sender in ETH_CEX_DEPOSITS:
            result["cex_connections"].append({
                "exchange": ETH_CEX_DEPOSITS[sender],
                "value_eth": entry["value_eth"],
                "tx_hash": entry["tx_hash"],
                "timestamp": entry["timestamp"],
                "direction": "withdrawal",
            })

    # 3. Multi-hop funding trace (upstream)
    current = address
    for hop in range(1, max_hops + 1):
        transfers = await _alchemy_transfers(http_url, {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "toAddress": current,
            "category": ["external", "internal"],
            "maxCount": "0x3",
            "order": "asc",
            "withMetadata": True,
        })

        source = None
        for t in transfers:
            if float(t.get("value", 0)) > 0:
                source = t
                break
        if not source:
            break

        hop_data = {
            "hop": hop,
            "from": source["from"].lower(),
            "to": current,
            "value_eth": float(source["value"]),
            "tx_hash": source.get("hash"),
            "timestamp": source.get("metadata", {}).get("blockTimestamp", ""),
            "label": ETH_KNOWN_LABELS.get(source["from"].lower()),
        }
        result["funding_hops"].append(hop_data)
        current = source["from"].lower()

        # Stop if we hit a labeled address (CEX, bridge, etc.)
        if hop_data["label"]:
            break

    # 4. Store in DB
    conn.execute("""
        INSERT OR REPLACE INTO eth_traces
        (address, traced_at, inbound_count, outbound_count,
         bridge_interactions, cex_connections, funding_hops,
         labels_found, raw_result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        address, _now(),
        len(result["inbound"]), len(result["outbound"]),
        json.dumps(result["bridge_activity"]),
        json.dumps(result["cex_connections"]),
        json.dumps(result["funding_hops"]),
        json.dumps(result["labels_found"]),
        json.dumps(result),
    ))
    conn.commit()

    return result


async def trace_l2_address_on_eth(http_url: str, address: str,
                                  conn: sqlite3.Connection) -> dict:
    """
    Check if an L2 address has any Ethereum mainnet history.
    Lightweight — just checks for existence (1 API call).
    """
    address = address.lower()
    transfers = await _alchemy_transfers(http_url, {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "fromAddress": address,
        "category": ["external"],
        "maxCount": "0x1",
        "order": "desc",
        "withMetadata": True,
    })

    has_mainnet = len(transfers) > 0
    last_seen = transfers[0].get("metadata", {}).get("blockTimestamp") if transfers else None

    return {
        "address": address,
        "has_mainnet_activity": has_mainnet,
        "last_mainnet_tx": last_seen,
    }


async def batch_check_mainnet_presence(http_url: str, addresses: list[str],
                                       conn: sqlite3.Connection) -> dict:
    """
    Check multiple addresses for mainnet presence. 1 CU per address.
    Use sparingly — for priority targets only.
    """
    results = {}
    for addr in addresses:
        try:
            r = await trace_l2_address_on_eth(http_url, addr, conn)
            results[addr] = r
        except Exception as e:
            results[addr] = {"error": str(e)}
    return results


# ---------------------------------------------------------------
# DB table creation
# ---------------------------------------------------------------

def ensure_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eth_traces (
            address TEXT PRIMARY KEY,
            traced_at TEXT NOT NULL,
            inbound_count INTEGER DEFAULT 0,
            outbound_count INTEGER DEFAULT 0,
            bridge_interactions TEXT,
            cex_connections TEXT,
            funding_hops TEXT,
            labels_found TEXT,
            raw_result TEXT
        )
    """)
    conn.commit()


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ethereum mainnet depth trace")
    parser.add_argument("--address", required=True, help="Address to trace on Ethereum")
    parser.add_argument("--hops", type=int, default=2, help="Max funding hops (default 2)")
    parser.add_argument("--check-only", action="store_true", help="Just check mainnet presence")
    args = parser.parse_args()

    # Load env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    eth_url = os.environ.get("ETH_HTTP_URL", "")
    if not eth_url:
        eth_wss = os.environ.get("ETH_WSS_URL", "")
        eth_url = eth_wss.replace("wss://", "https://") if eth_wss else ""
    if not eth_url:
        print("ERROR: Set ETH_HTTP_URL or ETH_WSS_URL")
        exit(1)

    db_path = Path(__file__).resolve().parent / "data" / "surveillance.db"
    conn = sqlite3.connect(str(db_path))
    ensure_tables(conn)

    async def _run():
        if args.check_only:
            return await trace_l2_address_on_eth(eth_url, args.address, conn)
        return await trace_eth_mainnet(eth_url, args.address, conn, max_hops=args.hops)

    result = asyncio.run(_run())
    print(json.dumps(result, indent=2))
    conn.close()
