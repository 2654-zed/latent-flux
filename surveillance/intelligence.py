"""
Layer 3 -- Phase 2 Intelligence Module

All Alchemy-powered analysis functions. No Arbiscan dependency.
Each function is independent and callable from admin endpoints.

Requires: Alchemy paid plan (archive access, alchemy_getAssetTransfers)
Optional: debug_traceTransaction (Growth plan+)
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from surveillance import db

logger = logging.getLogger("surveillance.intelligence")

# Known exchange deposit addresses (Arbitrum)
KNOWN_EXCHANGES = {
    "0xb38e8c17e38363af6ebdcb3dae12e0243582891d": "binance",
    "0xa7efae728d2936e78bda97dc267687568dd593f3": "okx",
    "0x1ab4973a48dc892cd9971ece8e01dcc7688f8f23": "gate_io",
}

USDC_ARBITRUM = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_E_ARBITRUM = "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"

EIP1167_PREFIX = "363d3d373d3d3d363d73"
EIP1167_SUFFIX = "5af43d82803e903d91602b57fd5bf3"


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
    """Generic JSON-RPC call."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    async with aiohttp.ClientSession() as session:
        async with session.post(http_url, json=payload, timeout=30) as resp:
            return await resp.json()


# ---------------------------------------------------------------
# Contract verification (code presence + proxy detection)
# ---------------------------------------------------------------

async def check_contract_verification(http_url: str, address: str,
                                      conn: sqlite3.Connection) -> dict:
    """Check if a contract has code and detect proxy patterns."""
    result = await _rpc_call(http_url, "eth_getCode", [address, "latest"])
    code = result.get("result", "0x")
    code_hex = code[2:] if code.startswith("0x") else code

    has_code = len(code_hex) > 0
    code_size = len(code_hex) // 2
    is_proxy = EIP1167_PREFIX in code_hex and EIP1167_SUFFIX in code_hex

    conn.execute(
        """INSERT INTO contract_verification (contract_address, has_code, code_size, is_proxy, checked_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(contract_address) DO UPDATE SET
               has_code = ?, code_size = ?, is_proxy = ?, checked_at = ?""",
        (address, int(has_code), code_size, int(is_proxy), _now(),
         int(has_code), code_size, int(is_proxy), _now()),
    )
    conn.commit()

    return {
        "address": address,
        "has_code": has_code,
        "code_size": code_size,
        "is_proxy": is_proxy,
    }


async def check_all_suspected(http_url: str, conn: sqlite3.Connection) -> dict:
    """Check verification for all SUSPECTED contracts not yet checked."""
    unchecked = conn.execute(
        """SELECT c.contract_address FROM contracts c
           LEFT JOIN contract_verification v ON c.contract_address = v.contract_address
           WHERE c.confidence_tier IN ('suspected', 'confirmed')
             AND v.contract_address IS NULL"""
    ).fetchall()

    results = {"checked": 0, "has_code": 0, "proxy": 0, "empty": 0}
    for row in unchecked:
        try:
            r = await check_contract_verification(http_url, row[0], conn)
            results["checked"] += 1
            if r["has_code"]:
                results["has_code"] += 1
            else:
                results["empty"] += 1
            if r["is_proxy"]:
                results["proxy"] += 1
        except Exception as e:
            logger.warning("Verification check failed for %s: %s", row[0], e)

    return results


# ---------------------------------------------------------------
# Multi-hop funding trace
# ---------------------------------------------------------------

async def trace_funding_hops(http_url: str, deployer: str,
                             conn: sqlite3.Connection,
                             max_hops: int = 2) -> list[dict]:
    """Trace funding chain: who funded the deployer, who funded the funder."""
    hops = []
    current = deployer

    for hop in range(1, max_hops + 1):
        transfers = await _alchemy_transfers(http_url, {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "toAddress": current,
            "category": ["external", "internal"],
            "maxCount": "0x5",
            "order": "asc",
            "withMetadata": True,
        })

        # Find first inbound with value
        source = None
        for t in transfers:
            if float(t.get("value", 0)) > 0:
                source = t
                break

        if not source:
            break

        hop_data = {
            "deployer_address": deployer,
            "hop_number": hop,
            "source_address": source["from"].lower(),
            "transfer_type": source.get("category", "external"),
            "value_eth": float(source["value"]),
            "tx_hash": source.get("hash"),
            "block_number": int(source.get("blockNum", "0x0"), 16) if source.get("blockNum") else None,
            "timestamp": source.get("metadata", {}).get("blockTimestamp", _now()),
            "is_exchange": KNOWN_EXCHANGES.get(source["from"].lower()),
        }

        conn.execute(
            """INSERT OR REPLACE INTO funding_hops
               (deployer_address, hop_number, source_address, transfer_type,
                value_eth, tx_hash, block_number, timestamp, is_exchange)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (hop_data["deployer_address"], hop_data["hop_number"],
             hop_data["source_address"], hop_data["transfer_type"],
             hop_data["value_eth"], hop_data["tx_hash"],
             hop_data["block_number"], hop_data["timestamp"],
             hop_data["is_exchange"]),
        )
        conn.commit()

        hops.append(hop_data)
        current = source["from"]  # trace next hop

    return hops


# ---------------------------------------------------------------
# USDC withdrawal destination trace
# ---------------------------------------------------------------

async def trace_usdc_withdrawals(http_url: str, address: str,
                                 conn: sqlite3.Connection) -> list[dict]:
    """Find all USDC outflows from an address (deployer or contract)."""
    results = []

    for token in [USDC_ARBITRUM, USDC_E_ARBITRUM]:
        transfers = await _alchemy_transfers(http_url, {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "fromAddress": address,
            "category": ["erc20"],
            "contractAddresses": [token],
            "maxCount": "0x20",
            "order": "desc",
            "withMetadata": True,
        })

        for t in transfers:
            results.append({
                "from": t["from"].lower(),
                "to": t["to"].lower(),
                "value": float(t.get("value", 0)),
                "token": token,
                "tx_hash": t.get("hash"),
                "block": int(t.get("blockNum", "0x0"), 16) if t.get("blockNum") else None,
                "timestamp": t.get("metadata", {}).get("blockTimestamp", ""),
                "is_exchange": KNOWN_EXCHANGES.get(t["to"].lower()),
            })

    return results


# ---------------------------------------------------------------
# Execution traces
# ---------------------------------------------------------------

async def trace_transaction(http_url: str, tx_hash: str,
                            conn: sqlite3.Connection) -> dict:
    """Run debug_traceTransaction with callTracer on a specific tx."""
    result = await _rpc_call(http_url, "debug_traceTransaction", [
        tx_hash, {"tracer": "callTracer"}
    ])

    if "error" in result:
        return {"error": result["error"].get("message", str(result["error"]))}

    trace = result.get("result", {})

    # Build human-readable summary
    summary_parts = []
    _walk_trace(trace, summary_parts, depth=0)
    summary = "\n".join(summary_parts[:20])  # first 20 lines

    conn.execute(
        """INSERT OR REPLACE INTO traces
           (tx_hash, block_number, from_address, to_address, trace_json, summary, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tx_hash, None, trace.get("from"), trace.get("to"),
         json.dumps(trace), summary, _now()),
    )
    conn.commit()

    return {"tx_hash": tx_hash, "summary": summary, "calls": len(summary_parts)}


def _walk_trace(node: dict, parts: list, depth: int) -> None:
    """Recursively walk a callTracer output, building summary lines."""
    indent = "  " * depth
    call_type = node.get("type", "CALL")
    to_addr = node.get("to", "???")[:18]
    value = node.get("value", "0x0")
    if value and value != "0x0":
        value_eth = int(value, 16) / 1e18 if isinstance(value, str) else value
        parts.append(f"{indent}{call_type} -> {to_addr}... value={value_eth:.6f} ETH")
    else:
        input_data = node.get("input", "")
        selector = input_data[2:10] if len(input_data) >= 10 else "?"
        parts.append(f"{indent}{call_type} -> {to_addr}... selector={selector}")

    if node.get("error"):
        parts.append(f"{indent}  REVERTED: {node['error']}")

    for call in node.get("calls", []):
        _walk_trace(call, parts, depth + 1)


# ---------------------------------------------------------------
# Retroactive bot history
# ---------------------------------------------------------------

async def scan_bot_history(http_url: str, address: str,
                           conn: sqlite3.Connection) -> dict:
    """Get historical activity summary for a bot candidate."""
    # Outbound transfers
    transfers = await _alchemy_transfers(http_url, {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "fromAddress": address,
        "category": ["external"],
        "maxCount": "0x5",
        "order": "asc",
        "withMetadata": True,
    })

    # Inbound (funding)
    inbound = await _alchemy_transfers(http_url, {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "toAddress": address,
        "category": ["external"],
        "maxCount": "0x5",
        "order": "asc",
        "withMetadata": True,
    })

    first_activity = None
    funding_source = None
    total_eth_moved = 0.0

    if transfers:
        first_activity = transfers[0].get("metadata", {}).get("blockTimestamp")
        total_eth_moved = sum(float(t.get("value", 0)) for t in transfers)

    if inbound:
        for t in inbound:
            if float(t.get("value", 0)) > 0:
                funding_source = t["from"].lower()
                break

    return {
        "address": address,
        "first_activity": first_activity,
        "funding_source": funding_source,
        "total_eth_moved": total_eth_moved,
        "outbound_tx_count": len(transfers),
        "inbound_tx_count": len(inbound),
    }
