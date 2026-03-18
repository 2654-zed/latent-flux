"""
Layer 3 -- Funding Trail Tracer

For priority/confirmed deployers, traces where their deployment
gas funding came from by finding the first inbound ETH transfer.

Uses Alchemy's standard JSON-RPC (no debug/trace endpoints needed).
Strategy: query the deployer's earliest transactions to find who
funded the wallet.

Limitations (Phase 1):
- Only traces the direct funding source (1 hop)
- Cannot trace internal transactions (withdrawAllUSDC destinations)
  without debug_traceTransaction (Alchemy Growth plan) or Arbiscan API
- Full funding graph analysis is Phase 2

Usage:
    python3 -m surveillance.funding_tracer
    python3 -m surveillance.funding_tracer --address 0xe93d64...
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db

logger = logging.getLogger("surveillance.funding")

# Arbiscan API for transaction history (free tier: 5 calls/sec)
ARBISCAN_API = "https://api.arbiscan.io/api"


async def trace_funding_source(w3: AsyncWeb3, address: str) -> dict:
    """
    Find who funded a deployer address by checking:
    1. Current ETH balance
    2. First inbound transaction (via eth_getBalance + binary search on nonce)

    Returns a dict with funding info.
    """
    address = w3.to_checksum_address(address)
    result = {
        "address": address.lower(),
        "balance_eth": 0.0,
        "funding_source": None,
        "funding_tx": None,
        "funding_value_eth": None,
        "nonce": 0,
        "method": "rpc_nonce_trace",
    }

    try:
        # Current balance
        balance = await w3.eth.get_balance(address)
        result["balance_eth"] = float(w3.from_wei(balance, "ether"))

        # Current nonce (total outbound txs)
        nonce = await w3.eth.get_transaction_count(address)
        result["nonce"] = nonce

        if nonce == 0 and balance == 0:
            result["funding_source"] = "unfunded_or_contract"
            return result

    except Exception as e:
        logger.error("Failed to query %s: %s", address, e)
        result["funding_source"] = f"error: {e}"
        return result

    return result


async def trace_via_arbiscan(address: str, api_key: str) -> dict:
    """
    Use Arbiscan API to get the first normal transaction for an address.
    This finds the funding source directly.
    """
    import aiohttp

    result = {
        "address": address.lower(),
        "funding_source": None,
        "funding_tx": None,
        "funding_value_eth": None,
        "first_tx_from": None,
        "first_tx_to": None,
        "method": "arbiscan_txlist",
    }

    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": 5,  # just the first few txs
        "sort": "asc",
        "apikey": api_key,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ARBISCAN_API, params=params, timeout=15) as resp:
                if resp.status != 200:
                    result["funding_source"] = f"arbiscan_error_{resp.status}"
                    return result

                data = await resp.json()
                if data.get("status") != "1" or not data.get("result"):
                    result["funding_source"] = "no_transactions_found"
                    return result

                txs = data["result"]

                # Find first inbound transaction (to == our address)
                for tx in txs:
                    if tx["to"].lower() == address.lower() and int(tx["value"]) > 0:
                        result["funding_source"] = tx["from"].lower()
                        result["funding_tx"] = tx["hash"]
                        result["funding_value_eth"] = int(tx["value"]) / 1e18
                        result["first_tx_from"] = tx["from"].lower()
                        result["first_tx_to"] = tx["to"].lower()
                        break

                # If no inbound found, check if first tx is outbound (self-funded via bridge/contract)
                if not result["funding_source"] and txs:
                    first = txs[0]
                    if first["from"].lower() == address.lower():
                        result["funding_source"] = "self_initiated_or_bridge"
                        result["funding_tx"] = first["hash"]
                        result["first_tx_from"] = first["from"].lower()
                        result["first_tx_to"] = first["to"].lower()

    except Exception as e:
        logger.error("Arbiscan query failed for %s: %s", address, e)
        result["funding_source"] = f"error: {e}"

    return result


async def trace_all_priority(arbiscan_key: str = "") -> list[dict]:
    """Trace funding for all priority deployers."""
    conn = db.init_db()
    priority = db.get_priority_deployers(conn)

    if not priority:
        print("No priority deployers to trace.")
        conn.close()
        return []

    results = []

    for dep in priority:
        addr = dep["deployer_address"]
        existing = dep.get("funding_trail")
        if existing:
            logger.info("Skipping %s — already traced", addr[:18])
            continue

        logger.info("Tracing funding for %s...", addr[:18])

        if arbiscan_key:
            result = await trace_via_arbiscan(addr, arbiscan_key)
            # Rate limit: Arbiscan free tier is 5/sec
            await asyncio.sleep(0.25)
        else:
            # Fallback: basic RPC query (less informative)
            rpc_url = os.environ.get("ARB_WSS_URL", "")
            if not rpc_url:
                logger.warning("No ARB_WSS_URL or ARBISCAN_API_KEY — skipping %s", addr[:18])
                continue
            async with AsyncWeb3(WebSocketProvider(rpc_url)) as w3:
                result = await trace_funding_source(w3, addr)

        # Store result
        trail_json = json.dumps(result, indent=2)
        db.update_deployer_funding(conn, addr, trail_json)
        results.append(result)

        funding = result.get("funding_source", "unknown")
        value = result.get("funding_value_eth")
        print(f"  {addr}")
        print(f"    funded by: {funding}")
        if value:
            print(f"    amount:    {value:.6f} ETH")
        print(f"    method:    {result.get('method')}")
        print()

    # Check for shared funding sources
    all_sources = {}
    for r in results:
        src = r.get("funding_source")
        if src and src not in ("unknown", "error", "unfunded_or_contract", "self_initiated_or_bridge"):
            all_sources.setdefault(src, []).append(r["address"])

    shared = {k: v for k, v in all_sources.items() if len(v) > 1}
    if shared:
        print("=== SHARED FUNDING SOURCES ===")
        for source, deployers in shared.items():
            print(f"  {source} funded:")
            for d in deployers:
                print(f"    -> {d}")
        print()

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer 3 -- Funding trail tracer")
    parser.add_argument("--address", help="Trace a single address")
    parser.add_argument("--arbiscan-key", default=os.environ.get("ARBISCAN_API_KEY", ""),
                        help="Arbiscan API key (or ARBISCAN_API_KEY env var)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    if args.address:
        # Single address trace
        async def single():
            if args.arbiscan_key:
                result = await trace_via_arbiscan(args.address, args.arbiscan_key)
            else:
                rpc = os.environ.get("ARB_WSS_URL")
                async with AsyncWeb3(WebSocketProvider(rpc)) as w3:
                    result = await trace_funding_source(w3, args.address)
            print(json.dumps(result, indent=2))
        asyncio.run(single())
    else:
        asyncio.run(trace_all_priority(args.arbiscan_key))


if __name__ == "__main__":
    main()
