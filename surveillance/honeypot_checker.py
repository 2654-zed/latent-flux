"""
Layer 3 -- Honeypot.is + GoPlus API integration

Secondary verification for SUSPECTED contracts using external APIs.
No auth required for Honeypot.is. GoPlus requires API key.

Rate limits:
  Honeypot.is: 1 request per 500ms (conservative)
  GoPlus: 5 requests per second with API key
"""

import asyncio
import json
import logging
import os
import sqlite3
from typing import Optional

import aiohttp

from surveillance import db

logger = logging.getLogger("surveillance.honeypot_checker")

# NOTE: Honeypot.is does NOT support Arbitrum (returns "Invalid chain" for chainID=42161).
# GoPlus returns empty results for unlisted contracts (our trap contracts aren't tokens).
# These APIs are useful for Ethereum mainnet tokens only.
# For Arbitrum trap detection, our bytecode classifier is the primary tool.
HONEYPOT_IS_URL = "https://api.honeypot.is/v2/IsHoneypot"
GOPLUS_URL = "https://api.gopluslabs.io/api/v1/token_security/42161"
CHAIN_ID = 1  # Honeypot.is only supports Ethereum mainnet


async def check_honeypot_is(address: str, session: aiohttp.ClientSession) -> Optional[dict]:
    """Check a single address against Honeypot.is API."""
    try:
        params = {"address": address, "chainID": CHAIN_ID}
        async with session.get(HONEYPOT_IS_URL, params=params, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "is_honeypot": data.get("honeypotResult", {}).get("isHoneypot", False),
                    "honeypot_reason": data.get("honeypotResult", {}).get("honeypotReason", ""),
                    "buy_tax": data.get("simulationResult", {}).get("buyTax", 0),
                    "sell_tax": data.get("simulationResult", {}).get("sellTax", 0),
                    "buy_success": data.get("simulationResult", {}).get("buySuccess", False),
                    "sell_success": data.get("simulationResult", {}).get("sellSuccess", False),
                    "source": "honeypot.is",
                }
            elif resp.status == 429:
                logger.warning("Honeypot.is rate limited")
                await asyncio.sleep(5)
                return None
            else:
                return None
    except Exception as e:
        logger.debug("Honeypot.is check failed for %s: %s", address, e)
        return None


async def check_goplus_batch(addresses: list[str], session: aiohttp.ClientSession,
                             api_key: str = "") -> dict:
    """Check multiple addresses against GoPlus API (batch of up to 50)."""
    try:
        addr_str = ",".join(addresses[:50])
        params = {"contract_addresses": addr_str}
        headers = {}
        if api_key:
            headers["Authorization"] = api_key
        async with session.get(GOPLUS_URL, params=params, headers=headers, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                results = {}
                for addr, info in data.get("result", {}).items():
                    results[addr.lower()] = {
                        "is_honeypot": info.get("is_honeypot") == "1",
                        "cannot_sell_all": info.get("cannot_sell_all") == "1",
                        "buy_tax": float(info.get("buy_tax", 0)),
                        "sell_tax": float(info.get("sell_tax", 0)),
                        "is_open_source": info.get("is_open_source") == "1",
                        "is_proxy": info.get("is_proxy") == "1",
                        "owner_address": info.get("owner_address", ""),
                        "source": "goplus",
                    }
                return results
            return {}
    except Exception as e:
        logger.debug("GoPlus check failed: %s", e)
        return {}


async def enrich_suspected_contracts(conn: sqlite3.Connection,
                                     use_goplus: bool = False) -> dict:
    """Run external checks on all SUSPECTED contracts."""
    suspected = conn.execute(
        "SELECT contract_address FROM contracts WHERE confidence_tier = 'suspected'"
    ).fetchall()
    addresses = [r[0] for r in suspected]

    print(f"Checking {len(addresses)} SUSPECTED contracts...")

    stats = {"checked": 0, "honeypot_confirmed": 0, "errors": 0}

    async with aiohttp.ClientSession() as session:
        # Honeypot.is — one at a time with 500ms delay
        for i, addr in enumerate(addresses):
            result = await check_honeypot_is(addr, session)
            await asyncio.sleep(0.5)

            if result:
                stats["checked"] += 1
                verdict_json = json.dumps(result)

                if result["is_honeypot"]:
                    stats["honeypot_confirmed"] += 1
                    reason = result.get("honeypot_reason", "external API confirmation")
                    # Upgrade to confirmed
                    conn.execute(
                        """UPDATE contracts SET
                            confidence_tier = 'confirmed',
                            confidence_reason = confidence_reason || '; HONEYPOT.IS CONFIRMED: ' || ?
                           WHERE contract_address = ? AND confidence_tier != 'confirmed'""",
                        (reason, addr),
                    )
                    conn.execute(
                        "DELETE FROM bytecode_cache WHERE source_contract = ?",
                        (addr,),
                    )
                    conn.commit()
                    print(f"  [{i+1}] {addr[:18]}... HONEYPOT CONFIRMED: {reason}")
                    if result.get("sell_tax", 0) > 0:
                        print(f"       buy_tax={result['buy_tax']:.1%} sell_tax={result['sell_tax']:.1%}")
                else:
                    if (i + 1) % 20 == 0:
                        print(f"  [{i+1}/{len(addresses)}] progress...")
            else:
                stats["errors"] += 1

        # GoPlus — batch of 50
        if use_goplus:
            goplus_key = os.environ.get("GOPLUS_API_KEY", "")
            print(f"\nGoPlus batch check ({len(addresses)} addresses)...")
            for batch_start in range(0, len(addresses), 50):
                batch = addresses[batch_start:batch_start + 50]
                results = await check_goplus_batch(batch, session, goplus_key)
                for addr, info in results.items():
                    if info.get("is_honeypot") or info.get("cannot_sell_all"):
                        stats["honeypot_confirmed"] += 1
                        conn.execute(
                            """UPDATE contracts SET
                                confidence_tier = 'confirmed',
                                confidence_reason = confidence_reason || '; GOPLUS CONFIRMED: honeypot=' || ? || ' cannot_sell=' || ?
                               WHERE contract_address = ? AND confidence_tier != 'confirmed'""",
                            (str(info["is_honeypot"]), str(info["cannot_sell_all"]), addr),
                        )
                        conn.execute(
                            "DELETE FROM bytecode_cache WHERE source_contract = ?",
                            (addr,),
                        )
                        conn.commit()
                        sell_tax = info.get("sell_tax", 0)
                        print(f"  GOPLUS CONFIRMED: {addr[:18]}... sell_tax={sell_tax:.1%}")
                await asyncio.sleep(0.25)

    return stats


async def main():
    conn = db.init_db()
    stats = await enrich_suspected_contracts(conn, use_goplus=bool(os.environ.get("GOPLUS_API_KEY")))

    print(f"\nResults:")
    print(f"  Checked:    {stats['checked']}")
    print(f"  Confirmed:  {stats['honeypot_confirmed']}")
    print(f"  Errors:     {stats['errors']}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
