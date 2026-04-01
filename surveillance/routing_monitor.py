"""
Layer 3 -- Component 3: 1inch Routing Monitor

Polls the 1inch API every 5 minutes to cross-reference flagged contracts
against live Arbitrum token routing. Two functions:

1. Token registry check: Are any of our flagged contracts appearing as
   tokens in 1inch's registry? If yes, they have real liquidity exposure
   and are higher priority.

2. Routing anomaly detection: For SUSPECTED contracts, check if 1inch
   routes AROUND a token that appears to offer better pricing. An
   aggregator avoiding a seemingly-better price is an implicit signal
   that the token is dangerous.

Read-only. No swaps, no approvals, no transactions.

Usage:
    python -m surveillance.routing_monitor --api-key YOUR_1INCH_KEY

Environment variable alternative:
    export ONEINCH_API_KEY=YOUR_1INCH_KEY
    python -m surveillance.routing_monitor
"""

import argparse
import asyncio
import logging
import os
import signal
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db

logger = logging.getLogger("surveillance.routing")

# Arbitrum One chain ID
CHAIN_ID = 42161

# 1inch API base (v6.0 -- current stable)
API_BASE = f"https://api.1inch.dev/swap/v6.0/{CHAIN_ID}"

# Token list endpoint (separate from swap API)
TOKEN_LIST_URL = f"https://api.1inch.dev/token/v1.2/{CHAIN_ID}"

# Well-known Arbitrum tokens for routing anomaly quotes
WETH_ARB = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
USDC_ARB = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDT_ARB = "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9"

# Poll interval
DEFAULT_POLL_SECONDS = 300  # 5 minutes

# Rate limiting: 1inch API has rate limits. We space requests
# to avoid hitting them.
REQUEST_DELAY_SECONDS = 1.0


class RoutingMonitor:
    """
    Async polling monitor that cross-references the surveillance database
    against 1inch's Arbitrum token registry and routing paths.
    """

    def __init__(self, api_key: str, conn: sqlite3.Connection,
                 poll_interval: int = DEFAULT_POLL_SECONDS):
        self.api_key = api_key
        self.conn = conn
        self.poll_interval = poll_interval
        self._running = False
        self._cycles = 0
        self._routing_hits = 0
        self._anomalies_found = 0

    async def start(self) -> None:
        """Begin the polling loop."""
        self._running = True
        logger.info(
            "Routing monitor started -- polling every %ds", self.poll_interval
        )

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    await self._poll_cycle(session)
                except Exception as e:
                    logger.error("Poll cycle failed: %s", e, exc_info=True)

                self._cycles += 1

                # DB heartbeat every cycle
                try:
                    db.write_heartbeat(
                        self.conn, "routing_monitor",
                        self._cycles, self._routing_hits,
                    )
                except Exception:
                    pass

                if self._cycles % 12 == 0:  # hourly log at 5min intervals
                    s = db.stats(self.conn)
                    logger.info(
                        "Heartbeat -- cycle %d, routing_hits=%d, anomalies=%d, "
                        "db: %d contracts (%d suspected, %d confirmed)",
                        self._cycles, self._routing_hits, self._anomalies_found,
                        s["contracts_total"],
                        s["contracts_by_tier"].get("suspected", 0),
                        s["contracts_by_tier"].get("confirmed", 0),
                    )

                # Wait for next poll
                for _ in range(self.poll_interval):
                    if not self._running:
                        break
                    await asyncio.sleep(1)

        logger.info(
            "Routing monitor stopped -- %d cycles, %d routing hits, %d anomalies",
            self._cycles, self._routing_hits, self._anomalies_found,
        )

    def stop(self) -> None:
        self._running = False

    async def _poll_cycle(self, session: aiohttp.ClientSession) -> None:
        """One full poll cycle: token registry check + routing anomaly scan."""
        logger.debug("Starting poll cycle %d", self._cycles + 1)

        # Phase 1: Fetch 1inch token list, cross-reference with our contracts
        await self._check_token_registry(session)

        # Phase 2: For SUSPECTED contracts, check for routing anomalies
        await self._check_routing_anomalies(session)

    # ------------------------------------------------------------------
    # Phase 1: Token registry cross-reference
    # ------------------------------------------------------------------

    async def _check_token_registry(self, session: aiohttp.ClientSession) -> None:
        """
        Fetch the 1inch token registry for Arbitrum and cross-reference
        against all contracts in our database that don't have routing_presence yet.
        """
        tokens = await self._fetch_token_list(session)
        if tokens is None:
            logger.warning("Failed to fetch token list -- skipping registry check")
            return

        # Normalize all 1inch token addresses to lowercase
        oneinch_addresses = {addr.lower() for addr in tokens.keys()}
        logger.debug("1inch registry: %d tokens on Arbitrum", len(oneinch_addresses))

        # Get our contracts that haven't been marked with routing_presence yet
        unrouted = db.get_contracts_without_routing(self.conn)
        if not unrouted:
            return

        hits = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        for contract in unrouted:
            addr = contract["contract_address"].lower()
            if addr in oneinch_addresses:
                db.update_contract_routing(self.conn, addr, now_iso)
                hits += 1
                self._routing_hits += 1

                token_info = tokens.get(addr, {})
                symbol = token_info.get("symbol", "???")
                name = token_info.get("name", "unknown")
                logger.info(
                    "ROUTING HIT: %s (%s / %s) -- contract in 1inch registry, "
                    "tier=%s, deployer=%s",
                    addr, symbol, name,
                    contract["confidence_tier"],
                    contract["deployer_address"][:18] + "...",
                )

        if hits:
            logger.info(
                "Registry check: %d new routing hits out of %d unchecked contracts",
                hits, len(unrouted),
            )

    async def _fetch_token_list(self, session: aiohttp.ClientSession) -> Optional[dict]:
        """Fetch the full token list from 1inch. Returns {address: token_info} or None."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            async with session.get(
                TOKEN_LIST_URL, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Response format: {"tokens": {"0xaddr": {symbol, name, ...}, ...}}
                    # or directly {"0xaddr": {symbol, name, ...}, ...}
                    if isinstance(data, dict):
                        if "tokens" in data:
                            return data["tokens"]
                        # Check if it's directly the address map
                        first_key = next(iter(data), "")
                        if first_key.startswith("0x"):
                            return data
                    return data
                elif resp.status == 429:
                    logger.warning("1inch rate limit hit (429) -- backing off")
                    await asyncio.sleep(10)
                    return None
                else:
                    body = await resp.text()
                    logger.warning(
                        "1inch token list returned %d: %s", resp.status, body[:200]
                    )
                    return None
        except Exception as e:
            logger.error("Failed to fetch token list: %s", e)
            return None

    # ------------------------------------------------------------------
    # Phase 2: Routing anomaly detection
    # ------------------------------------------------------------------

    async def _check_routing_anomalies(self, session: aiohttp.ClientSession) -> None:
        """
        For each SUSPECTED contract with routing_presence, check if 1inch
        routes around it despite apparent price advantage.

        Logic: Get a quote for a swap that COULD go through the suspected
        token. If 1inch's pathfinder avoids it while a direct path would
        be cheaper, that's a routing_anomaly signal.
        """
        suspected = db.get_suspected_contracts(self.conn)
        # Only check contracts that ARE in the routing registry
        routed_suspected = [c for c in suspected if c["routing_presence"]]

        if not routed_suspected:
            return

        logger.debug(
            "Checking %d suspected contracts with routing presence for anomalies",
            len(routed_suspected),
        )

        for contract in routed_suspected:
            addr = contract["contract_address"]

            # Rate limiting
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

            anomaly = await self._check_single_anomaly(session, addr)
            if anomaly:
                self._anomalies_found += 1
                # Update detection method if not already routing_anomaly
                if contract["detection_method"] != "routing_anomaly":
                    reason = (
                        f"{contract['confidence_reason']}; "
                        f"ROUTING ANOMALY: 1inch pathfinder avoids this token "
                        f"despite apparent price advantage -- {anomaly}"
                    )
                    db.update_contract_confidence(
                        self.conn, addr, "suspected", reason
                    )
                    logger.info(
                        "ROUTING ANOMALY: %s -- %s", addr, anomaly
                    )

    async def _check_single_anomaly(self, session: aiohttp.ClientSession,
                                    token_address: str) -> Optional[str]:
        """
        Check if 1inch avoids routing through a specific token.

        Strategy: Request a quote for WETH -> USDC. If the token_address
        does NOT appear in the route's token path despite being available
        in the registry, that's not necessarily anomalous (most tokens
        won't be on the optimal path). But if we can get a quote for
        WETH -> token -> USDC that shows a better rate than the direct
        route, and 1inch still avoids it, that IS anomalous.

        For Phase 1, we do a simpler check: can we get ANY quote involving
        this token? If the token is in the registry but 1inch refuses to
        quote it (returns error or empty route), that's a soft signal.
        """
        # Try to get a quote: token -> USDC (small amount)
        try:
            quote = await self._get_quote(
                session,
                src=token_address,
                dst=USDC_ARB,
                amount="1000000000000000000",  # 1 token (18 decimals assumed)
            )

            if quote is None:
                # 1inch refused to quote this token -- soft anomaly signal
                return "1inch returns no quote for token->USDC despite registry presence"

            # Quote succeeded -- token is routable, no anomaly from this check
            return None

        except Exception as e:
            logger.debug("Anomaly check failed for %s: %s", token_address, e)
            return None

    async def _get_quote(self, session: aiohttp.ClientSession,
                         src: str, dst: str, amount: str) -> Optional[dict]:
        """Get a swap quote from 1inch. Returns quote data or None."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        params = {
            "src": src,
            "dst": dst,
            "amount": amount,
        }

        try:
            url = f"{API_BASE}/quote"
            async with session.get(
                url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("1inch rate limit hit on quote -- backing off")
                    await asyncio.sleep(10)
                    return None
                elif resp.status == 400:
                    # Bad request -- token may be unquotable (insufficient liquidity, etc.)
                    return None
                else:
                    body = await resp.text()
                    logger.debug("Quote %s->%s returned %d: %s",
                                 src[:10], dst[:10], resp.status, body[:100])
                    return None
        except asyncio.TimeoutError:
            logger.debug("Quote timeout for %s->%s", src[:10], dst[:10])
            return None
        except Exception as e:
            logger.debug("Quote error for %s->%s: %s", src[:10], dst[:10], e)
            return None

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def routing_hits(self) -> int:
        return self._routing_hits

    @property
    def anomalies_found(self) -> int:
        return self._anomalies_found


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

async def run_routing_monitor(api_key: str, db_path: Optional[Path] = None,
                              poll_interval: int = DEFAULT_POLL_SECONDS,
                              write_queue=None) -> None:
    """Entry point: initialize DB, start routing monitor, handle shutdown."""
    if write_queue is not None:
        from surveillance.db_queue import QueueConnection
        resolved = str(db_path) if db_path else str(db.DEFAULT_DB_PATH)
        conn = QueueConnection(resolved, write_queue)
    else:
        conn = db.init_db(db_path)
    monitor = RoutingMonitor(api_key, conn, poll_interval)

    loop = asyncio.get_running_loop()

    def _shutdown():
        logger.info("Shutdown signal received")
        monitor.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    try:
        await monitor.start()
    finally:
        conn.close()
        logger.info("Database connection closed")


def main():
    parser = argparse.ArgumentParser(
        description="Layer 3 -- 1inch Routing Monitor (Arbitrum)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ONEINCH_API_KEY"),
        help="1inch API key (or set ONEINCH_API_KEY env var)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: surveillance/data/surveillance.db)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help=f"Seconds between polls (default: {DEFAULT_POLL_SECONDS})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "ERROR: No 1inch API key provided.\n"
            "  --api-key YOUR_KEY\n"
            "  or set ONEINCH_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    db_path = Path(args.db) if args.db else None

    try:
        asyncio.run(run_routing_monitor(args.api_key, db_path, args.poll_interval))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
