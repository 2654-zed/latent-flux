"""
Layer 3 — Extended Event Monitors

Six new monitoring capabilities that plug into the existing block processing loop:

1. DEX Liquidity Events — addLiquidity/removeLiquidity on major routers
2. Token Approval Monitoring — approve() calls to suspected/confirmed contracts
3. Mempool Watching — pending transaction subscription (separate loop)
4. Bridge Activity — cross-chain transfers from known org wallets
5. CEX Deposit Pattern Detection — addresses with high-inflow/zero-outflow pattern
6. New Token Pair Creation — PairCreated events from DEX factories

All read-only. Never sends transactions.

Usage:
    Integrated as sub-monitor in deployment_monitor.py via process_block()
    Mempool monitor runs as a separate async task.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Set

from web3 import AsyncWeb3

from surveillance import db

logger = logging.getLogger("surveillance.event_monitors")

# =====================================================================
# Contract addresses (Arbitrum + Base)
# =====================================================================

# DEX Routers
SUSHISWAP_V2_ROUTER_ARB = "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506"
UNISWAP_V3_ROUTER_ARB = "0xe592427a0aece92de3edee1f18e0157c05861564"
UNISWAP_UNIVERSAL_ARB = "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad"
PANCAKESWAP_V3_BASE = "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24"

DEX_ROUTERS = {
    SUSHISWAP_V2_ROUTER_ARB, UNISWAP_V3_ROUTER_ARB,
    UNISWAP_UNIVERSAL_ARB, PANCAKESWAP_V3_BASE,
}

# DEX Factories (for PairCreated events)
UNISWAP_V2_FACTORY_ARB = "0xf1d7cc64fb4452f05c498126312ebe29f30fbcf9"
SUSHISWAP_V2_FACTORY_ARB = "0xc35dadb65012ec5796536bd9864ed8773abc74c4"
UNISWAP_V3_FACTORY_ARB = "0x1f98431c8ad98523631ae4a59f267346ea31f984"
PANCAKESWAP_V3_FACTORY_BASE = "0x0bfbcf9fa4f9c56b0f40a671ad40e0805a091865"

DEX_FACTORIES = {
    UNISWAP_V2_FACTORY_ARB, SUSHISWAP_V2_FACTORY_ARB,
    UNISWAP_V3_FACTORY_ARB, PANCAKESWAP_V3_FACTORY_BASE,
}

# Bridge contracts
ARB_BRIDGE = "0x0000000000000000000000000000000000000064"  # ArbSys precompile
BASE_BRIDGE = "0x4200000000000000000000000000000000000010"  # L2StandardBridge
STARGATE_ARB = "0x53bf833a5d6c4dda888f69c22c88c9f356a41614"
STARGATE_BASE = "0x27a16dc786820b16e5c9028b75b99f6f604b5d26"

BRIDGE_CONTRACTS = {ARB_BRIDGE, BASE_BRIDGE, STARGATE_ARB, STARGATE_BASE}

# Function selectors
SEL_ADD_LIQUIDITY = "e8e33700"        # addLiquidity(address,address,uint,uint,uint,uint,address,uint)
SEL_ADD_LIQUIDITY_ETH = "f305d719"    # addLiquidityETH(address,uint,uint,uint,address,uint)
SEL_REMOVE_LIQUIDITY = "baa2abde"     # removeLiquidity
SEL_REMOVE_LIQUIDITY_ETH = "02751cec" # removeLiquidityETH
SEL_APPROVE = "095ea7b3"
SEL_TRANSFER = "a9059cbb"
SEL_TRANSFER_FROM = "23b872dd"
SEL_MINT = "6a627842"                 # mint(address) — Uniswap V2 pair mint
SEL_BURN = "89afcb44"                 # burn(address) — Uniswap V2 pair burn

LIQUIDITY_SELECTORS = {
    SEL_ADD_LIQUIDITY, SEL_ADD_LIQUIDITY_ETH,
    SEL_REMOVE_LIQUIDITY, SEL_REMOVE_LIQUIDITY_ETH,
    SEL_MINT, SEL_BURN,
}

# PairCreated event topic
PAIR_CREATED_TOPIC = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create monitoring tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS liquidity_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_hash         TEXT NOT NULL,
            block_number    INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            chain           TEXT NOT NULL,
            router_address  TEXT NOT NULL,
            caller_address  TEXT NOT NULL,
            event_type      TEXT NOT NULL,  -- add_liquidity, remove_liquidity, mint, burn
            selector        TEXT NOT NULL,
            token_address   TEXT,           -- extracted from calldata if possible
            linked_deployer TEXT,           -- if caller/token is a known deployer
            alert_level     TEXT DEFAULT 'info'  -- info, warning, critical
        );

        CREATE TABLE IF NOT EXISTS approval_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_hash         TEXT NOT NULL,
            block_number    INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            chain           TEXT NOT NULL,
            token_contract  TEXT NOT NULL,
            approver        TEXT NOT NULL,
            spender         TEXT NOT NULL,
            linked_deployer TEXT,
            alert_level     TEXT DEFAULT 'info'
        );

        CREATE TABLE IF NOT EXISTS bridge_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_hash         TEXT NOT NULL,
            block_number    INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            chain           TEXT NOT NULL,
            bridge_contract TEXT NOT NULL,
            sender          TEXT NOT NULL,
            value_wei       TEXT,
            org_link        TEXT,
            alert_level     TEXT DEFAULT 'info'
        );

        CREATE TABLE IF NOT EXISTS pair_creation_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_hash         TEXT NOT NULL,
            block_number    INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            chain           TEXT NOT NULL,
            factory_address TEXT NOT NULL,
            token0          TEXT,
            token1          TEXT,
            pair_address    TEXT,
            linked_deployer TEXT,
            alert_level     TEXT DEFAULT 'info'
        );

        CREATE TABLE IF NOT EXISTS cex_deposit_candidates (
            address         TEXT NOT NULL PRIMARY KEY,
            chain           TEXT NOT NULL,
            unique_senders  INTEGER NOT NULL DEFAULT 0,
            total_inflows   INTEGER NOT NULL DEFAULT 0,
            total_outflows  INTEGER NOT NULL DEFAULT 0,
            first_seen      TEXT NOT NULL,
            last_seen       TEXT NOT NULL,
            flagged         INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_liq_events_caller ON liquidity_events(caller_address);
        CREATE INDEX IF NOT EXISTS idx_liq_events_ts ON liquidity_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_approval_token ON approval_events(token_contract);
        CREATE INDEX IF NOT EXISTS idx_bridge_sender ON bridge_events(sender);
        CREATE INDEX IF NOT EXISTS idx_pair_creation_ts ON pair_creation_events(timestamp);
    """)


class EventMonitors:
    """
    Processes each block for 6 event types.
    Plugs into the deployment monitor's block processing loop.
    """

    def __init__(self, conn: sqlite3.Connection, chain: str = "arbitrum"):
        self.conn = conn
        self.chain = chain
        _ensure_tables(conn)

        # Cache known deployers and org wallets for fast lookup
        self._known_deployers: Set[str] = set()
        self._org_wallets: Set[str] = set()
        self._suspected_contracts: Set[str] = set()
        self._refresh_cache()

        self.events_logged = 0
        logger.info("EventMonitors initialized for %s", chain)

    def _refresh_cache(self) -> None:
        """Refresh cached address sets from DB. Called periodically."""
        try:
            self._known_deployers = set(
                r[0].lower() for r in self.conn.execute(
                    "SELECT deployer_address FROM deployers"
                ).fetchall()
            )
            self._org_wallets = set(
                r[0].lower() for r in self.conn.execute(
                    "SELECT deployer_address FROM deployers WHERE entity_type NOT IN ('unknown', 'mev_bot_factory', 'protocol')"
                ).fetchall()
            )
            self._suspected_contracts = set(
                r[0].lower() for r in self.conn.execute(
                    "SELECT contract_address FROM contracts WHERE confidence_tier IN ('suspected', 'confirmed')"
                ).fetchall()
            )
        except Exception as e:
            logger.warning("Cache refresh failed: %s", e)

    async def process_block(self, w3: AsyncWeb3, block: dict, timestamp_iso: str) -> None:
        """Process a block for all 6 event types."""
        block_number = block["number"]
        txs = block.get("transactions", [])

        # Refresh cache every 100 blocks
        if block_number % 100 == 0:
            self._refresh_cache()

        for tx in txs:
            to_addr = (tx.get("to") or "").lower()
            from_addr = (tx.get("from") or "").lower()
            input_data = (tx.get("input") or "0x")
            if isinstance(input_data, bytes):
                input_data = "0x" + input_data.hex()
            selector = input_data[2:10] if len(input_data) >= 10 else ""
            tx_hash = tx.get("hash", b"").hex() if isinstance(tx.get("hash"), bytes) else str(tx.get("hash", ""))
            value = tx.get("value", 0)

            # 1. DEX Liquidity Events
            if to_addr in DEX_ROUTERS and selector in LIQUIDITY_SELECTORS:
                self._handle_liquidity(tx_hash, block_number, timestamp_iso,
                                       to_addr, from_addr, selector, input_data)

            # 2. Token Approval to suspected contracts
            if selector == SEL_APPROVE and to_addr in self._suspected_contracts:
                spender = "0x" + input_data[34:74] if len(input_data) >= 74 else "unknown"
                self._handle_approval(tx_hash, block_number, timestamp_iso,
                                      to_addr, from_addr, spender)

            # 3. Bridge Activity from org wallets
            if to_addr in BRIDGE_CONTRACTS and from_addr in self._org_wallets:
                self._handle_bridge(tx_hash, block_number, timestamp_iso,
                                    to_addr, from_addr, value)

            # 5. CEX Deposit Pattern — track inflow/outflow ratios
            if value and int(value) > 0:
                self._update_cex_candidate(to_addr, from_addr, timestamp_iso)

        # 4. Check logs for PairCreated events (requires receipt — sample only)
        # We check logs from the block for factory events
        try:
            if block_number % 100 == 0:  # Sample every 100th block (~3-4 min on Arb, ~3 min on Base)
                await self._check_pair_created(w3, block_number, timestamp_iso)
        except Exception as e:
            logger.debug("PairCreated check skipped: %s", e)

    def _handle_liquidity(self, tx_hash: str, block: int, ts: str,
                          router: str, caller: str, selector: str, input_data: str) -> None:
        """Record a DEX liquidity event."""
        if selector in (SEL_ADD_LIQUIDITY, SEL_ADD_LIQUIDITY_ETH, SEL_MINT):
            event_type = "add_liquidity"
        else:
            event_type = "remove_liquidity"

        # Try to extract token address from calldata
        token_addr = None
        if len(input_data) >= 74:
            token_addr = "0x" + input_data[34:74]

        # Check if caller is a known deployer
        linked = None
        alert = "info"
        if caller in self._known_deployers:
            linked = caller
            alert = "warning"
            if caller in self._org_wallets:
                alert = "critical"
                logger.warning(
                    "CRITICAL: Org wallet %s called %s on router %s (tx: %s)",
                    caller[:14], event_type, router[:14], tx_hash[:18]
                )

        # Check if token is a known deployer's contract
        if token_addr and token_addr.lower() in self._suspected_contracts:
            alert = "critical"
            logger.warning(
                "CRITICAL: Liquidity %s for suspected contract %s (tx: %s)",
                event_type, token_addr[:14], tx_hash[:18]
            )

        try:
            self.conn.execute(
                """INSERT INTO liquidity_events
                   (tx_hash, block_number, timestamp, chain, router_address,
                    caller_address, event_type, selector, token_address,
                    linked_deployer, alert_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_hash, block, ts, self.chain, router, caller,
                 event_type, selector, token_addr, linked, alert),
            )
            self.conn.commit()
            self.events_logged += 1
        except Exception as e:
            logger.debug("Liquidity event insert failed: %s", e)

    def _handle_approval(self, tx_hash: str, block: int, ts: str,
                         token: str, approver: str, spender: str) -> None:
        """Record an approval to a suspected/confirmed contract."""
        linked = None
        alert = "info"
        if approver in self._known_deployers:
            linked = approver
            alert = "warning"

        # Check if spender is a DEX router (pre-staged drain pattern)
        if spender.lower() in DEX_ROUTERS:
            alert = "warning"
            logger.info(
                "Approval to suspected contract %s with DEX router spender (tx: %s)",
                token[:14], tx_hash[:18]
            )

        try:
            self.conn.execute(
                """INSERT INTO approval_events
                   (tx_hash, block_number, timestamp, chain, token_contract,
                    approver, spender, linked_deployer, alert_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_hash, block, ts, self.chain, token, approver,
                 spender, linked, alert),
            )
            self.conn.commit()
            self.events_logged += 1
        except Exception as e:
            logger.debug("Approval event insert failed: %s", e)

    def _handle_bridge(self, tx_hash: str, block: int, ts: str,
                       bridge: str, sender: str, value: int) -> None:
        """Record a bridge event from an org wallet."""
        # Find org link
        org_link = None
        try:
            r = self.conn.execute(
                "SELECT entity_type FROM deployers WHERE deployer_address = ?",
                (sender,),
            ).fetchone()
            if r:
                org_link = r[0]
        except Exception:
            pass

        logger.warning(
            "BRIDGE: Org wallet %s (%s) sent %s wei to bridge %s",
            sender[:14], org_link or "?", value, bridge[:14]
        )

        try:
            self.conn.execute(
                """INSERT INTO bridge_events
                   (tx_hash, block_number, timestamp, chain, bridge_contract,
                    sender, value_wei, org_link, alert_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_hash, block, ts, self.chain, bridge, sender,
                 str(value), org_link, "critical"),
            )
            self.conn.commit()
            self.events_logged += 1
        except Exception as e:
            logger.debug("Bridge event insert failed: %s", e)

    def _update_cex_candidate(self, to_addr: str, from_addr: str, ts: str) -> None:
        """Track inflow/outflow ratio for CEX deposit pattern detection."""
        try:
            # Update receiver (inflow)
            self.conn.execute("""
                INSERT INTO cex_deposit_candidates (address, chain, unique_senders, total_inflows, total_outflows, first_seen, last_seen)
                VALUES (?, ?, 1, 1, 0, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    unique_senders = unique_senders + CASE
                        WHEN address NOT IN (SELECT address FROM cex_deposit_candidates WHERE address = ?) THEN 1 ELSE 0 END,
                    total_inflows = total_inflows + 1,
                    last_seen = ?
            """, (to_addr, self.chain, ts, ts, to_addr, ts))

            # Update sender (outflow)
            self.conn.execute("""
                INSERT INTO cex_deposit_candidates (address, chain, unique_senders, total_inflows, total_outflows, first_seen, last_seen)
                VALUES (?, ?, 0, 0, 1, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    total_outflows = total_outflows + 1,
                    last_seen = ?
            """, (from_addr, self.chain, ts, ts, ts))

            # Flag addresses with CEX pattern: 20+ unique senders, 0 outflows
            # (checked periodically, not every tx)
        except Exception:
            pass  # High-volume table, suppress individual errors

    async def _check_pair_created(self, w3: AsyncWeb3, block_number: int, ts: str) -> None:
        """Check for PairCreated events from DEX factories."""
        try:
            logs = await w3.eth.get_logs({
                "fromBlock": block_number,
                "toBlock": block_number,
                "topics": [PAIR_CREATED_TOPIC],
            })
        except Exception:
            return

        for log in logs:
            factory = (log.get("address") or "").lower()
            tx_hash = log.get("transactionHash", b"").hex() if isinstance(log.get("transactionHash"), bytes) else str(log.get("transactionHash", ""))
            topics = log.get("topics", [])
            data = log.get("data", "0x")

            # PairCreated(address token0, address token1, address pair, uint)
            token0 = "0x" + topics[1].hex()[-40:] if len(topics) > 1 else None
            token1 = "0x" + topics[2].hex()[-40:] if len(topics) > 2 else None

            # Extract pair address from data
            pair_addr = None
            if isinstance(data, bytes) and len(data) >= 32:
                pair_addr = "0x" + data[:32].hex()[-40:]
            elif isinstance(data, str) and len(data) >= 66:
                pair_addr = "0x" + data[2:66][-40:]

            # Check if either token is from a known deployer
            linked = None
            alert = "info"
            for token in [token0, token1]:
                if token and token.lower() in self._suspected_contracts:
                    linked = token
                    alert = "critical"
                    logger.warning(
                        "CRITICAL: New DEX pair created with suspected token %s (pair: %s)",
                        token[:14], pair_addr[:14] if pair_addr else "?"
                    )

            try:
                self.conn.execute(
                    """INSERT INTO pair_creation_events
                       (tx_hash, block_number, timestamp, chain, factory_address,
                        token0, token1, pair_address, linked_deployer, alert_level)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (tx_hash, block_number, ts, self.chain, factory,
                     token0, token1, pair_addr, linked, alert),
                )
                self.conn.commit()
                self.events_logged += 1
            except Exception as e:
                logger.debug("Pair creation insert failed: %s", e)

    def flag_cex_candidates(self) -> int:
        """Flag addresses matching CEX deposit pattern. Call periodically."""
        try:
            result = self.conn.execute("""
                UPDATE cex_deposit_candidates
                SET flagged = 1
                WHERE unique_senders >= 20
                  AND total_outflows = 0
                  AND flagged = 0
            """)
            self.conn.commit()
            count = result.rowcount
            if count > 0:
                logger.info("Flagged %d new CEX deposit candidates", count)
            return count
        except Exception as e:
            logger.warning("CEX candidate flagging failed: %s", e)
            return 0
