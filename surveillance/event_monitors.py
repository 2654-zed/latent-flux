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

# Arbitrum-native DEXes
CAMELOT_V3_ROUTER_ARB = "0x1f721e2e82f6951c7b59a09dc21c1a9f4c882d1c"
CAMELOT_V2_ROUTER_ARB = "0xc873fecbd354f5a56e00e710b90ef4201db2448d"
TRADER_JOE_V21_ARB = "0xb4315e873dbcf96ffd0acd8ea43f689d8c20fb30"
UNISWAP_V2_ROUTER_ARB = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"

# Base-native DEXes
AERODROME_ROUTER_BASE = "0xcf77a3ba9a5ca399b7c97c74d54e5b1beb874e43"
AERODROME_V2_ROUTER_BASE = "0x6cb442acf35158d5eda88fe602221b67b400be3e"
BASESWAP_ROUTER_BASE = "0x327df1e6de05895d2ab08513aadd9313fe505d86"
VELODROME_ROUTER_BASE = "0xa062ae8a9c5e11aaa026fc2670b0d65ccc8b2858"
PANCAKESWAP_V2_ROUTER_BASE = "0x02a84c1b3bbd7401a5f7fa98a384ebc70bb5749e"

# Aggregators
ODOS_V2_ROUTER = "0xa669e7a0d4b3e4fa48af2de86bd4cd7126be4e13"
PARASWAP_V6 = "0x6a000f20005980200259b80c5102003040001068"
ZEROX_PROXY = "0xdef1c0ded9bec7f1a1670819833240f027b25eff"

DEX_ROUTERS = {
    SUSHISWAP_V2_ROUTER_ARB, UNISWAP_V3_ROUTER_ARB,
    UNISWAP_UNIVERSAL_ARB, PANCAKESWAP_V3_BASE,
    CAMELOT_V3_ROUTER_ARB, CAMELOT_V2_ROUTER_ARB,
    TRADER_JOE_V21_ARB, UNISWAP_V2_ROUTER_ARB,
    AERODROME_ROUTER_BASE, AERODROME_V2_ROUTER_BASE,
    BASESWAP_ROUTER_BASE, VELODROME_ROUTER_BASE,
    PANCAKESWAP_V2_ROUTER_BASE,
    ODOS_V2_ROUTER, PARASWAP_V6, ZEROX_PROXY,
}

# DEX Factories (for PairCreated events)
UNISWAP_V2_FACTORY_ARB = "0xf1d7cc64fb4452f05c498126312ebe29f30fbcf9"
SUSHISWAP_V2_FACTORY_ARB = "0xc35dadb65012ec5796536bd9864ed8773abc74c4"
UNISWAP_V3_FACTORY_ARB = "0x1f98431c8ad98523631ae4a59f267346ea31f984"
PANCAKESWAP_V3_FACTORY_BASE = "0x0bfbcf9fa4f9c56b0f40a671ad40e0805a091865"

CAMELOT_V2_FACTORY_ARB = "0x6eccab422d763ac031210895c81787e87b43a652"
AERODROME_FACTORY_BASE = "0x420dd381b31aef6683db6b902084cb0ffece40da"

DEX_FACTORIES = {
    UNISWAP_V2_FACTORY_ARB, SUSHISWAP_V2_FACTORY_ARB,
    UNISWAP_V3_FACTORY_ARB, PANCAKESWAP_V3_FACTORY_BASE,
    CAMELOT_V2_FACTORY_ARB, AERODROME_FACTORY_BASE,
}

# Bridge contracts
ARB_BRIDGE = "0x0000000000000000000000000000000000000064"  # ArbSys precompile
BASE_BRIDGE = "0x4200000000000000000000000000000000000010"  # L2StandardBridge
STARGATE_ARB = "0x53bf833a5d6c4dda888f69c22c88c9f356a41614"
STARGATE_BASE = "0x27a16dc786820b16e5c9028b75b99f6f604b5d26"

BRIDGE_CONTRACTS = {ARB_BRIDGE, BASE_BRIDGE, STARGATE_ARB, STARGATE_BASE}

# ERC-20 token contracts to monitor for org-wallet outbound transfers.
# Keyed by lowercase contract address -> (symbol, decimals, chain).
# Closes the observability gap where org funds exit as USDC/WETH rather
# than native ETH (discovered 2026-04-08 via 0xe69f81b8 trace: 90M USDC
# cycled through an org-adjacent EOA while our ETH-only scanner saw
# nothing).
ERC20_TOKENS = {
    # USDC
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": ("USDC", 6,  "base"),
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831": ("USDC", 6,  "arbitrum"),
    "0x0b2c639c533813f4aa9d7837caf62653d097ff85": ("USDC", 6,  "optimism"),
    # WETH
    "0x4200000000000000000000000000000000000006": ("WETH", 18, "base/optimism"),
    "0x82af49447d8a07e3bd95bd0d56f35241523fbab1": ("WETH", 18, "arbitrum"),
    # USDT
    "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2": ("USDT", 6,  "base"),
    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9": ("USDT", 6,  "arbitrum"),
    "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": ("USDT", 6,  "optimism"),
    # cbBTC (base native wrapped BTC by Coinbase)
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": ("cbBTC", 8, "base"),
}

# Bridge name lookup for logging / alerting.
BRIDGE_NAMES = {
    BASE_BRIDGE:   "L2StandardBridge",
    ARB_BRIDGE:    "ArbSys",
    STARGATE_ARB:  "StargateArbitrum",
    STARGATE_BASE: "StargateBase",
}

# Known bridge withdraw selectors.
BRIDGE_SELECTORS = {
    "32b7006d": "withdraw(address,uint256,uint32,bytes)",
    "a3a79548": "withdrawTo(address,uint256,uint32,bytes,address)",
    "25e16063": "withdrawEth(address)",
}

# Alert thresholds in ETH for bridge withdrawals.
BRIDGE_ALERT_ETH_MEDIUM = 10.0
BRIDGE_ALERT_ETH_HIGH   = 100.0

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


def _decode_erc20_transfer(selector: str, calldata: str) -> Optional[tuple[str, int]]:
    """Decode ERC-20 transfer / transferFrom calldata.

    Returns (recipient_address, raw_amount_as_int) or None on malformed
    input. Caller must apply token decimals separately.

    transfer(address _to, uint256 _value):
        selector a9059cbb + 32B _to + 32B _value
    transferFrom(address _from, address _to, uint256 _value):
        selector 23b872dd + 32B _from + 32B _to + 32B _value
    """
    if not calldata or len(calldata) < 10:
        return None
    params = calldata[10:] if calldata.startswith("0x") else calldata[8:]

    def _word(i: int) -> Optional[str]:
        start = i * 64
        w = params[start:start + 64]
        return w if len(w) == 64 else None

    try:
        if selector == "a9059cbb":  # transfer(to, value)
            w_to = _word(0)
            w_val = _word(1)
            if not w_to or not w_val:
                return None
            return ("0x" + w_to[-40:], int(w_val, 16))
        if selector == "23b872dd":  # transferFrom(from, to, value)
            w_to = _word(1)
            w_val = _word(2)
            if not w_to or not w_val:
                return None
            return ("0x" + w_to[-40:], int(w_val, 16))
    except Exception:
        return None
    return None


def _decode_bridge_l1_recipient(selector: str, calldata: str,
                                sender: str) -> Optional[str]:
    """Extract the L1 destination from bridge withdrawal calldata.

    L2StandardBridge on Base/Optimism exposes:
      withdraw(address _l2Token, uint256 _amount, uint32 _l1Gas, bytes _data)
          — L1 recipient == msg.sender (same EOA on L1), no arg
      withdrawTo(address _l2Token, address _to, uint256 _amount,
                 uint32 _l1Gas, bytes _data)
          — L1 recipient is the 2nd arg (`_to`)

    ArbSys on Arbitrum exposes:
      withdrawEth(address destination)
          — L1 recipient is the 1st arg

    Returns None for unknown selectors or malformed calldata.
    """
    if not calldata or len(calldata) < 10:
        return None
    params = calldata[10:]  # strip 0x + selector

    def _word(i: int) -> Optional[str]:
        start = i * 64
        w = params[start:start + 64]
        if len(w) != 64:
            return None
        return w

    if selector == "32b7006d":  # withdraw(l2Token, amount, l1Gas, data)
        return sender  # L1 recipient == sender (same EOA on L1)

    if selector == "a3a79548":  # withdrawTo(l2Token, _to, amount, l1Gas, data)
        # L2StandardBridge ABI: _to is the 2nd fixed-width word
        w = _word(1)
        if w:
            return "0x" + w[-40:]
        return None

    if selector == "25e16063":  # withdrawEth(destination) — ArbSys
        w = _word(0)
        if w:
            return "0x" + w[-40:]
        return None

    return None


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
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_hash               TEXT NOT NULL,
            block_number          INTEGER NOT NULL,
            timestamp             TEXT NOT NULL,
            chain                 TEXT NOT NULL,
            bridge_contract       TEXT NOT NULL,
            sender                TEXT NOT NULL,
            value_wei             TEXT,
            org_link              TEXT,
            alert_level           TEXT DEFAULT 'info',
            selector              TEXT,
            function_name         TEXT,
            value_eth             REAL,
            decoded_l1_recipient  TEXT,
            bridge_name           TEXT
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

        CREATE TABLE IF NOT EXISTS org_transfer_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_hash         TEXT NOT NULL,
            block_number    INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            chain           TEXT NOT NULL,
            from_address    TEXT NOT NULL,
            to_address      TEXT NOT NULL,
            value_eth       REAL,
            token           TEXT,
            org_id          TEXT,
            from_role       TEXT,
            selector        TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_liq_events_caller ON liquidity_events(caller_address);
        CREATE INDEX IF NOT EXISTS idx_liq_events_ts ON liquidity_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_org_transfer_from ON org_transfer_events(from_address);
        CREATE INDEX IF NOT EXISTS idx_org_transfer_ts ON org_transfer_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_approval_token ON approval_events(token_contract);
        CREATE INDEX IF NOT EXISTS idx_bridge_sender ON bridge_events(sender);
        CREATE INDEX IF NOT EXISTS idx_pair_creation_ts ON pair_creation_events(timestamp);
    """)

    # Idempotent column additions for older databases where bridge_events
    # was created before the scanner enhancements landed.
    for col, col_type in (
        ("selector",             "TEXT"),
        ("function_name",        "TEXT"),
        ("value_eth",            "REAL"),
        ("decoded_l1_recipient", "TEXT"),
        ("bridge_name",          "TEXT"),
    ):
        try:
            conn.execute(f"ALTER TABLE bridge_events ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Idempotent columns for ERC-20 org transfers (USDC/WETH/USDT/cbBTC).
    for col, col_type in (
        ("token_contract", "TEXT"),
        ("token_symbol",   "TEXT"),
        ("token_value",    "REAL"),
    ):
        try:
            conn.execute(f"ALTER TABLE org_transfer_events ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass


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

            # Primary org wallets: deployers with a non-generic entity_type.
            primary = set(
                r[0].lower() for r in self.conn.execute(
                    "SELECT deployer_address FROM deployers "
                    "WHERE entity_type NOT IN ('unknown', 'mev_bot_factory', 'protocol')"
                ).fetchall()
            )

            # Additional: addresses explicitly classified to an org.
            try:
                classified = set(
                    r[0].lower() for r in self.conn.execute(
                        "SELECT address FROM entity_classification WHERE org_id IS NOT NULL"
                    ).fetchall()
                )
            except sqlite3.Error:
                classified = set()

            # Secondary org wallets: EOAs that received >= 100 ETH from primary
            # org wallets (gas-station splash zone). These are where org funds
            # get parked before being bridged / swept off-chain. Closes the
            # 2026-04-07 19k ETH coverage gap where an unlabeled EOA received
            # org funds then immediately withdrew via L2StandardBridge.
            try:
                secondary = set(
                    r[0].lower() for r in self.conn.execute(
                        "SELECT to_address FROM org_transfer_events "
                        "WHERE org_id IS NOT NULL AND value_eth >= 100"
                    ).fetchall()
                )
            except sqlite3.Error:
                secondary = set()

            self._org_wallets = primary | classified | secondary

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
                                    to_addr, from_addr, value, input_data)

            # 5. CEX Deposit Pattern — DISABLED 2026-04-11
            # Root cause of disk-full crisis: wrote 2 rows per value-bearing tx
            # across all blocks, grew to 4.4M rows / 2GB+ WAL in production.
            # Re-enable only with rate limiting (e.g. sample 1-in-100 blocks).
            # if value and int(value) > 0:
            #     self._update_cex_candidate(to_addr, from_addr, timestamp_iso)

            # 6. Org wallet outbound transfers — capture where exit ramp money goes
            if from_addr in self._org_wallets and to_addr and to_addr != from_addr:
                org_role = None
                try:
                    r = self.conn.execute(
                        "SELECT entity_type FROM deployers WHERE deployer_address = ?",
                        (from_addr,),
                    ).fetchone()
                    org_role = r[0] if r else None
                except Exception:
                    pass

                # 6a. ERC-20 transfer / transferFrom to a monitored token
                # contract: decode recipient + amount from calldata and
                # record with token metadata. Prevents the $90M USDC
                # observability gap observed on 2026-04-08.
                if (to_addr in ERC20_TOKENS
                        and selector in ("a9059cbb", "23b872dd")):
                    decoded = _decode_erc20_transfer(selector, input_data)
                    if decoded:
                        recipient, raw_amount = decoded
                        symbol, decimals, _ = ERC20_TOKENS[to_addr]
                        token_value = raw_amount / (10 ** decimals)
                        self._handle_org_erc20_transfer(
                            tx_hash, block_number, timestamp_iso,
                            from_addr, recipient, to_addr, symbol,
                            token_value, selector, org_role,
                        )
                        continue  # don't also record as raw ETH event

                # 6b. Native ETH transfer (tx.value > 0) or other call
                value_eth = int(value) / 1e18 if value else 0
                self._handle_org_transfer(
                    tx_hash, block_number, timestamp_iso,
                    from_addr, to_addr, value_eth, selector, org_role,
                )

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
                       bridge: str, sender: str, value: int,
                       input_data: str) -> None:
        """Record a bridge event from an org wallet.

        Decodes the function selector and (where possible) the L1
        recipient, writes a row to bridge_events, and generates an
        alerts row if the ETH value crosses MEDIUM/HIGH thresholds.
        """
        # Find org link
        org_link = None
        try:
            r = self.conn.execute(
                "SELECT entity_type FROM deployers WHERE deployer_address = ?",
                (sender,),
            ).fetchone()
            if r:
                org_link = r[0]
            if not org_link:
                r = self.conn.execute(
                    "SELECT org_id FROM entity_classification WHERE address = ?",
                    (sender,),
                ).fetchone()
                if r and r[0]:
                    org_link = r[0]
        except Exception:
            pass

        # Normalize input
        calldata = input_data or "0x"
        if not calldata.startswith("0x"):
            calldata = "0x" + calldata
        selector = calldata[2:10] if len(calldata) >= 10 else ""
        function_name = BRIDGE_SELECTORS.get(selector)
        l1_recipient = _decode_bridge_l1_recipient(selector, calldata, sender)

        try:
            value_int = int(value) if value else 0
        except (TypeError, ValueError):
            value_int = 0
        value_eth = value_int / 1e18

        bridge_name = BRIDGE_NAMES.get(bridge, "unknown")

        # Alert level from value
        if value_eth >= BRIDGE_ALERT_ETH_HIGH:
            alert_level = "critical"
        elif value_eth >= BRIDGE_ALERT_ETH_MEDIUM:
            alert_level = "warning"
        else:
            alert_level = "info"

        logger.warning(
            "BRIDGE_WITHDRAWAL: %s (%s) -> %s %s (%s) value=%.4f ETH l1=%s",
            sender[:14], org_link or "?", bridge[:14], bridge_name,
            function_name or selector or "?", value_eth,
            (l1_recipient or "same")[:14],
        )

        try:
            self.conn.execute(
                """INSERT INTO bridge_events
                   (tx_hash, block_number, timestamp, chain, bridge_contract,
                    sender, value_wei, org_link, alert_level,
                    selector, function_name, value_eth, decoded_l1_recipient,
                    bridge_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_hash, block, ts, self.chain, bridge, sender,
                 str(value_int), org_link, alert_level,
                 selector, function_name, value_eth, l1_recipient,
                 bridge_name),
            )
            # Also generate an alerts row for MEDIUM+ withdrawals so the
            # alert feed surfaces them alongside other high-severity events.
            if alert_level in ("warning", "critical"):
                payload = (
                    f"{bridge_name} {function_name or selector} "
                    f"value={value_eth:.4f} ETH "
                    f"l1_recipient={l1_recipient or 'same EOA'} "
                    f"org_link={org_link or 'unknown'}"
                )
                self.conn.execute(
                    """INSERT INTO alerts
                       (alert_type, address, tx_hash, block_number, timestamp, payload, false_positive)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    ("BRIDGE_WITHDRAWAL", sender, tx_hash, block, ts, payload),
                )
            self.conn.commit()
            self.events_logged += 1
        except Exception as e:
            logger.debug("Bridge event insert failed: %s", e)

    def _handle_org_erc20_transfer(self, tx_hash: str, block: int, ts: str,
                                   from_addr: str, recipient: str,
                                   token_contract: str, symbol: str,
                                   token_value: float, selector: str,
                                   from_role: Optional[str]) -> None:
        """Record an ERC-20 transfer out of an org wallet to a real
        recipient. Writes to org_transfer_events with token metadata
        populated and value_eth=0.
        """
        org_id = None
        if from_role:
            if "org_002" in from_role:
                org_id = "org_002"
            elif from_role not in ("unknown", "mev_bot_factory", "protocol"):
                org_id = "org_001"
        try:
            self.conn.execute(
                """INSERT INTO org_transfer_events
                   (tx_hash, block_number, timestamp, chain, from_address, to_address,
                    value_eth, token, org_id, from_role, selector,
                    token_contract, token_symbol, token_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_hash, block, ts, self.chain, from_addr, recipient,
                 0.0, symbol, org_id, from_role, selector,
                 token_contract, symbol, token_value),
            )
            self.conn.commit()
            self.events_logged += 1
            if token_value > 1000:
                logger.info(
                    "ORG_ERC20: %s (%s) -> %s  %.2f %s",
                    from_addr[:14], from_role or "?", recipient[:14],
                    token_value, symbol,
                )
        except Exception as e:
            logger.debug("Org ERC20 transfer insert failed: %s", e)

    def _handle_org_transfer(self, tx_hash: str, block: int, ts: str,
                             from_addr: str, to_addr: str, value_eth: float,
                             selector: str, from_role: Optional[str]) -> None:
        """Record an outbound transfer from a known org wallet."""
        # Determine org_id from role
        org_id = None
        if from_role:
            if "org_002" in from_role:
                org_id = "org_002"
            elif from_role not in ("unknown", "mev_bot_factory", "protocol"):
                org_id = "org_001"

        # Determine token from selector
        token = "ETH"
        if selector in ("a9059cbb", "23b872dd"):  # transfer, transferFrom
            token = "ERC20"
        elif selector == "095ea7b3":
            token = "APPROVE"

        try:
            self.conn.execute(
                """INSERT INTO org_transfer_events
                   (tx_hash, block_number, timestamp, chain, from_address, to_address,
                    value_eth, token, org_id, from_role, selector)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tx_hash, block, ts, self.chain, from_addr, to_addr,
                 value_eth, token, org_id, from_role, selector),
            )
            self.conn.commit()
            self.events_logged += 1

            if value_eth > 0.1:
                logger.info(
                    "ORG_TRANSFER: %s (%s) -> %s %.4f %s",
                    from_addr[:14], from_role or "?", to_addr[:14], value_eth, token,
                )
        except Exception as e:
            logger.debug("Org transfer insert failed: %s", e)

    # cex_deposit_candidates writer removed 2026-04-22. The table accumulated
    # 2.69M rows with zero flags ever produced; flagger threshold was never
    # met. Code retained in git history (see commit archaeology). Kept
    # flag_cex_candidates as a no-op below for backward-compat with
    # deployment_monitor.py callers until they are also cleaned up.

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
        """No-op stub. cex_deposit_candidates was retired 2026-04-22
        (never produced flagged rows; table dropped). Kept as a no-op so
        deployment_monitor.py's periodic call does not KeyError. Remove
        when deployment_monitor.py line ~616 call is also deleted."""
        return 0

    def _defunct_flag_cex_candidates(self) -> int:
        """Historical implementation, retained under a renamed symbol so
        it does not get accidentally invoked. See commit log for context."""
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
