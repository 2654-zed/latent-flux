"""
Layer 3 -- Selector Monitor

Watches transactions hitting SUSPECTED/CONFIRMED contracts.
Extracts function selectors, tags known bot signatures, and
classifies gas patterns (static vs dynamic priority fees).

Runs as a component alongside the deployment monitor, sharing
the same WebSocket subscription. Processes every block's
transactions, not just contract creations.

Read-only -- never sends transactions.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from web3 import AsyncWeb3

from surveillance import db

logger = logging.getLogger("surveillance.selector")

# Known bot function selectors
BOT_SELECTORS = {
    "38ed1739": "legacy_v2_bot",        # swapExactTokensForTokens
    "ab9c4b5d": "aave_flashloan_bot",   # Aave flashLoan
    "ea0429f5": "aave_callback_bot",    # executeOperation callback
    "095ea7b3": "lazy_approval_bot",    # approve (flagged only with max uint256)
}

# max uint256 for lazy approval detection
MAX_UINT256 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

# transferFrom selector for drain detection
SEL_TRANSFER_FROM = "23b872dd"

# Static gas thresholds (exact round gwei values)
STATIC_GAS_VALUES = {0.0, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0}


def classify_gas_pattern(max_priority_fee_gwei: Optional[float]) -> Optional[str]:
    """Classify gas priority fee as static or dynamic."""
    if max_priority_fee_gwei is None:
        return None
    # Round to avoid float precision issues
    rounded = round(max_priority_fee_gwei, 4)
    if rounded in STATIC_GAS_VALUES:
        return "static"
    return "dynamic"


def identify_bot_tag(selector: Optional[str],
                     calldata_hex: str) -> Optional[str]:
    """Match a function selector against known bot signatures."""
    if not selector:
        return None

    tag = BOT_SELECTORS.get(selector)

    # Special case: approve with max uint256
    if selector == "095ea7b3" and MAX_UINT256 in calldata_hex:
        return "lazy_approval_bot"

    # For approve without max uint256, don't tag
    if selector == "095ea7b3" and tag == "lazy_approval_bot":
        return None

    return tag


class SelectorMonitor:
    """
    Processes full blocks to find transactions interacting with
    watched (SUSPECTED/CONFIRMED) contracts.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._watched: set[str] = set()
        self._exposure_contracts: set[str] = set()
        self._exposed_addresses: set[str] = set()
        self._events_logged = 0
        self._bots_tagged = 0
        self._drains_detected = 0
        self._last_refresh = 0.0
        self._vanity_checked: set[str] = set()  # cache to avoid re-checking

    def refresh_watched(self) -> None:
        """Reload watched sets from DB."""
        self._watched = db.get_watched_addresses(self.conn)
        self._exposure_contracts = db.get_exposure_contracts(self.conn)
        self._exposed_addresses = db.get_exposed_addresses(self.conn)

    @property
    def watched_count(self) -> int:
        return len(self._watched)

    @property
    def events_logged(self) -> int:
        return self._events_logged

    @property
    def bots_tagged(self) -> int:
        return self._bots_tagged

    async def process_block(self, w3: AsyncWeb3, block: dict,
                            timestamp_iso: str) -> None:
        """Scan all transactions in a block for interactions with watched contracts."""
        # Refresh watched set every 60 seconds
        now = asyncio.get_event_loop().time()
        if now - self._last_refresh > 60:
            self.refresh_watched()
            self._last_refresh = now

        if not self._watched:
            return

        block_number = block["number"]

        for tx in block["transactions"]:
            to_addr = tx.get("to")
            if to_addr is None:
                continue  # contract creation, handled by deployment monitor

            to_lower = to_addr.lower() if isinstance(to_addr, str) else to_addr

            if to_lower not in self._watched:
                continue

            # This transaction targets a watched contract
            await self._process_interaction(w3, tx, to_lower,
                                            block_number, timestamp_iso)

    async def _process_interaction(self, w3: AsyncWeb3, tx: dict,
                                   contract_address: str,
                                   block_number: int,
                                   timestamp_iso: str) -> None:
        """Extract selector, tag bots, classify gas, log to DB."""
        # Extract calldata
        input_data = tx.get("input", b"")
        if isinstance(input_data, bytes):
            calldata_hex = input_data.hex()
        else:
            calldata_hex = str(input_data).replace("0x", "")

        # Function selector = first 4 bytes
        selector = calldata_hex[:8] if len(calldata_hex) >= 8 else None

        # Bot tagging
        bot_tag = identify_bot_tag(selector, calldata_hex)

        # Gas analysis
        gas_price_wei = tx.get("gasPrice") or tx.get("maxFeePerGas")
        gas_price_gwei = float(gas_price_wei) / 1e9 if gas_price_wei else None

        max_priority_wei = tx.get("maxPriorityFeePerGas")
        max_priority_gwei = float(max_priority_wei) / 1e9 if max_priority_wei else None

        gas_pattern = classify_gas_pattern(max_priority_gwei)

        # Check revert status
        try:
            receipt = await w3.eth.get_transaction_receipt(tx["hash"])
            is_reverted = receipt["status"] == 0
        except Exception:
            is_reverted = False

        # Interacting address
        from_addr = tx["from"]
        if isinstance(from_addr, str):
            from_lower = from_addr.lower()
        else:
            from_lower = from_addr

        tx_hash = tx["hash"].hex() if isinstance(tx["hash"], bytes) else str(tx["hash"])

        # Capture transaction value
        raw_value = tx.get("value")
        value_wei = str(raw_value) if raw_value and int(raw_value) > 0 else None

        try:
            db.insert_transaction_event(
                self.conn,
                contract_address=contract_address,
                interacting_address=from_lower,
                function_selector=selector,
                bot_tag=bot_tag,
                gas_price_gwei=gas_price_gwei,
                max_priority_fee_gwei=max_priority_gwei,
                gas_pattern=gas_pattern,
                block_number=block_number,
                timestamp=timestamp_iso,
                is_reverted=is_reverted,
                tx_hash=tx_hash,
                value_wei=value_wei,
            )
            self._events_logged += 1

            # Tag vanity callers on first sight
            if from_lower not in self._vanity_checked:
                self._vanity_checked.add(from_lower)
                try:
                    from surveillance.vanity_detector import detect_vanity
                    vr = detect_vanity(from_lower)
                    if vr:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO vanity_tags (address, address_type, pattern_type, label, detected_at) VALUES (?, ?, ?, ?, ?)",
                            (from_lower, "caller", vr[0], vr[1], timestamp_iso),
                        )
                except Exception:
                    pass

            if bot_tag:
                self._bots_tagged += 1
                logger.info(
                    "BOT TAGGED: %s -> %s selector=%s tag=%s gas=%s reverted=%s",
                    from_lower[:18] + "...", contract_address[:18] + "...",
                    selector, bot_tag, gas_pattern, is_reverted,
                )

            # If reverted, this might also be a trap event
            if is_reverted:
                logger.info(
                    "REVERTED TX: %s -> %s selector=%s tx=%s",
                    from_lower[:18] + "...", contract_address[:18] + "...",
                    selector, tx_hash[:18] + "...",
                )

            # DRAIN DETECTION: transferFrom where spender is a watched contract
            # and the 'from' in calldata is an exposed address
            if selector == SEL_TRANSFER_FROM and not is_reverted:
                # transferFrom(address from, address to, uint256 amount)
                # calldata: selector + 32b from + 32b to + 32b amount
                if len(calldata_hex) >= 136:  # 8 + 64 + 64
                    victim_addr = "0x" + calldata_hex[32:72]
                    victim_lower = victim_addr.lower()
                    if (contract_address in self._exposure_contracts and
                            victim_lower in self._exposed_addresses):
                        self._drains_detected += 1
                        logger.warning(
                            "APPROVAL DRAIN DETECTED: contract %s draining %s via transferFrom tx=%s",
                            contract_address[:18] + "...", victim_lower[:18] + "...",
                            tx_hash[:18] + "...",
                        )
                        # Update exposure status
                        db.mark_exposure_drained(
                            self.conn, victim_lower, contract_address,
                            tx_hash, timestamp_iso, 0.0,
                        )
                        # Log alert
                        self.conn.execute(
                            """INSERT INTO alerts (alert_type, address, tx_hash, block_number, timestamp, payload)
                               VALUES ('APPROVAL_DRAIN', ?, ?, ?, ?, ?)""",
                            ("APPROVAL_DRAIN", victim_lower, tx_hash, block_number,
                             timestamp_iso, json.dumps({
                                 "victim": victim_lower,
                                 "contract": contract_address,
                                 "caller": from_lower,
                             })),
                        )
                        self.conn.commit()

        except Exception as e:
            logger.error("Failed to log tx event: %s", e)

        # Batch commit once per block (not per event)
        try:
            self.conn.commit()
        except Exception:
            pass
