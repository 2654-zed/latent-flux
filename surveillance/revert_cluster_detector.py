"""
Layer 3 -- Revert Cluster Detector

Tracks addresses that produce >5 reverted transactions in a single block.
These are likely active bot addresses — bots spray transactions and eat
reverts as a cost of doing business.

Cross-references against deployer addresses — a bot that is also a
deployer is a significant finding (self-deploying trap operator).

Runs inline during block processing, no extra RPC calls needed
beyond the receipt fetches already being done.
"""

import logging
import sqlite3
from collections import defaultdict
from typing import Optional

from web3 import AsyncWeb3

from surveillance import db

logger = logging.getLogger("surveillance.revert_detector")

# Minimum reverts in a single block to flag as bot candidate
REVERT_THRESHOLD = 5


class RevertClusterDetector:
    """
    Scans blocks for addresses with high revert counts.
    Call process_block() for each block with full transactions + receipts.
    """

    def __init__(self, conn: sqlite3.Connection, threshold: int = REVERT_THRESHOLD):
        self.conn = conn
        self.threshold = threshold
        self._candidates_found = 0
        self._deployer_hits = 0

    @property
    def candidates_found(self) -> int:
        return self._candidates_found

    @property
    def deployer_hits(self) -> int:
        return self._deployer_hits

    async def process_block(self, w3: AsyncWeb3, block: dict,
                            timestamp_iso: str) -> None:
        """
        Count reverted transactions per sender in this block.
        Flag addresses exceeding the threshold.
        """
        block_number = block["number"]
        revert_counts: dict[str, int] = defaultdict(int)

        for tx in block["transactions"]:
            from_addr = tx["from"]
            if isinstance(from_addr, str):
                from_lower = from_addr.lower()
            else:
                from_lower = from_addr

            # We need receipt to check revert status.
            # To avoid extra RPC calls, only check transactions that
            # look like bot activity: low gas, many txs from same sender.
            # First pass: count txs per sender in block.
            revert_counts.setdefault(from_lower, 0)

        # Count senders with multiple txs (potential bots spray txs)
        sender_tx_counts: dict[str, int] = defaultdict(int)
        for tx in block["transactions"]:
            from_addr = tx["from"]
            from_lower = from_addr.lower() if isinstance(from_addr, str) else from_addr
            sender_tx_counts[from_lower] += 1

        # Only fetch receipts for senders with 3+ txs in the block
        # (optimization: avoids receipt calls for regular users)
        heavy_senders = {addr for addr, count in sender_tx_counts.items() if count >= 3}
        if not heavy_senders:
            return

        revert_counts = defaultdict(int)
        for tx in block["transactions"]:
            from_addr = tx["from"]
            from_lower = from_addr.lower() if isinstance(from_addr, str) else from_addr
            if from_lower not in heavy_senders:
                continue

            try:
                receipt = await w3.eth.get_transaction_receipt(tx["hash"])
                if receipt["status"] == 0:
                    revert_counts[from_lower] += 1
            except Exception:
                continue

        # Flag addresses exceeding threshold
        for addr, count in revert_counts.items():
            if count >= self.threshold:
                self._flag_candidate(addr, block_number, count, timestamp_iso)

    def _flag_candidate(self, address: str, block_number: int,
                        revert_count: int, timestamp_iso: str) -> None:
        """Record a bot candidate and check deployer cross-reference."""
        try:
            db.upsert_bot_candidate(
                self.conn, address, block_number, revert_count, timestamp_iso
            )
            self._candidates_found += 1

            # Check deployer cross-reference
            candidate = self.conn.execute(
                "SELECT is_deployer FROM bot_candidates WHERE address = ?",
                (address,),
            ).fetchone()

            if candidate and candidate[0]:
                self._deployer_hits += 1
                logger.warning(
                    "BOT+DEPLOYER: %s produced %d reverts in block %d "
                    "AND is a known deployer -- possible self-deploying trap operator",
                    address, revert_count, block_number,
                )
            else:
                logger.info(
                    "BOT CANDIDATE: %s produced %d reverts in block %d",
                    address, revert_count, block_number,
                )
        except Exception as e:
            logger.error("Failed to flag bot candidate %s: %s", address, e)
