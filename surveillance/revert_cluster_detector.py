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

        revert_counts: dict[str, int] = defaultdict(int)
        # Collect selectors per sender for reverted txs
        revert_selectors: dict[str, list[str]] = defaultdict(list)
        # Collect target addresses per sender for reverted txs (for self-loop detection)
        revert_targets: dict[str, list[str]] = defaultdict(list)

        for tx in block["transactions"]:
            from_addr = tx["from"]
            from_lower = from_addr.lower() if isinstance(from_addr, str) else from_addr
            if from_lower not in heavy_senders:
                continue

            try:
                receipt = await w3.eth.get_transaction_receipt(tx["hash"])
                if receipt["status"] == 0:
                    revert_counts[from_lower] += 1
                    # Extract function selector from calldata
                    input_data = tx.get("input", b"")
                    if isinstance(input_data, bytes):
                        calldata_hex = input_data.hex()
                    else:
                        calldata_hex = str(input_data).replace("0x", "")
                    selector = calldata_hex[:8] if len(calldata_hex) >= 8 else "00000000"
                    revert_selectors[from_lower].append(selector)
                    to_addr = tx.get("to")
                    if to_addr:
                        to_lower = to_addr.lower() if isinstance(to_addr, str) else to_addr
                        revert_targets[from_lower].append(to_lower)
            except Exception:
                continue

        # Flag addresses exceeding threshold
        for addr, count in revert_counts.items():
            if count >= self.threshold:
                self._flag_candidate(
                    addr, block_number, count, timestamp_iso,
                    revert_selectors.get(addr, []),
                    revert_targets.get(addr, []),
                )

    def _flag_candidate(self, address: str, block_number: int,
                        revert_count: int, timestamp_iso: str,
                        selectors: list[str] | None = None,
                        targets: list[str] | None = None) -> None:
        """Record a bot candidate, store selectors, check deployer cross-reference."""
        try:
            db.upsert_bot_candidate(
                self.conn, address, block_number, revert_count, timestamp_iso
            )

            # Store each reverted selector
            for sel in (selectors or []):
                db.upsert_bot_selector(self.conn, address, sel, timestamp_iso)
            self.conn.commit()
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
                self._check_self_loop(address, block_number, targets or [])
            else:
                logger.info(
                    "BOT CANDIDATE: %s produced %d reverts in block %d",
                    address, revert_count, block_number,
                )
        except Exception as e:
            logger.error("Failed to flag bot candidate %s: %s", address, e)

    def _check_self_loop(self, address: str, block_number: int,
                         targets: list[str]) -> None:
        """
        Promote contracts to CONFIRMED when a BOT+DEPLOYER is observed
        repeatedly calling contracts they themselves deployed.

        This catches self-deploying trap operators who use a liveness pump
        (calling their own trap with a fixed selector) to keep the contract
        looking active to routers/scanners. The bot's own reverts against
        their own deployment are proof of operator intent.
        """
        if not targets:
            return
        unique_targets = set(targets)
        try:
            placeholders = ",".join("?" * len(unique_targets))
            rows = self.conn.execute(
                f"""
                SELECT contract_address, confidence_tier
                FROM contracts
                WHERE deployer_address = ?
                  AND contract_address IN ({placeholders})
                """,
                (address, *unique_targets),
            ).fetchall()
            for contract_addr, tier in rows:
                if tier == "confirmed":
                    continue
                reason = (
                    f"Self-loop: deployer {address} called own contract "
                    f"in reverted tx at block {block_number} (BOT+DEPLOYER)"
                )
                db.update_contract_confidence(
                    self.conn, contract_addr, "confirmed", reason
                )
                logger.warning(
                    "SELF_LOOP_TRAP: promoted %s to CONFIRMED "
                    "(self-deployed and self-called by %s)",
                    contract_addr, address,
                )
        except Exception as e:
            logger.error("Self-loop check failed for %s: %s", address, e)
