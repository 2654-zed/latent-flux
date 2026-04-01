"""
Layer 3 — Behavioral Confirmation

Automatically detects trap events by identifying when non-deployer
addresses interact with suspected contracts and get reverted.
This is the missing link between "suspected" and "confirmed" tiers.

Integrates into SelectorMonitor._process_interaction() — when a revert
is detected on a watched contract, this module checks whether the caller
is the deployer (self-test) or an external victim (trap fire).
"""

import logging
import sqlite3
from typing import Optional

from surveillance import db

logger = logging.getLogger("surveillance.behavioral_confirm")


class BehavioralConfirmation:
    """
    Watches reverted transactions on suspected/confirmed contracts.
    When a non-deployer address gets reverted, records a trap event
    and upgrades the contract to CONFIRMED.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        # Cache: contract_address -> deployer_address
        self._deployer_cache: dict[str, str] = {}
        # Cache: contract_address -> confidence_tier
        self._tier_cache: dict[str, str] = {}
        # Track tx hashes already processed to avoid duplicates
        self._seen_tx: set[str] = set()
        # Stats
        self.trap_events_created = 0
        self.self_tests_skipped = 0
        # Keep seen_tx bounded
        self._max_seen = 50000

    def _get_deployer(self, contract_address: str) -> Optional[str]:
        """Get deployer for a contract, with caching."""
        if contract_address in self._deployer_cache:
            return self._deployer_cache[contract_address]
        row = self.conn.execute(
            "SELECT deployer_address FROM contracts WHERE contract_address = ?",
            (contract_address,),
        ).fetchone()
        deployer = row[0] if row else None
        self._deployer_cache[contract_address] = deployer
        return deployer

    def _get_tier(self, contract_address: str) -> Optional[str]:
        """Get confidence tier for a contract, with caching."""
        if contract_address in self._tier_cache:
            return self._tier_cache[contract_address]
        row = self.conn.execute(
            "SELECT confidence_tier FROM contracts WHERE contract_address = ?",
            (contract_address,),
        ).fetchone()
        tier = row[0] if row else None
        self._tier_cache[contract_address] = tier
        return tier

    def check_revert(self, *,
                     contract_address: str,
                     caller_address: str,
                     tx_hash: str,
                     block_number: int,
                     timestamp: str,
                     function_selector: Optional[str] = None) -> bool:
        """
        Called when a reverted transaction is detected on a watched contract.
        Returns True if a trap event was created.
        """
        # Deduplicate
        if tx_hash in self._seen_tx:
            return False
        self._seen_tx.add(tx_hash)

        # Bound the seen set
        if len(self._seen_tx) > self._max_seen:
            # Drop oldest half (set doesn't preserve order, but that's fine)
            to_keep = set(list(self._seen_tx)[self._max_seen // 2:])
            self._seen_tx = to_keep

        # Only act on suspected contracts (confirmed already proven)
        tier = self._get_tier(contract_address)
        if tier != "suspected":
            return False

        # Check if caller is the deployer (self-test)
        deployer = self._get_deployer(contract_address)
        if deployer and caller_address.lower() == deployer.lower():
            self.self_tests_skipped += 1
            logger.debug(
                "Self-test skipped: deployer %s testing %s tx=%s",
                deployer[:18], contract_address[:18], tx_hash[:18],
            )
            return False

        # This is a genuine trap fire — external address got reverted
        failure_sig = f"revert:selector={function_selector or 'unknown'}"

        try:
            db.insert_trap_event(
                self.conn,
                trap_contract_address=contract_address,
                bot_address=caller_address,
                tx_hash=tx_hash,
                block_number=block_number,
                timestamp=timestamp,
                failure_signature=failure_sig,
                loss_estimate_usd=None,
            )
            self.trap_events_created += 1

            # Invalidate tier cache for this contract (now confirmed)
            self._tier_cache[contract_address] = "confirmed"

            logger.warning(
                "TRAP CONFIRMED: %s trapped %s in tx %s (selector=%s) — upgraded to confirmed",
                contract_address[:18] + "...",
                caller_address[:18] + "...",
                tx_hash[:18] + "...",
                function_selector or "?",
            )
            return True

        except Exception as e:
            logger.error("Failed to insert trap event: %s", e)
            return False

    def refresh_caches(self) -> None:
        """Clear caches periodically to pick up new data."""
        self._deployer_cache.clear()
        self._tier_cache.clear()
