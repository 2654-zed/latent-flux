"""
Layer 3 — Alert Engine

Generates alerts for significant surveillance events beyond
the original APPROVAL_DRAIN type. Integrates into the existing
deployment monitor and selector monitor flows.

Alert types:
- TRAP_CONFIRMED: A suspected contract behaviorally confirmed as a trap
- HIGH_VELOCITY_DEPLOYER: Deployer exceeds velocity threshold
- WATCHLIST_HIT: Watched entity detected in new deployment
- NEW_BOT_TRAPPED: A previously-unseen bot address got trapped
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger("surveillance.alert_engine")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_alert(conn: sqlite3.Connection, *,
                 alert_type: str,
                 address: str,
                 tx_hash: str = None,
                 block_number: int = None,
                 timestamp: str = None,
                 payload: dict = None) -> None:
    """Insert an alert into the alerts table."""
    conn.execute(
        """INSERT INTO alerts (alert_type, address, tx_hash, block_number, timestamp, payload)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (alert_type, address, tx_hash, block_number,
         timestamp or _now_iso(), json.dumps(payload) if payload else None),
    )


def alert_trap_confirmed(conn: sqlite3.Connection, *,
                         contract_address: str,
                         bot_address: str,
                         tx_hash: str,
                         block_number: int,
                         timestamp: str,
                         chain: str = None) -> None:
    """Alert when a suspected contract is behaviorally confirmed as a trap."""
    insert_alert(
        conn,
        alert_type="TRAP_CONFIRMED",
        address=contract_address,
        tx_hash=tx_hash,
        block_number=block_number,
        timestamp=timestamp,
        payload={
            "contract": contract_address,
            "victim_bot": bot_address,
            "chain": chain,
            "message": f"Contract {contract_address[:18]}... confirmed as trap. "
                       f"Bot {bot_address[:18]}... reverted in tx {tx_hash[:18]}...",
        },
    )
    logger.info("ALERT: TRAP_CONFIRMED %s chain=%s", contract_address[:18], chain)


def alert_high_velocity_deployer(conn: sqlite3.Connection, *,
                                 deployer_address: str,
                                 contract_count: int,
                                 threshold: int,
                                 chain: str = None,
                                 timestamp: str = None) -> None:
    """Alert when a deployer exceeds the velocity threshold."""
    insert_alert(
        conn,
        alert_type="HIGH_VELOCITY_DEPLOYER",
        address=deployer_address,
        timestamp=timestamp or _now_iso(),
        payload={
            "deployer": deployer_address,
            "contracts_this_session": contract_count,
            "threshold": threshold,
            "chain": chain,
            "message": f"Deployer {deployer_address[:18]}... deployed {contract_count} "
                       f"contracts (threshold: {threshold}) on {chain}",
        },
    )
    logger.info("ALERT: HIGH_VELOCITY_DEPLOYER %s count=%d", deployer_address[:18], contract_count)


def alert_watchlist_hit(conn: sqlite3.Connection, *,
                        entity_name: str,
                        hit_type: str,
                        deployer_address: str,
                        contract_address: str,
                        chain: str,
                        priority: str,
                        timestamp: str = None) -> None:
    """Alert when a watched entity is detected."""
    insert_alert(
        conn,
        alert_type="WATCHLIST_HIT",
        address=contract_address,
        timestamp=timestamp or _now_iso(),
        payload={
            "entity": entity_name,
            "hit_type": hit_type,
            "deployer": deployer_address,
            "contract": contract_address,
            "chain": chain,
            "priority": priority,
            "message": f"Watchlist entity '{entity_name}' detected: "
                       f"{hit_type} match on {chain}",
        },
    )
    logger.info("ALERT: WATCHLIST_HIT entity=%s chain=%s priority=%s", entity_name, chain, priority)
