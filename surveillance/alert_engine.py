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


# Epistemic classification per alert type.
# Added per COMMERCIAL_EPISTEMIC_AUDIT.md (2026-04-09).
# provable  = trigger is an on-chain event, independently verifiable
# assessed  = trigger involves behavioral interpretation or classification
# predictive = trigger involves forward-looking inference
ALERT_EPISTEMIC_TAGS = {
    "TRAP_CONFIRMED":           "provable",   # external revert with tx_hash
    "HIGH_VELOCITY_DEPLOYER":   "provable",   # N deployments counted on-chain
    "WATCHLIST_HIT":            "provable",   # deterministic address match
    "BRIDGE_WITHDRAWAL":        "provable",   # direct bridge call with value
    "X402_AGENT_DRAIN":         "assessed",   # Permit2 call decoded + rogue classification
    "X402_FACILITATOR_UNKNOWN": "assessed",   # unregistered address calling x402 selectors
    "X402_PERMIT2_EXPOSURE":    "assessed",   # new allowance on monitored token
    "X402_TRUST_AMPLIFICATION": "assessed",   # amplification factor vs baseline
    "DORMANT_ACTIVATION":       "assessed",   # first interaction != malicious activation
    "APPROVAL_DRAIN":           "assessed",   # temporal sequence interpreted as drain
    "SELF_LOOP_TRAP":           "provable",   # deployer called own contract (on-chain)
    "SUSPECTED_HIGH_TRAFFIC":   "assessed",   # suspected contract with 50+ distinct callers
    "COORDINATED_DEPLOYMENT":   "assessed",   # 3+ deployers deploy identical bytecode within 1 hour
}


def insert_alert(conn: sqlite3.Connection, *,
                 alert_type: str,
                 address: str,
                 tx_hash: str = None,
                 block_number: int = None,
                 timestamp: str = None,
                 payload: dict = None) -> None:
    """Insert an alert into the alerts table.

    Automatically injects epistemic_tag into the payload so feed
    consumers can filter by provability level.
    """
    if payload is None:
        payload = {}
    payload["epistemic_tag"] = ALERT_EPISTEMIC_TAGS.get(alert_type, "assessed")

    conn.execute(
        """INSERT INTO alerts (alert_type, address, tx_hash, block_number, timestamp, payload)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (alert_type, address, tx_hash, block_number,
         timestamp or _now_iso(), json.dumps(payload)),
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
            "message": f"Contract {contract_address} confirmed as trap. "
                       f"Bot {bot_address} reverted in tx {tx_hash}.",
        },
    )
    logger.info("ALERT: TRAP_CONFIRMED %s chain=%s", contract_address, chain)


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
            "message": f"Deployer {deployer_address} deployed {contract_count} "
                       f"contracts (threshold: {threshold}) on {chain}",
        },
    )
    logger.info("ALERT: HIGH_VELOCITY_DEPLOYER %s count=%d", deployer_address, contract_count)


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


def alert_coordinated_deployment(conn: sqlite3.Connection, *,
                                  deployed_code_hash: str,
                                  deployer_count: int,
                                  contract_count: int,
                                  deployers: list[str],
                                  time_window: str = "1 hour",
                                  trigger_contract: str,
                                  chain: str = None,
                                  timestamp: str = None) -> None:
    """Alert when 3+ deployers deploy identical bytecode within 1 hour (Sybil pattern)."""
    insert_alert(
        conn,
        alert_type="COORDINATED_DEPLOYMENT",
        address=trigger_contract,
        timestamp=timestamp or _now_iso(),
        payload={
            "deployed_code_hash": deployed_code_hash,
            "deployer_count": deployer_count,
            "contract_count": contract_count,
            "deployers": deployers,
            "time_window": time_window,
            "trigger_contract": trigger_contract,
            "chain": chain,
            "message": f"Coordinated deployment: {deployer_count} distinct deployers "
                       f"deployed {contract_count} contracts with identical bytecode "
                       f"({deployed_code_hash[:16]}...) within {time_window} on {chain}",
        },
    )
    logger.warning(
        "ALERT: COORDINATED_DEPLOYMENT hash=%s deployers=%d contracts=%d chain=%s",
        deployed_code_hash[:16], deployer_count, contract_count, chain,
    )
