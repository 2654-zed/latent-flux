"""
Layer 3 -- Dormant Activation Monitor

Detects when contracts from dormant fleets (deployers with 10+ contracts,
<10% previously active) receive their first interaction. This is the
"Dragon alert" -- when pre-staged infrastructure starts activating.

Runs every 30 minutes from the deployment monitor heartbeat loop.
Zero API calls -- pure SQLite.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger("surveillance.dormant_activation")


def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dormant_activations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deployer_address TEXT NOT NULL,
            contract_address TEXT NOT NULL,
            chain TEXT,
            fleet_size INTEGER,
            fleet_active_before INTEGER,
            fleet_active_after INTEGER,
            first_interaction_timestamp TEXT,
            first_interacting_address TEXT,
            detected_at TEXT,
            UNIQUE(contract_address)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_dormant_act_deployer
        ON dormant_activations(deployer_address)
    """)


def scan(conn: sqlite3.Connection) -> dict:
    """Check for newly activated contracts in dormant fleets.

    Returns dict with activation count and details.
    """
    ensure_table(conn)
    now = datetime.now(timezone.utc).isoformat()

    # Find dormant fleets: deployers with 10+ contracts
    fleets = conn.execute("""
        SELECT deployer_address, COUNT(*) as fleet_size
        FROM contracts
        GROUP BY deployer_address
        HAVING fleet_size >= 10
    """).fetchall()

    total_new = 0
    activations = []

    for fleet in fleets:
        deployer = fleet["deployer_address"]
        fleet_size = fleet["fleet_size"]

        # Find contracts from this deployer that:
        # 1. Have transaction_events (are active)
        # 2. Are NOT already in dormant_activations (newly active)
        newly_active = conn.execute("""
            SELECT c.contract_address, c.chain,
                   MIN(te.timestamp) as first_tx,
                   (SELECT interacting_address FROM transaction_events
                    WHERE contract_address = c.contract_address
                    ORDER BY timestamp ASC LIMIT 1) as first_caller
            FROM contracts c
            JOIN transaction_events te ON c.contract_address = te.contract_address
            WHERE c.deployer_address = ?
            AND c.contract_address NOT IN (
                SELECT contract_address FROM dormant_activations
            )
            GROUP BY c.contract_address
        """, (deployer,)).fetchall()

        if not newly_active:
            continue

        # How many were previously known as active?
        already_known = conn.execute("""
            SELECT COUNT(*) FROM dormant_activations WHERE deployer_address = ?
        """, (deployer,)).fetchone()[0]

        # Only alert if this is a fleet that was mostly dormant
        # (active count before these new ones was < 10% of fleet)
        if already_known >= fleet_size * 0.5:
            continue  # Already well-known active fleet, not dormant

        for contract in newly_active:
            conn.execute("""
                INSERT OR IGNORE INTO dormant_activations
                (deployer_address, contract_address, chain, fleet_size,
                 fleet_active_before, fleet_active_after,
                 first_interaction_timestamp, first_interacting_address, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (deployer, contract["contract_address"], contract["chain"],
                  fleet_size, already_known, already_known + len(newly_active),
                  contract["first_tx"], contract["first_caller"], now))
            total_new += 1

        if newly_active:
            # Get org attribution
            trail_row = conn.execute(
                "SELECT funding_trail FROM deployers WHERE deployer_address = ?",
                (deployer,),
            ).fetchone()
            org = ""
            if trail_row and trail_row["funding_trail"]:
                try:
                    org = json.loads(trail_row["funding_trail"]).get("org_link", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            activations.append({
                "deployer": deployer,
                "fleet_size": fleet_size,
                "newly_activated": len(newly_active),
                "total_active": already_known + len(newly_active),
                "org": org,
                "contracts": [c["contract_address"] for c in newly_active[:5]],
            })

            # Generate alert
            try:
                from surveillance.alert_engine import insert_alert
                insert_alert(
                    conn,
                    alert_type="DORMANT_ACTIVATION",
                    address=deployer,
                    timestamp=now,
                    payload={
                        "deployer": deployer,
                        "fleet_size": fleet_size,
                        "newly_activated": len(newly_active),
                        "total_active": already_known + len(newly_active),
                        "org": org,
                        "message": (
                            f"Dormant fleet activation: {deployer[:18]}... "
                            f"({len(newly_active)} new of {fleet_size} total). "
                            f"Org: {org or 'unattributed'}"
                        ),
                    },
                )
            except Exception as e:
                logger.debug("Alert generation failed: %s", e)

            logger.warning(
                "DORMANT ACTIVATION: %s fleet=%d newly_active=%d total_active=%d org=%s",
                deployer[:18], fleet_size, len(newly_active),
                already_known + len(newly_active), org or "none",
            )

    return {"new_activations": total_new, "fleets_activated": len(activations),
            "details": activations}
