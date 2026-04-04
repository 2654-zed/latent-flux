"""
Layer 3 -- Time-Lock Countdown Monitor

For TIMESTAMP-gated contracts, extracts activation timestamps from bytecode
analysis notes and maintains a countdown table. Alerts when any contract is
within 24 hours of its activation time.

Runs daily from the deployment monitor heartbeat loop.
Zero API calls -- pure SQLite + bytecode pattern parsing.
"""

import logging
import re
import sqlite3
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("surveillance.timelock_monitor")


def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS timelock_countdowns (
            contract_address TEXT PRIMARY KEY,
            chain TEXT,
            deployer_address TEXT,
            confidence_tier TEXT,
            activation_block INTEGER,
            activation_timestamp INTEGER,
            activation_iso TEXT,
            bytecode_evidence TEXT,
            status TEXT DEFAULT 'PENDING',
            detected_at TEXT,
            alerted_at TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timelock_status
        ON timelock_countdowns(status)
    """)


def scan_for_timelocks(conn: sqlite3.Connection) -> dict:
    """Find TIMESTAMP-gated contracts and extract activation times.

    Parses bytecode_pattern_notes for TIMESTAMP comparisons and attempts
    to extract the comparison value as an activation time.
    """
    ensure_table(conn)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    now_ts = int(now.timestamp())

    # Find suspected/confirmed contracts with TIMESTAMP in bytecode notes
    # that aren't already in the countdown table
    candidates = conn.execute("""
        SELECT c.contract_address, c.chain, c.deployer_address, c.confidence_tier,
               c.bytecode_pattern_notes, c.detection_block
        FROM contracts c
        WHERE c.bytecode_pattern_notes LIKE '%TIMESTAMP%'
        AND c.confidence_tier IN ('suspected', 'confirmed')
        AND c.contract_address NOT IN (SELECT contract_address FROM timelock_countdowns)
    """).fetchall()

    new_timelocks = 0
    for c in candidates:
        notes = c["bytecode_pattern_notes"] or ""

        # Extract activation_timestamp=NNNN from enhanced bytecode notes
        # The classifier now embeds the PUSH value: "activation_timestamp=1712345678 (2026-04-05 ...)"
        ts_match = re.search(r'activation_timestamp=(\d+)', notes)
        block_match = re.search(r'activation_block=(\d+)', notes)

        activation_ts = None
        activation_block = None

        if ts_match:
            ts = int(ts_match.group(1))
            if 1_700_000_000 <= ts <= 1_900_000_000:
                activation_ts = ts

        if block_match:
            activation_block = int(block_match.group(1))

        # Fallback: scan notes for any 10-digit number in valid timestamp range
        if not activation_ts and not block_match:
            for m in re.findall(r'\b(1[7-8][0-9]{8})\b', notes):
                ts = int(m)
                if now_ts - 86400 * 90 < ts < now_ts + 86400 * 365:
                    activation_ts = ts
                    break

        if activation_ts:
            act_iso = datetime.fromtimestamp(activation_ts, tz=timezone.utc).isoformat()
            status = "PENDING" if activation_ts > now_ts else "ACTIVATED"

            conn.execute("""
                INSERT OR IGNORE INTO timelock_countdowns
                (contract_address, chain, deployer_address, confidence_tier,
                 activation_timestamp, activation_iso, bytecode_evidence,
                 status, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (c["contract_address"], c["chain"], c["deployer_address"],
                  c["confidence_tier"], activation_ts, act_iso,
                  notes[:500], status, now_iso))
            new_timelocks += 1
        else:
            # Record as time-gated but without extractable activation time
            conn.execute("""
                INSERT OR IGNORE INTO timelock_countdowns
                (contract_address, chain, deployer_address, confidence_tier,
                 bytecode_evidence, status, detected_at)
                VALUES (?, ?, ?, ?, ?, 'UNKNOWN_TIME', ?)
            """, (c["contract_address"], c["chain"], c["deployer_address"],
                  c["confidence_tier"], notes[:500], now_iso))

    return {"new_timelocks": new_timelocks, "candidates_scanned": len(candidates)}


def check_countdowns(conn: sqlite3.Connection) -> dict:
    """Check for contracts within 24 hours of activation. Generate alerts."""
    ensure_table(conn)
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())
    now_iso = now.isoformat()
    threshold = now_ts + 86400  # 24 hours from now

    # Find PENDING contracts with activation within 24 hours
    imminent = conn.execute("""
        SELECT * FROM timelock_countdowns
        WHERE status = 'PENDING'
        AND activation_timestamp IS NOT NULL
        AND activation_timestamp <= ?
        AND activation_timestamp > ?
        AND alerted_at IS NULL
    """, (threshold, now_ts - 3600)).fetchall()  # Include recently activated (1hr grace)

    alerts = 0
    for tl in imminent:
        hours_remaining = (tl["activation_timestamp"] - now_ts) / 3600
        already_active = hours_remaining < 0

        # Generate alert
        try:
            from surveillance.alert_engine import insert_alert
            insert_alert(
                conn,
                alert_type="TIMELOCK_IMMINENT",
                address=tl["contract_address"],
                timestamp=now_iso,
                payload={
                    "contract": tl["contract_address"],
                    "chain": tl["chain"],
                    "deployer": tl["deployer_address"],
                    "tier": tl["confidence_tier"],
                    "activation_time": tl["activation_iso"],
                    "hours_remaining": round(hours_remaining, 1),
                    "status": "ACTIVATED" if already_active else "IMMINENT",
                    "message": (
                        f"Time-locked contract {tl['contract_address'][:18]}... "
                        f"{'ACTIVATED' if already_active else f'activates in {hours_remaining:.1f}h'}. "
                        f"Tier: {tl['confidence_tier']}"
                    ),
                },
            )
            alerts += 1
        except Exception as e:
            logger.debug("Alert failed: %s", e)

        # Mark as alerted
        conn.execute("""
            UPDATE timelock_countdowns SET alerted_at = ?,
            status = CASE WHEN activation_timestamp <= ? THEN 'ACTIVATED' ELSE status END
            WHERE contract_address = ?
        """, (now_iso, now_ts, tl["contract_address"]))

        logger.warning(
            "TIMELOCK %s: %s tier=%s %s",
            "ACTIVATED" if already_active else "IMMINENT",
            tl["contract_address"][:18],
            tl["confidence_tier"],
            f"activated" if already_active else f"in {hours_remaining:.1f}h",
        )

    # Also update status of past-due contracts
    conn.execute("""
        UPDATE timelock_countdowns SET status = 'ACTIVATED'
        WHERE status = 'PENDING' AND activation_timestamp IS NOT NULL
        AND activation_timestamp <= ?
    """, (now_ts,))

    return {"imminent_alerts": alerts}
