"""
Layer 3 — SQLite database access layer.

Handles schema initialization, upsert ordering (deployer before contract),
and all read/write operations for the surveillance system.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "surveillance.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create or open the database and apply schema if tables don't exist."""
    db_path = db_path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    # Check if schema is already applied
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='contracts'"
    )
    if cursor.fetchone() is None:
        # Read and execute schema DDL (skip PRAGMA lines — already set above)
        ddl = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(ddl)

    # Migration: add bytecode_cache table if missing (existing DBs from pre-cache sessions)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='bytecode_cache'"
    )
    if cursor.fetchone() is None:
        conn.executescript("""
            CREATE TABLE bytecode_cache (
                code_hash           TEXT    NOT NULL PRIMARY KEY,
                confidence_tier     TEXT    NOT NULL,
                confidence_reason   TEXT    NOT NULL,
                bytecode_signals    TEXT    NOT NULL DEFAULT '{}',
                source_contract     TEXT    NOT NULL,
                hit_count           INTEGER NOT NULL DEFAULT 0,
                created             TEXT    NOT NULL
            );
        """)

    # Migration: add heartbeat + connection_gaps tables
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='heartbeat'"
    )
    if cursor.fetchone() is None:
        conn.executescript("""
            CREATE TABLE heartbeat (
                component   TEXT NOT NULL PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                blocks      INTEGER NOT NULL DEFAULT 0,
                deployments INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE connection_gaps (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                component   TEXT    NOT NULL,
                disconnect  TEXT    NOT NULL,
                reconnect   TEXT,
                last_block  INTEGER,
                first_block INTEGER,
                reason      TEXT
            );
        """)

    return conn


# ------------------------------------------------------------------
# Deployer operations
# ------------------------------------------------------------------

def ensure_deployer(conn: sqlite3.Connection, deployer_address: str,
                    timestamp_iso: str, chain: str = "arbitrum") -> None:
    """
    Upsert a minimal deployer row. Called BEFORE inserting a contract
    to satisfy the FK constraint. If the deployer already exists,
    updates last_seen and increments total_contracts_deployed.
    """
    conn.execute(
        """
        INSERT INTO deployers (deployer_address, chain, first_seen, last_seen,
                               total_contracts_deployed)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(deployer_address) DO UPDATE SET
            last_seen = ?,
            total_contracts_deployed = total_contracts_deployed + 1
        """,
        (deployer_address, chain, timestamp_iso, timestamp_iso, timestamp_iso),
    )


def get_deployer(conn: sqlite3.Connection, address: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM deployers WHERE deployer_address = ?", (address,)
    ).fetchone()
    return dict(row) if row else None


def deployer_confirmed_count(conn: sqlite3.Connection, address: str) -> int:
    """Derived count — no denormalized field."""
    row = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE deployer_address = ? AND confidence_tier = 'confirmed'",
        (address,),
    ).fetchone()
    return row[0]


def deployer_suspected_count(conn: sqlite3.Connection, address: str) -> int:
    """Derived count — no denormalized field."""
    row = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE deployer_address = ? AND confidence_tier = 'suspected'",
        (address,),
    ).fetchone()
    return row[0]


def update_deployer_notes(conn: sqlite3.Connection, address: str,
                          notes: str) -> None:
    conn.execute(
        "UPDATE deployers SET deployment_pattern_notes = ? WHERE deployer_address = ?",
        (notes, address),
    )
    conn.commit()


def is_priority_deployer(conn: sqlite3.Connection, address: str) -> bool:
    """A deployer is priority if they have pattern notes set (flagged by analyst)."""
    row = conn.execute(
        "SELECT deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
        (address,),
    ).fetchone()
    return row is not None and row[0] is not None and len(row[0]) > 0


def get_returning_deployers(conn: sqlite3.Connection,
                            since_block: int) -> list[dict]:
    """
    Deployers who deployed contracts BOTH before and after since_block.
    Returns deployers with counts from each period.
    """
    rows = conn.execute(
        """
        SELECT d.deployer_address,
               d.deployment_pattern_notes,
               SUM(CASE WHEN c.detection_block < ? THEN 1 ELSE 0 END) as prev_count,
               SUM(CASE WHEN c.detection_block >= ? THEN 1 ELSE 0 END) as new_count
        FROM deployers d
        JOIN contracts c ON c.deployer_address = d.deployer_address
        GROUP BY d.deployer_address
        HAVING prev_count > 0 AND new_count > 0
        ORDER BY new_count DESC
        """,
        (since_block, since_block),
    ).fetchall()
    return [dict(r) for r in rows]


def get_max_block(conn: sqlite3.Connection) -> int:
    """Highest detection_block in the contracts table (session boundary marker)."""
    row = conn.execute("SELECT MAX(detection_block) FROM contracts").fetchone()
    return row[0] if row[0] is not None else 0


# ------------------------------------------------------------------
# Contract operations
# ------------------------------------------------------------------

def insert_contract(conn: sqlite3.Connection, *,
                    contract_address: str,
                    deployer_address: str,
                    detection_method: str,
                    detection_timestamp: str,
                    detection_block: int,
                    confidence_tier: str,
                    confidence_reason: str,
                    chain: str = "arbitrum",
                    **kwargs) -> None:
    """
    Insert a contract record. Automatically ensures the deployer row
    exists first (FK ordering constraint).

    Both the deployer upsert and contract insert are wrapped in a single
    transaction — if the contract INSERT fails (e.g., CHECK constraint
    on confidence_reason), the deployer increment rolls back too.
    """
    try:
        # Step 1: ensure deployer exists (inside implicit transaction)
        ensure_deployer(conn, deployer_address, detection_timestamp, chain)

        # Step 2: insert contract
        # Plain INSERT — not INSERT OR IGNORE. OR IGNORE silently swallows
        # ALL constraint violations including the confidence_reason CHECK.
        # Duplicate contract_address is a logic error that should surface.
        # The caller must check for existence before calling if re-insertion
        # is possible (e.g., during block replay).
        conn.execute(
            """
            INSERT INTO contracts (
                contract_address, chain, detection_method, detection_timestamp,
                detection_block, confidence_tier, confidence_reason,
                deployer_address, deployer_funding_source,
                routing_presence, routing_first_seen,
                has_asymmetric_transfer, has_conditional_revert,
                has_unusual_fee_structure, bytecode_pattern_notes,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contract_address, chain, detection_method, detection_timestamp,
                detection_block, confidence_tier, confidence_reason,
                deployer_address,
                kwargs.get("deployer_funding_source"),
                kwargs.get("routing_presence", 0),
                kwargs.get("routing_first_seen"),
                kwargs.get("has_asymmetric_transfer"),
                kwargs.get("has_conditional_revert"),
                kwargs.get("has_unusual_fee_structure"),
                kwargs.get("bytecode_pattern_notes"),
                _now_iso(),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def update_contract_confidence(conn: sqlite3.Connection, contract_address: str,
                               confidence_tier: str, confidence_reason: str) -> None:
    conn.execute(
        """
        UPDATE contracts SET confidence_tier = ?, confidence_reason = ?, last_updated = ?
        WHERE contract_address = ?
        """,
        (confidence_tier, confidence_reason, _now_iso(), contract_address),
    )
    conn.commit()


def update_contract_routing(conn: sqlite3.Connection, contract_address: str,
                            routing_first_seen: str) -> None:
    conn.execute(
        """
        UPDATE contracts SET routing_presence = 1, routing_first_seen = ?, last_updated = ?
        WHERE contract_address = ?
        """,
        (routing_first_seen, _now_iso(), contract_address),
    )
    conn.commit()


def update_contract_bytecode_signals(conn: sqlite3.Connection, contract_address: str,
                                     *, has_asymmetric_transfer: Optional[bool] = None,
                                     has_conditional_revert: Optional[bool] = None,
                                     has_unusual_fee_structure: Optional[bool] = None,
                                     bytecode_pattern_notes: Optional[str] = None) -> None:
    conn.execute(
        """
        UPDATE contracts SET
            has_asymmetric_transfer = COALESCE(?, has_asymmetric_transfer),
            has_conditional_revert = COALESCE(?, has_conditional_revert),
            has_unusual_fee_structure = COALESCE(?, has_unusual_fee_structure),
            bytecode_pattern_notes = COALESCE(?, bytecode_pattern_notes),
            last_updated = ?
        WHERE contract_address = ?
        """,
        (
            int(has_asymmetric_transfer) if has_asymmetric_transfer is not None else None,
            int(has_conditional_revert) if has_conditional_revert is not None else None,
            int(has_unusual_fee_structure) if has_unusual_fee_structure is not None else None,
            bytecode_pattern_notes,
            _now_iso(),
            contract_address,
        ),
    )
    conn.commit()


def get_contract(conn: sqlite3.Connection, address: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM contracts WHERE contract_address = ?", (address,)
    ).fetchone()
    return dict(row) if row else None


def get_contracts_by_tier(conn: sqlite3.Connection, tier: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM contracts WHERE confidence_tier = ? ORDER BY detection_timestamp DESC",
        (tier,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_contracts_by_deployer(conn: sqlite3.Connection, deployer: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM contracts WHERE deployer_address = ? ORDER BY detection_timestamp DESC",
        (deployer,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_contract_addresses(conn: sqlite3.Connection) -> set[str]:
    """Return all known contract addresses as a set (for fast membership checks)."""
    rows = conn.execute("SELECT contract_address FROM contracts").fetchall()
    return {r[0] for r in rows}


def get_contracts_without_routing(conn: sqlite3.Connection) -> list[dict]:
    """Contracts that haven't been checked against 1inch routing yet."""
    rows = conn.execute(
        "SELECT * FROM contracts WHERE routing_presence = 0 ORDER BY detection_timestamp DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_suspected_contracts(conn: sqlite3.Connection) -> list[dict]:
    """All SUSPECTED contracts — priority targets for routing anomaly checks."""
    rows = conn.execute(
        "SELECT * FROM contracts WHERE confidence_tier = 'suspected' ORDER BY detection_timestamp DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Trap event operations
# ------------------------------------------------------------------

def insert_trap_event(conn: sqlite3.Connection, *,
                      trap_contract_address: str,
                      bot_address: str,
                      tx_hash: str,
                      block_number: int,
                      timestamp: str,
                      failure_signature: str,
                      loss_estimate_usd: Optional[float] = None) -> None:
    """Insert a trap event and upgrade the contract to CONFIRMED."""
    conn.execute(
        """
        INSERT OR IGNORE INTO trap_events (
            trap_contract_address, bot_address, tx_hash, block_number,
            timestamp, loss_estimate_usd, failure_signature
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (trap_contract_address, bot_address, tx_hash, block_number,
         timestamp, loss_estimate_usd, failure_signature),
    )

    # Auto-upgrade contract to confirmed
    conn.execute(
        """
        UPDATE contracts SET
            confidence_tier = 'confirmed',
            confidence_reason = 'Behavioral confirmation: bot ' || ? || ' trapped in tx ' || ?,
            confirmation_tx_hash = ?,
            confirmation_timestamp = ?,
            confirmation_block = ?,
            last_updated = ?
        WHERE contract_address = ?
        AND confidence_tier != 'confirmed'
        """,
        (bot_address, tx_hash, tx_hash, timestamp, block_number,
         _now_iso(), trap_contract_address),
    )
    conn.commit()


def get_trap_events_for_contract(conn: sqlite3.Connection, address: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM trap_events WHERE trap_contract_address = ? ORDER BY block_number",
        (address,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_most_trapped_bots(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Which bot addresses get trapped most frequently?"""
    rows = conn.execute(
        """
        SELECT bot_address, COUNT(*) as trap_count,
               SUM(COALESCE(loss_estimate_usd, 0)) as total_loss_usd
        FROM trap_events
        GROUP BY bot_address
        ORDER BY trap_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Bytecode cache operations
# ------------------------------------------------------------------

def cache_lookup(conn: sqlite3.Connection, code_hash: str) -> Optional[dict]:
    """Look up a classification by code hash. Returns None on miss."""
    row = conn.execute(
        "SELECT * FROM bytecode_cache WHERE code_hash = ?", (code_hash,)
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE bytecode_cache SET hit_count = hit_count + 1 WHERE code_hash = ?",
        (code_hash,),
    )
    result = dict(row)
    result["bytecode_signals"] = json.loads(result["bytecode_signals"])
    return result


def cache_store(conn: sqlite3.Connection, code_hash: str,
                confidence_tier: str, confidence_reason: str,
                bytecode_signals: dict, source_contract: str) -> None:
    """Store a classification result keyed by code hash."""
    conn.execute(
        """
        INSERT OR IGNORE INTO bytecode_cache
            (code_hash, confidence_tier, confidence_reason,
             bytecode_signals, source_contract, hit_count, created)
        VALUES (?, ?, ?, ?, ?, 0, ?)
        """,
        (code_hash, confidence_tier, confidence_reason,
         json.dumps(bytecode_signals), source_contract, _now_iso()),
    )
    conn.commit()


def cache_stats(conn: sqlite3.Connection) -> dict:
    """Cache summary: total entries, total hits."""
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM bytecode_cache"
    ).fetchone()
    return {"entries": row[0], "total_hits": row[1]}


# ------------------------------------------------------------------
# Heartbeat + connection gap operations
# ------------------------------------------------------------------

def write_heartbeat(conn: sqlite3.Connection, component: str,
                    blocks: int = 0, deployments: int = 0) -> None:
    conn.execute(
        """
        INSERT INTO heartbeat (component, timestamp, blocks, deployments)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(component) DO UPDATE SET
            timestamp = ?, blocks = ?, deployments = ?
        """,
        (component, _now_iso(), blocks, deployments,
         _now_iso(), blocks, deployments),
    )
    conn.commit()


def read_heartbeat(conn: sqlite3.Connection, component: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM heartbeat WHERE component = ?", (component,)
    ).fetchone()
    return dict(row) if row else None


def read_all_heartbeats(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM heartbeat").fetchall()
    return [dict(r) for r in rows]


def log_connection_gap(conn: sqlite3.Connection, component: str,
                       disconnect_time: str, last_block: Optional[int],
                       reason: str) -> int:
    """Log a disconnect. Returns the gap ID for updating on reconnect."""
    cursor = conn.execute(
        """
        INSERT INTO connection_gaps (component, disconnect, last_block, reason)
        VALUES (?, ?, ?, ?)
        """,
        (component, disconnect_time, last_block, reason),
    )
    conn.commit()
    return cursor.lastrowid


def close_connection_gap(conn: sqlite3.Connection, gap_id: int,
                         reconnect_time: str, first_block: Optional[int]) -> None:
    conn.execute(
        "UPDATE connection_gaps SET reconnect = ?, first_block = ? WHERE id = ?",
        (reconnect_time, first_block, gap_id),
    )
    conn.commit()


def get_connection_gaps(conn: sqlite3.Connection,
                        since: Optional[str] = None) -> list[dict]:
    if since:
        rows = conn.execute(
            "SELECT * FROM connection_gaps WHERE disconnect >= ? ORDER BY disconnect",
            (since,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM connection_gaps ORDER BY disconnect"
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Stats (all derived, no denormalized fields)
# ------------------------------------------------------------------

def stats(conn: sqlite3.Connection) -> dict:
    """Summary statistics for the surveillance database."""
    contracts_total = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    contracts_by_tier = {}
    for tier in ("unknown", "suspected", "confirmed"):
        contracts_by_tier[tier] = conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE confidence_tier = ?", (tier,)
        ).fetchone()[0]

    deployers_total = conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
    events_total = conn.execute("SELECT COUNT(*) FROM trap_events").fetchone()[0]

    return {
        "contracts_total": contracts_total,
        "contracts_by_tier": contracts_by_tier,
        "deployers_total": deployers_total,
        "trap_events_total": events_total,
    }
