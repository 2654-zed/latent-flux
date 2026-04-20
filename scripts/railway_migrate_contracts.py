"""Run the contracts-table migrations on Railway manually.

Executes the same code path as surveillance.db.init_db() but scoped to just
the contracts table: adds deployed_code_hash, decayed_at, prior_confidence_tier
columns if missing, then rebuilds with the extended CHECK if needed.
"""
import sqlite3
import sys
from pathlib import Path

DB = Path("/app/surveillance/data/surveillance.db")
if not DB.exists():
    print(f"DB not found at {DB}", file=sys.stderr)
    sys.exit(1)

print(f"db: {DB}  size={DB.stat().st_size:,} bytes")
con = sqlite3.connect(str(DB), timeout=120)
con.execute("PRAGMA busy_timeout=60000")

def existing_cols() -> set:
    return {r[1] for r in con.execute("PRAGMA table_info(contracts)").fetchall()}

def ddl() -> str:
    return con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='contracts'").fetchone()[0]

print(f"rows before: {con.execute('SELECT COUNT(*) FROM contracts').fetchone()[0]:,}")
print(f"cols before: {sorted(existing_cols())}")

# Stage 1: add deployed_code_hash if missing
cols = existing_cols()
if "deployed_code_hash" not in cols:
    print("adding deployed_code_hash")
    con.execute("ALTER TABLE contracts ADD COLUMN deployed_code_hash TEXT")
    con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_code_hash ON contracts(deployed_code_hash)")
    con.commit()

# Stage 2: add decayed_at + prior_confidence_tier
cols = existing_cols()
if "decayed_at" not in cols:
    print("adding decayed_at + prior_confidence_tier")
    con.execute("ALTER TABLE contracts ADD COLUMN decayed_at TEXT")
    con.execute("ALTER TABLE contracts ADD COLUMN prior_confidence_tier TEXT")
    con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_decayed_at ON contracts(decayed_at)")
    con.commit()

# Stage 3: CHECK rebuild to allow 'unanalyzed'
current_sql = ddl()
if "'unanalyzed'" not in current_sql:
    print("rebuilding contracts table to extend CHECK constraint")
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute("""
            CREATE TABLE contracts_new (
                contract_address    TEXT    NOT NULL PRIMARY KEY,
                chain               TEXT    NOT NULL DEFAULT 'arbitrum',
                detection_method    TEXT    NOT NULL CHECK (detection_method IN (
                                        'bytecode_pattern', 'behavioral_trigger',
                                        'deployer_history', 'routing_anomaly'
                                    )),
                detection_timestamp TEXT    NOT NULL,
                detection_block     INTEGER NOT NULL,
                confidence_tier     TEXT    NOT NULL DEFAULT 'unknown' CHECK (confidence_tier IN (
                                        'unknown', 'suspected', 'confirmed', 'unanalyzed'
                                    )),
                confidence_reason   TEXT    NOT NULL CHECK (length(confidence_reason) > 0),
                confirmation_tx_hash    TEXT,
                confirmation_timestamp  TEXT,
                confirmation_block      INTEGER,
                deployer_address        TEXT NOT NULL,
                deployer_funding_source TEXT,
                routing_presence    INTEGER NOT NULL DEFAULT 0,
                routing_first_seen  TEXT,
                has_asymmetric_transfer  INTEGER,
                has_conditional_revert   INTEGER,
                has_unusual_fee_structure INTEGER,
                bytecode_pattern_notes   TEXT,
                last_updated        TEXT    NOT NULL,
                deployed_code_hash  TEXT,
                decayed_at          TEXT,
                prior_confidence_tier TEXT,
                FOREIGN KEY (deployer_address) REFERENCES deployers(deployer_address)
            )
        """)
        con.execute("""
            INSERT INTO contracts_new SELECT
                contract_address, chain, detection_method, detection_timestamp,
                detection_block, confidence_tier, confidence_reason,
                confirmation_tx_hash, confirmation_timestamp, confirmation_block,
                deployer_address, deployer_funding_source, routing_presence,
                routing_first_seen, has_asymmetric_transfer, has_conditional_revert,
                has_unusual_fee_structure, bytecode_pattern_notes, last_updated,
                deployed_code_hash, decayed_at, prior_confidence_tier
            FROM contracts
        """)
        con.execute("DROP TABLE contracts")
        con.execute("ALTER TABLE contracts_new RENAME TO contracts")
        con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_deployer ON contracts(deployer_address)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_confidence_tier ON contracts(confidence_tier)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_detection_timestamp ON contracts(detection_timestamp)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_routing_presence ON contracts(routing_presence)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_code_hash ON contracts(deployed_code_hash)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_contracts_decayed_at ON contracts(decayed_at)")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.execute("PRAGMA foreign_keys=ON")

print(f"rows after:  {con.execute('SELECT COUNT(*) FROM contracts').fetchone()[0]:,}")
print(f"cols after:  {sorted(existing_cols())}")
print(f"has 'unanalyzed' in CHECK: {'unanalyzed' in ddl()}")
con.close()
print("DONE")
