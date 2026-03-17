-- ============================================================
-- Layer 3 — On-Chain Trap Surveillance System
-- SQLite Schema DDL — Phase 1
--
-- MIGRATION NOTE (Phase 2):
--   deployers.funding_sources is a JSON array in Phase 1.
--   When Phase 2 cluster analysis needs graph queries across
--   funding relationships ("which deployers share a funding
--   source?"), promote to a junction table:
--
--     CREATE TABLE deployer_funding_links (
--         deployer_address TEXT NOT NULL,
--         funding_address  TEXT NOT NULL,
--         first_seen       TEXT NOT NULL,
--         PRIMARY KEY (deployer_address, funding_address),
--         FOREIGN KEY (deployer_address) REFERENCES deployers(deployer_address)
--     );
--     CREATE INDEX idx_funding_links_funder ON deployer_funding_links(funding_address);
--
--   Then migrate existing JSON arrays into rows and drop the column.
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ============================================================
-- Table: deployers
-- One row per deployer address.
-- Created first — contracts FK references this table.
--
-- NOTE: confirmed_trap_count and suspected_trap_count are
-- intentionally omitted. These are derived at read time via
-- COUNT() queries against contracts and trap_events. Storing
-- denormalized counts is a data integrity risk — if the
-- pipeline updates one table without the other, counts silently
-- lie. For Phase 1 data volumes, derived counts are safer.
-- ============================================================
CREATE TABLE deployers (
    deployer_address                TEXT    NOT NULL PRIMARY KEY,
    chain                           TEXT    NOT NULL DEFAULT 'arbitrum',

    first_seen                      TEXT    NOT NULL,  -- ISO 8601
    last_seen                       TEXT    NOT NULL,

    total_contracts_deployed        INTEGER NOT NULL DEFAULT 0,

    -- JSON arrays — lookup lists, not independently queried in Phase 1.
    -- See migration note at top for Phase 2 promotion plan.
    funding_sources                 TEXT    NOT NULL DEFAULT '[]',  -- JSON array of addresses
    known_associated_deployers      TEXT    NOT NULL DEFAULT '[]',  -- JSON array of addresses

    -- Behavioral fingerprint
    typical_deployment_interval_hours REAL,
    typical_gas_price_gwei           REAL,
    deployment_pattern_notes         TEXT
);

-- ============================================================
-- Table: contracts
-- One row per flagged contract address.
--
-- OPERATIONAL CONSTRAINT (upsert order):
--   The FK on deployer_address means the pipeline MUST upsert
--   into deployers first, then insert into contracts. For new
--   deployments where no deployer record exists yet, the
--   pipeline must INSERT OR IGNORE a minimal deployer row
--   before inserting the contract row. This is enforced in
--   db.py's insert_contract() method.
-- ============================================================
CREATE TABLE contracts (
    -- Identity
    contract_address    TEXT    NOT NULL PRIMARY KEY,
    chain               TEXT    NOT NULL DEFAULT 'arbitrum',

    -- Detection
    detection_method    TEXT    NOT NULL CHECK (detection_method IN (
                            'bytecode_pattern', 'behavioral_trigger',
                            'deployer_history', 'routing_anomaly'
                        )),
    detection_timestamp TEXT    NOT NULL,  -- ISO 8601
    detection_block     INTEGER NOT NULL,

    -- Confidence
    confidence_tier     TEXT    NOT NULL DEFAULT 'unknown' CHECK (confidence_tier IN (
                            'unknown', 'suspected', 'confirmed'
                        )),
    confidence_reason   TEXT    NOT NULL CHECK (length(confidence_reason) > 0),

    -- Confirmation (null until confirmed)
    confirmation_tx_hash    TEXT,
    confirmation_timestamp  TEXT,
    confirmation_block      INTEGER,

    -- Deployer
    deployer_address        TEXT NOT NULL,
    deployer_funding_source TEXT,

    -- 1inch routing
    routing_presence    INTEGER NOT NULL DEFAULT 0,  -- boolean: 0/1
    routing_first_seen  TEXT,

    -- Bytecode signals (null = not yet analyzed)
    has_asymmetric_transfer  INTEGER,  -- boolean: 0/1/null
    has_conditional_revert   INTEGER,
    has_unusual_fee_structure INTEGER,
    bytecode_pattern_notes   TEXT,

    -- Metadata
    last_updated        TEXT    NOT NULL,

    FOREIGN KEY (deployer_address) REFERENCES deployers(deployer_address)
);

-- ============================================================
-- Table: trap_events
-- One row per confirmed trap firing. Separate table — every
-- field is a real column, not a JSON blob. Enables:
--   SELECT * FROM trap_events WHERE loss_estimate_usd > 100
-- ============================================================
CREATE TABLE trap_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    trap_contract_address   TEXT    NOT NULL,
    bot_address             TEXT    NOT NULL,
    tx_hash                 TEXT    NOT NULL UNIQUE,
    block_number            INTEGER NOT NULL,
    timestamp               TEXT    NOT NULL,  -- ISO 8601
    loss_estimate_usd       REAL,              -- null if not determinable
    failure_signature       TEXT    NOT NULL,

    FOREIGN KEY (trap_contract_address) REFERENCES contracts(contract_address)
);

-- ============================================================
-- Table: bytecode_cache
-- Two-tier lookup by code hash (SHA-256).
--   Tier 1: init_code_hash (tx.input) — hit skips eth_getCode + classifier.
--   Tier 2: deployed_code_hash (eth_getCode result) — hit skips classifier only.
-- Both tiers store the same classification result. A single deployment
-- inserts up to 2 rows (one per hash). The code_hash column stores
-- either hash; the cache is content-addressed.
-- ============================================================
CREATE TABLE bytecode_cache (
    code_hash           TEXT    NOT NULL PRIMARY KEY,
    confidence_tier     TEXT    NOT NULL,
    confidence_reason   TEXT    NOT NULL,
    bytecode_signals    TEXT    NOT NULL DEFAULT '{}',  -- JSON
    source_contract     TEXT    NOT NULL,  -- first contract classified with this hash
    hit_count           INTEGER NOT NULL DEFAULT 0,
    created             TEXT    NOT NULL   -- ISO 8601
);

-- ============================================================
-- Indexes
-- ============================================================

-- contracts.deployer_address: FK lookup + "all contracts from
-- this deployer" is a core cluster analysis query.
CREATE INDEX idx_contracts_deployer
    ON contracts(deployer_address);

-- contracts.confidence_tier: primary operational filter —
-- "give me all SUSPECTED contracts".
CREATE INDEX idx_contracts_confidence_tier
    ON contracts(confidence_tier);

-- contracts.detection_timestamp: time-range monitoring —
-- "what was deployed in the last N hours?"
CREATE INDEX idx_contracts_detection_timestamp
    ON contracts(detection_timestamp);

-- contracts.routing_presence: boolean filter for 1inch
-- cross-reference — "which flagged contracts have live routing?"
CREATE INDEX idx_contracts_routing_presence
    ON contracts(routing_presence);

-- trap_events.trap_contract_address: FK lookup + confirmation
-- audit trail per contract.
CREATE INDEX idx_trap_events_contract
    ON trap_events(trap_contract_address);

-- trap_events.block_number: time-ordered event queries —
-- "what traps fired in the last N blocks?"
CREATE INDEX idx_trap_events_block
    ON trap_events(block_number);

-- trap_events.loss_estimate_usd: enables loss-threshold queries —
-- "which events caused > $100 in losses?"
CREATE INDEX idx_trap_events_loss
    ON trap_events(loss_estimate_usd);

-- trap_events.bot_address: "which bots get trapped most often?"
-- identifies poorly-coded Layer 1 scanners. Also enables reverse
-- lookup: given a bot, show every trap it triggered.
CREATE INDEX idx_trap_events_bot
    ON trap_events(bot_address);

-- deployers.last_seen: "which deployers are currently active?"
CREATE INDEX idx_deployers_last_seen
    ON deployers(last_seen);
