# Layer 3 Consumable Intelligence Surface

**Snapshot:** 2026-04-21, production DB at `/app/surveillance/data/surveillance.db` (Railway, 2.45 GB).
**Source:** direct read of Railway surveillance DB via `railway ssh`; fresh pull this session.
**Purpose:** reference document describing what signals Layer 3's corpus supplies to external software, their reliability characteristics, and their query access pattern. It does not specify how the signals should be consumed, filtered, or thresholded — those are downstream decisions.

The document excludes: raw transaction-event records, behavioral-profile internals (e.g. timezone inference, gas-fingerprint distributions inside `deployer_profiles`), bytecode-cache contents, and per-detector methodology. It includes: the stable operational outputs that function as signals — classifications, tiers, counts, timestamps, linkage columns.

**Audit discipline anchor.** Seven corrections (Correction #9–#15) landed 2026-04-19/20 under an exception-as-rule review. The reliability characterization in Section 2 reflects what those audits actually measured, not framework claims.

---

## Section 1 — Inventory of consumable tables and columns

The corpus has 70 tables. The subset below is the consumable surface. Tables not listed are internal state (caches, profiles, low-level events) that are not intended for external consumption.

For each subsection:
- **Rows:** live count at snapshot time
- **Update frequency:** how often the table changes
- **Consumable fields:** the columns an external consumer should read
- **Query pattern:** the minimal SQL template; `?` marks parameter positions
- Fields explicitly excluded from this surface are listed where relevant

### 1.1 Contract-level classification

**Table:** `contracts`
**Rows:** 132,501
**Update frequency:** continuous (new rows as new deployments are observed on Base / Arbitrum / Optimism)

**Consumable fields:**
- `contract_address` (primary key), `chain`
- `detection_method` ∈ {`bytecode_pattern`, `behavioral_trigger`, `deployer_history`, `routing_anomaly`}
- `confidence_tier` ∈ {`unknown`, `suspected`, `confirmed`, `unanalyzed`}
- `detection_timestamp`
- `decayed_at` — non-null if the contract was decayed from `suspected` → `unanalyzed`
- `prior_confidence_tier` — the tier before decay (audit trail for Correction #9)
- `deployed_code_hash` — keccak of runtime bytecode; stable cross-chain identity
- `has_asymmetric_transfer`, `has_conditional_revert`, `has_unusual_fee_structure` — ternary (0/1/null) bytecode-pattern flags
- `deployer_address` — FK to `deployers`

**Not consumable:** `bytecode_pattern_notes` (free text, methodology-leaking), `confidence_reason` (ditto).

**Query pattern:**
```sql
SELECT contract_address, confidence_tier, decayed_at, deployed_code_hash,
       has_asymmetric_transfer, has_conditional_revert, has_unusual_fee_structure,
       deployer_address
FROM contracts
WHERE contract_address = LOWER(?);
```

---

### 1.2 Deployer-level classification

**Table:** `deployers`
**Rows:** 47,823
**Update frequency:** one row per deployer, touched on every deployment by that EOA; `mainnet_first_tx` backfilled once per deployer via Etherscan v2

**Consumable fields:**
- `deployer_address` (primary key), `chain`
- `first_seen`, `last_seen`
- `total_contracts_deployed`
- `mainnet_first_tx` — Ethereum mainnet first-tx ISO timestamp, or empty string if no mainnet history. Populated for 36,097 of 47,823 deployers at snapshot time; fills continuously for new deployers.
- `entity_type` — free-text tag set by human investigation or admin endpoint
- `funding_trail` — JSON blob; the consumable key is `funder` (the 1-hop upstream address)

**Not consumable:** `behavioral_score`, `score_breakdown`, `typical_gas_price_gwei`, `typical_deployment_interval_hours`, `deployment_pattern_notes`, `known_associated_deployers` — behavioral-profile internals.

**Query pattern:**
```sql
SELECT deployer_address, first_seen, total_contracts_deployed,
       mainnet_first_tx, entity_type,
       json_extract(funding_trail, '$.funder') AS funder
FROM deployers
WHERE deployer_address = LOWER(?);
```

---

### 1.3 Observed harm events

**Table:** `trap_events`
**Rows:** 1,867
**Update frequency:** continuous; one row per confirmed bot-vs-trap reversion event

**Consumable fields:**
- `trap_contract_address`, `bot_address`
- `tx_hash`, `block_number`, `timestamp`
- `loss_estimate_usd` — may be null
- `failure_signature` — free-text short tag (e.g. `revert:selector=559e0fab`)

**Query pattern:**
```sql
-- All trap events for a contract
SELECT timestamp, bot_address, tx_hash, loss_estimate_usd, failure_signature
FROM trap_events
WHERE LOWER(trap_contract_address) = LOWER(?)
ORDER BY timestamp DESC;
```

This is the only table that records observable end-state harm. Every PPV measurement in Section 2 uses this table as the outcome proxy.

---

### 1.4 Alerts

**Table:** `alerts` (joined with `false_positives` for audit)
**Rows:** 23,433 alerts · 12 `false_positives` audit rows
**Update frequency:** continuous; alerts emit in real time from `deployment_monitor`, `selector_monitor`, `revert_cluster_detector`, `event_monitors`, `x402_monitor`, Alchemy Notify webhooks

**Consumable fields on `alerts`:**
- `alert_type` — enumerated (see table below)
- `address`, `tx_hash`, `block_number`, `timestamp`
- `payload` — JSON; shape varies by alert_type
- `false_positive` ∈ {0, 1}

**Consumable fields on `false_positives`:**
- `contract_address`, `chain`
- `fp_reason` — operator-supplied justification
- `fp_method` — tag describing how the FP was determined (`admin_bulk_mark`, `balanced_interaction`, `sustained_traffic`, `weak_detector_only`)
- `detector_blamed`, `assessed_at`

**Alert-type cardinality** (lifetime, non-FP, top 15):
```
LAUNDRY_PIPELINE            6,746
WATCHLIST_HIT               6,731
COORDINATED_DEPLOYMENT      5,401
CASHOUT_MOVEMENT            2,979
TRAP_CONFIRMED                398
DORMANT_ACTIVATION            368
HIGH_VELOCITY_DEPLOYER        316
X402_FACILITATOR_UNKNOWN      190
X402_AGENT_DRAIN              172
SUSPECTED_HIGH_TRAFFIC         55
TRUST_AMPLIFICATION            55
BRIDGE_WITHDRAWAL              19
LIVE_EXTRACTION_OBSERVED        2
BOT_DEATHWATCH                  1
```

**Query pattern:**
```sql
-- Real-time tail
SELECT alert_type, address, tx_hash, timestamp, payload
FROM alerts
WHERE COALESCE(false_positive, 0) = 0
  AND timestamp > ?
ORDER BY timestamp DESC;

-- FP check with audit row
SELECT a.alert_type, a.timestamp, fp.fp_reason, fp.detector_blamed
FROM alerts a
LEFT JOIN false_positives fp ON LOWER(fp.contract_address) = LOWER(a.address)
WHERE a.false_positive = 1;
```

Since Correction #14 the `/admin/mark-false-positive` endpoint writes both tables atomically; any alert with `false_positive = 1` and no `false_positives` row is pre-Correction-#14 legacy state.

---

### 1.5 Documented extraction events

**Table:** `extraction_events`
**Rows:** 8
**Update frequency:** manual INSERT by investigator; shadow classifier suggestion auto-filled by `surveillance.extraction_classifier`

**Consumable fields:**
- `event_id` (e.g. `EXTRACTION_008`)
- `event_type` — human-assigned label from a closed 7-value vocabulary
- `event_type_suggestion`, `event_type_suggestion_confidence` — classifier output
- `observed_at`, `documented_at`
- `total_usd_moved`, `nodes_active`
- `chain` — the chain the event occurred on (may be off-L3 chains: near, solana, bnb, ethereum)
- `monitored_chain` ∈ {0, 1} — `1` if the chain is in Layer 3's continuous monitoring set (Base / Arbitrum / Optimism), `0` for reference-only events
- `summary` — human-readable description

**Not consumable:** `raw_transactions` (JSON blob, may contain investigator notes), `notes`.

**Query pattern:**
```sql
SELECT event_id, event_type, event_type_suggestion,
       observed_at, chain, monitored_chain, total_usd_moved, summary
FROM extraction_events
WHERE monitored_chain = 1  -- L2 events only, for chain-scoped rollups
ORDER BY observed_at DESC;
```

---

### 1.6 Organizational classification

**Tables:** `org_wallets` (ground truth) · `org_candidates` (Tier-B clustering output)
**Rows:** 13 · 326
**Update frequency:**
- `org_wallets`: manual INSERTs (post-Correction-#11 seed of 13 entries); grows when investigator promotes a candidate
- `org_candidates`: nightly at 04:45 UTC via scheduled `surveillance.org_candidates --apply`

**Consumable fields on `org_wallets`:**
- `address`, `chain` (composite primary key)
- `org_id` — string label (`org_001`, `org_002`, ...)
- `role` — short tag (`treasury`, `cashout`, `gas_station`, `operator`, `laundry`, ...)
- `added_at`, `added_by`, `reason`

**Consumable fields on `org_candidates`:**
- `candidate_id` — stable hash over the deployer set
- `cluster_size` (3–50)
- `deployer_addresses` — JSON array
- `shared_funding_source`, `shared_chain`
- `shared_gas_fingerprint` — may be null (not all clusters have gas data)
- `first_seen`, `last_seen`, `detected_at`
- `status` ∈ {`pending`, ..., potentially `promoted`, `dismissed` — only `pending` values exist at snapshot}

**Query pattern:**
```sql
-- Is this address in a known org?
SELECT org_id, role, chain FROM org_wallets WHERE LOWER(address) = LOWER(?);

-- Which novel-org clusters reference this deployer?
SELECT candidate_id, cluster_size, shared_funding_source, status
FROM org_candidates
WHERE deployer_addresses LIKE '%' || LOWER(?) || '%' AND status = 'pending';
```

---

### 1.7 Infrastructure registry

**Table:** `infrastructure_registry`
**Rows:** 18
**Update frequency:** manual INSERT; new classifications seeded as investigators validate legitimate infrastructure

**Consumable fields:**
- `address`, `chain` (composite primary key — a single address can appear on multiple chains with CREATE2)
- `classification` — stable slug (e.g. `circle_cctp_message_transmitter_v2`)
- `verified_at`
- `notes` — may include retrospective-evidence flags (e.g. `retrospective_kelp_*` entries for post-hoc analysis)

**Snapshot content:**
- Circle CCTP v2 on base/arbitrum/optimism (4 classifications × 3 chains = 12 rows)
- Retrospective Kelp/LayerZero entries on ethereum + unichain (6 rows)

**Query pattern:**
```sql
SELECT classification, chain FROM infrastructure_registry
WHERE LOWER(address) = LOWER(?);
```

---

### 1.8 Bytecode-family membership

**Tables:** `bytecode_families` · `bytecode_family_members`
**Rows:** 1,452 · 9,364
**Update frequency:** family clustering runs nightly at 03:00 UTC (`bytecode_families --cluster`)

**Consumable fields on `bytecode_families`:**
- `family_id` (primary key), `family_name`
- `detection_tier` — internal tier tag (T1, T2, etc.)
- `member_count`, `unique_deployers`
- `avg_revert_rate`, `total_victims`
- `first_seen`, `last_updated`
- `is_cross_deployer` — boolean; true = multiple deployers share this bytecode template

**Consumable fields on `bytecode_family_members`:**
- `family_id`, `contract_address`, `deployer`
- `deployment_timestamp`, `revert_rate`, `unique_callers`

**Not consumable:** `representative_bytecode_prefix`, `selector_fingerprint` (disassembly methodology).

**Query pattern:**
```sql
-- Family for a contract
SELECT bfm.family_id, bf.member_count, bf.unique_deployers, bf.is_cross_deployer
FROM bytecode_family_members bfm
JOIN bytecode_families bf ON bf.family_id = bfm.family_id
WHERE bfm.contract_address = LOWER(?);
```

---

### 1.9 Trust amplification

**Table:** `trust_amplification`
**Rows:** 170
**Update frequency:** computed batch-wise via `trust_amplification --analyze`; cadence on-demand rather than scheduled

**Consumable fields:**
- `contract_address`
- `total_callers`, `router_callers`, `router_percentage` — aggregator reuse metrics
- `amplification_factor` — vs family baseline
- `revert_rate`
- `alert_level` ∈ {null, `CRITICAL`}

**Query pattern:**
```sql
SELECT router_percentage, amplification_factor, revert_rate, alert_level
FROM trust_amplification
WHERE contract_address = LOWER(?);
```

---

### 1.10 Approval exposure

**Tables:** `approval_watchlist` · `live_exposures`
**Rows:** 18,413 · 2
**Update frequency:** `approval_watchlist` continuous (Permit2 approvals on flagged contracts); `live_exposures` manual via admin endpoint

**Consumable fields on `approval_watchlist`:**
- `victim_address`, `contract_address` (the approved spender)
- `approve_tx_hash`, `approve_timestamp`
- `contract_tier` — snapshot of `confidence_tier` at approve time
- `drain_detected` ∈ {0, 1} — flipped to 1 when a sweep is observed
- `drain_tx_hash`, `drain_timestamp`, `drain_caller` — populated if `drain_detected = 1`

**Query pattern:**
```sql
-- Approvals a victim has granted to suspected/confirmed contracts
SELECT contract_address, approve_timestamp, contract_tier,
       drain_detected, drain_tx_hash
FROM approval_watchlist
WHERE LOWER(victim_address) = LOWER(?)
ORDER BY approve_timestamp DESC;
```

---

### 1.11 Aggregate daily metrics

**Tables:** `daily_metrics` · `camouflage_metrics` · `predictions`
**Rows:** 11 · 12 · 18
**Update frequency:**
- `daily_metrics` — nightly at 00:15 UTC
- `camouflage_metrics` — nightly at 00:20 UTC (since Correction #13, writes both `camouflage_ratio` and `adversary_low_revert_ratio`)
- `predictions` — nightly at 00:30 UTC; a prior prediction's `hit` + `actual_value` fields get scored as target dates arrive

**Consumable fields on `camouflage_metrics`** (post-Correction-#13):
- `date`, `chain`
- `total_active_contracts`, `camouflaged_count`, `camouflage_ratio` — population-level (all contracts with ≥10 tx that day)
- `adversary_total_contracts`, `adversary_low_revert_count`, `adversary_low_revert_ratio` — restricted to contracts in `confidence_tier IN ('confirmed', 'suspected')`
- `overt_count`, `moderate_count`, `avg_camouflaged_callers`, `avg_overt_callers`

**Query pattern:**
```sql
SELECT date, camouflage_ratio, adversary_low_revert_ratio,
       total_active_contracts, adversary_total_contracts
FROM camouflage_metrics
ORDER BY date DESC LIMIT 30;
```

---

### 1.12 Bot classifications

**Tables:** `bot_candidates` · `bot_strategies` · `bait_profiles`
**Rows:** 1,396 · 634 · 51
**Update frequency:** `bot_candidates` continuous; `bot_strategies` + `bait_profiles` via `strategy_fingerprint --classify-all --profile-baits --lifecycle` (manual cadence)

**Consumable fields on `bot_strategies`:**
- `bot_address`, `strategy_type`
- `contracts_targeted`, `avg_revert_rate`, `total_interactions`
- `trap_vulnerability` — tag
- `classified_at`

**Not consumable:** `primary_selectors`, `active_hours` (profile internals).

---

### 1.13 Entity classification (auxiliary)

**Table:** `entity_classification`
**Rows:** 1,080
**Update frequency:** manual + derived from admin endpoints

**Consumable fields:**
- `address`, `category`, `subtype`
- `confidence` — free-text tag
- `org_id` — link into `org_wallets` naming when present
- `source` — provenance tag

---

### 1.14 x402 monitoring outputs

**Tables:** `x402_events` · `x402_facilitators` · `x402_permit2_exposure`
**Rows:** 537,894 · 266 · 2,808
**Update frequency:** continuous; one row per observed x402 agent-payment event and per unique facilitator encountered

**Consumable fields on `x402_events`:**
- `tx_hash`, `block_number`, `timestamp`, `chain`
- `facilitator_address`, `payer_address`, `payee_address`
- `token_contract`, `token_symbol`, `amount`
- `x402_type` — event class
- `confidence` — free-text tag
- `selector`

Note: large table; scan queries should include a time or chain filter.

---

### 1.15 Risk score (compute-per-request, no persistent table)

**Location:** `surveillance.risk_scoring.score_contract(conn, address)`
**Persistence:** none. Risk scores are computed live per request (see Correction #6). No `risk_scores` table exists despite what earlier documentation claimed.

**Return shape** (post-Correction-#12):
```json
{
  "contract_address": "...",
  "stored_potential": 0-125,
  "approval_scope_score": 0-25,
  "capability_score": 0-25,
  "deployer_risk_score": 0-25,
  "org_context_score": 0-25,
  "observation_capability_score": 0-25,
  "realized_value": int,
  "volatility": float,
  "risk_score": float,
  "risk_tier": "MINIMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "components": { ... per-component breakdown ... }
}
```

**Tier boundaries:** `CRITICAL ≥ 50`, `HIGH ≥ 20`, `MEDIUM ≥ 8`, `LOW ≥ 3`, else `MINIMAL`.

**Not consumable:** the `components` substructures contain methodology fingerprints (specific signal patterns, weights, threshold values). External consumers should treat the top-level `risk_tier` and `*_score` fields as the surface and ignore `components`.

---

## Section 2 — Epistemic tier and reliability caveats per signal

Tier semantics per [`docs/lexicon.md#epistemic-tier-classification`](lexicon.md#epistemic-tier-classification):
- **Tier A (deductive):** verifiable on-chain or by direct arithmetic on the corpus. A third party can reproduce.
- **Tier B (inferential):** methodology-applied judgment; defensible but requires the methodology to be valid.
- **Tier C (speculative):** explicit prediction or architectural claim without confirmation in corpus.

Overclaiming corpus reliability creates false signals for downstream consumers. Underclaiming creates unnecessary constraints. The tiering below reports what the audit actually measured.

### 2.1 `contracts.confidence_tier`

| value | tier | PPV at snapshot | meaning |
|---|---|---|---|
| `confirmed` | **A** | **84.84%** (554 of 653 have at least one `trap_events` row) | Observed-harm backing. Reliable as a flag. |
| `suspected` | **B** | **0.01%** (9 of 62,954 have observable harm) | Detector-fired-something flag. Near-zero predictive value at any horizon. See Correction #9. |
| `unanalyzed` | **B** | **0.00%** (0 of 7,886) | Explicit "detector fired, ≥30 days passed, no observable harm." Decayed from `suspected`. See Correction #9. |
| `unknown` | — | **0.00%** (0 of 61,008) | No detection signal; not a claim. |

**Failure modes of `suspected`:**
- The 0.01% PPV is not a bug in any single detector; it reflects that `suspected` aggregates four `detection_method` values (`bytecode_pattern`, `behavioral_trigger`, `deployer_history`, `routing_anomaly`) with different precision profiles. External consumers that want a usable precision signal should filter on `detection_method` too — the corpus does not currently publish per-method PPV.
- Trap-fleet behavior (see the `0xcadf9ebe…` deployer: 20 contracts, 19 never received a single call) inflates `suspected` counts without any of them ever resolving to harm.

**Failure modes of `confirmed`:**
- The 84.84% reflects PPV on the population Layer 3 has detected-then-observed. It does not address recall (contracts Layer 3 missed entirely). No recall figure is publishable from within the corpus — there is no external ground truth.

**`decayed_at` and `prior_confidence_tier`:** Tier A — deterministic transition record. Re-upgrade on subsequent `trap_events` is not yet automated; a contract that decayed and later fires will remain `unanalyzed` until a re-promotion pass runs (flagged as open work in Correction #9).

---

### 2.2 Bytecode-signal flags (`has_asymmetric_transfer`, `has_conditional_revert`, `has_unusual_fee_structure`)

**Tier A** for the detection itself — the pattern was found in the bytecode. **Tier B** for the predictive interpretation. Ternary values: `1` = pattern matched, `0` = not matched, `NULL` = not analyzed.

**Known failure modes:**
- False negatives when bytecode is fetched only partially (post-Correction-#5 cache transplant staleness is patched, but rollout is not universal across all code paths).
- False positives on legitimate contracts that happen to use conditional reverts in access-control (the 12 entries in `false_positives` with `fp_method='weak_detector_only'` capture this class).

---

### 2.3 Deployer-level signals

**`total_contracts_deployed`** — Tier A, direct count.
**`first_seen` / `last_seen`** — Tier A.
**`mainnet_first_tx`** — Tier A when populated; enrichment via Etherscan v2. Empty string means "checked, no mainnet history"; NULL means "not yet checked." 36,097 of 47,823 deployers populated at snapshot time.
**`entity_type`** — Tier B. Free-text; set by manual curation or by admin endpoint.
**`funding_trail.funder`** — Tier A, 1-hop upstream funder address when populated (populated for 36,813 of 47,823).

**Not a consumable signal:** `behavioral_score` exists in the schema but is always `0.0` in the current corpus (no active scorer writes it). External consumers should not treat it as informative.

---

### 2.4 Organizational classification (`org_wallets` / `org_candidates`)

**`org_wallets`** — Tier A for the 13 seed rows covering `org_001` (11 wallets) and `org_002` (2 wallets). The rows document specific investigated cases.

**Critical caveat from Correction #11:** `org_003` and `org_004` appear in `entity_classification`, `daily_report.py`, and `diamond_model.py` as case labels, but **have zero wallet-level membership in `org_wallets`**. A downstream consumer joining `org_wallets` to get "which org is this address part of?" will miss 100% of `org_003`/`org_004` attribution because there are no wallet rows for them. This is the allowlist-vs-discovery gap the audit surfaced.

**`org_candidates`** — Tier B. 326 pending clusters from the nightly novel-org detector. The detector's heuristic is "3–50 deployers sharing a `funding_trail.funder` within a 72h window of each other's `first_seen`." Promotion to `org_wallets` is a manual decision; zero candidates have been promoted at snapshot time. A consumer should treat a cluster as a Tier B structural hypothesis, not an identity claim.

**Known failure modes of the candidate detector:**
- Excludes clusters > 50 members as CEX/faucet (by design); would miss an adversary operating at gas-station scale.
- Excludes clusters < 3 members; would miss pair-wise rotations.
- Requires `funding_trail.funder` populated; a deployer with no upstream trace never enters a cluster.

---

### 2.5 Observation-capability score (`observation_capability_score` in risk-scoring output)

**Tier B.** Added 2026-04-20 per Correction #12. 0–25 signal contributing to `stored_potential`. Components: bytecode markers (CALLER/TIMESTAMP/TXORIGIN), log-scaled distinct-EOA count, `infrastructure_registry` role match, Permit2-seen edge bump.

**Critical caveat:** the signal only computes against contracts that are in the `contracts` table. Layer 3's `contracts` table records new deployments observed since March 2026 on Base/Arbitrum/Optimism — it does **not** include pre-existing infrastructure (Uniswap V3 routers, Chainlink oracles, LayerZero endpoints, Aave pools deployed before corpus start). A consumer scoring such an address will receive `{"error": "contract not found in database"}`, not a zero score; a consumer scoring a just-deployed contract will receive `observation_capability_score = 0` because the EOA-count signal has not yet accumulated.

**Validation:** in-corpus high-EOA contracts show `observation_capability_score` of 6–15 points (1k-10k EOAs → tier 6; 10k+ EOAs + bytecode markers → 14–15). See Correction #12 for the table.

---

### 2.6 Camouflage ratios (`camouflage_metrics`)

**`camouflage_ratio`** — Tier A for the arithmetic (fraction of all contracts with ≥10 tx in a day and revert rate <10%). **Tier B for the interpretation** — the metric is population-level and includes legitimate DEX routers, which are also low-revert. The name is preserved for historical continuity per Correction #13.

**`adversary_low_revert_ratio`** — Tier A for the arithmetic, restricted to contracts in `('confirmed', 'suspected')`.

**Snapshot numbers (2026-04-21 lifetime):**
| cohort | total (10+ tx) | low-revert | ratio |
|---|---|---|---|
| population | 1,004 | 634 | 63.1% |
| adversary | 985 | 619 | 62.8% |
| **confirmed-only** | **342** | **87** | **25.4%** |

**Interpretive consequence (from Correction #13):** population and adversary ratios track within 0.3pp because the adversary set is 98% of the ≥10-tx population. Confirmed-only is 37.7pp *lower* than population — confirmed traps revert *more*, not less. The "70-79% Nash equilibrium" claim in prior materials does not hold against this data; Correction #13 supersedes it.

**Failure modes:**
- The `≥10 tx in a day` filter drastically narrows the population; <1% of the `contracts` table meets it. The ratios describe only high-activity contracts.
- `chain` column exists but is written as `'ethereum'` placeholder in the current implementation; per-chain breakdown is not populated despite the column.
- Historical rows (pre-Correction-#13) have NULL in the adversary columns; a consumer filtering on non-null adversary values currently gets only rows written after 2026-04-20.

---

### 2.7 Alert stream (`alerts.alert_type`)

Not all alert types carry the same tier. Rough guide:

| alert_type | tier | reliability note |
|---|---|---|
| `TRAP_CONFIRMED` | **A** | Generated at trap_event insertion; matches observable harm. |
| `WATCHLIST_HIT` | **A** | Deterministic — fires when a watched address transacts. |
| `LAUNDRY_PIPELINE` / `CASHOUT_MOVEMENT` | **B** | Generated by Alchemy Notify webhook when a watched org wallet is `from`/`to` of a movement. Reliability depends on whether the watched address is actually an org wallet (which is ground-truth by construction — there are only 9 watched addresses hardcoded in the webhook handler). |
| `COORDINATED_DEPLOYMENT` | **B** | Velocity-escalation pipeline. Over-fires: 5,401 in 24h is high volume relative to 1,420 new contracts. |
| `DORMANT_ACTIVATION` | **B** | Sleeping deployer wakes up. 368 lifetime; known not to all be adversarial (c0ffeefeed correction 2026-04-18). |
| `HIGH_VELOCITY_DEPLOYER` | **B** | Threshold-based: `>8 contracts in current session`. The confirmed contract from 2026-04-21 fired this. |
| `X402_FACILITATOR_UNKNOWN` | **B** | Every new x402 facilitator encountered. `x402_facilitators` table accumulates all of them; known-rogue tagging is separate and Tier-B. |
| `X402_AGENT_DRAIN` | **B** (Tier **A** for the on-chain event, Tier **B** for "this is a drain"; see Correction #8 for token-decimals caveat on non-stablecoins) | |
| `SUSPECTED_HIGH_TRAFFIC`, `TRUST_AMPLIFICATION`, `BRIDGE_WITHDRAWAL` | **B** | |
| `LIVE_EXTRACTION_OBSERVED`, `BOT_DEATHWATCH` | **A** | Narrow, precisely-defined events; low volume by design. |

**False-positive surface:** `COALESCE(false_positive, 0) = 1` filter is mandatory for any consumer treating the stream as actionable. Post-Correction-#14, every `false_positive = 1` row is supposed to have a matching `false_positives` audit row (exceptions are pre-Correction-#14 legacy rows in local DB — Railway had no silenced alerts).

---

### 2.8 Extraction events (`extraction_events.event_type` / `event_type_suggestion`)

**Tier A** for the documented facts (chain, total_usd_moved, observed_at).
**Tier B** for `event_type` (hand-assigned label) and `event_type_suggestion` (classifier output). Post-Correction-#15, classifier agreement is 8/8 on current rows; divergence between `event_type` and `event_type_suggestion` is the actionable signal that a label should be reviewed.

**Vocabulary is closed**: `full_pipeline_cycle`, `infrastructure_parasite`, `oracle_manipulation_lending_exploit`, `oft_adapter_admin_compromise`, `cross_chain_proof_verification_bypass`, `cross_chain_dvn_verification_failure`, `unclassified`. Adding a new category requires editing `surveillance/extraction_classifier.py::_RULES` — which leaves git history.

**`monitored_chain` filter matters.** Events with `monitored_chain = 0` (NEAR, Solana, BNB) are reference events not in Layer 3's continuous monitoring; their inclusion in corpus-wide USD rollups must be handled explicitly by the consumer.

---

### 2.9 Infrastructure registry

**Tier A**. Manually curated. 18 rows: Circle CCTP v2 on three L2s (12 rows) and retrospective Kelp/LayerZero entries on ethereum + unichain (6 rows). Adding entries requires investigator confirmation.

**Known gap:** the registry is sparse. Many legitimate infrastructure contracts are not registered (Uniswap routers, Chainlink, Aave, 1inch, etc.). An `infrastructure_registry` miss is NOT a signal that an address isn't infrastructure — it is a signal that the registry hasn't been seeded for that address. Downstream consumers using this for "is known legit" should treat absence as "unknown," not as "not legit."

---

### 2.10 Bytecode-family membership

**Tier A** for the membership itself (deterministic clustering on bytecode). **Tier B** for any predictive use.

`is_cross_deployer` is the most load-bearing single signal: a family with `is_cross_deployer = true` and high `member_count` from few `unique_deployers` indicates a template being reused across a coordinated set — the T1-d5351e977044 family with 435 deployers / 2 funders is the corpus's flagship instance of this, unexplained at snapshot time and still worth investigation.

---

### 2.11 Trust amplification

**Tier A** for the arithmetic (`router_percentage`, `amplification_factor`, `revert_rate`).
**Tier B** for the adversarial interpretation — high `amplification_factor` relative to family baseline means the contract attracts disproportionate aggregator traffic, which has correlated with trap behavior in the observed cases (`0xd4624228` Cantina rejection) but has not been formalized as a PPV claim.

`alert_level = 'CRITICAL'` rows deserve investigator attention but have not been individually validated at scale.

---

### 2.12 Approval exposure

**Tier A.** Every `approval_watchlist` row is an on-chain event (an ERC-20/Permit2 approval targeting a `confidence_tier IN ('suspected', 'confirmed')` contract). `drain_detected = 1` is likewise an observed sweep. Both deterministic.

---

### 2.13 Live-compute risk score

**Composite Tier** — each of the five components has its own tier:
- `approval_scope` — Tier A (arithmetic on observed approvals)
- `capability` — Tier A (bytecode markers observed) / Tier B (scoring weights are methodology)
- `deployer_risk` — Tier B (composed of prior-trap count, velocity flag, org-link, etc.)
- `org_context` — Tier A for org-membership arithmetic, Tier B for the bonus value
- `observation_capability` — Tier B (see §2.5)

The top-level `risk_tier` and `risk_score` are Tier B composites. Downstream consumers treating `risk_tier = 'CRITICAL'` as deterministic will overclaim; treating it as "the framework's composite flag" with a measured but not published PPV is accurate.

**No persistence.** Risk scores are per-request compute. Bulk-query use cases (e.g. "all CRITICAL contracts") require iterating `score_contract` across candidate rows; no snapshot table exists.

---

## Section 3 — Query characteristics

### 3.1 Latency

All consumable queries are direct SQLite reads against the local volume on Railway (2.45 GB database, WAL mode). Rough p50 latencies, per-query:

| query class | p50 |
|---|---|
| Single-row lookups (`contracts` / `deployers` / `org_wallets` / `infrastructure_registry` PK) | <5 ms |
| `trap_events` / `alerts` for one address | <20 ms (indexed on address) |
| Recent-window scan (`alerts WHERE timestamp > ?` for last 24h) | 50–200 ms |
| `transaction_events` single-contract scan | 20–500 ms depending on contract activity |
| `deployer_similarity` (556k rows) full scan | 1–3 seconds; not intended for per-request use |
| `score_contract` live compute | ~80 ms (per Correction #6 measurement) |
| `org_candidates` full table scan | <50 ms (326 rows) |
| `camouflage_comparison` aggregate (population vs adversary vs confirmed) | 5–15 seconds (scans full `transaction_events`) |

### 3.2 Local vs live-production

Two access modes are available for this surface:

1. **Live Railway via HTTP API** — `/api/v1/*` endpoints for risk, contract, deployer, org, ecosystem-stats, feed, etc. Documented at [`/docs`](https://spypy.up.railway.app/docs). Response envelope includes `fresh_tables` (per Correction #7) so consumers can see whether the data behind the response is stale. Authentication: none for public endpoints, `Bearer ADMIN_TOKEN` for `/admin/*`.

2. **Local Railway DB copy** — the DB is replicated to external analytics via periodic `railway ssh` pulls (one-shot full copy) or via the `/admin/sync-*` endpoints for specific columns. Lag between production and local copy depends on when the pull was last run.

The consumable surface described in Section 1 is queryable via both; latency characteristics above are for direct SQL against the local volume.

### 3.3 RPC dependencies (Alchemy)

This is the dimension that most tightly constrains the consumable surface. Layer 3's architectural discipline (per `claude.md` §Infrastructure) is to **minimize Alchemy API calls**. The constraint is enforced, not aspirational.

**Tables and signals that do NOT require RPC** (safe for unlimited consumption):
- All of Section 1 above. Every consumable field lives in SQLite and is populated by batch or continuous work that runs on-Railway.

**Tables and signals that DO require RPC and are therefore NOT part of the stable consumable surface:**
- `eth_traces` (12 rows) — populated by `eth_depth.py` which makes Alchemy calls per-trace.
- `goplus_results` (50 rows) — populated by `goplus_enrichment.py` which calls GoPlus API.
- `contract_verification` (85 rows) — populated by `intelligence.py` which calls Alchemy.
- Any signal that would need "current on-chain state" (token balance at block N, current `getConfig` read, current storage slot value) is **outside** the consumable surface because it would require per-query RPC.

**Violations to watch for:** a downstream consumer that queries `contracts.contract_address` and then wants to cross-check a current ERC-20 allowance or a current proxy-slot read on the same address is stepping outside the consumable surface — the corpus does not guarantee those values. Such a consumer must either (a) accept the corpus's last-known snapshot fields, or (b) own its own RPC budget for the live-state check. Layer 3 will not silently proxy such calls.

**Single per-deployer RPC already budgeted:** `auto_funder_tracer` makes one Alchemy call per new deployer to trace funding, and one Etherscan v2 call to populate `mainnet_first_tx`. These are write-path enrichments — they populate the columns documented in Section 1.2. Consumers read the populated columns; no per-read RPC is implied.

**Scheduler budget:** seven nightly jobs (00:15–04:45 UTC) run analysis and decay/candidate passes, all against SQLite. None call Alchemy. The scheduler budget is bounded.

### 3.4 Freshness markers

Post-Correction-#7, API responses include `fresh_tables` mapping table names to the most recent `last_updated` / `computed_at` timestamp on the relevant rows. Consumer should check freshness before treating a signal as actionable. Relevant columns:

| table | freshness column |
|---|---|
| `contracts` | `last_updated` |
| `deployers` | `last_seen` |
| `bytecode_families` | `last_updated` |
| `camouflage_metrics` | `date` |
| `daily_metrics` | `date` |
| `trust_amplification` | `last_updated` |
| `trap_events` | `timestamp` |
| `alerts` | `timestamp` |

---

## Maintenance conventions

This document is a snapshot reference. Revision triggers:
- New consumable table added (e.g. if a future correction exposes signals from a previously-internal table)
- New audit finding that changes a tier or a PPV figure materially
- Schema migration that changes the shape of a consumable column
- Correction to the correction log that refines reliability characterization

Updates are additive where possible (append new tier information rather than overwriting), destructive where a prior claim is wrong (update the row with a link to the correction).

**Out of scope for this document:**
- How signals should be combined into a decision
- Alerting / filtering thresholds
- Consumer-side caching strategies
- Rate-limiting or throttling recommendations
- Commercial-tier access stratification

Those belong in the downstream specification.
