# Cache Invalidation Audit — Wave 1

**Date:** 2026-04-17
**Scope:** Every mutation path on a source table, checked against every derived-data table that reads from it. Pattern-level sweep — siblings of Correction #5 (bytecode_cache staleness).
**Methodology:** No code changes. This report surfaces the invalidation gaps; remediation is a separate decision.

---

## Headline finding

The **derived-analysis tables are badly stale** — not because of a per-row invalidation gap, but because the analysis producers haven't run in ~3 weeks. The audit uncovered the same *class* of problem at a coarser timescale than the bytecode_cache issue.

| Derived table | Rows | Latest timestamp | Staleness (vs 2026-04-17) |
|---|---|---|---|
| `bytecode_families` | 405 | 2026-03-24 | **24 days** |
| `trust_amplification` | 32 | 2026-03-25 | **23 days** |
| `daily_metrics` | 9 | 2026-03-25 | **23 days** |
| `camouflage_metrics` | 9 | 2026-03-25 | **23 days** |
| `deployer_similarity` | 4,879 | 2026-03-26 | **22 days** |
| `strategy_lifecycle` | 8 | 2026-03-26 | **22 days** |
| `entity_classification` | 1,080 | 2026-04-12 | 5 days |
| `deployer_profiles` | 1,873 | (no timestamp col) | — |
| `bot_strategies` | 634 | (no timestamp col) | — |
| `bait_profiles` | 51 | (no timestamp col) | — |
| `behavioral_baselines` | 7 | (no timestamp col) | — |
| `behavioral_anomalies` | 121 | (no timestamp col) | — |
| `predictions` | 9 | (no timestamp col) | — |
| **`risk_scores`** | — | — | **TABLE DOES NOT EXIST** |

Over the same window (2026-03-24 → 2026-04-17) the corpus grew from ~76k contracts to 124k+ (local snapshot) / 593 confirmed vs 368 at deck-time. Analysis results served by `/api/v1/*` therefore reflect a corpus state ~50k contracts out of date.

Two consequences worth isolating:

1. **`risk_scores` is a ghost table.** `CLAUDE.md` lists it as a current-production table ("Stored potential scoring output"). It does not exist in the DB. `risk_scoring.py:score_contract` computes live and is invoked ad-hoc by the API. No precomputation, no caching. Not an invalidation gap — a documentation/config gap: customers following the docs cannot query the table and may assume it was missed by their SQL rather than absent by design. **HIGH** risk for credibility; **NONE** for correctness.
2. **The producers exist, the cron doesn't.** `CLAUDE.md` lists the analysis commands under "Running Common Operations" but none are wired into `run_surveillance.py`'s 24/7 loop. On Railway, they need external triggers (cron, Railway cron-jobs, manual). If that schedule isn't running, tables drift. It has not been running since late March.

---

## Per-mutation-site audit (the original mandate)

Risk legend:
- **HIGH** — incorrect classification served via `/api/v1/*`
- **MEDIUM** — stale figure in reports/briefs/dashboards, not in hot API path
- **LOW** — minor metric drift; doesn't change any user-visible verdict

### 1. Mutations on `contracts.confidence_tier` / `confidence_reason`

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `db.update_contract_confidence` (db.py:660) | `bytecode_cache` | **YES** (shipped Correction #5, 2026-04-17) | — | N/A |
| `db.insert_trap_event` auto-upgrade (db.py:811) | `bytecode_cache` | **YES** (shipped Correction #5) | — | N/A |
| `honeypot_checker.py:119,146` direct UPDATE | `bytecode_cache` | **YES** (shipped Correction #5) | — | N/A |
| `backfill_self_loops.py:82` direct UPDATE | `bytecode_cache` | **YES** (shipped Correction #5) | — | N/A |
| All four sites above | `deployer_profiles` (re: deployer's confirmed_trap_count, suspected_trap_count) | NO — producer-recompute only | **MEDIUM** | `/api/v1/deployer/{addr}` reads `deployer_profiles` directly. When a sibling contract gets promoted to confirmed, the deployer's profile row stays with the old counts until `deployer_profiler --profile-all` runs. 22+ days stale today. Recommendation: either (a) restart the deployer_profiler cron, or (b) recompute the confirmed/suspected counts live in the API reader rather than persisting. |
| All four sites above | `deployer_similarity` (uses profiles as input) | NO — producer-recompute only | **MEDIUM** | Same as above, one step further downstream. If `deployer_profiles` is stale, `deployer_similarity` compounds it. |
| All four sites above | `entity_classification` (criminal / confirmed labeling depends on confirmed trap count) | NO — producer-recompute only | **HIGH** | API reads entity_classification for org attribution and deployer category in `/api/v1/deployer`, `/api/v1/contract`, `/api/v1/org`. If a contract was promoted to confirmed after the classifier last ran, the deployer may still be categorized BENIGN. Latest entity_classification row: 2026-04-12 (5 days), better than average. Recommendation: verify `entity_classifier.py` cron is running daily; investigate 5-day gap. |
| All four sites above | `trust_amplification` (depends on `contracts.confidence_tier` in amplification calc) | NO — producer-recompute only | **HIGH** | API reads `trust_amplification` directly at `/api/v1/contract/{addr}` (api_v1.py:434). 32 rows last computed 2026-03-25 — 23 days stale. Any contract promoted to confirmed since then has an amplification row still pinned to pre-promotion evidence OR no row at all. Recommendation: restart `trust_amplification.py --analyze` cron; investigate why it hasn't written in 23 days. |
| All four sites above | `bytecode_families` / `bytecode_family_members` | NO — rebuild only via `--rebuild` | **MEDIUM** | Family detection_tier depends on member counts per tier. Stale tier labels on members → misstated family classification. API reads bytecode_families at `/api/v1/contract` (api_v1.py:387). 405 families, last computed 2026-03-24. Recommendation: include family-rebuild in the same cron restart. |
| All four sites above | `bait_profiles` | NO — producer-recompute only | **MEDIUM** | Bait classification depends on bytecode patterns + bot interactions. Not directly API-served but appears in case files. |
| All four sites above | `strategy_lifecycle` | NO — producer-recompute only | **LOW** | Aggregate metric only, not served as a verdict. |

### 2. Mutations on `contracts.routing_presence` / `routing_first_seen`

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `db.update_contract_routing` (db.py:703) | `bytecode_cache` | NO | **LOW** | Cache row does not mirror routing_presence. No drift. |
| `db.update_contract_routing` | `trust_amplification` | NO — producer-recompute only | **MEDIUM** | Amplification depends on router routing behavior; new routing_presence=true events should trigger re-analysis of the amplification row. Same stale-cron issue. |

### 3. Mutations on `contracts.bytecode_pattern_notes`

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `db.update_contract_bytecode_signals` (db.py:719) | `bytecode_cache.bytecode_signals` | NO | **LOW** | Currently dead code (no callers). Document or remove. If resurrected, must invalidate cache (mirrors the same JSON payload). |
| `backfill_timelocks.py:103` UPDATE | `bytecode_families` | NO — rebuild only | **LOW** | Timelock-bearing contracts get `bytecode_pattern_notes` updated but no tier change. Family stats not affected. |

### 4. Inserts into `trap_events`

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `db.insert_trap_event` (db.py:791) | `contracts.confidence_tier` (auto-upgrade) | **YES** — UPDATE contracts within same function | — | Correctly wired. |
| `db.insert_trap_event` | `bytecode_cache` (via the promotion) | **YES** (shipped Correction #5) | — | N/A |
| `db.insert_trap_event` | `deployer_profiles.confirmed_trap_count` | NO — producer-recompute only | **HIGH** | Same as mutation group (1): deployer confirmed count is a key risk input. If a deployer's third contract gets confirmed today, `deployer_profiles.confirmed_trap_count` stays at 2 until `deployer_profiler` runs. `/api/v1/deployer` serves the stale count. Recommendation: cron restart. |
| `db.insert_trap_event` | `entity_classification` (criminal labeling) | NO — producer-recompute only | **HIGH** | New confirmed trap → `entity_classification.category` may still be 'benign' until reclassifier runs. |
| `db.insert_trap_event` | `risk_scoring` live path | N/A (live computation) | — | risk_scoring reads `trap_events` live, not a derived table. No invalidation needed. |

### 5. Mutations on `deployers.entity_type` / `funding_trail` / `behavioral_score`

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `auto_funder_tracer.py:119,157,200,270` | `entity_classification` (gas_station category derived from entity_type) | NO — producer-recompute only | **HIGH** | API at `/api/v1/deployer`, `/api/v1/org` both serve entity_classification. When auto_funder promotes a deployer to `entity_type='gas_station'`, the classification row stays until reclassifier runs. |
| `pattern_scanner.py:409` | `entity_classification` | NO | **MEDIUM** | Same shape. |
| `longitudinal_scorer.py:336` (behavioral_score) | `entity_classification`, `deployer_profiles`, `/api/v1/deployer` | Producer itself updates `deployer_profiles` atomically, but downstream consumers don't re-read | **LOW** | behavioral_score is a feature; consumers use it in their own recomputations. |
| `db.update_deployer_notes` (db.py:529) | (no derived consumer found) | N/A | **LOW** | Notes are an analyst field, not a computed input. |
| `db.update_deployer_funding` (db.py:538) | `entity_classification` (funding_source derivatives) | NO | **MEDIUM** | Stale classifier. |

### 6. Continuous inserts (`transaction_events`, `alerts`)

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `transaction_events` INSERT (selector_monitor, etc.) | `camouflage_metrics` (revert rate aggregates) | NO — producer-recompute only | **HIGH** | `camouflage_tracker.py --compute-today` writes `camouflage_metrics`. Last row 2026-03-25 (23 days). Customers quoting "70-79% camouflage ratio" are citing a March-25 number. Recommendation: cron restart. |
| `transaction_events` INSERT | `behavioral_baselines` / `behavioral_anomalies` | NO — producer-recompute only | **MEDIUM** | Statistical norms drift as ingest continues. |
| `transaction_events` INSERT | `daily_metrics`, `predictions` | NO — producer-recompute only | **MEDIUM** | `trend_forecaster` hasn't written since 2026-03-25. |
| `alerts` INSERT (continuous) | `/stats` `recent_alerts` (reads alerts live) | N/A (live read) | — | Fine — live. |
| `alerts` INSERT | `detector_precision` (derived) | NO — producer-recompute only | **LOW** | Measurement of our own false-positive rate. Stale measurement harms self-audit, not customer. |

### 7. Mutations that touch `bytecode_cache.hit_count`

| mutation_site | downstream_table | invalidation_present | risk_level | recommendation |
|---|---|---|---|---|
| `db.cache_lookup` (db.py:857) increments hit_count | (no consumer that would go stale) | N/A | **LOW** | Metric-only. Not served. |

---

## Pattern summary

Two classes of invalidation gap surfaced, only one of which is the same class as Correction #5:

**Class A — Event-driven cache that must invalidate on source mutation.** Exactly one such structure in the codebase: `bytecode_cache`. Fixed in Correction #5. No siblings found.

**Class B — Producer-recompute derived tables that are only as fresh as the producer's last run.** Ten such tables (`deployer_profiles`, `deployer_similarity`, `entity_classification`, `trust_amplification`, `bytecode_families`, `bytecode_family_members`, `daily_metrics`, `camouflage_metrics`, `predictions`, `trust_amplification`, plus smaller `bot_strategies` / `bait_profiles` / `strategy_lifecycle` / `behavioral_*`). The invalidation "happens" by the producer overwriting the table on its next run. In practice almost all producers have not run in 22+ days. **This is the bigger operational issue and it is not a code bug — it is a scheduled-job configuration issue.**

Neither class produces per-API-call errors today because:
- For Class A, Correction #5 is in place and the 641 stale rows are queued for backfill.
- For Class B, the API serves whatever is in the derived table. If the producer last ran on 2026-03-25, the answer is "what the world looked like on 2026-03-25" — which is a defensible Tier B answer epistemically as long as the API discloses the computation timestamp. It currently does not.

---

## Items for decision (not yet done — awaiting review)

1. **Cron / scheduler on Railway.** Identify why `trust_amplification`, `camouflage_tracker`, `deployer_profiler`, `bytecode_families`, `trend_forecaster`, `strategy_fingerprint` have not written in 22+ days. Either re-enable the schedule or declare these outputs archived (with disclosure on API).
2. **API freshness headers.** Every endpoint that reads a derived table should return a `computed_at` timestamp in the response so customers can see when they are reading stale analysis.
3. **`risk_scores` table documentation.** CLAUDE.md mentions `risk_scores` as an existing table with "Computed" rowcount. It does not exist. Either (a) create and persist it as documentation implies, or (b) correct CLAUDE.md to say scoring is live-computed.
4. **Within-pipeline invalidation for entity_classification + deployer_profiles on `contracts.confidence_tier` / `deployers.entity_type` mutations.** These are the HIGH-risk Class B items — they feed user-facing verdicts. If cron stays unreliable, consider making the relevant API readers recompute from source on the hot path (losing some latency but gaining freshness).

---

## Scope of what this audit did NOT cover

- Correctness of the producer logic itself — only whether producers re-ran.
- Whether derived tables are indexed properly for API latency — separate concern.
- Writes from `run_surveillance.py` one-shot maintenance tasks (e.g., the periodic `DELETE FROM bytecode_cache WHERE confidence_tier='unknown'` at line 1395) — those are schema-hygiene, not derived-data invalidation.
- The x402-family (`x402_events`, `x402_facilitators`, `x402_permit2_exposure`) — handled separately by the x402 write-tracer diagnostic (awaiting 24h Railway deployment).
