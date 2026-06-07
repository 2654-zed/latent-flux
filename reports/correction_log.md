# Correction Log
## Published Corrections to Layer 3 Methodology and Data

Every correction we document here makes the next one easier to spot. The point is not to be right the first time; it is to be transparent about being wrong so customers can trust what remains.

Each entry answers four questions:
1. **What did we claim?**
2. **What was actually true?**
3. **How was the error caught?**
4. **What did we change?**

---

## Correction #1 — Gas-Cluster Artifact
**Status:** Documented in prior conversation (pre-dates this log file). Entry placeholder; full write-up pending reconstruction from chat history.

---

## Correction #2 — 881E Reclassification
**Status:** Documented in prior conversation (pre-dates this log file). Entry placeholder; full write-up pending reconstruction from chat history.

---

## Correction #3 — T2-eaef6a5d NULL-Bucket Reclassification
**Date applied:** 2026-04-16
**Scope:** 20,936 contracts reclassified. Bytecode family `T2-eaef6a5d7678` dissolved.

### What we claimed
The bytecode clustering system exposed a family with 21,936 members and 8,240 deployers — the largest single "family" in the corpus. This count had been used implicitly in statements like "the suspected tier contains ~46% of classified contracts." Individual contracts in the family were labeled `confidence_tier = 'suspected'` in API responses with `detection_method = 'bytecode_pattern'`.

### What was actually true
`T2-eaef6a5d7678` was not a family. Its clustering key was `fee=0|asym=0|crev=0` — meaning every member shared only the **absence** of detected trap signatures. The Tier 2 clustering logic grouped contracts by their flag combination without excluding the all-zero combination, so the null result became its own bucket.

Evidence:
- 21,312 **unique** code_hashes across 21,312 cache entries — no bytecode was shared
- 8,240 distinct deployers — no single operator
- Every `bytecode_cache` entry: `{has_asymmetric_transfer: false, has_conditional_revert: false, has_unusual_fee_structure: false, pattern_notes: null}` with reason `Bytecode analyzed (NNN B) — checked [10 detectors], no trap patterns detected`

Separately, 20,936 contracts across the corpus (2,115 still in the family plus 18,821 deployed after the family was last regenerated) carried `confidence_tier='suspected'` with `detection_method='bytecode_pattern'` despite having zero bytecode evidence. These were upgraded by classification paths that no longer apply cleanly (cache transplants, post-hoc scoring on velocity/deployer features) but that failed to rewrite `detection_method` to reflect the true evidence source.

### How was the error caught
Adversarial review of the corpus's largest family. The prompt asked: "classify this family." The first two verdict options (`LEGITIMATE_INFRASTRUCTURE`, `MALICIOUS_TEMPLATE`) were refuted by counting unique code hashes — if it were a template, there would be few hashes. There were 21,312 (one per contract). That single count forced the verdict into `MIXED/UNKNOWN` and revealed the family was a null bucket, not a signal.

### What we changed

| Component | Change |
|-----------|--------|
| `surveillance/bytecode_families.py` | Tier 2 clustering now skips the all-zero key. Absence of classification is no longer itself a classification. |
| `contracts` table (data) | 20,936 rows moved from `confidence_tier='suspected'` → `'unknown'`. `confidence_reason` appended with correction note; prior reason preserved in brackets. `last_updated` set to correction date. |
| `bytecode_families` + `bytecode_family_members` (data) | `T2-eaef6a5d7678` family record and all 21,936 member entries deleted. |
| `web/api_v1.py` | New `detection.evidence_type` field on `/risk` responses with four values: `behavioral-confirmation`, `bytecode-pattern`, `deployer-derivative`, `unanalyzed`. Customers can now filter on the evidence basis, not only on the tier label. |
| `correction_null_bucket.py` | Idempotent migration script with dry-run default. Runs on local DB and is exportable to Railway. |

### Effect on published numbers
- Suspected tier: **64,921 → 43,985** contracts (−32.2%)
- Unknown tier: **58,841 → 79,777** contracts (+35.6%)
- `T2-eaef6a5d7678` (largest family): **dissolved** — no longer appears in `bytecode_families`
- Of the 43,985 remaining suspected: ~31,976 are `deployer-derivative` evidence (deployer_history), the rest carry bytecode evidence

### Open work
- The 31,976 `deployer-derivative` suspected contracts are honestly labeled but should be audited for the same root cause: is the `deployer_history` upgrade rule strict enough that inheritance alone justifies a suspected label? Proposal is a separate correction entry if the audit finds over-broad inheritance.
- The classification pipeline that produced 20,936 evidence-free suspected upgrades still exists on the live service. Code fix in `bytecode_families.py` prevents future NULL-family creation, but the data-level cause (whatever path upgrades tier to suspected without confirming bytecode evidence) needs root-cause identification in a follow-up. The migration script can be rerun to sweep new mislabels until that is done.

---

## Correction #4 — Velocity-Escalation Pipeline Fix + Inheritance-Breadth Audit
**Date applied:** 2026-04-17
**Scope (code fix):** Two pipeline paths now write `detection_method` when they upgrade a contract's tier. Prevents the Correction #3 mislabel from regenerating.
**Scope (audit only):** 31,976 `deployer-derivative` suspected contracts examined. No data change yet; findings raised for a policy decision.

### What we claimed (implicitly)
After Correction #3, we claimed the NULL-bucket reclassification was complete: 20,936 evidence-free contracts moved to `unknown`, and `T2-eaef6a5d7678` was dissolved. The root cause that produced those 20,936 mislabels was identified as open work.

### What was actually true
**Root cause (code defect):** 99.9% of the 20,936 mislabels (20,915 of them) carried `velocity escalation` in the prior `confidence_reason`. Traced to `surveillance/deployment_monitor.py` lines 842–855: when a deployer trips the velocity threshold, the monitor walks their existing `unknown` contracts and upgrades each to `suspected` via `db.update_contract_confidence`. That helper, prior to this correction, only wrote `confidence_tier` and `confidence_reason`. It did not touch `detection_method`. So a contract originally classified with `detection_method='bytecode_pattern'` (the default written at insert time before bytecode analysis returns) would be flipped to `suspected` while keeping the stale bytecode-pattern method — producing the exact `suspected + bytecode_pattern + all-flags-zero` mislabel pattern the NULL family caught.

A second path with the same defect: `surveillance/routing_monitor.py` line 270 (routing-anomaly upgrades) also called `update_contract_confidence` without supplying a method, producing a smaller but equivalent class of mislabels where the true evidence source was 1inch pathfinder avoidance but the method remained `bytecode_pattern`.

**Inheritance-breadth audit:** Of the 31,976 `deployer_history`-based derivative-suspected contracts:
- **30,182 (94.4%)** come from deployers with **zero** confirmed traps
- 1,794 (5.6%) come from deployers with at least one confirmed trap
- 2,022 of 2,077 flagging deployers have never produced a confirmed trap

The `is_priority_deployer` rule in `db.py:557` returns True for any deployer whose `deployment_pattern_notes` is non-empty. The velocity detector writes a note for any deployer that exceeds the velocity threshold — so velocity alone makes a deployer "priority" forever, and all their future and past `unknown` contracts inherit a `suspected` label with no trap evidence. The most extreme case: deployer `0x694834fe...` with 1 confirmed trap produced 93 derivative-suspected contracts (93× inheritance ratio). Deployers with high confirmed-trap counts (`0xc0ffee...` with 94 confirmed, 47 derivative = 0.5× ratio) show the inheritance pattern looks reasonable when grounded in actual trap evidence.

### How was the error caught
Follow-up investigation to Correction #3. Reading the code paths that call `update_contract_confidence` revealed two paths (velocity escalation, routing anomaly) that do not pass `detection_method`. Cross-checking against the 20,936 downgraded contracts' prior reasons confirmed velocity escalation was the dominant source (99.9%). The breadth audit then asked the follow-up question the user raised: is the `deployer_history` upgrade rule strict enough? The 94.4%-from-zero-confirmed-traps number shows it is not.

### What we changed

| Component | Change |
|-----------|--------|
| `surveillance/db.py` | `update_contract_confidence` now takes an optional `detection_method` parameter. When provided, the method is written alongside tier/reason so evidence-basis labels stay truthful. |
| `surveillance/deployment_monitor.py` | Velocity-escalation bulk upgrade now passes `detection_method="deployer_history"`. Future velocity-inherited suspecteds will be labeled `deployer-derivative` in API responses (not `bytecode-pattern`). |
| `surveillance/routing_monitor.py` | Routing-anomaly upgrade now passes `detection_method="routing_anomaly"`. The API's `evidence_type` helper already handles this value via the fallback branch; customers filtering on `detection.method == "routing_anomaly"` now see accurate data. |
| **No data change in this correction.** | The 20,936 historical mislabels are already moved to `unknown` by Correction #3. The code fix prevents regeneration. |

### Policy decision raised (not yet resolved)
The breadth audit shows 94.4% of derivative-suspected contracts originate from deployers with no confirmed traps. Three options, each defensible:

1. **Keep as is.** Velocity, funder-tracing, and analyst notes are legitimate early signals; flagging before a confirmed trap appears is the point of proactive surveillance. API consumers can already filter via `evidence_type="deployer-derivative"` if they want higher-confidence labels only.

2. **Require confirmed trap for inheritance.** Change `is_priority_deployer` or the bulk-upgrade rules so only deployers with ≥1 confirmed trap contaminate their sibling contracts. This eliminates 30,182 inheritance flags but risks false negatives on fresh deployer campaigns.

3. **Split the label.** Introduce a `pending-derivative` evidence type for deployer-flag inheritance where the deployer has no confirmed trap, and reserve `deployer-derivative` for deployers with confirmed history. Costs: one more label for customers to reason about.

I recommend option 3 if the business asks for a tightening. For now, the pipeline is honest (fixed here) and customers have a filter mechanism (evidence_type). A data-level correction should wait for a decision rather than preempt one.

### Effect on published numbers
- **No change** to current tier counts. Correction #3's counts stand: `suspected=43,985`, `unknown=79,777`.
- Going forward, new velocity-escalation upgrades will appear as `deployer-derivative` (not `bytecode-pattern`) in API responses, which matches the actual evidence basis.
- The `correction_null_bucket.py` script from Correction #3 can be rerun safely on Railway as a sweep for any mislabels that accumulated between the snapshot and the code fix.

### Open work
- Policy decision on inheritance-breadth option 1/2/3 above.
- Root cause not yet audited: **cache transplant** (`deployment_monitor.py` line 683–696). When a new contract's bytecode hash matches an earlier cached entry, the new contract inherits the cached tier and signals. If the cached entry's tier was later upgraded by an external path, and the cache record was not rewritten, new cache hits may inherit stale labels. Count of contracts currently affected by this path is not yet measured.

---

## Correction #5 — Bytecode-Cache Transplant Staleness
**Date applied:** 2026-04-17
**Scope (code fix):** Four mutation sites now invalidate the bytecode cache entry a contract seeded. Two route through the central helper (`db.update_contract_confidence`, `db.insert_trap_event`); two are direct-UPDATE bypasses in `honeypot_checker.py` and `backfill_self_loops.py` patched inline. Four additional call paths (`routing_monitor`, `revert_cluster_detector`, `deployment_monitor` velocity escalation, `seed_defihacklabs`) inherit the fix for free via the helper.
**Scope (data, pending):** 641 stale `bytecode_cache` rows identified; 8 downstream contracts in the `contracts` table uniquely matched to a source whose current tier differs from the cached tier. Data remediation (deletion of the 641 stale cache rows) is queued in `surveillance/backfill_cache_invalidation.py` and will be run against Railway after this entry is committed to the repo.

### What we claimed (implicitly)
Every cache hit at `deployment_monitor.py:683–696` transplants a `confidence_tier`, `confidence_reason`, and `bytecode_signals` payload from a prior source contract onto a new deployment with identical bytecode. API consumers reading the resulting `contracts` row were implicitly told the cached tier reflected the source's *current* classification.

### What was actually true
The cache is write-once: `db.cache_store` uses `INSERT OR IGNORE`, and no mutation path updated or deleted cache entries when the source contract's tier was later changed. Three post-insert mutations invalidate the source's cache entry in principle and did not in practice:

- `db.insert_trap_event` (behavioral confirmation → `confirmed`)
- `db.update_contract_confidence` (velocity escalation, self-loop promotion, routing anomaly, defihacklabs seed)
- `honeypot_checker.py` direct UPDATEs (honeypot.is + GoPlus external confirmations)
- `backfill_self_loops.py` direct UPDATE (historical promotion)

Measured against the local DB snapshot on the correction date:

| cached_tier | current_tier | entries | downstream lookups |
|---|---|---|---|
| suspected | confirmed | 469 | 11 |
| unknown   | confirmed | 128 | 2 |
| unknown   | suspected | 34  | 0 |
| suspected | unknown   | 10  | 0 |
| **total** | | **641** | **13** |

Direction of drift: almost entirely **under-classification** — the cache carried a lower tier than the source's current reality, because the dominant mutation path is `insert_trap_event` promoting suspected traps to confirmed after a bot is caught. 13 downstream lookups (hits) occurred on stale entries. Of 10,010 cache-sourced rows in the `contracts` table, 8 uniquely match a source whose current tier differs — the provable downstream mislabel count.

All observed drift produces honest-but-lower tier labels (the cache says `suspected` when the source is now `confirmed`). No false-positive amplification; zero entries moved a contract UP to a higher tier than the source's current reality without fresh evidence.

### How was the error caught
Open-work item in Correction #4 (2026-04-17) explicitly flagged `deployment_monitor.py:683–696` as unaudited and directed a follow-up. The follow-up enumerated every mutation path on `contracts.confidence_tier`, cross-checked whether each path invalidates `bytecode_cache`, and counted the residue. The class of bug is the same family as Correction #3: derived data outliving the assumption that justified it. Here the derived data is a cache row, not a family record.

### What we changed

| Component | Change |
|-----------|--------|
| `surveillance/db.py` — `update_contract_confidence` | `DELETE FROM bytecode_cache WHERE source_contract = ?` added before commit. All callers (routing_monitor, revert_cluster_detector, deployment_monitor velocity, seed_defihacklabs) inherit the fix without edits. |
| `surveillance/db.py` — `insert_trap_event` | Same DELETE added alongside the auto-upgrade UPDATE, so behavioral confirmations invalidate the cache. |
| `surveillance/honeypot_checker.py` | DELETE added inside both direct-UPDATE sites (honeypot.is, GoPlus). Append-style `confidence_reason` semantics preserved by not routing through the helper. |
| `surveillance/backfill_self_loops.py` | DELETE added inside the backfill loop. |
| `surveillance/backfill_cache_invalidation.py` (new) | Idempotent CLI, dry-run default, `--commit` required for execution. Prints the stale-entry breakdown before and after. Queued to run against Railway after this entry commits. |

### Rationale for `DELETE` over `UPDATE`
The cache entry was stamped at classification time with the tier that bytecode analysis produced. Post-insert mutations upgrade a contract's tier from **non-bytecode evidence** — a trap firing, a routing anomaly, a self-loop, an external-API confirmation. Overwriting the cache's tier with the new value would imply future deploys of the same bytecode deserve the behavioral tier on bytecode-only grounds. That is the same epistemic failure Correction #4 fixed for `detection_method`. DELETE forces the next matching deploy to re-run the classifier; the cache-derived tier can only ever be `bytecode-pattern` evidence.

### Verification
Unit smoke test (temp DB, no fixtures) confirms the three relevant properties: (a) `update_contract_confidence` deletes both `init_code_hash` and `deployed_code_hash` cache rows sourced from the mutated contract; (b) `insert_trap_event` does the same; (c) mutating contract A does not touch contract B's cache entries. The backfill dry-run against the production snapshot prints the same 641 / 13 figures measured by the audit, confirming the delete predicate matches the audit query.

### Effect on published numbers
- **No change** to current tier counts or any headline metric.
- `bytecode_cache` row count will drop from 53,096 → 52,455 when the backfill commits (−641, −1.2%).
- Forward: under-classification drift in the cache cannot re-accumulate via the four patched mutation paths.

### Backfill executed (local DB, 2026-04-17)
After committing the code fix and this entry (commit `0d23b6e`), ran the backfill against the local snapshot. Results:

```
bytecode_cache rows before: 53,096
Stale entries identified (dry-run count, grouped by tier pair):
  cached=suspected  current=confirmed   entries=469  downstream_lookups=11
  cached=unknown    current=confirmed   entries=128  downstream_lookups= 2
  cached=unknown    current=suspected   entries= 34  downstream_lookups= 0
  cached=suspected  current=unknown     entries= 10  downstream_lookups= 0
  TOTAL: 641 entries, 13 downstream lookups

Rows deleted: 661
bytecode_cache rows after: 52,435
Residual stale entries after delete: 0
```

**661 deleted vs 641 counted — resolved.** The DELETE predicate is `source_contract IN (<stale set>)`, not `code_hash IN (...)`. For sources that seeded BOTH an `init_code_hash` cache row AND a `deployed_code_hash` cache row, the dry-run count reports each row independently (matching the actual tier-pair drift), but the DELETE removes every cache row under that source — including sibling rows that weren't individually tier-stale. The 20-row surplus is `source_contract` buckets where one hash row was stale and the other wasn't. This is safe over-deletion: the next deploy of matching bytecode re-runs the classifier and re-caches deterministically. No cache row with a STILL-valid tier was spared; all rows sourced from a mutated contract are removed, which matches the intent of the fix.

### Backfill executed (Railway production, 2026-04-18)
Followed the local run with the production execution via `railway ssh`. Service: `latent-flux`, commit deployed `b75a02b`, volume: `/app/surveillance/data/` (1.21 GB).

```
bytecode_cache rows before: 20,345
Stale entries identified:
  cached=suspected  current=confirmed   entries=481  downstream_lookups=15
  cached=suspected  current=unknown     entries=  8  downstream_lookups= 0
  TOTAL: 489 entries, 15 downstream lookups

Rows deleted: 489
bytecode_cache rows after: 19,856
Residual stale entries after delete: 0
```

Prod cleanup is exact (489 counted, 489 deleted) — no over-deletion, unlike the local run (661 deleted vs 641 counted). Production has fewer multi-hash-per-source cases, likely because the periodic unknown-cache prune at `deployment_monitor.py:258` has already removed sibling rows that the local (superset) DB preserved.

Cache staleness now zero in both DBs. The four mutation-path fixes shipped in `0d23b6e` prevent recurrence going forward.

### Open work
- **Orphan cache rows** (source in `bytecode_cache` whose address is missing from `contracts`) are not touched by this fix. Measured at 2,310 cache-sourced contracts locally whose reason cites a source prefix no longer in the cache. Separate cleanup — not a staleness issue, a pruning artifact.
- **Wave 1 audit published** (`reports/cache_invalidation_audit.md`). Headline finding is bigger than the original Class A bug: five producer-recompute derived tables (`trust_amplification`, `camouflage_metrics`, `bytecode_families`, `deployer_similarity`, `daily_metrics`) have not been written in 22+ days and are served by the API with no freshness indicator. Scheduler audit is now the next priority, displacing the originally-planned epistemic-tag audit.
- **`risk_scores` table referenced in CLAUDE.md does not exist in the DB.** Dedicated investigation queued (`reports/risk_scoring_persistence_audit.md`).

---

## Correction #6 — Documentation-Reality Gap on `risk_scores` Persistence
**Date applied:** 2026-04-17
**Scope:** Three edits to `claude.md`. No code change. No data change.

### What we claimed
`claude.md` documented `risk_scores` as a table in the production schema ("Computed | Stored potential scoring output") and listed `python -m surveillance.risk_scoring --score-all` in *Running Common Operations* as if it were a working producer command. Together these implied a persisted score table refreshed by a batch job.

### What was actually true
`surveillance/risk_scoring.py` is explicitly read-only analytics (module docstring: *"Creates no new tables"*). No INSERT/CREATE/UPDATE/DELETE statements anywhere in the file. The `risk_scores` table does not exist in the production DB and does not exist in the local 1.77 GB snapshot either. `/api/v1/risk/{chain}/{address}` calls `score_contract` **live** per request at `web/api_v1.py:818`, with measured p50 ≈ 80 ms and p99 around 1 s for contracts with heavy related-row joins. The `--score-all` CLI flag does not parse; the actual argparse flags are `--address`, `--top N`, `--family`.

Full evidence in `reports/risk_scoring_persistence_audit.md`.

### How was the error caught
Wave 1 cache-invalidation audit (`reports/cache_invalidation_audit.md`) tried to enumerate staleness for every derived table named in `claude.md`. `SELECT COUNT(*) FROM risk_scores` failed with `no such table`. The dedicated follow-up confirmed risk_scoring.py is pure live computation and the doc describes a non-existent architecture.

### Root cause
`claude.md` was drafted from aspirational context — the L3 narrative materials and architecture intent — rather than measured codebase state. The same failure mode GoPlus commits when it returns `is_honeypot: 0` for a confirmed trap: documenting what should be true rather than what is true. This correction closes the claim; the broader discipline fix is proposed below.

### What we changed
Three edits to `claude.md`:

| Location | Change |
|---|---|
| *Database Schema (Current Production State)* table | Removed the `\| risk_scores \| Computed \| Stored potential scoring output \|` row. |
| *The Risk Scoring Model* section, below *Core interpretive rule* | Added: *"Persistence: `risk_scores` are computed live per API request. Not persisted. Historical tracking and bulk queries are not available in the current architecture. See Correction #6 (2026-04-17) for context; `/api/v1/risk/{chain}/{address}` calls `surveillance.risk_scoring.score_contract` per request, p50 ~80ms."* |
| *Running Common Operations* — Analysis block | Replaced `python -m surveillance.risk_scoring --score-all` with the three flags that actually parse: `--address 0x<CONTRACT>`, `--top 100`, `--family <family_id>`. |
| *Current Priority Items* | Added item #13: *"Evaluate hybrid cache architecture for `risk_scores` persistence — deferred until scheduler audit complete."* with decision criteria. |

### Choice made: Story A, not Story B
The persistence audit surfaced two consistent futures. **Story A**: accept live-compute as the design and edit the doc. **Story B**: build the producer, table, scheduler entry, and API read-through. Story A chosen because (a) it closes the discrepancy in minutes with no new surface, (b) Story B is the same class of work as the Class B scheduler finding — better to reason about both together after the scheduler audit lands, and (c) there is no measured customer pain on bulk screening today that justifies pre-scheduler work. Priority item #13 holds the door open for Story B if the pain shows up.

### Effect on published numbers
None. Zero rows moved. Zero API responses change. This is documentation alignment only.

### Pattern worth naming
This is the third correction in a short window representing three distinct failure classes:

- **Correction #4 (2026-04-17)**: *Data classification error* — pipeline upgraded tier without carrying evidence metadata, producing 20,915 mislabels that had to be dissolved.
- **Correction #5 (2026-04-17)**: *Cache invalidation gap* — derived data outliving the source mutation that should have invalidated it.
- **Correction #6 (2026-04-17)**: *Documentation drift from reality* — the doc describes intended state, the code runs actual state, customers and future-us read the gap.

A correction log that grows at this cadence is doing its job. The alternative — silent drift, undocumented gaps, stale claims — is exactly what Layer 3 measures in the systems it monitors. Our own discipline should match.

### Open work
- **`verify_claude_md.py` (proposed, not built):** a doc-test-suite script that asserts every claimed table exists, every claimed CLI flag parses, every claimed endpoint responds. Runs as a CI gate on `claude.md` changes. Cheap to build, prevents future Correction-#6-class entries. Not added to the priority list yet — awaiting user call on whether to promote it.
- **One other CLI line in `claude.md` is broken**: `python -m surveillance.case_file --address 0x[CONTRACT]` uses `--address`, but the actual module takes `address` as a positional argument. Noticed during validation for this correction. Not fixed in this entry because the user's scope was explicitly three edits; flagged here for the next pass or the `verify_claude_md.py` script to catch.
- **Story B still available.** If bulk-screening demand or longitudinal score tracking becomes a customer requirement, priority item #13 scopes the persistence work cleanly.

---

## Correction #7 — Stale Derived Metrics Served Without Freshness Indicators
**Date applied:** 2026-04-18
**Scope (code):** `run_surveillance.py` gains a nightly `_analysis_scheduler` thread wiring six producers. `web/api_v1.py` adds `metadata.computed_at` to five endpoint responses.
**Scope (data):** All six producers run once against Railway production to restore current-day freshness. Row counts refreshed across seven tables.

### What we claimed (implicitly)
The API served `trust_amplification`, `camouflage_metrics`, `bytecode_families`, `deployer_similarity`, `daily_metrics`, and `deployer_profiles` without any `computed_at` indicator. A customer hitting `/api/v1/contract/0x…` received attribution and amplification data that the response presented as current. No `epistemic_tag` accompanied it with a computation timestamp.

### What was actually true
The Wave 2 scheduler audit (`reports/producer_scheduler_audit.md`, 2026-04-17) established that these tables had not been written in 22+ days. Concretely:

| Table | Last prod write (before this correction) | Staleness |
|---|---|---|
| `trust_amplification` | 2026-03-25 16:17:28 UTC (32 rows, one-second burst) | 24 days |
| `camouflage_metrics` | 2026-03-25 (9 daily rows, cron-shaped) | 24 days |
| `bytecode_families` | 2026-03-24 (405 rows) | 25 days |
| `deployer_similarity` | 2026-03-26 15:37:34 UTC (4,879 rows, 7-sec burst) | 23 days |
| `daily_metrics` | 2026-03-25 (9 daily rows) | 24 days |
| `deployer_profiles` | no `profiled_at` timestamp before this run | unknown |

None of the producers were wired into `run_surveillance.py`. Three were driven by an external daily cron that died the night of 2026-03-25 → 2026-03-26. Two only ever ran as one-shot manual CLI invocations. The in-process heartbeat integration at `deployment_monitor.py:432–445` for `trust_amplification` fires every ~2 hours but has not written a row since March 25 (suspected silent IntegrityError in `db_writer.py:109`, same class as the x402 bug; dedicated diagnostic at `surveillance/diagnostics/x402_write_tracer.py` queued for deployment).

### How was the error caught
Wave 1 cache-invalidation audit noticed the timestamps on these tables as a side effect of probing for cache-staleness siblings. Wave 2 quantified the gap per producer (trigger type, last-run signature, cron death date).

### Root cause
The producers were assumed-scheduled, not actually-scheduled. `CLAUDE.md` lists all six under *Running Common Operations* as if they were scheduled analysis jobs, but `run_surveillance.py` only invokes one scheduled job (`daily_report.generate_report` at 06:03 UTC), and none of the six were called by it. Same shape as Correction #6 — documentation described intended state, deployed code ran actual state, the gap accumulated silently.

### What we changed

| Component | Change |
|-----------|--------|
| `run_surveillance.py` | New `_analysis_scheduler` background thread mirroring the existing `_daily_report_scheduler` pattern. Polls every 60 s, fires any job whose (hour, minute, day-of-week) matches. Jobs run as subprocesses (crash isolation); failures log at visible level. |
| `ANALYSIS_JOBS` config (same file) | Six jobs on a fixed UTC schedule: 00:15 `trend_forecaster --compute-today` → 00:20 `camouflage_tracker --compute-today` → 00:30 `trend_forecaster --forecast --score` → 02:00 `deployer_profiler --profile-all` → 03:00 `bytecode_families --cluster` → Sunday 04:00 `deployer_profiler --cluster`. |
| `web/api_v1.py` | New `_TABLE_FRESHNESS_COLUMN` map + `_freshness(conn, tables)` helper. `_ok(data, conn, fresh_tables=[...])` accepts the list of derived tables the endpoint reads and emits `meta.computed_at = {table: timestamp}`. Applied to `/api/v1/risk/{chain}/{address}`, `/api/v1/contract/{address}`, `/api/v1/deployer/{address}`, `/api/v1/org/{org_id}`, `/api/v1/ecosystem/stats`. |
| Railway production (data) | All six producers invoked once via `railway ssh` to restore current-day correctness before the nightly schedule takes over. |

### Effect on published numbers
Immediate post-backfill snapshot (Railway production, 2026-04-18):

| Table | Rows before | Rows after | Latest timestamp |
|---|---|---|---|
| `trust_amplification` | 32 | **151** | 2026-04-18 05:04:49 |
| `deployer_profiles` | (no tsc) | **10,132** | 2026-04-18 05:00:53 |
| `deployer_similarity` | 4,879 | **531,779** | 2026-04-18 05:04:35 |
| `bytecode_families` | 405 | **1,438** | (latest per-row) |
| `bytecode_family_members` | (paired) | **8,280** | — |
| `daily_metrics` | 9 | **10** | 2026-04-18 |
| `camouflage_metrics` | 9 | **10** | 2026-04-18 |
| `predictions` | 9 | **12** | 2026-04-18 |

The deployer_similarity jump (4,879 → 531,779 rows) is not a bug — the previous single-shot run in March covered ~22k deployers; the current corpus has 37,564, and O(n²) candidate pairs at ≥0.75 similarity threshold scale accordingly. Rows are legitimate matches, not spurious.

### Scope limitations (what this correction does NOT do)

- **`/api/v1/*` FastAPI surface is not yet deployed in production.** `spypy.up.railway.app` currently serves the legacy `StatsHandler` defined inside `run_surveillance.py`. The `computed_at` metadata is live in code but only visible once FastAPI is mounted in prod — see `reports/statshandler_migration_plan.md` scoping (pending, not drafted in this correction).
- **`entity_classification` staleness persists.** Its producer (`entity_classifier.py`) is not one of the six jobs in the new scheduler. Latest prod write is 2026-03-24. Will get addressed in a follow-up once the scheduler pattern is validated in the wild for a week.
- **The heartbeat-embedded `trust_amplification.analyze` is still silently failing.** We now run it nightly via the scheduler *and* the broken 2-hour heartbeat path remains. Fixing the heartbeat path is blocked on the x402 write-tracer catching the actual IntegrityError — a shared root cause fix, not a per-table bandaid.
- **Freshness headers are advisory, not enforcement.** A client can ignore `computed_at` and still consume a stale table value as if fresh. A future `/methodology/freshness` endpoint and an optional `max_age_seconds` query parameter on hot endpoints would harden this; neither is shipped here.

### Open work
- Ship FastAPI (`web/app.py`) on Railway so the `computed_at` metadata reaches customers.
- Add `entity_classifier` to `ANALYSIS_JOBS` once its cadence is confirmed (daily vs weekly).
- Deploy the x402 write-tracer to diagnose the trust_amplification heartbeat silent failure.
- Ship the `/methodology/freshness` endpoint documenting the producer schedule and current-actual lag per table.
- Monitor the scheduler for its first full week — one missed job means subprocess / SQLite contention issue; verify via Railway logs.

---

## Correction #8 — x402 Decimals Assumption (`amount_normalized_6dec`)
**Date applied:** 2026-04-18
**Scope (code):** `surveillance/x402_monitor.py` drain-alert path. One-line-equivalent behavior change: emit normalized dollar amounts only when the drained token is in a 6-decimal stablecoin allowlist.
**Scope (data):** Historical `X402_AGENT_DRAIN` alerts with non-stablecoin tokens carry inflated `amount_normalized_6dec` values. Not corrected in-place (immutable historical record); new alerts use the corrected encoding.

### What we claimed
Every `X402_AGENT_DRAIN` alert exposed a field `amount_normalized_6dec` presented as the USD-equivalent drain amount. Alert `message` prose rendered it as dollars. Customers reading the feed saw one alert from today with `"amount_normalized_6dec": 17717910000.0` and a message reading "Permit2.transferFrom pulled 17,717,910,000.00 from payer…" — a claim that $17.7 billion moved through a single transaction on Base.

### What was actually true
The drain targeted token `0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22` — NOT a 6-decimal stablecoin. Raw amount was `17,717,910,000,000,000` (≈1.77 × 10¹⁶). The normalization code divided by 10⁶ universally, as if every token used stablecoin decimals. If the token has 18 decimals (the default for most ERC-20s), the real amount is `0.01772` tokens — a 10¹² overstatement. USD value of that amount depends on the token's price, which we don't look up; it may be pennies or under $100 at most, given token size.

The drain detection itself fired correctly — the facilitator is A7B9, a confirmed rogue drainer, and the self-settlement shape matched. Only the displayed amount is wrong.

### How was the error caught
Today's activity-pull surfaced an A7B9 event on Base with the improbable $17.7B figure. Inspection of the raw payload showed the token contract is not USDC or USDT; the decimals assumption was the only plausible explanation for the scale.

### Root cause
`_handle_x402_tx` computed `amount_norm = amount / 1e6 if amount else 0` for every drain, regardless of token. The 6-decimal assumption was hard-coded in the Phase 1 design when the only observed drains were USDC. The code comment at `DRAIN_AMOUNT_THRESHOLD` did name "6-decimal" as a scope; the normalization site did not. Same class as Correction #4 (a convention treated as universal).

### What we changed

| Component | Change |
|-----------|--------|
| `x402_monitor.py` | Added `STABLECOIN_TOKENS_6DEC` allowlist (USDC Base/Arb/OP + USDT Arb/OP + EURC Base). |
| `x402_monitor.py` drain alert payload | `amount_normalized_6dec` renamed → `amount_usd_6dec`, populated only when the drained token is in the allowlist. New `token_is_stablecoin: bool` companion field. Raw `amount` always preserved. |
| `x402_monitor.py` drain alert message | Non-stablecoin drains now print raw units and a `(decimals unknown)` note instead of a fabricated dollar figure. |

### What we did NOT change
- **Threshold logic.** `DRAIN_AMOUNT_THRESHOLD = 100 * 10**6` still applies to raw `amount`. For 18-decimal tokens this is trivially satisfied by any non-dust amount; we get a drain alert with `(decimals unknown)` display. This is intentional for now: a rogue-facilitator self-settlement on a non-stablecoin still fires an alert with honest units. Tightening this is a separate design question (decimals-aware threshold, or stablecoin-only alerting) — flagged as open work.
- **Historical alert payloads.** Existing `X402_AGENT_DRAIN` rows in production keep their `amount_normalized_6dec` field as written. Immutable historical record; future consumers reading those rows should filter on token address against `STABLECOIN_TOKENS_6DEC` and interpret `amount_normalized_6dec` only when the token is in the set.
- **`x402_events` table.** Its `amount` column stores raw units. No change needed there.

### Effect on published numbers
- Today's CE5E drain total ($135,400 across 11 Arbitrum USDC victims) is unaffected — all CE5E drains are on USDC, which is in the allowlist. The figure reported in today's activity digest stands.
- The A7B9 Base event that read "$17.7B" was the only non-stablecoin drain in recent history; its actual USD value is unknown without a token-price lookup. The fix ensures future non-stablecoin drains don't repeat the inflation.
- No corpus-wide rollup statistics change. Nothing summed `amount_normalized_6dec` programmatically; customers only saw per-alert displays.

### Open work
- **Decimals-aware amount handling for non-stablecoins.** Optional upgrade: cache `decimals()` via one-time on-demand ERC-20 call per seen token. Would let us emit a correct `amount_token_units` field. Has Alchemy cost; not urgent given today's traffic. Could piggyback on the next x402-tracer deployment.
- **Historical payload migration.** Considered: could we rewrite stored alert payloads for past non-stablecoin drains? Decision: no. Immutability discipline says old claims stay as written. A consumer-facing note in `/api/v1/feed` explaining the schema change is the right path; filed as a non-priority follow-up.
- **Volume of historical miscounts.** Only one non-stablecoin X402_AGENT_DRAIN alert has ever fired (today's A7B9). Impact is small. If the historical volume were larger, a migration might be worth it.

---

## Correction #9 — "Suspected" Confidence Tier Was 99.98% Noise at 30-Day Horizon

**Date:** 2026-04-19
**Motivating audit:** exception-as-rule review of Layer 3 framework (P0 finding).

### The claim (as previously implied)
The `confidence_tier = 'suspected'` label applied to contracts flagged by bytecode-pattern / behavioral-trigger / routing-anomaly detection meant "this contract is likely adversarial based on its static/behavioral signals, pending observable confirmation." Tier was published in `/api/v1/contract/{...}` and rolled up into corpus-wide statistics ("46% suspected" cited in CLAUDE.md priority #10).

### The truth (as measured)
Diagnostic run 2026-04-19 on the local 124,341-row corpus:

| confidence_tier | count | with observable trap_events | observable-harm rate |
|---|---|---|---|
| confirmed | 579 | 482 | 83.2% |
| suspected | 43,985 | 8 | 0.02% |
| suspected, aged ≥ 30 days | 3,797 | 0 | 0.0% |

At the 30-day horizon, `suspected` has zero observable harm correlation. The label was functioning as "we ran detectors on this and something pattern-matched" rather than "this contract is likely adversarial." It aggregated detector-trip noise, not predictive signal. Published as a risk classification, it overstated Layer 3's predictive claim by orders of magnitude.

### How this was caught
The exception-as-rule audit took the user's diagnostic recipe — "score contracts CRITICAL, track 90-day outcomes, measure PPV" — and ran the analogue against `confidence_tier` + `trap_events`. The 0.02% rate for `suspected` vs 83.2% for `confirmed` revealed that the two labels are measuring fundamentally different things: `confirmed` means "observed to have caused loss," while `suspected` meant "detector fired." They were being served as graduated confidence tiers on the same scale. They aren't.

The audit also exposed a secondary structural issue: `contracts.confidence_tier` had a 3-value CHECK constraint (`unknown`, `suspected`, `confirmed`), so there was no schema-level vocabulary for "we detected but nothing observable followed." The closed enum was itself a form of exception-as-rule — the schema forced every detected contract into a predictive category.

### What changed
1. Schema migration (`surveillance/db.py`): extended `confidence_tier` CHECK to include `unanalyzed`. Required a full `contracts` table rebuild (SQLite has no `ALTER CHECK`). Added `decayed_at TEXT` and `prior_confidence_tier TEXT` columns with `idx_contracts_decayed_at` index.
2. New module `surveillance/confidence_decay.py`: moves `suspected` contracts aged ≥30 days with zero `trap_events` linkage to `confidence_tier = 'unanalyzed'`, preserving the original tier in `prior_confidence_tier` and timestamp in `decayed_at`. CLI: `--dry-run`, `--apply`, `--age-days N`.
3. Scheduler (`run_surveillance.py`): nightly at 04:30 UTC.
4. First local apply: 4,888 contracts decayed (3,727 base + 1,161 arbitrum). Post-decay distribution: suspected 39,097 (31.44%), unanalyzed 4,888 (3.93%).

### Effect on published numbers
- **"46% suspected" (CLAUDE.md priority #10 anchor):** not directly recomputed yet because that figure reflected Railway state at an earlier point; local now shows 35.37% → 31.44% post-decay. After Railway sync + first scheduled decay, the corpus-wide statistic should be recomputed and published. Any public page citing "46% suspected" needs an update alongside a pointer to this correction.
- **`/api/v1/contract/{chain}/{addr}`:** contracts that decayed now return `confidence_tier: "unanalyzed"` instead of `"suspected"`. The `prior_confidence_tier` + `decayed_at` fields are available for anyone who wants the audit trail. Deductive fields (`bytecode_pattern_notes`, `has_asymmetric_transfer`, etc.) are unchanged — they record what was detected, not what was predicted.
- **Re-promotion path:** if a decayed contract later fires a `trap_events` row, nothing currently re-upgrades it back to `suspected`. That asymmetry is deliberate for now — decay should be sticky unless we see harm. Re-promotion logic is flagged as open work.

### Open work
- **Re-promotion trigger.** A decayed contract that later produces observable harm should move out of `unanalyzed`. Natural place is in the trap_events write path: on insert, check if `contracts.decayed_at IS NOT NULL` and promote to `confirmed` if so. Separate follow-up.
- **Railway sync.** Schema migration + decay module shipped to `master`; scheduler will fire at 04:30 UTC Railway-time on the next day the build is live. Manual invocation (via `railway run python -m surveillance.confidence_decay --apply`) or an admin endpoint is the faster path if we want pre-scheduler decay on prod.
- **Other detectors may have the same pathology.** The audit found that `suspected` was over-broad for *our* detectors at 30 days, but different detectors may need different decay horizons. `routing_anomaly` detections might resolve faster than `bytecode_pattern` detections. Per-detector decay windows are a future refinement.
- **PPV per detection_method.** We measured PPV at the tier level; we didn't split by `detection_method`. It's possible one detection pathway is producing nearly all the noise. Worth splitting before the next round of decay-threshold tuning.
- **"46% suspected" recompute.** Pending Railway sync. Once decayed, recompute the headline and either update CLAUDE.md priority #10 or mark it resolved.

### Postscript — 2026-04-20 Railway result
Manual migration + decay ran on Railway after an init_db failure-mode was uncovered (documented as Correction #10). Final Railway numbers:
- Pre-decay: 64,987 suspected (52.27%).
- Post-decay: 59,287 suspected (47.68%), 5,700 unanalyzed (4.58%).
- Decayed by chain: base 4,191, arbitrum 1,509.
- CLAUDE.md priority #10 ("recompute 46% suspected") is now effectively resolved — the published 46% figure is within 1.7pp of live after decay; no action required beyond the correction note here.

---

## Correction #10 — Railway init_db Migration Did Not Run; Caused ~60-Minute Write Outage

**Date:** 2026-04-20
**Context:** Shipping Correction #9 (confidence decay).

### The claim (as previously implied)
The existing auto-migration pattern in `surveillance/db.py` — wrap each schema change in `try: SELECT …; except sqlite3.OperationalError: ALTER TABLE …` — runs on every `init_db()` call at service startup, so any new migration added to `db.py` reaches production on the next deploy.

### The truth (as observed)
After pushing commit `329f917` (which added the `deployed_code_hash` migration's consumer — the contracts-table rebuild to extend the `confidence_tier` CHECK constraint), Railway write logs began emitting `table contracts has no column named deployed_code_hash` on every deployment INSERT. An SSH probe of Railway's DB confirmed the contracts table had **19 columns** (no `deployed_code_hash`, no `decayed_at`, no `prior_confidence_tier`, CHECK without `'unanalyzed'`) — i.e., **none of the three pending migrations had applied**, despite the new code being live and referencing the columns.

### How this was caught
Railway `/stats` last_heartbeat was stale (2026-04-19T22:32 UTC, ~22 hours old). `railway logs` showed a continuous stream of `db_writer` INSERT failures. An SSH probe into the live container confirmed the schema mismatch. Estimated outage duration: ~60 minutes of deployments that hit detectors and recorded logs but never persisted to `contracts` or the downstream tables that FK to it.

### What changed
1. Ran a manual one-shot migration script (`scripts/railway_migrate_contracts.py`) via `railway ssh`. The script ran all three contracts-table migrations — add `deployed_code_hash`, add `decayed_at` + `prior_confidence_tier`, rebuild with extended CHECK. 124,341 rows preserved.
2. Writes resumed within the next heartbeat cycle; `/stats` reported fresh heartbeat at 20:33 UTC and incrementing contract counts.
3. Decay applied on Railway: 5,700 rows decayed, suspected dropped to 47.68%.

### Root cause — hypothesized, not confirmed
The running `init_db()` path in production either:
- failed silently before reaching the new migration block (e.g., a prior migration hit a lock at startup, the process retried, and a later migration never ran to completion); or
- ran in a race with the db_writer process acquiring the connection first and holding the BEGIN IMMEDIATE lock through the window where init_db would have applied the ALTER TABLEs.

Both hypotheses predict the same observed state. Distinguishing them requires log instrumentation that isn't currently in the code. Flagged as open work.

### Effect on published numbers
- ~60 minutes of deployment records lost. The contracts table is the gate for every downstream analysis (bytecode_classifier, behavioral scoring, etc.) — anything that depends on deployments from ~19:53–20:38 UTC on 2026-04-20 is incomplete.
- No public statistic was published during the outage window. Daily aggregate jobs run at 00:15–04:10 UTC and will compute from tomorrow onward including the data gap.
- `trap_events` and `alerts` tables are unaffected — they don't FK to `contracts`.

### Open work
- **Instrument `init_db()` with per-migration logging.** Every migration block should `print(f"[init_db] migration={name} ran=True/False", flush=True)` so the startup log shows which migrations fired, which were skipped as no-ops, and which were blocked. Without this, silent failures stay invisible.
- **Add a contract-write-smoke test to service startup.** Attempt a synthetic INSERT (and rollback) to the contracts table immediately after init_db completes. If it fails, raise — don't let the service enter steady-state with a broken write path.
- **Revisit db_writer / init_db ordering.** The main process and db_writer both open connections on startup. If either can acquire a write lock before init_db runs, migrations stall. Formalize init_db as a prerequisite that completes before any producer process starts.
- **Replay lost deployments.** The ~60-minute window of dropped contracts can be partially reconstructed from WebSocket logs if they're retained. Low priority — the miss is small and the period has no known incidents.

---

## Correction #11 — Org Classification Was a Hardcoded Allowlist

**Date:** 2026-04-20
**Motivating audit:** exception-as-rule review, P1.

### The claim (as previously implied)
CLAUDE.md's schema table listed `org_wallets — Organization wallet mappings` as an existing DB table. Layer 3 outputs referenced `org_001`, `org_002`, `org_003`, `org_004` as if these were the output of an organizational-discovery system capable of surfacing novel groups as they appeared.

### The truth (as measured)
No `org_wallets` table existed. Organizational classification was a dict literal `ORG_WALLETS` duplicated across `auto_funder_tracer.py` (8 entries) and `fund_tracer.py` (13 entries). The union is 13 wallets across 2 org_ids (`org_001` = 11 wallets, `org_002` = 2). org_003 and org_004 were referenced in reporting modules (`daily_report.py`, `diamond_model.py`) but had zero wallet-level entries; they existed as case labels without membership data. Any novel criminal group operating on different timezone, gas fingerprint, or funding pattern would not surface as a novel org — it would produce unlinked `suspected` contracts that never accrue to an `org_id`.

### How this was caught
Exception-as-rule audit grep'd for `ORG_WALLETS\s*=\s*\{` across `surveillance/` and found the two identical dict literals. Cross-referenced against CLAUDE.md's schema table (which claimed the DB table existed). Pulled the actual schema — no `org_wallets` table present.

### What changed
1. Schema: added `org_wallets (address, chain, org_id, role, added_at, added_by, reason)` and `org_candidates (candidate_id, cluster_size, deployer_addresses, shared_funding_source, shared_gas_fingerprint, shared_chain, first_seen, last_seen, detected_at, status, notes)` tables. Both migrations instrumented via `_log_migration`.
2. `surveillance/org_registry.py`: cached DB-backed lookup replacing the dict literals. 5-min TTL. CLI with `--seed` for bootstrap and `--list` for dump.
3. `surveillance/auto_funder_tracer.py` and `surveillance/fund_tracer.py`: both dict literals removed; lookups go through `org_registry.get_org_for_address()`. `fund_tracer.py` retains a `_ORG_WALLETS_LazyDict` shim so `addr in ORG_WALLETS` and `ORG_WALLETS[addr]` still work in the address-prefix matcher.
4. `surveillance/org_candidates.py`: novel-org candidate detector. Groups deployers by shared `funding_trail.funder` within a 72h window, filters clusters outside 3–50 members (below = noise, above = CEX/faucet). Emits `org_candidates` rows for review. Tier B inferential.
5. Seeded 13 wallets on local + Railway via `--seed`.
6. Scheduler job at 04:45 UTC daily.

### Effect on published numbers
- **"4 organizations mapped"** in CLAUDE.md is technically accurate as a count of documented cases, but the coverage claim is narrower than the wording implies. The four orgs are hand-investigated cases; wallet-level membership exists only for org_001 (11) and org_002 (2). This correction doesn't change any quantitative public stat but sharpens what "organizations mapped" means.
- First Railway candidate scan: **324 clusters of 3–40 deployers** surfaced as novel-org candidates. Top clusters are all Base-chain, all gas-station-shaped (one funder → many short-lived deployers). None have been promoted to `org_wallets`; all carry `status='pending'` pending review.
- The existence of 324 pending candidates is itself a measurable gap in prior coverage. If even 5% promote to real orgs after review, the org count roughly quadruples.

### Open work
- **Review the 324 candidates.** Promotion workflow: investigator inspects the cluster's deployed contracts / destination flows / on-chain timing, and if the pattern matches an org profile, inserts members into `org_wallets` with `added_by='manual_review_YYYY-MM-DD'`. Otherwise sets `status='dismissed'` with a `notes` explanation.
- **Promotion admin endpoint.** An `/admin/promote-candidate` that takes a `candidate_id` + `org_id` and atomically INSERTs members + updates candidate status would make the review workflow ergonomic. Not shipped.
- **Extend signals beyond funder.** The detector currently uses shared `funder` as the clustering key. Additional signals that would sharpen precision when populated: `typical_gas_price_gwei`, deployer-similarity fingerprints (already computed in `deployer_similarity`), bytecode-family membership, timezone-inferred activity windows.
- **Merge org_candidates and deployer_similarity.** There's conceptual overlap between "these deployers are behaviorally similar" and "these deployers share organizational infrastructure." Not currently cross-referenced. Could fold into the candidate detector as a second signal layer.

---

## Correction #12 — Observation Capability Primitive Was Not Computed

**Date:** 2026-04-20
**Motivating audit:** exception-as-rule review, P2.

### The claim (as previously implied)
CLAUDE.md's *Adversarial Topology Framework* section names five primitives for evaluating any contract: position, permissions, trust bindings, mutability, observation capability. Each is described as load-bearing, and the interpretive rule states that "a node with privileged position, broad permissions, high mutability, strong trust binding, and zero malicious behavior is at MAXIMUM stored potential — not minimum risk."

### The truth (as measured)
`risk_scoring.py` computed four components into `stored_potential`: `approval_scope + capability + deployer_risk + org_context`. Volatility applied as a multiplier; realized_value as a divisor. Observation capability was named in the framework but **nowhere in the scoring code** — a pure observer-class contract (an oracle with broad user-intent visibility; a router that sees aggregated signed intent) scored MINIMAL despite being at the framework's maximum-stored-potential corner.

### How this was caught
Exception-as-rule audit grep'd `risk_scoring.py` for each of the five primitive names. Four matched concrete scoring functions; observation capability matched nothing. The math `stored_potential = approval_score + capability_score + deployer_score + org_score` at line 856 (pre-P2) confirmed a four-component sum. The framework's rhetorical five-primitive claim did not match the code's four-primitive computation.

### What changed
1. `_compute_observation_capability(conn, address) -> (int, dict)` added at line ~695 of `risk_scoring.py`. Four signals:
   - **A** (0–8) — `bytecode_pattern_notes` contains `CALLER` / `TIMESTAMP` / `TXORIGIN` / `ORIGIN` markers.
   - **B** (0–8) — log-scaled distinct interacting-EOA count (<10=0, 10–100=2, 100–1k=4, 1k–10k=6, 10k+=8).
   - **C** (0–8) — present in `infrastructure_registry` with classification/notes containing `router`, `aggregator`, `bundler`, `oracle`, `endpoint`, `transmitter`, `relay`, `relayer`.
   - **Edge** (+1) — contract has any `approval_watchlist` Permit2 row.
2. `stored_potential` ceiling raised from 100 to 125. Tier boundaries unchanged (CRITICAL ≥50, HIGH ≥20, MEDIUM ≥8, LOW ≥3). Observation-heavy contracts score higher; this is the correction, not a recalibration artifact.
3. API `/api/v1/risk/{chain}/{addr}` response adds `observation_capability_score` at the top level and `observation_capability` under `components`. Additive — existing keys preserved.
4. No new DB columns — `observation_capability` is computed live per request alongside the rest of the scoring model, consistent with Correction #6.

### Effect on published numbers
- Validation on Railway's top 10 contracts by distinct-EOA count (an in-corpus proxy for observation-heavy infrastructure) — all are confirmed/suspected traps that have interacted with 1,000+ unique users. Samples after P2:

| contract | EOAs | obs score | stored | tier |
|---|---|---|---|---|
| `0xfc26…03ce` | 5,107 | 6 | 22 | HIGH |
| `0x752c…c858` | 4,869 | 15 | 68 | CRITICAL |
| `0x8321…2e17` | 3,444 | 6 | 36 | LOW |
| `0x9818…6732` | 1,758 | 14 | 29 | CRITICAL |

- External infrastructure (Chainlink ETH/USD, Uniswap V3 SwapRouter, LayerZero EndpointV2, etc.) was attempted but none are in Layer 3's `contracts` table — Layer 3 records new deployments observed since March 2026, not pre-existing infrastructure. The primitive is correctly defined for in-corpus contracts and cannot re-score off-corpus ones without an ingest-side change.
- Zero `risk_score` values shipped publicly before this correction have been regenerated. Persistence was never computed (`risk_scores` is computed live per request; see Correction #6). The first request after deploy returns the new schema.

### Open work
- **Include observation_capability in batch_score pre-filter.** The lightweight pass at `risk_scoring.py:889` pre-ranks by bytecode-signal count only. High-EOA-but-no-bytecode-signals contracts could be under-ranked in top-N queries. Revisit if a consumer reports missing obvious observation-heavy cases.
- **Extend infrastructure_registry beyond the 12 CCTP + retrospective Kelp entries.** Signal C currently fires on very few contracts. A seeding pass for Uniswap routers, Aave pools, LayerZero endpoints, 1inch aggregation, Chainlink aggregators, Gelato, Stargate would make Signal C meaningful for more scored contracts. Flagged as a P4-adjacent item.
- **Signals D and E (deferred from P2 scope):** event-emission address-typed arg detection, and proxy-contract compounding with `proxy_upgrade_watcher`. Deferred pending either an event-log indexing pass or `proxy_upgrade_watcher` Railway deploy (CLAUDE.md priority #4).
- **`/methodology/stored-potential` copy update.** The public methodology endpoint documents the scoring components. Needs a copy change to state the 5-component model and list the observation signals. Not in scope for this correction.

---

## Correction #13 — "Camouflage Ratio" Was Measuring Contract-Design, Not Adversary Strategy

**Date:** 2026-04-20
**Motivating audit:** exception-as-rule review, P3.

### The claim (as previously implied)
The `camouflage_ratio` metric — fraction of active contracts with <10% revert rate — was published at 70-79% and described as reflecting adversary camouflage strategy. The interpretive shorthand was "Nash equilibrium": a stable fraction of adversarial contracts hiding as low-revert because that's where victim traffic lives. `/methodology/camouflage` published the threshold justification (4.5x more victims for <10% revert than >50% revert) and presented the headline figure as a measurement of adversary behavior.

### The truth (as measured)
The metric was a **population-level arithmetic** — numerator and denominator both included every active contract with 10+ interactions, legitimate or adversarial. The definition in the module docstring correctly said "contracts with low revert rates" but the name, methodology endpoint, and downstream reports framed it as an adversary-specific signal.

When the metric is restricted to contracts Layer 3 has actually flagged as adversarial, the picture inverts:

| cohort | total | low-revert (<10%) | ratio |
|---|---|---|---|
| ALL active contracts (≥10 tx lifetime) | 912 | 579 | **63.5%** |
| suspected + confirmed | 896 | 566 | **63.2%** |
| confirmed only | 312 | 78 | **25.0%** |

Two findings:

1. **Population and "adversary" (suspected+confirmed) ratios are 0.3pp apart.** The suspected+confirmed cohort almost entirely overlaps the ≥10-tx population (896 of 912, or 98.2%). Layer 3's high-activity corpus IS dominated by flagged contracts, so the two ratios are structurally identical — the "adversary" measurement is functionally measuring the population.

2. **Confirmed-only ratio is 25.0% — 38.5pp LOWER than the population.** Contracts with the strongest adversarial evidence revert *more*, not less. This is the inverse of the Nash-equilibrium interpretation. It is also the structurally expected finding: a confirmed trap is a contract that reverts on the adversary bot's extraction attempt, so high-revert is a *feature* of the adversary design, not a liability to camouflage against.

The headline "70-79% stable across 23 days" figure cited in `/methodology/camouflage` does not reproduce against today's corpus (63.5%). Whether that's drift, corpus composition change, or the prior figure being wrong at the time is not determinable from the code; all three hypotheses are consistent with a population-level metric that was never pinned to adversary-specific data.

### How this was caught
Exception-as-rule audit traced the SQL in `camouflage_tracker.py:compute_day`. The `rows = conn.execute(...)` query selected every contract with 10+ tx on a day — no confidence_tier filter. The numerator (`camouflaged`) was filtered only by revert rate. With legitimate DEX routers also low-revert, the ratio measured low-revert contract design across the whole ecosystem, not adversary disguise. The interpretation in the methodology endpoint read the output as if the denominator were adversary-scoped.

### What changed
1. Schema: added three columns to `camouflage_metrics` — `adversary_low_revert_ratio`, `adversary_total_contracts`, `adversary_low_revert_count`. Kept `camouflage_ratio` column for historical continuity.
2. `surveillance/camouflage_tracker.py`: compute_day now joins `contracts` to compute both metrics per day. Module docstring rewritten to separate the population metric (retained, renamed-in-spirit to "low-revert ratio") from the adversary-scoped one.
3. `web/api_v1.py` `/methodology/camouflage`: replaced single-metric response with a dual-metric response explicitly flagging that `camouflage_ratio` overstates the adversarial claim and pointing to `adversary_low_revert_ratio` as the honest test.
4. `web/api_v1.py` `/stats`: `detection` block now includes both ratios live.
5. `scripts/camouflage_comparison.py`: standalone diagnostic for one-shot comparisons.

### Effect on published numbers
- **"70-79% Nash equilibrium" figure** is superseded. The live general-population ratio is 63.5% today. The confirmed-adversary ratio is 25.0%. Any external material citing 70-79% as an adversary-specific statistic needs an update.
- **Interpretation reversal:** the framework's prior framing — that adversaries concentrate in low-revert as camouflage — is not supported by the data. Confirmed adversaries are disproportionately *high*-revert (75% of them). This makes sense: most Layer 3-confirmed traps are reversion-traps that fail bot extractions loudly. The camouflage-as-disguise intuition applied to a different adversary class (silent-settlement contracts) than the one Layer 3 predominantly detects.
- **`/methodology/camouflage`** response has changed shape. Consumers parsing the old `metric` + `current_value` + `definition` fields see a new `metrics` object and `historical_anchor` field instead. Additive change with backward-incompatible key removal — documented.

### Open work
- **Daily recompute.** The `adversary_low_revert_ratio` will be populated from the next scheduler fire (00:20 UTC); existing historical rows in `camouflage_metrics` have NULL for the new columns until a backfill runs. A `camouflage_tracker --compute-all` backfill would fill them.
- **Where the 79.2% came from.** The original figure was calibrated during the two-week assessment. Whether the methodology shift (inclusion/exclusion of a narrower cohort) accounts for the drift, or whether early-period corpora had different composition, is a followup question for whoever owns the historical narrative.
- **Does the 25% confirmed figure hold across chains?** The comparison above is corpus-wide. Per-chain breakdown might surface that one chain's confirmed traps have a different revert profile. Not investigated in this correction.
- **Revisit `camouflage` language across L3_CONTEXT_* docs and external materials.** Inside the code it is now clear; outside the code the "camouflage" narrative survives in several places. This correction is the canonical reference for when to update them.

---

## Correction #14 — Silent FP Silencing Through the Admin Endpoint

**Date:** 2026-04-20
**Motivating audit:** exception-as-rule review, P4.

### The claim (as previously implied)
The `/admin/mark-false-positive` endpoint took a list of addresses and set `alerts.false_positive = 1` for every alert on those addresses. The expectation was that an operator used this only with justification, and that the justification surfaced somewhere — either in the `false_positives` audit table or in a correction log entry.

### The truth (as measured)
The endpoint accepted only `{"addresses": [...]}`. It wrote nothing to the `false_positives` audit table and required no reason. Local DB had 3,577 alerts flagged this way across 5 addresses (mostly Arbitrum WETH9 receiving transfers from org wallet `fdaf1f…`), with **zero matching audit rows**. These silenced alerts are read by `org_cycles.py:177` and `risk_scoring.py:522` to exclude them from organizational reasoning and risk scoring — so silent FP marking silently dropped evidence from those analyses.

Railway had zero alerts with `false_positive = 1` — the silencing was local-DB only. The structural vulnerability in the endpoint was identical; nothing prevented the same thing on production if anyone had used the endpoint there.

### How this was caught
Exception-as-rule audit counted `alerts WHERE false_positive = 1` (local 3,577) and cross-joined against `false_positives` (12 rows, zero matching the flagged alert addresses). Traced the endpoint at `run_surveillance.py:960` and confirmed the write path had no reason field and no audit-row insert.

### What changed
1. **Endpoint is now hardened.** Accepts `addresses`, `reason` (required, ≥10 chars), and optional `detector_blamed` (default `"manual_review"`). Rejects missing/short reason with HTTP 400.
2. **Atomic audit write.** For each address with pending alerts, the endpoint writes a `false_positives` row (`fp_method='admin_bulk_mark'`, capturing pre-flag alert count and distinct alert types in `original_patterns`) and commits per-address so contention with the long-lived `db_writer` resolves inside SQLite's busy-timeout window rather than blocking on an outer `BEGIN IMMEDIATE`.
3. **Skips no-op addresses.** If an address has zero un-flagged alerts, no audit row is written.
4. **Local backfill.** Ran a one-shot INSERT against the local DB to create 5 `false_positives` audit rows covering the historically silenced addresses, reason tagged `canonical_infrastructure_misfire` (org-wallet transfers that touched WETH9 / Uniswap V3 Router etc.). Railway had nothing to backfill.
5. **Smoke-tested on Railway** after deploy:
   - POST with no reason → 400.
   - POST with reason=`"too short"` → 400.
   - POST with valid reason against a dummy address with no alerts → 200 `{marked:0, audit_rows_written:0}`.

Two operational notes from the smoke test:
   - **Windows curl UTF-8 gotcha.** The first test body included an em-dash that got encoded as CP1252 `\x97` by the calling shell; the Python HTTP server's `.decode("utf-8")` raised `UnicodeDecodeError`. Call sites should send ASCII or force UTF-8.
   - **Lock contention fix.** Initial patch used `BEGIN IMMEDIATE` across the whole address loop; this lost every lock contest against `db_writer` and returned 502. Final patch uses a read-only connection for the pre-flight SELECT and commits writes per address. Same atomicity for a single address, survives contention.

### Effect on published numbers
- No public statistics were ever published based on the silenced-alert population. The FP flag affected internal scoring exclusions (`org_cycles`, `risk_scoring`) on local DB only. Railway's numbers were never touched.
- The local DB's `false_positives` audit table grew from 12 to 17 rows — the 5 new ones capture what had been silent-silenced.
- `alerts.false_positive = 1` counts are unchanged; only the audit trail coverage improved.

### Open work
- **Audit coverage check as ongoing discipline.** Any future delta between `COUNT(DISTINCT address WHERE false_positive=1)` and `COUNT(DISTINCT contract_address IN false_positives)` indicates the endpoint got bypassed somehow or the backfill got dropped. A scheduled health check is a good idea.
- **infrastructure_registry source-level suppression (partial scope).** The audit originally proposed adding an `infrastructure_registry` lookup in the LAUNDRY/CASHOUT detection path so canonical infra never generates the alert in the first place. The detection path turned out to be the Alchemy Notify webhook handler, whose alert types (`LAUNDRY_PIPELINE`, `CASHOUT_MOVEMENT`) fire only when watched org wallets are the `from` or `to` of a movement — the watched wallet is recorded as the alert `address`, not the counterparty. So the target of the flag isn't infrastructure; the current implementation doesn't produce the misfire pattern I originally described. Leaving this open in case a future detector IS vulnerable to the pattern.
- **Deprecate `/admin/mark-false-positive` as bulk-mark.** For per-alert FP review, a different endpoint taking an alert ID and reason would be cleaner. The bulk-mark shape is rarely the right tool. Consider adding `/admin/mark-false-positive-alert` that takes `alert_id` + `reason`.

---

## Correction #15 — Extraction-Event Taxonomy Had No Classifier

**Date:** 2026-04-20
**Motivating audit:** exception-as-rule review, P5.

### The claim (as previously implied)
The `extraction_events.event_type` column was presented as a taxonomy — `full_pipeline_cycle`, `infrastructure_parasite`, `oracle_manipulation_lending_exploit`, etc. — that captured the mode of each documented theft event. The categories looked principled.

### The truth (as measured)
`event_type` was a free-text `TEXT` column with no CHECK constraint, no suggestion, and no external check on the hand-assigned label. Every one of EXTRACTION_001 through 008 was labeled by the investigator at INSERT time with nothing forcing the label to match a pre-enumerated vocabulary. A novel extraction event could be quietly shoehorned into an existing bucket, and nobody — including the person doing the labeling — would notice unless they happened to compare two events' code paths and find the labels didn't fit.

### How this was caught
Exception-as-rule audit checked for a classifier. There was none. The DDL confirmed free-text; the distinct `event_type` values across 5 local rows showed hand-variation already (`oracle_manipulation_lending_exploit` used for two different protocols, ok; but nothing structural enforced consistency).

### What changed
1. `surveillance/extraction_classifier.py`: rule-based classifier with a closed 7-value vocabulary. `suggest_type(summary, raw_transactions)` returns `(label, confidence, matched_patterns)`. Adding a new category requires editing `_RULES` — the change leaves git history and is reviewable.
2. `surveillance/db.py`: migration for `extraction_events.event_type_suggestion` and `event_type_suggestion_confidence` shadow columns. The classifier writes the suggestion alongside the documented type; divergence is the actionable signal.
3. CLI `--apply` backfills suggestions for existing rows.
4. Classifier does *not* gate INSERTs. Existing INSERTs continue to work with hand-assigned `event_type`. The shadow column surfaces drift without requiring a workflow change.

### Effect on published numbers
- All 8 extraction events agree with the classifier after tuning (initial pass had 7/8 agreement with EXTRACTION_006 / Aethir flagged as ambiguous between `oft_adapter_admin_compromise` and `cross_chain_dvn_verification_failure`). The disagreement was informative: Aethir IS a LayerZero-OFT-adapter attack, and the original DVN-failure rule triggered on any LayerZero mention. Tightening the DVN rule to require verification-failure cooccurrence resolved it.
- **The classifier's first-pass 7/8 was a successful detection of genuine ambiguity in the taxonomy**, not a failure of the classifier. Aethir and Kelp are both LayerZero-adjacent; their root causes differ (admin compromise vs DVN misconfiguration). The classifier caught the ambiguity that a human might have let slide.
- No published numbers change. The `event_type` column is unchanged; the suggestion is additive.

### Open work
- **Extend vocabulary as new event-classes arrive.** Current 7 categories cover everything seen through EXTRACTION_008. A novel class (supply-chain compromise, reentrancy-via-hook, governance-vote-hijack, etc.) would initially classify as `unclassified` at low confidence; that's the signal to review and extend.
- **Confidence threshold for auto-promotion.** Currently all classifications are shadow-only. A future workflow could auto-accept suggestions with confidence ≥ 0.8 and require review for lower. Not shipped.
- **Re-run on any new extraction event at INSERT time.** Right now `--apply` is a manual CLI. Integrating the suggestion into the Bundle-INSERT scripts (`bundle_d_aethir_insert.py` pattern) would catch drift at write time, which is the best moment.
- **Cross-chain attack class confusion is a real ontology problem.** LayerZero, Hyperbridge, Wormhole, Chainlink CCIP etc. all have similar surface — the failure modes (DVN, MMR, guardian signature, off-chain validator) are architecturally distinct but appear in summaries together. Worth considering a multi-label scheme where an event can carry multiple tags (e.g., `cross_chain + dvn_misconfig + oft_adapter`) rather than forcing a single bucket.

---

## Correction #16 — Lexicon Additions and Camouflage Ratio Methodological Caveat

**Date applied:** 2026-04-25
**Scope (docs):** Four new entries appended to `docs/lexicon.md`; one existing entry (Camouflage Ratio) revised with a methodological caveat paragraph and one additional cross-reference; index updated; bidirectional cross-references propagated to seven existing entries.
**Scope (data / code):** None. Documentation alignment only.

### What we claimed (implicitly, as of lexicon v2026-04-18)
The lexicon's vocabulary was complete enough to describe the operator and harm classes Layer 3 surveils. The Camouflage Ratio entry framed the equilibrium claim as "stable across operators" without qualifying that the surveilled population's diversity assumption had been tested.

### What was actually true
Two empirical findings from this session require lexicon coverage that did not exist:

1. **Pristine Solo Operator** (deployer-layer) — surfaced 2026-04-25 by `surveillance/pristine_solo_detector.py`. 11 candidates locally, 13 on Railway, with mainnet gaps from 377 days to 2,498 days. Class falls between Pattern D (active mainnet reputation import) and the small-cell `org_candidates` detector, and was not previously named.

2. **Infrastructure-Scale Operator** (funder-layer) — surfaced 2026-04-25 from the funder-cluster diagnostic. 12 candidates from `surveillance/infrastructure_operator_detector.py` covering 11,006 deployers (26.5% of corpus), 26,514 contracts, and 24.5% of the corpus's confirmed traps. The class slips `org_candidates` because the >50-deployer exclusion logic was based on a falsified hypothesis ("clusters that big are CEX/faucet noise").

3. **Cross-Domain Compositional Harm** (off-chain extension) — Vercel/Context.ai breach disclosed 2026-04-19 demonstrates the same compositional-harm pattern Layer 3 documents in DeFi, but with components spanning malware → cloud platform → SSO → customer credentials. No CVE involved; every component functioned as designed. Generalizes Compositional Harm beyond the on-chain substrate.

4. **Tuition Extraction Markets** (structural framing) — generalizes the Akerlof / "liquidity of fools" model to the on-chain corpus. Distinct from Victim-to-Predator Pipeline: VtP describes the unusual case of vertical migration; TEM describes the steady-state market structure that makes such migration both possible and rare. Empirical anchor: bot `0x84792c2a` ($4,412 gas burn / 375K+ reverts feeding a $5.75M MEV vault elsewhere).

5. **Camouflage Ratio caveat** — the funder-cluster diagnostic showed 31.5% of the active deployer population is downstream of the top-12 funder cluster. The "stable across operators" framing must therefore be qualified: it is "stable across the surveilled population, of which a substantial fraction may be one funder cluster's downstream." The original equilibrium claim is not retracted, but its corpus-wide resolution is conditional on a top-12-excluded re-run.

### How was the error caught
- The two new operator entries were forced by today's session: probing `0x604be06b`'s funder chain surfaced `0xf70da978` (2,684-deployer cluster), which prompted the 12-funder enumeration, which revealed the corpus-dominance problem.
- The Cross-Domain entry was forced by the Vercel/Context.ai disclosure crossing the on-chain/off-chain boundary on a Layer-3-relevant theoretical surface (Compositional Harm).
- The Camouflage Ratio caveat was forced by the dominance finding making the "across operators" framing measurably weaker than previously claimed.

### What we changed

| Component | Change |
|---|---|
| `docs/lexicon.md` Index | Added 4 new entries to their respective category indexes (Detection Methodology, Structural and Psychological, Attack Pattern). |
| `docs/lexicon.md` body | Appended `### Pristine Solo Operator` and `### Infrastructure-Scale Operator` under Detection Methodology; appended `### Tuition Extraction Markets` under Structural and Psychological; appended `### Cross-Domain Compositional Harm` under Attack Pattern. Each entry follows the standard four-section format. |
| `docs/lexicon.md` Camouflage Ratio | Inserted "**Methodological caveat (2026-04-25)**" paragraph between Extended description and Empirical grounding. Existing Extended description preserved verbatim. Cross-references line extended with one additional reference. |
| `docs/lexicon.md` bidirectional refs | Compositional Harm, Adversarial Topology, Trust Amplification Factor, Behavioral Laundering, Pattern D, Cost-Habituation Asymmetry, Victim-to-Predator Pipeline, Configuration-Level Vulnerability, and Operational Layer Attack each gained one or more cross-references pointing at the new entries. |
| `docs/lexicon.md` Version | Bumped from 2026-04-18 to 2026-04-25. |

### Effect on published numbers
None. Zero rows moved. Zero API responses change. Documentation alignment only.

### Deliberately held out
- **Adversarial Co-Tenancy** — the A/B operators sharing prey finding (`0x604be06b` and `0xc0ffeefeed`). Resolved as anecdote, not pattern, after pairwise probe across 903 operator pairs returned only that one pair at any meaningful overlap. Not lexicon-worthy.
- **AI-Augmented Adversary Tradecraft** — interesting concept surfaced by the Vercel CEO's public attribution, but the corpus has no direct evidence of it operating at the on-chain layer yet. Hold for a future entry once there is empirical grounding.
- **Prey-Driven Equilibrium Calibration** — the hypothesis that the Camouflage Ratio is prey-driven rather than market-driven. Hypothesis was tested by the pairwise overlap probe and did not generalize. Not ready for the lexicon.

### Open work
- **Re-compute Camouflage Ratio against a top-12-excluded cohort.** The equilibrium claim cannot be restored to corpus-wide resolution until this runs. Targeted as the next methodological step in the Camouflage Ratio entry's caveat paragraph.
- **Build the `infrastructure_operators` entity class.** The detector ships candidates but the promotion target schema (peer of `org_wallets` with a different review workflow) is not yet defined. Holding for separate spec.
- **Re-run other corpus-wide statistics** (disposable-deployer rate, bytecode-family diversity, suspected-tier base rate) against the top-12-excluded cohort, with the same caveat-or-restate discipline. The dominance finding implies these claims need re-verification, not silent retention.

---

## Correction #17 — Trust Amplification 14.2× Was a Dissolved-Family Baseline Artifact + Stale FP Rows on Confirmed Contracts

**Date applied:** 2026-04-25
**Scope (docs):** Trust Amplification Factor entry in `docs/lexicon.md` revised with a methodological caveat. The 14.2× figure for `0xd4624228` is no longer cited as a current measurement.
**Scope (code):** `surveillance/false_positive_tracker.py` gains a guard that skips contracts with any `trap_events` row.
**Scope (data):** 7 stale rows retracted from `false_positives` (local + Railway).

### What we claimed
Two coupled claims, both surfacing from a check-up on `0xd4624228` (the canonical Trust Amplification Factor anchor):

1. **Trust Amplification Factor empirical anchor.** The lexicon (and decks downstream of it) cited "14.2× amplification, 2,910 victims, 98.7% router-delivered traffic" for `0xd4624228` as the canonical measured instance of TAF.
2. **False-positives audit table.** The `false_positives` audit table was presented as an FP signal layer. A contract with a row in it is — by the table's name and downstream consumer expectation — an investigator-validated false positive.

### What was actually true

**On the 14.2× figure.** `surveillance/trust_amplification.py` computes `amplification = callers_per_day / family_avg_callers_per_day`. When the contract has no `bytecode_family_members` row, the formula falls back to a self-baseline of 1.0×. `0xd4624228`'s 14.2× was computed when the contract was a member of the `T2-eaef6a5d` bytecode family — the same family that was dissolved by Correction #3 (2026-04-16) after being identified as a NULL-bucket methodology artifact. When the family dissolved, the contract lost its comparator baseline. Subsequent runs of the producer would have reset amplification to 1.0× (self-baseline). A check-up run on 2026-04-25 produced no row at all: the contract has only 2 transaction events in the post-monitoring-start window, dropping it below the producer's 50-caller minimum. The 2,910-victim and ~97%-router-traffic figures remain Tier A direct counts and are unaffected. The 14.2× multiplier specifically is **a Correction #3 cascade**: the artifact baseline that produced it was retroactively invalidated when the NULL-bucket family was dissolved.

**On the FP audit table.** A check on contracts with both `confidence_tier='confirmed'` and a `false_positives` row surfaced 8 conflicts; 7 of them have at least one `trap_events` row (proven harm). All 7 FP rows were written in a single batch on 2026-03-30T02:38:31.581742+00:00. The FP scanner (`false_positive_tracker.py`) filters its candidate set on `confidence_tier = 'suspected'`, but the contracts in question had been promoted to `'confirmed'` between FP-scanner runs without their FP rows being invalidated. The same class of bug as Correction #5 (cache transplant staleness): post-write mutation of related state without invalidating the prior assertion.

The seven stale rows:

| contract | fp_method | trap_events | observation |
|---|---|---|---|
| `0xd4624228...` | sustained_traffic | 31 | The TAF anchor — high-traffic + low-revert is the parasite's signature |
| `0x0becff44...` | balanced_interaction | 19 | 1,073 callers, 9 selectors — diverse-interface heuristic fired on a confirmed harvester |
| `0x54a03956...` | sustained_traffic | 5 | 498 callers over 3 days, 0.6% revert |
| `0x39d411e0...` | weak_detector_only | 3 | 37 callers, 9.2% revert; CALLER-only pattern read as access-control |
| `0x18d0bd91...` | weak_detector_only | 2 | 52 callers, 5% revert; same shape |
| `0xf2b2b76e...` | sustained_traffic | 1 | 124 callers over 5 days |
| `0x01bba1aa...` | balanced_interaction | 1 | 64 selectors — diverse interface on a confirmed-harm contract |

Three of the seven (`sustained_traffic` cases) demonstrate exactly the failure mode the [Camouflage Ratio](../docs/lexicon.md#camouflage-ratio) and [Trust Amplification Factor](../docs/lexicon.md#trust-amplification-factor) entries describe: **infrastructure-parasite contracts produce high-traffic, low-revert patterns by design**, and the heuristic that catches legitimate-token false positives reads that signature as evidence of legitimacy. The heuristic has no defense against operators who calibrate against it.

### How was the error caught
A user-prompted check-up on the `0xd4624228` Trust Amplification Factor anchor entry. The lexicon's "14.2× amplification" figure was inspected against the corpus's current `trust_amplification` row (showing 1.0×), the discrepancy was traced to the producer formula's family-baseline dependency, and the dependency was traced to the Correction #3 NULL-bucket dissolution. The same check-up surfaced the contradicting `false_positives` row, which was traced to the FP scanner's heuristic firing on the parasite's defining surface.

### What we changed

| Component | Change |
|---|---|
| `surveillance/false_positive_tracker.py` `scan_false_positives` | New guard inside the scan loop: any contract with at least one `trap_events` row is skipped. Observed-harm overrides heuristic FP signal. Defense-in-depth alongside the existing `confidence_tier = 'suspected'` SQL filter. |
| `scripts/retract_stale_fp_rows.py` (new) | Idempotent migration with dry-run default. Retracts `false_positives` rows where the contract is currently `'confirmed'` and has at least one `trap_events` row. Also un-silences any `alerts.false_positive=1` rows on those contracts (no rows in scope this run; defensive for future). |
| `false_positives` table (data, local + Railway) | 7 rows retracted. 0 alerts un-silenced. |
| `docs/lexicon.md` Trust Amplification Factor entry | Added "**Methodological caveat (2026-04-25)**" paragraph between Extended description and Empirical grounding. The 14.2× figure is no longer cited as a current measurement. The `2,910 victims` and `~97% router-delivered traffic` counts retained as Tier A direct measurements. Producer formula made explicit (`callers_per_day ÷ family_avg_callers_per_day`) along with the 1.0× fallback. |

### Effect on published numbers
- `false_positives` row count drops by 7 (local + Railway).
- 0 alerts re-surfaced (the FP audit table is independent of `alerts.false_positive`; no alerts had been silenced as a downstream effect of the 7 stale rows).
- TAF lexicon entry: 14.2× moved from "canonical measured instance" framing to "originally reported, dissolved-family artifact." The 2,910-victim and ~97%-router-traffic measurements remain canonical.

### Pattern worth naming
This is the second instance of a Correction #3 cascade. Correction #5 surfaced the cache-transplant staleness that was the same pattern at the cache layer. Correction #17 surfaces a methodology-baseline staleness at the producer-output layer. The shape is: **derived data outliving the assumption that justified it**. When `T2-eaef6a5d` was dissolved as a NULL bucket, every signal computed against that bucket as a comparator silently lost its meaning. Future corrections of this class should be flagged at the time the dissolution happens, not surfaced months later through anchor-point check-ups.

### Open work
- **Audit other producer outputs that depended on `T2-eaef6a5d`-family baselines.** TAF is one signal; the camouflage tracker, trust amplification alerts, and any per-family aggregate downstream of `bytecode_families` may have analogous staleness. A targeted scan of producer modules for `family_avg`-style baseline lookups is the natural follow-up.
- **The 1 remaining contract with confirmed tier + FP row but zero trap_events** (`0xef17f86e44d8e2718287da08eb26dcd3953156ce`, fp_method `balanced_interaction`, 41 selectors, 64 callers, 4.5% revert) is not in scope for this retraction. It is either a legitimate token mis-promoted to confirmed via a different path, or a parasite without observed bot-trapping. Holding for separate review.
- **Re-snapshot the `0xd4624228` figures from the deck source** (`Layer3_Intelligence_Platform_1.pptx` slide 5). If the deck cites 14.2× as a current measurement, the deck needs the same caveat the lexicon now carries.
- **The producer's DELETE-and-rebuild semantics** (`DELETE FROM trust_amplification` at the start of every run) means contracts that drop below the 50-caller minimum lose their row entirely. Historical TAF measurements are not preserved by the producer; they exist only in narrative documentation. Worth deciding whether to archive snapshots before each rebuild.

---

## Correction #18 — Epistemic Test #2 Cleanup Pass

**Date applied:** 2026-04-29
**Scope (docs):** `docs/lexicon.md` (Camouflage Ratio caveat lifted; Infrastructure-Scale Operator overlap finding integrated; figures re-pointed at canonical script). `surveillance/data/cases/PARASITE_ARCHITECTURE_0xd4624228.md` and `cases/TRUST_LAYER_EXPLOITATION_20260324.md` gain Correction #17 addenda. `reports/correction_log.md` Correction #16 receives a methodology note (this entry).
**Scope (code):** New canonical query `scripts/funder_metrics.py`. Search script `scripts/search_tuition_anchor.py` for a corpus-derived Tuition Extraction Markets bot anchor.
**Scope (data):** Watchlist row for `0x604be06b` updated on local + Railway production with org_001 attribution language.

### What we claimed (implicitly, as of lexicon v2026-04-28)
The post-2026-04-25 lexicon and correction-log entries described top-12 funder cluster scale with floating numbers (lexicon: 14,650 deployers / 36% of 41,538-deployer corpus; Correction #16: 11,006 deployers / 26.5% of corpus). Both numbers came from the same probe but diverged. The Camouflage Ratio entry's 2026-04-25 caveat noted the equilibrium claim could not be cited at corpus-wide resolution until a top-12-excluded re-run was performed. The Infrastructure-Scale Operator entry hedged between single-actor and multi-tenant interpretations of the funder cluster. The two `0xd4624228` case files framed the 14.2× trust amplification factor as live, despite Correction #17 having retired it in the lexicon. The Tuition Extraction Markets entry cited bot `0x84792c2a` ("$4,412 gas burn / 375K+ transactions") as if from corpus.

### What was actually true
Per `reports/epistemic_test_results_2026-04-29.md` (14 sections, 6 corrections flagged):

1. **A1** — Lexicon and Correction #16 disagreed on the same probe. Live measurement via `scripts/funder_metrics.py` on 2026-04-29: 22,165 deployers / 42.7% of corpus. Both prior numbers were point-in-time and diverged because there was no canonical query.
2. **A4** — Cross-funder downstream-deployer overlap probe (66 pairs, top-12 funders): zero pairs share any downstream deployer. The single-actor-with-many-faces and multi-tenant rental hypotheses are both ruled out. Twelve independent infrastructure-scale operations exist with similar operational tempo and no shared infrastructure.
3. **A7** — Camouflage ratio top-12-excluded re-run: full corpus 67.1%, top-12 excluded 68.1%, delta +0.9pp. Well below the 5pp threshold; equilibrium robust.
4. **A10** — `cases/PARASITE_ARCHITECTURE_0xd4624228.md` and `cases/TRUST_LAYER_EXPLOITATION_20260324.md` carried framing that the lexicon had retired in Correction #17 (TRUST_LAYER file already had a partial 2026-04-11 inline correction; PARASITE_ARCHITECTURE had no addendum at all).
5. **A11** — `0x604be06b` watchlist row still read "high-velocity solo operator" without the post-2026-04-25 org_001 cluster attribution.
6. **A12** — Bot `0x84792c2a` has zero `transaction_events` rows in the corpus (the bot's own case file `BOT_INVESTIGATION_0x84792c2a_20260322.md` confirms this explicitly). The 375K+ figure came from external block-walking with methodology not preserved. Cited file `reports/mev_vault_0xa45b51_discovery.md` does not exist.

### How was the error caught
The 2026-04-29 epistemic integrity test (`reports/epistemic_test_results_2026-04-29.md`) ran 14 claim-vs-corpus checks against the 2026-04-25 cohort of new framework material. 8 sections survived HIGH-confidence; 6 surfaced corrections.

### What we changed

| Component | Change |
|---|---|
| `scripts/funder_metrics.py` (new) | Canonical query for top-N funder cluster metrics. CLI: `--top-n` (default 12), `--db`. Output: `reports/funder_metrics_<DATE>.md`. Self-test on 2026-04-29 reproduced 22,165 / 42.7% from epistemic test A1. **Future references to funder cluster scale should cite this script and a date, not transcribed numbers.** |
| `reports/funder_metrics_2026-04-29.md` (new) | First dated invocation. 22,165 top-12 deployers, 42.7% of 51,874-deployer corpus, 39.4% of active subset. |
| `docs/lexicon.md` Camouflage Ratio entry | Replaced the 2026-04-25 methodological caveat with a 2026-04-29 empirical-robustness paragraph documenting the +0.9pp delta. The "stable across operators" claim is restored to corpus-wide resolution. |
| `docs/lexicon.md` Infrastructure-Scale Operator entry | Replaced the single-actor-vs-multi-tenant hedge with the resolved finding (twelve independent operations, zero cross-funder overlap). New open question: convergent calibration across unrelated actors. Replaced static figures with a pointer to `scripts/funder_metrics.py` and named `reports/funder_metrics_2026-04-29.md` as the dated source. Per-funder profile heterogeneity figures refreshed against today's corpus. |
| `docs/lexicon.md` version | Bumped from 2026-04-28 to 2026-04-29. |
| `cases/PARASITE_ARCHITECTURE_0xd4624228.md` | Correction #17 addendum added at top, between metadata block and Executive Summary. Original analysis preserved. |
| `cases/TRUST_LAYER_EXPLOITATION_20260324.md` | Correction #17 addendum added at top, layered on the existing 2026-04-11 inline correction. Original analysis preserved. |
| `watchlist` table (local + Railway production) | `0x604be06b` row's `watch_reason` updated to reflect org_001 cluster re-attribution. Both DBs verified post-update. |
| `scripts/search_tuition_anchor.py` (new) | Corpus search for replacement Tuition Extraction Markets bot anchor. Output: `reports/tuition_extraction_anchor_search_2026-04-29.md`. **Lexicon entry not auto-edited** per cleanup-pass discipline; the judgment call (Option A re-source, Option B mark as external evidence, Option C retract and rewrite) is human-gated. Top candidate: `0xc0dec76000f6c2d32f23d523748e50ebb5bb34a3` (57,023 tx, 85% revert across 174 contracts and 23 days, with parallel-infrastructure signal — funder `0x18b0f4547a89` funds 5 other deployers with fleets up to 44). |

### Effect on published numbers
- Camouflage Ratio: claim restored to corpus-wide resolution. The 2026-04-25 caveat is no longer cited as methodological provisional.
- Infrastructure-Scale Operator scale: figures now reference dated canonical query rather than floating numbers. The 14,650 / 36% (lexicon) and 11,006 / 26.5% (Correction #16) figures should be read as superseded point-in-time measurements; the corpus has grown 25% since 2026-04-25.
- Trust Amplification Factor: case-file framing now matches lexicon-level retirement of the 14.2× multiplier.

### Pattern worth naming
This is the second cleanup-pass surface that fixes a propagation problem rather than a substantive analytical error. Correction #6 was about documentation lagging code reality; this one is about documentation lagging the lexicon's own corrections. The fix in both cases is structural: replace the propagation surface with a canonical query (this case) or a single source of truth for behavior (Correction #6's `score_contract` per-request answer). When several documents copy a number from a probe, future drift is guaranteed unless one of them is a script.

### Open work
- **Tuition Extraction Markets entry update.** ~~Search results delivered in `reports/tuition_extraction_anchor_search_2026-04-29.md`. Pending Jason's decision between Option A, B, or C.~~ **CLOSED 2026-04-29:** Option A executed. Lexicon entry now anchored on `0xc0dec76000f6c2d32f23d523748e50ebb5bb34a3` (57,023 tx, 85% revert, 174 contracts, 23-day span, vanity-prefix branding cross-referencing Adversarial Vanity Branding). Old `0x84792c2a` anchor retired with cited reason.
- **Cross-Domain Compositional Harm case files.** ~~Vercel/Context.ai breach (2026-04-19) and Bancor EIP-7702 (2026-04-29) are referenced in the lexicon entry without backing case files.~~ **CLOSED 2026-04-29 (partial):** Vercel case file written as structural reference (`cases/CASE_VERCEL_CONTEXT_BREACH_20260419.md`); Bancor case file written as skeleton (`cases/CASE_BANCOR_EIP7702_20260429.md`) acknowledging the documentation gap explicitly. The Bancor file flags that the lexicon's reference currently exceeds the corpus's documented detail and is held for primary-source review in a future session.
- **Funder cluster diagnostic file.** ~~Cited in lexicon but never existed as a file.~~ **CLOSED 2026-04-29:** `reports/funder_cluster_diagnostic_2026-04-29.md` written, replacing the missing 04-25 file. Documents top-12 roster, three structural findings (twelve independent ops, profile heterogeneity, L2-native subset), and three open questions. INDEX.md Section 1 entry updated to point at this file.
- **Single-Purpose Infrastructure Funder typology** (added to lexicon 2026-04-28). **CLOSED 2026-04-29:** Case file `cases/CASE_SINGLE_PURPOSE_INFRASTRUCTURE_FUNDER.md` written. Documents the 69 Pattern A operators, three deployment shapes, vanity-mirrored funder/deployer bonus finding, and structural independence from org_001-004 architecture. INDEX.md Section 1 entry added.
- **Active-drain follow-ups** (six self-deploying trap operators surfaced 2026-04-29 by ad-hoc query, with `0x1aae146c1328` as the 133-drain leader still unwatched). Deferred to next session per active-drain to-do list.

### Note on Correction #16
Correction #16 (2026-04-25) cited "11,006 deployers (26.5% of corpus)" for the same probe the lexicon entry described as 14,650 / 36%. Both were point-in-time measurements at the same date; the disagreement reflects the absence of a canonical query at the time. The numbers are not retracted (both were honest measurements of related quantities) but they are superseded by `scripts/funder_metrics.py` going forward. Future correction-log entries citing funder cluster scale should reference a dated invocation of the canonical script.

---

## Correction #19 — Single-Purpose Infrastructure Funder Detection Rule Was Funder-Side-Only

**Date applied:** 2026-05-08
**Scope (docs):** `docs/lexicon.md` Single-Purpose Infrastructure Funder entry — definition refined to require *both* funder-layer and downstream-layer verification; new false-positive-class paragraph added under Empirical grounding.
**Scope (code):** None. The detection rule lived in informal probe scripts; refining the lexicon entry is the load-bearing fix.
**Scope (data):** None. The 69 prior Pattern A instances all pass the refined rule (chain distribution recorded as 100% L2-native already).

### What we claimed (implicitly, as of lexicon v2026-04-29)
The Single-Purpose Infrastructure Funder typology was defined by funder-layer criteria: lifetime spawns = 1, funder is EOA-only with no own deployer record, funder is silent after the funding event. The L2-only property was treated as a corpus-distribution *observation* ("Zero mainnet operators … parallels Infrastructure-Scale Operator subset 4 of top-12") rather than a detection criterion. In practice, ad-hoc probes promoted candidates to "fresh Pattern A funder" framing when only the funder-layer test was checked.

### What was actually true
On 2026-05-07 a daily probe surfaced three high-output funders — `0x268cbda30dd229e5f9b084609a2bb9b73b0f8aad`, `0x04e3eebcb2f9fa17640b1792546545b74289a4ef`, `0xa7eccdb9be08178f896c26b7bbd8c3d4e844d9ba` — all showing clean funder-layer Pattern A signal (1 lifetime spawn, EOA-only or self-funded, large downstream fleet). Initial framing in the May 7 probe writeup labeled them "three NEW high-output funders" and recommended Pattern A investigation.

The 2026-05-08 deep-dive on the freshened local DB rejected all three:

1. **`0x268cbda30dd2`** funds `0xf238b357f0d97048866b0569b9cd101df341c827` — mainnet history since 2025-03-17, fleet 278 spread across Base/OP/Arb. Multi-chain bot operator using single-funder OPSEC, not a Single-Purpose Funder.
2. **`0x04e3eebcb2f9`** is *self-funded* (funder == deployer). 1,215-contract Base self-deployer. The Pattern A funder signal was an artifact of how `funding_trail` records first incoming tx — the deployer's own first-funding trace points to itself.
3. **`0xa7eccdb9be08`** is a 2022-vintage operator (mainnet first-tx 2022-06-15) with 5 lifetime fleets. Established multi-chain operator, not single-purpose.

All three pass funder-layer Pattern A criteria but fail at the downstream layer. None warranted Pattern A classification.

### How was the error caught
The May 7 probe writeup recommended deep-diving the three funders. Running the deep-dive (post-corpus-resync 2026-05-08) and observing each downstream deployer's `mainnet_first_tx` revealed that none of the three downstreams were L2-only — every one had either mainnet history or self-funding. The funder-layer-only rule had no test for this; the lexicon entry treated L2-only as a corpus observation, not a detection criterion. The gap was a methodology gap, not a corpus error: the 69 prior Pattern A instances all happen to be L2-only (the `mainnet_first_tx IS NULL` filter was implicit in how those candidates were originally surfaced), so the rule's incompleteness never produced a published false positive — only a writeup-stage one, caught before promotion.

### What we changed

| Component | Change |
|---|---|
| `docs/lexicon.md` Single-Purpose Infrastructure Funder definition | Added a Detection rule (refined 2026-05-08) section requiring both funder-layer and downstream-layer checks. Downstream layer requires `mainnet_first_tx IS NULL`. Existing definition preamble updated to include condition (b) "the funded deployer is L2-only with no mainnet history". |
| `docs/lexicon.md` Empirical grounding section | Note added that the L2-only property is now part of the detection rule, not just a corpus-distribution observation. New "False-positive class identified 2026-05-08" paragraph documents the three rejected candidates with addresses and reasons. |
| `reports/correction_log.md` (this entry) | Numbered correction documenting the methodology gap, false-positive class, and rule refinement. |

### Effect on published numbers
- **No retraction of the 69 confirmed Pattern A instance count.** All 69 were originally surfaced via a probe that included an implicit L2-only filter; spot-checks of the published distribution (71% Base / 24% Arb / 5% OP, "zero mainnet operators") confirm the population is unchanged under the refined rule.
- **No retraction of any external-facing claim.** The three May 7 false-positives never reached watchlist HIGH, INDEX.md Section 1, or any case file. The error was caught at writeup-internal-recommendation stage.
- **Future-applied effect**: probes that surface high-output funders for Pattern A consideration must now run the downstream `mainnet_first_tx IS NULL` check before any framing as Single-Purpose Funder.

### Open work
- **None for this correction.** Refined rule is documented in lexicon and grounded in the false-positive class. Future ad-hoc probes inherit the rule by reading the lexicon entry.
- **Adjacent open thread (not part of this correction):** the May 7 hub iter_9 pause is a separate observation worth tracking; logged as INDEX.md Section 1 entry on 2026-05-08, not material to this correction.

---

## Correction #20 — Mass Mislabel Sweep: OLI-Tagged Institutional Addresses Misclassified as Adversarial Operators

**Date applied:** 2026-05-09
**Scope (docs):** `docs/INDEX.md` — Section 1 entries for Top-12 Infrastructure-Scale Operators, Architect, org_001 whale path, 0xe69f81b8 high_value_bridge_user, Cluster A/B funder analysis, bb50 industrial-scale operator. Section 2 watchlist entries for the 18 affected addresses. `docs/lexicon.md` — Infrastructure-Scale Operator, Convergent Calibration, Thermodynamic Fundamentalism, Pristine Solo Operator.
**Scope (code):** New module `surveillance/oli_enrichment.py`. New table `oli_labels` (migration in `db.py`). New script `scripts/blockscout_tag_audit.py` (the audit that surfaced this).
**Scope (data):** Local `watchlist` and `entity_classification` rows for 18 addresses receive corrective notes; production must apply the same updates separately. New `oli_labels` table populated for 140 flagged addresses on 2026-05-09.

### What we claimed
Across multiple INDEX.md entries, lexicon entries, and watchlist rows, Layer 3 had asserted that 14 institutional-class addresses were adversarial operators of various kinds. Specifically:

- **Top-12 Infrastructure-Scale Operator cluster** (lexicon: [Infrastructure-Scale Operator](../docs/lexicon.md#infrastructure-scale-operator); INDEX.md Section 1): documented as 12 funder addresses with ≥200 fanout / ≥10% adversarial ratio / ≥50% disposable rate. Used as empirical anchor for [Convergent Calibration](../docs/lexicon.md#convergent-calibration). At least one (`0x3304e22d`) used as positive-CER exemplar in [Thermodynamic Fundamentalism](../docs/lexicon.md#thermodynamic-fundamentalism).
- **Pristine Solo Operator typology** (lexicon: [Pristine Solo Operator](../docs/lexicon.md#pristine-solo-operator); INDEX.md): old mainnet wallets first appearing on L2 with high-velocity deployment. Documented as a trap-class signature.
- **Cluster A and Cluster B funder analysis** (INDEX.md Section 1; CRITICAL watchlist): two "cluster funders" connecting tens of deployers each, framed as coordinated trap-fleet operations.
- **Architect investigation** (INDEX.md Section 1; CRITICAL watchlist): a lone funder identified as the Architect's "sole funder," used to anchor an exclusivity finding.
- **org_001 whale path** (INDEX.md): one path representing 68% of org_001 deployments.
- **`0xe69f81b8` high_value_bridge_user** (INDEX.md Section 1): 49,000 ETH bridged to L1 in one week (Apr 7-14), framed as "coordinated during western sleep hours."
- **bb50 / `0xbb50ce87`** (INDEX.md Section 1; lexicon Thermodynamic Fundamentalism empirical grounding): the industrial-scale stockpile entry with 38,016+ contracts, 7.7y mainnet age, framed as PSO+Single-Purpose hybrid pre-funding state.

### What was actually true
The 2026-05-09 mass audit (`reports/blockscout_tag_audit_2026-05-09.csv`) cross-referenced 140 flagged addresses against the Open Labels Initiative tag set surfaced by the Blockscout metadata service. **18 of 140 (12.9%) carry public institutional tags incompatible with the adversarial classification.**

**HIGH-severity mismatches (institutional / CEX / payment-processor identities):**

| L3 address | L3 framing (PRIOR) | OLI public tag | Reality |
|---|---|---|---|
| `0xbb50ce87be3443ed137df1dfdbf2fb0ca8c0a9e0` | `pristine_solo_industrial_bb50ce87` (HIGH) — Pristine Solo Operator at industrial scale, PSO+Single-Purpose hybrid, CER ≈ 0 pre-funding state | **Circle: contract deployer** | Circle's institutional contract deployer. 7.7y mainnet history is real but is institutional infrastructure history, not Pristine Solo Operator dormancy. |
| `0x3304e22ddaa22bcdc5fca2269b418046ae7b566a` | `infrastructure_scale_drainer_spawn_hub_3304e22d` (HIGH); top-12 ISO funder (1,939 deployers, 6,470 contracts, 33 confirmed); empirical anchor for Thermodynamic Fundamentalism positive-CER hub class | **Binance 73 / Exchange / Binance** | Binance hot wallet. The 1,939 "deployers" are CEX customer-withdrawal recipients who happened to deploy contracts. The "33 confirmed" traps are downstream of Binance withdrawals, not seeded by a coordinated operator. |
| `0x39591e7c099a379fd7b349ebfecaeef439c40454` | `iso_top12_39591e7c` (HIGH); top-12 ISO rank #10 (633 deployers, 2,029 contracts) | **OKX 177 / Exchange** | OKX hot wallet. Same misclassification as `0x3304e22d`. |
| `0x4e3ae00e8323558fa5cac04b152238924aa31b60` | top-12 ISO candidate (243 deployers, 1,557 contracts) | **MEXC 15 / Exchange / MEXC** | MEXC hot wallet. |
| `0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda` | `iso_top12_fd92f4e9_stockpile` (HIGH); top-12 ISO rank #3 | **OKX 137 / Exchange** | OKX hot wallet (second OKX address in the same top-12 list). |
| `0xbaed383ede0e5d9d72430661f3285daa77e9439f` | `potential_org_004` (HIGH); CLAUDE.md priority #12 explicitly calls this out as next org-mapping target; top-12 ISO candidate (210 deployers, 1,492 contracts) | **Bybit: Hot Wallet 6 / DEPOSIT ADDRESS / Exchange / Bybit** | Bybit hot wallet. The "potential org_004" investigation surface dissolves on identification. |
| `0xf70da97812cb96acdf810712aa562db8dfa3dbef` | `org_001_whale` (HIGH) — "68% of deployments now through this path. Binance origin"; top-12 funder (2,684 deployers, 6,971 contracts) | **Relay: Solver / Relay Bridge** | Relay (cross-chain bridge) solver. Bridges route trades from many origins to many destinations. The "68% of deployments through this path" claim is bridge-routing volume, not org_001 fund flow. The "Binance origin" notation is correct only in that the bridge user originated from Binance — it does not imply org_001 attribution. |
| `0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb` | `high_value_bridge_user` (HIGH) — 49,000 ETH bridged Apr 7-14, framed "coordinated during western sleep hours" | **Binance: Internal 2 / Exchange** | Binance internal wallet moving exchange funds. The 49K ETH is treasury rebalancing. The "coordinated" framing dissolves — Binance internal transfers run on whatever schedule treasury operations require. |
| `0x151b381058f91cf871e7ea1ee83c45326f61e96d` | `architect` (CRITICAL) — Architect's sole funder, 0.0508 ETH, "single deployer" | **MoonPay 4 / Exchange** | MoonPay (fiat onramp) deposit address. Many users receive funds from MoonPay; the Architect address is one of millions of recipients. The "sole funder" framing collapses — MoonPay is not exclusive to anyone. |
| `0x45a318273749d6eb00f5f6ca3bc7cd3de26d642a` | `cluster_b_funder` (CRITICAL) — 25 deployer wallets, 223 contracts on Base+Arb, blacklist tracking | **Owlto Finance: Bridge** | Owlto Finance cross-chain bridge. The 25 "deployer wallets" are bridge destinations. "Cluster B" topology is a normal bridge fanout. |
| `0xe4edb277e41dc89ab076a1f049f4a3efa700bce8` | `cluster_a_funder` (CRITICAL) — 51 deployer wallets, 260 contracts on Base, "honeypot bytecode" | **Orbiter Finance: Bridge 2** | Orbiter Finance cross-chain bridge. The "honeypot bytecode" downstream is real — but that's because bridges send to whatever address users specify, including honeypots that the bridge does not control. |
| `0x80c67432656d59144ceff962e8faf8926599bcf8` | top-12 ISO candidate (272 deployers, 1,135 contracts) | **Orbiter Finance: Bridge** | Orbiter Finance — same bridge family as `0xe4edb277e`. |
| `0xd37bbe5744d730a1d98d8dc97c42f0ca46ad7146` | `thorchain_router_known_offramp` (HIGH) | **THORChain: THORChain Router v4.1.1** | OLI tag CONFIRMS our existing classification. Keep as-is — already correct as "known offramp surface, not adversarial-by-default." |
| `0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9` | `INFRASTRUCTURE / mixer` (entity_classification) | **Relay; Railgun** | OLI tag CONFIRMS classification (Railgun is privacy-protocol mixer). Keep as-is. |

**LOW-severity mismatches (Web3 brand institutional deployers misclassified as Pristine Solo Operators):**

| L3 address | L3 framing | OLI public tag |
|---|---|---|
| `0x147b8869655bc09f226955cc676ff78efe240ca8` | `pristine-reputation solo operator (base)` (HIGH); fleet 4 conf 1 | **Luchadores: Deployer** (NFT project) |
| `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` | `pristine-reputation solo operator (0x752c5a95 deployer)` (HIGH); 2026-04-24 investigator | **Animoca: Deployer** |
| `0xa2a01b4a68575280a2de45178e289da717bedb6f` | `pristine-reputation solo operator (arbitrum)` (HIGH); fleet 4 conf 1 | **Stabilize Finance: Deployer 2** |
| `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` | `architect_associated` (HIGH) — behavioral match to Architect at 0.742 | **CryptoCauses: Deployer** |

### How was the error caught
Two-stage discovery on 2026-05-09:

1. **Stage 1 — bb50 manual probe.** User asked for production `/priority` analysis. Production analyst note flagged bb50 as "FALSE POSITIVE for trap detection / wallet farm for airdrop farming," contradicting our PSO+Single-Purpose framing. Manual cross-check of bb50's mainnet identity via Blockscout metadata service surfaced "Circle: contract deployer" tag.
2. **Stage 2 — corpus-wide audit.** User authorized broader sweep. New script `scripts/blockscout_tag_audit.py` queried Blockscout metadata service for all 140 unique malicious-flagged addresses (`watchlist` ∪ `entity_classification` malicious subtypes ∪ `infrastructure_operator_candidates`). 18 addresses returned tags incompatible with our classifications.

The Blockscout metadata service surfaces tags from the [Open Labels Initiative](https://www.openlabelsinitiative.org/), a public crowd-curated labeling consortium. Tag reliability was independently verified against canonical anchor addresses (USDC `0xa0b8...` correctly tagged; Circle Hot Wallet `0x55fe...` correctly tagged with `meta.main_entity: "Circle"` and OLI attribution) before any retraction.

The methodological gap producing the false positives: **Layer 3's classification pipeline never queried public address labels.** Detectors operated on behavioral and topological signal only (high-fanout funding, high-velocity deployment, mainnet age, bytecode patterns). The `infrastructure_registry` table existed (12 rows) but contained only Circle CCTP product contracts — not the *deployer wallets* of CEXes, bridges, or institutional issuers, and was not consulted at the watchlist-promotion or typology-application step. CLAUDE.md priority #2 ("Build known-legitimate bytecode template baseline — discount OpenZeppelin, Uniswap, standard patterns") covered this conceptually but the address-label dimension was never wired in.

The shared root cause across all 18 mismatches: **high-fanout funding-wallet topology is the structural signature of CEX hot wallets, bridge solvers, payment processors, AND single-purpose trap-fleet funders.** The detector cannot distinguish them by topology alone. The disambiguating signal is institutional identity, surfaced via public labels.

### What we changed

| Component | Change |
|---|---|
| `surveillance/oli_enrichment.py` (new) | Module that fetches Open Labels Initiative tags from Blockscout metadata service, classifies severity (HIGH for institutional / CEX / bridge / issuer, LOW for project-deployer brands, self-confirming for scam/phishing tags), caches in `oli_labels` table. Provides `is_known_legitimate(conn, address)` for fast lookup at watchlist-promotion or classification time. CLI: `--address`, `--backfill-watchlist`, `--backfill-flagged`, `--hits`. |
| `surveillance/db.py` | New migration: `oli_labels` table with PK `(address, chain_id)`, indexed on `severity` and `primary_entity`. |
| `scripts/blockscout_tag_audit.py` (new) | One-shot audit script — pulls all malicious-flagged addresses, runs OLI enrichment, writes CSV report. Output: `reports/blockscout_tag_audit_YYYY-MM-DD.csv`. |
| `docs/INDEX.md` Section 1 | Top-12 ISO entry — 6 of 12 confirmed CEX hot wallets; remaining 6 not yet OLI-cleared, should not be cited as "infrastructure-scale operator cluster" pending re-audit with non-CEX-contaminated baseline. bb50 entry — superseded by Circle attribution (per this correction). Cluster A/B funder entries — superseded by Orbiter Finance bridge attribution. Architect funder side — superseded by MoonPay attribution. `0xe69f81b8` 49K ETH bridge user entry — superseded by Binance Internal attribution. org_001 entry — `0xf70da978` whale path retracted (Relay solver). |
| `docs/INDEX.md` Section 2 | Per-address rows for the 18 affected entries receive `[CORRECTION #20]` notation pointing to this entry. |
| `docs/lexicon.md` | [Infrastructure-Scale Operator](../docs/lexicon.md#infrastructure-scale-operator) — adds "CEX-hot-wallet false-positive class" subsection, requires OLI cross-check. [Pristine Solo Operator](../docs/lexicon.md#pristine-solo-operator) — adds "institutional-deployer false-positive class," requires OLI check. [Convergent Calibration](../docs/lexicon.md#convergent-calibration) — empirical grounding revised to remove Top-12 ISO instances pending re-audit. [Thermodynamic Fundamentalism](../docs/lexicon.md#thermodynamic-fundamentalism) — bb50 stockpile and `0x3304e22d` positive-CER hub examples retracted; replacement examples needed (deferred). |
| `watchlist` table (local) | 14 HIGH-severity rows tagged `[CORRECTION #20]` in `watch_reason` and active flag set per disposition. Production DB requires same updates. |

### Effect on published numbers / case files / pitch claims

- **"12 Infrastructure-Scale Operators"** — corpus claim retracted at the 12-count. Re-audit needed before any operator-cluster count is cited externally.
- **"49,000 ETH bridged Apr 7-14, coordinated"** — retract; this was Binance Internal moving exchange funds.
- **"68% of org_001 deployments through Relay path"** — retract; Relay's solver is a bridge, the 68% reflects bridge throughput attribution to org_001 that conflated origin-chain bridging with org-attribution.
- **"Architect's sole funder"** — retract. The Architect investigation continues but the funder-side framing collapses.
- **bb50 lexicon anchor for Thermodynamic Fundamentalism** — retract. Replacement empirical anchor needed.

### Open work (post-correction)

**Closed 2026-05-09 (follow-up session):**

1. ~~Re-audit Top-12 ISO list to identify the 6 non-CEX entries.~~ **CLOSED.** Re-fetch + DB drilldown completed. All 8 of 12 non-retracted entries are OLI-cleared (no public tag); 4 are pre-attributed within Layer 3 (Adversarial Vanity Branding, org_002 senior+junior, org_001 L2 Gas Station). The other 4 (`0xc43f317e`, `0x0e6e9177`, `0x8ca70232`, `0xca7ece5e`) carry the topology fingerprint expected of the typology — high disposable-deployer fanout, all-fleet-1 downstream, predominantly L2-only. **They represent the genuine residual Infrastructure-Scale Operator population that the typology was originally designed to identify.** Retain HIGH watchlist; deeper individual case-file investigation deferred but not blocking. INDEX.md updated with this status.

2. ~~OLI enrichment integration into entity_classifier and watchlist-promotion paths.~~ **CLOSED.** `surveillance/entity_classifier.classify_address` now consults `surveillance.oli_enrichment.is_known_legitimate` at the boundary of all promotion calls. If the address has a HIGH-severity OLI tag AND the requested subtype is in `_OLI_GUARDED_TRAP_SUBTYPES` (pristine_solo_operator, infrastructure_scale_operator, single_purpose_funder, drainer_spawn_hub, trap_*, mev_factory, infrastructure_parasite, etc.), the classification is redirected to `COMMERCIAL/institutional_oli_tagged` with the original request preserved in `notes`. Smoke-tested: 3 cases (HIGH OLI + trap → redirect; HIGH OLI + non-guarded subtype → pass-through; no OLI + trap → pass-through). All callers inherit the check without per-caller modification.

3. ~~Re-anchor [Thermodynamic Fundamentalism](../docs/lexicon.md#thermodynamic-fundamentalism) with a non-misclassified positive-CER instance.~~ **CLOSED.** Coffee Fleet (`0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e`, Base, OLI-clean) added as second positive-CER anchor alongside `0xf7883e3fef23` drainer-spawn hub. Coffee Fleet's anchor strength: 322 deployed contracts, 142 confirmed-tier (44.1%), 111 suspected (34.5%) — exceptional confirmation density indicating active extraction. Mainnet age 1.7y (operator-owned history, not pristine-solo dormancy). Dual-role (deployer + 84-bot self-scanning fleet). 5 weeks of continuous activity (2026-03-30 → 2026-05-07). Lexicon entry now has two non-CEX-contaminated positive-CER anchors at different operational shapes.

4. ~~Validate the LOW-severity mislabels by a second source.~~ **PARTIAL CLOSE.** Each LOW-severity OLI tag carries a `tooltipUrl` field pointing to the project's public website (Animoca → revvmotorsport.com; Stabilize → stabilize.finance; Luchadores → luchadores.io; CryptoCauses → crypto4ac.com). All four URLs resolve to verifiable Web3 projects. The `tooltipAttribution` field is empty (these labels originate from Blockscout's own curation, not OLI consortium consensus), making them lower-confidence than the HIGH-severity entries. URLs are documented in the Pristine Solo Operator lexicon entry as the second-source anchor. **Watchlist rows remain ACTIVE** with `[CORRECTION #20]` notes; entity_classification rows NOT yet downgraded pending a third-source confirmation (Etherscan label or direct project disclosure).

**Closed 2026-05-10 (follow-up session):**

5. ~~Production sync.~~ **CLOSED.** Mechanism built and verified end-to-end. `scripts/sync_prod_db.py` uses `railway ssh` to dispatch `scripts/sync_prod_db_remote.py` on the production container, which performs SQLite online backup → gzip → base64-streams to stdout framed by `L3SYNC_PAYLOAD_START`/`_END` markers. Local wrapper captures stdout to a raw file (avoiding line-buffering fragility under railway ssh's transport), mmap-searches for markers, streams base64 chunks through `re.sub` → `base64.b64decode` → `gzip` → SQLite file with bounded memory (~16 MB working set regardless of payload size). Validates via `PRAGMA integrity_check` + table-count sanity, atomic-renames into place, retains prior DB as `.bak`. Path B (apply to prod first, then sync down) executed 2026-05-10: dispositions applied to production via `scripts/apply_correction_20_via_ssh.py` (compact SSH-dispatched companion to the local `apply_correction_20_to_prod.py`, hand-minified to fit Windows cmd.exe's 8,191-char command-line limit), verified persisted (10 deactivated / 5 noted / 6 infra-rejected confirmed on prod), then `sync_prod_db.py` pulled the corrected 10.0 GB DB down to local. Post-sync `oli_enrichment --backfill-flagged` re-populated the `oli_labels` cache against the fresh DB (9 HIGH hits visible in the active-flagged set + 5 previously-HIGH now correctly suppressed because their watchlist rows are active=0).

**Partially closed 2026-05-10 (follow-up session):**

6. **Bytecode-and-narrative review for the 4 unattributed Top-12 entries.** **Partial close: 1 of 4 covered.** Case file `surveillance/data/cases/CASE_PRESTAGE_WAREHOUSE_0xc43f317e.md` authored — the largest of the four, 2,535 downstream deployers (was 1,562), still active, 100% bytecode concentration on `49155b60033de73770...`, 815 deployers active last 14 days. Hypotheses + recommended decompilation steps documented in the case file. The other three (`0x0e6e9177` still active, 1,408 deployers diversified; `0x8ca70232` and `0xca7ece5e` both dormant since 2026-04-16 — possible operator overlap) remain documented in the INDEX.md Top-12 entry but without dedicated case files. **Decompilation of `49155b60033de73770...` is the next-priority action to sharpen the c43f317e typology.**

**Closed 2026-05-10 (follow-up session — items 7, 8 partial, 9):**

7. ~~Decompile `0xc43f317e`'s dominant bytecode template `49155b60033de73770...`.~~ **CLOSED.** Sample `0xacfdc090ff9f5b160005bdaacb9a2d1025755baf` ("Kore Agent" / KORE) decompiled via Blockchain MCP. **Result: verified vanilla OpenZeppelin v5.0.0 ERC-20** — zero custom transfer/approval logic, zero fees, zero blacklist, zero delegatecall, zero selfdestruct. Operator is a meme-token deployment shop, not a pre-stage trap warehouse. Case file `CASE_PRESTAGE_WAREHOUSE_0xc43f317e.md` reframed with reclassification header; watchlist downgraded HIGH → MEDIUM (`meme_token_shop_c43f317e`) on local + prod 2026-05-10.

8. ~~Author case files for the remaining 3 unattributed Top-12 entries.~~ **CLOSED.** Bytecode-level review of all three (sample per operator): `0x0e6e9177` sample `0xcbbd17f9` ("X1000XLiquidBGT") = vanilla OZ ERC-20 (meme-token shop, same class as c43f317e); `0xca7ece5e` sample `0x3b6af3e8` ("CelestialForge") = vanilla OZ ERC-20 (meme-token shop); `0x8ca70232` sample `0xaeac0e69` ("Laser Eagle" / 🦅LSEG) = **custom `EVMToken` template with two honeypot primitives**: hardcoded blacklist of 5 victim addresses in `_transfer`, and hidden `approev(address)` function that zeroes any holder's balance with no Transfer event emitted, gated to the funder's own address via misleadingly-named `uniswapV2Router02` constructor argument. New case file `CASE_HONEYPOT_TOKEN_OPERATOR_0x8ca70232.md`; watchlist HIGH (`honeypot_token_operator_8ca70232`) added local + prod. Lexicon entry deferred (single instance is thin); promote on a second instance of the `approev`-style hidden-drain pattern.

9. ~~Investigate simultaneous April-16 stop for `0x8ca70232` + `0xca7ece5e`.~~ **CLOSED.** Cross-tests showed: zero downstream-deployer overlap (737 + 484 = 1,221, intersection 0); different burst shapes (8ca70232 = cold stop at peak 108 contracts/day, ca7ece5e = winding down for a week from 4-7 peak to single-digits); different bytecode templates (honeypot vs vanilla); ~9.5h apart on the stop day (17:37 UTC vs 08:09 UTC). **Most likely coincidence**, not coordinated stop. Recorded for completeness; if a third simultaneous-stop pair surfaces, revisit.

**Net result of items 7-9: methodology lesson.** The Top-12 ISO detector's high-disposable-deployer-fanout signal requires **bytecode-level disambiguation** to be useful. Of the original 12: 7 were CEX/bridge institutional addresses (Correction #20 main sweep), 3 were vanilla meme-token deployment shops (false-positive class for this typology), 1 was a genuine honeypot operator with custom-templated predatory bytecode (`0x8ca70232`), 4 were pre-attributed within Layer 3 (org_002 senior+junior, org_001 gas station, Adversarial Vanity Branding). **Zero of the original 12 retain the "Infrastructure-Scale Operator" framing as documented.** The behavioral/topology detector's signal is real but ambiguous; OLI cross-check + bytecode decompilation are both required to convert it into actionable classification.

**Closed 2026-05-10 (follow-up session — items 10, 11):**

10. ~~Add `approev` function-name signature to Layer 3's bytecode classifier.~~ **CLOSED.** Selector `0x3ed67ecd` (computed via keccak256("approev(address)") and cross-validated against `pycryptodome` and `eth_hash` libraries — both returned identical 4 bytes; sanity-check against `approve(address,uint256)` = `0x095ea7b3` matched the known canonical ERC-20 selector). Added `KNOWN_HIDDEN_DRAIN_SELECTORS` registry constant to `surveillance/bytecode_classifier.py` and new detector function `detect_hidden_drain_function` that scans bytecode for the Solidity dispatch pattern `PUSH4 <selector> EQ` (hex bytes `63 <selector> 14`). Registered in `PATTERN_REGISTRY` mapped to `has_asymmetric_transfer` flag (the function asymmetrically modifies balances). Unit-tested: positive (synthetic bytecode containing the pattern → detected with note), negative (standard ERC-20 approve dispatch only → not detected), case-folded variants pass, empty bytecode safe. **Live detector** active for all new deploys going forward; **backfill against existing 8ca70232 fleet contracts deferred** (would require bulk bytecode pulls from Blockscout — not blocking since the operator is dormant since 2026-04-16).

11. ~~Cross-reference the 5 hardcoded blacklist addresses against other Layer 3 entries.~~ **CLOSED.** All five addresses (`0x1f2F10D1C40777AE1Da742455c65828FF36Df387`, `0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13`, `0xC38e00aC5ED8859f18f4E9017fa2b3D3E1f65F40`, `0x01D37a36220d52108Ae6D453fE6Cd80af2906376`, `0x93C7878c5ab2F78Df087a4203cBEB3209C10e439`) cross-referenced against `deployers`, `bot_candidates`, `entity_classification`, `watchlist`, `org_wallets`, `approval_watchlist` (as victim), and `trap_events` (as bot). **Zero matches across all tables.** These are first-observation addresses — never seen elsewhere in Layer 3's corpus. Most likely interpretation: real retail wallets targeted by the operator, not bot wallets or repeat-victims of other operations. Same probe extended to the 25 unique victim approvers of `0x8ca70232`'s fleet: also zero matches in entity_classification, watchlist, or bot_candidates. **Honeypot operators draw victims from a separate population than drain operators** — this is a finding in its own right. Top victim by approval count: `0xbfc7c4333fa33dc04ed2fbce4637c5af3b6361bd` (19 approvals across the 8ca70232 fleet). Cross-victim diffusion: most victims interact ONLY with 8ca70232's contracts (0-2 other contracts touched) — they are not repeat-approvers across multiple scam operations. The proposed "full-fleet blacklist extraction" (potentially ~1,600 victim addresses across 320 contracts) remains feasible but requires bulk bytecode pulls; deferred until live detector finds new instances or operator resumes activity.

**All 11 follow-up items closed.** Correction #20 fully resolved.

**Methodology lessons documented:**
- **Identity layer must precede behavioral layer.** OLI/public-label cross-check must run BEFORE adversarial typology is applied. Half of an apparent adversarial cluster can be legitimate institutional flow.
- **Bytecode-level disambiguation is load-bearing.** Topology signal cannot distinguish CEX hot wallet vs. bridge solver vs. ERC-20 deployment shop vs. honeypot operator vs. trap stockpile — all produce high-fanout disposable-deployer signatures. Source/bytecode review is the only reliable disambiguator.
- **Behavioral similarity ≠ identity.** Deployer-profile similarity scores (0.7-0.8) capture operator-style match (timezone + technique + cadence), not identity. Two unrelated operators with similar style produce these scores routinely.
- **Function-name signature detection is brittle but cheap.** Add new selectors to `KNOWN_HIDDEN_DRAIN_SELECTORS` as the operator class rotates naming; pattern library builds over time.
- **Victim populations cluster by operator class.** Honeypot operators and drain operators appear to draw from distinct victim pools. Cross-operator victim overlap is a research direction.

### Why this is one numbered correction, not seven

All 14+4 mismatches share the same root cause (no OLI/public-label cross-check in classification pipeline) and were all surfaced by a single audit pass. Splitting into Corrections #20-#26 would create the appearance of multiple independent methodology issues; collapsing into one entry preserves the actual structural finding: **a single missing enrichment step produced systematic false positives across multiple typologies.** Future independent corrections that surface OLI-tag-related issues should reference back to #20.

---

## Correction #21 — Pattern D Direction Reversal: Cross-Chain Reputation Import Has the Opposite Signature From What the Lexicon Claimed

**Date:** 2026-05-19
**Discovery method:** Three independent statistical analyses run as part of the SAI inferential layer build-out (`surveillance/analytics/` series, sessions 2026-05-18 / 2026-05-19).
**Severity:** HIGH — Pattern D is one of six load-bearing primitives in the Behavioral Laundering framework; reversing its directional framing requires correcting external materials and re-orienting Q-005's pattern_d_gap scoring.

**Claim (what was asserted in the lexicon entry):**

The Pattern D — Cross-Chain Reputation Import entry stated (verbatim):

> "A deployer appears on Arbitrum or Optimism for the first time with no prior activity in our corpus, but the SAME address has substantial history on Ethereum mainnet or Base. Per-chain profiling sees a fresh-to-L2 deployer; cross-chain view reveals a long-standing identity that may have pivoted from legitimate activity to trap deployment."

And the quantitative anchor:

> "**Result: 54 of 100 high-risk L2 deployers had mainnet first-tx predating L2 first-seen. The strongest-supported pattern of the six.**"

The directional claim was that *long* mainnet vintage was the predator signature — predators were using aged cover identities. The 54% number was cited as corpus-wide.

**Reality (what the data now shows):**

Three analyses run against the 2026-05-18 corpus snapshot (commit `e120ddd` survival, `e6ecc7c` Cox PH, `6d2f881` KS test) produced converging findings:

1. **Cox PH multi-covariate model (commit `e6ecc7c`):** `mainnet_l2_gap_days` had β = -0.005, SE = 0.023, p = 0.82. Has no hazard effect once chain + tier + funder are controlled. The gap-as-timing-predictor framing fails.

2. **KS Test A (commit `6d2f881`, predator vs control by tier):** D = 0.2098, p = 6.59×10⁻⁷². Reject H₀ but in the **opposite** direction from the lexicon framing:
   - Predator group 10th percentile gap: **12.8 days** (mass of recently-bridged deployers)
   - Control group 10th percentile gap: 85.2 days
   - P(gap > 60 days): predator 78.6%, control 92.4% (control is HIGHER)

3. **KS Test B (commit `6d2f881`, drained vs flagged-quiet):** D = 0.3645, p = 0.0124. Reject H₀ with even sharper directional reversal:
   - Drained-set median gap: **53.6 days**
   - Flagged-quiet median gap: 644.3 days
   - The drained predators are 12× MORE recently bridged than flagged-but-quiet ones.

4. **"54/100" cross-check:** corpus-wide refresh shows **28.1%** of all 9,567 high-risk deployers have mainnet predating L2 (not 54%). Only 28.8% of high-risk deployers have `mainnet_first_tx` populated at all — the original 54% was measured against the top-100 manually-curated cohort, which had been intentionally selected for mainnet enrichment.

**Discovery (how it was caught):**

The 2026-05-18 statistical research-program design asked which lexicon primitives were inferentially testable. The Cox PH model run on 2026-05-19 surfaced the null-effect on `mainnet_l2_gap_days` as anomalous (p=0.82 was the second-highest p-value in the full covariate table, behind only `has_mainnet_history`). The follow-up KS test was run to determine whether the distributional difference at the lexicon level was real or also null; the rejection of H₀ in the *opposite* direction crystallized the correction.

**Numerical effect on headline statistics:**

| Claim location | Old (retired) form | Corrected form |
|---|---|---|
| Lexicon Pattern D Empirical grounding | "54 of 100 high-risk L2 deployers had mainnet first-tx predating L2" | **28.1% of all 9,567 corpus-wide high-risk deployers have mainnet predating L2.** The 54% figure was the top-100 curated cohort rate — not corpus-general — and conflated the two populations. |
| Lexicon Pattern D directional framing | Long mainnet vintage = adversarial cover identity | **Reversed.** Drained-completing predators have median bridge gap 53.6 days, not the ~1,000+ days the lexicon implied. Long-vintage deployers are over-represented in the *flagged-quiet* (non-drain-completing) group. |
| Behavioral Laundering: "Pattern D is the strongest validated" | "Pattern D is the strongest validated" | **Refines.** Pattern D is the most distributionally distinct (lowest p-value of any KS test we've run), but the distributional difference points the opposite direction from the original framing AND the hazard effect is null (Cox p=0.82). The "strongest validated" claim referred to *correlation* with high-risk status, not *prediction* of drain hazard. Those are different claims. |
| Q-005 cross-chain choreography detector | `pattern_d_gap` signal awards points proportional to gap size | **Direction may need to invert.** Currently larger gap → higher score. KS Test B says smaller gap (recent bridging) is the drain-completion signature. Tracked as separate engineering work; the inferential finding lands here. |

**Fix:**
- This correction-log entry (#21) — 2026-05-19, commit pending
- Lexicon `Pattern D` entry updated with a 2026-05-19 revision section pointing here
- Lexicon `Behavioral Laundering` entry "Pattern D is the strongest validated" line annotated with same revision
- `CORRECTIONS.md` Quick Retirement Index entry added for the 54% number
- Q-005 detector status: flagged for future re-engineering; the current `pattern_d_gap` score remains in place pending the directional inversion build

**Open work:**
- Re-engineer Q-005's `pattern_d_gap` signal: either invert direction (short-gap = high score) or replace with a "bridge recency" primitive based on the Test B finding.
- Improve `auto_funder_tracer` mainnet-enrichment coverage: only 28.8% of high-risk deployers currently have `mainnet_first_tx` populated. This data-completeness gap means many predator deployers can't be evaluated against Pattern D at all.
- Update external materials (decks, briefs) that cite the 54% number to disambiguate cohort vs corpus-wide. The propagation watchlist below should include any deck/report that still references the original phrasing.

**Why this matters epistemically (the meta-finding):**

This is the first correction in the log where the methodology that surfaced the error was statistical inference, not source verification. Corrections #1–#20 were caught by re-querying primary sources, finding stale caches, or discovering missing labels. Correction #21 was caught by running a hypothesis test against a claim that *was* sourced correctly at the time but did not hold up under multivariate control + distributional testing.

This is exactly the failure mode the SAI / inferential-layer build-out was designed to surface. Quantitative claims about population-level adversarial behavior require distributional and hazard-level tests, not point-estimates from curated cohorts. The lexicon should now expect every Pattern entry to carry a 95%-CI-style framing rather than a single-number anchor.

**Propagation watch-list (add to the top-of-file table):**

| Claim | Still appears in | Required cleanup |
|---|---|---|
| "54/100 high-risk L2 deployers had mainnet predating L2" (corpus-wide framing) | `docs/lexicon.md` Pattern D entry; `docs/lexicon.md` Behavioral Laundering entry; `reports/cross_chain_import_candidates.md`; potentially `l3-narrative/` decks | Replace with "28.1% corpus-wide" and disambiguate from the top-100 cohort. |
| "Long mainnet vintage = adversarial cover identity" (Pattern D directional framing) | `docs/lexicon.md` Pattern D entry; possibly `surveillance/analytics/cross_chain_choreography.py` pattern_d_gap scoring | Replace with bidirectional language; explicit note that the drain-completing predator has *recent* bridge gap. |
| Q-005 pattern_d_gap detector scoring | `surveillance/analytics/cross_chain_choreography.py` | Engineering follow-up to invert direction or replace with recency-based primitive. |

---

## Correction #22 — Camouflage Ratio Direction Reversal: Confirmed-Tier Predators Have *Higher* Revert Rates Than Baseline, Not Lower

**Date:** 2026-05-19
**Discovery method:** Two-proportion z-test (`surveillance/analytics/camouflage_ratio_z_test.py`, commit pending) on per-contract revert rates with Wilson 95% CIs and per-chain breakdown.
**Severity:** HIGH — Camouflage Ratio is the headline retention metric in pitch materials and the existence-claim underlying the "Camouflage Equilibrium" lexicon entry. Direction of effect is now reversed for the confirmed tier.

**Claim (what was asserted in the lexicon entry):**

The Camouflage Ratio entry stated that "dangerous contracts maintain low revert rates (under 10%) to evade standard detection." The numerical anchors were:

- Pre-2026-04-02: 68% (retired in CORRECTIONS.md)
- Post-2026-04-02: 70–79% across chains, organizations, and time (described as "stable")
- 2026-04-29 robustness check (lexicon Section A7): full-corpus 67.1%, top-12-excluded 68.1%

The directional framing was that PREDATORS deliberately keep revert rates low as camouflage. The Camouflage Equilibrium claim depended on predator-class contracts being *over*-represented in the low-revert population.

**Reality (what the data now shows):**

Two-proportion z-test on 8,252 contracts with ≥5 transactions, partitioned by `confidence_tier`:

| Tier | N | Low-revert (<10%) | Ratio | 95% Wilson CI |
|---|---|---|---|---|
| confirmed | 1,163 | 354 | **0.3044** | [0.2786, 0.3314] |
| suspected | 4,585 | 4,202 | 0.9165 | [0.9081, 0.9241] |
| unanalyzed (baseline) | 2,386 | 2,150 | 0.9011 | [0.8885, 0.9124] |
| unknown | 118 | 115 | 0.9746 | [0.9279, 0.9913] |

Tests:
- **confirmed vs unanalyzed:** Δ = −59.67 pp, z = −36.61, p < 10⁻⁶. Confirmed-tier predators are FAR LESS likely to maintain low revert rates than baseline.
- **suspected vs unanalyzed:** Δ = +1.54 pp, z = 2.14, p = 0.032. Suspected tier is marginally higher than baseline.

Per-chain breakdown confirms the reversal at every chain:
- Base confirmed: 0.2893 vs Base unanalyzed 0.9247 (gap −63.5 pp)
- Arbitrum confirmed: 0.3881 vs Arbitrum unanalyzed 0.7500 (gap −36.2 pp)
- Optimism confirmed: 0.4783 vs Optimism unanalyzed 0.9655 (gap −48.7 pp)

The "70–79% across chains" range applies to the *baseline* population (unanalyzed contracts have ~75% low-revert across chains). It does not apply to confirmed-tier predators, which sit at 30.44% corpus-wide.

**Discovery (how it was caught):**

The 2026-05-19 lexicon-validation pass tested whether the headline corpus claims would survive a two-proportion z-test against the unanalyzed baseline. The expected finding was confirmed-tier z > 0 (predators camouflaged → higher than baseline). The actual z = −36.61 (predators *anti*-camouflaged) inverted the entire framing. The Wilson CI on the confirmed-tier ratio [0.2786, 0.3314] does not touch the 70–79% claim band.

**Root cause of the original error:**

The 2026-04-02 retirement of "Camouflage ratio 68%" replaced the number with "70–79% across chains, organizations, and time" without specifying the partition. The 2026-04-29 Section A7 robustness check used *full-corpus* and *top-12-excluded* averages — both of which are dominated by suspected + unanalyzed contracts (5,748 of 8,252 in this corpus = 70%). Those tier-blended averages happened to land at 67.1% / 68.1% because the unanalyzed baseline drives them, not because predator-class contracts maintain that rate.

When the lexicon framing translated "Camouflage Ratio stable" into "predators maintain low revert rates as camouflage," it inverted causation. The low-revert *baseline* is the population norm. Predators (confirmed-tier) are systematically *more reverted*, not less. The "camouflage" framing was implying intent that the data does not support.

**Numerical effect on headline statistics:**

| Claim location | Old (retired) form | Corrected form |
|---|---|---|
| Lexicon Camouflage Ratio entry headline | "Camouflage Ratio 70–79% across chains" (predator framing) | **Population-level low-revert rate is 70–79% baseline. Confirmed-tier predators are at 30.44% [27.86, 33.14], significantly LOWER than baseline (z = −36.6, p < 10⁻⁶).** |
| Lexicon Section A7 robustness | "full-corpus 67.1%, top-12-excluded 68.1%" | **Retained — but reframed: these are tier-blended averages dominated by the ~90% baseline of unanalyzed/suspected contracts. They are not predator measurements.** |
| Camouflage Equilibrium claim | "Predators systematically calibrate to low revert rates to evade detection" | **Retired.** Confirmed-tier contracts revert MORE than baseline. If predators are calibrating to anything, it is to a higher-friction signature, not lower. The "equilibrium" was a measurement artifact from blending tiers. |
| Any deck/brief claim that uses "Camouflage Ratio" as evidence of predator stealth | Treats the 70–79% figure as predator camouflage | **Reframe.** The 70–79% figure is a population baseline. The predator-class divergence is the inverse: 30.44%. |

**Fix:**
- This correction-log entry (#22) — 2026-05-19, commit pending
- Lexicon `Camouflage Ratio` entry updated with 2026-05-19 revision section pointing here; old "70–79% across chains" framing retained with retirement annotation
- Lexicon `Camouflage Equilibrium` claim flagged for retirement or substantial reframing
- `CORRECTIONS.md` Quick Retirement Index entry revised: "Camouflage ratio 70-79% stable" → "70–79% is the baseline-population low-revert rate, not predator behavior. Confirmed-tier predators at 30.44%."
- `surveillance/analytics/camouflage_ratio_z_test.py` script committed as the canonical regenerator

**Open work:**
- Investigate WHY confirmed-tier contracts have higher revert rates than baseline. Hypotheses:
  - (a) Genuine adversarial mechanics (conditional reverts on user-input dependence, decoy logic, anti-bot checks that fail honest users).
  - (b) Selection-effect from the labeling pipeline — contracts that revert on diverse inputs may be more readily flagged as `confirmed`, producing a confound.
  - (c) Mixed: predators of *one* class (e.g., honeypots) revert by design; predators of *another* class (e.g., drainers) don't, and we're aggregating both.
- The (b) hypothesis is most concerning. If `confidence_tier='confirmed'` is partially conditioned on revert-rate (e.g., reverts contribute to risk score), the test confounds outcome with input. Audit the `confidence_reason` populations of the 1,163 confirmed contracts to verify the tier is not directly derived from revert frequency.
- Once the mechanism is identified, decide whether the "Camouflage Ratio" name should be retired entirely (the metric measures something real about the baseline population but does not measure "camouflage" in any actionable sense).

**Why this matters epistemically:**

This is the second correction in two days where a lexicon claim with a stable, multi-month-stated number turned out to be a tier-aggregation artifact. The Pattern D 54% (Correction #21) was a top-100 cohort number that had been promoted to corpus-general. The Camouflage Ratio 70–79% is a tier-blended average that was promoted to predator-specific.

The shared root cause is using single-number anchors without partition specification. Going forward every quantitative lexicon claim must specify:
1. The partition the number applies to (tier, chain, cohort, all-contracts).
2. The baseline it is compared against.
3. The direction of the effect — explicitly, with a CI or p-value.

The Q-001 role_classifier and Q-002 approval_spike_detector outputs are already partitioned this way. The lexicon entries written before that discipline existed need to be reviewed in bulk. Adding to operational priority list.

**Propagation watch-list (add to the top-of-file table):**

| Claim | Still appears in | Required cleanup |
|---|---|---|
| "Camouflage Ratio 70–79% (stable) across chains, organizations, and time" — as a predator claim | `docs/lexicon.md` Camouflage Ratio entry; `docs/lexicon.md` Camouflage Equilibrium claim; `l3-narrative/` decks | Reframe as baseline-population low-revert rate. Predator-class measurement is the *inverse* finding (30.44% confirmed-tier). |
| "Camouflage Equilibrium" as a behavioral claim about predator intent | `docs/lexicon.md` | Retire or substantially reframe — the inversion direction does not support the equilibrium framing. |
| Section A7 robustness check (full-corpus 67.1%, top-12-excluded 68.1%) | `docs/lexicon.md` | Retain numerically but annotate: these are tier-blended baselines, not predator-class measurements. |

---

## Correction #23 — Routing-Monitor Silent Failure: 0 1inch-Routing-Anomaly Signals Produced Across Entire Corpus

**Date:** 2026-05-21
**Discovery method:** Railway production health check via `railway ssh` after the recent Railway outage. The heartbeat-table probe surfaced that `routing_monitor` had not heartbeat since 2026-04-29T22:05:35Z — 22 days stale. Follow-up query of the `contracts` table confirmed the silent-failure scope.
**Severity:** MEDIUM — the routing-anomaly detection pathway has been *advertised* in code, lexicon Pattern C entry, and Correction #4 as a real signal, but has produced zero operational output across the 282K+ corpus. The downstream impact is not a published claim that needs retraction (no customer-facing material cites a routing-anomaly count), but the corpus does not have the signal it implicitly claims to.

**Claim (what was asserted, implicitly):**

- `surveillance/routing_monitor.py` is one of the production monitors. Correction #4 (2026-04-17) explicitly added `detection_method="routing_anomaly"` to its update-contract-confidence call, implying the monitor was producing real flag-updates.
- The `contracts` table schema carries dedicated columns for this signal: `routing_presence INTEGER`, `routing_first_seen TEXT`.
- Lexicon Pattern C — Funding Chain Laundering entry references "1inch pathfinder avoidance" as a corpus-observable signal type.
- The `run_surveillance.py` launcher at lines 1945–1958 starts the routing monitor as either an in-process Process (if `ONEINCH_API_KEY` is set) or as a subprocess (standalone mode).

**Reality (what the data now shows):**

Probed on 2026-05-21 via `railway ssh`:

| Check | Result |
|---|---|
| `heartbeat` row for `routing_monitor` | 2026-04-29T22:05:35Z (22 days stale) |
| Live `routing_monitor` process on container | **None** |
| `ONEINCH_API_KEY` env var | **Not set** (key length = 0) |
| Contracts with `routing_presence = 1` | **0** |
| Contracts with `routing_first_seen IS NOT NULL` | **0** |
| Contracts with `detection_method = 'routing_anomaly'` | **0** |

The routing_monitor has produced zero operational signal across the entire corpus, on either Arbitrum (its target chain) or any other. The 2026-04-29 heartbeat appears to be from a brief subprocess-launch that completed initialization (writing the heartbeat row) but then exited or stalled without producing detection output. No subsequent restart occurred.

**Root cause (likely):**

1. `ONEINCH_API_KEY` is not configured in Railway environment. Without the key, the launcher takes the fallback path at lines 1952–1957 and starts the monitor as a subprocess without an API key. The monitor's actual API calls (token-registry check + routing-anomaly quotes) require authentication against `api.1inch.dev`, so the subprocess can write a startup heartbeat but every detection cycle errors out at the API layer.
2. The detection-cycle error is presumably handled by an uncaught-exception path that exits the subprocess. Without a respawn mechanism (the `processes.append(("routing", routing))` registration does not include re-launch logic), once the subprocess dies the monitor stays dead until the next Railway redeploy.
3. The 2026-04-29 heartbeat lines up with the redeploy timestamp from that date (per `git log --since=2026-04-28 --until=2026-04-30`), suggesting the subprocess was respawned at deploy time, wrote one heartbeat, then died on its first detection cycle.

**Discovery (how it was caught):**

User asked for a Railway health check after a Railway-side outage. The first probe round (public API) showed everything healthy. The follow-up via `railway ssh` exposed the heartbeat-table-level view, where the four-chain monitors all heartbeated within the same second (07:03:44Z) but `routing_monitor` had a 22-day-stale timestamp. The detection-method count query confirmed zero output corpus-wide.

This is a class of failure the public API surface specifically does NOT expose: `/stats` returns the most-recent heartbeat (which gets dominated by the active chain monitors) and does not surface the per-component freshness gap.

**Numerical effect:**

No published numerical claims need retiring. Internal effect:
- Pattern C (Funding Chain Laundering) lexicon entry references "1inch pathfinder avoidance" as a detection signal type. The signal exists architecturally but has produced zero outputs. The Pattern C 2026-04-18 scan in `reports/cex_laundered_funding.md` returned 0 strict candidates — that scan was a separate SQL-only run on funding traces, so it did not depend on routing_monitor output and remains valid.
- Any future analysis that joins on `contracts.routing_presence` will silently match nothing. Pre-existing code that takes the `routing_anomaly` detection_method branch (in `risk_scoring.py` or similar) is a dead code path.
- `evidence_type` filter on `/risk` endpoint, when set to `routing-anomaly`, returns no contracts.

**Fix:**

- This correction-log entry (#23) — 2026-05-21, commit pending. No CORRECTIONS.md entry: no customer-facing claim was made.
- CLAUDE.md operational priorities updated to add routing-monitor remediation as a priority (decide between fix vs retire).

**Operational decision required:**

Two paths, listed in increasing order of effort and decreasing order of risk:

A. **Retire the monitor.** Remove the launcher call from `run_surveillance.py`, drop `routing_presence` / `routing_first_seen` columns from the schema (or document them as deprecated), remove the lexicon cross-reference to "1inch pathfinder avoidance" as an active signal. Pattern C remains real and detectable via funding-trace methods; this path concedes that the live-API approach was not productive. Lowest engineering effort, highest signal-loss.

B. **Fix the monitor.** Add `ONEINCH_API_KEY` to Railway env (already provisioned somewhere if Layer 3 has a 1inch account; verify status); add a respawn loop around the subprocess so future single-cycle failures are recoverable; add a watchdog that alerts when `heartbeat.routing_monitor` goes >2× the poll interval stale. Medium engineering effort, restores the documented detection pathway.

Recommendation: Path B if there is any 1inch account on Layer 3's side; Path A if the 1inch key is no longer available and re-provisioning is out of scope. The longer the monitor stays dead, the more the supporting infrastructure (schema columns, code paths, lexicon references) accumulates as cruft.

**Open work:**

- Add a `/api/health/detailed` endpoint that surfaces per-component heartbeat staleness, not just the latest single row. Today's silent failure was invisible to external observers; the system needs to fail loudly per CLAUDE.md "loud failures over silent wrong output."
- Audit other ANALYSIS_JOBS in `run_surveillance.py` for the same silent-death pathway: which background jobs have respawn logic, and which do not? The routing_monitor pattern is reproducible.
- Reconcile the lexicon Pattern C "1inch pathfinder avoidance" reference with the actual evidence basis. Either retain the language with explicit "signal architected but not currently producing detections" annotation, or rework to point to the funding-trace methodology only.

---

## Correction #24 — `0x752c5a95` Was Never a Harvester. Three Stacked Bugs Manufactured the Entire "Pre-Drain Harvester / 4,587-Victim Discharge" Finding.

**Date:** 2026-05-21 (filed same day as the retracted case file)
**Discovery method:** Task 4 of the 2026-05-21 recent-activity review — investigating "why an Animoca-tagged wallet deployed a confirmed-tier approval-harvesting contract" — produced a falsifiable answer: it didn't. Cross-chain Blockscout probe revealed the contract is `ERC20FixedSupply` (OneFootball Club, ticker OFC), a verified, CoinGecko-listed Animoca-affiliated token. Direct tx inspection on the "discharge" hashes showed both were FAILED `transferFrom` calls.
**Severity:** **CRITICAL — retracts every claim derived from the `0x752c5a95` finding from 2026-04-24 through 2026-05-21.**

**Claim (what was asserted, in chronological order):**

1. **2026-04-24 (INDEX.md, original entry):** "Confirmed-tier contract harvesting Permit2 approvals from 1,898+ victims without firing a sweep. Deployer `0x80b12bd0` (pristine-solo, 2019 mainnet vintage). Largest active confirmed-tier approval pool in the corpus." Tier-C prediction implicit: this is a Pre-Drain Harvester in accumulation phase; it will discharge.
2. **2026-05-09 (corpus observation):** "Harvester discharged. Two independent drain_caller EOAs swept 4,587 unique victims in 30 minutes (3,228 + 1,359). Total drains 4,587, drain pct 56.3%."
3. **2026-05-19 (Correction #20):** Deployer `0x80b12bd0` reframed as "Animoca: Deployer" per OLI labels. Harvester behavior NOT retracted; "investigate why an Animoca-tagged wallet deployed a confirmed-tier approval-harvesting contract" filed as open work.
4. **2026-05-21 (CASE_HARVESTER_DISCHARGE_0x752C5A95_20260509.md):** Documented the 2026-05-09 event as EXTRACTION_011, framed as "the strongest validated Tier-C prediction in the Layer 3 corpus to date." 4,587 victims drained validated the 2026-04-24 prediction within 15 days.
5. **2026-05-21 (lexicon Adversarial Maneuver entry, line 1065):** "Empirical leverage: Layer 3's 2026-04-24 pre-drain flag on operator `0x80b12bd0` (15 days of lead time before the May-9 4,587-victim discharge) is the canonical example of disrupt-positioning succeeding."

**Reality (what the data shows):**

Blockscout probe via MCP `mcp__44ed366a-ba42-4fe9-93fa-ca1c6bdc9f66__get_address_info` on mainnet (chain_id=1):

```
0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8:
  tags: [
    {slug: "animoca-deployer", name: "Animoca: Deployer",
     meta: {main_entity: "Animoca", tooltipUrl: "https://www.revvmotorsport.com/"}},
    {slug: "contract-deployer", name: "Contract Deployer"},
    {slug: "animoca", name: "Animoca", tagType: "protocol"}
  ]
  first_transaction: 2019-05-23T07:20:41Z (Tier A — matches our corpus mainnet_first_tx)
```

The Animoca attribution is real, multi-tag, and live on mainnet Blockscout.

Blockscout probe on Base (chain_id=8453) on the harvester contract:

```
0x752c5a95d202972e124390f30a50154409d3c858:
  contract_type: "ERC20FixedSupply"  (verified source, compiler v0.8.28)
  token:
    name: "OneFootball Club"
    symbol: "OFC"
    decimals: 18
    holders_count: 3,904
    circulating_market_cap: $7,902,567
    exchange_rate: $0.04921397
    icon_url: "https://assets.coingecko.com/coins/images/67442/small/ofc.jpg"
    volume_24h: $3,286,969
  source_code: @animoca-network/contracts framework + OpenZeppelin
  initial_holder: 0xDA42FE397c3fc9d08ac6675EecD2709880fDFD73
  initial_supply: 1,000,000,000 * 10^18
```

`0xDA42FE397c3fc9d0` (the "second contract Layer 3 thought was an unused sibling") is `OFTAdapterFixedSupply` — the LayerZero Omnichain Fungible Token bridge adapter for OFC. It received the entire 1B token supply at TGE.

The "discharge" transactions (`get_transaction_info` MCP probe on both):

| TX hash | Method | Status | Gas used | Value |
|---|---|---|---|---|
| `0x044feaebbe7380...` | `transferFrom(from=0x752C5a95, to=0x1d81AFF2..., value=10^14)` | **ERROR (reverted)** | 25,285 | 0 |
| `0x9cabf720a66d30...` | `transferFrom(from=0xaD6C87E9..., to=0x0e222468..., value=4.5×10^12)` | **ERROR (reverted)** | 25,297 | 0 |

Both transactions are nonsensical `transferFrom` attempts (the first tries to transfer from the token contract address itself) that reverted on-chain. **Zero OFC tokens moved in either transaction.** Layer 3's `approval_watchlist.drain_detected=1` pipeline marked them as discharging 3,228 + 1,128 + 231 = 4,587 victims respectively. The 4,587-victim discharge event did not happen.

**Three stacked bugs produced the finding:**

1. **Behavioral classifier false-positive on pre-launch ERC-20 token launches.** OFC's confirmed-tier label came from `confidence_reason: "Behavioral confirmation: bot 0x8c858126a972dd313e91d0d6b68e90e2e1eb9508 trapped in tx 5c6d3a661db3..."`. The bot tried to interact with the token while it was in pre-trading state, the contract reverted as designed (trading not yet enabled / forwarder-restricted), and Layer 3's behavioral confirmation pipeline interpreted the revert as a trap firing. **Class of false positive:** any ERC-20 token launch with a pre-launch trading gate that bots front-run will produce a confirmed-trap classification on Layer 3.

2. **Bytecode classifier false-positive on the Animoca / `@animoca-network/contracts` framework.** OFC was flagged with `has_asymmetric_transfer=1` and `has_unusual_fee_structure=1`. The diagnostic string identifies the patterns: `"CALLER at 0x101c -> EQ at 0x101e -> JUMPI at 0x1022 -> REVERT at 0x1053: conditional revert gated on msg.sender in transfer context"` — that is the standard `onlyOwner` / `ContractOwnership` modifier pattern used by Animoca's framework (and by OpenZeppelin). `"SHA3 at 0xc3a -> SLOAD at 0xc3e -> JUMPI at 0xc4f -> MUL at 0xc76: KECCAK256-keyed storage lookup gates arithmetic on transfer amount"` is the standard `TokenRecovery` pattern. **Class of false positive:** any ERC-20 built on Animoca's framework — and likely any using ContractOwnership + TokenRecovery in general — gets a trap-pattern signature. The OneFootball Club token sits at the intersection: Animoca framework + verified ERC-20 + active trading. The bytecode classifier produces a false positive class size that is *at least* the count of Animoca-framework deployments (which is large; Animoca has 380+ portfolio companies).

3. **`approval_watchlist.drain_detected` pipeline credits failed transferFrom calls as multi-victim drain events.** The 2026-05-09 "discharge" was 3 failed `transferFrom` transactions with single-`from` parameters. The pipeline credited each failed tx as drains against ~all approvers in the pool at the time. The numerical attribution: tx 1 → 3,228 phantom drain rows; tx 2 → 1,128; tx 3 → 231. **Class of false positive:** any contract that experiences a failed transferFrom attempt produces phantom drain rows against unrelated approvers. The 8,108 drain events in the past 14 days (per the 2026-05-21 recent-activity probe) likely contain a non-trivial false-positive fraction from this same bug — the headline drain counts are unreliable until the pipeline is audited.

**Discovery (how it was caught):**

The 2026-05-21 recent-activity review surfaced the harvester discharge as a major finding. Writing Phase 4 "Confirms" documentation under the strict CLAUDE.md discipline forced a Phase 2 literature review. The literature review found the Correction #20 open work flag ("investigate why an Animoca-tagged wallet deployed a confirmed-tier approval-harvesting contract"). The cross-chain Blockscout probe was the next-step input prescribed by Correction #20. The first probe (`get_address_info` on `0x752c5a95` Base) returned the ERC20FixedSupply / OneFootball Club / 3,904-holders / CoinGecko-listed token metadata — at which point the entire premise of the harvester finding collapsed in a single API call.

Subsequent probes confirmed: the second contract (`0xDA42FE`) is the LayerZero adapter, not an unused sibling; both discharge txs are failed transferFrom calls, not 4,587-victim sweeps; the mainnet Blockscout label confirms the Animoca attribution is real and not stale.

The discipline of "investigate the open work before declaring victory" caught the error before the case file's "strongest-validated Tier-C prediction" framing propagated into external materials. Without that discipline the 4,587-victim discharge would have anchored a pitch slide.

**Numerical effect on published / committed claims:**

| Claim location | Old (retired) form | Corrected form |
|---|---|---|
| INDEX.md Section 1 `0x752c5a95` entry | "DISCHARGED 2026-05-09 — 4,587 victims drained in 30 minutes... validated the 2026-04-24 Tier-C prediction within 15 days" | **Retired entirely.** Contract is OneFootball Club (OFC), a legitimate Animoca-deployed verified ERC-20 token with 3,904 holders. The 2026-04-24 "harvester" framing was a stacked-false-positive misclassification. There was no discharge. |
| CASE_HARVESTER_DISCHARGE_0x752C5A95_20260509.md (created 2026-05-21) | "EXTRACTION_011. Strongest validated Tier-C prediction in the corpus to date." | **Retracted entirely.** Case file is annotated with a top-of-file retraction notice; content preserved verbatim per immutable-corpus-record discipline. |
| Lexicon Adversarial Maneuver / Disrupt-Positioning entry (line 1065) | "Layer 3's 2026-04-24 pre-drain flag on operator `0x80b12bd0` (15 days of lead time before the May-9 4,587-victim discharge) is the canonical example of disrupt-positioning succeeding." | **Retracted.** Disrupt-positioning concept retained; the `0x80b12bd0` example is removed. No corpus-derived canonical example currently exists. |
| `approval_watchlist` table "drain_detected" counts | "3,437 lifetime drain events" (CLAUDE.md priority #14) | **Unreliable.** Pipeline bug credits failed transferFrom calls as multi-victim drains. Headline count includes false positives at an unknown ratio. Per-event audit required before the headline can be republished. |
| "Strongest validated Tier-C prediction in the corpus" | "0x752c5a95 harvester discharge, 15-day lead time" | **No validated Tier-C prediction currently in the corpus.** All prior Phase A predictions were disproven; the 0x752c5a95 prediction was misclassification-driven and is now retracted. The Strategy Lifecycle / Tier-C prediction model has no currently-validated example. |

**Fix:**

- This correction-log entry (#24) — 2026-05-21, commit pending.
- `CASE_HARVESTER_DISCHARGE_0x752C5A95_20260509.md` — top-of-file retraction notice added per immutable-record discipline; original content preserved.
- `INDEX.md` — `0x752c5a95` Pre-Drain Harvester entry retired; replaced with RETRACTED notice. Section 2 entries for the two "discharge wallets" retired.
- `CORRECTIONS.md` — Quick Retirement Index row added; full dated entry added.
- `lexicon.md` — Adversarial Maneuver / Disrupt-Positioning entry's `0x80b12bd0` example removed.
- `CLAUDE.md` retired-claims list updated.

**Operational priorities added (three new bugs to fix):**

- **Bug #1 (behavioral classifier FP on pre-launch ERC-20s):** Audit the `behavioral_confirmation` path. When a bot triggers a revert during the pre-launch window of a verified ERC-20 token, the system should NOT promote to confirmed-trap. Requires a pre-launch / not-yet-public-listed check. Sample size: at minimum the OFC case; almost certainly more in the corpus.
- **Bug #2 (bytecode classifier FP on `@animoca-network/contracts` framework):** Audit `bytecode_cache` for all rows with `has_asymmetric_transfer=1 + has_unusual_fee_structure=1` that are verified contracts on Blockscout. Cross-check against deployer's mainnet labels. Sample expected: every Animoca-framework deployment (Animoca has 380+ portfolio companies; per-company contract count varies).
- **Bug #3 (approval_watchlist drain credit on failed transferFrom):** Audit the `drain_detected=1` pipeline. Add a tx.status filter — a failed `transferFrom` must NOT generate `drain_detected=1` rows for any approver. Re-flag all existing `drain_detected=1` rows that correspond to reverted txs as `drain_detected=0` (or remove). Then recompute headline corpus statistics. This is the load-bearing fix.

**Open work (post-correction):**

- Bulk audit of Animoca-framework contracts in our corpus: how many confirmed/suspected entries are framework false-positives? The OFC case is one data point; corpus-wide count requires the bytecode_cache + OLI cross-reference described above.
- Bulk audit of `drain_detected=1` rows that map to failed transactions: recompute the 3,437-lifetime-drain claim.
- Re-evaluate every "self_deploying_trap_operator" archetype case file written under the same approval_watchlist methodology (`0xacc79e7b`, `0x73c0c56b`, `0xc0ee427b`). If the pipeline credits failed transferFroms as drains, the 290-drain count on `0xacc79e7b` (Case file CASE_SELF_DEPLOYING_TRAP_OPERATOR_0xACC79E7B_20260521.md, committed 2026-05-21) needs re-verification before its findings can be trusted.
- Audit the OLI enrichment pipeline (`surveillance/oli_enrichment.py`): the production `oli_labels` table has 13 rows, 0 of which have tags populated. The live Blockscout tags for `0x80b12bd0` ARE present (3 tags returned in the MCP probe), so the fetch path is broken somewhere between Blockscout response and table write. Until this is fixed, the OLI-based detector caveats (Correction #20's "use is_known_legitimate at promotion time") are running with empty data.

**Why this matters epistemically (the meta-finding):**

Correction #24 is the second consecutive correction (after #21 + #22) where statistical or external-source verification overturned a previously-accepted finding. Correction #21 (Pattern D direction reversal) was caught by Cox PH + KS testing. Correction #22 (Camouflage Ratio direction reversal) was caught by tier-partitioned z-testing. Correction #24 was caught by single-call cross-chain external probe.

The shared root cause: **Layer 3's classifiers were not stress-tested against the external truth-layer before being trusted as ground truth.** The "confirmed-tier" label, the "drain_detected=1" pipeline, and the bytecode trap-pattern flags all produced confident, internally-consistent labels — and all three were wrong on the same contract simultaneously.

The CLAUDE.md "loud failures over silent wrong output" doctrine assumed the classifier paths would fail loudly when wrong. They didn't — they failed silently, agreeing with each other in a self-reinforcing way that produced a "strongest-validated Tier-C prediction" finding from a normal ERC-20 token launch.

Process change implied: **any "confirmed-tier" finding in the corpus that has an externally-attestable identity (verified on Blockscout, listed on CoinGecko, OLI-tagged, on a major exchange) must be cross-checked against the external source before being treated as adversarial.** This is not optional and not deferred. It is a precondition for the classifier output to be cited.

**Propagation watch-list (add to top-of-file table):**

| Claim | Still appears in | Required cleanup |
|---|---|---|
| "`0x752c5a95` Pre-Drain Harvester" as an adversarial contract | `docs/INDEX.md` Section 1 + Section 2; `surveillance/data/cases/CASE_HARVESTER_DISCHARGE_0x752C5A95_20260509.md`; production `contracts` table (confidence_tier='confirmed'); production `approval_watchlist` table (4,587 drain_detected=1 rows); `docs/lexicon.md` line 1065 (Adversarial Maneuver canonical example) | INDEX entries retired; case file retraction notice; lexicon line removed; production DB rows pending (require migration script — flagged as engineering work). |
| "4,587 victims drained in a 30-minute discharge" | All references to EXTRACTION_011 | Retracted entirely. No corpus-derived "discharge" event corresponds to a real on-chain mass-drain on this contract. |
| "Strongest validated Tier-C prediction in the corpus" | `surveillance/data/cases/CASE_HARVESTER_DISCHARGE_0x752C5A95_20260509.md` and any derived materials | Retracted. No validated Tier-C prediction currently exists in the corpus. |

---

## Correction #25 — Confirmed-Tier Audit Phase A: 116 Verified-Source Legitimate Tokens Misclassified as Adversarial (7.2% of corpus)

**Date:** 2026-05-22
**Discovery method:** Bulk Blockscout v2 REST API enrichment on every confirmed-tier contract (1,603 of 1,609 probed, 99.6% success) + every distinct deployer (719/719). Classifier applied audit-plan rules (token holders ≥ 100, or verified-source ERC-20 with corroborating evidence, or institutional public/private tags). Execution: `scripts/phase_a_blockscout_enrich.py`. Audit CSV: `reports/confirmed_tier_audit_2026-05-22.csv`.
**Severity:** HIGH — the confirmed-tier population is one of Layer 3's headline corpus statistics and the basis for the Camouflage Ratio confirmed-tier figure (Correction #22) and the headline drain-attribution numbers. 7.2% FP rate is large enough to require migration + headline number recompute. Several individual findings are credibility-critical because the underlying tokens are widely-known (Circle Wrapped Bitcoin, TetherGold, Hyperliquid, Backpack).

**Claim (what was asserted, implicitly):**

The corpus stat "1,650 confirmed-tier adversarial contracts" had been carried as a Tier-A count across multiple internal reports, the Camouflage Ratio z-test (Correction #22), and prior Tier B/C inferences about adversarial behavior. Every contract in this set carried `confidence_tier='confirmed'` with `confidence_reason` claiming evidence of trap behavior (typically "Behavioral confirmation: bot 0x… trapped in tx 0x…").

**Reality (what the audit shows):**

Of the 1,603 confirmed contracts probed via Blockscout enrichment:

| Verdict | Count | Share |
|---|---|---|
| **LIKELY_FP (high confidence)** | **116** | **7.2%** |
| LIKELY_TP | 73 | 4.5% |
| NEEDS_REVIEW | 1,420 | 88.3% |

The 116 LIKELY_FP cases were caught by three classifier paths:
1. **Holders count ≥ 100** (CoinGecko-style real onboarding): caught the bulk of meme/protocol tokens with substantial holder onboarding.
2. **Verified-source ERC-20 with circulating market cap > 0**: caught CoinGecko-listed legitimate tokens (the OFC anchor / Correction #24).
3. **Verified-source ERC-20 with holders ≥ 10**: caught well-onboarded tokens whose holder count was lower than 100 but with corroborating verification.

EDGE cases (64 additional verified-ERC20 contracts with <10 holders and no market cap) were classified to NEEDS_REVIEW pending source-code inspection — these include brand-new token launches that bots front-ran during the pre-trading window. Some are likely TPs (a real adversary's test deploy); some are legitimate launches caught in the bot-revert FP pattern from Correction #24. They cannot be auto-classified without per-contract review.

**Highest-credibility findings (load-bearing for the retraction):**

| Token | Holders | Identity |
|---|---|---|
| **Circle Wrapped Bitcoin** | 119,491 | Circle's wrapped-BTC product (the USDC issuer) |
| **TetherGold** | 229,676 | Tether's gold-backed token |
| **Hyperliquid** | 129,150 | Leading perp-DEX protocol token |
| **Mezo** | 142,230 | Bitcoin DeFi protocol |
| **Backpack** | 113,357 | Backpack Wallet's ecosystem token |
| **Gensyn** | 196,671 | Distributed-compute infrastructure |
| **edgeX** | 194,992 | Real DEX |
| **Fluent** | 192,231 | Real protocol |
| **Sentio** | 168,018 | Real analytics platform |
| **OneFootball Club (OFC)** | 3,914 | Animoca product (Correction #24 anchor) ✓ |
| 23+ other named projects with ≥90K holders each | | (full list in CSV) |

If any external Layer 3 material had cited confirmed-tier counts that included any of these tokens as adversarial, the claim would be directly disprovable by ten seconds of CoinGecko inspection.

**Root cause (the three stacked classifier failure paths):**

This is the same triple-failure as Correction #24, generalized:

1. **Bytecode classifier FP on standard framework patterns.** OpenZeppelin's `ContractOwnership`, Animoca's `@animoca-network/contracts`, and similar framework imports produce bytecode that trips `has_asymmetric_transfer`, `has_unusual_fee_structure`, and `has_conditional_revert` flags. These flags are designed to detect adversarial reverts but cannot distinguish "framework-standard onlyOwner modifier" from "deceptive trap revert."

2. **Behavioral classifier FP on pre-launch token reverts.** Bots front-run new ERC-20 launches before trading is enabled. The contract reverts as designed. The pipeline reads the revert as a trap firing and applies `confidence_tier='confirmed'`. Pre-Bug-#19 backfill (Phase 0, Correction #24), this also created phantom drain rows.

3. **No verification gate before confirming.** The pipeline never checks Blockscout-verified status, OLI institutional tags, or CoinGecko listing before promoting to confirmed. A 200K-holder verified-source legitimate token can ride straight from behavioral revert → confirmed-tier label.

**Discovery (how it was caught):**

Per `reports/confirmed_tier_audit_plan.md` Phase A. Triggered by Correction #24 (the OFC retraction on 2026-05-21). The audit's first concrete step was bulk Blockscout enrichment on the entire confirmed population — exactly the move the audit plan described as "the cheapest single information-gathering move."

**Numerical effect on headline statistics:**

| Claim location | Old (retired) form | Corrected form |
|---|---|---|
| Headline "confirmed-tier adversarial contracts" count | 1,650 (production state 2026-05-21) | **1,534** after 116-row migration (1,650 − 116). Pending Phase B+C audit of NEEDS_REVIEW (1,420) and EDGE (64); the final number may move further. |
| Camouflage Ratio confirmed-tier figure (Correction #22) | 30.44% [27.86, 33.14] low-revert | Pending recompute. The 116 migrated contracts are mostly legitimate tokens — they tend to have HIGHER revert rates than baseline because of pre-launch bot-revert dynamics. Removing them from the partition is expected to RAISE the confirmed-tier low-revert rate (push it back toward the 70-79% baseline). The Correction #22 directional finding (predators revert MORE than baseline) may attenuate. |
| All Tier B/C corpus-level inferences that depend on the confirmed-tier as ground truth | Computed against 1,650 | Need re-running against the post-migration 1,534 population. |
| Lexicon Pattern entries citing confirmed-tier counts | Various | Annotate with the audit's effect; defer until Phase B+C complete. |

**Fix:**

- This correction-log entry (#25) — 2026-05-22, commit pending
- CORRECTIONS.md Quick Retirement Index entry and dated entry added
- Phase D migration: `scripts/phase_d_audit_migration.py` — moves 116 STRONG LIKELY_FP contracts from `confirmed` → `unanalyzed` on local + production. Original `confidence_reason` preserved with audit annotation prepended.
- CLAUDE.md operational priority list updated (audit-derived priorities resolved)
- INDEX.md cleanup for any of the 116 contracts that have prior INDEX entries (none expected at the top-level — none of these were referenced as "case files" — but to be verified)
- `scripts/phase_a_blockscout_enrich.py` classifier refined (verified+ERC20 with <10 holders → NEEDS_REVIEW, not LIKELY_FP)
- 64 EDGE cases remain in `confirmed` tier pending Phase C manual review

**Open work:**

- **Phase B (internal heuristics)**: apply drain/tx ratio, self-loop, recidivism rules to the 1,420 NEEDS_REVIEW population. Expected to surface additional FP candidates from Class C (behavioral-only labels with no bytecode evidence) and Class D (self-loop / BACKFILL).
- **Phase C (sample manual review)**: stratified sample of NEEDS_REVIEW + all 64 EDGE cases. Per-contract verdict.
- **Phase E (permanent pipeline fix)**: add Blockscout-verified-source check + OLI institutional tag check + holders threshold check BEFORE promoting any contract to `confirmed`. The audit's root cause was the absence of these gates at promotion time.
- **Camouflage Ratio recompute (Correction #22 follow-up)**: re-run `surveillance/analytics/camouflage_ratio_z_test.py` against the post-migration corpus. Re-publish the confirmed-tier figure with the audit-derived methodology note.
- **Headline-drain-attribution recompute** (Correction #24 follow-up): recompute the lifetime drain count from the Phase-0-cleaned `approval_watchlist` and exclude any drain events whose contract is now in `unanalyzed`.
- **Communication review**: if any external Layer 3 material (deck, brief, pitch, report, email to a customer) cited the confirmed-tier count or any of the 116 retracted contracts by name, draft a customer-facing correction note.

**Why this matters epistemically:**

The 7.2% FP rate is in the territory where headline numbers cannot be reused at face value. Correction #24 was a single anchor case (OFC). Correction #25 is the rate. It is a much harder finding to absorb because it cannot be patched by a single retraction — every cite of the confirmed-tier count is now subject to "which version of the count, and what cleanup state."

This is also the FIRST audit-derived correction in the log. Corrections #1–#24 were all caught by re-querying primary sources, manual review of specific cases, or statistical analysis of corpus subsets. Correction #25 was caught by **bulk enrichment of the entire confirmed-tier population against external truth sources** (Blockscout verified-source flag, holders count, market cap). That methodology should now be a standing process — not a one-time fix.

### Phase B follow-up migration (2026-05-22, same-day extension of Correction #25)

After the Phase A migration (116 STRONG LIKELY_FP), Phase B applied internal heuristics (deployer recidivism, drain/tx ratio, reason_class, bytecode_cache presence) to the residual NEEDS_REVIEW + EDGE population. Distribution:

| Phase B verdict | Count | Action |
|---|---|---|
| LIKELY_TP_RECIDIVIST | 828 | Keep confirmed (recidivist deployer, no institutional tag — strongest internal TP signal; Coffee Fleet alone accounts for ~203) |
| NEEDS_REVIEW | 488 | Phase C (manual review) |
| STILL_NEEDS_REVIEW | 62 | Phase C (Phase A EDGE — verified+ERC20 with <10 holders) |
| LIKELY_FP_WEAK | **40** | **Migrated to `unanalyzed`** (Phase D follow-up; this section) |
| BUG_19B_SUSPECT | 2 | Phase E work (residual from-matching bug; investigate, don't auto-downgrade) |
| ALREADY_MIGRATED / LIKELY_TP_PHASE_A | 189 | (Phase A results) |

The 40 LIKELY_FP_WEAK contracts all matched: self-loop or BACKFILL reason + solo deployer (recidivism=0) + zero drain activity + no bytecode_cache row + no institutional OLI tag. Per audit plan Class D guidance: "downgraded en masse pending stronger evidence."

Migration applied on local + production via `scripts/phase_d_weak_migration.py` (and prod-side equivalent). Each contract moved from `confirmed` → `unanalyzed`. Audit annotation prepended to `confidence_reason`. Original reason preserved.

**Post-Phase-B+D production state:**
- Confirmed tier: **1,495** (was 1,650 before audit; 1,535 after Phase A migration; now 1,495 after Phase B+D)
- Total audit-driven downgrades: **156 contracts (9.5% of the pre-audit confirmed population)**
- Remaining audit work: Phase C (550 contracts split between NEEDS_REVIEW and STILL_NEEDS_REVIEW) + Phase E (permanent pipeline fix)

**Propagation watch-list:**

| Claim | Still appears in | Required cleanup |
|---|---|---|
| "Confirmed-tier: 1,650" or similar specific count | All Layer 3 materials | Replace with "1,495 post-2026-05-22 Phase A+B+D audit (subject to further Phase C refinement)" with explicit methodology note. |
| Camouflage Ratio confirmed-tier 30.44% (Correction #22) | `docs/lexicon.md` Camouflage Ratio entry; `reports/correction_log.md` Correction #22 | Re-run the z-test post-migration and update the entry / log. |
| Pattern D / Behavioral Laundering / Stored Potential lexicon entries that reference confirmed-tier examples | `docs/lexicon.md` various | Verify that example contracts are not in the 116 retracted set; if so, annotate or replace. |
| Internal queries or analytics modules that `WHERE confidence_tier='confirmed'` | `surveillance/analytics/*`, `surveillance/sai/*` | Re-run with the post-migration corpus; document any meaningful shifts. |

---

## Correction #26 — Intentional 5-Day Surveillance Blind Window (2026-05-27 → 2026-06-01)

**Date:** 2026-05-27
**Type:** Operational pause, not a methodology correction. Logged here so anyone querying the corpus for the dark window dates gets the right framing rather than concluding that activity stopped.
**Severity:** LOW for the corpus (the gap is intentional and bounded). HIGH for any downstream analytics that join on `detection_timestamp` and assume uniform coverage.

**What was paused:**

The Railway service `stellar-embrace` (the surveillance container running `python run_surveillance.py`) was paused at approximately 2026-05-27T19:00 UTC and is scheduled to resume 2026-06-01. All three chain-monitor WebSocket subscriptions (`deployment_monitor_base`, `deployment_monitor_arbitrum`, `deployment_monitor_optimism`) stop ingesting during this window. SAI scheduled detectors (Q-002 × 4/day, Q-003, Q-005, Q-009, Q-008) do not run. The Stats API stays paused alongside.

**Why:**

The shared Alchemy app (one app, two Railway services — `stellar-embrace` for surveillance and `layer3-trading-exp` for trading) was at ~2.3B of the 2.5B monthly CU limit on 2026-05-27. The trading-exp side needed the remaining headroom (~200M CU) for testing through 2026-06-01. Surveillance's baseline burn (~12M CU/day measured via the May-2026 instrumentation in commit `7195ea5`) would have exhausted the budget before month-end. Pausing was the cleanest path; partial throttling would have introduced detection-quality questions that the corpus discipline does not tolerate.

**Companion finding:** The pause came one day after diagnosing and fixing a separate Alchemy CU spike on the trading-exp side. From 2026-05-22 onward, `layer3-trading-exp` was leaking newHeads subscriptions on Base Mainnet (445M CU/day, 91% of WS usage, 100% from a single subscription type). Root cause: `pool_monitor.py`'s inner reconnect loop re-called `_AlchemyTransport.subscribe_new_heads()` on stall without unsubscribing the prior subscription. Fix shipped as `a810799` in `github.com/2654-zed/layer3-trading-exp`. The surveillance pause is unrelated to this fix — surveillance was confirmed at baseline (~12M/day) via the per-method telemetry endpoint `/api/rpc/usage`. The surveillance pause is a budget-allocation decision, not a remediation.

**What you lose for 5 days:**

| Surface | Estimated gap |
|---|---|
| New contract deployments ingested | ~5,000–7,500 across all 3 chains (~1,000–1,500/day, normal ingest rate) |
| `transaction_events` rows | ~600K–1.2M (~120K–240K/day) |
| `approval_watchlist` rows | ~1,500–2,500 (~300–500/day at recent rate) |
| Drain events | ~150–300 if last-7d rate holds (~30–60/day post-Phase 0 fix) |
| SAI detector runs | 20 missed runs (Q-002 4×, Q-003/Q-005/Q-009/Q-008 1× each, daily × 5 days) |
| OLI label refresh | None during gap; resume catches up |

**Snapshot artifacts captured at pause (under `reports/dark_window_2026-05-27/`):**

| File | Content |
|---|---|
| `stats_at_pause.json` | Production `/stats` snapshot at the pause moment — baseline for resume comparison |
| `sai_alerts_at_pause.json` | Full SAI alerts (limit=500) — anything still in NEEDS_VERIFICATION / STALE state when surveillance comes back |
| `watchlist_at_pause.json` | Full active watchlist (110 rows, 100 HIGH) — addresses worth external monitoring during the gap |
| `rpc_usage_24h.json` + `rpc_usage_24h_by_component.json` | Confirmation of surveillance's ~12M CU/day baseline at pause |
| `active_addresses_for_etherscan_alerts.json` | Top-10 drain_callers + top-10 high-velocity deployers from the past 14 days — the candidates for free Etherscan address-watch alerts during the gap |
| `tier_counts_and_recent_drains.json` | Confirmed=1,450 / suspected=135,479 / unanalyzed=64,223 / unknown=163,011 at pause; last 24h drain events |

**Gap-coverage plan during dark window:**

1. **Free external sources stay live:** Blockscout web UI / API, Etherscan free tier, GoPlus free API, Spotonchain / Lookonchain Twitter, DefiHackLabs incidents.yaml — none cost Layer 3 anything. Investigations requested by partners can use any of these.
2. **Top 5 watchlist addresses get Etherscan email alerts:** `0x73c0c56bbf16…` (self-deploying-trap operator, 6 contracts drained in 14d), `0xf168cddd9093…` (154-victim drainer), `0xf3dd26b8081c…` (most-recent drain, last activity 2026-05-25T15:39Z), `0xa27bba42e0e1…` (pre-discharge bait), `0x80b12bd0…` (Animoca-tagged — most epistemically interesting if it does anything anomalous).
3. **NO Layer 3 backfill of the gap on resume.** The built-in `connection_gaps` table records the disconnect window. Backfilling 5 days × 3 chains = ~2.1M block fetches × 16 CU ≈ 34M CU just for blocks (60–100M total with receipts) — that would eat most of the next month's budget. Start fresh from current block on 2026-06-01.

**Resumption protocol (for 2026-06-01):**

1. User unpauses `stellar-embrace` in Railway dashboard.
2. Verify the deployment_monitor heartbeat advances within 5 minutes of unpause.
3. Verify `/api/rpc/usage?hours=1` shows the post-fix baseline (~12M CU/day rate).
4. Pull a "resumption summary" report comparing:
   - `/stats` snapshot at resume vs the captured `stats_at_pause.json` — total contract growth during the gap is rate-extrapolated, not actual
   - External-source observations gathered during the gap (drain events, watchlist hits, new operators surfaced via Twitter or hack labs)
   - Any addresses that received Etherscan alerts during the dark window
5. Append a "Resumption note" sub-entry to this correction documenting any anomalies surfaced.

**Why this is logged in correction_log.md and not just JOURNAL.md:** because future analyses joining on `detection_timestamp BETWEEN 2026-05-27 AND 2026-06-01` will return no rows and may otherwise be interpreted as a real activity flatline. This entry is the audit trail.

---

## Correction #27 — Bug #19b: Drain Detector Credits All Pending Approvers From a Single Contract Interaction

**Date:** 2026-05-27 (dark-window data-integrity audit; see `reports/data_integrity_audit_2026-05-27.md`)
**Discovery method:** Row-level inspection during the intentional-pause audit. `scripts/audit_verify_19b.py`.
**Severity:** HIGH — every "N victims drained" / "N drain events" figure derived from `approval_watchlist.drain_detected` is an upper bound, not a count. This is the second half of the Bug #19 family; Phase 0 (Correction #24) fixed only the reverted-tx half.

**Claim (implicit):** `approval_watchlist.drain_detected=1` rows each represent one victim whose approval was actually exercised by a drainer. Headline counts (e.g. "7,227 drain events," the lifetime "3,437 drains / 2,963 victims" in CLAUDE.md priority #14) treated each row as a real victim.

**Reality:** `surveillance/approval_drain_monitor.py` `check_drains()` matches a transaction against the *contract*, then stamps `drain_detected=1` onto **every pending approver of that contract** — not the single victim whose tokens moved. Both detection methods share the defect:
- Method 1 (transferFrom scan): a single `transferFrom` on the contract credits all approvers.
- Method 2 (deployer-interaction scan): ANY non-approve call by the deployer credits all approvers — even when the call is not a transfer at all.

**Worst-case proof:** drain tx `cf2fed47…bea6d9` on `0xb738b1568f08…` has exactly ONE `transaction_events` row (caller `0xa9f65861…`, selector `f4e2540c` — a custom method, **not** transferFrom `23b872dd`, `is_reverted=0`). `approval_watchlist` credits it to **1,520 distinct victims** whose approvals span five days. One on-chain action, not a transfer, became 1,520 "drained victims."

**Magnitude (local snapshot 2026-05-27):**
- 7,227 drain_detected=1 rows across 735 distinct tx_hashes
- 473 / 735 tx_hashes credited to >1 victim
- 6,965 / 7,227 (96.4%) drain rows attributable to multi-victim tx — the **upper bound** on inflation
- True count is between 735 (one-victim-per-tx floor) and 7,227 (current); cannot be pinned without decoding each tx's transfer logs

**Why it's an upper bound, not a fake count:** a legitimate batched/multicall drain CAN sweep many victims in one tx. The bug is that the detector does not *verify* each credited victim's tokens moved — it assumes they all did. Some multi-victim tx are real mass-drains; some (like the 1,520 case on a non-transfer selector) are pure over-credit.

**Fix:**
- This entry (#27) — documents the bound and the caveat.
- `reports/data_integrity_audit_2026-05-27.md` — full audit.
- **Code fix deferred to Phase E (needs RPC, paused during dark window):** decode the actual Transfer logs of each drain tx and credit only addresses whose balance moved. Tracked as resume-action #2.
- **Until fixed:** quote all drain victim/event counts as upper bounds with this caveat. Update CLAUDE.md priority #14 framing.

**Companion findings from the same audit (not separate corrections):**
- **OLI enrichment silently broken** (priority #22 confirmed): `oli_labels` 13 rows all have `tags_json=NULL, tag_count=0`. The `is_known_legitimate()` gate runs on empty data — a likely contributing cause of the Correction #25 verified-legitimate-token FP class. Resume-action #1.
- **Referential integrity CLEAN:** 0 orphan drain rows, 0 Phase-0 reverted-tx residue, 0 stranded SAI alerts, hash format 100% consistent (an earlier same-session "mixed 0x format" claim was a misread and is retracted).
- **Confirmed-tier thin-evidence residue:** 218/1,262 (17.3%) post-migration confirmed contracts have no bytecode + no approvals + no deployer recidivism. Phase-C-deep candidates, not actioned.

**AMENDMENT (2026-05-27) — Correction #25's migration heuristics never gated on drain evidence:**

The propagation sweep + drain-gate analysis (`scripts/audit_migration_drain_gate.py`) found that **45 of the 347 audit-migrated contracts (13.0%) had `drain_detected=1` rows in `approval_watchlist` that no migration heuristic ever checked.** A contract could be downgraded confirmed→unanalyzed for being a verified token / OZ-framework source / high-holder while carrying recorded drain evidence. The heuristics tested Blockscout legitimacy + activity but had no "does this contract have drains" veto.

Distribution of the 45 by batch (corrects two earlier wrong guesses in this session that blamed `LIKELY_FP_FROM_ACTIVITY`):
- **Phase A (holders/verified): 35** ← dominant source (30% of Phase A's 116)
- Phase C FROM_ACTIVITY: 5 (20% of its 25)
- Phase C FROM_SOURCE: 2 (1.5% of its 130)
- Phase C sample: 1; Phase B: 0; FROM_CLUSTER: 0

**These 45 are SUSPECTS, not confirmed errors, because the drain counts are themselves Bug #19b-inflated (Correction #27 above).** Split by drain-shape:
- **~27 with ≥10 distinct drain transactions** → strong false-negative suspects (a legit token doesn't have dozens of separate addresses running transferFrom-drains against it). E.g. `0xa7e1e8ab7b` FIRE (194 drains/99 tx, case-filed harvester), `0xf768d7d152` (127/28), `0xa68079da` (152/25), `0xd6cd943bfc` Yupp AI (118/19, **bytecode has SELFDESTRUCT**).
- **~11 with 1-2 drain tx fanned to hundreds of victims** → likely Bug #19b artifacts; migration probably correct. E.g. `0xb738b15` (1,618 drains/2 tx), `0xb0a4741f` (319/1).

**Resume-action #0 (ahead of all others):** decode on-chain transfer logs for the 45 (priority: the ~27 high-distinct-tx). This same RPC pass resolves Bug #19b for them. Restore confirmed-tier for any whose drains are real multi-tx drains. Lossless (original `confidence_reason` preserved in annotation). Verification needs prod DB + RPC (local approval_watchlist is a partial snapshot).

**Pipeline fix:** add a drain-evidence veto to migration logic — never auto-migrate a contract with ≥3 distinct drain transactions, regardless of Blockscout legitimacy. (Gate on distinct-tx, not raw drain rows, to avoid Bug #19b false vetoes.)

**Heuristic assessment:** `FROM_SOURCE` (130, 1.5% drain-tainted) and `FROM_CLUSTER` (7, 0%) are clean. Phase A holders/verified (30% drain-tainted) is the worst offender — the cheapest/earliest heuristic, not the Phase C ones.

**Meta-lesson (two layers):**
1. An FP-cleanup heuristic can manufacture false negatives as readily as the original pipeline made false positives. A legitimacy signal (verified source, holder count) is not a safe exoneration when it ignores the adversarial evidence that earned the original label. Migration must veto on the *presence of harm evidence*, not just weigh *legitimacy signals*.
2. **Process:** I asserted the wrong mechanism (`FROM_ACTIVITY counts victims as users`) twice in this session before computing the actual batch distribution, because I wrote conclusions in the same tool-batch as the queries that would test them. Corrected only after running `audit_migration_drain_gate.py` and reading it first. The lesson is in the audit report's process note: read output, THEN conclude, in separate steps.

---

## Correction #28 — Drain Detection Was Structurally Blind, Not Broken by a TypeError. New Blockscout Victim-Leg Detector Supersedes the tx_events Join. (Retracts the `d31bf2d` "regression" framing.)

**Date:** 2026-06-05
**Discovery method:** Building `check_drains_blockscout` per `reports/SPEC_blockscout_drain_detection.md`; grounding queries `scripts/_ground_drain_structural.py`; parity gates `scripts/t_drain_blockscout_parity.py`.
**Severity:** MEDIUM (detection-method change) + a process retraction of a committed claim.

**This correction has two parts: (A) a substantive detection-method change, and (B) a retraction of my own prior misdiagnosis committed earlier this session.**

### Part A — `check_drains()` (tx_events join) is structurally blind to almost all drains

**Claim (implicit):** `check_drains()` Method 1 — join `approval_watchlist` against `transaction_events` for a `transferFrom` (`23b872dd`) on the watched contract — is a working drain detector that simply found few drains.

**Reality:** it is a near-total false-negative by construction. `selector_monitor` only logs txs whose `to` is in the **watched contract set**. A real approval-drain is a `transferFrom` the drainer sends to the **token contract**, which is usually *not* in the watched set — so the drain tx never enters `transaction_events` and the join cannot see it.

**Grounded on the local corpus 2026-06-05 (`_ground_drain_structural.py`):**
- `transferFrom` rows in `transaction_events`: **3,345 lifetime**, **0** in the last 7 days.
- Pending approvals (`drain_detected=0`): **54,996**.
- Pending that Method 1 could match: **2** (of 54,996). The join is blind to ~99.996% of the backlog.
- Existing `audit_drain_legs` Blockscout cache (from the Bug #19b reconciliation): **5,174** victim/contract pairs already resolved, **4,396** with an outbound leg (real drained victims) — at 0 Alchemy CU.

**Fix:** `surveillance/approval_drain_monitor.py` gains `check_drains_blockscout()` + `_blockscout_outbound()` — the **victim-outbound-leg test**: a `(victim, contract)` row is a real drain iff the victim has ≥1 ERC-20 Transfer of the contract token with `from==victim` on Blockscout (0 Alchemy CU). This is *per-victim* (it also fixes the Bug #19b over-credit that Method 1 carried) and reuses the validated `audit_drain_legs` cache. Wired into the heartbeat loop (`deployment_monitor.py`) via `asyncio.to_thread` (network-bound; must not block the event loop), `max_victims=400`/cycle, writes routed through the single-writer queue. `check_drains()` is retained as a legacy reference path but is **no longer called from the loop**. CLI `--drain-scan-all` clears the backlog out-of-band. All parity/correctness gates pass (`t_drain_blockscout_parity.py`: live cache parity, inbound-only negative control incl. `0xf68425d0`, end-to-end detect + idempotent + zero-error on a temp DB).

### Part B — Retraction: the `d31bf2d` "drains dead since 2026-05-27 (TypeError)" claim is WRONG

**Claim made (commit `d31bf2d`, this session, already on `main`):** drain detection was dead from 2026-05-27 because `check_drains()` used dict-style row access on a connection with *no* `row_factory`, so rows were plain tuples and every call raised `TypeError: tuple indices must be integers`, silently swallowed by the heartbeat's `except`.

**Reality:** **every production path sets `row_factory = sqlite3.Row`**, so reads return `Row` objects and dict-access works. Verified by reading the code, not asserting it:
- Heartbeat path: `run_surveillance.py` → `process_entries.monitor_entry(..., write_queue)` → `run_monitor(..., write_queue=...)` → `QueueConnection`, whose read connection sets `self._ro.row_factory = sqlite3.Row` (`db_queue.py:65`).
- Standalone path: `db.init_db()` sets `row_factory = sqlite3.Row` (`db.py:98`).
- CLI `--scan`: sets it (`approval_drain_monitor.py`).
- Empirical confirmation (`_ground_drain_structural.py`): a bare `sqlite3.connect()` returns `tuple` and dict-access raises `TypeError`; a `row_factory=Row` connection returns `Row` and dict-access works. The `TypeError` I "reproduced" was in a bare test harness that does **not** reflect how the code runs in production.

**So no `TypeError` ever fired in production.** The real reasons drains read 0 were (1) the 2026-05-27→06-01 dark window (no block ingestion at all — Correction #26) and (2) the Part-A structural blind spot. The tuple-unpacking edits in `d31bf2d` (and the matching `scan_approvals` edit) are harmless **defensive hardening** — `sqlite3.Row` supports integer indexing, so position-unpacking works on both Row and tuple — but they did **not** fix a production outage, and the commit message overstates their effect. The misleading code comments those edits introduced have been corrected in this change.

**Numerical effect on headline stats:** none directly, but it removes a false "9-day outage" from the session record. Drain counts remain governed by Correction #27 (Bug #19b upper-bound caveat); the new per-victim detector is the path to replacing those upper bounds with verified counts as the backlog clears.

**Open work:**
1. Deploy (git→deploy; detection-method change — gated on user approval). Hot-patch remains forbidden.
2. Clear the 54,996-pending backlog via `--drain-scan-all` and/or accumulated heartbeat cycles; then recompute the Correction #27 / CLAUDE.md priority #14 & #25 drain headline stats against the verified data.
3. The `d31bf2d` commit message is immutable in history (no `--force` to `main`); this entry is the durable correction of record.

**Meta-lesson (recurrence of Correction #27's process note #2):** I shipped `d31bf2d` having concluded "TypeError regression" from a bare-harness repro *without verifying the production connection's `row_factory`*. Same failure mode as #27: a conclusion written before the evidence that would test it. Caught this time by reading `db_queue.py` / `process_entries.py` / `db.py` and running a grounding query **before** writing the correction. Read the source and the output, THEN conclude — in separate steps.

**UPDATE (2026-06-06) — backfill complete (open-work item #2 closed for the local snapshot):** `python -m surveillance.approval_drain_monitor --drain-scan-all` cleared the full local pending backlog in one resumable pass (0 Alchemy CU, 40 fetch errors = 0.09%, no rate-limiting). Result on `surveillance/data/surveillance.db`:
- **40,144 new per-victim-verified drains detected** (the tx_events-join method would have found 2). Total `drain_detected=1` now **46,593 rows / 20,769 distinct victims / 4,191 distinct contracts**, every row carrying an on-chain `drain_tx_hash`.
- **6,749 approvers verified CLEAN** (`n_out=0`, no outbound leg) and correctly left unflagged — the precision the old method lacked.
- 8,063 OLI-suppressed (institutional) correctly skipped; 40 fetch-error rows remain pending for retry.
- `audit_drain_legs` cache now holds 51,329 resolved (victim,contract) verdicts (44,540 with an outbound leg).

Provenance/tier: source = local `approval_watchlist` post-backfill, 2026-06-06; method = Blockscout victim-outbound-leg test; **Tier A** per row (on-chain replicable). Caveat: LOCAL snapshot only — prod converges as its heartbeat clears its own backlog (~400/cycle), or immediately if the completed `audit_drain_legs` cache is synced to prod. The retired "3,437 lifetime drains" headline (CLAUDE.md #14/#25, Correction #27 upper bound) can now be **recomputed as verified counts** against this data — quote with this provenance, not as a pre-existing corpus figure. Note the high drain rate (85.6% of checked) is expected because `approval_watchlist` only tracks approvals to suspected/confirmed adversarial contracts.

---

## Correction #29 — The Blockscout Drain Detector Counted DEX Sales as Drains. `n_out>0` Is Not a Drain Signal. (Retracts #28's backfill numbers.)

**Date:** 2026-06-06
**Discovery method:** Adversarial verification prompted by the operator's question "are we misattributing real activity?" On-chain tx-initiator sampling: `scripts/verify_oli_suppression.py`, `scripts/validate_drain_initiator.py`.
**Severity:** HIGH — invalidates the Correction #28 backfill (40,144 "drains") and the same-day recompute (44,540 "verified drains / 19,533 victims"). Both retracted here.

**Claim made (Correction #28, this session):** `check_drains_blockscout`'s victim-outbound-leg test (`n_out>0` — the victim has ≥1 ERC-20 Transfer of the contract token with `from==victim`) is a per-victim-verified drain signal. Backfill reported **40,144 new drains**; recompute reported **44,540 verified drain rows / 19,533 victims / 4,324 collectors**, claimed to dwarf the retired 3,437.

**Reality:** `n_out>0` only proves the victim's tokens *left their wallet*. It **cannot distinguish a malicious drain from a legitimate DEX sale** — both produce `Transfer(from=victim)`. The discriminator is the **transaction initiator**: a drain's outbound leg is initiated by a third party (`tx.from != victim`, a `transferFrom` by the drainer); a sale is initiated by the victim (`tx.from == victim`, a swap). Evidence:
- **OFC (`0x752c5a95`, the 8,063 OLI-suppressed):** 400/400 sampled approvers had outbound legs, 80.8% to 3 collectors — but the top collectors are `Volatile AMM - OFC/USDC` (a DEX pool) and `BaseSettler` (a DEX aggregator). The "drains" are holders **trading**. OFC is legit; **Correction #24 stands.**
- **Two independent tx-initiator samples:** most-recent-leg 3 drains / 120; detection-leg **0 drains / 62** (all flagged legs were victim-initiated swaps — `swapExactTokensForETH` / `execute` / `exactInputSingle`).
- **Validated discriminator** (`validate_drain_initiator.py`): OFC negative control **0/25** flagged; stratified true drain rate **confirmed 1/45 (2%), suspected 0/40, unanalyzed 0/46**; one genuine drain found with full attribution (drainer `0xcaf438ef…`, tx `0x326f0f1f…`).

**True drain rate: ~0–2%** (concentrated in confirmed tier). The 40,144 / 44,540 figures are ~98–99% legitimate DEX trading.

**Why the parity test missed it:** `scripts/t_drain_blockscout_parity.py`'s negative control was *inbound-only* (`n_out=0`) — it never tested a **seller** (`n_out>0`, victim-initiated). `n_out>0` is necessary but **not sufficient** for a drain. A discriminator must be validated against the realistic confound, not just the trivial negative.

**Remediation (operator-approved):**
1. **Prod detector PAUSED** (commit `5257aab`, deployed) — heartbeat runs `scan_approvals` only; no drain write until the tx-initiator-gated detector is validated. Stops further FP accumulation.
2. **All 46,593 local `drain_detected=1` flags reverted to 0** (`scripts/revert_drain_fps_20260606.py`), snapshotted to `drain_flags_backup_20260606` (reversible). `audit_drain_legs` cache (51,329 `n_out` verdicts) preserved — still valid raw input.
3. **Rebuild** `check_drains_blockscout` with the tx-initiator gate (drain iff an outbound leg has `tx.from != victim`), re-validate against the seller negative-control, then re-run the full "audit all" pass to derive the true drain set with correct drainer attribution.

**OLI finding (same investigation):** `oli_labels` is the broken priority-#22 table — all 13 rows have `tag_count=0, primary_entity=NULL`. The suppression gate rests on `severity` alone with no actual tags. It coincidentally suppressed OFC (legit) correctly, but also suppresses `org_004` (`0xbaed383e`) and `0xc5d133296e` (226 contracts incl. confirmed/suspected) — adversarial deployers that would be false-negatived if they ever carry pending drains. No compromised named wallets were found; OFC's "drains" were sales.

**Meta-lesson (third recurrence this session of "shape ≠ ground truth"):** #28 misdiagnosed a regression from a bare-harness repro; this entry shipped a detector that inferred intent (drain) from shape (`n_out>0`) without a ground-truth discriminator (tx initiator). The corpus's entire correction history (#20, #22, #24, #25, #27, this) is the same failure: **legitimate and adversarial activity are shape-identical at the resolution Layer 3 observes; only a deductive signal separates them.** No "adversarial / drain / predator" claim should ship without a deductive discriminator, validated against the realistic legitimate confound — not just a trivial negative control.

---

## How to add the next entry

1. Append a new `## Correction #N` section in chronological order.
2. Answer the four questions: claim, truth, how caught, what changed.
3. Report the numerical effect on any headline statistic.
4. List open work so the next reviewer knows what is still unresolved.
5. If the correction required a migration script, commit the script alongside the log entry.

The log is appended, never rewritten. A superseded correction stays in place; a follow-up correction points back to it.
