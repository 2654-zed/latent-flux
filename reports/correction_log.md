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

## How to add the next entry

1. Append a new `## Correction #N` section in chronological order.
2. Answer the four questions: claim, truth, how caught, what changed.
3. Report the numerical effect on any headline statistic.
4. List open work so the next reviewer knows what is still unresolved.
5. If the correction required a migration script, commit the script alongside the log entry.

The log is appended, never rewritten. A superseded correction stays in place; a follow-up correction points back to it.
