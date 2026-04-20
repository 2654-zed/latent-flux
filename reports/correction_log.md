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

---

## How to add the next entry

1. Append a new `## Correction #N` section in chronological order.
2. Answer the four questions: claim, truth, how caught, what changed.
3. Report the numerical effect on any headline statistic.
4. List open work so the next reviewer knows what is still unresolved.
5. If the correction required a migration script, commit the script alongside the log entry.

The log is appended, never rewritten. A superseded correction stays in place; a follow-up correction points back to it.
