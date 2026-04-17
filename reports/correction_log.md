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

### Open work
- **Backfill not yet executed on Railway.** Will run `python -m surveillance.backfill_cache_invalidation --commit` against the production volume after this entry is committed to the repo. Residual stale count after backfill should be zero; script verifies and reports.
- **Orphan cache rows** (source in `bytecode_cache` whose address is missing from `contracts`) are not touched by this fix. Measured at 2,310 cache-sourced contracts whose reason cites a source prefix no longer in the cache. Separate cleanup — not a staleness issue, a pruning artifact.
- **Cache-staleness class pattern search** (Wave 1) will enumerate other derived-data tables whose invalidation is not guaranteed when source rows mutate (e.g., `deployer_similarity`, `bytecode_families`, `risk_scores`, `entity_classification`). Report-only pass scheduled; not remediation.

---

## How to add the next entry

1. Append a new `## Correction #N` section in chronological order.
2. Answer the four questions: claim, truth, how caught, what changed.
3. Report the numerical effect on any headline statistic.
4. List open work so the next reviewer knows what is still unresolved.
5. If the correction required a migration script, commit the script alongside the log entry.

The log is appended, never rewritten. A superseded correction stays in place; a follow-up correction points back to it.
