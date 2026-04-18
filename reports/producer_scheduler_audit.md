# Producer Scheduler Audit — Wave 2

**Date:** 2026-04-17
**Scope:** Five Class B derived tables flagged in Wave 1 as stale for 22+ days. Explicitly excludes `risk_scores` (confirmed live-compute per Correction #6).
**Tables audited:** `trust_amplification`, `camouflage_metrics`, `bytecode_families`, `deployer_similarity`, `daily_metrics` (+ paired `predictions`, `bytecode_family_members`, `deployer_profiles`).
**Output:** Findings + architecture recommendation. No code changes.

---

## Phase 1 — Producer identity, trigger, last run, why stopped

### Table: `trust_amplification`
- **Producer:** `surveillance/trust_amplification.py::analyze()` (INSERT at line 271).
- **Triggers found:**
  1. CLI: `python -m surveillance.trust_amplification --analyze` — manual.
  2. In-process: `surveillance/deployment_monitor.py:432–445` calls `analyze(..., emit_alerts=True, quiet=True)` every `heartbeat_count % 120 == 100` → **every ~2 hours** for each of the three chain monitors.
- **Last wrote:** 2026-03-25 16:17:27 → 16:17:28 UTC. **All 32 rows landed in a single ~1-second window.**
- **Inference:** The single-second burst is the signature of a one-shot manual CLI run, not the scheduled heartbeat path. If the heartbeat-embedded call were working, we would see writes every 2 hours across 22+ days — at least hundreds of rows spread across dates. We see 32 rows in one second, then silence. The heartbeat path is either silently failing (exception caught at `logger.debug` level on line 444, invisible at default INFO) or never actually executing at all.
- **Why stopped:** Manual CLI not rerun since 2026-03-25. Suspected secondary failure: in-process heartbeat path broken — likely the same silent-IntegrityError class as x402_events (see diagnostic tracer queued for deploy). Requires Railway logs to confirm.

### Table: `camouflage_metrics`
- **Producer:** `surveillance/camouflage_tracker.py` (INSERT OR REPLACE at line 81).
- **Triggers found:**
  1. CLI flags: `--compute-today`, `--compute-all`. Manual invocation only.
  2. **No programmatic integration** — not imported anywhere outside its own module. No cron wire-up in `run_surveillance.py` or `Procfile`.
- **Last wrote:** 2026-03-25 (chain='ethereum' rows only — corpus-wide aggregate, not per-L2 breakdown).
- **Row history:** 9 rows, dates `2026-03-17 → 2026-03-25` (9 consecutive days, then silence).
- **Inference:** An external daily cron populated this table for 9 days, then stopped on the night of 2026-03-25 → 2026-03-26. The scheduler that drove it is not in this repo.
- **Why stopped:** External cron died; no visibility into what it was (user's laptop cron, a removed Railway cron job, a now-defunct systemd timer — unknown from code).

### Table: `bytecode_families` + `bytecode_family_members`
- **Producer:** `surveillance/bytecode_families.py` (INSERT OR REPLACE at lines 196, 207).
- **Triggers found:**
  1. CLI flag: `--cluster`. Manual only.
  2. **No programmatic integration.** Not imported by any monitor.
- **Last wrote:** 2026-03-24 23:10:37 UTC.
- **Row history:** 405 rows spanning 2026-03-17 → 2026-03-24 (7 consecutive days of incremental daily additions).
- **Inference:** Daily cron pattern, same shape as camouflage / daily_metrics but stopped one day earlier.
- **Why stopped:** Same class — external cron died. Also note Correction #3 (2026-04-16) dissolved the `T2-eaef6a5d` family after the last run, so the code has changed since the last write; next run will clean the all-zero bucket but the base mechanism works.

### Table: `deployer_similarity` + `deployer_profiles`
- **Producer:** `surveillance/deployer_profiler.py` (INSERT OR REPLACE at lines 381 + 552).
- **Triggers found:**
  1. CLI flags: `--profile-all` (writes deployer_profiles), `--cluster` (writes deployer_similarity), `--find-similar` (read-only), `--top` (read-only).
  2. **No programmatic integration.**
- **Last wrote:** 2026-03-26 15:37:27 → 15:37:34 UTC. **All 4,879 similarity rows landed in a ~7-second window.**
- **Inference:** One-shot manual run on 2026-03-26. Not a daily cron — the single-batch signature is clear.
- **Why stopped:** Producer was never scheduled; only ever run manually once.

### Table: `daily_metrics` + `predictions`
- **Producer:** `surveillance/trend_forecaster.py` (INSERT OR REPLACE at lines 79, 147, 162, 176).
- **Triggers found:**
  1. CLI flags: `--compute-today`, `--backfill`, `--forecast`, `--score`. Manual only.
  2. **No programmatic integration** — NOT invoked by `run_surveillance.py`'s daily_report scheduler at line 1516. That scheduler calls `daily_report.generate_report()`, a *different* module that does NOT write to `daily_metrics`.
- **Last wrote:** 2026-03-25 (chain='ethereum' rows).
- **Row history:** 9 rows, dates 2026-03-17 → 2026-03-25, paired with camouflage_metrics.
- **Inference:** Same daily cron as camouflage_metrics. Stopped 2026-03-26.
- **Why stopped:** Same class — external cron died.

### Summary table — Phase 1

| Table | Producer | Trigger type | Last wrote | Row pattern | Probable cause of staleness |
|---|---|---|---|---|---|
| `trust_amplification` | `trust_amplification.analyze()` | manual CLI + broken in-process heartbeat | 2026-03-25 | 32 rows in 1 sec | Manual run only; heartbeat path silently failing |
| `camouflage_metrics` | `camouflage_tracker.py` | external daily cron | 2026-03-25 | 9 daily rows, dead Mar 26 | External cron died |
| `bytecode_families` | `bytecode_families.py --cluster` | external daily cron | 2026-03-24 | 405 rows over 7 days | External cron died |
| `deployer_similarity` | `deployer_profiler.py --cluster` | manual CLI | 2026-03-26 | 4,879 rows in 7 sec | One-shot manual run; never scheduled |
| `daily_metrics` | `trend_forecaster.py --compute-today` | external daily cron | 2026-03-25 | 9 daily rows, dead Mar 26 | External cron died |

**Common root cause:** None of these five are wired into the deployed `run_surveillance.py` process. The `daily_report_scheduler` at line 1516 IS scheduled in-process (06:03 UTC daily) and runs reliably — proof that the pattern works when used — but only calls `daily_report.generate_report()`, not the five producers. An external cron appears to have driven three of them (`camouflage_metrics`, `bytecode_families`, `daily_metrics`) for 7–9 days in mid-March, then stopped. The other two (`trust_amplification`, `deployer_similarity`) only ever ran manually. **There is no scheduler running any of these producers today.**

---

## Phase 2 — Recommended cadence + scheduling mechanism

Semantic-driven cadence, not arbitrary:

| Table | Recommended cadence | Rationale | Orchestration needed? |
|---|---|---|---|
| `daily_metrics` | **Daily, 00:15 UTC** | Aggregates prior UTC day's ingest. Consumed by `predictions` + briefs. | CLI accepts `--compute-today`; just needs scheduling. |
| `predictions` | **Daily, 00:30 UTC** (after daily_metrics) | Depends on daily_metrics rows; must run after them. | CLI accepts `--forecast --score`; needs dependency ordering. |
| `camouflage_metrics` | **Daily, 00:20 UTC** | Same dependency on full-day transaction_events. Independent of daily_metrics. | CLI accepts `--compute-today`. |
| `bytecode_families` | **Daily, 03:00 UTC** (off-peak) | Clustering is expensive (reads all contract bytecode_signals). Daily cadence is fine — family membership shifts slowly. | CLI accepts `--cluster`. Accepts the full-table REPLACE pattern. |
| `deployer_similarity` | **Weekly, Sunday 04:00 UTC** | O(n²) similarity computation over 25k+ deployers. Weekly is the natural cadence for organizational attribution updates. | CLI accepts `--cluster`. Output is `INSERT OR REPLACE` keyed on (deployer_a, deployer_b) pairs. |
| `deployer_profiles` | **Daily, 02:00 UTC** (before similarity) | Profiles feed into similarity computation. Must run first. | CLI accepts `--profile-all`. |
| `trust_amplification` | **Every 2 hours** (already designed) | High-signal churn as new contracts accrue router traffic. Already in heartbeat loop — fix the silent failure rather than move to cron. | Existing code; needs writer-process diagnosis (x402 tracer will catch it). |

All existing CLIs are standalone and composable — no orchestration wrapping needed beyond a scheduler that invokes them in dependency order.

---

## Phase 3 — Scheduler architecture recommendation

Three options per the brief, plus one lightweight hybrid:

### Option A: Central in-process scheduler module
Extend the existing `_daily_report_scheduler` pattern (`run_surveillance.py:1496–1516`) into a general-purpose scheduler thread that fires multiple jobs at their configured times. Single process, shared connection pool, no additional infrastructure.

**Pros:**
- Zero new infrastructure (no Railway cron jobs to configure).
- Proof-of-concept already running daily at 06:03 UTC — pattern is validated.
- Works identically in local dev and prod.

**Cons:**
- Couples scheduler lifetime to `run_surveillance.py` process. If the monitor crashes, the scheduler stops too.
- Daemon threads can't be cleanly observed; failures log to stdout and may be lost unless captured explicitly.
- No independent retry on producer failure; if `trust_amplification.analyze` hangs, the whole scheduler thread hangs with it.

### Option B: Railway cron jobs, one per producer
Each producer gets its own Railway cron entry invoking the CLI (`python -m surveillance.camouflage_tracker --compute-today`). Railway supports this natively.

**Pros:**
- Failure isolation — one producer dying doesn't affect others.
- Railway dashboard surfaces per-job run history and exit codes.
- Natural scaling — heavy producers can request more resources.

**Cons:**
- Each job starts a fresh Python process → higher cold-start cost (~3–5s each).
- Producer processes open their own DB connection, independent of the single-writer architecture — they become writers. Must verify they use the write queue path, not direct sqlite3.connect, to avoid WAL contention.
- Railway-specific; local dev loses the scheduling surface (still need manual CLI runs for testing).

### Option C: Event-driven (invalidation-on-write)
Trigger recompute every time the source table mutates enough to matter. For example, `deployer_similarity` recomputed when a new deployer passes threshold, `bytecode_families` recomputed when new bytecode classifications accumulate past N.

**Pros:**
- Freshest possible data — derived state always reflects current source.
- No wasted compute on quiet days.

**Cons:**
- Wildly complex to implement correctly for O(n²) producers like `deployer_similarity` — every new deployer triggers full recomputation.
- Unbounded latency if trigger logic debounces poorly.
- Same class of bug we just fixed (cache invalidation, Correction #5) at a producer-level. More surface for mistakes.

### Recommendation: hybrid, weighted toward Option A

**Primary: in-process scheduler (Option A) for daily and weekly producers.** Extend the `_daily_report_scheduler` pattern into a `_analysis_scheduler` that fires at configured UTC times, each call wrapped in a try/except that logs failures at ERROR level (not DEBUG like the current heartbeat-trust-amplification code). Producers run in the same process as the monitors, share the write queue, inherit its failure modes but are observable through the same logs.

**Secondary: fix the in-process heartbeat integration for `trust_amplification`.** Two-hour cadence is correct; the silent-failure is the bug. Once the x402 write-tracer lands on Railway and surfaces the actual IntegrityError, the same fix applies to trust_amplification (both paths swallow exceptions at DEBUG/pass level inside db_writer.run).

**Tertiary: Option B for new, isolated producers that don't yet exist.** When a new producer ships, start with a Railway cron entry — it's observable, isolated, and doesn't require touching `run_surveillance.py`. Only migrate into the in-process scheduler when the producer is stable and frequency warrants it.

**Explicitly reject Option C for this codebase.** The `INSERT OR REPLACE` full-table pattern of every current producer is fundamentally incompatible with incremental/event-driven recomputation. Converting them would be a rewrite, not a scheduling fix.

**Concrete schedule (if Option A adopted):**
```
  00:15 UTC  trend_forecaster.compute_today      → daily_metrics
  00:20 UTC  camouflage_tracker.compute_today    → camouflage_metrics
  00:30 UTC  trend_forecaster.forecast_and_score → predictions
  02:00 UTC  deployer_profiler.profile_all       → deployer_profiles
  03:00 UTC  bytecode_families.cluster           → bytecode_families + _members
  04:00 UTC  (Sunday only) deployer_profiler.cluster → deployer_similarity
  06:03 UTC  daily_report.generate_report        (existing; unchanged)
  every 2h:  trust_amplification.analyze         (existing heartbeat; diagnose silent failure)
```

Total: six new scheduled slots plus the existing daily_report. Cumulative nightly compute window 00:15 → 04:10 UTC = ~4 hours off-peak.

---

## Phase 4 — `computed_at` header recommendations for API responses

Priority ranking by customer-visibility:

| API endpoint | Reads table | Add `computed_at` header | Priority |
|---|---|---|---|
| `/api/v1/contract/{addr}` | `trust_amplification` | YES | **HIGH** — served as Tier B intelligence; current 23-day staleness is misleading |
| `/api/v1/deployer/{addr}` | `deployer_profiles` + `deployer_similarity` | YES | **HIGH** — risk summary component; customers make decisions on this |
| `/api/v1/org/{org_id}` | `entity_classification` | YES (secondary) | **MEDIUM** — entity_classification is 5 days stale on average, better than the rest |
| `/api/v1/ecosystem/stats` | `daily_metrics`, `camouflage_metrics` | YES | **MEDIUM** — aggregate stats; less decision-critical |
| `/api/v1/contract/{addr}` | `bytecode_families` | YES | **MEDIUM** — family attribution is informational in most cases |
| `/api/v1/risk/{chain}/{addr}` | (live compute) | `"live": true` instead of timestamp | **LOW** — already fresh by definition |

**Header shape (proposal):**
```json
{
  "…existing response…",
  "metadata": {
    "epistemic_tag": "assessed",
    "computed_at": {
      "trust_amplification": "2026-03-25T16:17:28Z",
      "deployer_profiles": "2026-03-26T15:37:30Z"
    },
    "freshness_policy": "daily recompute; see /methodology/freshness"
  }
}
```

Customers who care about freshness can filter on the timestamps; customers who don't continue consuming as before. Matches the epistemic-tag discipline already in place.

**`/methodology/freshness` endpoint** (new, recommended): publishes the producer schedule + current actual lag per table. Same pattern as `/methodology/confidence` and `/methodology/camouflage` already exposed.

---

## Follow-up actions if Option A (hybrid) approved

1. Add `_analysis_scheduler` thread to `run_surveillance.py` alongside `_daily_report_scheduler`.
2. Diagnose the in-process `trust_amplification.analyze` silent failure (use the x402 write-tracer pattern — it's the same root cause family).
3. Add `computed_at` fields to the 5 HIGH/MEDIUM endpoints listed in Phase 4.
4. Ship `/methodology/freshness`.
5. Draft Correction #7 documenting: "22+ days of stale derived metrics served by API without freshness indicators — scheduler never ran in production."
6. Run all six producers once to generate fresh rows (restoring current-day correctness before the nightly cron kicks in).

Each step is scoped enough to ship independently and reversibly. None are done as part of this audit.

---

## Out of scope

- `risk_scores` table (confirmed live-compute per Correction #6).
- Root-cause diagnosis of the trust_amplification silent failure (blocked on Railway logs / x402 tracer deployment).
- `behavioral_baselines`, `behavioral_anomalies`, `bot_strategies`, `bait_profiles`, `strategy_lifecycle` — also Class B producer-recompute tables but not in the five-table scope of this audit. Same architecture recommendation would apply.
- Migration of StatsHandler to FastAPI (separate `statshandler_migration_plan.md` scoped).
