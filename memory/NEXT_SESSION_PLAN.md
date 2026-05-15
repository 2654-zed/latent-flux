# Next-Session Plan

**Authored:** 2026-05-13 (end of session that landed commit `b217c41` — regime monitor live + first flux_manifold consumer)
**Read this file at session start.** Then check `memory/STATE.md` and the latest entry of `memory/JOURNAL.md` for any state changes.

---

## Goal

Investigate the 3 candidates surfaced by the regime monitor's first scan against the production corpus. Convert algorithmic alerts into either confirmed findings (case files, INDEX updates) or ruled-out hypotheses (journal notes).

The integration that produced these alerts is `surveillance/regime_monitor.py` (committed today). Today's scan wrote 29 entries to `regime_alerts` table; this plan targets 3 of them.

---

## Pre-flight checks (run these first)

```bash
# Local DB recency
sqlite3 surveillance/data/surveillance.db "SELECT MAX(last_seen) FROM deployers"

# Regime alerts from prior session still present?
sqlite3 surveillance/data/surveillance.db \
  "SELECT signal_name, COUNT(*) FROM regime_alerts GROUP BY 1"
# Expected: ~29 rows across 4 signals (new_deployers_total, confirmed_traps,
# suspected_traps, approval_events)

# Tests still green?
python -m pytest tests/surveillance/ -q
# Expected: 14 passed
```

If any check fails: resync, re-run regime_monitor, or fix test breakage before proceeding to Phase A.

---

## Phase A — Surveillance investigation (~2 hours)

### A1. April-23 approval-events spike → April-25 deployer spike correlation

**Hypothesis:** The `0xb0b0b690` vanity-funder mass-fund event on 2026-04-25 (8,052 new deployers, P(CP)=1.000) was preceded by approval-side victim accumulation on 2026-04-23 (4,329 approvals vs ~1,500 baseline, P(CP)=0.853).

**Steps:**
1. Group approval_watchlist by `(date, contract_address)` for 2026-04-22 → 2026-04-26. Top contracts by approval count.
   ```sql
   SELECT DATE(approve_timestamp) AS d, contract_address, COUNT(*) AS n
   FROM approval_watchlist
   WHERE approve_timestamp BETWEEN '2026-04-22' AND '2026-04-27'
   GROUP BY d, contract_address ORDER BY n DESC LIMIT 30
   ```
2. Trace those contracts back to deployers. Are they downstream of `0xb0b0b690` or a different operator?
3. Profile victim addresses that approved on 2026-04-23. Bots vs retail. Cross-reference `bot_candidates`.
4. Check whether `0xb0b0b690` or its first downstream wallets were active before 2026-04-23.

**Success criteria:** Yes/no determination on causal link. If yes, documented attack sequence.

**Output:** `surveillance/data/cases/CASE_APR23_25_STAGING.md` if linked; journal note otherwise.

**Predicted outcome (write before starting, check at end):**
- A1 prediction: [LIKELY — staging pattern fits the b0b0b690 vanity-prefix operator profile, which is documented as deliberately calibrated]

**Time:** ~45-60 min.

### A2. May-5 confirmed-traps spike causal chain (207 confirmed, 4-10× surrounding days)

**Hypothesis space:**
- (a) iter_8 of drainer-spawn hub `0xf7883e3f` (May-5 was iter_8 spawn day per INDEX.md)
- (b) bytecode classifier rule was added/modified producing retroactive re-classification
- (c) backfill / re-scan job confirmed earlier-deployed contracts

**Steps:**
1. Top deployer drives the spike:
   ```sql
   SELECT deployer_address, COUNT(*) FROM contracts
   WHERE confidence_tier='confirmed' AND detection_timestamp LIKE '2026-05-05%'
   GROUP BY 1 ORDER BY 2 DESC LIMIT 10
   ```
2. Compare top deployer to iter_8 wallet `0xa8c7ac1cdc33...` and its downstream.
3. Git log for classifier changes: `git log --oneline --since="2026-05-04" --until="2026-05-06" -- surveillance/bytecode_classifier.py`
4. Bytecode hash distribution on May-5: one template or many?

**Success criteria:** Attribution to (a), (b), or (c).

**Output:** Journal note + INDEX.md update if a new finding.

**Predicted outcome:**
- A2 prediction: [SPLIT — partly (a) iter_8 contributed but unlikely to account for 207 alone; more likely a backfill or classifier change]

**Time:** ~30 min.

### A3. Coffee Fleet vs. approval-events decay (Apr-23 → Apr-25)

**Hypothesis:** Coffee Fleet's victim acquisition slowed Apr-23 → Apr-25 (totals: 4,329 → 2,039 → 1,424 → 1,217). Bots learned to avoid? Operators retired contracts? Victim pool saturated?

**Steps:**
1. Coffee Fleet daily approval counts:
   ```sql
   SELECT DATE(approve_timestamp), COUNT(*) FROM approval_watchlist
   WHERE LOWER(contract_address) IN
     (SELECT LOWER(contract_address) FROM contracts
      WHERE deployer_address='0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e')
   GROUP BY 1 ORDER BY 1
   ```
2. Coffee Fleet share of daily totals across the window.
3. Coffee Fleet deployment activity in same window.

**Success criteria:** Characterization — Coffee-Fleet-specific decay or systemic.

**Output:** Journal note + lexicon update if generalizable pattern.

**Predicted outcome:**
- A3 prediction: [SYSTEMIC — the decay is total-corpus-level; Coffee Fleet is one of many operators; expect Coffee Fleet's share roughly constant]

**Time:** ~30 min.

---

## Phase B — Engineering follow-ups (only if Phase A finishes early)

| Item | Effort | Value | Order |
|---|---|---|---|
| **B1.** Coalesce consecutive regime alerts into "episode" objects (V2 regime monitor) | ~60 min | Reduces 29 daily alerts → ~6 episode entries; makes alerts actionable | 2nd |
| **B2.** Wire `regime_monitor.py` into `run_surveillance.py` as scheduled daily job | ~30 min | Production integration deploys on worker restart | 3rd |
| **B3.** Fix INV-016 (extraction_events schema gap; guard migration with table-existence check; expand smoke test) | ~15 min | Closes latent bug; enables full-init-db smoke test | 1st |

**Recommended ordering if all three:** B3 → B1 → B2.

---

## Phase C — Reflection loop pass (mandatory)

Standard 7 steps per `memory/LOOP.md`. Expected cost ~10-15 min for an action-mode session producing 1-3 findings.

Specific disciplines for this session:
- **Step 5 — Surprise Logging:** the A1/A2/A3 predictions above are deliberately pre-registered. At session end, compare prediction vs. observation for each. Without that comparison, "surprise" becomes post-hoc rationalization.
- **Step 6 — Coherence Check:** verify the 29 regime alerts in `regime_alerts` are still present (data agrees with the alerts that motivated the work). If alerts are gone, something deleted them between sessions — major coherence issue.

---

## Working assumptions

1. **Local DB:** synced 2026-05-10 (per STATE.md). Phase A queries against Apr-23 → May-05 window — fully captured. Resync only if A1 results need post-2026-05-10 data.
2. **ADR-006 skip count starts at 0** (cleared on the 4th pass).
3. **Expect 0-2 new UNKNOWNs from investigation.** Surveillance work historically surfaces them.

---

## Out of scope

- New corpus analysis beyond the 3 candidates (stay focused).
- DSL-side work (`flux_manifold/`). Not blocked, just not this session's focus.
- README/doc polishing beyond what investigation findings require.

---

## Open questions (decide before starting)

1. **Sync first or not?** If sync was requested at session start, the data window may extend beyond 2026-05-13. Re-run regime_monitor against the fresh DB before Phase A to capture any newly-detected alerts. Otherwise the existing 29 alerts are the work surface.
2. **Case file vs. journal note threshold:** if a finding names a NEW entity not yet in `docs/INDEX.md`, write a case file. Otherwise journal note + INDEX update.
3. **Negative-result threshold:** for each hypothesis, agree on the negative-result evidence ahead of time. E.g., for A2 the iter_8 hypothesis is RULED OUT if `0xa8c7ac1cdc33...`'s downstream accounts for < 50% of May-5 confirmed contracts.

---

*If this plan is wrong by the time you read it (corpus changed, regime alerts gone, etc.), update `memory/STATE.md` first, then re-derive Phase A targets from the current `regime_alerts` table.*
