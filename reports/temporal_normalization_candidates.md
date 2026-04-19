# Pattern B — Temporal Normalization Candidates

**Date:** 2026-04-18
**Scope:** Query-only investigation. Corpus: 10,132 profiled deployers as of today's producer refresh.
**Epistemic:** Tier B inferential. Result: **negative candidate list, meaningful baseline finding.**

---

## What we looked for

Operators deliberately deploying on a Western-workday schedule (9–5 Europe-business hours UTC) to evade behavioral detection, while their contracts score suspicious on OTHER dimensions. Specifically:
- `timezone_guess = 'europe_business'`
- `active_window_pct >= 0.7` (activity concentrated in that UTC window)
- `active_days >= 14` (sustained, not a one-shot pattern)
- At least one contract in `confidence_tier ∈ ('suspected', 'confirmed')`
- Not already attributed to `org_001`–`org_004`

## Baseline finding (useful even though candidate count is 0)

Within the population of deployers who have ≥14 active days AND ≥0.7 window concentration (the "sustained schedule" cohort), trap-adjacency rates by timezone_guess:

| timezone_guess | deployers | with ≥1 trap | rate |
|---|---|---|---|
| europe_business | 14 | 0 | **0.0%** |
| asia_morning | 8 | 0 | 0.0% |
| night_shift | 5 | 1 | 20.0% |
| americas_evening | 2 | 0 | 0.0% |

Only 29 deployers across all timezone bins meet the sustained-schedule filter. Of those, **zero `europe_business` deployers have any trap-adjacent contracts.** The only sustained-schedule deployer with trap adjacency is in the `night_shift` bucket — which matches org_001's primary cluster (per Correction-adjacent work and `project_l3_narrative.md`'s deck summaries: org_001 is 59% night_shift + asia_morning).

## What this means

**The Pattern B hypothesis is not supported by the current corpus.** Interpreted carefully:

1. Operators willing to invest in sustained 14+ day schedule normalization — i.e., the kind who would adopt europe_business to blend in — are not currently producing trap-adjacent contracts in our window.
2. The "westernized schedule is a laundering signal" framing is not yet falsifiable at our scale because **too few deployers sustain any schedule pattern long enough to evaluate.** 10,132 profiled deployers but only 29 pass the sustained-concentration filter; of 1,418 europe_business deployers, only 14 are sustained.
3. The corpus-age constraint bites here too. A deployer would need to operate continuously for weeks to build the schedule fingerprint; most high-risk deployers in our data operate in bursts (deployment_style = 'burst' is common). Burst operators don't have a "workday" pattern to fake.

## Cross-chain comparison against known orgs

From the prod snapshot:

```
org_001  night_shift         89
org_001  asia_morning        50
org_001  europe_business     28
org_001  americas_afternoon  26
org_001  unknown             23
org_001  americas_evening    20
```

org_001 has a **23.5% europe_business representation** — if anything org_001 *already* over-indexes on europe_business relative to the corpus baseline (~14%). That's either (a) org_001 actually has real European-timezone operators, (b) they're already practicing Pattern B and we haven't caught it through this filter, or (c) the timezone_guess heuristic is noisy for burst deployments. All three are plausible.

If (b) — that org_001 is already doing Pattern B — the filter would need to be Pattern B ∩ org_link = org_001, which isn't what we were looking for (the handoff framed Pattern B as *un-attributed* operators trying to blend in). 28 org_001 deployers in europe_business is not the Pattern B we want to detect; they're known-org europe-timezone operators.

---

## What would make Pattern B detectable

Three improvements, ranked:

1. **Weekly vs weekday activity ratio as a distinct feature.** Real Western devs deploy during weekdays only; adversarial Pattern B fakers would show the same pattern, but so would burst-happening-to-hit-Tuesday-to-Friday. Need explicit `weekend_activity_ratio` in deployer_profiles (currently not exposed). Would help separate "genuine pattern" from "accidental concentration."

2. **Corpus-age floor of 60 days.** A sustained schedule pattern needs ~30 deploys to evaluate; a 30-day corpus caps how many deployers hit that count.

3. **Score against known-legitimate developer timezone baselines.** If we had a reference set of legitimate developers on Base (e.g., Coinbase's own contracts, major protocol deployers) with their timezone distribution, the Pattern B filter becomes "outlier deployers in europe_business compared to the legitimate europe_business baseline" rather than "all europe_business deployers." Right now we don't have that reference baseline populated in `infrastructure_registry`.

## Cross-ref

- org_001 cluster membership: `entity_classification.category = 'CRIMINAL' AND subtype = 'org_deployer'`
- Temporal analysis primitives: `deployer_profiles.timezone_guess`, `peak_hour`, `active_window_start`, `active_window_pct`, `hour_distribution`
- Producer that populates these: `deployer_profiler --profile-all` (scheduled nightly 02:00 UTC per Correction #7)

---

## What this does NOT claim

- No operator is accused of temporal laundering based on this scan.
- The absence of candidates does not prove Pattern B doesn't happen in the ecosystem; it means **the fingerprint isn't observable at our current corpus age and schedule-concentration thresholds.**
- Re-scan trigger: corpus age ≥ 60 days AND `weekend_activity_ratio` column added to profiles.
