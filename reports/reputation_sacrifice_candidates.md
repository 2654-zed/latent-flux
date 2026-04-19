# Pattern A — Reputation-Building Sacrifice Candidates

**Date:** 2026-04-18
**Scope:** Query-only investigation against the 2026-04-18 corpus state. No new modules, no alerts fired, no classification changes.
**Epistemic:** Tier B inferential. Candidates flagged for review, not for automatic elevation.

---

## What we looked for

Signal: a deployer whose first 3+ deployments were classified `unknown` (no trap signatures at ingest, no confirmations by our classifiers) followed by a later deployment that was promoted to `confirmed` via **behavioral trigger** (bot trapped in a tx). The trajectory duration ≥ 3 days (so it's a sequence, not a same-block burst), and the deployer deployed at least 5 contracts total.

Guardrail against false positives: the behavioral-confirmation requirement — the final contract must have caught a bot, not just been inherited-upgraded via `update_contract_confidence`. This matters because Correction #4 dissolved a class of mislabels where velocity-escalation upgraded contracts without evidence.

## Result

**4 candidates out of 5,810 deployers** that have ≥5 contracts in the corpus. The signal is extremely narrow but clean.

| Deployer | Contracts | Trajectory days | Chain | Final confirmation |
|---|---|---|---|---|
| `0x614737b68f3b7fa65a165c4057c2412d4a3a9271` | 5 | 23.8 | Base | bot `0xcffdfadd…` trapped |
| `0x5eb7a6583f6386d040f83429a9d46138cb9e8ea3` | 11 | 21.5 | Base | bot `0x36f746e2…` trapped |
| `0x809088835c4cf6a1af0de599da08ff355a9723db` | 11 | 10.4 | Arbitrum | bot `0x8ba99de8…` trapped |
| `0x021868f2e3d49c059ef52b539aaa933b437e0321` | 18 | 7.0 | Base | bot `0x21349…cd193` trapped |

---

## Interpretation

All four deployers match the surface pattern: quiet early contracts, loud late contract. Whether that matches the advisor-parasite **reputation sacrifice** model depends on follow-up:

- **The hypothesis requires the early contracts to be substantively different from the final one** — real functionality, real users, real activity — not just dormant-on-purpose placeholders waiting to be classified.
- Our ingest doesn't record sustained functionality for unknown contracts (we track deployment + interaction events). Without deeper per-contract activity analysis, these four are structural candidates, not confirmed reputation sacrifices.

To promote any of these to confirmed Pattern A, we'd want:
- Per-contract interaction volume over the trajectory — did the "quiet" early contracts have real caller diversity?
- Bytecode family per contract — are the early contracts in the same family as the final weapon, or different?
- Deployer profile score trajectory — did the deployer's behavioral score drop at the final deployment?

Those are all SQL-queryable enrichments, one more pass. Not done in this scan.

---

## What the absence of more candidates tells us

5,810 deployers filtered down to 4. Two possible reasons:

1. **The pattern is genuinely rare.** Most operators don't bother with reputation building — phishing + Permit2 sweeping is so lucrative (CE5E doing $137K/day) that the patient-trust-building model isn't competitive for the raw economics.
2. **Our classifier catches early trap bytecode reliably enough that "quiet early deployments" is a vanishing population among deployers who later deploy a confirmed trap.** The filter `all early contracts = unknown` requires the classifier to have missed them, which only happens for contracts with genuinely trap-free bytecode.

Both are consistent with a 30-day corpus. Reputation sacrifice at advisor scale (months of trust-building) would need a longer observation window — same corpus-age constraint that capped Pattern F.

---

## What this does NOT claim

- None of these 4 deployers are accused of running a reputation-sacrifice scheme. Structural fit only.
- No entity_classification updates, no alerts.
- The "quiet early contracts" in each trajectory may be legitimate — tests, dev contracts, or genuine products the same operator also runs. The framework requires proof the early contracts were exploitative toward users, not just that they existed.

---

## Cross-ref

- Pattern F (advisor-parasite) shares the "patient long-game extraction" framing. Pattern A is the deployer-scale version; Pattern F is the victim-extraction-cadence version. Both are corpus-age-constrained.
- Pattern D (cross-chain import) has 54 candidates on the same 100-deployer sample (see separate report). Deployer `0x4885631c…` appears in both the Pattern C (CEX-funded) and Pattern D (mainnet-imported) candidate lists — one entity on multiple signal surfaces, worth cross-referencing.

---

## Scoped module (per handoff) — deployer_trajectory_analyzer.py

Not implemented. Proposed interface:

- `analyze_trajectory(conn, deployer_address) -> dict`
- Reads all contracts for the deployer, orders by detection_timestamp
- Computes per-contract enrichment (family, tier, caller count if available, behavioral signature)
- Returns a trajectory object with an early-vs-late "delta" score
- Called on-demand from investigation scripts; not a periodic producer

**Estimated cost:** ~150 LoC new module, O(1) SQL per deployer. Fast. Could become a column on `deployer_profiles` if we decide we want the delta as a persistent behavioral fingerprint.

Decision gated on your review. The 4 candidates above are small enough that building the module for 4 cases is not justified today; build it only if (a) re-scan after corpus age ≥ 90 days returns more candidates, or (b) you want the delta as a persistent fingerprint on deployer_profiles for other uses.
