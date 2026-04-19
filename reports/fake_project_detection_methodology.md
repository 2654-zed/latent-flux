# Pattern E — Fake Legitimate Projects: Detection Methodology Proposal

**Date:** 2026-04-18
**Scope:** Methodology proposal only. No queries run, no candidates produced. Per the handoff: "This is the hardest to detect algorithmically. Scope a methodology before implementing."
**Epistemic:** This document is a plan. Every claim here is provisional until the methodology gets executed against the corpus, at which point findings would move to a separate `pattern_e_candidates.md`.

---

## The pattern to detect

A project has real on-chain activity — deployed token, DEX liquidity, trades, users — but the *economics* don't make sense: no revenue model, no product beyond the token, no path to profitability. The project operator either:

1. **Exits via rug-pull** (classic, but Pattern E is narrower — the rug is not the payload)
2. **Uses the project as reputation collateral** for a subsequent weaponized deployment (Pattern A intersection)
3. **Recycles the project's accumulated capital into new deployer wallets** that subsequently run trap operations

The detection signal Pattern E uniquely targets is #3: the project didn't collapse outright, it **wound down** in a controlled way, and the ETH that was liquidity is now funding the next trap fleet.

## Why this is harder than Patterns A–D

- **No trap bytecode in the "legitimate" phase.** The ERC-20 token and DEX pool are standard contracts. Our bytecode classifier sees nothing suspicious.
- **No drain event.** Funds flow out via legitimate withdrawal patterns (LP removal, team token sales). No Permit2.transferFrom signature to match against.
- **The weaponized deployer is a *different address* than the project operator.** The link between the two is a fund transfer through intermediate wallets, indistinguishable at the tx level from any other ETH transfer.
- **No victim-reported signal.** Users lost money but frame it as bad investment, not fraud.

The detection has to operate on a graph structure (fund flow + contract lifecycle) rather than per-contract bytecode features. That's fundamentally different from the signal types our current pipeline emits.

---

## Proposed three-phase methodology

### Phase 1 — Project lifecycle detection

For each contract in our `contracts` table that our classifier labeled `unknown` at deploy time (no trap signatures):

- **Sustained activity threshold:** transaction_events count ≥ 100 over ≥ 7 days → it's a "real" project, not a dormant contract
- **Liquidity presence:** token appears in pair_creation_events OR has Transfer event volume suggesting DEX routing → treated as tradeable
- **Lifecycle endpoint:** activity drops to <1 tx/day for ≥ 7 days after the sustained-activity period → project has wound down
- **Terminal state:** contract is NOT self-destructed (that would be rug-pull, out of Pattern E scope), but is effectively inactive

Output: a list of "wound-down projects" — contracts that had real activity then went quiet.

**Corpus-age impact:** 30-day corpus caps "sustained activity ≥ 7 days + wound-down ≥ 7 days" window at maybe 16 days of sustained activity. Will surface some candidates but not many. Re-scan at 90 days for substantially better coverage.

### Phase 2 — Capital recycling trace

For each wound-down project identified in Phase 1:

- Identify the project's controller addresses — the deployer of the token, LP positions, team wallets
- Trace outbound fund flows from those controllers in the period surrounding wind-down
- **Key signal:** controller wallet sends a meaningful ETH balance (≥ 0.1 ETH or equivalent in USDC) to a **previously inactive address**, which then deploys one or more contracts within N days

The graph walk is: project_controller → intermediate_wallet₁ → intermediate_wallet₂ → new_deployer → new_contract. We only have data for the first hop via our `transaction_events` and approval tracking; subsequent hops require either RPC tracing or a graph-indexing pass over transaction_events we don't currently do.

**Required enhancement:** an outbound-fund-trace function similar to `auto_funder_tracer` but running *forward* from controller wallets rather than backward from new deployers. Call it `capital_recycling_tracer`. Same Alchemy-call budget discipline (one call per hop).

### Phase 3 — Destination deployer evaluation

For each destination-deployer identified in Phase 2:

- Compute the deployer's risk profile at the time of first contract deployment
- If the new deployer deploys contracts flagged suspected or confirmed trap, the graph chain is complete: **wound-down project → intermediate wallets → new trap deployer**
- The full chain is the Pattern E signature

**Output:** confirmed Pattern E would be a tuple `(defunct_project_address, controller_wallet, fund_flow_path, destination_deployer, new_trap_contracts_deployed)`. This is a rich attribution object — structurally equivalent to Correction-log quality evidence, Tier B-high.

---

## What Pattern E requires that doesn't exist yet

1. **Wound-down project detection logic.** Net new. ~100 LoC module. Runs as a periodic scan (weekly cadence matches the slow pattern rhythm). Could be wired into the _analysis_scheduler.

2. **`capital_recycling_tracer`.** New module. Mirrors `auto_funder_tracer` but forward-direction. ~150 LoC + ~3–10 RPC calls per wound-down project candidate. If 50 wound-down projects per scan × 5 calls = 250 Alchemy calls per weekly scan. Overdue for approval check (200-call per-investigation budget could be raised to a per-week budget for this specific producer).

3. **Graph representation for fund-flow chains.** Currently transaction_events is a flat table of interaction records. For the Phase 2 trace to work at scale, we'd want either:
   - An on-the-fly recursive query (simple but slow)
   - A materialized `wallet_connection` table computed nightly by a new producer
   - An external graph DB (overkill)

   The on-the-fly query is the cheapest first version. Let it scale-fail and migrate to materialized if needed.

4. **A re-scan cadence.** Pattern E unfolds over months. Weekly scan with a 30-day lookback is the right shape. Less frequent than daily-metrics, more frequent than the pattern can evade.

---

## Relationship to existing surveillance

- **Pattern A (reputation sacrifice)** overlaps partially: in Pattern A, the *same deployer* runs both legitimate and weaponized contracts. In Pattern E, the operator switches wallets (more sophisticated). Both are detectable; Pattern A from single-deployer trajectory, Pattern E from multi-wallet fund-flow trace.
- **`org_cycles` module** already does some temporal analysis on deployers; Pattern E would read from that but compute at a different level (project-lifecycle, not deployer-activity-pattern).
- **`extraction_events` table** is the output target for confirmed cases. Pattern E at full maturity would produce `EXTRACTION_*` rows labeling the defunct project + recycled funds + new trap deployer as a single extraction chain.

---

## Estimated implementation cost

If approved:

- **Wound-down project detection:** 1 day (1 new module, one migration, one periodic job)
- **Capital recycling tracer:** 2 days (new module, ERC-20 Transfer indexing for target addresses, RPC budget configuration)
- **Graph query support:** 1 day (recursive SQL helper, or materialized table producer)
- **Integration testing:** 1 day (against existing corpus to calibrate thresholds)
- **First scan + report:** 0.5 day

Total: ~5 days of focused work for the full Pattern E pipeline. Could be sequenced as three separate implementation steps, each landing independently and producing intermediate value (wound-down project list → fund-flow traces → attributed chains).

---

## Recommended decision path

**Not ready to build today.** Preconditions:

1. Corpus age ≥ 60 days before the wound-down detection threshold (7-day sustained + 7-day quiet) has enough signal.
2. Pattern A re-scan at 90 days confirms reputation-sacrifice is or isn't common — if common, Pattern E is a natural extension; if rare, Pattern E is likely also rare and may not pay back the 5-day implementation cost.
3. An RPC budget for the capital_recycling_tracer gets explicitly approved (the current 200-call-per-investigation limit would need to expand to recurring weekly budget).

**Do today:** memorialize this methodology. Link from `behavioral_laundering_detection_scope.md` as the expanded version of Pattern E. When corpus age and A-pattern signal conditions clear, revisit the build decision.

---

## What this document does NOT claim

- No candidates identified. No deployers accused.
- No schema changes.
- No new modules built.
- Methodology unvalidated against real cases in our corpus.
- Cost estimates are engineering-gut-feel, not measured.

The value of this document is as a **reservation of scope** — if Pattern E becomes important, this is what executing it would look like. Current value: low (no findings). Future value: high (if reputation-sacrifice patterns become common or a customer asks about project-exit-to-trap-deployment correlation, the methodology is already framed).
