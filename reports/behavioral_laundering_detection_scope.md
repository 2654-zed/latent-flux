# Behavioral Laundering — Detection Scope

**Date:** 2026-04-18
**Purpose:** Consolidated scope doc tracking each behavioral-laundering pattern, its detection status, what data would be needed to promote from hypothesis to detection, and the cost to build. Deliverable #8 from the Part 2 behavioral-laundering handoff.

**Epistemic frame for all patterns listed here:** Tier B inferential. None of these patterns are currently operationalized into active detection; this file is the planning surface where we decide which ones to build.

---

## Pattern index

| ID | Name | Investigation status | Detection readiness | Next step |
|---|---|---|---|---|
| A | Reputation-building sacrifices | Not yet investigated | — | Phase 1 query per Part 2 handoff |
| B | Temporal pattern normalization | Not yet investigated | — | Phase 2 query per Part 2 handoff |
| C | Funding-chain laundering via CEX | Not yet investigated | Requires RPC budget | Phase 3 (blocked on Alchemy approval) |
| D | Cross-chain reputation import | Partially validated via EXTRACTION_004 (Rhea) | Weak; on-chain method unclear | Phase 4 (Etherscan approved, not run) |
| E | Fake legitimate projects | Not yet investigated | — | Phase 5 (methodology proposal) |
| **F** | **Advisor-parasite extraction** | **Scanned 2026-04-18, negative** | **Blocked on Transfer-event indexer** | **Re-scan when corpus age ≥ 90 days** |

Patterns A–E remain in the state the Part 2 handoff scoped. Pattern F is the first pattern where both the investigation *and* the first scan have been run; the rest of this document is Pattern F memorialization.

---

## Pattern F — Advisor-Parasite Extraction

### The pattern

An operator cultivates a long-term trust relationship with a victim (framed as advisor, guide, guru, coach, or expert) and extracts wealth through repeated small tolls, fees, or approvals over months or years. Structurally, the on-chain fingerprint is hub-and-spoke: many victim wallets approve a single collector address, and transfer small amounts to it at irregular cadence. Phenomenologically, the distinction from phishing is that the victim doesn't know they're being exploited because the advisor is consistently helpful — the extraction is wrapped in ongoing interaction. Recovery is almost impossible because the victim can't distinguish advisor-extracted funds from legitimate losses.

### Why it's distinct from Patterns A–E

- **Not Pattern A (reputation sacrifice):** the advisor doesn't deploy trap contracts in their own name; they *use* trap-adjacent or benign infrastructure.
- **Not Pattern B (temporal normalization):** the advisor may actually operate 24/7 or on opportunity-driven schedules; there's no reason to fake a 9–5 pattern.
- **Not Pattern C (CEX funding laundering):** that concerns where the *attacker* gets funded. Pattern F concerns where the *victim* pays from.
- **Not Pattern D (cross-chain import):** advisor-parasite can operate on one chain indefinitely; cross-chain isn't the evasion mechanism.
- **Not Pattern E (fake projects):** the advisor doesn't need to run a project; they can be a person with a Telegram channel who charges 0.05 ETH per signal.

### Ideal on-chain fingerprint

1. **Hub contract or EOA** receives small payments from **50–300 distinct approvers** (too few = niche operation not worth modeling; too many = likely a public protocol).
2. **Duration ≥ 60 days** with continuous, not burst, incoming flow.
3. **Per-victim cadence:** 1–5 outbound transfers per month from each victim, not a single sweep.
4. **Victim retention:** approvers retain positive balance throughout, continue interacting with the broader ecosystem.
5. **Amount clustering:** amounts at human-chosen round numbers (0.01 / 0.05 / 0.1 ETH or 50 / 100 / 500 USDC) rather than protocol-computed values.
6. **Not a public protocol:** spender is not in infrastructure_registry, is not 1inch / LI.FI / Uniswap / CCTP.
7. **Chain concentration:** single-chain operation (advisor's community is typically one ecosystem).

### What we can observe today

| Signal | Source | Status |
|---|---|---|
| Hub-and-spoke approval shape | `approval_events` grouped by `spender` | **Visible**, with false positives from DeFi routers |
| Duration | `approval_events.timestamp` min/max | **Visible**, but limited by corpus age (30 days as of 2026-04-18) |
| Per-victim cadence | Not fully indexed | **Partially visible** — we count approvals per approver, but not actual extraction transfers |
| Retention | Cross-ref against `X402_AGENT_DRAIN` victim set | **Statistically weak** at current drain-victim-set size (128) |
| Amount clustering | `approval_events` does not carry amount; `transaction_events.value_wei` populated for only 2.5% of rows | **Not currently observable** |
| Public-protocol exclusion | `infrastructure_registry` (12 rows) + naming of top DeFi contracts | **Incomplete** — our registry is Circle CCTP only; hundreds of legitimate protocols unlabeled |
| Chain concentration | `approval_events.chain` | **Visible** |

### Detection gap: what we'd need to operationalize

Ranked by leverage:

1. **Narrow Transfer-event indexer scoped to approver cohorts.** For each high-diversity spender identified by the approval-count query, subscribe to ERC-20 Transfer events where the approver is `from` and the spender (or its downstream graph) is `to`. Builds the actual per-victim extraction cadence record that separates advisor (small steady outflow) from phishing (single sweep to zero) from legitimate protocol use (irregular, event-tied). **Estimated cost:** ~200 LoC new module, one WebSocket subscription, low volume (scoped to ~5,000 total addresses = the approver cohorts of ~50 candidate spenders).
2. **Approver account-age enrichment.** One-time `eth_getTransactionCount` per approver against earliest mainnet block. Distinguishes new-to-crypto advisor victims from longtime DeFi users. **Estimated cost:** ~5,000 RPC calls at current candidate scale; exceeds the 200-call Phase 3 budget from the Part 2 handoff. Needs separate approval or a narrower subset.
3. **Populated `infrastructure_registry` for legitimate-protocol exclusion.** Currently the top 16 candidates in the advisor-parasite scan were dominated by unlabeled legitimate protocols (Base Uniswap Router, 1inch, LI.FI, vanity-prefix protocols). A richer registry would let the scan cleanly exclude them and surface only true unknowns. **Estimated cost:** incremental — add entries as they're identified. Each new entry follows the same draft-review-commit pattern as Circle CCTP.

### Re-scan trigger conditions

The 2026-04-18 scan returned negative for reasons that are structural, not definitive:

- Corpus age ≤ 30 days (advisor pattern unfolds over months)
- Retention metric doesn't discriminate at small drain-victim-set scale
- Per-victim extraction cadence not indexed
- Legitimate-protocol exclusion list incomplete

**Re-run conditions that would likely produce signal:**

1. **Corpus age ≥ 90 days** (calendar trigger: 2026-06-17 at earliest). The duration distribution of candidate spenders will broaden; genuine advisor relationships will show the month-scale retention that separates them from burst traffic.
2. **Transfer-event indexer deployed for ≥ 30 days.** Once we have per-approver-to-spender outbound flow data, the extraction-cadence signal becomes directly observable rather than inferred.
3. **`infrastructure_registry` grown to ~50 entries** covering the major DeFi protocols visible in our approval_events top-spender list. At that point, the residual unknowns in the scan output become statistically meaningful candidates rather than mostly-unlabeled-legit-infra.

**Any one of these three triggers should prompt a re-scan.** In the absence of all three, the scan will continue to return 16 candidates and no confirmed advisor-parasites, and re-running is wasted compute.

### What the 2026-04-18 scan established that's still useful

- **No current rogue drainer in our corpus (CE5E, A7B9, E3B2, E717, D270) matches the advisor-parasite profile.** They're all one-shot-sweep operators. If advisor-parasites exist in the ecosystem, they're a *different* population than the drainers we track.
- **Six spenders remain unidentified** (not in `infrastructure_registry`, not known DeFi brands, not confirmed traps): `0x57df6092…`, `0xccc88a9d…`, `0xd8ba9d1a…`, `0x9dda6ef3…`, `0x6131b5fa…`, `0xec3576c5…`, `0xac4c6e21…`, `0x337685fd…`, `0x1b02da8c…`. Their current behavior profile doesn't match advisor-parasite, but also doesn't cleanly match legitimate protocol. Worth cross-referencing against external sources (Dune dashboards, DefiLlama protocol lists) before the next scan rerun.
- **The 30-day corpus limit is now a measured fact, not a guess.** Future scope docs should reference this as the corpus-age floor for detecting slow-extraction patterns.

### Source material

- Full scan methodology and candidate list: `reports/advisor_parasite_candidates.md` (2026-04-18)
- Scan script: `scripts/advisor_parasite_scan.py`
- Candidate refinement script: `scripts/advisor_parasite_refine.py`
- Pattern framing originates from the behavioral-laundering handoff (2026-04-18) plus the user's subsequent elaboration on "advisor-as-parasite model operates on a relationship of trust over time."
- Canonical one-shot-sweep counterexample: `reports/case_CE5E_drainer_operation.md`

---

## Patterns A–E (deferred; see Part 2 handoff for specifications)

Not expanded here. Each pattern should get its own detection-readiness section matching Pattern F's structure once investigated. The seven deliverable reports in the Part 2 handoff remain the authoritative spec for A–E; this scope doc will accumulate them as they land.

---

## Cross-references

- Part 2 handoff: the 2026-04-18 USDC-Bridge-plus-behavioral-laundering brief (original conversation context).
- Pattern D partial validation: `reports/extraction_event_004_rhea_finance.md` — Rhea Finance Subject Wallet was funded via cross-chain onboarding (intents.near), demonstrating Pattern D in a NEAR/Solana context.
- Infrastructure registry discipline: `reports/circle_bridge_infrastructure.md` + Correction #7 in `reports/correction_log.md`.
