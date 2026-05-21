# Confirmed-Tier Audit — Design Document

**Filed:** 2026-05-21
**Author:** SAI inferential-layer review, post-Correction-#24
**Status:** PLAN — no audit executed yet. This document defines what an audit would look like, so the audit's design can be reviewed and chosen before any execution.

---

## Why this audit is needed (the trigger)

Correction #24 (filed 2026-05-21) retracted the entire `0x752c5a95` "Pre-Drain Harvester" finding, including:

- The 2026-04-24 INDEX entry that flagged it as "the largest active confirmed-tier approval pool in the corpus"
- The 2026-05-09 "discharge event" (4,587 victims drained in 30 minutes)
- The 2026-05-21 case file framed as "the strongest validated Tier-C prediction in the corpus to date"
- The lexicon Adversarial Maneuver entry citing the case as "the canonical example of disrupt-positioning succeeding"

What actually happened: `0x752c5a95` is **OneFootball Club (OFC)** — a verified Animoca-deployed ERC-20 token with 3,904 holders and $7.9M market cap, listed on CoinGecko. The "discharge" was three failed `transferFrom` transactions. **Three independent Layer 3 classifiers simultaneously misclassified a legitimate token as an adversarial harvester:**

| Classifier | Why it fired | What it should have done |
|---|---|---|
| Bytecode pattern (`has_asymmetric_transfer + has_unusual_fee_structure`) | Pattern matches Animoca's `@animoca-network/contracts` framework's standard `ContractOwnership` + `TokenRecovery` modifiers (the `CALLER → EQ → JUMPI → REVERT` signature is literally `onlyOwner`) | Not flag verified contracts whose deployer carries a public-entity Blockscout label |
| Behavioral confirmation (`confidence_reason: "Behavioral confirmation: bot X trapped in tx Y"`) | Bot front-running pre-launch ERC-20 → contract reverted as designed → pipeline read revert as trap firing | Distinguish "designed pre-launch revert" from "deceptive trap revert" |
| `approval_watchlist.drain_detected=1` pipeline | Failed `transferFrom` calls credit phantom drain rows against entire approval pool | Only credit drain rows when the corresponding tx succeeded |

**The implication: every confirmed-tier label in the corpus that depends on any of these three classifiers is in a population that contains an unknown false-positive fraction.** Until that fraction is measured and bounded, the corpus's headline numbers (1,404→1,650 confirmed contracts; 3,437 lifetime drain events; 94 drainers; 2,963 victims; the Camouflage Ratio's 30.44% confirmed-tier number) cannot be cited.

This document defines the audit that bounds the false-positive fraction.

---

## What the confirmed tier currently looks like

Production state as of 2026-05-21 (via `railway ssh` direct query):

### Population

| Metric | Value |
|---|---|
| Total confirmed contracts | **1,650** |
| Distinct deployers across the confirmed population | 722 |
| Confirmed contracts where the deployer has ≥2 confirmed contracts (recidivism) | 1,107 (67.1%) |
| Confirmed contracts whose deployer has any mainnet history | 1,161 (70.4%) |
| Confirmed contracts whose deployer has >365 days mainnet history before L2 first-seen | 328 (19.9%) — the pristine-solo FP-risk pool from Correction #20 |

### By chain

| Chain | Count | Share |
|---|---|---|
| Base | 1,446 | 87.6% |
| Arbitrum | 175 | 10.6% |
| Optimism | 29 | 1.8% |

### By detection_method

| Method | Count | Share |
|---|---|---|
| bytecode_pattern | 897 | 54.4% |
| deployer_history | 753 | 45.6% |

Note: per Corrections #3 + #4, `bytecode_pattern` and `deployer_history` are **labels written by the pipeline**, not necessarily reflective of the actual evidence. A `bytecode_pattern` label can survive on a contract whose bytecode flags are empty (it's just the default label written at insert-time before bytecode analysis returns).

### By `confidence_reason` (first 30 chars, top patterns)

| Pattern | Count | Share | FP risk |
|---|---|---|---|
| "Behavioral confirmation: bot 0…" | 1,330 | 80.6% | **HIGH** if the bot revert was a pre-launch revert, not a trap revert |
| "Behavioral confirmation: 1 vic…" | 70 | 4.2% | LOWER — a real victim approval+drain pair is stronger evidence than a bot revert |
| "BACKFILL: Self-loop detected…" | 65 | 3.9% | HIGH — retroactive backfill from a single self-loop event; thin evidence |
| "Behavioral confirmation: 2 vic…" | 15 | 0.9% | LOW |
| "Behavioral confirmation: 3+ vic…" | 38 | 2.3% | LOW (real victim aggregation) |
| "Self-loop: deployer 0x…" | 93 | 5.6% | HIGH — deployer's own bot fires the trap, weaker than independent victim |
| Other (`CONFIRMED TRAP`, etc.) | 39 | 2.4% | Per-pattern review needed |

### Bytecode evidence depth

Only **88 of 1,650 (5.3%)** confirmed contracts have a `bytecode_cache` row that the pipeline can cite as evidence. The other 94.7% were confirmed via behavioral classifier alone, without retained bytecode classification.

Among the 88 with cache rows, flag patterns are:

| `(asym/rev/fee)` | Count | Share |
|---|---|---|
| `asym/-/-` (asymmetric only) | 28 | 31.8% |
| `-/-/-` (no flags) | 20 | 22.7% — **confirmed despite no bytecode evidence** |
| `-/-/fee` (fee only) | 18 | 20.5% |
| `-/rev/-` (revert only) | 7 | 8.0% |
| `asym/rev/-` | 6 | 6.8% |
| `asym/-/fee` | 4 | 4.5% |
| `asym/rev/fee` (all three) | 4 | 4.5% — the OFC pattern; strongest "trap" signature, but also the strongest framework-FP signature |
| `-/rev/fee` | 1 | 1.1% |

The flag combinations don't cleanly separate predator from legitimate. OFC (a legitimate token) carried `asym/-/fee`; that's the same flag set as 4 other contracts in the corpus.

### Approval pool / drain evidence

| Metric | Value |
|---|---|
| Confirmed contracts with ≥1 approval-pool row | 139 (8.4%) |
| Confirmed contracts with ZERO approval activity | **1,511 (91.6%)** |
| Total approval rows across confirmed contracts | 20,134 |
| Total `drain_detected=1` rows | 10,117 |
| Distinct drain tx hashes on confirmed contracts | 614 |

**91.6% of the confirmed tier has no approval-watchlist activity whatsoever.** Those 1,511 contracts were classified confirmed-tier on the basis of: (a) a behavioral trap-fire (most), (b) a self-loop event (small subset), or (c) a bytecode pattern at deploy time. The approval-watchlist methodology — the most concrete on-chain harm evidence Layer 3 has — was never engaged for them.

### The Bug #19 false-positive signature (drain-per-tx ratio)

The smoking gun for Correction #24's `approval_watchlist` bug. For each confirmed contract with drain events, compute `drains ÷ distinct_drain_tx_hashes`:

| Ratio bin | Contracts | Interpretation |
|---|---|---|
| ≥100 drains per tx | **4** | Almost-certainly Bug #19 — phantom rows credited from a small number of failed txs |
| 30–100 | 3 | Bug #19 suspect |
| 10–30 | 11 | Either Bug #19 OR a legitimate batched drain |
| 3–10 | 12 | Plausible small-batch drain; needs per-case review |
| <3 | 26 | Looks like real per-tx drain events |

**Of the 10,117 drain_detected=1 rows on confirmed contracts, 7,206 (71.2%) come from contracts with drain/tx ratios ≥30 — the Bug #19 false-positive cluster.** The top 4 alone (ratio ≥100) account for the majority of the inflated drain count.

This means: **the 10,117 lifetime drain count is dominated by ~7 contracts whose drain rows are mostly phantom.** If those are subtracted, the real drain count falls to ~3,000 — but it also means the headline "drain events" number is concentrated in a tiny subset and not representative of a broad-base drainer population.

---

## False-positive risk classes (the audit's threat model)

Eight identified FP-risk classes, ranked by expected prevalence in the confirmed-tier population. The audit's sampling strategy should ensure each class is covered.

### Class A — Verified ERC-20 tokens with on-chain pre-launch reverts (the OFC pattern)

**The Correction #24 class.** A legitimate ERC-20 deploys with a pre-trading window. Bots front-run and try to interact before trading is enabled. Contract reverts. Pipeline reads revert as a trap firing.

- **Probable prevalence:** Significant. Per CoinGecko, hundreds of new ERC-20s launch per week across Base/Arbitrum/Optimism, many with pre-launch gates.
- **Detection:** Cross-reference confirmed contracts against Blockscout verified-source flag + CoinGecko token listing + a holders-count threshold (≥100 holders is a strong "real token" signal).
- **OFC anchor:** 0x752c5a95 (retracted).

### Class B — Animoca-framework and OpenZeppelin-pattern contracts

**The Correction #24 bytecode-FP class.** Contracts using `ContractOwnership` + `TokenRecovery` patterns trip `has_asymmetric_transfer` + `has_unusual_fee_structure` flags.

- **Probable prevalence:** Likely sizeable — Animoca has 380+ portfolio companies; many use the `@animoca-network/contracts` framework. OZ patterns appear in many more.
- **Detection:** Cross-reference confirmed contracts against Blockscout verified-source flag + check for OZ / Animoca framework imports in source code.
- **Anchor:** 0x752c5a95 (OFC), 0xDA42FE (OFTAdapterFixedSupply).

### Class C — Behavioral-confirmation-only labels (no corroborating evidence)

The 1,562 contracts (94.7%) without a `bytecode_cache` row, plus the 20 that have a cache row with no flags set. These were confirmed by behavioral evidence alone — a bot reverted in their transaction.

- **Probable prevalence:** Already 94.7% of confirmed-tier. This is the broadest FP class.
- **Detection:** No internal signal exists. Requires external verification (verified source, deployer label, holders count).

### Class D — Self-loop / BACKFILL confirmations

The 158 contracts (9.6%) where the deployer's own wallet fired the trap. These are weak evidence — a deployer testing their own contract, or any single self-interaction, triggers this label. The BACKFILL subset (65 contracts) is even weaker: retroactive labeling from a single historical event.

- **Probable prevalence:** 9.6% by row count.
- **Detection:** Internal — these can be flagged from `confidence_reason LIKE 'Self-loop%' OR 'BACKFILL%'` and downgraded en masse to suspected-tier pending stronger evidence.

### Class E — Cache-transplant survivors

Per Correction #3 / #4, cache-transplant could inherit confirmed-tier from a stale family. The pipeline fix in Correction #4 prevents new mislabels, but historical cache-transplant rows that still carry the confirmed label may exist.

- **Probable prevalence:** Unknown, possibly small (the bulk migration ran 2026-04-16/17).
- **Detection:** Cross-reference `bytecode_cache.source_contract` against the source contract's current tier. If source ≠ confirmed but the dependent rows ARE confirmed via cache, that's a transplant artifact.

### Class F — Fee-on-transfer / reflection tokens (legitimate tokenomics)

Some legitimate tokens (SafeMoon-style reflection, FoT tokens, tax tokens) have unusual transfer logic that legitimately trips `has_unusual_fee_structure`. These are not always frauds — many are working-as-designed memetic tokens that the holders knowingly opted into.

- **Probable prevalence:** Moderate. On Base in particular, the chain hosts many meme/FoT tokens.
- **Detection:** Source-code inspection — does the source explicitly document the fee mechanism?

### Class G — Anti-bot defensive contracts (legitimate defensive infrastructure)

Some legitimate contracts (NFT mints, fair-launch protocols, vesting contracts) revert by design against MEV/sandwich bots. The behavioral classifier reads the revert as a trap.

- **Probable prevalence:** Small but non-zero.
- **Detection:** Source-code inspection + deployer reputation check.

### Class H — Standard suspected-tier confirmations (likely true positives)

The audit's purpose is NOT to find true positives — those should remain confirmed. The largest residual class is the contracts that correctly carry the confirmed label: deployers with multiple confirmed contracts (recidivism), high-fanout funder linkage (org_001/X402 operators), trap-pattern bytecode + real victim drain attribution, etc.

- **Probable prevalence:** The complement of Classes A–G. The audit's success criterion is bounding Classes A–G; the residual is the credible adversarial population.

---

## What "true positive" should look like (definition for the audit)

Before auditing, define the criteria a contract must meet to remain confirmed-tier. Three tiers of evidence strength:

### Strong evidence (any one of these justifies confirmed)
- Successful `transferFrom` drain via `approval_watchlist` with attested USD value loss and a victim public statement of harm.
- Deployer has ≥3 confirmed contracts and the funder is on a documented adversarial watchlist (org_001, X402 facilitators, Coffee Fleet, etc.).
- Direct bytecode anomaly: SELFDESTRUCT in a token contract, hidden `admin` setter, or known-malicious-template match (e.g., GoPlus-flagged template).

### Moderate evidence (need two of these together)
- Bytecode flags `asym + fee + rev` (all three) AND no Blockscout verified source.
- Deployer with mainnet history >1 year, fleet ≥2 confirmed, no OLI/Blockscout institutional label.
- ≥3 distinct real-victim drain events (one drain row per victim per tx, with successful tx status).

### Weak evidence (insufficient alone — downgrade to suspected pending corroboration)
- Single behavioral confirmation (one bot revert).
- Self-loop self-interaction.
- BACKFILL labels.
- Bytecode flags on a contract with verified Blockscout source.
- Drain rows from a tx that reverted.

## What "verified legitimate" should look like (audit-exit definition)

For the auditor to mark a confirmed contract as a confirmed FP, evidence required:

### Strong (one of these alone justifies retraction)
- Blockscout-verified source code matching a well-known open-source framework (OpenZeppelin, Animoca, Uniswap, etc.).
- CoinGecko listing with substantial holders (≥100) and 24h volume (≥$10K).
- Deployer carries a public-entity Blockscout label confirmed by tooltipURL pointing to a real institutional website.
- The contract is referenced positively by another L1 institution (e.g., a Compound/Aave market, an Uniswap pool).

### Moderate (need two)
- Blockscout-verified source + non-trivial holders count (≥10).
- Deployer has institutional mainnet vintage (>1 year) AND deploys only verified contracts.
- Active normal trading observable on DEX aggregators.

### Weak / circumstantial (do not justify retraction alone)
- Long deployer mainnet history (Correction #20 — pristine-solo signal is symmetric).
- Bytecode flag absence (Class C — much of the confirmed tier has no bytecode evidence anyway).
- Low approval pool / no drain events (Class C absence; doesn't prove legitimacy).

---

## Audit methodologies (the menu)

Six methodologies, ranked low to high cost. Use combinations:

### Tier 1 — Bulk OLI / Blockscout label cross-reference (cheap, fast)

**What:** For every confirmed contract, fetch the contract's Blockscout `get_address_info` metadata. Capture: verified source flag, token metadata (name/symbol/decimals/holders/market cap), public_tags, private_tags. Same probe on the deployer.

**Cost:** ~5 minutes of Blockscout API calls per chain at 100 req/min for 1,650 contracts = ~50 minutes for contracts + ~30 minutes for ~722 deployers. Negligible.

**Yield:** Direct flag for Class A (verified ERC-20) + Class B (Animoca/OZ) + identifies institutional deployers. Likely catches the majority of FPs at low cost. **First audit step.**

**Risk:** Blockscout label coverage isn't 100%; some legitimate contracts aren't labeled. Negative result is not proof of adversarial.

### Tier 2 — Verified-source-code cross-reference (cheap-moderate)

**What:** For contracts that Tier 1 reveals as Blockscout-verified, fetch the verified source code metadata (compiler, file list, framework imports). Cluster by framework signature: `@openzeppelin/contracts`, `@animoca-network/contracts`, `@uniswap/v3-core`, etc.

**Cost:** Subset of Tier 1 results. Add ~10 minutes for source-code fetches on the verified subset.

**Yield:** Directly classifies Class B (Animoca-framework) and analogous framework-FP classes. Generates the "is this a known-legitimate framework deployment?" judgment without per-contract human review.

### Tier 3 — Token-existence and market-data cross-reference (cheap, parallel to Tier 1)

**What:** For Tier-1 contracts that have an ERC-20 token signature, cross-reference against CoinGecko / Coinmarketcap / DEX aggregators for listing + holder count + volume. A verified contract listed on CoinGecko with >100 holders is almost certainly a legitimate token launch.

**Cost:** Free CoinGecko API has rate limits but covers the population we care about.

**Yield:** Class A direct hits. The OFC anchor would have been caught in 10 seconds via this method.

### Tier 4 — `approval_watchlist` tx-status re-classification (load-bearing)

**What:** For every `drain_detected=1` row with a `drain_tx_hash`, fetch the transaction status (success/error). Re-flag the row to `drain_detected=0` if the tx reverted. Recompute headline drain counts.

**Cost:** Probably the biggest single audit job. 614 distinct drain tx hashes against ~10K confirmed-contract rows, plus the full corpus (~3,437 lifetime drain events per CLAUDE.md priority #14). Blockscout `get_transaction_info` at 1/sec = ~1 hour minimum for the confirmed-tier subset, much longer for full corpus.

**Yield:** Directly fixes the Correction #24 Bug #19 mis-attribution. Re-flags the 7,206 phantom drain rows from the ratio-≥30 contracts. Recomputes the headline corpus number from ~10K events to whatever the actual successful-drain count is.

### Tier 5 — Sample-based manual review

**What:** Sample N contracts from each FP risk class. Inspect bytecode, deployer history, and any documented case files. Per-contract verdict.

**Cost:** ~30 minutes per contract. For 1,650 contracts at 100% review = 825 hours = 20 weeks at one full-time analyst. For a sample of 50 per class × 8 classes = 400 reviews = 200 hours = 5 weeks.

**Yield:** Ground-truth for the audit's confusion matrix. Necessary if the audit needs to report an FP rate with a confidence interval.

### Tier 6 — Full pipeline re-classification

**What:** Build a new classification pipeline that pre-checks each candidate against Tiers 1–3 BEFORE applying the confirmed label, then re-run the pipeline on the entire historical corpus.

**Cost:** Significant engineering work. Two to four weeks. The output IS the corpus going forward.

**Yield:** The new pipeline is the audit's permanent fix. Strongly recommended after Tiers 1–4 surface the FP rate and shape.

---

## Sampling strategy (if not auditing the full population)

If a full audit is infeasible, draw a stratified sample:

| Stratum | Population | Sample size | Why |
|---|---|---|---|
| Approval-active confirmed (139) | 139 | All — full audit | Small enough to fully audit. Highest signal because drain rows can be tx-status checked directly. |
| Self-loop / BACKFILL (158) | 158 | All — full audit | Small. The reason class is thin enough that en-masse downgrade may be the right action. |
| Long-vintage deployer (328) | 328 | 50 | Per Correction #20, this class is high-FP-risk for pristine-solo institutional deployers. |
| Recidivist deployers (1,107 contracts, 200ish deployers with ≥2) | ~200 deployers | 50 deployers, all their contracts | If the deployer is legit, all of its contracts are FPs together. Sample deployers, not contracts. |
| Behavioral-only, no approval, no recidivism (~700–900) | ~800 | 100 | The largest residual pool. Stratified random sample. |

Total sample size if going this route: ~500 contracts, ~125 hours of analyst time at 15 min/contract.

---

## Phased plan (recommended)

### Phase 0 — Pre-audit fix on the load-bearing bug

Before any auditing, fix Bug #19. The `approval_watchlist.drain_detected=1` rows are still being written today; every day this is unfixed adds more phantom rows. Fix is a one-line filter: only credit drain rows when the corresponding tx status is success.

The fix may itself recompute and downgrade ~7,000 phantom drain rows. That's the cheapest single audit action and resolves the most data-quality-relevant bug.

**Duration:** 1–2 days for engineering + verification.

### Phase A — Bulk external enrichment (Tiers 1 + 2 + 3)

Run all three cheap methodologies in parallel:

1. Blockscout `get_address_info` on every confirmed contract + every distinct deployer (~2,500 API calls).
2. Verified-source-code fetch for the verified subset.
3. CoinGecko / token-list cross-reference for the ERC-20 subset.

Output: a `confirmed_tier_audit_2026-05-XX.csv` with one row per confirmed contract, columns:
- contract_address, chain, current_tier, current_reason
- blockscout_verified (bool), token_name, token_symbol, holders_count, market_cap, primary_blockscout_tag
- deployer_blockscout_tag, deployer_mainnet_first_tx
- coingecko_listed (bool), coingecko_url
- preliminary_verdict: {LIKELY_FP, LIKELY_TP, NEEDS_REVIEW}

**Duration:** 1 day for the bulk fetch + 1 day to write the classifier rules + 1 day for manual review of the LIKELY_FP set.

**Estimated yield:** Likely catches 50–80% of the FPs at near-zero cost. The OFC anchor is caught at second 30. Phase A is the *minimum* audit.

### Phase B — Internal heuristics

Apply purely-internal FP heuristics to the residual (Phase-A-NEEDS_REVIEW) set:

- Drain/tx ratio ≥30 → flag as Bug #19 suspect.
- Self-loop / BACKFILL reason → flag as weak-evidence suspect.
- No bytecode_cache row + behavioral-confirmation-only → flag as Class-C-suspect.
- Recidivist deployer (≥2 confirmed) + no institutional label → flag as likely TP, do not downgrade.

**Duration:** 1–2 days.

### Phase C — Manual targeted review

Random sample within each Phase-B flag class. Verdict per contract: keep / downgrade. ~50 contracts × 30 min/contract = 25 hours.

**Duration:** 1 week part-time.

### Phase D — Migration script + corpus update

For every contract Phase A+B+C identified as FP:
- Move from `confirmed` to `unanalyzed` (the Correction #3 precedent — don't move to suspected, which still carries adversarial connotation).
- Annotate `confidence_reason` with the audit's verdict and date.
- Update INDEX.md entries.
- Recompute corpus headline statistics.
- Write a single Correction #25 documenting the audit's scope, methodology, and the contracts moved.

**Duration:** 1 week.

### Phase E — Permanent pipeline fix (the Tier-6 work)

Update the classification pipeline so the FP classes can never re-occur:

- Add a Blockscout-verified-source check before promoting a contract to confirmed via behavioral confirmation. Verified contracts go to `verified-pending-review` instead of `confirmed`.
- Add a deployer-OLI-label check before confirming via deployer_history. Institutionally-labeled deployers go to `institutional-pending-review`.
- Fix the `approval_watchlist` tx-status filter (Bug #19).
- Audit the OLI enrichment fetch path (Bug #22) so the `is_known_legitimate()` check has data.

**Duration:** 2–4 weeks.

---

## What "done" looks like

The audit succeeds when:

1. The headline confirmed-tier count carries an explicit FP-rate estimate with a Wilson 95% CI based on a stratified sample.
2. Every Class-A / Class-B FP confirmed in the audit has been moved to `unanalyzed` and documented in a numbered Correction.
3. The `approval_watchlist` Bug #19 fix is shipped and historical phantom rows are re-flagged.
4. The pipeline now refuses to confirm a Blockscout-verified contract or an OLI-labeled deployer's deployment without human review.
5. CLAUDE.md operational priorities #19–22 are resolved or moved to "fixed."
6. The corpus headline drain statistic is recomputed and published with explicit methodology.
7. The Camouflage Ratio analysis (Correction #22) is re-run on the post-audit corpus. The previously-reported 30.44% confirmed-tier figure may shift substantially.

---

## What's at stake if we don't audit

If the audit doesn't happen, the most likely failure modes:

1. **External-facing materials cite FP claims.** A pitch deck slide referencing the "0x752c5a95 4,587-victim discharge" or analogous future findings — the failure mode Correction #24 documented.
2. **The Camouflage Ratio confirmed-tier figure (30.44%) becomes the new 14.2× principle.** If the confirmed tier has even 30% Class-A/B FPs, the partition-based statistics are wrong by an unknown direction and magnitude.
3. **New confirmed labels keep being produced from the same three classifier paths.** The corpus accumulates more FPs daily until the pipeline is fixed.
4. **The credibility cost of a future external retraction.** Layer 3's positioning emphasizes "directionally correct, methodologically sound" — a discovered FP after deck publication is worse than a pre-publication audit-revealed FP.

---

## Audit cost summary

| Phase | Engineering effort | Analyst time | Calendar |
|---|---|---|---|
| Phase 0 — Bug #19 fix | 1 dev-day | — | 1–2 days |
| Phase A — Bulk external enrichment | 1 dev-day | 1 analyst-day for the LIKELY_FP review | 2–3 days |
| Phase B — Internal heuristics | 1 dev-day | — | 1–2 days |
| Phase C — Sample manual review | — | 25 analyst-hours | 1 week part-time |
| Phase D — Migration + corpus update | 2 dev-days | 1 analyst-day | 1 week |
| Phase E — Permanent pipeline fix | 1–2 dev-weeks | 1 analyst-day for QA | 2–4 weeks |

**Minimum viable audit (Phases 0 + A only): ~3 days of engineering, 1 analyst day. Catches Correction-#24-class FPs.**

**Full audit through Phase D: ~2 weeks. Resets the confirmed-tier corpus to defensible.**

**Full program through Phase E: ~6 weeks. Makes the FP class not re-occur.**

---

## Open design questions (for you to think about before deciding)

1. **Audit scope: confirmed-only, or confirmed+suspected together?** The suspected tier (118K contracts) is even larger and has its own FP risks. Auditing both is more thorough but proportionally more work. Recommended: confirmed first; suspected after the pipeline fix lands.

2. **Manual review depth: 15 min, 30 min, or 60 min per contract?** Determines the time/cost of Phase C.

3. **Downgrade target: `unanalyzed`, `suspected`, or a new tier like `verified-pending-review`?** A new tier is more transparent but adds schema complexity.

4. **Communication to existing customers: yes / no / how?** If any external materials (deck, pitch, report) cite confirmed-tier counts, the audit's outcome needs a customer-facing note. Drafting language is a separate task.

5. **Audit cadence: one-time, quarterly, monthly?** If quarterly, the Phase A bulk-enrichment becomes a recurring job. Easy to schedule; provides ongoing FP-rate tracking.

6. **Whose FP definition takes precedence when methodologies disagree?** E.g., what if Blockscout doesn't have a label but the deployer has 5 confirmed contracts and a Bug-#19-affected drain history. Recommended: tie-breaker is verified-source-code + token holders; if neither, defer to manual review.

7. **Public auditability: do we publish the audit dataset?** A published CSV of "contract → audit verdict → evidence" is a strong epistemic move. Recommended after Phase D.

---

## Adjacent audits the same machinery enables

The audit infrastructure built for the confirmed tier directly enables:

- **Suspected-tier audit** (118K contracts). Run the same Tier 1–3 enrichment; report FP rate per `confidence_reason` subclass.
- **Headline drain-count audit** (3,437 lifetime events). Apply the Phase-0 fix and re-flag phantom rows. Publish corrected number.
- **Camouflage Ratio re-computation** (Correction #22). Re-run after the confirmed-tier audit completes; report the post-audit partition.
- **Pattern D re-classification** (Correction #21). The cross-chain choreography detector's input population is now suspect for the same FP class; the Q-005 re-engineering work mentioned in Correction #21 should incorporate the Phase E pipeline fix.

---

**Next decision required (from you):**

1. Approve / reject / modify the phased plan.
2. Approve / reject / modify the sampling strategy.
3. Approve / reject / modify the FP-class taxonomy.
4. Allocate engineering / analyst time to Phase 0.

Once approved, the first concrete action is the Bug #19 tx-status filter fix. After that, Phase A's bulk Blockscout enrichment is the cheapest single information-gathering move and would either confirm or refute the "Correction-#24-class FPs are widespread" hypothesis within 1 day.
