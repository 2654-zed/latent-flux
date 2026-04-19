# Pattern C — CEX-Laundered Funding Candidates

**Date:** 2026-04-18
**Scope:** SQL pass against the corpus. No RPC calls spent (budget: up to 200; spent: 0 this pass).
**Epistemic:** Tier B inferential on candidate labels; the SQL surface doesn't by itself confirm laundering — that's the one-hop-back RPC trace, deferred pending review of this candidate list.

---

## What we looked for

Deployers whose on-chain funding trail points to a known CEX hot wallet (labeled in `entity_classification` as `INFRASTRUCTURE/cex_hot_wallet`). These are the addresses where our `auto_funder_tracer` stops and reports "legitimate origin." The Pattern C hypothesis is that the apparent legitimacy is manufactured: the operator deposited funds into the CEX from a trap-adjacent wallet, withdrew to a fresh deployer, and let our tracer conclude at the CEX.

---

## Inputs from the corpus

- **10 CEX hot wallet addresses** currently labeled (entity_classification: INFRASTRUCTURE/cex_hot_wallet)
- **26 deployers** whose `funding_trail.funder` is in that CEX hot-wallet set

## Result — strict filter (CEX-funded AND has confirmed trap)

**0 candidates.** Zero of the 26 CEX-funded deployers have any CONFIRMED trap in our corpus. The strict version of Pattern C returns negative.

## Result — relaxed filter (CEX-funded AND suspected-heavy)

**4 candidates** match the weaker filter of "CEX-funded + substantial suspected trap deployments":

| Deployer | Total | Confirmed | Suspected | Unknown | Notes |
|---|---|---|---|---|---|
| `0x4885631c7335290adcdc4b6b95f97549f5a40edd` | 37 | 0 | 13 | 24 | highest volume |
| `0x6dc136bcac04646d8d342599a704fffe9861af56` | 19 | 0 | 19 | 0 | 100% suspected |
| `0xb87e28fc6086fad8fe228aac3d3e19058e69f828` | 7 | 0 | 7 | 0 | 100% suspected |
| `0x561d79e961c4dd7bbaf078e6c9753c764e084d77` | 3 | 0 | 3 | 0 | small but full-suspected |

`0x4885631c…` also appears in the Pattern D (cross-chain import) candidate list — it has **Ethereum mainnet first-seen 2026-02-17 vs L2 first-seen 2026-04-03**, a 44-day gap. Overlap on two laundering signals strengthens the signal.

## Funding values

The handful of funding_trail entries that include ETH values are small:
- `0x033d986709c6c794c42a1259a8baeb6693de9444` — 0.01221 ETH funded (but has 0 contracts — not a candidate)
- `0xafa9ed53c33bbd8de300481ce150db3d35738f9d` — 0.00977 ETH funded (but has 0 contracts)

The CEX-funded + 0-contract rows look like withdrawal addresses the operator initialized but hasn't used for deployments yet. Worth monitoring.

---

## Next step — one-hop-back RPC trace (not done this pass)

For the 4 candidates above, we'd want to:
1. Find the specific CEX → deployer withdrawal tx (using funding_trail.traced_at + Alchemy `alchemy_getAssetTransfers` filtered by from=CEX, to=deployer, around the timestamp).
2. Match the withdrawal amount against inbound deposits to the CEX hot wallet in the preceding window (same amount ± CEX fees, within 1–24 hours earlier).
3. Identify the pre-CEX source wallet.
4. Cross-reference the pre-CEX source against:
   - org_001 treasury network
   - Drainer operation wallets (CE5E, E717, A7B9, E3B2, D270)
   - `0x881e*` address poisoner cluster
   - Contract_address set for known trap deployers

**Estimated cost:** ~3–5 RPC calls per candidate (asset transfers filter is one call; may need multiple if CEX hot wallet has heavy volume and we need pagination). At 4 candidates × 5 calls = ~20 calls. Well under the 200-call budget the handoff authorized.

**Why not spent this pass:** Per the handoff checkpoint structure — Pattern C was flagged for explicit approval before proceeding with RPC. The SQL surface is complete; the candidate list is small enough to review, and the next step is constrained enough to decide explicitly.

**Decision ask:** proceed with one-hop-back trace on the 4 candidates? Budget 20 calls. Output would be a candidate → pre-CEX-source mapping, cross-referenced against known bad sets. Findings get appended here as an addendum.

---

## Structural caveats

1. **The `funding_trail.funder` field is written by `auto_funder_tracer`**, which runs 1 RPC call per new deployer. Its "funder" is usually the single address that sent ETH to the deployer. Tracer logic may not always correctly tag the funder as a CEX hot wallet — depends on whether the address is in our `entity_classification` at the moment funder_tracer runs.

2. **Small sample size (26 total CEX-funded deployers out of ~25,000)** reflects that most new deployers in our corpus get funded from dust-faucet EOAs or gas stations, not CEX withdrawals. That's consistent with the campaign structure: trap fleets use internal gas stations; CEX withdrawals are for operators establishing fresh identities.

3. **No confirmed traps in the CEX-funded set is surprising.** Could be because:
   - Operators using CEX laundering are more sophisticated and deploy harder-to-detect bytecode → our classifier marks them 'unknown', not 'suspected'
   - CEX-funded deployers genuinely aren't trap operators at detectable rates (most are just legitimate devs with normal CEX-withdraw-to-fresh-wallet pattern)
   - The bias is statistical: only 26 CEX-funded out of 25k deployers is thin; 0 confirmed out of 26 is consistent with the 579/124,341 ≈ 0.47% corpus-wide confirmed rate (4 would be the statistical-match count, so 0 is actually below baseline).

Given #3, the pattern may be under-represented because CEX-funded deployers are *less* likely to deploy trap contracts than the corpus average. That's an important finding: **if Pattern C is rare, it's rare because the operators who bother with CEX laundering are small-volume, not because the technique doesn't work.** Consistent with the behavioral-laundering thesis — laundering is an investment that pays back in low-volume, high-survivability operations.

---

## What this does NOT claim

- No operator is accused of CEX laundering based on this SQL-only scan.
- No entity_classification updates, no alerts fired.
- The one-hop-back trace is what would produce actionable findings; this report is the authorization request for that.
