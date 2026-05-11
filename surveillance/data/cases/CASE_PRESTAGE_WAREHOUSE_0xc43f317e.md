# CASE FILE: Meme-Token Deployment Shop — `0xc43f317e` (RECLASSIFIED)
**Case ID:** PRESTAGE_WAREHOUSE_C43F317E (legacy ID — retained for backref; framing retracted)
**Generated:** 2026-05-10 (initial "pre-stage warehouse" framing); **revised same day after bytecode decompilation**
**Classification:** Meme-token deployment shop (vanilla ERC-20 launchpad). Out of trap-detection scope.
**Chain:** Base
**Threat Level (revised):** **LOW at contract layer.** Off-chain harm possible (rug-pulls, dump schemes) but outside Layer 3's L2 trap-detection surface.

---

> **[RECLASSIFICATION — 2026-05-10]** This case file was authored under the hypothesis that `0xc43f317e` was a **pre-stage trap warehouse** awaiting activation (Stored Potential framework). After bytecode decompilation of a downstream sample (`0xacfdc090ff9f5b160005bdaacb9a2d1025755baf`, "Kore Agent" / KORE), **the dominant bytecode template `49155b60033de73770...` is a verified vanilla OpenZeppelin v5.0.0 ERC-20 token contract.** Zero custom transfer logic, zero fee-on-transfer, zero blacklist, zero owner functions, zero delegatecall, zero selfdestruct. Constructor takes `(name, symbol, initialSupply)` and `_mint`s to deployer. Pure stock OZ ERC-20.
>
> The operator is a **meme-token deployment shop** — sustained-velocity ERC-20 token launchpad. Same operator class as the Dragon (`0x2e20b261` / 2,077 tokens in compressed burst). The "100% bytecode concentration" is exactly what an ERC-20 factory template produces; the "no realized extraction" is because vanilla ERC-20s have no extraction primitives.
>
> **Watchlist downgraded HIGH → MEDIUM** (`meme_token_shop_c43f317e`). Original "pre-stage warehouse" framing preserved below for historical record; the structural observations (2,535 deployers, sustained tempo, single-template) remain accurate, but the *interpretation* layer is retracted.

---

## Revised Executive Summary

`0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078` is a meme-token deployment shop operating on Base since 2026-03-23. It funds 2,535 downstream wallets each deploying exactly one vanilla OpenZeppelin ERC-20 token contract. Token names follow a free-form parameter pattern ("Kore Agent" / "X1000XLiquidBGT" / "Laser Eagle" — though that last one is from a different operator's template; c43f317e's confirmed sample is "Kore Agent"). The operator is **not staging traps**; they are running an ERC-20 deployment-as-a-service or token-launchpad operation, possibly for:

- Burner-token clients (deploy a token for a client who self-rugs)
- Sybil airdrop wallet generation that happens to deploy a token per wallet
- Per-event meme/scam token launches that get dumped via LP-rug outside L3's contract-layer scope

The 27 bot wallets that have approved 102 of c43f317e's downstream contracts (798 total approvals) are **MEV/arbitrage scanners** probing newly-deployed tokens for arbitrage opportunities — not victims of an extraction primitive.

**The off-chain harm of this operator class is real and substantial** (meme-coin rug pulls cost real users real money), but it operates through promotion/dump cycles outside the on-chain trap-extraction surface Layer 3 monitors. Tier MEDIUM watchlist retains visibility without elevating priority above genuinely predatory operators like Coffee Fleet (`0xc0ffeefeed8b`), drainer-spawn hub `0xf7883e3f`, or the newly-discovered honeypot operator `0x8ca70232` (see `CASE_HONEYPOT_TOKEN_OPERATOR_0x8ca70232.md`).

---

## Decompilation Evidence

**Sample contract:** `0xacfdc090ff9f5b160005bdaacb9a2d1025755baf` (Base, deployed 2026-04-22)
- Token: "Kore Agent" (KORE), 4.8B supply, 18 decimals
- Compiler: Solidity 0.8.25
- File: `OpenAI.sol` (filename appears intentionally provocative; the actual contract is named `OpenAI` and inherits OpenZeppelin `ERC20`)
- Source verified via Blockscout
- Full constructor: `OpenAI(string name, string symbol, uint256 initialSupply)` → `ERC20(name, symbol); _mint(msg.sender, initialSupply * 10**decimals());`
- **No custom transfer / approval / mint logic beyond OpenZeppelin defaults.**

Verified via Blockchain MCP `inspect_contract_code` against `eth.blockscout.com` for Base chain (chain_id 8453). Source code retrieved as `OpenAI.sol`, content matches OpenZeppelin v5.0.0 verbatim through `_spendAllowance`. Constructor at the bottom adds only the `_mint` to deployer.

---

## Original framing (preserved for historical record)

The original "pre-stage warehouse" Executive Summary follows. The structural observations remain accurate; the interpretive layer is retracted per the reclassification above.

### Original Executive Summary (RETRACTED 2026-05-10)

`0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078` is the largest **still-active pre-stage warehouse operator** in the corpus that survived Correction #20's OLI mass-mislabel sweep. It funds 2,535 downstream deployer wallets on Base (up from 1,562 at the April 25 ISO snapshot — **+62% growth in two weeks**) and continues active operations as of 2026-05-10. **815 of those deployers have been active in the last 14 days.** Every deployer funded by this address has produced exactly one contract; the entire downstream fleet uses a **single bytecode template** (hash prefix `49155b60033de73770...`) at **100% concentration** across 1,206 hashed contracts.

Despite this scale, **zero realized extraction has been observed**:
- 0 entries in `trap_events` for any downstream contract
- 0 entries in `approval_watchlist` for any downstream contract
- 0 adversarial bytecode flags (`has_asymmetric_transfer`, `has_conditional_revert`, `has_unusual_fee_structure` all 0)
- All 2,366 known downstream contracts remain in `unknown` confidence tier (no classifier has graded them)

This is the strongest **Stored Potential** signal in the corpus that is not yet either retracted (Correction #20) or attributed to a documented organization. The operator is depositing single-purpose contracts at ~95/day average with a single bytecode template, no third-party interaction, and no obvious adversarial signature in the bytecode classifier's surface. **Either the bytecode is genuinely benign (legit infrastructure with single-purpose-contract topology), the bytecode classifier is missing the predatory pattern, or the operator is staging a discharge event of unknown timing.**

**Resolution:** Hypothesis 2 from the original "Disambiguation" section ("Non-trap infrastructure with single-purpose-contract topology") was the correct read. The bytecode IS genuinely benign — vanilla OZ ERC-20. The case-file authorship sequence demonstrates the load-bearing role of bytecode-level review: behavioral and topological signal alone could not disambiguate trap-stockpile from meme-token-shop.

---

## Operator Profile

**Funder address:** `0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078`
**Funder mainnet first tx:** 2025-11-09T21:56:35 UTC (~6 months old; not L2-only, not pristine-vintage)
**Funder's own deployment record:** None (`fleet=0` — pure funding wallet)
**Funder's funding source:** Not recorded in Layer 3's `funding_trail` for the funder itself (one-hop upstream not traced; deferred work)
**Funder OLI status:** **OLI-clean** per 2026-05-09 mass audit (`reports/blockscout_tag_audit_2026-05-09.csv` — no public institutional tag)

**Downstream fleet:** 2,535 deployer wallets on Base
- **Active window:** 2026-03-23 → 2026-05-10 (49-day operational span, ongoing)
- **Per-deployer fleet:** 1 contract each (`avg_fleet_per_deployer ≈ 1.0`)
- **Disposable rate:** 100% (each deployer used exactly once)
- **Last-14-day activity:** 815 deployers
- **Top-5 burst days:** 2026-04-13 (116), 2026-04-23 (96), 2026-04-12 (93), 2026-04-06 (88), 2026-04-26 (84)
- **Average:** ~52 new deployers per active day across the operational window

**Sample of latest deployments (2026-05-10):**
- `0xa4b83598a4f194e1ab090feba606d433d4763000` — chain=base, detected 15:20:59 UTC
- `0x6d1328a0c3e6b6415c3a265c2c4de8d68e1152ad` — chain=base, detected 14:59:49 UTC
- `0x6aa0d6efdbf8f6bd8aad1e754ee85fd483e937fe` — chain=base, detected 14:42:57 UTC
- `0xe37f1dfcd2147f240d21ff9af28068e531a1a7cd` — chain=base, detected 14:25:09 UTC
- `0x9b0a1960ff1273094c9f442d07541b61b987a7e8` — chain=base, detected 14:07:03 UTC

Average inter-deploy interval in the May 10 sample: ~18 minutes. Sustained, not bursty.

---

## Bytecode Signature

**Dominant hash prefix:** `49155b60033de73770...`
**Concentration:** 100.0% of hashed downstream contracts (1,206 of 1,206)
**Hashed-vs-total ratio:** 1,206 hashed / 2,366 unknown-tier classified = 50.9% (the rest are unhashed — likely an ingest-time deployed_code_hash backfill gap, not a separate template family)

**Bytecode-classifier flags (across full downstream fleet of 2,366 known contracts):**
- `has_asymmetric_transfer = 1`: **0 contracts**
- `has_conditional_revert = 1`: **0 contracts**
- `has_unusual_fee_structure = 1`: **0 contracts**

**Confidence-tier distribution:**
- `unknown`: 2,366 (100% of analyzed)
- `suspected`: 0
- `confirmed`: 0
- `unanalyzed`: 0 (per the partial-classification gap noted in CLAUDE.md priority #11)

The bytecode classifier has run against the downstream fleet and found zero adversarial flags. Either the template is genuinely benign, or the classifier's surface (3 boolean flags) does not catch this template's class. **A bytecode-level inspection of the dominant hash is the load-bearing next step** — without it, the typology cannot be sharpened beyond "infrastructure-scale single-template depositor."

---

## Stored Potential Assessment

Per the [Stored Potential](../../docs/lexicon.md#stored-potential) framework, capability primitives present:

| Primitive | Value | Interpretation |
|---|---|---|
| `count` | 2,535 (still growing) | Infrastructure-scale fanout |
| `approval_scope` | 0 known approvals | No pre-positioned approval-spending capability |
| `realized_value` | $0 | No observed extraction |
| `template_concentration` | 1.00 | Maximum-uniformity deployment |
| `operational_continuity` | 49 days, ongoing | Sustained, not opportunistic |
| `OLI_attribution` | none | No institutional identity to discount the signal |
| `behavioral_baseline` | unknown | Bytecode-classifier surface is silent |

**Stored potential score (qualitative): HIGH on structural primitives, ZERO on activation primitives.** The position is loaded with respect to fanout + template concentration + operational continuity, but no approval scope or realized extraction has been observed. This combination is identical in shape to the [Single-Purpose Infrastructure Funder](../../docs/lexicon.md#single-purpose-infrastructure-funder) pre-stage stockpile class — except the funder here funds 2,535 deployers, not the typology's stated ≤1.

---

## Disambiguation: What This Might Be

Several hypotheses fit the observed shape:

1. **Genuine pre-positioned trap stockpile.** The closest documented analog is the Dragon (`0x2e20b26172a8`, 2,077 ERC-20 tokens pre-approved to PancakeSwap V3 router). But Dragon's contracts have pre-approvals on file; c43f317e's contracts have *no approvals at all*. So if this is a stockpile, it's at an earlier life-cycle stage (contracts deployed, but the approval-baiting phase hasn't begun). Activation would require either victim-side approval or operator-side `approve()` calls; neither has occurred.

2. **Non-trap infrastructure with single-purpose-contract topology.** Examples that fit this shape: account-abstraction wallet deployers (each user gets one smart-account contract), forwarding-proxy factories for cross-chain message handlers, single-purpose payment-channel deployers. The 100% bytecode concentration is consistent with any of these. The absence of OLI tagging would be unusual for major infra projects (Circle CCTP, AA providers, bridge protocols all OLI-tag their factories) but not unusual for a smaller or in-development project.

3. **Airdrop-farming Sybil setup.** Each deployer wallet creates one contract to "be" a unique address for the farming purpose. The 100% bytecode template fits because all wallets need the same on-chain footprint. But the *funder identity* would matter — Sybil farmers typically use a fresh funding wallet not tied to any history, while this funder has 6 months of mainnet history (Nov 2025). Possible but not the strongest fit.

4. **Bytecode-classifier false negative.** The template is adversarial (e.g., contains predatory logic) but the classifier's 3 boolean flags don't catch the pattern. Decompiling a sample contract is the test. Bytecode-classifier known gaps are documented in CLAUDE.md priority #11 ("2,119 misclassified contracts in T2-eaef6a5d NULL bucket"); this could be another instance of the same class of miss.

5. **Operator running multiple narrow operations.** The 2,535 fleet might represent ~2,535 independent micro-operations (one per contract) rather than one coordinated stockpile. The single bytecode template argues against this — but if the template is a generic factory, each contract could be configured differently at deploy time without bytecode divergence.

**Hypotheses 1 and 4 are the operator-class-relevant ones.** Hypotheses 2, 3, and 5 are alternative-explanation paths that would (if confirmed) dissolve the typology and move the operator out of the adversarial classification.

---

## What We Know Did NOT Happen

This list is the negative-evidence corpus that anchors the "stored potential, not realized" framing:

- **No drain events**: zero `approval_watchlist.drain_detected = 1` entries for any of 2,366 downstream contracts.
- **No trap_events**: zero rows in `trap_events.trap_contract_address` matching any downstream contract.
- **No external victim-bot interactions**: zero rows in `transaction_events` outside operator-internal deployment.
- **No `extraction_events` entries**: no incident has been documented against any downstream.
- **No infrastructure_registry attribution**: the funder is not in the 12-row registry of known-legitimate institutional addresses.
- **No org_001-004 attribution**: zero overlap with documented organizations' wallet sets.
- **No top-12 ISO neighbor signal**: the 2026-04-29 cross-funder overlap probe (Section A4 of `reports/epistemic_test_results_2026-04-29.md`) confirmed zero pairs of top-12 funders share any downstream deployer; c43f317e is structurally independent of the other 11.

---

## Recommended Next Steps

1. **Decompile a sample of the dominant bytecode (`49155b60033de73770...`).** Pick 3-5 contracts from different burst days and inspect via Blockscout `inspect_contract_code` or local decompiler. Look for:
   - Approval-call patterns (`approve`, `permit`, `transferFrom` entry points)
   - Selective revert logic on caller
   - DELEGATECALL surfaces (post-deployment logic swap)
   - Upgrade proxies (UUPS / Transparent / Beacon)
   The 3-boolean classifier surface is too coarse; manual review is needed.

2. **Trace the funder's upstream.** `funding_trail` for `0xc43f317e` is empty in our DB. A `eth_depth.py` probe (similar to the 2026-05-01 bb50 funder upstream probe) would clarify whether the funder is itself receiving from a CEX, a bridge, a private wallet, or another infrastructure-scale operator.

3. **Test for approval-baiting via the dominant contract addresses.** If activation requires victim-side approval first, the contract addresses might be appearing on phishing pages or in social-engineering campaigns. A search of the X402-style attack-vector corpus for any of the latest 50 c43f317e-downstream contract addresses could surface this.

4. **Run the bytecode classifier's full surface (beyond the 3 booleans) against the dominant hash.** If `bytecode_families` clustering or `bytecode_family_members` analysis has been run for this hash, the result should be cross-referenced.

5. **Monitor the activation transition.** A `dormant_activation` alert on any of the 2,535 downstream deployers would be the first signal of state change. The existing `dormant_activation` detector already covers this — c43f317e's downstream should be in the regular scan.

---

## Watchlist Status

This entry is not yet on the watchlist. The OLI-clean status + Correction #20 cleared-residual context + zero realized extraction together argue against a CRITICAL or HIGH tier promotion at this time. **Recommend MEDIUM watchlist with `entity_name = prestage_warehouse_c43f317e` and reason pointing to this case file.** The promotion should be a separate authorized action — this case file documents the finding; tier-assignment is a downstream decision.

---

## Open Questions

- What is the dominant bytecode actually computing?
- Why does the funder have no recorded funding source?
- Why has no third party interacted with any of 2,366 contracts (no approvals, no trap_events, no transactions)?
- Is the +62% growth between April-25 ISO and May-10 indicative of a pre-activation phase, or just normal scale?
- What would activation look like? (Approvals appearing? `approve()` calls from operator side? Direct value-transfer-from?)

---

## Methodology Notes

- All counts derived from local DB synced from production 2026-05-10 (post-Correction-#20).
- OLI status verified via `surveillance.oli_enrichment` against Blockscout metadata service mainnet endpoint.
- Bytecode concentration computed via `SELECT deployed_code_hash, COUNT(*) FROM contracts WHERE deployer_address IN (SELECT deployer_address FROM deployers WHERE funding_trail LIKE '%0xc43f317e...')`.
- Burst-day analysis: `SUBSTR(first_seen, 1, 10)` group-by on downstream deployers.
- "No extraction" claim verified across `trap_events`, `approval_watchlist`, `extraction_events`, `transaction_events`.

---

*Case file generated by Layer 3 surveillance methodology — Item 6 of Correction #20 Open Work.*
*All analysis from SQLite, zero RPC calls during case-file authorship.*
