# Bipartite SBM on Funder→Deployer Graph

**Date:** 2026-05-19
**Module:** `surveillance/analytics/sbm_funder_deployer.py`
**Method:** Spectral biclustering (SVD-based, Dhillon normalization) → k-means on shared SVD-embedding → permutation test for confirmed-tier enrichment per block.

## Question

Are there latent organizational structures in the funder→deployer bipartite graph that the existing per-funder `auto_funder_tracer` + `entity_classification` pipeline does not surface? Specifically: do funder *communities* exist whose deployers are over-enriched for confirmed-tier predator contracts beyond what individual funder degree predicts?

## Setup

- Edges: `(funder, deployer)` from `deployers.funding_trail.funder`, kept where the funder appears at least 5 times. 305 funders survived the threshold, mapping 42,387 deployers via 42,387 edges.
- Spectral biclustering with K = 8 blocks.
- Permutation test (n = 1000) per block, two-sided, Bonferroni-adjusted alpha = 0.05/8 = 0.00625.
- Corpus baseline confirmed-tier rate: **0.49%** (208 of 42,387 deployers have at least one confirmed-tier contract).

## Result

| Cluster | N funders | N deployers | Confirmed | Rate | Enrichment | p-perm |
|---|---|---|---|---|---|---|
| 0 | 2 | 7,516 | 56 | 0.75% | **1.52×** | <0.001 *** |
| 1 | 66 | 3,220 | 19 | 0.59% | 1.20× | 0.433 |
| 2 | 55 | 6,372 | 17 | 0.27% | 0.54× | 0.011 * |
| **3** | **126** | **4,432** | **75** | **1.69%** | **3.45×** | **<0.001 *** |
| 4 | 48 | 7,603 | 35 | 0.46% | 0.94× | 0.713 |
| **5** | **1** | **8,507** | **0** | **0.00%** | **0.00×** | **<0.001 *** |
| 6 | 5 | 2,812 | 6 | 0.21% | 0.43× | 0.037 * |
| **7** | **2** | **1,925** | **0** | **0.00%** | **0.00×** | **0.005 *** |

Four clusters pass the Bonferroni threshold: clusters 0, 3, 5, and 7. Two are predator-enriched (0 and 3). Two are predator-free infrastructure (5 and 7).

## Headline finding — Cluster 3 (predator cooperative)

- **126 funders, 4,432 deployers, 3.45× baseline confirmed-tier rate, p < 0.001.**
- Top funders by within-block degree:
  - `0xb451e56ebdca2f672178f4219b170e6bdb2970b3` — 498 deployers
  - `0xbaed383ede0e5d9d72430661f3285daa77e9439f` — 378 deployers ← **this is org_004**
  - `0xe4edb277e41dc89ab076a1f049f4a3efa700bce8` — 358 deployers
  - `0x5e809a85aa182a9921edd10a4163745bb3e36284` — 176 deployers
  - `0x2cff890f0378a11913b6129b2e97417a2c302680` — 125 deployers
- Chain mix: Base 3,783, Arbitrum 311, Optimism 141, unknown 197.
- **Framing — Extends.** CLAUDE.md operational priority #12 names org_004 (`0xbaed383e`) as the next organizational mapping target. The current per-funder mapping attributes 378 deployers to it. The SBM places it in a community of 126 co-clustered funders, 4,432 deployers, and 3.45× baseline confirmed-tier enrichment. This is not "org_004 has 4,432 deployers" — it is "org_004 sits inside a funder cooperative whose collective behavioral signature exceeds what its individual attribution captures." The four other named funders above are the immediate co-cluster anchors to investigate.

## Headline finding — Cluster 0 (smaller predator cluster)

- **2 funders, 7,516 deployers, 1.52× baseline, p < 0.001.**
- `0xf70da97812cb96acdf810712aa562db8dfa3dbef` — 4,346 deployers
- `0x3304e22ddaa22bcdc5fca2269b418046ae7b566a` — 3,170 deployers
- Chain mix: Base 6,805 (90.5%), Arbitrum 422, Optimism 129, unknown 160.
- These two funders together account for ~18% of the bipartite-graph deployers. The 1.52× enrichment is modest but significant given the cluster size. **Investigation target.**

## Headline finding — Cluster 5 (clean-infrastructure factory)

- **1 funder, 8,507 deployers, 0% confirmed-tier, p < 0.001 vs baseline.**
- `0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07` — entire cluster.
- Chain mix: **8,500 Optimism**, 7 unknown. No Base. No Arbitrum.
- This is the known b0b0b690 factory pattern. The SBM independently confirms it as a single-funder mono-chain infrastructure cluster with zero predator content. **Discount.** Any corpus-wide statistic that aggregates over this cluster will be biased toward the clean-Optimism-factory mode.

## Headline finding — Cluster 7 (smaller clean-infrastructure factory)

- **2 funders, 1,925 deployers, 0% confirmed-tier, p = 0.005.**
- `0x0e6e91775d24d34b90e0f3d806a90705f0199999` — 1,602 deployers
- `0x623777cc098c6058a46cf7530f45150ff6a8459d` — 323 deployers
- Chain mix: Base 1,766 (91.7%), unknown 158, Optimism 1.
- Second clean-infrastructure cluster, this time on Base. Different mechanism from b0b0b690 (multi-funder, smaller scale).

## What the SBM does NOT find

- No K=8 block partitions the corpus into chain-pure predator communities. Cluster 3's predator-enriched community is multi-chain (Base-dominant but spans Arbitrum + Optimism).
- No block recovers the previously-mapped org_001 / org_002 / org_003 cleanly. Either those organizations span multiple SBM blocks (likely, given their multi-funder construction) or the K=8 partition is too coarse to separate them. Re-running at K=16 is the natural next step.
- The two "anti-predator" clusters (2 and 6) are below baseline but not Bonferroni-significant. They likely represent legitimate-DeFi funder clusters (clusters where the deployer behavior is consistently clean) but the result is not actionable at this K.

## Methodological notes

- The bipartite graph is extremely sparse: every funder edges to many deployers, but each deployer has exactly one funder (their primary funding-trail source). The SVD-based biclustering effectively groups funders that fund similar populations of deployers and groups deployers by which funder community they belong to. The block structure that emerges is therefore close to a partition of funders, with deployers riding along their primary funder's block label.
- Permutation null is constructed by sampling `n_block` deployers uniformly without replacement from the corpus and computing the resampled confirmed-tier rate. n = 1000 iterations. The p-values for the four significant blocks all show observed enrichment far outside the null distribution.
- Random seed: 20260519 (`np.random.default_rng`). K-means initialization: "++" (Arthur-Vassilvitskii) with seed 20260519. Reproducible.

## Recommended follow-up

1. **Investigate Cluster 3's top 5 funders** (`0xb451e56e…`, `0xbaed383e` = org_004, `0xe4edb277…`, `0x5e809a85…`, `0x2cff890f…`) for shared infrastructure: do they fund overlapping deployer subsets, share funder origins, or coordinate temporally? If yes, the cluster is best modeled as a cooperative; if no, the SBM has aggregated structurally unrelated funders whose individual deployer populations happen to be predator-enriched (likely a chain-of-funder feedback effect).
2. **Investigate Cluster 0's two funders** (`0xf70da978…`, `0x3304e22d…`) for the same questions.
3. **Re-run SBM at K = 16** to test whether org_001, org_002, org_003 emerge as separable blocks at finer granularity.
4. **Cross-check Cluster 5 (b0b0b690) against `entity_classification`** to verify it is already labeled as known-clean infrastructure. If not, this is a labeling gap.
5. **The 0.49% corpus baseline** assumes the contracts table is the right population. If we re-run with funder→deployer→drained-event projection instead of confirmed-tier, the enrichment numbers will shift — that test is the natural extension to validate Cluster 3 as a drain-completion community (not just a label-correlation community).
