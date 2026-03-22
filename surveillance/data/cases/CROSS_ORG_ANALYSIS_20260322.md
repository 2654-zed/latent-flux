# CROSS-ORG INTERFERENCE ANALYSIS
**Generated:** 2026-03-22 23:20 UTC
**Organizations:** org_001 (DELEGATECALL network) vs org_002 (tx.origin campaign)

---

## Relationship Assessment
**Classification:** INDEPENDENT
**Confidence:** HIGH — zero overlap across all 5 signal types

---

## Wallet Overlap
**Shared wallets: 0**

org_001 has 11 known wallets (treasury, operators, cashout, exit ramps, laundry).
org_002 has 357 known wallets (2 treasuries + 355 disposable deployers).

Zero addresses appear in both organizations. No shared infrastructure at the wallet level.

---

## Direct Interactions
| Direction | Interactions |
|---|---|
| org_001 wallets -> org_002 contracts | **0** |
| org_002 wallets -> org_001 contracts | **0** |

Neither org has ever interacted with the other's contracts. No probing, no sabotage, no testing. They are invisible to each other at the contract interaction layer.

---

## Victim Pool Overlap
| Metric | Count |
|---|---|
| org_001 unique victims | 1 |
| org_002 unique victims | 1,660 |
| Shared victims | **0** |
| Overlap | **0.0%** |

Zero shared victims. This is partly structural — org_001 has minimal tracked victim interactions in the DB (most extraction was traced via Alchemy, not captured by the tx_events monitor), while org_002's 1,660 victims come from the tx.origin contracts' on-chain interactions. But even accounting for coverage differences, the populations don't intersect.

---

## Deployment Timing Correlation
| Metric | Value |
|---|---|
| Hours with org_001 deployments | 13 |
| Hours with org_002 deployments | 79 |
| Hours with BOTH deploying | 8 |

Overlapping deployment hours (both active simultaneously):

| Hour | org_001 | org_002 |
|---|---|---|
| Mar 19 21:00 | 5 | 5 |
| Mar 19 23:00 | 32 | 5 |
| Mar 21 19:00 | 1 | 4 |
| Mar 21 21:00 | 7 | 4 |
| Mar 22 02:00 | 20 | 4 |
| Mar 22 03:00 | 3 | 4 |
| Mar 22 05:00 | 4 | 4 |
| Mar 22 06:00 | 3 | 5 |

**No reactive pattern.** org_002 maintains a steady 4-5 deploys/hour regardless of org_001's activity. org_001 has bursts (32 in one hour) that don't affect org_002's cadence. They deploy on independent schedules with no correlation.

---

## Chain Separation
| Chain | org_001 | org_002 |
|---|---|---|
| Arbitrum | 45 contracts | 4 contracts |
| Base | 52 contracts | **363 contracts** |

org_001 operates on both chains roughly equally. org_002 is **98.9% Base-only**. The 4 Arbitrum contracts are likely noise or early experiments. The orgs have minimal geographic overlap — org_001's Arbitrum operations and org_002's Base operations occupy different ecosystems entirely.

---

## Funding Source Analysis
**Shared funding sources: UNKNOWN**

org_001's treasury (`0xf186cb`) and org_002's treasuries (`0x238d71`, `0xde8eb9`) were traced via Alchemy API. No shared upstream funders were found in any of the hop-tracing investigations. org_001's gas station (`0x8c826f`) serves 6,375+ addresses but does NOT appear in org_002's funding chain. org_002 uses its own treasury wallets with nonces of 9,283 and 15,486 — established infrastructure independent of org_001.

---

## Competitive Impact Assessment

org_002 scaled from 43 to 367 contracts between March 20-22, overlapping with org_001's continuous cashout operations. During this period:

- org_001's cashout pipeline continued operating normally (EXTRACTION_001 and _002 documented)
- org_001's revert rate on its contracts didn't change
- org_001's deployment cadence was unaffected

org_002's entry into the Base market increased the overall trap density but did not visibly impact org_001's operations. They fish in different pools: org_001 uses DELEGATECALL upgradeable traps that attract MEV bots via DEX routing, while org_002 uses tx.origin buy-not-sell traps that catch direct token traders. Different mechanisms, different victim profiles.

---

## Assessment

**These are independent operations run by different people, using different techniques, on mostly different chains, targeting different victim populations.**

Evidence:
1. Zero wallet overlap (0/368 addresses shared)
2. Zero cross-contract interactions (neither has ever touched the other's contracts)
3. Zero victim overlap (0 shared addresses in victim pools)
4. No deployment timing correlation (independent schedules)
5. No funding source overlap (different treasury infrastructure)
6. Different chains (org_001 dual-chain, org_002 Base-only)
7. Different trap mechanics (DELEGATECALL vs tx.origin)
8. Different operational patterns (org_001 irregular bursts, org_002 continuous marathon)
9. Different opsec (org_001 reuses operators, org_002 uses disposable deployers)
10. Different funding (org_001 uses USDC via treasury, org_002 uses ETH via gas station pattern)

**What would change this assessment:**
- Discovery of a shared funding source 2+ hops upstream
- A wallet appearing in both org structures
- One org's contracts referencing the other's addresses in bytecode
- A shared operational tool (same bot framework, same deployment script signature)

None of these signals are present. Classification: **INDEPENDENT** with **HIGH** confidence.
