# CASE FILE: ORG_001 — Complete Infrastructure Map
**Case ID:** ORG_001
**Classification:** DELEGATECALL Trap Network (Arbitrum + Base)
**Last Updated:** 2026-04-11
**Status:** ACTIVE — infrastructure expanding, shadow wallets discovered
**Compiled from:** fund_tracer.py, org_cycles.py, trace_exit_ramp.py, trace_laundry_new.py, entity_classification DB, watchlist DB, deployer_profiles DB, eth_traces DB, CASE_ORG_001_ETHEREUM_DEPTH.md, FUND_FLOW_TRACE reports

---

## Executive Summary

org_001 is a professional, multi-chain trap operation running honeypot contracts on Arbitrum and Base. The operation traces to two CEX origins (Coinbase, Binance), employs 559+ deployers, has deployed 7,400+ contracts, and victimized 3,400+ bots. Total identified assets exceed **$80M** when including the Binance origin branch (33,333 ETH withdrawal).

On 2026-04-11, two **vanity-spoofed shadow wallets** were discovered during a data gap investigation. These wallets use address prefix collision to impersonate existing org_001 infrastructure — a previously unobserved OPSEC technique indicating the operator is actively evolving counter-forensic capabilities.

---

## Complete Wallet Graph

### Tier 1: CEX Origins (KYC Layer)

| Address | Role | Chain | Notes |
|---|---|---|---|
| `0x503828976d22510aad0201ac7ec88293211d23da` | Coinbase Hot Wallet 1 | Ethereum | Origin A — 0.05 ETH seed, May 2023 |
| `0x28c6c06298d514db089934071355e5743bf21d60` | Binance Hot Wallet 1 | Ethereum | Origin B — 33,333 ETH withdrawal, Sep 2021 |

### Tier 2: Ethereum Mainnet Buffer Layer

| Address | Role | Chain | First Seen | Notes |
|---|---|---|---|---|
| `0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d` | Buffer Wallet | Ethereum | 2023-05 | Coinbase -> Treasury relay |
| `0x4976a4a02f38326660d17bf34b431dc6e2eb2327` | Binance Buffer | Ethereum | 2021-09 | 33,333 ETH pass-through |
| `0xf3d63166f0ca56c3c1a3508fce03ff0cf3fb691e` | Intermediate | Ethereum | 2022-11 | Binance -> Staging relay |
| `0x81f91aca8c05b3eefebc00171139afefac17c9a6` | Staging Wallet | Ethereum | 2023-12 | 30.4 ETH -> Whale Trader |

### Tier 3: Ethereum Mainnet Operations

| Address | Role | Chain | First Seen | Notes |
|---|---|---|---|---|
| `0x4c968f6beecf1906710b08e8b472b8ba6e75f957` | **Central Treasury** | Ethereum | 2023-05 | Hub for all downstream ops. Coinbase-funded. |
| `0xf70da97812cb96acdf810712aa562db8dfa3dbef` | **Whale Trader** | Ethereum | 2024-01 | Primary funding channel (68% of deployments as of Mar 28). Binance-funded. Moving 5-92 ETH/tx. |
| `0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1` | MEV Bot | Ethereum | 2024-08 | Funded 20x from treasury Aug-Sep 2024 |
| `0x391e7c679d29bd940d63be94ad22a25d25b5a604` | Revenue Collector | Ethereum | 2024-06 | Sweeper -> MEV bot |
| `0x5e0f8e7337c8955d2124b8e85ca74af884b3e124` | WETH Wrapping Station | Ethereum | 2026-01 | DeFi operations |
| `0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390` | Revenue Source | Ethereum | 2023-09 | ~22 ETH into treasury (MEV profits?) |
| `0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae` | LI.FI Diamond (3rd party) | Ethereum | -- | Cross-chain bridge for L1->L2 |

### Tier 4: L2 Core Infrastructure (Arbitrum + Base)

| Address | Role | Chain | Classified | DB Status |
|---|---|---|---|---|
| `0xf186cb00e49e18491db5783ff04fae3818102ff7` | **Treasury** | Arb+Base | org_001:treasury | CONFIRMED (entity_classification) |
| `0xe93d64f3fbc352131e79fc5578cbe44b66697f86` | **Operator** | Arb+Base | org_001:operator | CONFIRMED |
| `0xfd51e33d44b376ef346d24a130a51035db09c1dc` | **Operator 2** | Arb+Base | org_001:operator_2 | CONFIRMED |
| `0xc6962004f452be9203591991d15f6b388e09e8d0` | **Cashout** | Arb+Base | org_001:cashout | CONFIRMED |
| `0x8c826f795466e39acbff1bb4eeeb759609377ba1` | **Gas Station** | Eth+Arb+Base | org_001:gas_station | CONFIRMED. 503 deployers, 2,444 contracts. |
| `0x360e68faccca8ca495c1b759fd9eee466db9fb32` | **Vault / Treasury Branch** | Arb | org_001:treasury_branch | CONFIRMED. Receives WBTC from exit_cex. |

### Tier 5: Exit Infrastructure

| Address | Role | Chain | Classified | DB Status |
|---|---|---|---|---|
| `0x01989c93890aed05a63d179b03424997075b6acf` | **CEX Exit / Shadow Wallet 1** | Arb | org_001:exit_cex | **CRITICAL** — VANITY-SPOOFED. See Shadow Wallet section. |
| `0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb` | **Laundry (WBTC)** | Arb | org_001:laundry | CONFIRMED. Swaps through WBTC/USDC pools. |
| `0x96daa0b8a5499ea9323421ed0cda06b345caab73` | **LP Staging / Shadow Wallet 2** | Arb | org_001:lp_staging | **CRITICAL** — VANITY-SPOOFED. See Shadow Wallet section. |
| `0x27920e8039d2b6e93e36f5d5f53b998e2e631a70` | **LP Companion / ETH Wrapper** | Arb | org_001:lp_companion | CONFIRMED |
| `0x51c72848c68a965f66fa7a88855f9f7784502a7f` | **DeFi Exit Channel** | Arb | org_001:defi_exit_channel | CONFIRMED. Routes to USDC/USDT pools. |

### Tier 6: Shadow Wallet CEX Funding (Newly Discovered 2026-04-11)

**Shadow Wallet 1** (`0x01989c93...75b6acf`) — funded by 5 CEX hot wallets:

| CEX Hot Wallet | Nonce Range | Est. Volume | Notes |
|---|---|---|---|
| CEX Hot 1 | ~2.2M | High | Part of high-volume exchange deposit pool |
| CEX Hot 2 | ~3.5M | High | Same exchange cluster |
| CEX Hot 3 | ~4.8M | High | Same exchange cluster |
| CEX Hot 4 | ~6.1M | High | Same exchange cluster |
| CEX Hot 5 | ~7.9M | High | Highest nonce — most active hot wallet |

*Note: Exact CEX hot wallet addresses pending full trace. Nonce ranges 2.2M-7.9M indicate major exchange infrastructure (Binance/OKX-tier volume).*

**Shadow Wallet 2** (`0x96daa0b8...caab73`) — funded by sybil gas distribution network:

| Address | Role | Amount | Notes |
|---|---|---|---|
| `0x17548bc3...` | **Gas Distributor** | 60 ETH total | Distributed 3 ETH each to 20 wallets |
| 20x sybil wallets | Gas recipients | 3 ETH each | Used for transaction fees on shadow ops |

*Note: The 20 downstream sybil addresses pending full enumeration from the 0x17548bc3 outbound trace.*

---

## OPSEC Technique: Vanity Address Spoofing

**Discovery date:** 2026-04-11
**Discovery context:** Data gap investigation, Apr 8-11 period

### The Technique

org_001 generated vanity addresses with matching **prefixes** to existing infrastructure wallets. In block explorers and monitoring tools, addresses are often displayed truncated (e.g., `0x01989c93...acf`), making the spoofs visually identical to the originals.

| Real Address | Shadow (Spoofed) Address | Matching Prefix |
|---|---|---|
| `0x01989c93890aed05**cbcda4e62eec1b2eb4c55b1b**` (LP_POOL_2 in KNOWN_INFRA) | `0x01989c93890aed05**a63d179b03424997075b6acf**` | `0x01989c93890aed05` (8 bytes) |
| `0x96daa0b8...` (original LP staging) | `0x96daa0b8a5499ea9323421ed0cda06b345caab73` | `0x96daa0b8` (4 bytes) |

### Why This Matters

1. **Evades automated monitoring.** Any system matching on address prefixes (common in dashboards, alerts, log truncation) would classify shadow wallet transactions as belonging to known infrastructure.
2. **Confuses manual review.** An analyst scanning transaction lists sees familiar prefixes and skips deeper inspection.
3. **Enables parallel extraction.** Shadow wallets can receive/move funds independently while appearing to be the known wallet in truncated views.
4. **Volume during detection gap.** ~$2M moved through each shadow wallet during the Apr 8-11 data gap period.

---

## Fund Flow Diagram

```
                         CEX ORIGINS (KYC LAYER)
                    ┌────────────────────────────────┐
                    │                                │
              Coinbase Hot 1                   Binance Hot 1
              (0x503828...)                    (0x28c6c0...)
                    │                                │
                    │ 0.05 ETH                       │ 33,333 ETH
                    v                                v
              Buffer Wallet                    Binance Buffer
              (0x66666f...)                    (0x4976a4...)
                    │                                │
                    │ 2.75 ETH                       │ 0.5 ETH
                    v                                v
             CENTRAL TREASURY ◄─── 10 ETH ───  Staging Wallet
             (0x4c968f...)                     (0x81f91a...)
                    │                                │
          ┌─────────┼──────────┐                     │ 30.4 ETH
          │         │          │                     v
          v         v          v              WHALE TRADER
     Gas Station  MEV Bot  Revenue            (0xf70da9...)
     (0x8c826f)  (0x5babe)  Collector         386 deployers
     173 deployers          (0x391e7c)        68% of deployments
          │                                        │
          │  ┌─────── LI.FI Bridge ───────────┐    │
          │  │   (0x1231de... cross-chain)     │    │
          v  v                                 v    v
    ┌──────────────────────────────────────────────────┐
    │              L2 OPERATIONS (Arbitrum + Base)       │
    │                                                    │
    │   Treasury (0xf186cb) ◄──── Trap Profits          │
    │       │                                            │
    │       ├──► Operator (0xe93d64) ──► Deploy Traps   │
    │       ├──► Operator 2 (0xfd51e3) ──► Deploy Traps │
    │       │                                            │
    │   [Trap fires → victim ETH/tokens extracted]       │
    │       │                                            │
    │       v                                            │
    │   Cashout (0xc69620)                               │
    │       │                                            │
    │       ├──► LP Staging (0x96daa0*) ──► Cashout     │
    │       │       [shadow wallet — spoofed prefix]     │
    │       │                                            │
    │       ├──► CEX Exit (0x01989c*) ──► WBTC pools    │
    │       │       [shadow wallet — spoofed prefix]     │
    │       │       │                                    │
    │       │       └──► Vault (0x360e68) [WBTC store]  │
    │       │                                            │
    │       ├──► DeFi Exit (0x51c728) ──► USDC/USDT     │
    │       │                               pools        │
    │       │                                            │
    │       └──► Laundry (0xfdaf1f) ──► WBTC/USDC      │
    │               [WBTC conversion pipeline]           │
    │                                                    │
    │   LP Companion (0x279208) ── ETH wrapping ops     │
    │                                                    │
    └──────────────────────────────────────────────────┘
                              │
                              v
              ┌───────────────────────────────┐
              │     SHADOW WALLET FUNDING      │
              │                               │
              │  Shadow 1 (0x01989c...acf)    │
              │    ◄── 5 CEX hot wallets      │
              │    Nonces: 2.2M - 7.9M        │
              │    (major exchange infra)      │
              │                               │
              │  Shadow 2 (0x96daa0...ab73)   │
              │    ◄── Gas Distributor         │
              │    (0x17548bc3...)             │
              │    └── 20 sybil wallets       │
              │        (3 ETH each = 60 ETH)  │
              └───────────────────────────────┘
```

---

## Total Assets Identified

| Category | Estimated Value | Basis |
|---|---|---|
| Binance origin branch | ~$80M+ at withdrawal | 33,333 ETH @ ~$2,400 (2021 price range) |
| Whale Trader active flows | Multi-ETH daily | 5-92 ETH per tx, active daily |
| Central Treasury throughput | ~30 ETH visible | Coinbase inflows + revenue streams |
| L2 Trap extraction (confirmed) | ~$257K | extraction_events table total |
| L2 Trap extraction (estimated) | ~$520K+ | Earlier fund flow trace estimate |
| Shadow wallet volume (Apr 8-11) | ~$4M | ~$2M per shadow wallet during data gap |
| Sybil gas network | ~60 ETH ($150K+) | 20 wallets x 3 ETH from 0x17548bc3 |
| **Total identified infrastructure value** | **$80M+** | Conservative — Binance branch dominates |

*Note: The $80M+ figure is dominated by the 33,333 ETH Binance withdrawal. The operational L2 trap extraction ($520K+) is a small fraction of total capital. This suggests L2 traps are a subsidiary revenue stream within a much larger trading/MEV operation.*

---

## Operational Scale

| Metric | Value | Source |
|---|---|---|
| Total deployers (gas station path) | 503 | deployers table |
| Total deployers (whale trader path) | 1,819 | deployers table |
| Contracts (gas station path) | 2,444 | deployers table |
| Contracts (whale trader path) | 5,018 | deployers table |
| **Combined deployers** | **559+ profiled** (2,322 total in deployers table) | deployer_profiles + deployers |
| **Combined contracts** | **7,462+** | deployers table |
| Bot victims | ~3,400 | diamond_model |
| Confirmed trap events | 88 | trap_events join |
| Active chains | Arbitrum, Base | org_cycles.py |
| Trap types | DELEGATECALL proxy, conditional revert, V3 callback | diamond_model |
| Camouflage rating | MEDIUM | diamond_model |
| Anti-forensic techniques | Unicode WETH impersonation, multi-exit channel, **vanity spoofing** | diamond_model + new discovery |
| Operating timezone | Americas (UTC-5 to UTC-8) | diamond_model |
| Operational pattern | Night shift | diamond_model |

---

## Funding Channel Shift (Observed Mar 24-28)

The operation shifted its primary funding from the Coinbase-origin gas station to the Binance-origin whale trader during active observation.

| Date | Gas Station | Whale Path | Total | Whale % |
|---|---|---|---|---|
| Mar 24 | 272 | 184 | 456 | 40% |
| Mar 25 | 107 | 192 | 299 | 64% |
| Mar 26 | 158 | 229 | 387 | 59% |
| Mar 27 | 26 | 29 | 55 | 53% |
| Mar 28 | 103 | 216 | 319 | **68%** |

---

## Exit Ramp Architecture

Funds flow from Cashout through four parallel exit channels:

### Channel 1: CEX Exit (Shadow Wallet 1 — 0x01989c...acf)
- Receives USDC from Cashout
- Swaps to WBTC via Uniswap V3 pools
- Sends WBTC to Vault (0x360e68)
- **Two-way**: also sends WETH back to Cashout
- Funded independently by 5 CEX hot wallets (nonces 2.2M-7.9M)

### Channel 2: LP Staging (Shadow Wallet 2 — 0x96daa0...ab73)
- Receives USDC from Cashout
- Returns WETH to Cashout (LP yield extraction pattern)
- Supported by sybil gas network (20 wallets, 3 ETH each from 0x17548bc3)

### Channel 3: DeFi Exit (0x51c72848)
- Receives USDC from Cashout
- Routes to USDC/USDT DEX pools
- Outbound to unclassified addresses (0x389938, 0x7fcdc3)

### Channel 4: Laundry WBTC Pipeline (0xfdaf1f17)
- Swaps through WBTC/USDC and USDC/WBTC pools
- Converts extracted funds to WBTC (harder to trace, smaller market)
- Three separate pool interactions observed

---

## Known DEX Pools Used by Exit Infrastructure

| Pool Address | Label | Used By |
|---|---|---|
| `0x51c72848c68a965f66fa7a88855f9f7784502a7f` | LP Pool 1 / DeFi Exit | Cashout, DeFi Exit Channel |
| `0x01989c93890aed05cbcda4e62eec1b2eb4c55b1b` | LP Pool 2 | Exit CEX, LP operations |
| `0x0e4831319a...` | WBTC/USDC Pool | Laundry, Exit CEX |
| `0x6985cb98ce...` | WBTC/USDC Pool | Laundry |
| `0x5a17cbf5f8...` | USDC/WBTC Pool | Laundry |
| `0x5969ef...` | WBTC Pool | Exit CEX |

---

## Timeline of Discoveries

| Date | Discovery | Significance |
|---|---|---|
| Sep 2021 | 33,333 ETH Binance withdrawal | Origin of Binance branch |
| May 2023 | Coinbase withdrawal -> buffer -> treasury | Origin of main operation |
| Sep 2023 | Revenue starts flowing into treasury | Operation becomes profitable |
| Aug 2024 | MEV bot funded 20x in 1 month | Mainnet bot scaling |
| Jan 2025 | WETH wrapping station activated | DeFi integration |
| Feb 2025 | L2 gas station funded | L2 trap operation begins |
| Mar 2025 | V3 callback traps deployed | New capability adopted |
| 2026-03-19 | First L3 surveillance detection | org_001 enters our database |
| 2026-03-22 | Cross-org analysis confirms org_001 independent from org_002 | No shared infrastructure |
| 2026-03-23 | Fund flow traces reveal exit ramp architecture | 4-channel exit system mapped |
| 2026-03-24 | Entity classification assigned to all 10+ wallets | DB formalized |
| 2026-03-25 | Diamond model created | Full adversary profile |
| 2026-03-26 | Ethereum depth trace reveals CEX origins | Two KYC chains identified |
| 2026-03-28 | Funding channel shift detected (40% -> 68% whale) | Operational evolution during observation |
| 2026-03-29 | Watchlist entries for gas station + whale | Real-time monitoring enabled |
| **2026-04-11** | **Shadow wallets discovered (vanity spoofing)** | **New OPSEC technique. ~$4M volume in data gap.** |
| **2026-04-11** | **CEX hot wallet funding of shadow wallet 1 identified** | **5 hot wallets, nonces 2.2M-7.9M** |
| **2026-04-11** | **Sybil gas network for shadow wallet 2 identified** | **20 wallets, 60 ETH from 0x17548bc3** |

---

## Intelligence Gaps (What We Don't Know)

### Critical
1. **Actual cash-out mechanism.** We see funds flow to CEX Exit and through WBTC pools, but the final fiat off-ramp is not visible on-chain. Is it through the same CEX accounts (Coinbase/Binance) that funded the operation, or a separate set?
2. **Shadow wallet full transaction history.** The Apr 8-11 data gap means we have incomplete records of shadow wallet activity. Full trace needed.
3. **5 CEX hot wallet addresses.** The exact addresses of the 5 CEX hot wallets funding shadow wallet 1 are pending full enumeration from the trace report.
4. **20 sybil wallet addresses.** The full list of 20 sybil wallets funded by 0x17548bc3 for shadow wallet 2 needs enumeration.

### High Priority
5. **Are there more shadow wallets?** If they spoofed 2 addresses, there may be more vanity-generated addresses mimicking other infrastructure wallets (treasury, operator, laundry).
6. **Whale trader outbound targets.** 0xec0c2f (91.9 ETH), 0x3ab435 (38.3 ETH), 0xb92fe92 (multiple large txs) — could be exchange deposits, DeFi, or additional operational wallets.
7. **Revenue source (0x9e22ebec).** 22 ETH flowed into central treasury — origin unknown (MEV profits? Other chain extraction?).
8. **LI.FI bridge destinations.** Which L2 chains beyond Arbitrum and Base are they bridging to?
9. **Binance account holder identity.** 33,333 ETH withdrawal in 2021 is whale-tier. Personal or institutional?

### Lower Priority
10. **Unclassified exit destinations.** 0x389938, 0x7fcdc3 from DeFi Exit Channel — need tracing.
11. **March 2026 new wallet (0xb0e99b4e).** 0.335 ETH from treasury — new operational wallet?
12. **org_001 relationship to org_003.** Both operate on Arbitrum. Any shared infrastructure?

---

## Database Cross-References

### entity_classification table
10 org_001 addresses classified (CRIMINAL/INFRASTRUCTURE categories). Shadow wallets updated 2026-04-12 with CRITICAL priority and vanity-spoofing notes.

### watchlist table
4 org_001-related entries (whale trader, gas station, shadow wallet 1, shadow wallet 2). Shadow wallets added at CRITICAL priority on 2026-04-12.

### deployer_profiles table
26 deployers profiled with org_link=org_001 (all funded by gas station 0x8c826f).

### sload_patterns table
69 contracts with org_id=org_001.

### diamond_model table
Full adversary profile at case_id=org_001, confidence=CONFIRMED.

### Related Case Files
- `CASE_ORG_001_ETHEREUM_DEPTH.md` — L1 depth trace, CEX origins, funding hierarchy
- `FUND_FLOW_TRACE_20260323_024240.md` — Initial exit ramp analysis (19 flows)
- `FUND_FLOW_TRACE_20260323_041436.md` — Updated with DEX pool classifications
- `CROSS_ORG_ANALYSIS_20260322.md` — org_001 vs org_002 independence confirmed

---

*Generated by Layer 3 Surveillance System*
*All data sourced from public blockchain records and local surveillance database*
*No on-chain interactions performed — read-only intelligence*
