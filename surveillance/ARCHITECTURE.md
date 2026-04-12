# Layer 3 Surveillance System — Architecture

## Overview

A passive blockchain surveillance system monitoring smart contract deployments on Base and Arbitrum in real-time. Detects trap contracts (honeypots, fee skimmers, buy-not-sell tokens), traces organizational structures, and documents criminal extraction operations.

**Corpus:** 25,000+ contracts | 128,000+ transaction events | 9,700+ deployers | 620+ bots
**Organizations tracked:** 3 confirmed + 1 independent parasite
**Total USD traced:** $257,000+

---

## Real-Time Monitors (run on Railway 24/7)

### deployment_monitor.py
**Core.** Subscribes to `newHeads` via WebSocket on both chains. For each block, extracts all contract creation transactions, fetches bytecode, and passes to the classifier. Manages 4 sub-monitors. Auto-reconnects with infinite retry. Hourly WAL checkpoint.

- Input: Alchemy WebSocket (ARB_WSS_URL, BASE_WSS_URL)
- Output: contracts + deployers tables
- Sub-monitors: selector_monitor, revert_cluster_detector, event_monitors, auto_funder_tracer

### selector_monitor.py
**Sub-monitor.** Processes every transaction in every block against monitored contracts. Extracts function selectors, tags known bot signatures, classifies gas patterns (static vs dynamic), detects approval drains. Records to transaction_events table.

### revert_cluster_detector.py
**Sub-monitor.** Counts reverted transactions per address per block. Flags addresses exceeding 5 reverts/block as bot candidates. Clusters bots sharing function selectors. Records to bot_candidates + bot_candidate_events.

### event_monitors.py
**Sub-monitor.** Six monitoring capabilities:
1. **DEX Liquidity** — addLiquidity/removeLiquidity on 16 router addresses
2. **Token Approvals** — approve() on suspected/confirmed contracts
3. **Bridge Activity** — transfers from org wallets to 4 bridge contracts
4. **Pair Creation** — PairCreated events from 6 DEX factories (sampled every 100 blocks)
5. **CEX Deposit Pattern** — flags addresses with 20+ senders and 0 outflows
6. **Org Wallet Transfers** — all outbound transfers from classified org wallets

### auto_funder_tracer.py
**Sub-monitor.** Traces the funding source for every new deployer at deployment time (1 Alchemy API call per deployer). Checks if funder is a known org wallet (instant flag) or funds 5+ other deployers (gas station pattern). Hop-traces high-score deployers every 1000 blocks.

### routing_monitor.py
Polls the 1inch API every 5 minutes. Cross-references flagged contracts against live Arbitrum token routing. Detects routing anomalies (1inch avoids a token despite apparent price advantage).

### bytecode_classifier.py
Static analysis of deployed EVM bytecode. 10 pattern detectors: asymmetric_transfer, blacklist_check, tx_origin_conditional, callback_trap, hidden_fee, selfdestruct, delegatecall_in_token, timestamp_activation, origin_eoa_gate, obfuscated_fee. Uses PUSH-data skipping to avoid false positives.

---

## Analysis Modules (run on demand)

### longitudinal_scorer.py
Behavioral scoring across sessions. 7 positive signals (multi-session, USDC flow, return ratio, patience, template reuse, bot interaction, gas station funding) + 4 negative signals. Scores deployers 0.0-1.5. Designed to catch "card counters" who stay below velocity detection.

### entity_classifier.py
Classifies all addresses into a taxonomy: INFRASTRUCTURE, INSTITUTIONAL, COMMERCIAL, CRIMINAL, BOT, INDIVIDUAL, UNKNOWN. 1,080 addresses classified from 7 sources. Confidence protection prevents overwriting high-confidence labels.

### fund_tracer.py
Post-extraction fund flow classification. Traces where money goes after exit ramp extraction. Known registries: 10 CEX hot wallets, 20 DEX routers, 5 DEX pools, 4 bridges, 2 mixers. Integrates with entity_classifier.

### org_cycles.py
Temporal activity pattern analysis. Identifies active/dormant cycles, time-of-day patterns, day-of-week patterns, funding-to-deployment lag, deployment-to-cashout lag. Predicts next active window with confidence rating.

### case_file.py
Generates structured Markdown intelligence briefs from the database for any contract address. Includes: contract identity, trap mechanisms, traffic analysis, deployer profile, organizational links, kill chain timeline, risk assessment, evidence hashes, recommended actions.

### daily_report.py
Automated daily intelligence report. Covers: corpus growth with day-over-day deltas, hot contracts, camouflage ratio, org activity, new deployers, cross-chain count, longitudinal scoring, system health, detection summary, active watchlist. Scheduled at 06:03 UTC on Railway.

### rib_scorer.py
Relational Intelligence Benchmark. Builds a NetworkX graph from the DB, runs 4 baseline detectors (random, degree centrality, PageRank, Louvain), scores them against org_001 ground truth. System scores 84.6% precision, 91.7% recall on org identification.

### rib_export.py
Exports org_001 and surrounding graph as an anonymized temporal edge list. 25,602 edges, 14,249 nodes, 12 ground truth nodes with roles. Deterministic UUIDs (seeded). For benchmark distribution.

---

## Database Schema

**SQLite with WAL mode.** Railway persistent volume at `/app/surveillance/data`. Local at `surveillance/data/surveillance.db`.

### Core Tables
| Table | Rows | Purpose |
|---|---|---|
| contracts | 25,000+ | Every flagged contract with bytecode analysis |
| deployers | 9,700+ | Every deployer with funding trail + entity type |
| transaction_events | 128,000+ | Every interaction with monitored contracts |
| bot_candidates | 620+ | Addresses with high revert counts |
| bot_candidate_selectors | 4,000+ | Function selectors used by each bot |
| bot_candidate_events | 11,500+ | Per-block revert counts |
| bytecode_cache | 20,000+ | Classification results keyed by code hash |

### Intelligence Tables
| Table | Purpose |
|---|---|
| extraction_events | Documented live extraction cycles (3 events, $257K) |
| pattern_matches | Treasury pattern evidence (org_001, org_002, org_003) |
| entity_classification | 1,080 addresses with type/subtype/confidence |
| stats_snapshots | DB state at time of extraction events |
| known_selectors | Tagged function selectors (15 entries) |

### Monitoring Tables
| Table | Purpose |
|---|---|
| alerts | Webhook alerts (57K after dedup) |
| live_exposures | Open approval drain risks (2 active) |
| liquidity_events | DEX liquidity activity (populates on Railway) |
| approval_events | Token approvals on suspected contracts |
| bridge_events | Cross-chain transfers from org wallets |
| pair_creation_events | New DEX pair creation events |
| cex_deposit_candidates | Addresses matching CEX deposit pattern |
| org_transfer_events | Outbound transfers from org wallets |

### System Tables
| Table | Purpose |
|---|---|
| heartbeat | Monitor health (4 components) |
| connection_gaps | WebSocket disconnect/reconnect log |

---

## HTTP API (run_surveillance.py)

`ThreadingHTTPServer` on Railway port 8080. GET endpoints for data queries, POST endpoints for admin operations (auth via ADMIN_TOKEN).

### GET Endpoints
`/stats` `/suspected` `/priority` `/cross-chain` `/bots` `/bot-deployers` `/bot-selectors` `/clusters` `/cluster-events` `/known-selectors` `/funding` `/funding-hops` `/verification` `/traces` `/alerts` `/exposures` `/tx-events` `/liquidity-events` `/pair-creations` `/cex-candidates` `/bridge-events` `/rib/scores` `/rib/export` `/dump`

### POST Endpoints (admin auth required)
`/admin/deployer-notes` `/admin/bot-candidate` `/admin/flag-address` `/admin/upgrade-contracts` `/admin/known-selector` `/admin/bot-cluster` `/admin/auto-assign-clusters` `/admin/entity-type` `/admin/mark-false-positive` `/admin/add-exposure` `/admin/compact-db` `/admin/register-webhook`

---

## Infrastructure

- **Railway:** Persistent deployment, Volume mount at /app/surveillance/data
- **Alchemy:** Arbitrum + Base WebSocket + HTTP (free tier, ~14% usage)
- **Arbiscan V2 API:** Transaction history queries
- **1inch API:** Routing cross-reference
- **GitHub:** 2654-zed/latent-flux (auto-deploy to Railway on push)

---

## Sync + Local Development

`sync_railway_db.py` pulls all tables from Railway `/dump` endpoint into local SQLite. Run before any local analysis:

```bash
python3 sync_railway_db.py
```

---

## Organizations Tracked

| Org | Pattern | Contracts | Victims | Funding | Chain |
|---|---|---|---|---|---|
| org_001 | DELEGATECALL traps | 93 | ~3,400 | $520K+ traced | Arb+Base |
| org_002 | tx.origin buy-not-sell | 367+ | 1,660+ | 215+ ETH via 2 treasuries | Base |
| org_003 | SHA3@0x1508 fee-skimmer | 6 | 727 | INVISIBLE (ghost deployers) | Base |
| Parasite | Uniswap routing injection | 1 | 2,910+ | Disposable, emptied | Base |

---

## Key Findings

1. **Trust Amplification: 96.6% router dominance** — The parasite contract `0xd4624228` received 96.6% of its 2,910 callers via Uniswap's Universal Router (selector `3593564c`), averaging 1,332 callers/day. *(Note: the originally claimed "14.2x amplification factor" compared this to a hand-picked set of 20 contracts averaging 94 callers/day. That comparison set was never persisted and the number cannot be reproduced from stored data. See CORRECTIONS.md 2026-04-02. The 96.6% router dominance IS verified.)*
2. **Three-tier anti-forensic model (org_001):** Transaction layer (custom selector drains — zero log events, invisible to event-log forensics), Victim layer (Unicode WETH impersonation — evades human token name inspection), Intelligence layer (vanity address spoofing — 7-char prefix matching evades organizational monitoring and chain analysis). The intelligence-layer capability is the highest counter-intelligence sophistication observed in the corpus.
3. **Trap-as-a-Service:** 229 contracts from 213 deployers share parameterized bytecode templates with anti-forensics offset randomization
4. **$5.75M MEV Vault** discovered through a "failing" bot's funder network (Bot_A investigation)
5. **Revert rate trend:** 3% → 34% over 7 days as trap density increases
6. **Camouflage ratio:** 69-73% of active contracts maintain <10% revert rate
7. **Zero mixer usage** by org_001 — launders entirely through standard DEX pools
