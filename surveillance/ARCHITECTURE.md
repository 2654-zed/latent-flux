# Layer 3 Surveillance System — Architecture

## Overview

A passive blockchain surveillance system monitoring smart contract deployments on Base, Arbitrum, and Optimism in real-time. Detects trap contracts (honeypots, fee skimmers, buy-not-sell tokens), traces organizational structures, documents criminal extraction operations, and provides a pre-transaction risk API for AI agent frameworks.

**Corpus (2026-04-16):** 124,341 contracts | 1.17M transaction events | 36,115 deployers | 1,296 bots
**Organizations tracked:** 3 confirmed + 1 independent parasite + 1 active drain operation (7 rogue facilitators)
**Drain operation documented:** $10-15M+ across 22 months (Permit2 drains + address poisoning, dual-vector)

---

## Quick Start — How to Run

**Clone + configure:**

```bash
git clone https://github.com/2654-zed/latent-flux.git
cd latent-flux
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Environment variables (.env or shell):**

```
ARB_WSS_URL=wss://arb-mainnet.g.alchemy.com/v2/<YOUR_KEY>
BASE_WSS_URL=wss://base-mainnet.g.alchemy.com/v2/<YOUR_KEY>
OP_WSS_URL=wss://opt-mainnet.g.alchemy.com/v2/<YOUR_KEY>
ADMIN_TOKEN=<any-random-string>
PORT=8080
```

**Run the full surveillance pipeline:**

```bash
python run_surveillance.py
```

This starts: WebSocket listeners on all 3 chains, bytecode classifier, revert cluster detector, event monitors, funder tracer, x402 monitor, single-writer SQLite process, and HTTP API on port 8080.

**Query the live API:**

```bash
curl http://localhost:8080/stats                                           # corpus metrics
curl http://localhost:8080/api/v1/agent/screen/base/0x<contract>           # pre-tx risk screen
curl http://localhost:8080/api/v1/agent/facilitator/<address>              # x402 facilitator check
curl "http://localhost:8080/dump?token=$ADMIN_TOKEN&table=contracts&limit=10"  # raw data
```

**Generate the daily intelligence brief (syncs from production first):**

```bash
ADMIN_TOKEN=<token> python -m surveillance.daily_brief --generate
```

**Production:** Deployed on Railway with persistent volume at `/app/surveillance/data`. Public URL `https://spypy.up.railway.app`. Auto-deploys from `master` branch.

---

## Real-Time Monitors (run on Railway 24/7)

### deployment_monitor.py
**Core.** Subscribes to `newHeads` via WebSocket on Arbitrum, Base, and Optimism. For each block, extracts all contract creation transactions, fetches bytecode, and passes to the classifier. Manages sub-monitors per chain. Auto-reconnects with infinite retry. Single-writer architecture: all monitors push writes via `multiprocessing.Queue` to a dedicated DB writer process (prevents WAL bloat and multi-process lock contention).

- Input: Alchemy WebSocket (ARB_WSS_URL, BASE_WSS_URL, OP_WSS_URL)
- Output: contracts + deployers tables
- Sub-monitors: selector_monitor, revert_cluster_detector, event_monitors, auto_funder_tracer, x402_monitor

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

### x402_monitor.py
Monitors the x402 payment protocol — EIP-3009 `transferWithAuthorization` and Permit2 `transferFrom` settlements. Maintains the facilitator registry (60 known, 4 unknown, 7 rogue). Fires `X402_AGENT_DRAIN` alerts on self-settlement patterns with rogue facilitators, classified as `REAL_DRAIN` vs `PASS_THROUGH` based on deposit source.

### risk_scoring.py
Stored-potential risk model. Scores every contract across five dimensions — approval scope, bytecode capability, deployer risk, org context, realized value — with a volatility multiplier (DELEGATECALL 2.5x, SELFDESTRUCT 3.0x, metamorphic 3.5x, timestamp-gate 2.0x). Implements the interpretive framework from `L3_TOPOLOGY_FRAMEWORK.md`: `risk = (stored_potential × volatility) / max(realized_value, 1)`. Tiers: CRITICAL ≥ 50, HIGH ≥ 20, MEDIUM ≥ 8, LOW ≥ 3, MINIMAL < 3.

### poisoning_watcher.py
Monitors high-value addresses for address-poisoning attempts (Unicode homoglyph tokens, phishing airdrops, vanity-prefix dust). Fires `POISONING_ATTEMPT` alerts. Scan state persists per `(address, chain)` so only new events are processed.

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

## HTTP API (run_surveillance.py + web/api_v1.py)

`ThreadingHTTPServer` on Railway port 8080. Two surfaces:

### Raw Data API (no auth required for read endpoints)
`/stats` `/suspected` `/priority` `/cross-chain` `/bots` `/bot-deployers` `/bot-selectors` `/clusters` `/cluster-events` `/known-selectors` `/funding` `/funding-hops` `/verification` `/traces` `/alerts` `/exposures` `/tx-events` `/liquidity-events` `/pair-creations` `/cex-candidates` `/bridge-events` `/rib/scores` `/rib/export` `/dump`

### Commercial API v1 (`/api/v1/*`, Bearer auth for tiered endpoints)
- **Tier A (free, for AI agents):** `/api/v1/agent/screen/{chain}/{address}` — pre-transaction risk screen returning `risk_score`, `risk_tier`, `capabilities`, `approval_exposure`, and machine-readable `recommendation` (DO_NOT_APPROVE / CAUTION / PROCEED / UNVERIFIED). `/api/v1/agent/facilitator/{address}` — x402 facilitator validation.
- **Tier 1 (screening):** `/api/v1/risk/{chain}/{address}`, `/api/v1/check/{address}`, `/api/v1/screen` (batch), `/api/v1/verify/{chain}/{address}` (deductive-only output for liability-conscious consumers)
- **Tier 2 (intelligence feed):** `/api/v1/feed`, `/api/v1/feed/stats`, `/api/v1/watch`, `/api/v1/ecosystem/stats`
- **Tier 3 (org intelligence):** `/api/v1/org/{org_id}` with `attribution_confidence`

### Admin POST Endpoints (ADMIN_TOKEN required)
`/admin/deployer-notes` `/admin/bot-candidate` `/admin/flag-address` `/admin/upgrade-contracts` `/admin/known-selector` `/admin/bot-cluster` `/admin/auto-assign-clusters` `/admin/entity-type` `/admin/mark-false-positive` `/admin/add-exposure` `/admin/compact-db` `/admin/register-webhook` `/admin/upload-db` (emergency DB replacement)

---

## Infrastructure

- **Railway:** Persistent deployment, Volume mount at /app/surveillance/data
- **Alchemy:** Arbitrum + Base + Optimism WebSocket + HTTP
- **Arbiscan V2 API:** Transaction history queries
- **1inch API:** Routing cross-reference
- **GitHub:** 2654-zed/latent-flux (auto-deploy to Railway on push)
- **Interpretive framework:** `L3_TOPOLOGY_FRAMEWORK.md` at repo root — five-primitive risk model that changes how data is read (stored potential vs behavioral snapshot)

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
