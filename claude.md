# Layer 3 — Claude Code Implementation Brief
**Last updated:** April 25, 2026
**Purpose:** This is the implementation context for Claude Code sessions working on Layer 3. Read this first before touching any code.

---

## Session-Start Protocol

The corpus is now large enough that finding-rediscovery is the dominant failure mode for fresh sessions. Several recent sessions have independently "discovered" infrastructure that was already characterized in `surveillance/data/cases/` (org_001 wallets re-attributed; Coffee Fleet single-operator hypothesis restated as new; `0xd4624228` partially reframed without consulting `PARASITE_ARCHITECTURE_0xd4624228.md`). The cause is structural — sessions begin without a literature-review step. These four phases are the fix.

Skipping a phase is the failure. Finding nothing in a phase is fine.

### Phase 1 — Index Loading (mandatory, no exceptions)

Read at session start, in order:
1. `claude.md` (this file)
2. `docs/lexicon.md`
3. `docs/INDEX.md`
4. `reports/correction_log.md`
5. `CORRECTIONS.md` (project root — distinct from `reports/correction_log.md`)

Both correction logs exist and serve different purposes. `CORRECTIONS.md` records customer-facing claim retractions; `reports/correction_log.md` records numbered methodology corrections. **Always check both.**

State at session start:

```
Phase 1 complete:
  claude.md (last updated [date]),
  lexicon.md (version [date]),
  INDEX.md (version [date]) — covers [N] organizations, [M] addresses, [K] patterns,
  reports/correction_log.md (latest #[N]),
  CORRECTIONS.md (latest entry [date]).
```

If `INDEX.md` is older than the newest file in `surveillance/data/cases/`, or `reports/correction_log.md` has not been updated in >30 days, flag staleness before proceeding.

### Phase 2 — Task-Specific Literature Review

Before producing any analysis, identify what the task touches and resolve each piece against documented prior work:

- **Every address in the task** → search Section 2 of `INDEX.md`. For any address not in the index, grep `surveillance/data/cases/` and `reports/` before claiming it is undocumented.
- **Every pattern, hypothesis, or framework concept** → search `docs/lexicon.md`, then Section 3 of `INDEX.md`, then both correction logs.
- **Every entity name** (`org_xxx`, `Entity_xxx`, named cases like Dragon / Coffee Fleet / Routing Parasite) → look up in Section 1 of `INDEX.md`.

Produce the list before analyzing:

```
Prior context for this task:
- 0xABC: classified as [role] in [file], watchlist [tier or "none"].
- Pattern Y: documented in lexicon entry [Z], status [DOCUMENTED / RETIRED / OPEN / ANECDOTE].
- 0xDEF: not in INDEX.md, no grep matches in cases/ or reports/. Treating as undocumented; will not claim novel until Phase 3 verification.
```

Default posture: addresses are documented somewhere unless proven otherwise. The index Section 2 is grep-able. Use it.

### Phase 3 — Prior-State Declaration

Before producing analytical findings, declare the prior state explicitly:

```
Prior to this investigation, the following is documented:
[1-paragraph summary citing specific files and key facts]
```

OR (only after Phase 2 grep failed to surface documentation):

```
No prior documentation found for [item] in (claude.md, lexicon.md, INDEX.md, both correction logs, cases/, reports/). Flagging as potentially novel.
```

The declaration is the load-bearing artifact. Without it, analysis defaults to silent rediscovery. **Skipping this phase is the protocol's primary failure mode.**

### Phase 4 — Delta Framing

Findings must frame their relationship to prior state in one of five shapes:

- **Confirms** — "Prior file [X] documented [Y]; this investigation confirms with additional evidence [Z]."
- **Refines** — "Prior file [X] documented [Y]; this investigation refines to [Z] based on [evidence]."
- **Contradicts** — "Prior file [X] documented [Y]; this investigation contradicts based on [evidence]. Recommend correction-log entry."
- **Extends** — "Prior file [X] documented [Y] up to [boundary]; this investigation extends with [Z]."
- **Novel** — "No prior file documents [Y]; this finding is novel because [explicit reason it would have appeared in prior files if known]."

Default posture: findings are confirmation, refinement, contradiction, or extension of prior work. Pure novel discovery is the rare case, not the default. A finding framed Novel without explicit justification of why prior work would have surfaced it (if known) is not Phase-4-compliant.

### Phase 5 — Output Discipline (when warranted)

When a session produces a finding that warrants a case file (genuinely novel entity; substantially refining a prior file; documenting a new wallet role), the session must:

1. **Generate the case-file content** under `surveillance/data/cases/`. Match the existing naming convention:
   - `CASE_ORG_{NNN}_{TITLE}.md` for organizational characterizations
   - `CASE_ENTITY_{NNN}_{NAME}.md` for entity-level dives
   - `CASE_{NAME}_{0xADDR}.md` for named cases
   - `BOT_INVESTIGATION_{0xADDR}_{date}.md` / `PARASITE_ARCHITECTURE_{0xADDR}.md` / `CASE_ZERODAY_{0xADDR}.md` for specialized formats
   - `CASE_0x{prefix8}_{chain}_{YYYYMMDD}_{HHMMSS}.md` is auto-generated by `surveillance/case_file.py`; don't write these by hand
2. **Update `docs/INDEX.md`** — Section 1 entry if entity-level, Section 2 entries for every wallet with a role, Section 3 if a new pattern or hypothesis surfaces, Section 4 if it's an extraction event, Section 5 if it's a bytecode family.
3. **If a prior claim is being retired** → append to `reports/correction_log.md` (numbered) and/or `CORRECTIONS.md` (date-titled), per the discipline of each log.

The case-file commit and the INDEX.md update commit are the same commit. A new case file without an index update is the failure mode this protocol exists to prevent.

### Index Maintenance Commitment

Every commit that adds, removes, or substantially edits a file in `surveillance/data/cases/` or `reports/` MUST include a paired update to `docs/INDEX.md`. The stretch-goal `scripts/check_index.py` automates drift detection: lists addresses in case files but absent from INDEX.md, lists INDEX.md entries pointing to nonexistent files, returns nonzero on drift. Run it as a pre-commit check or as Phase 1 staleness verification.

---

## What Layer 3 Is

Layer 3 is a production behavioral intelligence platform monitoring smart contract deployments on Base, Arbitrum, and Optimism in real time. It has been running continuously since March 17, 2026.

**Core thesis:** Harm in permissionless systems emerges from composition of correctly-executing components, not from individual code defects. Code auditing finds bugs. Layer 3 measures what happens when there are no bugs — stored potential, organizational infrastructure, trust exploitation, and compositional risk.

**What the system does:** Captures every new contract deployment across three chains, classifies bytecode at ingest, accumulates behavioral data over time, maps organizational structures through fund flow analysis, scores stored potential through a multi-component risk model, and serves intelligence through 19 API endpoints.

---

## Current Corpus (as of 2026-05-09, from production `/stats`)

| Metric | Value | vs. April 2026 baseline |
|---|---|---|
| Contracts monitored | 282,401 (1,404 confirmed + 115,514 suspected + 39,842 unanalyzed + 125,641 unknown) | 3.13x (was 90,276+) |
| Transaction events | 16,678,335 | 6.84x (was 2.44M+) |
| Unique deployers | 66,805 | 2.59x (was 25,834) |
| Confirmed traps | 1,404 | 3.59x (was 391) |
| Suspected traps | 115,514 | 2.35x (was 49,190) |
| Bot candidates | 4,244 | 6.69x (was 634) |
| Funder coverage | 91.9% (61,375 of 66,805 deployers traced) | new line item |
| Gas stations identified | 253 | new line item |
| Org links found | 4,382 | new line item |
| Cross-chain shared deployers | 1,046 | new line item |
| Organizations mapped | 4 | unchanged |
| API endpoints | new surface (post-refactor) — `/stats /suspected /priority /bots /tx-events /known-selectors /clusters /cluster-events /health`. The 19-endpoint Tier 1/2/3/A surface documented below is from the prior architecture; reconcile against current routes before relying on the list. | needs spec reconciliation |
| Analysis modules | 54 | unchanged (modules in repo; runtime composition not re-audited) |
| Database tables | 50+ | unchanged |

**Pulled from:** `https://stellar-embrace-production-2020.up.railway.app/stats` on 2026-05-09 (heartbeat 13 min before pull: `deployment_monitor_optimism 2026-05-09T22:09:10.140408+00:00`).

**Note on the API surface:** The 19 endpoints documented in the section below (`/risk/{chain}/{address}`, `/check/...`, `/screen/...`, `/feed`, `/dump`, `/org/...`, `/deployer/...`, etc.) all return 404 on the new service. The `/dump` route returns 403 (alive but token-rotated) — re-enable for delta sync once new token is set. The other Tier 1/2/3/A routes appear to have been replaced with a stripped surface; section needs a reconciliation pass.

---

## Infrastructure

- **Deployment:** Railway, persistent volume, auto-restart supervisor
- **Chains:** Base, Arbitrum, Optimism (one WebSocket connection per chain)
- **Database:** SQLite at `data/surveillance.db` — do not migrate without explicit instruction
- **Language:** Python 3.13
- **Dependencies:** FastAPI, web3.py, aiohttp, sqlite3 (stdlib). Zero ML libraries.
- **Node provider:** Alchemy (WebSocket + REST). API key in environment.

**Design constraint (strictly enforced):** Minimize Alchemy API calls. All analysis runs from SQLite. One persistent WebSocket per chain. One REST call per new deployer for funding trace (`auto_funder_tracer`). On-demand RPC for `eth_depth.py` (manual investigation) and `intelligence.py` (manual). Never build features requiring periodic RPC polling. Approval drain monitor checks SQLite every 30 minutes, not RPC.

---

## Module Inventory (54 Python files)

### Core Monitoring (runs 24/7)
| Module | Function |
|---|---|
| `deployment_monitor.py` | WebSocket monitor — new contract detection, 4 sub-monitors |
| `selector_monitor.py` | Function selector extraction, bot tagging, gas patterns |
| `revert_cluster_detector.py` | Flags >5 reverts/block as bot candidates |
| `event_monitors.py` | 6 monitors: DEX liquidity, approvals, bridges, pair creation, CEX deposits, org wallet transfers |
| `auto_funder_tracer.py` | Traces funding source for every new deployer (1 Alchemy call each) |
| `routing_monitor.py` | 1inch API polling — CURRENTLY DOWN, API key expired |

### Analysis Modules
| Module | Function |
|---|---|
| `bytecode_classifier.py` | 10 pattern detectors for trap signatures at deployment time |
| `bytecode_families.py` | Template clustering — 718 families, 73K+ members |
| `entity_classifier.py` | Address classification from 7 behavioral sources |
| `longitudinal_scorer.py` | Behavioral scoring (7 positive + 4 negative signals) |
| `camouflage_tracker.py` | Daily revert-rate classification + trend |
| `trust_amplification.py` | Router exploitation detection + amplification factor |
| `trend_forecaster.py` | Daily metrics + 48h predictions with scoring |
| `behavioral_baseline.py` | Statistical norms + z-score anomaly detection |
| `diamond_model.py` | Diamond Model intelligence framework |
| `fund_tracer.py` | Post-extraction fund flow classification |
| `org_cycles.py` | Activity cycle analysis with timezone inference |
| `eth_depth.py` | Ethereum mainnet on-demand depth tracing |
| `strategy_fingerprint.py` | Bot strategy classification, bait profiling, lifecycle tracking |
| `deployer_profiler.py` | Behavioral fingerprinting — 14 dimensions, similarity clustering |
| `goplus_enrichment.py` | GoPlus API benchmark — detection gap measurement |
| `risk_scoring.py` | Stored potential model (core product) |
| `drain_detector.py` | Permit2 drain event detection (Path B, self-settlement) |
| `x402_monitor.py` | x402 protocol monitoring across 4 phases |
| `proxy_upgrade_watcher.py` | Detects implementation changes on proxy contracts |
| `vanity_attention_scanner.py` | Inverted detection — scores addresses by poisoning attention received |

### Reporting & Intelligence
| Module | Function |
|---|---|
| `daily_brief.py` | Comprehensive daily intelligence brief (automated 06:03 UTC) |
| `daily_report.py` | Automated daily intelligence report |
| `case_file.py` | Generates structured case briefs for any contract |
| `rib_scorer.py` | Relational Intelligence Benchmark |
| `rib_export.py` | Anonymized dataset export |

### Infrastructure
| Module | Function |
|---|---|
| `db.py` | Database access layer with auto-migrations |
| `intelligence.py` | On-demand Alchemy analysis functions |

---

## Database Schema (Current Production State)

Key tables:

| Table | Rows | Purpose |
|---|---|---|
| `contracts` | 90,276+ | All monitored contracts |
| `transaction_events` | 2.44M+ | Every interaction with monitored contracts |
| `deployers` | 25,834+ | Deployer profiles with funding trails |
| `alerts` | 57,464+ | Detection alerts |
| `bytecode_cache` | 30,682+ | Raw bytecode storage |
| `bytecode_families` | 718 | Template clusters |
| `bytecode_family_members` | 73,106 | Family membership |
| `bot_candidates` | 634+ | Identified bots |
| `bot_candidate_events` | 18,876+ | Bot interaction history |
| `entity_classification` | 1,080+ | Address taxonomy |
| `behavioral_anomalies` | 121+ | Z-score outliers |
| `trust_amplification` | 32+ | Router exploitation metrics |
| `camouflage_metrics` | Daily | Daily camouflage ratios |
| `daily_metrics` | Daily | Daily aggregate metrics |
| `predictions` | Daily | Forecast + scoring |
| `diamond_model` | 4 | Intelligence cases |
| `extraction_events` | 5+ | Documented theft events. Columns include `chain` and `monitored_chain` (added 2026-04-18). `monitored_chain=0` for off-chain reference events (Rhea on NEAR, Drift on Solana); `monitored_chain=1` for events on Base/Arbitrum/Optimism. Chain-scoped rollups MUST filter on `monitored_chain=1` to avoid mixing off-chain dollars with L2 dollars. |
| `infrastructure_registry` | 12+ | Known-legitimate high-stakes infrastructure (Circle CCTP v2 seeded 2026-04-18; Uniswap routers, Aave pools etc. as they land). PK `(address, chain)` — same address can appear on multiple chains (CCTP v2 uses CREATE2 so the same address is canonical everywhere). Primary classification location; `entity_classification` may cross-link later. This becomes the "known-legitimate template baseline" (priority #2). |
| `org_wallets` | — | Organization wallet mappings |
| `deployer_profiles` | 25,834 | Behavioral fingerprints |
| `deployer_similarity` | 4,879+ | High-similarity pairs |
| `bot_strategies` | 634 | Strategy classifications |
| `bait_profiles` | 51 | Bait pattern profiles |
| `strategy_lifecycle` | 8 | Strategy saturation tracking |
| `eth_traces` | 8+ | Ethereum mainnet depth traces |
| `connection_gaps` | 225+ | WebSocket disconnections |
| `approval_watchlist` | Active | Permit2 approvals on suspected contracts |
| `drain_events` | Continuous | Detected drain events |

Auto-migrations handled by `db.py`. Schema changes require migration entries, never manual ALTER TABLE in production.

---

## The Risk Scoring Model

Core equation in `risk_scoring.py`:

```
risk_score = (stored_potential × volatility) / max(realized_value, 1)
```

**Stored potential (0-100):**
- `approval_scope` (0-25): UNLIMITED/NEVER = 25, bounded with expiration = lower
- `capabilities` (0-25): DELEGATECALL +10, asymmetric_transfer +8, conditional_revert +7, unusual_fee +5, SELFDESTRUCT +5
- `deployer_risk` (0-25): prior confirmed +10, suspected count +5, total contracts +5, org link +5, velocity +3
- `org_context` (0-25): org_001 linked +15, other org +10, gas station funded +5, cross-deployer family +3

**Volatility multiplier:**
- 3.0x: SELFDESTRUCT present
- 2.5x: DELEGATECALL + ownership not renounced
- 2.0x: timestamp-gated logic
- 1.0x: fixed / burned keys

**Realized value (0-25):** Normalized extraction amount. Deployer-adjacent deposits excluded (gaming resistance).

**Tier boundaries:** CRITICAL ≥50, HIGH ≥20, MEDIUM ≥8, LOW ≥3, MINIMAL <3

**Core interpretive rule:** A contract with maximum capability, maximum permissions, zero extraction, and high volatility is at PEAK stored potential — not minimum risk. The absence of realized value is the danger signal.

**Persistence:** `risk_scores` are computed live per API request. Not persisted. Historical tracking and bulk queries are not available in the current architecture. See Correction #6 (2026-04-17) for context; `/api/v1/risk/{chain}/{address}` calls `surveillance.risk_scoring.score_contract` per request, p50 ~80ms.

---

## API Endpoints (19 Live)

### Tier 1 — Screening (high volume, low latency)
- `GET /risk/{chain}/{address}` — decomposed risk score with component breakdown
- `GET /check/{chain}/{address}` — binary safe/unsafe response
- `GET /screen/{chain}/{address}` — detailed screening with recommendation
- `GET /verify/{chain}/{address}` — pure deductive facts (Tier A only)

### Tier 2 — Intelligence Feed
- `GET /feed` — recent high-priority alerts
- `GET /feed/stats` — feed statistics
- `POST /watch/{chain}/{address}` — add to watchlist
- `GET /watch/{chain}/{address}` — check watchlist status
- `DELETE /watch/{chain}/{address}` — remove from watchlist

### Tier 3 — Platform
- `GET /org/{org_id}` — organizational intelligence
- `GET /org` — list all mapped organizations
- `GET /deployer/{address}` — deployer profile with similarity clusters
- `GET /contract/{chain}/{address}` — full contract intelligence
- `GET /ecosystem/stats` — corpus-wide statistics

### Tier A — Agent (no auth, machine-speed)
- `GET /agent/screen/{chain}/{address}` — binary recommendation for AI agents
- `GET /agent/facilitator/{address}` — x402 facilitator validation

### Public
- `GET /docs` — API documentation
- `GET /methodology/confidence` — confidence tier methodology
- `GET /methodology/camouflage` — camouflage ratio methodology

---

## The Epistemic Framework

Every output is tagged with epistemic status:

- **Tier A (Deductive):** Verifiable on-chain. "This bytecode contains DELEGATECALL at offset 0x2a1." Anyone can verify.
- **Tier B (Inferential):** Analytical judgment based on documented methodology. "This deployer is likely connected to org_001 based on behavioral similarity at 0.87."

The `/verify` endpoint returns ONLY Tier A. The `/methodology/*` endpoints publish scoring algorithms. Every alert carries an `epistemic_tag`.

**The correction log:** `reports/correction_log.md` tracks every retracted or revised claim with root cause. Recent corrections:
- `0x785ce546` reclassified from "victim" to "controlled intermediary"
- Camouflage ratio revised from 79.2% to 70-79% range
- GoPlus 14.2x trust amplification retracted as unverifiable
- T2-eaef6a5d identified as NULL bucket methodology artifact (April 2026)
- 881E reclassified as address poisoner (not just drainer-adjacent)

**Rule:** Any new claim that contradicts or revises an existing claim requires a correction log entry with root cause analysis. Never silently update.

---

## Development Conventions

### How to write prompts for Claude Code tasks

Use the multi-phase structured format:

1. **Phase 1: Data inventory (read before acting).** Specify exactly which files to read first and which SQL queries to run before any analysis.
2. **Phase 2: Analysis.** What to do with the data, explicit boundaries on what counts as evidence.
3. **Phase 3: Verdict/output.** Classification options with evidence requirements, output format specification.
4. **Acceptance criteria.** What "done" means, explicit falsifiability conditions.
5. **What NOT to build.** Explicit negative scope.

### Working discipline

- **Read before acting.** Every significant task starts with reading relevant modules and running diagnostic queries. Never write code against assumed architecture.
- **No Alchemy calls without explicit approval.** If a task seems to require RPC, flag it first. Most tasks can run from SQLite.
- **Loud failures over silent wrong output.** If classification can't be determined, return UNKNOWN. If a query fails, raise. If data is missing, say so. Never guess to fill a field.
- **Conservative over aggressive on classifications.** False negatives acceptable. False positives destroy credibility.
- **Immutable historical record.** Once confirmed data enters the DB, it is appended to, not edited. New information creates new entries.
- **Schema-first for any new capability.** Design the table structure before writing pipeline code.

### Code patterns to follow

- All analysis modules follow pattern: `python -m surveillance.{module} --{action}` (e.g., `--compute`, `--analyze`, `--generate`)
- All new modules include a `__main__` block with CLI flags
- Database access through `db.py` only — no direct sqlite3 connections elsewhere
- Auto-migrations registered in `db.py` migration dict
- Alerts written to `alerts` table with `epistemic_tag` field populated

### Code patterns to avoid

- Direct RPC calls outside approved modules
- Direct sqlite3 connections bypassing `db.py`
- Silent retries on API failures (log and escalate)
- Mutable updates to confirmed records
- ML dependencies (zero ML by architectural decision)

---

## The Adversarial Topology Framework

When analyzing any contract, address, or system component, evaluate across five topological primitives:

1. **Position** — Where does this node sit relative to user assets? Can it observe, intercept, or modify?
2. **Permissions** — What edges exist between this node and user assets? Maximum scope, not currently exercised scope.
3. **Trust bindings** — What assumptions cause users to treat this node as safe?
4. **Mutability** — Can this node change behavior without re-consent? Proxy upgrades, version bumps, implementation swaps.
5. **Observation capability** — What can this node see? Transaction data, address inputs, behavioral patterns.

**Interpretive rule:** A node with privileged position, broad permissions, high mutability, strong trust binding, and zero malicious behavior is at MAXIMUM stored potential — not minimum risk.

This framework transfers to non-blockchain domains (browser extensions, AI agents, mobile apps, SaaS integrations). Any work extending Layer 3's methodology to new domains should map the five primitives first.

---

## Current Priority Items

1. **Compute/publish positive predictive value per risk tier** — 0.8% base rate means CRITICAL PPV must be computed and disclosed
2. **Build known-legitimate bytecode template baseline** — discount OpenZeppelin, Uniswap, standard patterns
3. **Add approval staleness dimension to risk model** — time since approval granted as modifier
4. **Deploy proxy upgrade watcher on Railway** — built but not yet deployed, urgent after 21-day blind spot
5. **Ship suspected + high traffic auto-escalation** — any suspected contract with 50+ callers generates WARNING
6. **Ship cross-deployer family velocity detector** — identical bytecode from multiple deployers in minutes = coordinated
7. **Vanity attention scanner v2** — block-walking for zero-value detection (v1 has documented gap)
8. **ERC-20 approve() tracking expansion** — approval_scope currently only tracks Permit2
9. **Audit 2,119 misclassified contracts in T2-eaef6a5d NULL bucket** — downgrade to UNANALYZED
10. **Recompute "46% suspected" statistic** excluding NULL bucket derivative flags
11. **Draft appeals/recourse policy** for `/methodology` endpoint
12. **Investigate org_004 (0xbaed383e)** — next organizational mapping target
13. **Evaluate hybrid cache architecture for `risk_scores` persistence** — deferred until scheduler audit complete. Decision criteria: bulk-query demand, longitudinal tracking need, acceptable staleness window. See Correction #6 and `reports/risk_scoring_persistence_audit.md`.
14. **Drain-wave USD attribution gap (flagged 2026-05-04)** — `approval_watchlist` schema captures `victim_address`, `contract_address`, `drain_caller`, `drain_tx_hash`, and `drain_timestamp`, but has no field for per-drain USD value. As of 2026-05-04, 3,437 lifetime drain events recorded across 94 distinct drainer wallets and 2,963 victims, but **zero USD attribution** on any drain. The 2026-05-04 capital-total query (`scripts/capital_total.py`) flagged this as the single largest unmeasured-harm gap in the corpus. Estimated impact: median drain ~$500-2000 → drain-wave alone is ~$1.7M–$6.9M unaccounted for. Fix: add `drain_value_usd` column + backfill via on-chain `transferFrom` log parsing (token amount × oracle price at block timestamp). Without this, every "how much have you seen drained" answer requires a hand-waved estimate.
15. **`trap_events.loss_estimate_usd` column unused (flagged 2026-05-04)** — schema has the field, **0 of 2,159 lifetime rows populated**. Either the field was deprecated and the column should be dropped, OR the populator was never built. The 2026-05-04 capital-total query surfaced this as the second-largest USD-attribution gap. Fix path A (drop): formal deprecation with migration; fix path B (populate): build a `trap_loss_estimator.py` module that joins `trap_events` against `transaction_events` for the trapped tx and computes USD value at trap-time. Path B is preferred if the data is reachable from existing tx records.

---

## What NOT to Build

- No trading logic, execution, or flash loans
- No contract deployment
- No interaction with flagged contracts — read-only always
- No ML models (explicit architectural decision)
- No polling of RPC endpoints (WebSocket + on-demand only)
- No silent data modifications (correction log for all revisions)
- No features that require the Latent Flux primitives unless explicitly specified (the original LF integration plan was superseded by threshold heuristics + z-scores)

---

## Reference Documents

- `L3_CONTEXT_ARCHITECTURE.md` — system state document (what Layer 3 IS)
- `L3_CONTEXT_INTELLIGENCE.md` — findings document (what Layer 3 HAS FOUND)
- `reports/correction_log.md` — every retracted or revised claim
- `reports/daily_brief_YYYY-MM-DD.md` — automated daily intelligence briefs
- `reports/case_ORG_001_*.md` — organizational case files

---

## Running Common Operations

```bash
# Status
sqlite3 data/surveillance.db "SELECT COUNT(*) FROM contracts;"
sqlite3 data/surveillance.db "SELECT COUNT(*) FROM transaction_events;"

# Analysis
python -m surveillance.camouflage_tracker --compute-today
python -m surveillance.trust_amplification --analyze
python -m surveillance.behavioral_baseline --compute --detect-anomalies
python -m surveillance.trend_forecaster --compute-today --forecast --score
python -m surveillance.risk_scoring --address 0x<CONTRACT>   # score one contract
python -m surveillance.risk_scoring --top 100                # rank by stored_potential
python -m surveillance.risk_scoring --family <family_id>     # score all members of a family
python -m surveillance.daily_brief --generate
python -m surveillance.strategy_fingerprint --classify-all --profile-baits --lifecycle
python -m surveillance.deployer_profiler --profile-all --cluster
python -m surveillance.goplus_enrichment --benchmark

# Investigation
python -m surveillance.case_file --address 0x[CONTRACT]
python -m surveillance.eth_depth --address 0x[ADDRESS]
python -m surveillance.deployer_profiler --find-similar 0x[ADDRESS]

# API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## First Steps for Any New Session

1. Read this file completely
2. Read `L3_CONTEXT_ARCHITECTURE.md` for current system state
3. Read `L3_CONTEXT_INTELLIGENCE.md` for current findings
4. Check `reports/correction_log.md` for recent revisions
5. Check the priority list above for current focus areas
6. For any specific task, read the modules that task will touch BEFORE writing code
7. Confirm understanding of scope and constraints before proceeding

When in doubt: ask, don't assume. The codebase has been built with specific architectural discipline. Breaking that discipline silently costs more than asking costs.
