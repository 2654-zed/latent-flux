# Layer 3 Commercial Product Audit
## Deductive vs Inferential Framework

**Audit date:** 2026-04-09
**Auditor:** Claude Opus 4.6 (automated, against production data)
**Production corpus at audit time:** 80,633 contracts | 1,335,445 transaction events | 21,951 deployers | 3 chains

---

## Section 1: Classification of Every Customer-Facing Output

### Legend
- **D** = Deductive: verifiable on-chain fact or deterministic consequence of on-chain state
- **I** = Inferential: behavioral pattern matching, threshold heuristics, organizational attribution, or prediction
- **M** = Mixed: contains both D and I components not currently separated in the output

---

### 1.1 API Endpoints (web/api_v1.py)

#### GET /risk/{chain}/{address} and GET /check/{address}

| Output field | Classification | Verification method |
|---|---|---|
| `address`, `chain` | D | Input echo; trivially verifiable |
| `confidence_tier` (confirmed/suspected/unknown) | **M** | D component: "confirmed" requires on-chain revert from external caller (trap_events row with tx_hash). I component: "suspected" is a policy decision based on bytecode pattern count threshold (MIN_PATTERNS_FOR_SUSPECTED=1). A customer can verify the trap_event tx_hash on-chain but cannot independently reproduce the suspected classification without running the same bytecode analyzer with the same thresholds. |
| `confidence` (0.0-1.0 score) | **I** | Computed via a formula that weights tier assignment, bytecode signal count, and victim count. The formula is deterministic given inputs, but the inputs include inferential tier assignment. Not independently reproducible without access to the scoring algorithm. |
| `risk_level` (CRITICAL/HIGH/MEDIUM/LOW/UNKNOWN) | **I** | Derived from confidence score. Inherits all inferential components. |
| `detection_methods` (array of pattern names) | **D** | Each pattern name corresponds to a specific opcode sequence found at a specific byte offset in deployed bytecode. Customer can verify by reading the contract's bytecode and checking the cited offsets. |
| `bytecode_pattern_notes` | **D** | Contains byte offsets (e.g., "CALLER at 0x1a3 -> EQ at 0x1a8 -> JUMPI at 0x1b2 -> REVERT at 0x1c9"). Independently verifiable by disassembling the contract. |
| `revert_statistics.total_interactions` | **D** | Count of rows in transaction_events for this contract. Verifiable by replaying blocks and counting. |
| `revert_statistics.reverts` | **D** | Count of is_reverted=1 rows. Each corresponds to a receipt with status=0 on-chain. |
| `revert_statistics.revert_rate` | **D** | Arithmetic: reverts / total_interactions. Deterministic. |
| `revert_statistics.unique_victims` | **D** | COUNT(DISTINCT interacting_address). Each address verifiable on-chain. |
| `attribution.deployer` | **D** | tx.from of the contract creation transaction. On-chain fact. |
| `attribution.deployer_contracts_total` | **D** | Count of contracts WHERE deployer_address = X. Each contract's creation tx is on-chain. |
| `attribution.org_id` | **I** | Organizational attribution via funding-chain tracing. Multi-hop inference from on-chain value transfers. Could be wrong: shared funders don't prove shared operators; the same CEX withdrawal address may fund unrelated parties. |
| `attribution.timezone` | **I** | Inferred from deployment hour distribution (peak hours mapped to timezone labels). The hour distribution is deductive; the timezone label is inferential. An operator could deploy from a VPN or automated scheduler that doesn't reflect their actual timezone. |
| `attribution.technique` | **M** | D: which bytecode patterns the deployer's contracts exhibit. I: the label "delegatecall" or "callback" as a technique classification is a categorical inference from pattern presence. |
| `approval_exposure.pending_approvals` | **D** | Count of approval_watchlist rows WHERE drain_detected=0 AND contract_address=X. Each row traces to an on-chain approve() tx. |
| `approval_exposure.approvals_drained` | **D** | Count of approval_watchlist rows WHERE drain_detected=1. Each drain row traces to an on-chain transferFrom() or deployer interaction tx. |
| `bytecode_family.family_id` | **I** | Cluster membership assigned by bytecode_families.py based on bytecode_pattern_notes prefix similarity. The clustering algorithm is a policy choice; different prefix lengths or similarity metrics would produce different families. |
| `bytecode_family.family_size` | **D** given family membership | Count of contracts in the family. Arithmetic on the clustering result. |
| `external_benchmark.goplus` | **M** | D: the GoPlus API was called and returned these flags. I: the `match_vs_layer3` comparison assumes both systems are measuring the same thing. GoPlus and Layer 3 use different methodologies; "L3_ONLY" means Layer 3 flagged and GoPlus didn't, but GoPlus may have different detection scope, not necessarily a "miss." |
| `trust_amplification.amplification_factor` | **M** | D: callers_per_day is arithmetic on transaction_events. I: family_avg_callers_per_day depends on bytecode family membership (inferential clustering). The ratio itself is deterministic given the inputs, but one input is inferential. |
| `trust_amplification.router_percentage` | **D** | Fraction of callers using selector 0x3593564c. Each call is an on-chain transaction. |
| `diamond_model` | **I** | Formal adversary profiling (adversary, capability, infrastructure, victims). Every component involves attribution and organizational inference. |
| `extraction_events` | **M** | D: the tx hashes and value amounts cited in extraction events are on-chain. I: the narrative ("extraction cycle") is an interpretation of the transaction sequence. |
| `detector_precision` | **M** | D: the count of times each detector fired is arithmetic. I: "precision" implies a ground-truth label for each firing (was it really a trap?), which is only available for the confirmed tier. Precision on suspected-tier firings is unknown. |

#### POST /screen (batch)

| Output field | Classification | Notes |
|---|---|---|
| Per-address `risk_level` | **I** | Same as /risk endpoint. |
| Per-address `confidence` | **I** | Same formula. |

#### GET /feed and GET /feed/stats

| Output field | Classification | Notes |
|---|---|---|
| Alert events (array) | **M per alert type** | See alert classification below. |
| Alert counts by severity | **D** | Arithmetic on alerts table. |

#### GET /org and GET /org/{org_id}

| Output field | Classification | Notes |
|---|---|---|
| `org_id` | **I** | Organizational identity is an attribution, not a measured property. |
| `scale.deployers`, `scale.contracts` | **M** | D: count of addresses linked via funding chain. I: the funding chain itself is multi-hop inference. The API now includes `attribution_method` and `methodology_note` to make this explicit. |
| `operational_tempo.timezone_inference` | **I** | See attribution.timezone above. |
| `operational_tempo.techniques` | **M** | See attribution.technique above. |

#### GET /deployer/{address}

| Output field | Classification | Notes |
|---|---|---|
| `profile` (gas, timezone, technique, cadence) | **M** | D: all metrics are arithmetic on transaction_events. I: the labels ("americas_afternoon", "burst", "delegatecall") are categorical inferences. |
| `similar_deployers` (similarity >= 0.70) | **I** | Behavioral similarity is a weighted composite of 6 dimensions. The score is deterministic given inputs, but the weights and threshold are policy choices. A different weighting would produce different matches. |
| `funding_trail` | **M** | D: the value transfer from funder to deployer is an on-chain tx. I: interpreting this as "organizational link" is inference. The funder could be a CEX withdrawal, a shared service, or an unrelated party. |

#### GET /ecosystem/stats

| Output field | Classification | Notes |
|---|---|---|
| `corpus.contracts`, `corpus.deployers`, `corpus.bots` | **D** | Row counts on tables. |
| `detection.confirmed_threats` | **D** | Count of contracts WHERE confidence_tier='confirmed'. Each confirmation traces to a trap_event with on-chain tx_hash. |
| `detection.suspected_threats` | **I** | Count of contracts WHERE confidence_tier='suspected'. Suspected is a classification decision. |
| `detection.camouflage_ratio` | **M** | D: the ratio is arithmetic (count of contracts with revert_rate < 0.10 / total with 10+ interactions). I: the 10% threshold is arbitrary. |
| `organizations.mapped` | **I** | Count of distinct org_ids. Organizational mapping is inferential. |
| `organizations.wallet_rotations_detected` | **I** | Behavioral similarity matching. |

---

### 1.2 Alert Types (alert_engine.py, x402_monitor.py, event_monitors.py)

| Alert type | Classification | D component | I component |
|---|---|---|---|
| `TRAP_CONFIRMED` | **M** | D: external caller reverted on this contract (tx_hash verifiable). | I: interpreting the revert as "trap confirmation" assumes the revert was caused by the contract's trap mechanism, not a bug or user error. |
| `HIGH_VELOCITY_DEPLOYER` | **D** | Deployer exceeded N contracts in this session. Each deployment is an on-chain creation tx. | — (threshold is configurable but the count is factual) |
| `WATCHLIST_HIT` | **D** | Address matched a watchlist entry. The deployment is on-chain. | — (the watchlist itself is curated, but the match is deterministic) |
| `X402_FACILITATOR_UNKNOWN` | **M** | D: an EIP-3009 or Permit2 selector was called by this address. | I: "unknown facilitator" means the address isn't in our registry. It could be a legitimate non-x402 relayer. |
| `X402_AGENT_DRAIN` | **M** | D: Permit2.transferFrom was called, calldata decoded, self-settlement shape detected, amount verified. | I: "drain" interpretation assumes the transfer was unauthorized. Path B additionally requires prior `rogue` classification (inferential). |
| `BRIDGE_WITHDRAWAL` | **D** | Direct call to L2StandardBridge or ArbSys with decoded value and L1 recipient. All on-chain. | — |
| `DORMANT_ACTIVATION` | **M** | D: a contract that had zero interactions now has interactions. | I: "activation" implies intent; the contract may have been called by an unrelated scanner. |
| `APPROVAL_DRAIN` | **M** | D: approve() was followed by transferFrom() on the same contract. | I: temporal sequence doesn't prove causation; the transferFrom may be unrelated to the approval. |

---

### 1.3 Daily Brief Sections (daily_brief.py)

| Section | Classification | Notes |
|---|---|---|
| Key Metrics vs Yesterday | **D** | Arithmetic on daily_metrics table. |
| Prediction Scorecard | **M** | D: actual values are measured. I: predicted values are forecasts. Hit/miss evaluation is D (comparison). |
| Watchlist Alerts | **D** | Address match against curated list. |
| Trust Amplification Alerts | **M** | See trust_amplification analysis above. |
| Camouflage Watch | **M** | See camouflage ratio above. |
| Behavioral Anomalies (z-scores) | **M** | D: z-score is arithmetic on behavioral baseline. I: "anomaly" threshold is a policy choice. |
| Diamond Model Index | **I** | Adversary profiling is organizational attribution. |
| Strategy Lifecycle Monitor | **I** | Saturation indices and lifecycle stage assignments are model-based predictions. |
| Bait Detection Summary | **I** | Bait type classification is inferential pattern matching. |
| Zero-Day Trap Watch | **M** | D: self-test detection (deployer is sole caller, 10+ calls). I: "zero-day" label implies future activation intent. |
| Approval Drain Watchlist | **M** | D: pending approval count. I: "at risk" framing implies future drain will occur. |
| x402 Activity | **M** | D: event counts and exposure counts. I: facilitator classification and "stored potential" framing. |

---

### 1.4 Classification Counts

| Classification | Count of distinct output fields/sections |
|---|---|
| **D (Deductive)** | 18 |
| **I (Inferential)** | 16 |
| **M (Mixed)** | 22 |
| **Total** | 56 |

**39% of outputs are mixed.** The product does not currently separate the deductive components from the inferential components in most of its mixed outputs. A customer receiving a risk_level=CRITICAL cannot distinguish which parts of that assessment are on-chain facts vs which are heuristic judgments without reading the methodology documentation.

---

## Section 2: Three Epistemic Product Tiers

### Tier A — Provable Intelligence (Deductive Only)

Claims Layer 3 can make that a customer can independently verify with their own RPC endpoint. These survive any audit, any challenge, any "prove it" request.

| Claim | Verification method |
|---|---|
| "This contract's bytecode contains CALLER->EQ->JUMPI->REVERT at offsets 0x1a3, 0x1a8, 0x1b2, 0x1c9" | `eth_getCode(address)` + disassemble. Customer checks cited offsets. |
| "This contract has been called 1,243 times with 1,156 reverts (93% revert rate)" | Replay blocks, count receipts with status=0 vs status=1. |
| "This deployer created 23 contracts" | Query creation txs where tx.from = deployer_address. |
| "This deployer was funded by address X in tx Y" | `eth_getTransactionByHash(Y)`. Value transfer is on-chain. |
| "This address has a Permit2 allowance of MAX_UINT160 to spender Z, never-expiring" | `eth_call` to Permit2.allowance(owner, token, spender). Anyone can read this. |
| "This address's USDC balance is 0 after tx Z" | `eth_call` to USDC.balanceOf(address) at the block after tx Z. |
| "This tx called Permit2.transferFrom with decoded payer=A, recipient=B, amount=C, token=D" | `eth_getTransactionByHash` + decode calldata. Receipt confirms status=1. |
| "This contract was called by 89 distinct addresses" | COUNT(DISTINCT interacting_address) from replayed blocks. |
| "Address X called bridge contract Y with value Z ETH at block N" | `eth_getTransactionByHash`. All fields on-chain. |
| "79.2% of contracts with 10+ interactions have revert rates below 10%" | Replay all monitored contracts, count reverts, apply threshold. Arithmetic. |
| "This contract's callers arrive at 504/day; its bytecode family averages 12/day" | Count distinct callers, divide by span. Count family siblings. Arithmetic on on-chain data (given family membership). |

**What Tier A cannot include:** organizational attribution, timezone inference, confidence scores, risk levels, facilitator classifications, predictions, or any label that requires a judgment call.

**Customer value proposition:** "We read the chain continuously and present verified on-chain facts you'd have to replay blocks to compute yourself. Every number traces to a tx hash or contract address you can check."

---

### Tier B — Assessed Intelligence (Inferential, With Evidence Chain)

Claims that require behavioral analysis, pattern matching, or organizational attribution. Layer 3 adds unique value beyond raw on-chain data, but these claims carry epistemic risk.

| Claim | Evidence chain (deductive base) | Failure mode | Confidence methodology |
|---|---|---|---|
| "This contract is a suspected trap" | Bytecode contains 1+ trap patterns (specific offsets cited). Deductive base: the patterns exist. | The patterns could appear in non-malicious contracts (false positive). The bytecode classifier has 10 detectors; each has a precision rate documented in detector_precision. | `confidence_tier = 'suspected'` when MIN_PATTERNS_FOR_SUSPECTED (1) fires. Precision per detector is tracked empirically. |
| "This contract is a confirmed trap" | External caller reverted (trap_event with tx_hash). Deductive base: the revert happened. | The revert could be caused by a bug, insufficient gas, or user error, not the trap mechanism. | Behavioral confirmation requires the caller to be a non-deployer address. Confidence = 0.90+ because external-victim-revert is the strongest available signal. |
| "These 308 deployers are part of org_001" | Funding chain: each deployer received ETH from an address in the org_001 funding tree. Deductive base: each value transfer is on-chain. | Shared funders don't prove shared operators. A CEX withdrawal, a gas station service, or an airdrop could create spurious funding links. | `attribution_method: 'funding_chain'` with explicit methodology_note. Conservative count vs expanded count both published. |
| "This operator's timezone is Americas afternoon" | Deployment hours cluster at UTC 18:00-22:00. Deductive base: each deployment timestamp is on-chain. | VPNs, automation scripts, or timezone-shifted operators would produce misleading distributions. | Peak-hour concentration metric published. High-concentration (>40% in 4-hour window) increases confidence. |
| "These two deployers are the same operator (wallet rotation)" | Behavioral similarity >=0.85 across 6 dimensions + deployer A's last_seen precedes deployer B's first_seen. Deductive base: all dimension metrics are arithmetic on on-chain data. | Two unrelated operators could coincidentally share timezone, gas patterns, and technique preferences. 0.85 threshold is arbitrary. | Similarity score published with per-dimension breakdown. Customer can evaluate which dimensions carry weight for their use case. |
| "This facilitator is rogue" | Self-settlement pattern (tx.from == decoded.to), victims have unlimited never-expiring Permit2 allowances, victim current balance = 0. Deductive base: all three conditions are on-chain reads. | The pattern could be a legitimate batch-settlement operation where the facilitator temporarily holds funds. The balance-zero check is a snapshot, not a proof of unauthorized extraction. | Classification requires forensic investigation + human judgment. Tagged as 'rogue' only after spot-check verification. Source field documents the evidence chain. |
| "This bytecode family is a TaaS (Trap-as-a-Service) distribution" | 435 deployers from 2 funders use the same bytecode template. Deductive base: bytecode similarity, deployer counts, funder counts are all on-chain. | Template sharing could be coincidental (common Solidity compiler output) or from a legitimate service (OpenZeppelin clones). The 2-funder concentration is suggestive but not proof of TaaS. | Family size, unique deployers, funder count all published. Cross-deployer flag indicates family spans multiple operators. |

**What Tier B adds over Tier A:** Context. Attribution. Pattern recognition. Behavioral history. The things that turn "this contract has a 93% revert rate" (Tier A fact) into "this contract is one of 23 deployed by the same operator who runs a timestamp-activated trap fleet on Base" (Tier B assessment).

**Customer value proposition:** "We add behavioral intelligence that no raw on-chain data provider offers. Every assessment includes the evidence chain so you can evaluate the inference yourself. We publish our confidence methodology and our correction log so you know what we've gotten wrong before."

---

### Tier C — Predictive Intelligence (Forward-Looking, Explicitly Speculative)

Claims about what will or could happen. Highest value, highest risk.

| Prediction | Stored potential (deductive base) | Assumption about human behavior (inferential gap) | Confirmation / falsification |
|---|---|---|---|
| "These 13,954 undrained approvals are at risk of being drained" | Each approval is an on-chain approve() tx to a suspected/confirmed trap contract. The approval persists until revoked. | The trap operator will choose to exercise the drain. Many trap operators never drain (1,462 of 15,416 = 9.5% drain rate). 90.5% of tracked approvals have NOT been drained. | Confirmed: drain_detected flag flips when transferFrom is observed. Falsified: approval revoked without drain. |
| "This dormant fleet will activate" | 75,509 contracts have zero interactions. Some deployers hold multiple pre-staged contracts. | The operator will choose to add liquidity and attract victims. Historical activation rate is low (~1.2% in the corpus). | Confirmed: first external interaction detected. Falsified: contract remains dormant beyond a reasonable window. |
| "This Permit2 allowance will be consumed" | Permit2.allowance() returns unlimited, never-expiring for the (victim, token, drainer) triple. On-chain fact. | The drainer will call Permit2.transferFrom. The victim will not revoke. The victim will deposit new funds. All three are human decisions. | Confirmed: transferFrom observed. Empirical base: we observed 2 of 3 channels exercised on victim 0x785ce546. |
| Compositional zero-day threat synthesis (8 attack configurations in POTENTIAL_ATTACKS.md) | Each component cites specific table rows, addresses, or selector hashes from the corpus. | The components will be assembled into the described configuration. No full chain has been observed end-to-end. | Confirmed: full attack chain observed in the wild. Partially confirmed: individual components observed (e.g., Attack 1's permission harvesting validated by Permit2 drain finding). |
| "org_002 will sustain 48+ contracts/day" | Week 2 deployment rate was 48.4 contracts/day (9.22x acceleration from Week 1). | The operator will maintain their infrastructure and operational tempo. | Confirmed/falsified by next-week deployment count. Short-term predictions on established operators are the most reliable category. |

**What Tier C adds over Tier B:** Forward-looking intelligence. The transition from "this is what exists" to "this is what will happen." The stored-potential bridge makes this possible: deductive state reads + inferential behavioral models + explicit speculation about human decisions.

**Customer value proposition:** "We measure stored potential — the gap between what exists on-chain and what could happen next. Our predictions are explicitly speculative, grounded in deductive state reads, and falsifiable. We track our prediction accuracy in the daily brief scorecard."

---

## Section 3: Pricing Structure Analysis

### Current Pricing Tiers

| Pricing tier | Monthly | Current features | Epistemic tier delivered |
|---|---|---|---|
| Tier 1 — Blocklist API | $500/mo | /risk, /check, /screen (single contract screening) | **Mostly M (mixed)** — the primary output is `risk_level` and `confidence`, both inferential. The deductive components (revert stats, bytecode patterns, approval exposure) are embedded in the response but the headline output is a classification judgment. |
| Tier 2 — Intelligence Feed | $2,500/mo | +/feed, /feed/stats, /watch, /ecosystem/stats | **Mostly I** — alerts are mixed (see classification above), ecosystem stats include inferential counts (suspected threats, organizations mapped, rotation detections). The deductive components (event counts, alert timestamps) are metadata, not the value proposition. |
| Tier 3 — Full Platform | $5,000/mo | +/org, /deployer, /contract (full intelligence) | **All three (A+B+C)** — organizational intelligence is Tier B (assessed), deployer profiling is Tier B, predictions and dormant fleet tracking are Tier C. The deductive base (funding chains, deployment facts) supports but doesn't constitute the intelligence product. |

### Mismatch Analysis

**Tier 1 mismatch:** A customer paying $500/mo for a "Blocklist API" receives a risk_level that looks like a binary safe/unsafe signal but is actually a mixed-epistemic output. A customer who wants only Tier A (provable facts) gets Tier A facts buried inside a Tier B wrapper. A customer who wants liability-safe screening gets an inferential judgment they can't fully audit without understanding the scoring algorithm.

**Tier 2 mismatch:** Intelligence Feed customers receive alerts that range from fully deductive (BRIDGE_WITHDRAWAL, HIGH_VELOCITY_DEPLOYER) to heavily inferential (DORMANT_ACTIVATION intent, X402_FACILITATOR_UNKNOWN classification). The feed doesn't distinguish alert epistemic status — a BRIDGE_WITHDRAWAL (provable) and a DORMANT_ACTIVATION (interpreted) appear at the same level of apparent certainty.

**Tier 3 is correctly positioned:** Full Platform customers explicitly buy organizational intelligence and predictive analysis. They understand they're getting assessed intelligence. The methodology documentation in API responses (attribution_method, methodology_note, rotation criteria) supports informed consumption.

### Should pricing restructure around epistemic tiers?

**No — but the epistemic status should be explicit in every output.**

Restructuring pricing around epistemic tiers would create an awkward product: "Pay $500 for facts only, $2,500 for facts + assessments, $5,000 for everything." The problem is that customers don't buy epistemic purity; they buy screening, alerting, and intelligence. A customer asking "is this contract safe?" wants an answer, not a dissertation on what's provable.

**The fix is labeling, not restructuring:**

Every API response should include an `epistemic_status` field:
```json
{
  "risk_level": "CRITICAL",
  "epistemic_status": {
    "classification": "assessed",
    "deductive_base": ["bytecode patterns at cited offsets", "93% revert rate on 1,243 interactions", "12 pending approvals"],
    "inferential_components": ["confidence score formula", "suspected tier threshold"],
    "verification": "Customer can verify deductive base via eth_getCode + block replay. Classification methodology published at [link]."
  }
}
```

This lets:
- **Liability-conscious customers** (exchanges, compliance teams) consume only the deductive base and make their own classification decisions
- **Operational customers** (MEV operators, trading desks) consume the full assessed output including risk_level and confidence score
- **Intelligence customers** consume Tier B/C organizational and predictive intelligence with full evidence chains

### Where does Layer 3's moat actually live?

**The moat is in Tier B — the inferential layer.**

- **Tier A (deductive)** is reproducible. Any team with Alchemy access and a bytecode disassembler can verify the same on-chain facts. The competitive advantage is freshness and coverage (continuous monitoring vs one-time scan), not unique insight.

- **Tier B (inferential)** is where Layer 3 has no competitors. Wallet rotation detection, organizational attribution, behavioral fingerprinting, trust amplification analysis, and the self-loop promotion rule are novel analytical methods that require both the continuous data collection infrastructure AND the analytical methodology. A competitor would need to rebuild 23 days of behavioral history to start producing comparable assessments.

- **Tier C (predictive)** is the highest-value output but the thinnest evidence base. Predictions about dormant fleet activation, drain timing, and organizational tempo are grounded in Tier A/B data but depend on assumptions about human behavior. Prediction accuracy is tracked but the track record is short (23 days).

**The honest moat statement:** "Our facts are reproducible. Our analysis is novel. Our predictions are early. The correction log proves we know the difference."

---

## Section 4: Honest Product Descriptions

### Tier 1 — Contract Screening ($500/mo)

**What you get:** Real-time risk assessment for any contract on Base, Arbitrum, or Optimism. Query by address; receive a risk profile with on-chain evidence and behavioral classification.

**What's provable:** Every risk profile includes verifiable on-chain facts: bytecode patterns with byte offsets you can check by disassembling the contract. Revert statistics computed from replayed blocks. Approval exposure counts traced to specific approve() transactions. Deployer identity from the creation transaction.

**What's assessed:** The risk_level (CRITICAL/HIGH/MEDIUM/LOW) and confidence score are behavioral classifications, not measurements. They are computed from a published algorithm that combines on-chain facts with threshold-based heuristics. The thresholds are policy choices informed by 23 days of continuous surveillance and 1,462 confirmed drains. The algorithm is deterministic given inputs, but reasonable people could disagree on the thresholds.

**What we've gotten wrong:** CORRECTIONS.md documents 15+ corrections to prior claims including inflated organization sizes, unverifiable benchmark comparisons, and stale-data artifacts. Every correction includes root cause and fix. The corrections are part of the product, not a bug — they demonstrate that the system detects and corrects its own errors.

**Verification:** Every deductive component in a risk profile can be independently verified with an RPC endpoint. The API includes methodology documentation for every assessed component. We publish our detector precision rates so you can evaluate false-positive risk for your use case.

### Tier 2 — Intelligence Feed ($2,500/mo)

**What you get:** Real-time alerts on trap confirmations, high-velocity deployers, bridge withdrawals, Permit2 drains, dormant fleet activations, and watchlist hits. Plus ecosystem-level statistics on detection coverage, camouflage ratios, and organizational activity.

**What's provable:** Alert triggers are on-chain events: a contract was called and reverted (TRAP_CONFIRMED), a deployer created N contracts (HIGH_VELOCITY_DEPLOYER), a Permit2.transferFrom was executed (X402_AGENT_DRAIN), a bridge withdrawal was submitted (BRIDGE_WITHDRAWAL). Each alert includes tx_hash and block_number for independent verification.

**What's assessed:** Alert interpretation involves judgment. A TRAP_CONFIRMED alert means an external caller reverted — but reverts have many causes besides trap mechanisms. A DORMANT_ACTIVATION alert means a previously-quiet contract received its first interaction — but the interaction may be a scanner probe, not a malicious activation. A X402_FACILITATOR_UNKNOWN alert means an EIP-3009 call came from an unregistered address — but most EIP-3009 traffic is legitimate gasless payments, not x402 abuse (empirically: 0/61 facilitators in our production data match the public x402 registry).

**What's predictive:** The feed includes stored-potential metrics: 13,954 undrained approvals, 44,811 armed-but-unfired bytecode patterns, 2,402 Permit2 exposures. These are deductive counts of on-chain state. The implication that they represent "risk" is predictive — only 9.5% of tracked approvals have historically been drained.

**Energy-to-truth ratio:** In our operational history, we've produced 8 verified major findings, 2 high-confidence inferences that could go either way, 4 findings that were wrong and corrected, and 1 finding that was misleadingly framed. The correction rate is declining as we've institutionalized production-first verification.

### Tier 3 — Full Intelligence Platform ($5,000/mo)

**What you get:** Everything in Tiers 1 and 2, plus organizational intelligence (mapped criminal enterprises with deployer networks, funding chains, and operational profiles), deployer behavioral fingerprinting (timezone inference, gas patterns, technique preferences, wallet rotation detection), and full case file access.

**What's provable:** Funding chains trace on-chain value transfers. Deployer profiles are arithmetic on deployment timestamps, gas prices, and bytecode patterns — all on-chain data. Behavioral similarity scores are deterministic given the weighting formula.

**What's assessed:** Organizational attribution ("these 308 deployers are part of org_001") is the core Tier 3 product and it is explicitly inferential. We trace funding chains on-chain but interpret them as organizational links — an inference that could be wrong when funders are shared services, CEXes, or unrelated parties. We detect wallet rotations via behavioral similarity (>=0.85 threshold) but two unrelated operators could coincidentally match. We infer operator timezones from deployment hours but operators could use automation or VPNs.

**What's predictive:** Tier 3 includes compositional threat intelligence — 8 hypothetical attack configurations built from observed components. Each cites specific corpus evidence. None have been observed end-to-end. Two have partial empirical validation (the Permit2 drain finding validated the Permission Harvesting pattern; the coffee fleet activation validated the Dormant Fleet pattern). The rest are forward-looking threat synthesis.

**The correction log as trust infrastructure:** CORRECTIONS.md is not a weakness. It is the mechanism by which we distinguish what we know from what we claimed. Every customer should read it. A security vendor that has never corrected itself either has never been wrong (unlikely) or has never checked (dangerous). We check.

**The moat:** Nobody else has 23 days of continuous behavioral surveillance across 80,633 contracts on 3 chains with organizational attribution, wallet rotation detection, and predictive threat synthesis. The data accumulates daily. The analytical methods are novel. The correction discipline is public. A competitor starting today cannot replicate the behavioral history; they can only reproduce the on-chain facts.

---

## Summary Statistics

| Epistemic classification | Count of distinct customer-facing outputs |
|---|---|
| D (Deductive) | 18 (32%) |
| I (Inferential) | 16 (29%) |
| M (Mixed, unseparated) | 22 (39%) |

**Where the commercial moat lives:** Tier B (inferential analysis). The deductive base is reproducible by anyone with RPC access. The predictions are too early to have a track record. The behavioral analysis — organizational attribution, wallet rotation detection, trust amplification, self-loop promotion, drain pattern forensics — is where Layer 3 produces value no one else can.

**The single most defensible claim Layer 3 can make to a paying customer:**

> "We continuously monitor 80,633 contracts across 3 chains. Of the 15,416 approvals we've tracked on suspected trap contracts, 1,462 have been drained (9.5%). We can show you every approval on every contract you're considering interacting with, tell you which ones have been drained before, and measure the behavioral patterns of the deployers behind them. The on-chain facts are independently verifiable. The behavioral assessments are documented with explicit methodology and a public correction log. When we've been wrong, we've said so."

That claim contains: a deductive fact (15,416 approvals tracked, 1,462 drained), a behavioral assessment (suspected trap contracts), a service description (continuous monitoring), and a trust mechanism (correction log). It does not overclaim. It does not make an inferential claim sound deductive. It positions the epistemic discipline as a feature, not a footnote.
