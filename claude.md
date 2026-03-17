# Layer 3 — On-Chain Trap Surveillance System
## Claude Code Onboarding Brief

---

## What We Are Building

This is not an arbitrage bot. Do not build an arbitrage bot.

We are building a **passive surveillance system** that watches the on-chain behavior of trap deployers on Arbitrum — actors who deploy honeypot contracts, poison pill ERC-20s, and bait liquidity positions designed to exploit poorly-coded arbitrage bots.

**The position:**

- **Layer 1** — arbitrage bots scanning for price gaps, executing mechanically
- **Layer 2** — trap setters deploying contracts designed to exploit Layer 1 bots
- **Layer 3 (us)** — observing Layer 2 trap setters without their knowledge, building a structured database of their behavior, contracts, and deployment patterns

Layer 2 does not know Layer 3 exists. Their on-chain behavior is fully public and permanent. We have structural advantage: no capital at risk, no contracts deployed, no timing pressure. Pure intelligence gathering on a public ledger.

**Why this works:**
Trap setters are focused on bots. Their operational security is zero — everything they do is on a public chain. They cannot hide deployment patterns, cannot erase transaction history, and have no reason to change behavior because they don't know they're being watched. New trap deployers enter constantly. Human failure to write correct bot behavior is not a solvable problem. The surface never closes.

**The data is the product.** We are not building a trading tool. We are building a dataset that becomes more valuable every day it runs. Customers (Layer 1 operators) pay to avoid traps. The database sells itself once it's provably accurate.

---

## Design Philosophy — Non-Negotiable

**1. Ground truth only.**
A suspected trap and a confirmed trap are different tiers. Every database entry must have traceable provenance: how it was detected, what confirmed it, which on-chain event is the proof. No field gets filled with inference. `null` is an honest value. Fabrication is not.

**2. Conservative over aggressive.**
False negatives are acceptable. False positives destroy credibility permanently. If we cannot prove a contract is a trap, it stays `yellow` (suspected) or `unknown`. We never upgrade a label without on-chain confirmation.

**3. Immutable historical record.**
Once a confirmed event enters the database it is never edited — only appended to. New information creates a new entry with a timestamp. We do not retroactively revise history.

**4. Freshness matters.**
A trap flagged 48 hours after firing is useless to the operator who got burned in hour one. Detection must be near real-time. Stale data implies coverage we don't have.

**5. The schema is more important than the pipeline.**
Design the data structure first. The pipeline's only job is to populate that structure correctly. Do not write pipeline code before the schema is locked.

---

## Data Schema — Build This First

Before any pipeline code, implement this schema. Every field must be justifiable from on-chain data alone.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

class ConfidenceTier(Enum):
    UNKNOWN = "unknown"       # No signal yet
    SUSPECTED = "suspected"   # Bytecode exhibits trap patterns, unconfirmed
    CONFIRMED = "confirmed"   # Behavioral event has proven it

class DetectionMethod(Enum):
    BYTECODE_PATTERN = "bytecode_pattern"     # Static analysis of contract code
    BEHAVIORAL_TRIGGER = "behavioral_trigger" # Bot interacted, tx failed/reverted abnormally
    DEPLOYER_HISTORY = "deployer_history"     # Known trap deployer deployed this contract
    ROUTING_ANOMALY = "routing_anomaly"       # 1inch routes around despite apparent price advantage

@dataclass
class TrapEvent:
    """A single confirmed trap firing event."""
    trap_contract_address: str          # The trap contract
    bot_address: str                    # Address that got trapped
    tx_hash: str                        # Transaction hash of the failure
    block_number: int                   # Block number
    timestamp: datetime                 # Block timestamp
    loss_estimate_usd: Optional[float]  # Estimated loss if determinable, else null
    failure_signature: str              # Revert reason or failure pattern

@dataclass  
class ContractRecord:
    """One record per flagged contract address."""
    # Identity
    contract_address: str
    chain: str                          # "arbitrum" to start
    
    # Detection
    detection_method: DetectionMethod
    detection_timestamp: datetime
    detection_block: int
    
    # Confidence
    confidence_tier: ConfidenceTier
    confidence_reason: str              # Human-readable explanation of why this tier
    
    # Confirmation (null until confirmed)
    confirmation_tx_hash: Optional[str]
    confirmation_timestamp: Optional[datetime]
    confirmation_block: Optional[int]
    confirmation_event: Optional[TrapEvent]
    
    # Deployer
    deployer_address: str
    deployer_funding_source: Optional[str]  # Address that funded deployer, if traceable
    
    # 1inch routing
    routing_presence: bool              # Does 1inch route through this contract?
    routing_first_seen: Optional[datetime]
    
    # Bytecode signals (populated from static analysis)
    has_asymmetric_transfer: Optional[bool]
    has_conditional_revert: Optional[bool]
    has_unusual_fee_structure: Optional[bool]
    bytecode_pattern_notes: Optional[str]
    
    # History
    trap_events: list[TrapEvent] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DeployerRecord:
    """One record per deployer address."""
    deployer_address: str
    chain: str
    
    first_seen: datetime
    last_seen: datetime
    
    confirmed_trap_count: int
    suspected_trap_count: int
    total_contracts_deployed: int
    
    funding_sources: list[str]          # Addresses that funded this deployer
    known_associated_deployers: list[str]  # Other deployers in same cluster
    
    contracts: list[str]                # All contract addresses from this deployer
    
    # Behavioral fingerprint
    typical_deployment_interval_hours: Optional[float]
    typical_gas_price_gwei: Optional[float]
    deployment_pattern_notes: Optional[str]
```

---

## Build Sequence — Phase 1 Only

**Phase 1 goal:** Passive surveillance running continuously. No product. No API. Just data accumulation with high provenance quality.

Three components, build in this order:

### Component 1: Contract Deployment Monitor

Watch every new contract deployment on Arbitrum in real time via WebSocket.

```
Input: Arbitrum WebSocket node
Output: Stream of (contract_address, deployer_address, block_number, timestamp, bytecode)
```

- Connect to Arbitrum via WebSocket (`wss://arb-mainnet.g.alchemy.com/v2/...` or equivalent)
- Subscribe to new blocks
- For each block, extract all contract creation transactions (`to == null`)
- Extract: deployer address, deployed contract address, bytecode, block number, timestamp
- Push each new contract to the bytecode classifier immediately

Do NOT try to execute or interact with flagged contracts. Read-only.

### Component 2: Bytecode Classifier

Static analysis only. No execution. Classify each new contract as SUSPECTED or UNKNOWN based on bytecode patterns.

Trap signatures to detect (start with these, expand over time):

```python
TRAP_PATTERNS = {
    "asymmetric_transfer": [
        # CALLER == owner allows transfer, others revert
        # Look for: CALLER opcode followed by conditional REVERT
        # EVM opcodes: 0x33 (CALLER), 0xfd (REVERT), 0x57 (JUMPI)
        "conditional_revert_on_caller",
    ],
    "blacklist_check": [
        # mapping(address => bool) blacklisted; require(!blacklisted[msg.sender])
        # Look for: SLOAD of address-keyed mapping followed by conditional REVERT
        "address_keyed_sload_before_revert",
    ],
    "buy_not_sell": [
        # transfer() succeeds on buy (from DEX), reverts on sell (to DEX)
        # Look for: different code paths based on tx.origin or msg.sender
        "tx_origin_conditional",
    ],
    "callback_trap": [
        # Flash loan callback that doesn't complete repayment
        # Look for: callback signature without corresponding transfer back
        "incomplete_callback_pattern",
    ],
    "fee_on_transfer_anomaly": [
        # Fee logic that's asymmetric or concealed
        "hidden_fee_logic",
    ]
}
```

Output per contract: `{confidence_tier, detected_patterns, pattern_notes}`

**Important:** When in doubt, mark UNKNOWN. Do not force a classification.

### Component 3: 1inch Routing Monitor

Use the 1inch API (key available) to check whether flagged contracts are appearing in live routing.

**Endpoint to use:** `GET https://api.1inch.io/v5.2/42161/tokens` (Arbitrum chain ID: 42161)

Poll every 5 minutes for new tokens on Arbitrum. Cross-reference new token contract addresses against the deployment monitor. If a newly deployed contract appears in 1inch's token registry:
- Update `routing_presence: true` in the ContractRecord
- Upgrade priority of bytecode analysis (contracts with routing exposure catch real traffic)
- Flag for accelerated monitoring

**Also use:** `GET /v5.2/42161/quote` to check if a suspected trap token appears in routing paths between major pairs. If 1inch routes *around* a token that appears to offer better pricing, that's a `routing_anomaly` detection signal.

The 1inch API gives you two things the deployment monitor alone cannot: confirmation that a contract has real liquidity exposure, and implicit avoidance signals from the aggregator's own pathfinder behavior.

---

## How Latent Flux Primitives Map to This System

The README describes LF's architecture. Here is how the primitives apply to the surveillance system specifically:

**⧖ ReservoirState → Deployer behavioral baseline**
Each deployer address gets a ReservoirState that accumulates their deployment behavior over time: timing intervals, gas price patterns, contract interaction sequences. The reservoir builds a continuous behavioral fingerprint. Deviation from baseline (e.g., deployer suddenly changes gas pattern or deployment cadence) surfaces as drift — potential signal of new campaign.

**⊗ AttractorCompetition → Contract classification**
Each new contract gets classified against known attractor basins: `honeypot`, `poison_pill`, `bait_liquidity`, `legitimate`. The state (bytecode feature vector) flows toward the nearest attractor. Basin membership determines classification tier. This replaces brittle if/else rule matching with geometric pattern recognition that generalizes to novel trap variants.

**↺ RecursiveFlow → Deployer cluster resolution**
When a new deployer appears, run recursive flow to resolve cluster membership: does this deployer's funding source, gas pattern, and timing connect them to a known deployer cluster? The fixed-point iteration converges on either a known cluster label or isolates them as a new entity.

**◉ FoldReference → Data integrity enforcement**
Apply fold-reference critique at every pipeline stage to catch data quality failures: null addresses, impossible timestamps, NaN feature vectors, missing provenance fields. Corrections are logged. Nothing corrupt enters the database.

**≅ DriftEquivalence → Behavioral similarity matching**
When a new contract's bytecode feature vector is within tolerance of a known confirmed trap's feature vector, flag as SUSPECTED. The tolerance threshold is the key parameter — conservative at first, tightened as the corpus grows.

**Do not force LF primitives where they don't fit.** The WebSocket listener is vanilla Python async. The 1inch polling loop is vanilla REST. LF is the analysis layer on top of raw data ingestion — not a replacement for it.

---

## What NOT to Build

- No trading logic, no execution, no flash loans, no sandwiching
- No contract deployment of any kind
- No interaction with flagged contracts — read-only at all times
- No API layer yet — Phase 1 is data accumulation only
- No dashboard or visualization — that's Phase 3
- No price prediction or profit estimation
- Do not touch the backtest/ directory — that work is closed
- Do not attempt to recreate or extend the arbitrage research

---

## Infrastructure Notes

- **Chain:** Arbitrum One (chain ID 42161) only to start
- **Node:** Alchemy or QuickNode Arbitrum endpoint via WebSocket
- **1inch API:** Available (key to be provided) — Arbitrum endpoint at `https://api.1inch.io/v5.2/42161/`
- **Storage:** SQLite to start — simple, local, no infra overhead. Schema maps directly to the dataclasses above.
- **Language:** Python. Async where needed (WebSocket listener). Sync elsewhere.
- **Dependencies:** `web3.py`, `aiohttp`, `sqlite3` (stdlib), existing LF primitives

---

## Acceptance Criteria for Phase 1 Complete

1. WebSocket listener runs continuously on Arbitrum, captures all new contract deployments in real time
2. Every new deployment is immediately classified by the bytecode analyzer — UNKNOWN, SUSPECTED, or passes to confirmation queue
3. 1inch polling runs every 5 minutes, cross-references new tokens against deployment monitor, updates `routing_presence` fields
4. Every record in the database has complete provenance: detection method, detection timestamp, confidence tier with reason
5. No null addresses, no fabricated fields, no inferred classifications without on-chain evidence
6. The deployer record table links every contract to its deployer address
7. When a trap fires (bot interacts with SUSPECTED contract and fails), the record automatically upgrades to CONFIRMED with the TrapEvent attached
8. The system runs for 24 hours without crashing and produces at least one SUSPECTED classification from real Arbitrum data

**Falsification criteria:** If after 72 hours of running the database contains zero SUSPECTED entries, the bytecode classifier thresholds are too conservative and need tuning — not a failure of the system design.

---

## First Task

Read this brief completely. Then:

1. Design the SQLite schema from the dataclasses above — tables, indexes, foreign keys
2. Show the schema DDL for review before writing any other code
3. Wait for approval before proceeding to the pipeline components

The schema is the foundation. Everything else is built on top of it. Get it right before building anything else.