# L3_TOPOLOGY_FRAMEWORK

**Purpose:** Interpretive framework for Layer 3 analysis. Reference this when analyzing any contract, address, or system component. Overrides the default behavioral-snapshot interpretation.

---

## The Five Primitives

When evaluating any node (contract, EOA, protocol, facilitator), score across these five dimensions rather than observed behavior alone.

### 1. Position

Where does this node sit relative to user assets? Can it observe, intercept, or modify transactions?

| Indicator | Example | How to read it |
|-----------|---------|----------------|
| Transaction-insight permission | Router, aggregator, MEV bot | Privileged observer — sees user intent before execution |
| DELEGATECALL capability | Proxy contract | Privileged modifier — can change semantics of calls routed through it |
| Approval target | Permit2, token spender | Privileged actor — can move user assets without re-consent |
| Custody holder | Vault, lending pool | Privileged retainer — holds user assets directly |

A contract with a privileged position has the capability to act on user assets regardless of its current behavior.

### 2. Permissions

What edges exist between this node and user assets? Map the **maximum permission surface**, not the currently exercised permissions.

| Edge type | Key questions |
|-----------|---------------|
| Approval scope | UNLIMITED or bounded? Expiration NEVER or finite? Which tokens? |
| Execution rights | Can it call arbitrary functions on behalf of users? (multicall, execute) |
| Delegation authority | Can it delegate its own rights to other contracts? |
| Upgrade authority | Can it change its own implementation? |

A permission granted is a permission available. An unexercised approval with UNLIMITED/NEVER scope is functionally identical to an exercised one.

### 3. Trust Bindings

What assumptions cause users or systems to treat this node as safe? Trust bindings are **assumptions attackers exploit** — they are liability surfaces, not safety guarantees.

| Binding | Exploitation vector |
|---------|---------------------|
| Router endorsement (1inch, Uniswap routing) | "It's in the router so it must be safe" → attacker injects into router path |
| Low revert rate | "It works" → camouflaged trap activates only on specific conditions |
| Audit status | "It was audited" → audit was for v1, current implementation is v7 |
| Installation history | "Lots of users use it" → sybil installation or downstream dependency |
| GoPlus/scanner clean | "Third-party cleared it" → scanner tested wrong invariant or was bypassed |

**High trust binding = less scrutiny = more dangerous.** The most attractive attack surfaces are the ones with the strongest trust bindings, because those are the ones analysts stop examining.

### 4. Mutability

Can this node change its behavior without re-consent?

| Mechanism | Signal | Volatility |
|-----------|--------|------------|
| Proxy upgrade | EIP-1967 slot writable, owner not renounced | HIGH (2.5x) |
| Version bump | Multiple implementations in history | MEDIUM |
| Implementation swap | DELEGATECALL to mutable address | HIGH (2.5x) |
| Unburned ownership | `owner()` returns non-zero, non-timelock | MEDIUM |
| Timestamp-gated logic | Behavior changes at predetermined block/time | MEDIUM (2.0x) |
| SELFDESTRUCT capability | Can erase evidence + redeploy | CRITICAL (3.0x, 3.5x with CREATE2) |

Mutability is volatility. A node with high mutability is a node where the analysis you did yesterday may be irrelevant today.

### 5. Observation Capability

What can this node see? Observation capability compounds over time and enables future exploitation even without current malicious behavior.

| Observation type | Exploitation path |
|------------------|-------------------|
| Transaction inputs | Replay or reorder attacks |
| Address graphs | Target selection for phishing, poisoning |
| Fund flows | Behavioral profiling of operators |
| Approval state | Drain target prioritization |
| Behavioral patterns | Automation bypass construction |

A node that has observed 100,000 transactions has built a model of its users that it can act on at any future point. Observation is potential energy.

---

## Risk Assessment

```
risk = (position × permissions × mutability) × (1 / trust_binding_strength)
```

Current behavior is **not** a term in this equation. Behavior is the outcome that potential produces when exercised. Score the potential.

---

## The Interpretive Inversion

**A node with privileged position, broad permissions, high mutability, strong trust binding, and zero malicious behavior is at maximum stored potential — not minimum risk.**

This is the rule that changes how every data point reads.

| Default interpretation | Topology interpretation |
|------------------------|------------------------|
| "0% revert rate → legitimate" | "0% revert rate with broad permissions → camouflaged trap awaiting trigger" |
| "Contract hasn't fired → low risk" | "Contract has capability to fire + hasn't fired → peak stored potential" |
| "Router-endorsed → safe" | "Router-endorsed → trust binding active, scrutiny suppressed" |
| "Well-audited → low risk" | "Well-audited v1, current v7 → trust binding from obsolete verification" |
| "100K users → trusted" | "100K users → observation model built, extraction can be timed" |

The detection model shifts from "what did this do?" to "what is this positioned to do?"

---

## Application to Layer 3 Data

When analyzing any finding, evaluate the five primitives using the data Layer 3 already produces:

| Primitive | Layer 3 data sources |
|-----------|---------------------|
| Position | `contracts.routing_presence`, `contracts.bytecode_pattern_notes` (DELEGATECALL), `approval_watchlist` (as spender) |
| Permissions | `approval_events`, `approval_watchlist` (scope + expiration), `x402_permit2_exposure` |
| Trust bindings | `goplus_results`, `trust_amplification` (router %), `contracts.routing_presence`, `entity_classification` (verified org) |
| Mutability | `contracts.has_delegatecall`, `contracts.bytecode_pattern_notes` (SELFDESTRUCT, CREATE2), `upgrade_events`, `timelock_countdowns`, `deployers.entity_type` (ownership status) |
| Observation | `transaction_events.count`, `approval_watchlist.victim_count`, `contracts.total_unique_callers` |

The `risk_scoring.py` module implements this framework: `stored_potential` captures position + permissions + deployer_risk + org_context; `volatility` multiplier captures mutability; `realized_value` in the denominator ensures that exercised behavior **reduces** the score (spent potential is less dangerous than stored potential).

---

## Practical Interpretive Rules

When reviewing any finding, apply these rules before concluding:

1. **A suspicious-looking contract that has done nothing is more dangerous, not less.** Do not downgrade based on absence of malicious behavior.

2. **A third-party "safe" classification is evidence of trust binding, not safety.** Weight scanner clean results inversely — they increase the attack value of the node.

3. **Mutability converts yesterday's analysis into tomorrow's obsolete data.** Never rely on a point-in-time classification for a mutable node. Re-assess at interaction time.

4. **Observation capability accumulates.** A node that has seen N transactions has a model worth N attacks. Old nodes with many users are not safer; they are richer in attack material.

5. **Trust bindings are attack surfaces ranked by binding strength.** The highest-value targets have the strongest trust bindings — that's where the compound risk lives.

6. **Position determines what's possible; permissions determine what's permitted; mutability determines what can change; trust bindings determine what will be noticed; observation capability determines what's known.** All five compound. None are sufficient alone. None are safe to ignore.

---

## When to Reference This Document

Reference this framework when:
- Interpreting any contract, address, or entity finding
- Writing risk-related analysis, alerts, case files, or customer-facing output
- Explaining why a "clean-looking" contract should still be flagged
- Designing new detectors — ensure they measure potential, not behavior
- Evaluating trust-binding sources (routers, audits, scanners) — their endorsement is a vulnerability vector, not a validation

The framework complements CLAUDE.md (ground truth, conservative classification, immutable record) by providing the interpretive lens that turns ground-truth data into actionable intelligence.
