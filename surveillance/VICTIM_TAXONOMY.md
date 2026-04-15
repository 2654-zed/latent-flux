# Layer 3 Victim Taxonomy

Three victim tiers with distinct detection approaches, on-chain signatures, and customer bases.

## Tier M — Machine Victims

Bots, scanners, automated MEV strategies.

**On-chain signature:** High revert rates, known selectors (`9b3ca70c`, `b2460c48`, etc.), rapid-fire interactions, deployer-pattern address prefixes (`0xc0ffee`, `0x0984ad6f`).

**Detection approach:**
- Behavioral baseline analysis (strategy_lifecycle, bot_candidate_selectors)
- Revert pattern detection (revert_cluster_detector)
- Trap confirmation via self-loop and revert clustering
- Bytecode family matching (shared trap templates targeting known bot strategies)
- Bait profiling (bait_profiles table: which bait type targets which strategy)

**Current coverage:**
- 1,278 bot candidates tracked
- 8 strategy types classified (CALLBACK_ARBITRAGE, L2_OPTIMIZED_SCANNER, BLIND_SCANNER, etc.)
- 1,789 trap events (primarily Tier M — bots hitting traps)
- 51 bait profiles across 6 bait types

**Blind spots:**
- Novel strategy types not in the classifier
- Cross-chain bot rotation (bot changes address per chain)
- Bots that learn and stop hitting traps (survivorship bias in our data)

**Customer:** MEV operators, bot developers, protocol teams. They want to know which contracts will eat their gas, which strategies are saturated, and where the trap density is highest.

**Product:** Trap detection feed, strategy saturation index, bytecode family alerts.

---

## Tier H — Human Victims

Retail users, wallet holders, manual traders.

**On-chain signature:** Low nonce (1-10), single approval event, zero revert history, funds from exchanges/bridges, no prior interaction with the draining contract.

**Detection approach:**
- Stored potential risk scoring (risk_scoring.py — the loaded-gun detector)
- Approval exposure monitoring (approval_watchlist: 16,064 tracked, 14,611 pending drain)
- Trust amplification measurement (high caller diversity + low revert = camouflaged trap)
- SUSPECTED_HIGH_TRAFFIC escalation (suspected contracts accumulating retail callers)
- Pass-through detection (distinguishing real victim extraction from laundering flow)
- Address poisoning detection (Unicode homoglyph tokens, vanity address matching)

**Current coverage:**
- 16,064 approval exposures tracked across 8,409 unique victim addresses
- 1,463 confirmed drains
- Stored potential model: CRITICAL tier = ~1,700 contracts (actionable daily feed)
- Pass-through classifier separates real theft ($2.3M est.) from wash volume ($1.6M est.)
- 7 rogue Permit2 facilitators tracked across 3 chains
- Address poisoning detection via vanity prefix matching

**Blind spots:**
- Off-chain phishing (the initial approval signature happens off-chain; we see the Permit2 allowance but not the social engineering)
- Approval revocation tracking (we know who's exposed but not who revoked)
- Token-specific risk (we track USDC/USDT approvals but not all ERC-20s)
- The 21-day DELEGATECALL proxy gap (fixed with SUSPECTED_HIGH_TRAFFIC, but the general class of 0%-revert traps remains harder to detect than reverting traps)

**Customer:** Wallet providers, exchanges, compliance teams. They want to protect their users from approving malicious contracts, get alerts when their users' approvals are being exploited, and have defensible data for regulatory reporting.

**Product:** Approval exposure dashboard, stored potential risk scores, drain alert feed, pass-through vs real-drain classification.

---

## Tier A — Agent Victims

AI agents making autonomous on-chain transactions.

**On-chain signature:** Identical to Tier H for Permit2 drains (the chain doesn't know the signer is an AI agent). Distinguishing features: facilitator-mediated transactions (x402 pattern), EIP-3009 transferWithAuthorization calls, high-frequency micro-payments ($0.01-$10 range), agent-specific wallet addresses (often freshly generated, low nonce).

**Detection approach:**
- Facilitator validation (x402_facilitators registry: 60 known, 7 rogue)
- Approval scope enforcement (flag agents that grant UNLIMITED/NEVER Permit2 allowances)
- Facilitator behavioral fingerprinting (nonce velocity, self-settlement ratio, vanity address patterns)
- Agent-specific stored potential (same risk_scoring model but with approval_scope weighted higher — agents approve programmatically and may not verify scope)
- Prompt injection surface monitoring (not on-chain — requires integration with agent platforms)

**Current coverage:**
- x402 monitor: tracks EIP-3009 and Permit2 settlement patterns
- 71 facilitators classified (60 known, 4 unknown, 7 rogue)
- 1,059 Permit2 exposure rows across 837 unique owners
- X402_AGENT_DRAIN alert type with REAL_DRAIN / PASS_THROUGH classification
- Drain detection transfers directly from Tier H — same Permit2.transferFrom signature

**Why this tier matters now (not later):**

The infrastructure for draining AI agents already exists and is actively being used to drain humans. When the first AI agent gets drained, it will happen through one of these vectors:

1. **Compromised MCP server** returns a tool result containing a malicious approval request. The agent signs it because the tool result looks legitimate. On-chain, this is indistinguishable from a phished human signing a malicious approval.

2. **Prompt injection in API response** causes an agent to call `approve(Permit2, MAX_UINT256)` on a token it holds. The agent's framework passes the call through because it can't distinguish a legitimate payment authorization from an injected one.

3. **Rogue facilitator** in the x402 ecosystem submits a `transferWithAuthorization` that sweeps the agent's balance. The agent granted the authorization for a $0.01 API payment; the facilitator replays it for the full balance (if scope wasn't enforced).

4. **Vanity-spoofed facilitator** mimics a legitimate x402 payment processor. The agent's address book has a truncated address match, so it approves the spoofed facilitator.

In all four cases, the on-chain exhaust is identical to a Tier H drain. The detection transfers. The prevention doesn't — agents need pre-transaction validation that humans get from reading a MetaMask popup.

**Blind spots:**
- No agent-specific wallet identification yet (we can't distinguish an agent wallet from a human wallet on-chain)
- No MCP server / tool-use integration (the attack surface is off-chain)
- No approval scope enforcement tooling (agents need a library that refuses to grant unlimited approvals)
- The facilitator registry is manually curated — needs automated classification

**Customer:** AI agent platforms (Anthropic tool-use, OpenAI function-calling, autonomous agent frameworks), x402 ecosystem participants, MCP server operators. They want pre-transaction validation: "is this facilitator safe?", "is this approval scope reasonable?", "has this contract been flagged?"

**Product:** Facilitator validation API, approval scope policy engine, agent-specific risk scoring endpoint, real-time drain detection webhook.

---

## Cross-Tier Signals

Some detection capabilities serve all three tiers:

| Capability | Tier M | Tier H | Tier A |
|-----------|--------|--------|--------|
| Bytecode classification | Primary | Secondary | Secondary |
| Revert pattern detection | Primary | — | — |
| Approval exposure tracking | — | Primary | Primary |
| Stored potential scoring | — | Primary | Primary (higher weight on approval_scope) |
| Trust amplification | Secondary | Primary | Primary |
| Facilitator validation | — | — | Primary |
| Address poisoning detection | — | Primary | Primary |
| Pass-through classification | — | Primary | Primary |
| Deployer risk profiling | Primary | Primary | Secondary |
| Org attribution | Secondary | Secondary | — |

---

## Mapping to Existing Alert Types

| Alert Type | Primary Tier | Secondary |
|-----------|-------------|-----------|
| TRAP_CONFIRMED | M | — |
| HIGH_VELOCITY_DEPLOYER | M | H |
| DORMANT_ACTIVATION | M | — |
| WATCHLIST_HIT | M, H | — |
| TRUST_AMPLIFICATION | H | M |
| SUSPECTED_HIGH_TRAFFIC | H | M |
| COORDINATED_DEPLOYMENT | M, H | — |
| X402_AGENT_DRAIN | H | A |
| X402_FACILITATOR_UNKNOWN | A | H |
| BRIDGE_WITHDRAWAL | — | — (infra) |
| LAUNDRY_PIPELINE | — | — (infra) |

---

## Revenue Mapping

| Tier | Product | Customer | Pricing Model |
|------|---------|----------|--------------|
| M | Trap detection feed + strategy saturation | MEV operators, bot devs | Per-query or subscription (high volume, low margin) |
| H | Approval dashboard + stored potential scores | Wallets, exchanges, compliance | SaaS subscription (medium volume, high margin) |
| A | Facilitator validation API + scope enforcement | Agent platforms, x402 ecosystem | Per-validation or enterprise license (low volume today, high growth) |

Tier A is the growth bet. The victim population doesn't exist at scale yet, but the detection infrastructure is already built and the customer base (AI agent platforms) is the fastest-growing segment in crypto tooling.
