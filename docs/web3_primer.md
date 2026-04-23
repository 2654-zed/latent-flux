# Web3 Primer — Substrate for Reading Layer 3

**Audience:** traditional-cybersecurity reader, no prior web3 exposure.
**Purpose:** minimum-viable substrate to read Layer 3's corpus, lexicon, and `claude.md` — nothing broader.

This is not a neutral web3 tutorial. Layer 3 takes a specific analytical stance — compositional harm over code bugs, observation over intervention, behavior over intent — and this primer delivers the substrate that stance needs. Where Layer 3's vocabulary collides with mainstream web3 usage (the word "operator" is the clearest case), the primer flags it. Much of the posture will feel familiar from traditional security — indicators over motive, attack-surface thinking, trust-boundary analysis. The unfamiliar parts are the substrate.

Reading order after this primer: `docs/lexicon.md`, then `docs/layer3_consumable_intelligence.md`, then `claude.md`.

---

## 1. The substrate: EOAs, contracts, chains, gas

A blockchain address is either an externally-owned account (EOA) — controlled by a secp256k1 keypair — or a contract account, which is a blob of bytecode deployed at a deterministic address. Only EOAs can originate transactions. Contracts execute code in response to being called. Every transaction is a signed tuple (from, to, value, data, gas, nonce), and blocks give those transactions a canonical ordering.

Layer 3 monitors Base, Arbitrum, and Optimism. These are three separate L2 rollups — chains that batch user transactions, execute them cheaply off-mainnet, and periodically post compressed state proofs back to Ethereum mainnet for final settlement. Users get faster and cheaper transactions; Ethereum mainnet contributes security. Base, Arbitrum, and Optimism are three independent rollups with three independent state machines; a contract on Base does not exist on Arbitrum unless separately deployed there, and an EOA's history on one is not visible to the other. For reputation purposes, treat them as three different namespaces: a deployer with six months of Base history and zero Ethereum mainnet history is *new on mainnet*, period. The corpus records this distinction in `deployers.chain` and enriches each deployer with `mainnet_first_tx` — the first time that EOA transacted on Ethereum mainnet, or the empty string if never. This column is the substrate for Pattern D (Cross-Chain Reputation Import): an adversary active on one chain can present as "established" on another, and the mainnet-first-tx gap exposes the gap between actual longevity and apparent longevity.

Gas is the per-instruction cost of executing a transaction, paid in the chain's native token. Users pay to read the chain free; they pay to *act*. This matters later because the user pays gas to revoke a bad approval just as the attacker pays gas to exploit one — the cost asymmetry is the attacker's, not the user's (this is Cost-Habituation Asymmetry in the lexicon).

The substrate claim: contracts execute deterministically. Given the same state and input, the output is fixed. This is precisely why the interesting question is not whether code has bugs.

## 2. Tokens, approvals, Permit2 — the permission model

ERC-20 tokens are not held by the owner in any literal sense. They live as a balance mapping inside the token contract: `balances[0xAlice] = 1000`. There are two ways Alice's balance can decrease. She calls `transfer(Bob, 100)` herself, or she has previously called `approve(Spender, N)` — and Spender now calls `transferFrom(Alice, Bob, 100)`, which the token contract permits because `allowances[Alice][Spender] >= 100`.

The key word is *previously*. Approvals are standing authorizations. The closest TradFi analogue is a pre-signed withdrawal slip with no amount cap, no expiry, and no notification when used. Most DEX frontends (DEX = decentralized exchange, e.g. Uniswap — covered in §3) default to approving `type(uint256).max` — "all of it, forever" — because re-prompting for each trade is bad UX. Every one of those approvals is durable. When the contract that received the approval turns out to have been a trap, or has its admin key rotated, or gets upgraded via proxy, the approval remains live until explicitly revoked — which itself costs gas.

Permit2 is Uniswap's attempt to compress this. A user grants one approval to the Permit2 router, and individual protocols ask Permit2 for time-bounded, amount-bounded sub-permissions via signed messages. It concentrates approval risk at a single, audited choke point — at the cost of making Permit2 itself a maximum-leverage target.

Layer 3's `approval_watchlist` records every approval granted to a `suspected` or `confirmed` contract, and flips `drain_detected` from 0 to 1 at the moment a sweep is observed. The risk-scoring model's `approval_scope_score` (0–25) comes from this substrate.

Permissions are an independent axis from code correctness. A contract whose code is perfect but which holds `type(uint256).max` approvals from 10,000 users is a different kind of object than a contract with the same code and no approvals. Stored Potential begins here.

## 3. AMMs, routers, aggregators — the trading graph

A Uniswap V2 pool is a single contract holding two token balances `x` and `y`, enforcing the constant-product invariant `x * y = k` on every trade: a trader who pulls some `y` out must deposit enough `x` to keep the product constant, and the current ratio of balances is the price. No order book, no market maker — the math is the market maker. Liquidity providers deposit a matched pair and receive LP tokens proportional to their share; fees accrue to the pool and are claimable against those tokens. Uniswap V3 generalizes this with concentrated liquidity — the math is more complex, the shape is the same: *users surrender custody of tokens into a single contract*. This is pooled custody. A compromise of one pool drains every LP in it simultaneously, without per-LP interaction. The lexicon entry is Pooled Custody Amplification.

On top of pools sit routers and aggregators. Uniswap's Universal Router composes multi-hop paths across V2/V3/V4 pools in a single call. 1inch, CowSwap, and OpenOcean do the same across every DEX on-chain. The user picks a quote in a frontend; the aggregator picks *the contracts that receive the user's funds in transit*. Those contracts are a function of quoted price, not user intent. A malicious or negligent contract that offers competitive quotes will be routed into silently.

This is the Trust Amplification Factor: a contract's realized traffic is not bounded by the users who chose it, because aggregators choose on their behalf. Layer 3's `trust_amplification` table records `router_percentage` (what fraction of a contract's callers came via a known aggregator) and `amplification_factor` (traffic volume vs family baseline). A contract with high `amplification_factor` is being *trusted by the aggregator*, and that trust flows to every user who ever hit "swap."

The framing claim here: decentralization does not remove risk. It removes counterparty risk — there is no broker to front-run you or go insolvent — and replaces it with compositional risk, which is the joint behavior of pools, routers, aggregators, and the contracts they happen to route through. The surface moves. It does not shrink.

## 4. Bridges and cross-chain plumbing: LayerZero, DVNs, OFT, CCTP

A bridge moves value between chains. Ethereum cannot directly see Base's state, so the canonical pattern is: lock tokens on chain A, mint a representation on chain B. Unlocking the original requires a proof, delivered by a verifier, that the mint happened (or was authorized). Every bridge is a pooled-custody amplifier: the escrow on chain A holds the capital of everyone who has ever bridged.

LayerZero is a generic cross-chain messaging layer. Its core contract, `EndpointV2`, lets any protocol send a message from chain A to chain B. Who verifies that the message is legitimate on delivery? A **DVN** (Decentralized Verifier Network) — one or more parties that sign off that the source-chain event happened. Each protocol configures its own DVN set via `EndpointV2.getConfig`. The safe posture is N-of-M with N ≥ 2 and diverse verifiers. In the degenerate case, a protocol configures 1-of-1: a single DVN signs, and its compromise is sufficient to authorize arbitrary cross-chain mints. The code behaves correctly in both configurations — the only difference is a configuration value.

**OFT** (Omnichain Fungible Token) is LayerZero's token template. An OFT adapter is the contract that consumes LayerZero messages and performs the burn/mint. It trusts the endpoint; the endpoint trusts the configured DVN set.

Circle's **CCTP v2** is an architectural alternative. Instead of a verifier network, CCTP burns USDC on the source chain and mints the equivalent on the destination chain under Circle's own attestation. It is centralized along the attestation axis and decentralized along the execution axis. Layer 3's `infrastructure_registry` seeds the CCTP v2 contracts (four classifications, including MessageTransmitter and TokenMessenger) across Base, Arbitrum, and Optimism as known-legitimate — twelve rows total. Via **CREATE2** (a deployment opcode that derives the contract address from `keccak256(0xff ++ deployer ++ salt ++ keccak256(bytecode))`, making addresses reproducible across chains when the inputs match), CCTP v2 occupies the same canonical address on every chain — which is why a single classification entry suffices per contract type.

The Kelp / rsETH incident (recorded in the corpus as EXTRACTION_008, ~$292M lost) is the canonical Compositional Harm case. The Kelp OFT adapter was correctly written. LayerZero's EndpointV2 was correctly written. The DVN that signed was correctly written. rsETH's token accounting was correctly written. The harm emerged from the *composition*: the OFT adapter trusted the endpoint, the endpoint trusted a 1-of-1 DVN, the DVN was compromised off-chain, arbitrary rsETH was minted on Ethereum and backed against real Kelp custody. Every component passed its own audit. The 1-of-1 DVN was a **Configuration-Level Vulnerability** — not code, not a bug — with CRITICAL Stored Potential for the 56.7 days it remained live. This is the case that forces the lexicon.

## 5. Mutability: proxies, admin keys, configuration as code

Most non-trivial contracts on-chain are proxies. A proxy is a small contract with a single responsibility: `delegatecall` every incoming call to a separate *implementation* contract, while keeping the proxy's own storage. The user interacts with the proxy's address — which is stable — while the logic behind it can be swapped by calling the proxy's admin function. The proxy's storage persists; the behavior changes.

Who can swap? Whoever holds the admin key. For some protocols this is a DAO (decentralized autonomous organization — the protocol's on-chain governance body, typically tokenholders voting on proposals) whose votes drive a multisig with a timelock. For others it is an EOA controlled by one person. The distinction does not appear in the bytecode at a glance; it appears in what the admin slot is set to and what governs that address.

Configuration is the softer sibling. `EndpointV2.getConfig` returns the DVN set for a given protocol. The storage slot holding that configuration is writable by the protocol's admin. Changing it from 2-of-3 to 1-of-1 is a configuration change, not a code deployment. No bytecode moves. No audit is automatically re-triggered. Observers who watch code changes will not see it; observers who read configuration state at decision-time will.

Layer 3 treats mutability as an independent axis of Stored Potential. The risk-scoring model's volatility multiplier (1.0x to 3.0x) reflects this: fixed/burned keys score 1.0x; timestamp-gated logic 2.0x; DELEGATECALL with non-renounced ownership 2.5x; SELFDESTRUCT present 3.0x. Same code, same balance — different posture — different score. The `proxy_upgrade_watcher` module watches implementation slots for changes to catch the upgrade moment.

This is **Configuration-Level Vulnerability**: the surface where the same bytecode has materially different behavior depending on values a privileged address can change.

## 6. The adversarial landscape: bots, MEV, traps, drains

Pending transactions are publicly visible in the **mempool** — each node's pool of signed but unconfirmed transactions waiting to be included in a block — before they confirm. This is the basis of **MEV** (Maximal Extractable Value): a searcher sees Alice's pending swap, calculates that her trade will move the price, inserts a buy in front of her and a sell after her, and pockets the difference. Sandwiching, backrunning, and liquidation-racing are all MEV variants. MEV is a tax visible to anyone reading the mempool.

Deployment is itself a public event. A bot scanning new contracts will call a fresh deployment within seconds of its first block — probing for a known-exploitable pattern, trying common **function selectors** (the first four bytes of `keccak256(functionSignature)`; a contract dispatches an incoming call by matching the call's first four bytes against its selectors, so the set of selectors a contract exposes is effectively its ABI fingerprint), and checking for a held balance to drain. Bots match on *selector fingerprint* (which selectors a contract exposes) and *bytecode family* (which template it was copied from — a cluster of contracts sharing a common implementation is a "family" in Layer 3's `bytecode_families` schema). Layer 3's `selector_monitor`, `bot_candidates`, and `revert_cluster_detector` are the substrate for identifying these bots.

**Traps** are the inverse. An operator deploys a contract that *looks* like a known-exploitable target — the right selectors, the right balances, the right surface — but reverts on the exploit call and silently extracts the bot's gas and any submitted funds. Every revert is logged in `trap_events` with the trap contract, the bot address, the failure signature, and a loss estimate. At snapshot time the corpus has 1,867 confirmed trap events; contracts with at least one trap event get promoted to `confidence_tier = 'confirmed'` (84.84% positive predictive value — PPV — per Correction #9).

**Drains** work the other way. A user has previously granted approval to a contract. When the operator is ready, a sweep transaction calls `transferFrom` against every granted allowance, moving tokens from victims to a collector address. The `drain_events` table records these; `approval_watchlist.drain_detected` flips from 0 to 1 at the moment of sweep.

Vocabulary note: Layer 3 uses **"operator"** to mean the adversarial actor running a fleet of deployers. In mainstream web3 (Lido node operator, sequencer operator, etc.) "operator" is a legitimate role. Throughout the L3 corpus, assume the adversarial sense unless otherwise specified.

## 7. What transparency does and doesn't give you

Every transaction on Base, Arbitrum, and Optimism is publicly readable. Every contract's bytecode is publicly readable. Every token balance is publicly computable. This is real and worth something — but it does not mean intent is visible. It means *behavior* is visible, and nothing else.

Adversaries read the same data. A sophisticated operator deploying a trap fleet will calibrate revert rates to match the legitimate-aggregator-routed baseline: if real DEX routers revert on roughly 25% of calls (from *slippage* — the price moving between the quote and the on-chain execution — and from hitting gas limits), and a free scanner flags contracts with revert rates above 40%, then a trap calibrated to 20% is invisible. This is the **Camouflage Ratio**. Layer 3 measures it in `camouflage_metrics`, split between population-wide (`camouflage_ratio`) and adversary-set (`adversary_low_revert_ratio`, restricted to suspected/confirmed contracts). The confirmed-only rate at snapshot is 25.4% — confirmed traps revert *more* than the population average, not less, which itself corrected a prior framework claim (Correction #13).

A second-order effect: when detection heuristics are published, adversaries adopt them as calibration targets. This is **Publishing-Induced Recursive Evasion**. The `goplus_enrichment` module benchmarks this directly — contracts Layer 3 flags as confirmed often clear GoPlus's scan, because the visible detection surface has already been optimized around.

Layer 3 handles this with two disciplines. First, every output carries an epistemic tier. **Tier A** is deductive — verifiable on-chain by anyone with the block data (a bytecode flag, a transfer event, a deployment timestamp). **Tier B** is inferential — methodology-applied judgment (a risk score, a candidate-cluster assignment). The `/verify` endpoint returns only Tier A; `/methodology/*` publishes the scoring algorithms so consumers can audit them. Second, every retracted or revised claim lands in `reports/correction_log.md` with a root-cause entry. The two are complementary: tiers signal confidence; the log signals when confidence turned out to have been wrong.

Public detection is attack surface. The framing claim is not rhetorical.

## 8. Putting it together: Layer 3's posture

Layer 3's analytical unit is not a vulnerability. It is **Stored Potential** — a measurement of how much harm a node *could* produce if the topology around it composed adversarially — expressed as a 0–125 score with five components. Stored Potential is what the risk-scoring model outputs live per request; the `risk_tier` label (MINIMAL / LOW / MEDIUM / HIGH / CRITICAL) is a Tier B composite of it.

The substrate for the measurement is the Adversarial Topology Framework:

1. **Position** — where the node sits relative to user assets; can it observe, intercept, or modify?
2. **Permissions** — what edges exist between the node and user assets, at *maximum* scope, not currently-exercised scope?
3. **Trust bindings** — what assumptions cause users or protocols to treat the node as safe?
4. **Mutability** — can the node change behavior without re-consent?
5. **Observation capability** — what can the node see? Addresses, amounts, timing, behavioral patterns?

These five primitives transfer out of blockchain. A browser extension has position, permissions, trust bindings, mutability, and observation capability. So does an AI agent with tool access, or a SaaS integration with OAuth scopes. This is the piece of Layer 3 that plugs directly into a traditional security posture — the framework's ambition is to work anywhere assets and permissions compose.

The corpus is read-only by construction. *No trading logic, no contract deployment, no interaction with flagged contracts.* The discipline is in `claude.md` as "What NOT to Build," and it is not a style preference. Observation is the method. Acting would contaminate the measurement (Layer 3's own transactions would appear in the data it is analyzing) and expand the attack surface (every privileged action is new Stored Potential on Layer 3's side). The correction log exists because the observational stance implies accountability for claims — every revised claim has a root cause, every retracted one has a note.

After this primer: read `docs/lexicon.md` first, then `docs/layer3_consumable_intelligence.md` for the schema surface, then `claude.md` for working discipline. `docs/lexicon_reasoning_experiment.md` is optional — it shows what this substrate buys you when five frontier models each read the lexicon cold.
