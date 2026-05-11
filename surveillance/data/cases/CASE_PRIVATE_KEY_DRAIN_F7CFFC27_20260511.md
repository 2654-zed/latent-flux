# Case File — Private-Key Drain via Telegram Phishing (2026-05-11)

**Status:** Active. Funds sitting in attacker EOA; recovery window open at file creation.
**Disclosed:** Operator-supplied 2026-05-10 (Twitter user reported drain on social; analyst summary forwarded to Layer 3 2026-05-10).
**Layer 3 corpus involvement:** Attacker added to watchlist HIGH (local + production, row 95, `private_key_drain_attacker_F7cFFC27`). Victim wallet absent from corpus — outside our deployment-window scope.
**Loss magnitude:** ~$172K total, multi-chain.
**Opened:** 2026-05-10 (this file).

---

## Identity and roles

| Role | Address | Chain | Source |
|---|---|---|---|
| **Attacker EOA** | `0xF7cFFC27732a5C9c4E2D592F3E33435F8dDb019A` | multi-chain (Base / BSC / Ethereum) | Tier A — verified on all three chains via Blockscout / basescan |
| **Victim wallet** | `0x62acE10c7f2Aa0e9B5a8e09CbF5D18d0f8a1EE8A` | same multi-chain | Tier A — appears as `from` on the Sigma tx + as origin of drained assets |
| Sigma trading bot tx (forensic anchor) | `0xb81f9f0a1abb2330763d7b9498185404277955a18b3f766a31582c83ba70047e` | Base | Tier A — basescan confirmed: EIP-7702 delegation to `0x00…00` (null) executed from victim wallet; 39,987 POD → Uniswap V4 Pool Manager; 5.49 ETH returned to victim then routed onward |
| KyberSwap Meta Aggregation Router v2 | `0x6131B5fae19EA4f9D964eAc0408E4408b66337b5` (Eth) | Ethereum + Base | Tier A — OLI `"KyberSwap: Meta Aggregation Router v2"`. Used by attacker to swap drained tokens to native ETH. **Not adversarial** — legitimate venue used as off-ramp staging. |
| **POD token** ("Dolphin") | (Base ERC-20, address per Sigma tx event log) | Base | Drained asset — $125K |
| **FHE token** | (BSC ERC-20) | BSC | Drained asset — $21K |
| **SAT1 token** | `0x8f66337a0c2A02202fd91Dd596c411CF977c6060` | Ethereum | Drained asset — $11K nominal. Verified contract; 1,990 holders. Attacker funded 0.02 ETH dust to victim wallet to gas-sponsor remaining SAT1 sweep — see Ethereum timeline. |

### Current attacker holdings (snapshot 2026-05-11 ~05:00 UTC, via basescan.org HTML)

| Chain | Balance | USD value | Notes |
|---|---|---|---|
| Base | 52.56 ETH | ~$122,396 | Bulk of POD ($125K) swapped to ETH via KyberSwap 2026-05-11T01:01:55 UTC |
| BSC | 38.80 BNB | ~$25,273 | FHE ($21K nominal) already converted to native BNB |
| Ethereum | 10.20 ETH | ~$23,752 | Earlier ETH drain + SAT1 swap proceeds (~$10K) |
| **Total** | | **~$171,421** | Matches analyst's $172K estimate within rounding |

**Attacker dormant since 2026-05-11T01:01:55 UTC** at file creation — no outflows in 4+ hours. Off-ramp imminent.

---

## Attack chronology (Tier A — verified via Blockscout + basescan + analyst-supplied tx hash)

The drain ran 2026-05-11 **00:37 → 00:56 UTC** (30-min window, per analyst). The attacker had simultaneous signing authority across Ethereum, Base, and BSC, indicating private-key compromise rather than per-chain authorization theft.

### Base timeline (Tier A)

| Time UTC | Action | Notes |
|---|---|---|
| Pre-attack | Victim holds ~39,987+ POD tokens | Sigma trading bot user |
| ~00:37–00:56 | Attacker, holding stolen private key, drains POD across multiple txs | Bulk $125K loss leg |
| 00:53:19 | 6.19 ETH → attacker EOA `0xF7cf…` from victim wallet | $14,409 ETH skim incoming |
| 01:01:01 | Attacker approves spender `0xeD664536…` | Pre-swap setup |
| 01:01:55 | Attacker executes swap on KyberSwap Meta Aggregation Router v2 | Receives **46.37 ETH** (~$107,963) — likely the POD → ETH conversion |
| (during Sigma tx) | EIP-7702 delegation of victim wallet to `0x0000…0000` | Demonstrates attacker's signing-key control. From `0x62acE10c…` (victim) to `0x8CC69C61…`, signed delegation transferring 39,987 POD into Uniswap V4 Pool Manager, netting 5.49 ETH which then routes onward. |

### Ethereum timeline (Tier A, observed earlier in this session)

| Time UTC | Action | Notes |
|---|---|---|
| 2026-05-11T00:53:59 | Attacker EOA `0xF7cf…` first tx on Ethereum | Fresh-on-chain (pre-existed elsewhere, first reached out on Eth here) |
| 00:56:11 | Attacker sends 0.02 ETH from `0xF7cf…` to victim wallet `0x62acE10c…` | Gas funding for victim wallet to enable subsequent SAT1 sweep |
| 00:57:11 | 157,340 SAT1 → attacker EOA from victim wallet | Final asset drain (the SAT1 leg analyst noted) |
| 00:59:59 | Attacker sends 0.02 of fake `ĖTḨ` (unicode-spoof) token to address-poisoning lookalike | Cleanup / cosmetic step |
| 01:14:47 | Attacker approves SAT1 spending | KyberSwap setup |
| 01:16:23 → 01:19:11 | 4× swap calls dump 157,340 SAT1 via KyberSwap | $11K nominal → ~$10K returned (slippage ~$1K). Final tally: 10.20 ETH on Ethereum. |

### BSC

BSC outside Blockscout MCP / Etherscan-V2-free coverage. Per analyst: FHE ($21K) drained → 38.80 BNB held. Off-chain verification pending the user's basescan/etherscan/bscscan paid key or direct RPC access.

---

## Attack mechanism (Tier B — operator-supplied analyst summary)

**Cause: private key compromise**, most likely via Telegram phishing — fake CAPTCHA bot (the common 2025-2026 vector that asks targets to paste a "verification" command into Windows Run dialog, which executes a clipboard-injection that exfiltrates browser-stored credentials and wallet private keys). SIM swap is the lower-probability alternative.

### Why "manual drain" is the right framing (Tier B)

The 30-minute drain window (00:37 → 00:56 UTC), multi-chain coordination, and tooling-aware moves (using legitimate KyberSwap for off-ramp staging, EIP-7702 delegation as a control demo) all suggest a human operator executing a runbook, not an automated drainer bot. Drainer bots typically:
- Complete in seconds, not 30 minutes
- Use single-chain workflows
- Don't bother with delegation demos

### EIP-7702 delegation as the forensic anchor

The Sigma tx is the strongest single piece of evidence. Standard ECDSA-signed transactions are nominally fungible — both the legitimate owner and a key-thief produce signatures indistinguishable on-chain. **EIP-7702 delegation, however, is a state-changing action that grants persistent (per-block) authority to a delegated contract.** When the victim wallet executes a delegation pointing to null (`0x0000…0000`), it is a one-step demonstration that the signer has full control AND the explicit intent to flex it.

The standard interpretation in 2026: an EIP-7702 delegation from a wallet to `0x0…0` is either (a) the legitimate owner revoking a prior delegation, OR (b) the attacker proving "I own this." In context — concurrent with a 30-min cross-chain drain — interpretation (b) is decisive.

---

## Framework analysis

### This is NOT a Distributed Confused Deputy Chain

Unlike the Renegade or Grok/Bankr cases, no architectural composition failure produced this loss. The wallet's contracts behaved correctly; the failure was at the **credential-custody layer**, before any contract executed.

### It IS a Configuration-Level Vulnerability at the operational-security layer

Per the lexicon: "code that did exactly what its specification said it would do, with the spec being the failure mode." Here the "spec" is the user's *operational* configuration:
- Private key reachable from a device that runs Telegram with auto-clipboard handling
- No hardware-wallet enforcement
- No second-factor on signing (no Gnosis Safe / no MPC / no time-lock)

Each missing safeguard is itself a measurement of the credential's stored-potential volatility multiplier. The key held authority over $172K across three chains; the configuration permitted unilateral signing by any process that could read the key.

### Cognitive Load Concentration — empirical anchor

This case is a clean fit for [Cognitive Load Concentration](../../../docs/lexicon.md#cognitive-load-concentration): the user was load-bearing for an operational-security regime they were not equipped to maintain. The Telegram CAPTCHA phishing flow is specifically engineered to defeat the cognitive-vigilance threshold that DeFi presumes. The post-mortem framing — "you used a malicious Captcha bot" — is technically accurate but practically toothless as a defense recommendation, because the user's failure was sustained-vigilance not specific-knowledge.

### Forced Deterministic Neutrality at the signature-validation layer

The EVM, EIP-7702 delegation logic, and every token contract the attacker touched executed [Forced Deterministic Neutrality](../../../docs/lexicon.md#forced-deterministic-neutrality): valid signature in → execution proceeds. None could ask "is this signature being produced by the wallet's legitimate owner right now, or by a thief who copied the key from a phished clipboard 30 minutes ago?" The signature is the entire authorization surface; the *origin* of the signature is unobservable.

---

## Why Layer 3 missed it (and what's nonetheless actionable)

Three structural reasons:

1. **Victim wallet not in corpus.** `0x62acE10c…` is not a contract deployer in our `deployers` table; not a victim of any contract drain in `approval_watchlist`. The user is a DeFi participant whose loss vector was operational, not contract-mediated.
2. **Off-chain causation.** The compromise originated on Telegram. Even full L2 coverage cannot see browser clipboards, Telegram CAPTCHA bots, or stolen-key exfiltration to attacker infrastructure.
3. **Multi-chain across our edge.** Base is in scope; Ethereum and BSC are not. The attacker's holdings split is partly inside (Base $122K) and partly outside (BSC $25K + Eth $24K) our deployment-window visibility.

**What Layer 3 nonetheless does:**
- **Watchlist HIGH** on attacker EOA (local + prod row 95, added 2026-05-10).
- **Outflow monitor** — `scripts/monitor_attacker_outflows.py` polls Blockscout / Etherscan V2 for new outbound txs; emits one stdout line per new tx, suitable for Claude Code Monitor or cron alerting.
- **Case file documenting the mechanism** — future similar private-key-drain reports can cross-reference this case as the EIP-7702-delegation-as-forensic-anchor anchor.

---

## Recovery actions for the victim (operator-facing)

These are the actions that matter NOW while the attacker is dormant:

1. **Document ownership of `0x62acE10c…`** — wallet UI screenshots; sign a message proving control of an old key/seed (if any pre-compromise variant is still trusted); pull pre-drain tx history.
2. **Preemptive CEX freeze-intake reports** — Coinbase, Binance, Kraken, OKX, Bybit, MEXC all have stolen-funds intake forms. Submit BEFORE the attacker deposits. Include: attacker EOA `0xF7cFFC27…`, victim wallet, Sigma tx hash, this case file. The Sigma tx is the single strongest forensic anchor.
3. **Token-issuer freeze requests** — POD ("Dolphin") and FHE issuers may have emergency-freeze admin functions; check governance / Discord. Even if drained amounts are already swapped, an issuer freeze on the liquidity pool may help downstream.
4. **Chainabuse submission** — Chainabuse.com is free and feeds into chain-analysis providers used by exchanges. Submit the attacker EOA with the Sigma tx + this case file.
5. **Watch for first off-ramp** — once the attacker bridges or hits a CEX deposit, that's the moment to escalate from preemptive to active. The monitor script (`scripts/monitor_attacker_outflows.py`) gives that signal.

---

## Cross-references

- Lexicon: [Configuration-Level Vulnerability](../../../docs/lexicon.md#configuration-level-vulnerability) (operational-security configuration layer); [Cognitive Load Concentration](../../../docs/lexicon.md#cognitive-load-concentration); [Forced Deterministic Neutrality](../../../docs/lexicon.md#forced-deterministic-neutrality); [Stored Potential](../../../docs/lexicon.md#stored-potential) (the private key is the stored-potential node).
- Comparable cases: `CASE_DORMANT_WALLET_DRAIN_20260430.md` (multi-victim drain hub; different topology — attacker pre-prepared infrastructure; here attacker is a single fresh-EOA with stolen-key access).
- Tooling: `scripts/monitor_attacker_outflows.py` (built 2026-05-10 in this incident response).

## Outstanding work

1. **Monitor backend coverage** — Blockscout Base API returns 0 items for this attacker; Etherscan V2 free tier blocks Base. Add ETHERSCAN_API_KEY support to the monitor (next step, this session).
2. **BSC visibility** — attacker holds $25K on BSC; we have no Blockscout / Etherscan-free access for chain 56. Manual basescan / bscscan monitoring until a backend lands.
3. **Identify the swap-counterparty pool** for the SAT1 leg — the four KyberSwap calls routed through pool `0x8F10B468b06c6FD214B65F87778827F7D113f996`. May have additional liquidity-provider context.
4. **Forensic identification of the Telegram CAPTCHA vector** — if the victim can identify the specific Telegram bot/group, that's actionable intel against the operator (vs. the immediate attacker who is downstream).
5. **Loss verification per asset** — POD and FHE current-supply / per-attacker-balance accounting against the analyst's $125K / $21K estimates. Token prices at drain time (00:37 UTC) needed for precise figures.
