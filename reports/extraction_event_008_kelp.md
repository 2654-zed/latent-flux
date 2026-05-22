# Extraction Event 008 — KelpDAO LayerZero DVN Configuration + RPC Poisoning Failure

**Event date:** 2026-04-18
**Chain:** Ethereum (destination, monitored_chain=1). Source: Unichain.
**Purpose:** Corpus expansion. Largest April-2026 exploit; configuration-layer enabler + operator-layer RPC poisoning proximate cause.
**Status:** Draft INSERT. Nothing executed. Paired with deeper retrospective at `reports/kelp_retrospective_replay.md` (scoped separately, 50-call RPC budget).

**2026-05-22 revision.** LayerZero Labs published their official incident report on 2026-05-18 (`kelpdao-incident-report.pdf` in the corpus archive). The report upgrades the attribution from Tier B → Tier A and reveals the proximate attack vector was RPC poisoning, not signer-key compromise or any DVN-internal failure. The 1-of-1 DVN configuration was the *enabler* (any compromise of a single DVN's infrastructure = exploit), but the *mechanism* was operational infrastructure compromise of LayerZero Labs' GCP environment. All prior Tier A facts in this file survive; the framing is expanded. New Tier A facts below.

---

## The one-line frame

**The configuration was publicly observable for weeks pre-exploit** — `EndpointV2.getConfig` returns `requiredDVNCount=1` on both chains; LayerZero's own documented best practice is `>=2 required DVNs`. The exploit took zero code defects, zero compromised admin keys, zero novel mechanism — just an attacker willing to compromise a single validator in a system that accepted single-validator attestation.

## Tier A claims

Sourced from `github.com/DK27ss/KelpDAO-294m-PoC` + Blockaid statement + LayerZero network response.

**Attack**
- Attack tx: `0x1ae232da212c45f35c1525f851e4c41d529bf18af862d9ce9fd40bf709db4222`
- Ethereum block: 24,908,285 (pre-state block 24,908,284)
- Entry: single call to `EndpointV2.lzReceive`, gas used 94,456
- Source chain: Unichain (srcEid 30320)
- 116,500 rsETH minted to attacker at nonce 308 — ≈ 18% of rsETH circulating supply
- 40,000 rsETH at nonce 309 BLOCKED via Kelp's `TransfersBlocked` at 2026-04-19 18:23:11 UTC

**Key addresses**
- Kelp OFTAdapter (Ethereum): `0x85d456B2DfF1fd8245387C0BfB64Dfb700e98Ef3`
- Required DVN (Ethereum): `0x589dedbd617e0cbcb916a9223f4d1300c294236b`
- Required DVN (Unichain): `0x282b3386571f7f794450d5789911a9804fa346b4`
- Attack recipient (fresh address): `0x8B1b6c9A6DB1304000412dd21Ae6A70a82d60D3b`
- **OApp delegate (single EOA, added 2026-05-22 from PDF):** `0x1f7A03b70C5448DFd0a2C5a7865169253c2C769b` — the key that **modified the channel configuration from 2-of-2 to 1-of-1** before the exploit. Single EOA controlled the security policy of a $292M bridge. Should be watchlisted as a configuration-authority anchor.
- **LayerZero Endpoint (Ethereum, immutable):** `0x1a44076050125825900e736c501f859c50fE728c`

**Configuration at time of attack (verifiable via `getConfig(configType=2)`)**
| Chain | requiredDVNCount | optionalDVNCount | optionalDVNThreshold |
|---|---|---|---|
| Unichain (source) | 1 | 0 | 0 |
| Ethereum (destination) | 1 | 0 | 0 |

**Downstream propagation**
- Attacker deposited stolen rsETH to Aave V3 as collateral on **Ethereum AND Arbitrum**
- Borrowed ~$236M WETH against the rsETH
- Bad debt on Aave V3 post-unbacked

## Tier A — RPC poisoning kill chain (added 2026-05-22 from LayerZero Labs report)

The PDF reveals the **proximate** cause was not DVN-internal. The 1-of-1 configuration was the *enabler* (single point of failure); the *mechanism* was operational compromise of LayerZero Labs' GCP infrastructure:

| Date (UTC) | Event |
|---|---|
| 2026-03-06 12:59–13:06 | LayerZero developer (Developer1, on a team running RPC infrastructure) socially engineered to clone malicious GitHub repo `github[.]com/pi2infra-can-4/gtn-candidate-repo.git`. Repo dropped **FLATROOF** + **ROOFDECK** Rust macOS backdoors. EDR missed both. **SSO + MFA were enforced** but bypassed via session-key harvest. |
| 2026-03-30 → 2026-04-16 | Attacker used Developer1's session keys to access LayerZero's GCP + GitHub via VPN services (ExpressVPN, NordVPN, Mullvad). Reconnaissance + persistence. |
| 2026-04-16 → 2026-04-18 | Lateral movement to GKE clusters. Patched the `op-geth` running process on **two separate GKE clusters in two regions** via injected ELF Position-Independent Executable (CARVED1 launcher → CARVED2 monitor using `funchook` to hook Go `syscall.Syscall` / `syscall.Syscall6`). The malware was discriminating: returned correct responses to LayerZero monitoring tools but tampered responses to the DVN signing service. |
| 2026-04-18 16:30 | Denial-of-Service attack on **external** RPC providers. Forced LayerZero's failover logic to rely exclusively on the two poisoned internal RPC nodes. |
| 2026-04-18 17:35 | Exploit. Single `EndpointV2.lzReceive` call mints 116,500 rsETH to attacker. |

**Signer keys were NEVER compromised.** Forensic analysis confirms no key exfiltration. The DVN signed legitimately, with legitimate keys, over a hash of source-chain state that had been *fabricated* by the poisoned RPC layer. This is qualitatively different from "key compromise" or "signer compromise" framings — the right name is **RPC-Layer Honesty-Assumption Violation** (new lexicon entry).

## Tier A — Attribution upgrade (2026-05-22)

Prior framing (Tier B inference by analogy with Drift heist patterns): "Lazarus/DPRK profile, consistent with prepared automation."

Revised framing (Tier A, multi-source independent corroboration):

- **Mandiant + CrowdStrike:** HIGH-confidence DPRK; MEDIUM-confidence **UNC4899** (aka TraderTraitor, Jade Sleet, Pressure Chollima).
- **tanuki42 + tayvano (independent researchers):** confirm UNC4899 via de-mixing of attacker funding back to known UNC4899 infrastructure-payment addresses.
- **UNC4899 lineage:** assessed with high confidence to be aligned with the DPRK Reconnaissance General Bureau. Same group as the **Bybit Safe{Wallet} heist (Feb 2025, $1.5B)**. Pattern reuse: macOS-targeted social engineering → developer-credential harvest → infrastructure compromise → single-attack-window exploit → wallet-laundering ring.

## Tier B interpretations

- **Attack family (revised):** compound failure — configuration (1-of-1 DVN, enabler) + operational (RPC poisoning, proximate cause) + code-assumption (signing service implicitly trusted its RPC layer). Distinct from Aethir (purely operational — key compromise) and Hyperbridge (purely code-level MMR bypass). All three are April-2026 cross-chain infrastructure cluster events; Kelp is the compound-failure exemplar.
- **Stored-potential pre-attack:** CRITICAL. Per the core interpretive rule, maximum capability (mint authority) + maximum trust binding (Kelp + LayerZero) + maximum victim exposure (rsETH holders) + zero constraint (1-of-1 DVN). Exactly the state the framework identifies as loaded-not-safe.
- **Detection surface:** DVN configuration enumeration. LayerZero DVN configs are read-only public data on-chain; a surveillance module that polled OApp configurations and flagged `requiredDVNCount=1` would have scored Kelp CRITICAL with significant lead time. The retrospective replay report quantifies "significant" via historical `getConfig` calls at earlier blocks.
- **Why not Circle/Tether freeze:** rsETH is a yield-bearing LRT (Liquid Restaking Token), not a stablecoin. No Circle or Tether pathway applies. Recovery likely requires protocol-native action by Kelp and downstream counterparties (Aave, LayerZero).

## What Layer 3 currently catches vs misses

**Catches (retrospectively verifiable via Phase 7 of kelp_retrospective_replay.md):**
- Attack recipient's downstream Arbitrum Aave V3 deposits (if the attacker used our monitored chain post-Ethereum-mint, our `transaction_events` / `approval_events` should have captured it)

**Misses:**
- The entire Ethereum origin — we don't monitor Ethereum
- LayerZero OApp configurations — no module enumerates these
- DVN signing-activity baselines — not indexed
- Unichain — not monitored
- Code-level MMR-style bugs (see EXTRACTION_007 — bridges can fail in three orthogonal ways; our current surface catches none)

**Potentially catches if extensions approved:**
- DVN configuration monitoring (new module scoped in notes; cost estimate: ~150 LoC)
- Cross-chain mint-event conservation check (from EXTRACTION_007 notes): bridged-asset inbound mints vs source-chain burns — would have flagged the forged lzReceive within blocks

## Cluster framing (Aethir / Hyperbridge / Kelp)

| | Aethir (EXTRACTION_006) | Hyperbridge (EXTRACTION_007) | Kelp (EXTRACTION_008) |
|---|---|---|---|
| Date | 2026-04-09 | 2026-04-13 | 2026-04-18 |
| Failure layer | Operational (key) | Code (validation) | Configuration (DVN) |
| Loss | $400K | $237K | $292M |
| Audit-catchable? | No | Yes (in principle) | No |
| Publicly observable pre-attack | No | No | Yes (on-chain config read) |

Three orthogonal attack vectors in nine days against the same structural target. The diversity of failure modes is the commercial framing point: traditional audits (code-only) catch at most 1 of 3; Layer 3's stored-potential lens catches all 3 by measuring capability-vs-constraint directly.

## Cross-links

- **EXTRACTION_006 (Aethir)** — operational precursor, 9 days earlier
- **EXTRACTION_007 (Hyperbridge)** — code-layer cluster sibling, 5 days earlier
- **EXTRACTION_005 (Drift)** — parallel "stored potential via removed constraint" at governance layer; Drift dropped multisig threshold + timelock; Kelp accepted 1-of-1 DVN from deployment
- **`reports/kelp_retrospective_replay.md`** — the deep retrospective (scoped 50-RPC budget, 8 phases, paused at every phase boundary) that answers "what signals would Layer 3 have caught if it monitored Ethereum and LayerZero OApp configurations"
- **Pattern D (lexicon)** — sibling concept. Pattern D = adversaries importing aged-identity reputation across chains. **Kelp = adversaries compromising the operator infrastructure that bridges chains** (operator-layer compromise, distinct from reputation import). See revised Pattern D entry.
- **UNC4899 / TraderTraitor (lexicon threat-actor entry, added 2026-05-22)** — same threat actor as Bybit Safe{Wallet} heist (Feb 2025, $1.5B). Pattern reuse documented in lexicon entry.
- **LayerZero Labs incident report (2026-05-18)** — official post-mortem, source for the 2026-05-22 revision facts above. Archived as `kelpdao-incident-report.pdf`.

## What the INSERT contains

Full Tier A facts in summary + raw_transactions JSON. Notes carry Tier B interpretations with explicit labels. `chain='ethereum'`, `monitored_chain=1`.

Note: EXTRACTION_007 and EXTRACTION_008 are both on Ethereum / `monitored_chain=1`. That doesn't mean our active monitor covered them — Ethereum is NOT currently in our ingest set. `monitored_chain=1` here reflects that Ethereum IS on the "chains we would monitor if we expanded" roster, vs 0 for truly out-of-scope (NEAR, Solana, BNB). The field distinguishes first-class-target chains from reference-data chains. If that's the wrong convention, the classification will be easy to revise; no other table joins this column today.

## What this file does NOT claim

- No prevention claim. Flagging stored potential is not the same as preventing exploitation. Layer 3 has no enforcement layer.
- No attribution of the compromised DVN operator. Whether the attacker obtained signing keys via compromise or was the DVN operator is not established.
- No final Recovery figure. Coordination between Kelp / LayerZero / Aave is ongoing as of documentation date.
- No quote-ready claim without retrospective validation. The retrospective replay report is where "would-have-caught" claims get Tier A substantiation (historical getConfig calls, DVN signing-activity baselines, etc.). Until that report lands, all "Layer 3 would have caught X" statements here are labeled Tier B.
