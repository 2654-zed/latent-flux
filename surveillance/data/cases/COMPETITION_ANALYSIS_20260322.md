# TRAP COMPETITION & UNDERMINING ANALYSIS
**Generated:** 2026-03-22 23:30 UTC
**Corpus:** 19,149 contracts, 75,331 transaction events, 7,260 deployers

---

## Executive Summary
There IS evidence of deployer-on-deployer reconnaissance — 22 deployers were caught interacting with other deployers' contracts, with one confirmed case of a deployer copying a competitor's bytecode pattern after probing their trap. However, there is NO evidence of reactive deployment timing (nobody is rushing to copy successful traps within hours). The trap economy is in an **expansion phase**: operators share code templates freely (350 deployers use the same tx.origin pattern), but they're not actively undermining each other. The competition is for victim attention, not sabotage.

---

## Bytecode Families

**106 distinct bytecode pattern fingerprints are shared across multiple deployers.**

The trap economy runs on shared templates. The biggest families:

| Family Pattern | Contracts | Unique Deployers | First Seen | Assessment |
|---|---|---|---|---|
| tx.origin conditional at 0x314 | **350** | **350** | Mar 18 22:35 | org_002 campaign (1 contract per deployer) |
| CALLER->SLOAD gate at 0x2ab | 57 | 51 | Mar 18 23:22 | Shared template, varied deployers |
| CALLER->SLOAD gate at 0x2ac | 54 | 50 | Mar 18 23:10 | Same family, offset variant |
| CALLER->SLOAD gate at 0x2aa | 43 | 42 | Mar 18 23:07 | Same family, offset variant |
| CALLER->SLOAD gate at 0x2ad | 41 | 38 | Mar 18 23:03 | Same family, offset variant |
| CALLER->EQ gate at 0xea | 25 | 23 | Mar 20 20:30 | Conditional revert template |
| DELEGATECALL at 0x1235 | 19 | 19 | Mar 19 02:40 | Upgradeable proxy template |

**Key finding:** The CALLER->SLOAD family (offsets 0x2aa through 0x2af) comprises **229 contracts from 213 deployers** — the same base template with a 1-2 byte offset variation in the SLOAD gate position. These are NOT exact copies. The offset shifts suggest **parameterized deployment from a shared tool or service** where each deployer gets a slightly customized version. This is a trap-as-a-service pattern.

---

## Deployer Reconnaissance

**22 deployers were caught hitting other deployers' contracts.** 2,045 total cross-deployer interactions.

### Confirmed Copy-After-Recon

**`0x0b034b10d7bb219d...`** — THE SMOKING GUN
- Probed 2 contracts from another deployer: 43 interactions, **81% reverted** (testing the trap)
- First recon: Mar 21 18:59
- **Then deployed 1 contract with MATCHING bytecode pattern**
- Assessment: **CONFIRMED CODE COPY AFTER RECONNAISSANCE**

### Likely Reconnaissance (deployed after probing)

| Deployer | Contracts Hit | Interactions | Own Deploys After | Pattern Match |
|---|---|---|---|---|
| `0x0b034b10...` | 2 | 43 (81% rev) | 1 | **YES** |
| `0xa18478d1...` (MEV bot) | 11 | 125 (38% rev) | 53 | No (MEV, not competing) |
| `0x2f73a5c9...` | 1 | 1,241 (24% rev) | 1 | No data |
| `0xbf7b58c6...` | 1 | 70 (31% rev) | 1 | No data |
| `0xdd9734f5...` | 4 | 11 (9% rev) | 8 | No data |
| `0xc305b4f4...` | 1 | 26 (0% rev) | 1 | No data |

**`0xa18478d1`** is our cleared MEV bot factory (518 contracts). Its 125 interactions with other deployers' contracts are MEV attempts, not reconnaissance. It's a victim of traps, not a competitor.

**`0x2f73a5c9`** hit one contract 1,241 times — this looks like a bot farming a specific contract, not reconnaissance. The single deployment afterward may be unrelated.

**`0x0b034b10`** is the only clear case: probed, mostly reverted (testing defenses), then deployed with matching bytecode. Classic competitive espionage.

---

## Caller Migration

**Wallets hitting 3+ suspected/confirmed contracts:**

| Wallet | Contracts Hit | Different Deployers | Assessment |
|---|---|---|---|
| `0x2e20b2...` (Dragon) | 169 | 1 (self) | Approving own contracts, not migrating |
| `0xa70b3f...` | 156 | 5 | **Active multi-deployer scanner** |
| `0xbd418c...` | 147 | 5 | **Active multi-deployer scanner** |
| `0xd660fa...` | 86 | 1 (self) | Self-interaction |
| `0xb6526e...` | 31 | 5 | Multi-deployer scanner |

The migration pattern shows **systematic bot scanners** (`0xa70b3f`, `0xbd418c`, `0xb6526e`) that hit contracts from 5 different deployers. These are not victims migrating between traps — they're automated scanners probing every new contract. The high contract count (147-156) confirms industrial-scale scanning.

No evidence of victim redirection (wallets moving from one trap to a "better" one). Victims hit one contract and don't come back.

---

## Reactive Deployment Timing

**No evidence of reactive deployment.**

For each of the top 5 high-traffic contracts, we checked whether similar contracts appeared within 6 hours of the trap reaching 10 unique callers:

| Trap | Victims | 10-Caller Milestone | Similar Deploys Within 6h |
|---|---|---|---|
| `0x9da33e` | 1,466 | Mar 21 05:13 | **0** |
| `0xd46242` | 931 | Mar 22 19:32 | **0** |
| `0xa65319` | 840 | Mar 20 01:58 | **0** |
| `0xb15e7a` | 143 | Mar 19 15:01 | **0** |
| `0x0697a1` | 132 | Mar 21 17:33 | **0** |

Zero similar bytecode deployments in the reactive window. While 433 contracts deployed within 6 hours of `0xd46242` hitting its milestone, NONE shared its bytecode pattern. Operators are deploying on their own schedules, not reacting to competitors' success.

---

## Trap-as-a-Service Signal

The most significant finding isn't direct competition — it's the **CALLER->SLOAD gate family**: 229 contracts from 213 deployers, all sharing the same base bytecode with 1-2 byte offset variations. This isn't 213 people independently writing the same code. This is a **template distribution system**.

Possibilities:
1. **Open-source trap template** shared in a Telegram group or forum
2. **Paid trap-as-a-service** where an operator sells customized deployment scripts
3. **Single operator using 213 disposable wallets** (but the offset variations argue against this — a single script would produce identical offsets)

The offset variation (0x2aa through 0x2af for the same SLOAD gate) is consistent with a template compiler that slightly randomizes bytecode layout per deployment to evade exact-match detection. This is anti-forensics baked into the deployment tool.

---

## Assessment

| Signal | Evidence Level | Finding |
|---|---|---|
| Deliberate undermining | **WEAK** | 1 confirmed copy-after-recon case out of 19,149 contracts |
| Code copying | **STRONG** | 106 pattern families shared across deployers, but most is template sharing not stealing |
| Victim redirection | **NONE** | No evidence of victims migrating between similar traps |
| Reactive deployment | **NONE** | No time-correlated competitive responses detected |
| Shared tooling | **STRONG** | 229 contracts with parameterized offset variations = trap-as-a-service |

**The trap economy on Base/Arbitrum is in expansion phase.** There's enough victim surface (1,660+ unique victims on org_002 alone, 1,238 on 0x9da33e) that operators don't need to compete — they can all profitably fish in the same ocean. The one confirmed recon-and-copy case (`0x0b034b10`) is the exception, not the rule.

The competition phase will arrive when victim traffic saturates and returns per contract decline. When that happens, expect to see: shorter reactive deployment windows, more aggressive parameter tuning (lower fees to attract traffic), and deliberate griefing of competitors' contracts. The surveillance system is now instrumented to detect all three.
