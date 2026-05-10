# Entity_005 — "The Architect"
**Generated:** 2026-03-30
**Revised:** 2026-03-30 (gas cluster debunked, see Corrections); 2026-05-09 (Correction #20: funder + 0xc5d... behavioral match retracted, see Corrections)
**Status:** Active R&D — zero production deployment (primary deployer unchanged)
**Chain:** Arbitrum (exclusive)
**Threat Level:** CRITICAL (pre-production)

> **[CORRECTION #20 — 2026-05-09]** Two attribution surfaces around the Architect are retracted; the primary deployer finding is unchanged.
> - **Funder `0x151b381058f91cf871e7ea1ee83c45326f61e96d`** is OLI-tagged as **MoonPay 4 / Exchange** — a fiat onramp address with millions of recipients. The "sole funder, single-deployer" framing collapses; MoonPay is not exclusive to anyone. The 0.0508 ETH was a MoonPay deposit to the Architect's wallet, not a deliberate funding event by an Architect-aligned actor.
> - **Behavioral match `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` (0.742 similarity)** is OLI-tagged as **CryptoCauses: Deployer** — a separate Web3 project, not an Architect alternate. Behavioral-similarity 0.742 was computed on profile-shape dimensions (timezone, technique, cadence) without identity disambiguation; OLI lookup was never run.
> - **Primary Architect finding (`0x9209c9f7dcb61937f1ec8160c22c0b2365079474`) is NOT retracted.** The 21-contract R&D session pattern, the 4-mechanism weapon achievement, the night-shift Arbitrum-exclusive operation — all stand. The other behavioral matches (`0x4cfe37d2`, `0x30e88ee4`, `0x7930e138`, `0xd28e6a7a`) are not yet OLI-cleared.
>
> See `reports/correction_log.md#correction-20` for the full numbered correction.

---

## Executive Summary

Entity_005 is the most sophisticated trap developer in our corpus. Operating exclusively on Arbitrum in R&D mode, they systematically iterate through contract architectures across disciplined 7-contract sessions, building toward a 4-mechanism weapon (SELFDESTRUCT + DELEGATECALL + TIMESTAMP + CALLER) that no other operator has achieved.

The Architect's distinctiveness lies in their session discipline (exactly 7 contracts per session, consistent escalation pattern), the 4-mechanism weapon (unique in the corpus), and 2 confirmed behavioral matches at 0.74-0.80 similarity. An initial hypothesis that a gas fingerprint cluster of 46 deployers was connected was investigated and debunked — 0.0200xx gwei is the Arbitrum default (54% of all Arbitrum deployers use it), not a distinctive signal.

---

## The Architect — Primary Address

**Deployer:** `0x9209c9f7dcb61937f1ec8160c22c0b2365079474`
**Funder:** ~~`0x151b381058f91cf871e7ea1ee83c45326f61e96d` (0.0508 ETH, sole deployer)~~ **[CORRECTION #20 → MoonPay 4 / Exchange — fiat onramp deposit, not Architect-aligned funding]**
**Contracts:** 21 (3 documented sessions, Session 4 pending sync)
**Chain:** Arbitrum only
**Gas:** 0.020012 gwei (invariant — Arbitrum default, not distinctive)
**Timezone:** Night shift (01:00-02:00 UTC) / late session at 17:00 UTC

### Session Timeline

| Session | Date | Time (UTC) | Contracts | Peak Complexity |
|---|---|---|---|---|
| 1 | 2026-03-25 | 01:01-01:25 | 7 | SD+DC+TS+CL (6 patterns) |
| 2 | 2026-03-26 | 02:08-02:19 | 7 | SD+DC+TS+CL (6 patterns) |
| 3 | 2026-03-29 | 17:11-17:31 | 7 | SD+DC+TS+CL (6 patterns) |
| 4 | 2026-03-30 | TBD | 7 (reported) | Pending sync |

### Capability Evolution Matrix

| Session | Contract | Time | CL | DC | SD | TS | Total | Notes |
|---|---|---|---|---|---|---|---|---|
| **S1** | 0x8ecada | 01:01 | X | | | | 1 | Baseline CALLER check |
| S1 | 0xec3103 | 01:07 | X | | | | 1 | Same — validating |
| S1 | 0xde31be | 01:11 | X | | X | X | 5 | Added SELFDESTRUCT + TIMESTAMP |
| S1 | 0xc750dc | 01:16 | X | X | | X | 4 | Swapped: added DELEGATECALL, removed SD |
| S1 | 0x0fb3aa | 01:20 | | | | | 0 | Clean control |
| S1 | 0xccd25c | 01:24 | X | X | X | X | **6** | **Full weapon — first achieved** |
| S1 | 0x0170cb | 01:25 | | | | | 0 | Clean control |
| **S2** | 0xbfc38e | 02:08 | X | | | | 1 | Reset to baseline |
| S2 | 0x158bfe | 02:09 | X | | | | 1 | Validation |
| S2 | 0x532bb7 | 02:12 | X | X | | X | 5 | DC+TS+CL |
| S2 | 0x2d4fbd | 02:13 | X | X | X | X | **6** | Full weapon |
| S2 | 0x16e42a | 02:16 | X | X | X | X | 5 | SD+DC+TS+CL variant |
| S2 | 0x79040e | 02:18 | | | | | 0 | Clean control |
| S2 | 0x9cca9d | 02:19 | X | X | X | X | **6** | Full weapon — third instance |
| **S3** | 0x6f55f5 | 17:11 | X | | | | 1 | Baseline |
| S3 | 0xc19e29 | 17:15 | | | | | 0 | Clean |
| S3 | 0xb0b658 | 17:17 | X | | | | 1 | Baseline variant |
| S3 | 0xb620e5 | 17:18 | X | X | | X | 5 | DC+TS+CL |
| S3 | 0x3b48fd | 17:19 | X | X | X | X | **6** | Full weapon |
| S3 | 0xe784e1 | 17:24 | X | X | X | X | **6** | Full weapon |
| S3 | 0xef883a | 17:31 | X | X | X | X | 5 | SD+DC+TS+CL variant |

**Pattern:** Each session repeats the same escalation (simple → complex → full weapon → control). By Session 3, 4 of 7 contracts are full-weapon or near-full-weapon. The operator is refining, not exploring — they found their architecture in Session 1 and are iterating the implementation.

---

## Behavioral Matches (Profiler-Confirmed)

Two deployers match The Architect's behavioral fingerprint through the deployer profiler's multi-dimensional similarity scoring (timezone, technique, cadence, chain — NOT gas price):

| Deployer | Contracts | Timezone | Technique | Similarity Score |
|---|---|---|---|---|
| `0x9209c9f7` (Architect) | 21 | night_shift | CL+DC+SD+TS | — |
| `0x4cfe37d2` | 17 | night_shift | caller_check | **0.799** |
| ~~`0xc5d13329`~~ | ~~16~~ | ~~night_shift~~ | ~~delegatecall~~ | ~~0.742~~ **[CORRECTION #20 → CryptoCauses: Deployer — separate Web3 project, REMOVED from Architect-cluster]** |

These are potential alternate wallets. Both are Arbitrum-exclusive, night-shift deployers with similar technique profiles. If confirmed, the operation extends to ~54 contracts across 3 wallets.

Three additional behavioral matches at 0.71 (`0x30e88ee4`, `0x7930e138`, `0xd28e6a7a`) are lower confidence but worth monitoring.

---

## Cross-Organization Check

| Dimension | org_001 | org_002 | org_003 | Parasite | Architect |
|---|---|---|---|---|---|
| Shared funder | NO | NO | NO | NO | — |
| Shared bytecode family | NO | NO | NO | NO | — |
| Gas overlap | NO | NO | NO | NO | — |
| Chain overlap | NO (Base primary) | NO (Base only) | NO (Base only) | NO (Base only) | Arbitrum only |
| Timing overlap | NO (daytime) | NO (24/7) | NO | NO | Night shift |

**The Architect is fully independent.** No connection to any known organization across any dimension. Different chain, different timezone, different funding, different techniques.

---

## Behavioral Fingerprint

| Attribute | Value | Uniqueness |
|---|---|---|
| Gas price | 0.020012 gwei | Arbitrum default — not distinctive (54% of deployers use 0.0200xx) |
| Timezone | Night shift (01:00-17:00 UTC) | Shared with 2 confirmed behavioral matches |
| Chain | Arbitrum exclusive | Distinctive — most operators favor Base |
| Session pattern | Exactly 7 contracts, ~25 min | Unique in corpus |
| Technique | 4-mechanism combo (SD+DC+TS+CL) | Unique in corpus |
| Deployment style | Burst | Common |
| Funder | Single-use, 0.0508 ETH | Standard |
| Interactions | Near-zero (1 self-test) | Consistent with R&D |

---

## Threat Assessment

### What Happens When They Go to Production

The Architect's weapon deploys a contract that:
1. **Looks clean at deployment** — passes initial scan (clean control contracts prove they know how to deploy without triggering detectors)
2. **DELEGATECALL allows post-deployment logic swap** — the contract can become anything after the scanner checks it
3. **TIMESTAMP activates on a schedule** — benign before the timer, trap after
4. **CALLER gates the deployer** — operator always passes, victims always fail
5. **SELFDESTRUCT erases evidence** — after extraction, bytecode disappears from chain

**Expected deployment chain:** Base (where victim traffic is)
**Expected timing:** After Session 4 produces a stable build (possibly next 48-72 hours)
**Expected architecture:** The full 4-mechanism weapon, likely with a clean initial state that upgrades to trap mode via DELEGATECALL after deployment

### Why This Is Different

Every other trap in our corpus is detectable at deployment time by at least one of our 10 pattern detectors. The Architect's architecture is designed to be **clean at deployment and dangerous later**. The DELEGATECALL upgrade path means the bytecode we classify at deployment time isn't the bytecode that extracts value from victims. This is the first adversarial architecture specifically designed to defeat static analysis — including ours.

---

## Watchlist Status

| Entry | Type | Priority | Status |
|---|---|---|---|
| `0x9209c9f7` | deployer | CRITICAL | Active — Session 4 detected |
| `0x151b3810` | funder | ~~CRITICAL~~ **DEACTIVATED** | **[CORRECTION #20 → MoonPay 4]** |
| `0x4cfe37d2` | deployer (associated) | HIGH | Active (0.799 similarity) — not yet OLI-cleared |
| `0xc5d13329` | deployer (associated) | ~~HIGH~~ **kept active w/ note** | **[CORRECTION #20 → CryptoCauses: Deployer]** — LOW-confidence OLI tag, pending second-source verification |
| `0x30e88ee4` | deployer (associated) | HIGH | Active (0.711 similarity) — not yet OLI-cleared |
| `0x7930e138` | deployer (associated) | HIGH | Active (0.710 similarity) — not yet OLI-cleared |
| `0xd28e6a7a` | deployer (associated) | HIGH | Active (0.719 similarity) — not yet OLI-cleared |

---

## Recommended Actions

1. **Monitor Session 4 contracts** when sync completes — check if the architecture has evolved
2. **Watch for Base deployment** — any watchlist hit on Base from Architect or associated addresses is the production signal
3. **Run DELEGATECALL post-deployment monitor** — 148 candidates identified. This is the specific mechanism The Architect will use to bypass our classifier. Module built: `delegatecall_monitor.py`
4. **Profile the 2 high-confidence matches** (`0x4cfe37d2`, `0xc5d13329`) — check their contracts for the same session pattern and technique escalation

---

## Corrections

**2026-03-30:** Gas fingerprint cluster (46 deployers, 0.0200xx gwei) debunked. Investigation found 0.0200xx is the Arbitrum default gas price — 54% of all Arbitrum deployers use this range. The 46-deployer "cluster" was the general Arbitrum population, not a distinctive signal. `0xd660fa35` (164 contracts, night shift) confirmed INDEPENDENT of The Architect — different funder (`0x34aeddd4` vs `0x151b3810`), all-clean bytecode (255/255 no patterns vs multi-mechanism weapons), steady deployment style vs burst. Case file revised to remove all gas-based attribution. Behavioral matches (0.71-0.80) retained — they're based on timezone + technique + cadence, validated independently of gas price.

**2026-05-09 (Correction #20 — OLI mass mislabel sweep):**
- **Funder retraction:** `0x151b381058f91cf871e7ea1ee83c45326f61e96d` is OLI-tagged (Open Labels Initiative via Blockscout metadata service) as **MoonPay 4 / Exchange** — a fiat onramp address. The original "sole funder, 0.0508 ETH, single deployer" framing reflected a behavioral observation (one deposit, one downstream) but the identity layer was never queried; in reality MoonPay sends to millions of recipients and "single deployer" is just what the corpus happened to surface from MoonPay's downstream. Watchlist row deactivated. The 0.0508 ETH funding event remains real but its *meaning* changes — it's a user (the Architect) buying ETH via fiat onramp, not an Architect-aligned actor staging funds.
- **Behavioral match retraction:** `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` is OLI-tagged as **CryptoCauses: Deployer** — a separate Web3 project. The 0.742 behavioral-similarity score was profile-shape match (timezone + technique + cadence), not identity match. Removed from Architect-cluster; watchlist note added (kept active pending second-source OLI verification per Correction #20 §"Open work").
- **Cross-organization independence claim** in the table above remains structurally correct: the Architect's primary deployer is still independent of org_001-003. Update only: CryptoCauses (a Web3 project) is now noted as a separate, non-adversarial entity that previously appeared in Architect-cluster attribution.
- **Methodology lesson:** the behavioral-fingerprint similarity dimensions (timezone, technique, cadence) describe *operator style*, not *operator identity*. Two unrelated operators with similar style produce similarity scores in the 0.7-0.8 range. The OLI cross-check is now part of the typology-promotion pipeline; future behavioral matches should be OLI-checked before attribution.

See `reports/correction_log.md#correction-20` for the full numbered correction.

---

*Case file generated by Layer 3 Surveillance*
*Investigation: Entity_005 "The Architect"*
*All analysis from SQLite — zero RPC calls*
