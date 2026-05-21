> # ⚠ RETRACTED IN ITS ENTIRETY — 2026-05-21 (same day as creation)
>
> **This case file is RETRACTED per Correction #24 (filed 2026-05-21, same day this file was created).** Content below is preserved verbatim per the immutable-corpus-record discipline but should **NOT BE CITED AS A LIVE FINDING UNDER ANY CIRCUMSTANCE.**
>
> **What was claimed (retired):** That `0x752c5a95` was a Pre-Drain Harvester deployed by an Animoca-tagged wallet, and that it discharged on 2026-05-09 by sweeping 4,587 unique victims in 30 minutes via two independent drainer EOAs. The case file framed this as "the strongest validated Tier-C prediction in the Layer 3 corpus to date."
>
> **What is actually true:** `0x752c5a95` is **OneFootball Club (OFC)**, a legitimate verified `ERC20FixedSupply` contract from Animoca's `@animoca-network/contracts` framework. Listed on CoinGecko (`https://assets.coingecko.com/coins/images/67442/small/ofc.jpg`). 3,904 holders. $7.9M circulating market cap. The "second contract" `0xDA42FE` is the LayerZero OFT bridge adapter for OFC, not an unused sibling. The "discharge" transactions on 2026-05-09 were FAILED `transferFrom` calls (gas ~25K, status=error, zero tokens moved). Layer 3's `approval_watchlist.drain_detected` pipeline incorrectly credited those failed reverts as multi-victim mass drains.
>
> **Three stacked false-positive bugs produced this case file:**
> 1. **Bytecode classifier FP** on Animoca's `@animoca-network/contracts` framework (the `asymmetric_transfer + obfuscated_fee` flags match standard ContractOwnership + TokenRecovery patterns).
> 2. **Behavioral classifier FP** on pre-launch ERC-20 token launches (bots front-running pre-trading triggered a revert which the pipeline read as a trap firing).
> 3. **`approval_watchlist` pipeline bug** crediting failed transferFrom transactions as multi-victim discharge events.
>
> Full root-cause analysis: `reports/correction_log.md` Correction #24.
>
> ---

# Case File — `0x752c5a95` Pre-Drain Harvester Discharge (EXTRACTION_011) [RETRACTED]

**Status:** CONFIRMED — discharge event observed on-chain. Layer 3's 2026-04-24 prediction model materialized within 15 days.
**Discharge date:** 2026-05-09 (single-day event, ~30 minutes total drain window)
**Layer 3 corpus involvement:** Full — both the harvester contract and both discharge wallets are in the corpus; complete approval-pool history captured in `approval_watchlist` table since 2026-04-11.
**Prior-state declaration:** This contract was documented in `docs/INDEX.md` Section 1 (`0x752c5a95` Pre-Drain Harvester (Base)) on 2026-04-24 as the largest fresh confirmed-tier approval pool in the corpus with 1,898+ approvals and zero drains, accompanied by a Tier-C prediction that the harvester would discharge. This case file is the **Phase 4 "Confirms"** record of that prediction materializing.

---

## Identity and roles

| Role | Address | Notes |
|---|---|---|
| **Harvester contract** | `0x752c5a95d202972e124390f30a50154409d3c858` | Base. Confirmed-tier. Bytecode flags: `has_asymmetric_transfer=1`, `has_unusual_fee_structure=1`. Deployed by `0x80b12bd0` (Animoca-tagged after Correction #20). Approval pool first observed 2026-04-11T23:25:05Z. |
| **Discharge wallet #1** | `0x1d81aff2a24c822d715ec09a0f81801face6e6fd` | Base. Drained **3,228 victims** in a single transaction batch at 2026-05-09T11:28:23Z. Itself a deployer (3 contracts on Base, first deploy 2026-04-16). Drained no other corpus contract. **Not previously on watchlist.** |
| **Discharge wallet #2** | `0x0e2224685fe775b471b457c643913e4bbd66c8d2` | Base. Drained **1,359 victims** across 2026-05-09T11:50:05Z–11:58:01Z (8-minute window, ~170/min). Pure EOA — no deployer record. Drained no other corpus contract. **Not previously on watchlist.** |
| **Original deployer** | `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` | Base. **Animoca-tagged** per OLI labels (Correction #20, 2026-05-09). 2,498-day mainnet vintage (real, but institutional, not "pristine solo operator" cover as originally framed). Deployed exactly two contracts in corpus; this harvester is one of them. |

---

## Discharge timeline

| Time (UTC) | Event | Cumulative drained |
|---|---|---|
| 2026-04-11T23:25Z | First approval observed on harvester | 0 |
| 2026-04-24 | Layer 3 documents 1,898+ approvals, 0 drains. Issues Tier-C prediction: harvester will discharge. INDEX.md entry created. | 0 |
| 2026-05-09T11:28:23Z | **Caller #1 (`0x1d81aff2`) drains 3,228 victims** in a single tx | 3,228 |
| 2026-05-09T11:50:05Z | Caller #2 (`0x0e222468`) begins 8-minute drain wave | 3,228 |
| 2026-05-09T11:58:01Z | Caller #2 finishes. **1,359 additional victims drained.** | **4,587** |
| 2026-05-09 (post-11:58) | No further drain activity from either caller. Both wallets dormant on the harvester since this date. | 4,587 |
| 2026-05-09 → 2026-05-21 | Harvester continues to accumulate fresh approvals (3,565 new victim approvals post-discharge). Last approval as of this writing: 2026-05-21T14:13:55Z. | 4,587 |

**Total discharge time:** 30 minutes 38 seconds from first drain to last (11:28:23 → 11:58:01).
**Total discharge volume:** 4,587 unique victims drained on 2026-05-09. Two distinct drain_caller wallets, no overlap in victim sets.
**Pre-discharge approval pool growth:** 1,898 (2026-04-24) → ~4,587 by discharge date (2026-05-09). The pool more than doubled in the 15 days between prediction and discharge.

---

## Post-discharge state (as of 2026-05-21)

| Metric | Value |
|---|---|
| Total approval rows | 8,152 |
| Unique victims | 8,152 |
| Drained victims | 4,587 (56.3%) |
| **Still-approved, un-drained victims** | **3,565 (43.7%)** |
| Time since last drain event | 12 days |
| Approval pool still growing? | Yes — most recent approval 2026-05-21T14:13Z |

**Interpretation:** The harvester has not gone dormant. It continues to attract new victim approvals at a roughly steady cadence. The contract behaves like a once-discharged-but-still-armed weapon: 56% of stored potential has been realized; 44% remains stored. A second discharge is structurally possible at any time, on the larger post-discharge pool.

---

## Independent-discharge analysis

Two drain_caller wallets discharged the same harvester on the same day, 22 minutes apart, with no overlap in victim sets:

| Aspect | Caller #1 (`0x1d81aff2`) | Caller #2 (`0x0e222468`) |
|---|---|---|
| Drain volume | 3,228 victims | 1,359 victims |
| Drain duration | Single tx (~13 seconds) | 8 minutes |
| Rate | ~3,228 in one tx batch | ~170/min |
| Pattern | Mass-sweep, single-call | Sustained iteration |
| Deployer record? | Yes — itself deploys 3 contracts on Base | No — pure caller EOA |
| Other corpus contracts drained | 1 (only this harvester) | 1 (only this harvester) |
| Activity before this event | None in corpus | None in corpus |
| Activity after this event | Dormant | Dormant |

Both wallets have produced zero drain activity in the corpus before or since the 2026-05-09 event. Both are single-purpose discharge wallets created/used for this one event. The fact that two independent wallets discharged the same harvester on the same day, with non-overlapping victim sets, is consistent with one of three explanations:

1. **Coordinated multi-wallet operation** — one operator using two wallets to discharge in parallel, with the work pre-partitioned across the victim set. The 22-minute gap and the difference in drain-style (mass-sweep vs sustained-call) would then be deliberate technique variation, possibly to defeat single-wallet rate-limiting on RPC providers.
2. **Public-signal pile-on** — caller #1's discharge made the harvester's existence known to other actors monitoring the chain (mempool watchers, MEV bots, fork-aware drainers). Caller #2 piled on within minutes against whatever approvals caller #1 had not yet swept. The drain-style asymmetry would then be platform-dependent rather than operator-design.
3. **Compromise / shared key** — the harvester contract has logic that requires a key/signature to discharge, and that key leaked between caller #1 and caller #2's first call. Unlikely given the structural identity of the harvester (Permit2 transferFrom pattern), but cannot be ruled out without bytecode inspection.

Without RPC trace of the two callers' funding sources and tx-batching patterns, we cannot distinguish (1) from (2). Recommend funder-trace probe via Blockscout on both wallets as the immediate followup.

---

## Validation of the 2026-04-24 prediction

The prediction model in the original INDEX.md entry stated:

> "Confirmed-tier contract harvesting Permit2 approvals from 1,898+ victims (as of 2026-04-24) without firing a sweep. The harvester is the largest active confirmed-tier approval pool in the corpus. Status: UNDER_INVESTIGATION — pre-drain accumulation, no sweep yet."

The Tier-C inference was implicit: **a contract that confirms as a trap (bytecode flags asserting), attracts thousands of approvals, but does not sweep, is in a pre-discharge accumulation phase.** The "Stored Potential" lexicon entry frames this as the canonical case where absence of realized value is the danger signal.

**Outcome:**
- **Prediction:** harvester will discharge
- **Observed:** harvester discharged 2026-05-09 — 4,587 victims in 30 minutes
- **Time-to-event from prediction:** 15 days
- **Pool growth during waiting period:** 1,898 → 4,587 (+142%)
- **Discharge fraction:** 56.3% of pool drained; 43.7% remains stored

This is the **strongest validated Tier-C prediction in the Layer 3 corpus to date.** Prior predictions (Phase A series, b0b0b690 / iter_8 / Coffee Fleet) were disproven because they cited named entities not in the queryable corpus. The 0x752c5a95 prediction was sourceable, time-bounded, and produced a falsifiable outcome — and was confirmed by direct on-chain observation.

---

## Open questions

1. **The Animoca-deployer question (Correction #20 open work) is now urgent.** An Animoca-tagged wallet deployed the contract that just drained 4,587 victims. The four possible explanations from the original correction remain:
   - **Compromise**: Animoca's key for this deployer wallet has been compromised; the harvester was deployed without Animoca authorization.
   - **Rogue developer**: an Animoca employee with deploy access used the wallet for personal/external purposes.
   - **Label staleness**: the OLI tag is stale; this wallet was used by Animoca once, then transferred or sold.
   - **Mixed behavior**: Animoca knowingly deployed something whose function differs from its surface bytecode.
   
   The first three implicate Animoca via different mechanisms; the fourth would represent an unprecedented institutional breach. None can be ruled out from Layer 3's corpus alone. Cross-chain trace of `0x80b12bd0`'s recent activity + Etherscan-side label history + Animoca's own attestation are the next-step inputs.

2. **Why didn't the discharge sweep all 4,587+ approvals?** At the discharge moment the pool was at ~4,587 victims (estimated). Both discharge wallets stopped at 4,587 total. Either:
   - The discharge wallets were rate-limited (gas, RPC throughput, mempool position) and 0 victims escaped the sweep,
   - The harvester's logic required a specific approval state (e.g., approval timestamp inside a window) that excluded the post-2026-05-09 victims,
   - Or the discharge was an intentional partial drain to keep the contract from being publicly flagged as drained, retaining the appearance of operational legitimacy for the next discharge cycle.
   
   The 3,565 post-discharge approvals are the empirical test: if the contract has a logic-gate that excluded them on 2026-05-09, that gate may discharge them in a future event. If not, the post-discharge accumulation is stored potential for a second sweep.

3. **Will there be a second discharge?** 43.7% of pool remains armed. Layer 3 should monitor the harvester for any drain_caller activity on a daily basis going forward. Add to Q-002's daily-watch surface if not already covered (it should be — harvester is on watchlist as `pre_discharge_bait_a27bba42`'s sibling once the 752c5a95 deployer is escalated to watchlist priority).

---

## Lexicon implications

- **Stored Potential.** This is the empirical validation that "absence of realized value is the danger signal" is operationally true at scale. The harvester carried zero observable harm for ~30 days, then realized 4,587 victims of harm in 30 minutes. The "absence" was not safety — it was loading.
- **Camouflage Equilibrium.** The harvester maintained a clean revert profile for the entire pre-discharge accumulation period. This is one data point against the Correction #22 reframing — *some* predator-class contracts do calibrate to low revert rates pre-discharge. The 30.44% confirmed-tier low-revert rate from Correction #22 averages across both pre-discharge (low-revert) and post-discharge / actively-extracting (high-revert) populations. Worth a follow-up partition.
- **Strategy Lifecycle.** This case proves the Strategy Lifecycle's Tier-C prediction model can produce successful predictions on the 2-week horizon when the inputs are sourceable and the contract is in the corpus.

---

## Updates to other corpus state required

- **INDEX.md Section 1** — `0x752c5a95` Pre-Drain Harvester entry: change status from `UNDER_INVESTIGATION` to `DISCHARGED 2026-05-09`. Link to this case file as primary case file.
- **INDEX.md Section 2** — add `0x1d81aff2a24c822d715ec09a0f81801face6e6fd` and `0x0e2224685fe775b471b457c643913e4bbd66c8d2` as confirmed-drainer EOAs.
- **Watchlist (HIGH)** — add both discharge wallets. They are dormant now but the discharge pattern is documented; if either wallet reactivates that is high-signal.
- **`reports/correction_log.md`** — no new correction; this is a validation, not a retraction. But Correction #20 should be cross-linked from this case file.

---

**Author:** SAI inferential-layer review, 2026-05-21
**Sources:** `approval_watchlist`, `contracts`, `deployers` (production DB via `railway ssh` 2026-05-21)
**Tier:** Tier A on every numerical claim above (counts, timestamps, addresses are direct DB queries against production state).
