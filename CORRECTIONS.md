# Layer 3 Corrections Log

A living record of claims made, errors found, and how they were fixed. Every entry includes discovery date, discovery method, what was wrong, what was actually true, and the commit or report where the fix landed.

**Purpose:** Make the project defensible under scrutiny. Customers and reviewers can reconstruct "what did Layer 3 claim on date X and was it true?" by reading this file chronologically.

**Format for each entry:**
```
## [YYYY-MM-DD] Short title
- **Claim:** what was asserted
- **Reality:** what the data actually shows
- **Discovery:** how it was caught
- **Fix:** commit/report where it was corrected
- **Severity:** HIGH / MEDIUM / LOW — impact if a customer had acted on the wrong claim
```

---

## 2026-04-02 Epistemic Integrity Audit — 6 Corrections

### "GoPlus detects 0 of the top 50 contracts"
- **Claim:** 100% detection gap vs GoPlus across 50 contracts
- **Reality:** GoPlus API was unreachable during testing (TLS SSL error). The "0/50" figure was from a prior session that never successfully queried GoPlus. At audit time, there was no GoPlus data in the database.
- **Discovery:** Section A1 of epistemic integrity test. Searched all 54 tables for GoPlus data, found none.
- **Fix:** Removed GoPlus claims from API responses. Added TODO to rerun benchmark when API accessible. Later discovered `goplus_results` table DID contain 50 prior benchmark results — now exposed in `/risk` endpoint as of commit 14bc4f8.
- **Severity:** HIGH — this was a central pitch claim with no verifiable backing.

### "org_001 has 899 deployers and 2,042 contracts"
- **Claim:** Specific headcount for org_001
- **Reality:** Actual numbers depend entirely on attribution method. `entity_classification`: 16 deployers / 462 contracts. `deployer_profiles.org_link`: 26 deployers. `funding_trail` JSON: 308 deployers / 1,856 contracts. Union of all methods: 324 deployers / 1,875 contracts. The "899" figure matched no clean query.
- **Discovery:** Section A2 of epistemic test. Ran three independent counts and compared.
- **Fix:** `/org/{org_id}` endpoint now returns `attribution_method: "funding_chain"` and a `methodology_note` showing the conservative count alongside. Commit d3ee103.
- **Severity:** HIGH — numbers were off by up to 20x depending on method.

### "Trust amplification factor of 14.2x"
- **Claim:** Router parasite operates at 14.2x the normal bytecode family rate
- **Reality:** No trust amplification table existed in the DB at audit time. The number came from an ad-hoc analysis that was never persisted. *Later update:* the table now has 32 rows with real data, including the `0xd4624228` parasite at router_percentage=96.6%. The original "14.2x" still cannot be reproduced from stored data.
- **Discovery:** Section C2 of epistemic test.
- **Fix:** Removed from API responses. Now re-exposed with verified fields (`total_callers`, `router_percentage`, `amplification_factor`) from the populated table. Commits d3ee103 (remove) and 14bc4f8 (re-add with real data).
- **Severity:** HIGH — key pitch number with no backing.

### "Camouflage ratio 68%"
- **Claim:** 68% of active contracts have <10% revert rate
- **Reality:** Actual ratio is 79.2% (computed fresh from transaction_events on contracts with 10+ interactions). Off by 11 percentage points.
- **Discovery:** Section A3 of epistemic test. Recomputed from raw data.
- **Fix:** API `/ecosystem/stats` now computes live from `transaction_events` instead of reading stale `camouflage_metrics` table. Commit d3ee103. The "stable across chains/weeks" claim still holds — spread is <15pp.
- **Severity:** MEDIUM — directionally correct, magnitude wrong.

### "832 wallet rotations, 302 high-confidence"
- **Claim:** 832 rotations detected, 302 high-confidence
- **Reality:** `deployer_similarity` has 4,879 pairs total. Thresholds produce 4,879 (≥0.70), 1,236 (≥0.80), 348 (≥0.85), 27 (≥0.90). None match "832" or "302" cleanly. With the temporal succession filter (≥0.85 AND one deployer's last_seen < other's first_seen), the count is **274**.
- **Discovery:** Section A4 of epistemic test.
- **Fix:** API `/ecosystem/stats` now reports 274 with explicit criteria: `rotation_criteria: "similarity >= 0.85 with temporal succession"`. Commit d3ee103.
- **Severity:** MEDIUM — wrong number but the underlying capability is real.

### "49 victim-to-predator conversions"
- **Claim:** 49 addresses were trap victims before becoming trap deployers
- **Reality:** The original query found 84 addresses that appear in both `transaction_events` (as reverted caller) AND `deployers` table. With a 24-hour filter requiring the first revert to precede first deployment by at least a day (to exclude operators who got reverted while probing), the count drops to **2**. Most of the original 84 were deployer-first operators, not victims-turned-predators.
- **Discovery:** Section A5 of epistemic test. Re-ran with stricter temporal filter.
- **Fix:** Number removed from external materials. The victim-to-predator narrative is essentially dead with the corrected filter.
- **Severity:** MEDIUM — narrative collapsed from "49 confirmed" to "2 with strict filter".

---

## 2026-04-05 Destroyed Implementation → EOA

- **Claim:** `0x93614117...` stored in `0x08b8b941`'s proxy slot 1 was a self-destructed implementation contract, indicating anti-forensic behavior. This was flagged as the "strongest evidence of anti-forensic behavior in our corpus" and as a potential infrastructure-layer extraction confirmation.
- **Reality:** `0x93614117...` is an EOA (externally owned account), not a destroyed contract. It has nonce > 0, is actively transacting on Aave V3, Uniswap V3, and ParaSwap. Zero bytes from `get_code` is normal for an EOA — not evidence of SELFDESTRUCT.
- **Discovery:** Blockscout API returned `is_contract: false`. Confirmed by checking the address's current transaction activity — actively doing DeFi operations from March 31 to April 5.
- **Fix:** Proxy watcher updated (commit a259755) to check nonce > 0 before flagging zero-code address as destroyed. Infrastructure layer report corrected to note: "Initial analysis flagged a potential anti-forensic implementation destruction. Verification revealed the address is an EOA, not a destroyed contract. The closed-loop infrastructure ecosystem is confirmed but evidence of extraction is inconclusive."
- **Severity:** HIGH — the wrong interpretation was used to support an infrastructure-layer extraction hypothesis. Correction prevents false "anti-forensic" confirmations in future proxy checks.

## 2026-04-05 Drain TX Hashes Don't Match On-Chain Events

- **Claim:** We can recover dollar values from the 963 documented drain events by fetching transaction receipts and parsing Transfer event logs.
- **Reality:** The drain transactions call a custom function (`e37136db`, unknown ASCII) that emits **zero log events**. Successful transactions with empty logs mean the extraction mechanism doesn't use standard ERC-20 Transfer events. Receipt parsing finds nothing.
- **Discovery:** Built drain_value_scanner.py, ran it on 500 drain receipts, found 0 transfer events. Debugged by fetching individual receipts and inspecting logs directly.
- **Fix:** Pivoted to checking current balances on trap contracts and their deployers. Result: ~$194 total across top 20 drain contracts. Most extracted value has already been moved out via the laundry pipeline. True drain amounts require `debug_traceTransaction` (trace-enabled RPC, not available).
- **Severity:** MEDIUM — the "we can value drains" claim was wrong, but we now know the extraction mechanism is invisible at the event log level. This is itself a finding about the anti-forensic design of these traps.

## 2026-04-05 Infrastructure-Layer Extraction Hypothesis Deflated

- **Claim:** `0x08b8b941` is an infrastructure operator extracting value from its own funded deployers, confirmed via the destroyed implementation and post-deployment callback pattern.
- **Reality:** The "destroyed implementation" was an EOA. Broader search across the corpus found **zero additional closed-loop operators**. The pattern (fund deployers + deploy service contracts + receive callbacks from funded deployers) is unique to this one address and the evidence of extraction is inconclusive. The funded deployers call the hub AFTER deploying (not before), which is suspicious but not diagnostic.
- **Discovery:** Phase 3 broader search ran against all 50 top funders, returned zero additional matches. Post-deployment call pattern is explainable as shared infrastructure usage, not necessarily extraction.
- **Fix:** Infrastructure layer report updated with honest assessment: "The infrastructure-layer threat model remains architecturally valid but lacks a confirmed instance in this corpus." Commit a259755.
- **Severity:** MEDIUM — the hypothesis is still valid as a concept but should not be claimed as confirmed.

## 2026-04-04 "Serial Victim Trapped 6x in 15 Minutes" Misread

- **Claim:** Bot `0x5901663c...` was trapped 6 times in 15 minutes across different contracts.
- **Reality:** The bot has 9 total trap events spread across Apr 3-4 (not 15 minutes). The "6 in 15 minutes" mental model conflated separate sessions. Actual pattern: 6 traps in 1 hour on Apr 3 (13:26-14:10), then 3 more traps in 15 minutes on Apr 4 (05:05-05:19). Still notable behavior but initial framing was imprecise.
- **Discovery:** Full timeline dump during April 5 investigation.
- **Fix:** Corrected in April 5 trap analysis. Updated framing to "9 traps across 2 days, clustered in rapid-fire sessions."
- **Severity:** LOW — the underlying observation (bot repeatedly hits traps without learning) holds, just the time window was wrong.

## 2026-04-05 Seven Optimism Trap Fires — Actually Three Chains Active

- **Claim:** "First Optimism trap fire" on April 4.
- **Reality:** Three Optimism trap fires had already occurred: April 3 00:00:29, April 3 10:35:13, April 4 12:34:43. The April 4 was the third, not the first. Missed in the daily summary because earlier detections were pre-sync.
- **Discovery:** Queried all trap events directly from Railway.
- **Fix:** Updated April 5 analysis to note Optimism trapping started April 3.
- **Severity:** LOW — chronology was wrong by a day.

---

## Infrastructure Corrections (Non-Data)

### 2026-03-31 WAL Bloat — 4.5GB Volume Fill
- **Problem:** SQLite WAL file grew unbounded, filling 5GB Railway volume in days.
- **Root cause sequence:**
  1. Initial: per-event commits generated massive WAL traffic. Fixed with batch commits per-block.
  2. Still growing: WAL checkpoint thread couldn't get exclusive access while monitors held connections. Fixed with aggressive PASSIVE checkpoints.
  3. Still growing: PASSIVE only moves pages, doesn't truncate. Fixed with connection cycling (close + reopen to allow TRUNCATE).
  4. Still growing: architectural — SQLite not designed for multi-process writers. Fixed with single-writer architecture (multiprocessing.Queue + dedicated writer process). Commit 0be703e.
- **Result:** WAL now stays under 100MB. DB grows only from real data.

### 2026-04-05 Railway Deploy Failures — Pre-Deploy Command
- **Problem:** Railway deploys failing repeatedly with "no such table: contracts" errors during startup.
- **Root cause:** The Railway agent had added `python -m surveillance.backfill_timelocks` as a pre-deploy command. The backfill script tries to read from the contracts table before the volume is mounted, causing it to fail and block deployment.
- **Fix:** Remove pre-deploy command entirely. Backfill is a one-shot admin script, not a deploy hook.
- **Severity:** HIGH — service was down through multiple deploy cycles.

### 2026-04-05 Multiprocessing Spawn Loop
- **Problem:** After switching from `fork` to `spawn` start method, child processes re-imported `run_surveillance.py` at module level, which re-executed the HTTP server start, DB cleanup, and writer spawn code — causing infinite process creation.
- **Fix:** Reverted to `fork` start method. Linux (Railway) supports fork, which inherits parent memory without re-executing module-level code. The original reason for switching to spawn (fd inheritance) was theoretical and not observed in practice.
- **Severity:** HIGH — crashed Railway deploy loop.

---

## 2026-04-07 Pipeline-Down Misread

- **Claim:** "Surveillance pipeline is down — last heartbeat 2026-04-06 20:23 UTC, ~13 hours stale"
- **Reality:** The Railway production pipeline was healthy and writing the entire time. The local SQLite at `surveillance/data/surveillance.db` was a stale dev checkout that hadn't been synced. The user's logs from Railway showed continuous activity through 2026-04-07 20:39+ that I had no visibility into until I ran `sync_railway_db.py`.
- **Discovery:** User pasted Railway logs showing live monitor activity on the same date my local query said "no rows".
- **Fix:** Ran `sync_railway_db.py` to pull production state. All future analysis sessions should begin with a sync (or query the live `/dump` endpoint directly) before claiming the pipeline is down.
- **Severity:** HIGH — would have triggered an unnecessary "restart Railway" intervention if the user had acted on it.

## 2026-04-07 Coffee Fleet Size Off By 4x

- **Claim:** Coffee fleet `0xc0ffeefeed8b9d27` has fleet size 55 (from `dormant_activations.fleet_size` and `deployers.total_contracts_deployed`)
- **Reality:** The actual contracts table shows **209 contracts** under that deployer. The `deployers.total_contracts_deployed` field is **stale** — it was last updated on 2026-03-30 when the deployer first crossed an alert threshold and hasn't been refreshed since. 60 of the 209 are confirmed (with on-chain victim evidence).
- **Discovery:** Joining `contracts WHERE deployer_address=?` while writing the case file. Counted directly instead of trusting the deployer record's stale field.
- **Fix:** Case file `surveillance/data/cases/CASE_COFFEE_FLEET_0xc0ffeefeed8b.md` documents the correct number. Need a periodic background job to refresh `deployers.total_contracts_deployed` from the contracts table — currently only gets updated at deployer-creation time.
- **Severity:** MEDIUM — under-counted the largest single trap fleet in the corpus by 4x in any report that referenced the deployer record.

## 2026-04-07 Self-Loop Trap Operator Was Logged But Never Promoted

- **Claim:** Implicit — that BOT+DEPLOYER warnings logged by the revert detector were sufficient to flag self-deploying trap operators
- **Reality:** The warning was a `logger.warning()` only — no contracts ever got promoted, no entity_type ever changed. `0x0e4c51936a3f74b4` had been hammering its own trap `0x55ab3b8397...` with selector `9aa6e55a` 2,015 times across 7 days (and selector `042ff559` 5,135 times against other targets) and the system kept logging "possible self-deploying trap operator" without acting on it. The trap stayed `suspected` until manually promoted on 2026-04-07.
- **Discovery:** While decoding the heaviest revert source in the live log capture from the user.
- **Fix:** `surveillance/revert_cluster_detector.py` now captures the `to` address of every reverted tx in `process_block`, and `_check_self_loop` queries the contracts table for any addresses the deployer is calling that they themselves deployed. Self-call → automatic promotion to `confirmed` with reason `"Self-loop: deployer X called own contract..."`. Commit `552baab`.
- **Severity:** HIGH — the system had the data and the warning but never closed the loop. There are likely other dual-role addresses still sitting at `suspected` that the new rule will catch on next encounter.

## 2026-04-07 19,000 ETH Trace Coverage Gap (closed via direct block scan)

- **Claim:** Initially "the 18,994 ETH already drained off" without saying where
- **Reality:** The 19k ETH was bridged to Ethereum L1 via the canonical Base bridge `0x4200000000000000000000000000000000000010` in tx `0x675afa1eb19e72ff3b453563a2b270e76d25b727292a331db633e918b2aad720` at block 44386396 (93 blocks / ~3 minutes after arrival). The recipient `0xe69f81b8...` called `withdraw()` (not `withdrawTo`) so the L1 destination is the **same address** on Ethereum mainnet, pending 7-day challenge period for finalization. `org_transfer_events` did not capture this because the bridge call is a direct contract call to a system precompile, not a top-level value transfer pattern that the org_transfer scanner watches.
- **Discovery:** Alchemy `alchemy_getAssetTransfers` returned 0 results for both inbound and outbound (Base support gap). Found the tx via brute-force `eth_getBlockByNumber` scan over blocks 44386300 → 44386450 looking for `tx.from == recipient`.
- **Fix:** Need to add a scanner that watches outbound calls from gas-station-class addresses to the Base/Optimism L2StandardBridge addresses (`0x4200000000000000000000000000000000000010`) and decodes the `withdraw`/`withdrawTo` calldata. Filed as follow-up.
- **Severity:** MEDIUM — we're missing org-level fund movements when the destination is L1 mainnet via the canonical bridge. For org_001 specifically this is a significant blind spot since the largest single tx of 2026-04-07 ($57M+ at $3K/ETH) used this exact path.

## 2026-04-07 Cross-Chain Rotation Misattribution

- **Claim:** "0x09d6f2c4b854 was funded by org_001 across all 3 chains within 90 seconds — cross-chain rotation"
- **Reality:** The deployer `0x09d6f2c4b854c5ec4e552e3754d38973ac83bbdd` only exists on optimism (1 contract). The 3 different `ORG LINK: deployer 0x09d6f2c4b854 funded by 0x8c826f795466 (org_001:gas_station)` log lines weren't 3 separate cross-chain deployments — they were the `auto_funder` running on each chain (base, arbitrum, optimism) and observing the same arbitrum-side funding tx. The funding tx is on arbitrum (block 450105987); the deployer first appears on optimism 8 minutes later. So the EOA *did* cross-chain (funded on arbitrum, deployed on optimism), but it's a single cross-chain hop, not a 3-way fan-out.
- **Discovery:** Looking at the actual deployer record in the synced DB and seeing `chain=optimism` only, then checking `funding_trail` JSON and finding the arbitrum tx.
- **Fix:** Updated investigation report. Real pattern is 1-hop chain rotation, not multi-chain simultaneous deployment. Still a valid Attack 5 instance but smaller in scope than originally reported.
- **Severity:** LOW — direction was right, magnitude was overstated.

---

## Summary of Wrong Numbers Previously Used in External Materials

| Claim | Correct Number | Status |
|-------|---------------|--------|
| "GoPlus detects 0/50" | 50 GoPlus results stored, most L3_ONLY matches. Not "0/50" but an honest gap of "L3 caught, GoPlus didn't check or rated clean" | FIXED in API |
| "org_001: 899 deployers" | 16-324 depending on method; primary funding_chain = 308 | FIXED in API |
| "Trust amplification 14.2x" | Data now exists (32 rows), actual amplification_factor per contract | FIXED in API |
| "Camouflage 68%" | Actual: 79.2% | FIXED in API |
| "832 wallet rotations" | Actual with temporal filter: 274 | FIXED in API |
| "49 victim-to-predator" | Actual with 24h filter: 2 | REMOVED from narrative |
| "Anti-forensic implementation destruction" | It was an EOA | FIXED in report + proxy watcher |
| "Drain dollar exposure = $211K" (parasite event) | Correct for parasite, but NOT total | Available for specific contracts with extraction_event data |
| "Infrastructure-layer extraction confirmed" | Architecturally valid, unconfirmed | FIXED in report |
| "Coffee fleet size 55" | Actual: 209 contracts (60 confirmed) | FIXED in case file 2026-04-07 |
| "Pipeline is down" (2026-04-07) | Pipeline was healthy; local DB was stale | FIXED via sync 2026-04-07 |

---

## What This Log Does Not Cover

- Claims made in prior sessions that weren't audited
- Numbers in Week 1/Week 2 reports that predate the epistemic test and weren't retroactively corrected in those documents
- Any number from before March 31 where the underlying data has since changed (the corpus grew from ~29K to 102K contracts, most pre-epistemic-test numbers are now stale by definition)

**The epistemic test is a point-in-time audit, not a running verification.** This log should be updated every time a claim is challenged or found wrong, not just during audits.
