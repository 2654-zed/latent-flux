# Kelp Retrospective Replay — Forensic Report

**Attack:** KelpDAO rsETH LayerZero OFT exploit, 2026-04-18, ~$292M
**Attack tx:** `0x1ae232da212c45f35c1525f851e4c41d529bf18af862d9ce9fd40bf709db4222` at Ethereum block 24,908,285
**Question this report answers:** What signals would Layer 3 have caught if it had been monitoring Ethereum the way it monitors Arbitrum / Base / Optimism?
**Discipline:** Tier A deductive claims separated from Tier B inferential. No claim of prevention (we have no enforcement layer). Every gap identified is an extension candidate, not evidence the framework is inadequate.
**Budget:** 50 RPC calls across the full 8-phase investigation. Phase 1 below used zero.

---

## Phase 1 — Corpus presence check (complete)

**Execution:** `scripts/kelp_retro_phase1.py` on the Railway production DB, 2026-04-18.

### Addresses probed

| Label | Address |
|---|---|
| Kelp OFTAdapter (Ethereum) | `0x85d456b2dff1fd8245387c0bfb64dfb700e98ef3` |
| Kelp required DVN (Ethereum) | `0x589dedbd617e0cbcb916a9223f4d1300c294236b` |
| Kelp required DVN (Unichain) | `0x282b3386571f7f794450d5789911a9804fa346b4` |
| Attack recipient | `0x8b1b6c9a6db1304000412dd21ae6a70a82d60d3b` |
| Ethereum endpoint receive library | `0xc02ab410f0734efa3f14628780e6e695156024c2` |
| Unichain endpoint send library | `0xc39161c743d0307eb9bcc9fef03eeb9dc4802de7` |

### Tables checked

`contracts`, `deployers`, `transaction_events`, `approval_events`, `entity_classification`, `bytecode_cache`, `bytecode_family_members`, `infrastructure_registry`, `alerts`, `trap_events`. Plus wide LIKE-scans against JSON payload / funding_trail / confidence_reason / bytecode_pattern_notes text columns for the recipient and adapter addresses.

### Result

**Zero hits across every table and every column for every address.** None of the six key addresses appears in our corpus in any form.

### Tier A interpretation

- The absence on the Ethereum-only addresses (4 of 6: OFTAdapter, ETH DVN, receive library, and the Unichain-only contracts) is **fully consistent with our monitoring footprint**: Layer 3 ingests Arbitrum / Base / Optimism, not Ethereum or Unichain. These addresses should not appear.
- The absence of the **attack recipient** is more substantive. The public post-mortem reports "The attacker deposited stolen rsETH to Aave V3 on Ethereum AND Arbitrum as collateral, borrowed ~$236M WETH." The Arbitrum leg is inside our monitoring scope. **The recipient address does not appear in our Arbitrum `transaction_events`, `approval_events`, or any other table.** That has two possible explanations:
  1. The attacker used a **different wallet** on Arbitrum — the recipient `0x8B1b…0D3b` received the rsETH on Ethereum and either bridged to Arbitrum under a different address, or moved the rsETH through intermediate wallets before the Aave deposit.
  2. The Aave V3 deposit happened, but our ingest doesn't capture it because **Aave V3 on Arbitrum is not in our `contracts` table** (legitimate infrastructure, no trap bytecode signatures at ingest time), and our `approval_events` is scoped to Permit2-family and flagged-contract approvals, not general-purpose approvals to arbitrary DeFi protocols.

### Tier B interpretation

**Explanation #2 is almost certainly the proximate cause of the miss.** Our pipeline is built to catch trap contracts. Aave V3 is not a trap contract. When the attacker's wallet (whether `0x8B1b…` itself or a downstream address) approved USDC/rsETH to Aave V3 on Arbitrum and deposited the collateral, there was no trap-adjacent signature for our detectors to fire on. The activity was indistinguishable from legitimate DeFi use. This is a detection-surface choice, not a bug.

**This is a concrete methodology gap worth naming up front, before later phases elaborate it.** Our current ingest is biased toward the adversarial bytecode population. Attacks that propagate through legitimate infrastructure leave no residue in our corpus unless a trap-adjacent address shows up as a party.

### What this means for the overall retrospective

- **Phases 2–6 are about what we could have caught had we monitored Ethereum / LayerZero OApp configs.** The answer is expected to be "a lot, with significant lead time," but every claim there is hypothetical — predicated on infrastructure we don't yet run.
- **Phase 7 (Arbitrum leg) is the one place where we had the access but missed the signal.** Phase 1 has already effectively answered Phase 7 with this negative finding. The gap is not "we didn't see the tx"; it's "our ingest pipeline doesn't index interactions with Aave V3 because Aave V3 isn't trap-adjacent at deploy time." Phase 7 will confirm this more precisely by checking whether ANY address associated with the attack shows up in our Arbitrum data — not just the initial recipient.
- **Phase 8 (methodology gap inventory) has its first entry for free**: "ingest scope does not include general-purpose DeFi approvals / interactions, so cross-chain attack downstream propagation through legitimate DeFi infrastructure is invisible unless the attacker uses a previously-flagged wallet." That's a concrete, defensible gap with a clear extension path — index ERC-20 approvals to a curated set of major DeFi protocols (Aave, Compound, Pendle, etc.), or more broadly add a "symptomatic activity" detector that watches for unusually-large collateral deposits by new/rarely-seen addresses on monitored chains.

### Pause point

Phase 1 complete. No RPC calls spent (0 of 50 budget). No writes to any table. Ready for Phase 2 (infrastructure_registry retrospective population) once findings above are reviewed.

**Phase 2 pre-flight questions for your approval:**
1. The original handoff proposes new `infrastructure_registry` entries marked as "retrospective reference" rather than `monitored_chain=1`. The registry currently has no `monitored_chain` column — only `(address, chain, classification, verified_at, verification_source, notes)`. Proposed approach: use `classification='kelp_oft_adapter_retrospective'` or similar retrospective-tagged classification strings; use the `notes` field to indicate "retrospective reference — not a forward-watch entry." Or: add a `retrospective INTEGER DEFAULT 0` column. Preference?
2. The Unichain DVN is on a chain we don't monitor at all. Adding it to `infrastructure_registry` with `chain='unichain'` creates the table's first non-monitored-chain entry. Acceptable, or do you want `infrastructure_registry` scoped to chains we monitor + Ethereum?
3. Budget for Phase 2: 0 RPC calls (all four addresses already known from the post-mortem; no verification lookups needed). Phase 3 is the first phase that spends budget (5 calls for historical `getConfig` reads).

---

## Phases 2–8 — pending approval

Scoped in the original handoff. Not executed. Summary of per-phase RPC budget:

| Phase | Description | RPC budget |
|---|---|---|
| 2 | infrastructure_registry retrospective entries | 0 |
| 3 | Historical `getConfig` at 5 pre-attack blocks | 5 |
| 4 | Attack recipient funding trace | 20 |
| 5 | Pre-attack anomaly scan (DVN signing history) | 10 |
| 6 | Attack tx anomaly scoring | 0 (local) |
| 7 | Arbitrum leg retrospective | 0 (local) |
| 8 | Methodology gap inventory + synthesis | 0 |
| **Total** | | **≤ 35 of 50** |

---

## What this document will NOT claim

- No claim that Layer 3 would have **prevented** the Kelp attack. We have no enforcement layer.
- No quote-ready "Layer 3 caught Kelp" sentence unless Phase 3 validates that the catastrophic configuration was readable via `getConfig` at the blocks we probe. That's a Tier A dependency — until we return the actual `getConfig` response from a pre-attack block, any statement that "Layer 3 would have flagged Kelp 30 days in advance" is unsubstantiated.
- No attribution of the compromised DVN operator. Whether the attacker obtained signing keys via off-chain compromise or was themselves operating the DVN is outside our observation surface.
- No recovery-projection claims. Kelp + LayerZero + Aave coordination is ongoing.
