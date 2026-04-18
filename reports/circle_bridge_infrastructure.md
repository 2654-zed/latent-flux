# Circle Bridge / CCTP Infrastructure — Identification Report

**Date:** 2026-04-18
**Scope:** Part 1 Phase 1 of the Circle Bridge handoff — identify Circle's cross-chain USDC infrastructure on Base, Arbitrum, Optimism; establish the epistemic tier of each claim; draft (not execute) the storage layer needed to track them.
**Status:** Findings + draft migration + draft INSERTs. Nothing executed. Pause for review.

---

## Q3 resolution (epistemic_tag schema probe)

Before the investigation, I checked whether `alerts.epistemic_tag` has a CHECK constraint. **It doesn't — the column doesn't exist in the live schema.** The `alerts` table currently stores `{id, alert_type, address, tx_hash, block_number, timestamp, payload, false_positive}`. Existing producers (e.g. `trust_amplification.py:299`) encode `epistemic_tag` as a JSON key inside the `payload` blob — convention, not enforcement.

Implication: per your Q3 answer, I use existing string values in the JSON payload and let `alert_type` carry the nuance. The mapping for this workstream:

| Finding class | `alert_type` | JSON `epistemic_tag` |
|---|---|---|
| Circle Bridge verified contract | `verified_infrastructure` | `deductive` |
| Impersonator (Part 1 Phase 2) | `infrastructure_impersonation` | `assessed` |
| Behavioral laundering (Part 2) | `reputation_laundering_candidate` | `assessed` |

No schema change. Matches how `trust_amplification` / `camouflage_tracker` already tag their output.

---

## Tier A findings — verified CCTP v2 addresses (deductive)

Source: `developers.circle.com/cctp/evm-smart-contracts` (fetched 2026-04-18).

CCTP v2 uses CREATE2 deterministic deployment, so **the same address is used on every EVM chain Circle supports.** This is an important observation for our work: the canonical Circle cross-chain USDC infrastructure isn't per-chain addresses we'd individually watchlist — it's a single cross-chain identity with a different domain ID per chain.

| Component | Address (all EVM chains) | Role |
|---|---|---|
| TokenMessengerV2 | `0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d` | Entry point for `depositForBurn`; user-facing |
| MessageTransmitterV2 | `0x81D40F21F12A8F0E3252Bccb954D722d4c464B64` | Cross-chain message relay; calls TokenMinter on destination |
| TokenMinterV2 | `0xfd78EE919681417d192449715b2594ab58f5D002` | Mint/burn authority over USDC; trusted by USDC token contract |
| MessageV2 | `0xec546b6B005471ECf012e5aF77FBeC07e0FD8f78` | Message encoding / validation library |

CCTP domain IDs for the three chains we monitor: Base=6, Arbitrum=3, Optimism=2.

**"USDC Bridge" as a consumer product:** Circle's docs as fetched do not identify any contract explicitly labeled "USDC Bridge" or "BridgeOperator" distinct from CCTP core. Circle USDC Bridge appears to be a consumer-facing UI wrapper over CCTP v2. No separate contracts identified in developer documentation. If distinct operator contracts emerge, update the registry. The consumer-wrapper question is answerable once Circle's bridge product is actually launched and visible on-chain; right now CCTP v2 is the infrastructure to register.

**CCTP v1 addresses:** Circle's current developer documentation no longer surfaces v1 addresses (the v2 page replaces them). I did not hardcode v1 addresses from memory because they should be verified before any production classification. If v1 is still relevant to the ecosystem, a separate WebFetch pass against Circle's GitHub repo or historical docs is the honest path. Flagging this as an open question.

---

## Corpus presence check

Zero across the board. None of the CCTP v2 addresses (or the three v1 TokenMessenger addresses I'm 80%-confident on from public knowledge) appear in our corpus:

| Table | Circle contract rows |
|---|---|
| `contracts` | 0 |
| `transaction_events` (as contract_address OR interacting_address) | 0 |
| `approval_events` (as spender) | 0 |
| `entity_classification` | 0 |
| `bytecode_cache` (as source_contract) | 0 |
| `contracts` (as deployer_address) | 0 |

This is consistent with system design, not a bug:
- Our pipeline captures **new contract deployments** on monitored chains. CCTP v2 was deployed well before our corpus start (2026-03-17), so these contracts are outside our ingest window.
- The ~90k contracts we monitor are bytecode we classified as trap-candidate or unknown; a legitimate, heavily-used bridge does not deploy new contracts per user interaction, so CCTP hasn't seeded our bytecode family trees.
- We don't capture the full approval firehose; `approval_events` is populated only when a tracked contract is the spender or when Permit2 is involved. CCTP TokenMessenger receives USDC via `depositForBurn` (ERC20.transferFrom) — generating `approval` to TokenMessenger would appear only if our indexer tracked general approval events, which we don't.

---

## Stored potential — framework-level analysis (not data-driven)

**I cannot compute a `score_contract` value for Circle's CCTP contracts.** `risk_scoring.score_contract` reads `contracts`, `deployers`, `transaction_events`, `trap_events`, `approval_events` — all of which return zero rows for CCTP addresses. A computed score would be artifactually low (no trap events → realized_value denominator = 1, but stored potential components also zero → score ≈ 0). That would be wrong: CCTP is at maximum capability, maximum trust binding, high mutability. The score would report zero risk because we don't have the data to measure it.

**The honest response is to apply the stored-potential framework by reasoning, not computation, and flag the data gap.** Per the Adversarial Topology framework in `CLAUDE.md`:

| Primitive | CCTP TokenMessengerV2 assessment |
|---|---|
| **Position** | Trusted cross-chain USDC mover. Sits between user's wallet and destination-chain USDC supply. Can observe all bridge flows. Cannot see destination wallet except what the user encodes in the message. |
| **Permissions** | Holds `depositForBurn` authority (burns user USDC on source chain, triggers mint on destination). Implicit permission: any user who calls it approves it as a one-shot spender. Single call = max-scope behavior for that call, but not retained after. |
| **Trust bindings** | "Circle" brand. USDC token contract trusts `TokenMinterV2` for mint authority. Wallets (Metamask, Coinbase) surface CCTP as a preferred bridge in UI. Users assume correctness because Circle issues USDC. |
| **Mutability** | CCTP v2 uses proxy patterns (`MessageTransmitterV2` is upgradeable per Circle's architecture docs). Circle retains admin authority. An upgrade could change message validation, minter authority delegation, or attestation requirements — without re-consent from users. |
| **Observation capability** | High. Every cross-chain USDC flow through CCTP passes through these contracts. The `depositForBurn` and `MessageReceived` events expose sender, amount, destination domain, destination address. Circle sees the full cross-chain USDC graph. |

**Qualitative stored-potential tier: VERY HIGH.**
- Capability: 25/25 — mint authority over USDC is maximal
- Permissions: 20/25 — single-call scope per user, but scope of "a single call" is the user's full bridge amount
- Trust binding: 25/25 — brand + wallet-UI preference + USDC issuance relationship
- Mutability: 15/25 — upgradeable proxy, Circle-controlled admin
- Observation: this isn't a stored-potential component in our current model, but it's elevated.

Per the core interpretive rule from CLAUDE.md ("A contract with maximum capability… is at PEAK stored potential — not minimum risk. The absence of realized value is the danger signal"), CCTP is exactly the case the framework describes as loaded. The absence of any realized malicious value is the expected state for legitimate infrastructure; the framework says this loading is meaningful regardless.

**This is Tier B (inferential) reasoning, not Tier A.** Tier A would require CCTP call-logs in our DB, which we don't have.

---

## Draft migration — `infrastructure_registry` table

Not executed. Proposed shape follows Q1 answer.

```sql
-- Migration: infrastructure_registry
-- Purpose: registry of known-legitimate high-stakes infrastructure that
-- the surveillance methodology should handle correctly. First entries:
-- Circle CCTP v2 contracts. Over time: Uniswap routers, Aave pools, etc.
-- This becomes the "known-legitimate template baseline" (priority #2 in
-- CLAUDE.md current priorities). entity_classification can reference this
-- later if that turns out to be the right cross-link design.

CREATE TABLE IF NOT EXISTS infrastructure_registry (
    address              TEXT    NOT NULL,
    chain                TEXT    NOT NULL,
    classification       TEXT    NOT NULL,
    verified_at          TEXT    NOT NULL,
    verification_source  TEXT    NOT NULL,
    notes                TEXT,
    PRIMARY KEY (address, chain)
);

CREATE INDEX IF NOT EXISTS idx_infra_registry_class
    ON infrastructure_registry(classification);
```

Migration hook in `db.py` follows the existing pattern at lines 48–63 (bytecode_cache migration): check `sqlite_master` for the table; if absent, `executescript` the DDL above.

**Q: Why PRIMARY KEY (address, chain) rather than just (address)?**
Because CCTP v2 uses the same address on every chain. A row per (address, chain) is needed to separately encode classification per chain (which is always the same for CCTP but may differ for other infra — e.g., a Uniswap router on Base vs Arbitrum has the same address but different governance implications).

---

## Draft INSERT statements — Circle CCTP v2

Not executed. Preloaded with `verified_at = '2026-04-18'` and `verification_source` pointing at the fetched Circle docs URL.

```sql
-- Per-chain CCTP v2 entries. Four contracts × three chains = twelve rows.
-- CCTP v2 address is CREATE2-deterministic (same on every EVM chain).
INSERT INTO infrastructure_registry
    (address, chain, classification, verified_at, verification_source, notes)
VALUES
    ('0x28b5a0e9c621a5badaa536219b3a228c8168cf5d', 'base',     'circle_cctp_token_messenger_v2',     '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 TokenMessenger — user-facing depositForBurn entry point. Stored potential: VERY HIGH (see reports/circle_bridge_infrastructure.md). Epistemic: deductive from Circle docs.'),
    ('0x28b5a0e9c621a5badaa536219b3a228c8168cf5d', 'arbitrum', 'circle_cctp_token_messenger_v2',     '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 TokenMessenger — user-facing depositForBurn entry point. Same address on all chains via CREATE2.'),
    ('0x28b5a0e9c621a5badaa536219b3a228c8168cf5d', 'optimism', 'circle_cctp_token_messenger_v2',     '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 TokenMessenger — user-facing depositForBurn entry point. Same address on all chains via CREATE2.'),

    ('0x81d40f21f12a8f0e3252bccb954d722d4c464b64', 'base',     'circle_cctp_message_transmitter_v2', '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 MessageTransmitter — cross-chain message relay. Circle-controlled upgradeable proxy (Tier B mutability claim).'),
    ('0x81d40f21f12a8f0e3252bccb954d722d4c464b64', 'arbitrum', 'circle_cctp_message_transmitter_v2', '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 MessageTransmitter — cross-chain message relay. Same address on all chains.'),
    ('0x81d40f21f12a8f0e3252bccb954d722d4c464b64', 'optimism', 'circle_cctp_message_transmitter_v2', '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 MessageTransmitter — cross-chain message relay. Same address on all chains.'),

    ('0xfd78ee919681417d192449715b2594ab58f5d002', 'base',     'circle_cctp_token_minter_v2',         '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 TokenMinter — mint/burn authority over USDC on destination chain. Maximum-capability node in the stablecoin ecosystem.'),
    ('0xfd78ee919681417d192449715b2594ab58f5d002', 'arbitrum', 'circle_cctp_token_minter_v2',         '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 TokenMinter — mint/burn authority over USDC on destination chain. Same address on all chains.'),
    ('0xfd78ee919681417d192449715b2594ab58f5d002', 'optimism', 'circle_cctp_token_minter_v2',         '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 TokenMinter — mint/burn authority over USDC on destination chain. Same address on all chains.'),

    ('0xec546b6b005471ecf012e5af77fbec07e0fd8f78', 'base',     'circle_cctp_message_v2',              '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 Message library — message encoding/validation helpers. Read-only in practice.'),
    ('0xec546b6b005471ecf012e5af77fbec07e0fd8f78', 'arbitrum', 'circle_cctp_message_v2',              '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 Message library — message encoding/validation helpers.'),
    ('0xec546b6b005471ecf012e5af77fbec07e0fd8f78', 'optimism', 'circle_cctp_message_v2',              '2026-04-18', 'https://developers.circle.com/cctp/evm-smart-contracts (fetched 2026-04-18)', 'CCTP v2 Message library — message encoding/validation helpers.');
```

12 rows total. All classifications follow the convention `circle_cctp_<role>_v<version>` to leave room for v1 if we backfill and for non-CCTP Circle infra if it surfaces.

**Deferred (not in this draft):**
- CCTP v1 per-chain addresses — need verification source, not committing from memory.
- A "circle_usdc_bridge_frontend" classification for any consumer-facing wrapper product — not identified in fetched docs; open question for the next fetch pass.

---

## Phase 2 — impersonator detection (scoped, not yet run)

Blocked on your approval of Phase 1. If green-lit, Phase 2 runs without further schema work:

- `vanity_attention_scanner` against each Circle address → surfaces vanity lookalikes planted in victim tx histories
- `bytecode_families` search for families containing CCTP-style selectors (`depositForBurn`=0x6fd3504e, `receiveMessage`=0x57ecfd28, per CCTP v2 interface) with deployers NOT in infrastructure_registry
- `deployer_profiles` scan for deployers with ≥3 contracts mimicking CCTP selectors

Output: `reports/circle_bridge_impersonators.md`. Any findings get `alert_type=infrastructure_impersonation` + `epistemic_tag=assessed`, plus a `deployer_profiles.org_link = "circle_impersonator_cluster"` candidate flag (not committed; proposed only).

---

## Phase 3 — monitor proposal (scope-only, per Q2)

Per Q2 clarification, Phase 3 is design/cost in this report; no code written. Proposed:

**New monitor:** `event_monitors.py::monitor_circle_bridge()` would subscribe to `DepositForBurn` and `MessageReceived` events on TokenMessengerV2 and MessageTransmitterV2 respectively. Stores `(block, tx_hash, sender, amount, destination_domain, destination_address)` into a new `cross_chain_flow_events` table.

**Cost to build:** ~150 LoC new module + one schema migration (~40 LoC). Reuses the WebSocket subscription machinery of existing `event_monitors`.

**Cost to run:** Zero additional RPC budget — events are delivered through the existing WSS subscriptions. One extra INSERT per cross-chain flow; at Base's CCTP volume, maybe ~50–200 events/day across our three chains.

**Value:** Aggregate cross-chain USDC volume tracking gives us:
- Correlation between Circle Bridge launch and observed victim patterns (do drainers migrate to CCTP because of anti-fraud friction on third-party bridges?)
- Baseline ecosystem data for the "which stablecoin recovers more reliably" commercial question
- Detection surface for impersonators — any contract emitting fake `DepositForBurn` events at unusual addresses

**Decision needed:** Build this monitor after Phase 1 lands, or wait until Phase 2 surfaces active impersonation (which would be the trigger for needing flow baselines)?

---

## Open questions

1. **Is "Circle USDC Bridge" a distinct consumer product with its own contracts, or a re-marketing of CCTP v2?** Developer docs don't identify a distinct contract; the product page at `circle.com/usdc-bridge` may or may not. Worth a second WebFetch before closing Phase 1.
2. **Do we want CCTP v1 in the registry too?** v1 is still active on some chains. If yes, I need verified v1 addresses from Circle's GitHub repos, not memory.
3. **`entity_classification` cross-link.** You deferred the `circle_bridge_operator` entity_classification write. Once `infrastructure_registry` has rows, do we want `entity_classification.category = 'legitimate_infrastructure'` with a FK to the registry? Not in scope for Phase 1 but worth deciding before Phase 2.

---

## What was not built

- No migration executed. `infrastructure_registry` does not yet exist in the DB.
- No INSERT executed. No rows added anywhere.
- No `entity_classification` writes (per your explicit deferral in Q1 answer).
- No monitor module (per your Q2 answer — scope-only in this report).
- No alerts generated.
- No behavioral data-driven stored-potential score for Circle contracts (cannot be computed with current corpus coverage; would require ingesting CCTP call logs, which is a pipeline shape-change not in scope).

Pause here per checkpoint. Next steps on your approval: execute migration + INSERTs on local, mirror on Railway, proceed to Phase 2.
