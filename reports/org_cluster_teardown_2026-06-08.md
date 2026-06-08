# Forensic Teardown — Every Named Org / Cluster / Entity vs Ground Truth

**Date:** 2026-06-08
**Method:** cross-reference every named organization, role-wallet, classified entity, and facilitator against the only ground-truth harm signal in the corpus — the tx-initiator-verified drain set (266 drains / 259 victims / 59 drained contracts / 104 drainers, Correction #29) — plus on-chain identity spot-checks (Blockscout, 0 Alchemy CU). Operator-requested.
**Posture:** adversarial. A "criminal organization / drainer / trap" claim must show ground-truth harm from its OWN monitored-chain activity, not shape, labels, or a catalog SUM.

---

## 0. Verdict

The entire named org/cluster/entity layer has **near-zero ground-truth adversarial output.** Across **5 organizations (~5,000 deployers, ~10,300 contracts), 53 curated role-wallets, 22 adversarial-classified entities, and 7 "rogue facilitators,"** the verified drain set contains **1 drained contract and 0 draining deployers/role-wallets.** The one documented real harm outside that is D270's ~$5K OP drain (itself the retired "$3.1 quadrillion" event). Everything else is **funding-chain clustering + curated role-narratives + shape/association labels**, and — for org_001 — a headline ($285M) fabricated by a SQL sweep of unrelated external incidents (Correction #30).

This is the house of cards at the organizational layer: the same shape-vs-intent error that inflated the drain count, one level up.

---

## 1. Organizations (funding_trail clusters)

| Org | deployers | contracts | confirmed | **verified-drain contracts** | **deployers that drain** |
|---|---|---|---|---|---|
| org_001 | 1,445 | 6,842 | 18 | **1** | **0** |
| org_002 | 3,507 | 3,249 | 2 | **0** | **0** |
| org_005 | 13 | 69 | 0 | **0** | **0** |
| org_006 | 18 | 119 | 1 | **0** | **0** |
| org_007 | 7 | 28 | 0 | **0** | **0** |
| **total** | **~4,990** | **~10,307** | **21** | **1** | **0** |

- **Membership = shared funding trail**, which is method-arbitrary (org_001 counted at 899 → "use 308" → 1,445) and not evidence of coordination — sharing a funding source (Coinbase/Binance) is near-universal.
- **org_001's 18 "confirmed"** are mostly FPs: **3 verified Chainlink `AccessControlledOCR2Aggregator` oracles**, 6 "bait-deposit" inferences (0 drains), ~7 "bot-interacted" (0 drains), 2 named tokens (Yupp AI carries the lone drain). Genuine: ~1.
- **org_001's "$285M+"** is a `SUM(...) WHERE event_id LIKE 'EXTRACTION_00%'` artifact that swept in EXTRACTION_005 ($285M Drift hack, **Solana**, **DPRK**), 004 ($18.4M Rhea, **NEAR**), 009 ($5M Wasabi) — external incidents on unmonitored chains. Real org_001 traced ≈ **$257K** (Correction #30).
- **org_002/005/006/007:** funding clusters with **0 ground-truth harm**.

**Verdict:** all five are funding/infrastructure clusters, not coordinated criminal organizations. org_001 has a ~1-contract real sliver; the rest, none.

## 2. Curated role-wallets (`org_wallets`, 53)

Treasury / operator / cashout / gas_station / laundry / exit_cex / lp_staging / defi_exit_channel across the 5 orgs. **0 of 53 appear as a drainer or drained contract in the verified set.** The org graphs additionally list **legitimate infrastructure as "nodes"**: Coinbase, Binance, **LI.FI Bridge** (`0x1231deb6…`), WETH wrapper, MEV bots.

**Verdict:** a curated role-narrative over a funding cluster + real infrastructure. No demonstrated harm.

## 3. Classified entities (`entity_classification`, 1,080)

Mostly non-adversarial: 617 `unclassified_bot`, 334 `private_infrastructure`, plus explicitly-legit categories (`dex_router` 19, `cex_hot_wallet` 10, `dex_pool` 5, `bridge` 4, `lp_manager` 3, `gas_station` 23). Adversarial-classified: **16 `known_attacker` + 6 `trap_contract` = 22**.
- The 16 `known_attacker` are labeled circularly (`notes: deployers.entity_type=known_attacker`), conf=HIGH, with **0 in the verified drain set**.
- The 6 `trap_contract` are org_001's bait-deposit contracts — **0 drains**.

**Verdict:** the catalog is ~98% bots/legit infrastructure; the 22 adversarial labels have **0 ground-truth harm**.

## 4. The 7 "rogue facilitators" (`x402_facilitators` classification=`rogue`)

A7B9, E3B2, E717, CE5E, D270, 881E, F71C — EOAs labeled "DRAINER"/"SUSPECT". All `total_volume=0.0`; **0 in the verified approval-drain set**; on-chain they are modest active EOAs (18–50 recent token transfers; A7B9 holds 287 ETH, rest ~0) with **no industrial-drainer signature** (no mass victim inflows). Labels rest on nonce magnitude + funding association (881E "PROBABLE", F71C "funded by D270"). **Only D270 has documented real harm: the ~$5K OP drain** (the event retired from "$3.1 quadrillion", decimals bug).

**Verdict:** 1 tiny real drainer (D270, ~$5K), 6 thin/association-based labels. Not a documented facilitator network.

## 5. x402 facilitators (437 addresses) and bytecode families

- `x402_facilitators`: 371 unknown, 60 `known`, 26 `benign_relayer`, 7 `rogue` (§4). The non-rogue are **legitimate Coinbase x402 payment infrastructure** (e.g. `x402ExactPermit2Proxy`, sourced from Coinbase's GitHub). **0 of 437 in the verified drain set.**
- `bytecode_families` / `bytecode_family_members`: **0 rows** — the "T2-eaef6a5d family" and "Coffee fleet" exist only as case-file/doc narratives, not as a populated, queryable adversarial set. T2-eaef6a5d is the ~75% NULL bytecode bucket (CLAUDE.md priority #9, uncharacterized — i.e. *unclassified*, not *adversarial*).

---

## 6. Total ground-truth adversarial output across the entire named layer

| Source | Real harm |
|---|---|
| 5 orgs / 53 role-wallets / 22 entities / 7 facilitators / 437 x402 | **1 drained contract (org_001) + D270's ~$5K OP drain + the 2 named tokens (1 drain)** |

Out of ~5,000 deployers, ~10,300 contracts, and ~500 named addresses, the demonstrable on-chain harm is **a single-digit number of contracts and ~$5K**. The "$285M+ criminal empire" was a query artifact.

## 7. Recommendation

- **Demote the org/cluster layer to "funding/infrastructure clusters" (leads), not "criminal organizations" (verdicts).** Same standard as the tier demotion: no "organization" claim without ground-truth harm from its own monitored-chain activity + an explicit attribution chain.
- **Retract org-scale and $-extraction claims** from pitch/narrative materials (org_001 $285M done in CORRECTIONS.md; org_002–007 carry no harm to cite).
- **Keep** the small real signal as forensics: the 266 drains / 104 drainers, org_001's ~1 real contract, D270's OP drain.
- The `extraction_events` incident catalog (Drift/Rhea/Wasabi) is legitimate research — keep as a catalog, never auto-attributed to an org.
