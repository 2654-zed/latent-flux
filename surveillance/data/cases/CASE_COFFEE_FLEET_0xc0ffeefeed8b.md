# Case File — Coffee Fleet `0xc0ffeefeed8b9d27`

**Status:** Active. Largest single-deployer trap fleet in the corpus.
**Opened:** 2026-04-07
**Chain:** base only

---

## Identity

**Deployer:** `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e`
**Funder of deployer:** `0x7c8b9874f7be10ba196d3bb6fe1f45556c0bc1b5`
**Original funding tx:** `0xc99fe50ee735019fefd48c6886ceffcf9116bd419f19b5c71e4f5a1a0505a78a`
**Original funding date:** 2024-09-17 20:33:29 UTC (0.1 ETH)
**Pre-deployment dormancy:** ~18 months (funded Sep 2024, first contract deployed Mar 2026)

The deployer record's `total_contracts_deployed` field shows `55` and is **stale** — actual count in the contracts table is `209` (corrected 2026-04-07).

---

## Scale

| Metric | Value |
|---|---|
| Contracts deployed | **209** |
| Confirmed (with on-chain victim evidence) | **60** |
| Suspected | 142 |
| Unknown | 7 |
| First contract | block 43545766 / 2026-03-19 00:41 UTC |
| Most recent contract | block 44397390 / 2026-04-07 17:48 UTC |
| Active deployment window | 19 days, ongoing |

## Victim record

| Metric | Value |
|---|---|
| Total trap_events | **366** |
| Unique victim bots | **84** |
| First victim hit | 2026-03-21 12:58 UTC |
| Most recent hit | 2026-04-07 20:33 UTC |

**100% of victim bots have `0xc0ffee` vanity prefixes.** Not "most" — all 84.

This is the defining feature of this case. Coffee fleet is **exclusively trapping coffee fleet**. There are no non-coffee-fleet victims in the entire 19-day window. Two interpretations:

1. **Single operator running both sides.** One actor controls the trap-deployer + a fleet of `0xc0ffee` MEV scanner bots that systematically scan all `0xc0ffee` contracts. The "victims" are part of the same operation — possibly a load test, possibly a self-laundering pattern that creates fake "trap fired" evidence on-chain to bait third-party scanners into scoring the contracts as more dangerous than they are.

2. **Closed coffee-fleet adversary ecosystem.** Multiple independent operators all chose `0xc0ffee` vanities, and they have converged on a self-contained scan-and-trap loop nobody outside the cluster touches. Less likely given the perfect bipartition.

The 19-day all-coffee-fleet record is too clean to be coincidence. **Working hypothesis: single operator.**

---

## Top victim contracts

| Contract | Hits | Bots |
|---|---|---|
| `0xedfc1a1fc08b48faca8db52d7714499475612f41` | 36 | 36 |
| `0x76c322661e376a0db4be5885dbbce73f9d49015c` | 27 | 27 |
| `0x9c5ef32f2da0fb43623c5febfb94d6e0d54926d9` | 22 | 22 |
| `0x564414fb446ea13b95b8bfe642b170f74d746fd9` | 20 | 20 |
| `0x65e0c2f55cc17bd538bd8defdc1a29ab1abde8e0` | 17 | 17 |
| `0x620a9fa4f2f0af230273b9845a83e7283b762a57` | 17 | 17 |
| `0x7bde2eac44b46d802c2268fef19b2b8489de1ec8` | 17 | 17 |
| `0x7b9b868588a023388a931dfdef1a665393081971` | 16 | 16 |
| `0xf2a82dbbde3ef1647fabbd9cdfe8f3165d453363` | 16 | 16 |
| `0x1d8f51d546f20caa903bf0f4381060a3afe8017f` | 15 | 15 |

Note: every confirmed contract has a 1:1 hits-to-bots ratio. Each bot only ever trips a given contract once — bots learn (or rotate). This is consistent with a scanner sweep where each scanner address is single-use.

---

## Bytecode profile

Sampled 30 suspected contracts and computed sha256 prefixes:
- **30/30 unique** — no template reuse
- Bytecode sizes range **1,609 → 10,199 bytes**
- Each contract is freshly compiled with distinct constants

This matches the same anti-fingerprint pattern as `0x0e4c51`'s tokens (Investigation 1, 2026-04-07): same author distinguishable by deployer + behavior, but bytecode hashes deliberately diversified to evade static-fingerprint scanners.

The 60 confirmed contracts have on-chain confirmation reasons of two flavors:
- **Backfill confirmations** (older, blocks 43545766 → 43877672): `Behavioral confirmation: N victims trapped (backfill)` — historical scan of bot losses
- **Live confirmations** (recent, blocks 44080574 → 44349036): `Behavioral confirmation: bot 0xc0ffee...` — caught in real time, attributing the specific coffee-fleet bot that tripped

---

## Activation pattern

`dormant_activations` table records 9 distinct first-callers on a single coordinated wake at **2026-04-04 14:06:02** (chain=base, fleet active going from 0 → 9 in one batch). All 9 first-callers are coffee-fleet vanities:

```
0xc0ffee077edd3997c2a65ef68c71a5bc6400051a
0xc0ffee2a32bc8d7799764ef72caa075276908484
0xc0ffee410b604164c6394b1e918362a70bf8d091
0xc0ffee4582039cc176c77a0da7f61293abcd65cb
0xc0ffee59f94f54f4f293f01672976408bc1cad7f
0xc0ffee8cae1d4279e42fba7a6cafeb8e1401140f
0xc0ffeeb5141ee829d163e56ab1e1519240d3979c
0xc0ffeee770e501395a49833100b41f31429b8f9c
0xc0ffeeec990e2e50d1589bc9120769455a104d6d
```

Subsequent activations (visible in the live log capture from 20:46:13 UTC on 04-07) showed fleet `0xc0ffeefeed8b9d27` going through 18 active contracts, with another fleet `0xe29a2cbd0c5a300d` (size 11) waking 6 contracts in one block — a 55% wake-up rate.

---

## Open questions

1. **What's the trap mechanism?** The 60 confirmed contracts are not byte-identical, but they're from the same author. Decompile a few of the highest-victim ones (`0xedfc1a1f`, `0x76c32266`, `0x9c5ef32f`) and characterize the actual fail mode — what calldata triggers the revert?

2. **Why is this profitable?** If the operator is running both sides (one-operator hypothesis), what's being extracted? Possibilities:
   - On-chain "evidence laundering" — bait scanners into over-scoring the contracts so they get blacklisted from competitor MEV bots, removing competition for the operator's real trades
   - Reputation-building for the trap fleet — make it look battle-tested so other actors trust its surface
   - Fee accumulation from trapped txs (gas refunds, MEV searcher fees)

3. **Did the funder `0x7c8b9874f7...` fund any other deployers?** If yes, those deployers are siblings of the coffee fleet and worth investigating.

4. **Is there an L1 / cross-chain analogue?** Coffee-fleet vanities are easy to grind on any EVM chain; check arbitrum + optimism for `0xc0ffee*` deployers.

5. **Decode the trap selectors.** The fleet's contracts likely use the same `0xc0ffee_` style selectors that are in the bot side's `bot_candidate_selectors` table — cross-reference.

---

## Detection flags

- Add `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e` to permanent watchlist as confirmed trap operator (entity_type = `trap_operator`).
- Add the funder `0x7c8b9874f7be10ba196d3bb6fe1f45556c0bc1b5` to funder watchlist.
- All 209 contracts under this deployer should be scored at minimum `suspected`. The 60 confirmed should remain confirmed. The 142 currently suspected can stay; the 7 unknown should be promoted to suspected based on deployer attribution alone.

## Linked entities

- `0xe29a2cbd0c5a300d` — separate dormant fleet (size 11) observed waking 6 contracts in the same evening (2026-04-07 20:46:13). Not yet investigated whether this is a sibling.
- 9 coordinated first-callers from 2026-04-04 14:06 wake event listed above.
