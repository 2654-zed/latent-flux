# Case File — Coffee Fleet `0xc0ffeefeed8b9d27`

**Status:** Active. Confirmed dual-role (trap deployer + self-scanning bot fleet).
**Opened:** 2026-04-07
**Last updated:** 2026-04-08 (rewritten from production data after local-sync correction)
**Chain:** base only

> **Note on prior version:** An earlier draft of this file reported 209 contracts / 60 confirmed / 366 trap hits / 84 unique victim bots. Those numbers came from the local SQLite file, which is a stale-sync superset of Railway production (rows from deleted/reset earlier Railway states were never removed locally). This rewrite uses data pulled directly from the Railway production DB on 2026-04-08. See CORRECTIONS.md entry `2026-04-08 Correction-of-the-Correction`.

---

## Identity

**Deployer:** `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e`
**Funder:** `0x7c8b9874f7be10ba196d3bb6fe1f45556c0bc1b5`
**Original funding tx:** `0xc99fe50ee735019fefd48c6886ceffcf9116bd419f19b5c71e4f5a1a0505a78a`
**Original funding:** 2024-09-17 20:33:29 UTC, 0.1 ETH on base, block 19907931
**Deployer first_seen:** 2026-03-30 00:04:37 UTC (block 44019865)
**Pre-deployment dormancy:** ~18.5 months

---

## Scale (production, as of 2026-04-08)

| Metric | Value |
|---|---|
| Contracts deployed | **60** |
| Confirmed (on-chain victim evidence) | **20** |
| Suspected | 39 |
| Unknown | 1 |
| First deploy | block 44019865 / 2026-03-30 00:04:37 UTC |
| Latest deploy | block 44434493 / 2026-04-08 14:25:33 UTC |
| Active window | 9 days, still deploying |
| Chain | base only (not multi-chain) |

## Victim record (production)

| Metric | Value |
|---|---|
| Total trap_events | **20** |
| Unique victim bots | **15** |
| First hit | 2026-04-06 21:43:35 UTC |
| Last hit | 2026-04-08 02:40:41 UTC |

**All 15 victim bots have `0xc0ffee` vanity prefixes.** 100% coffee-fleet-on-coffee-fleet. Zero non-coffee-fleet victims.

This is the central finding and it is unchanged from the earlier draft. Production data confirms it directly: the coffee-fleet deployer is exclusively trapping coffee-fleet vanity bots, and no other bot class has ever been observed tripping one of these contracts.

Two possible interpretations remain:

1. **Single operator running both sides.** One actor controls both the trap-deployer and the `0xc0ffee` scanner fleet. The "victims" are part of the same operation — likely scanner self-tests, failure-mode probing, or a deliberate on-chain confirmation theater aimed at third-party risk scorers.

2. **Closed coffee-fleet adversary ecosystem.** Multiple independent `0xc0ffee`-vanity operators have converged on a self-contained scan-and-trap loop nobody outside the cluster participates in.

The data cannot distinguish between these without wallet-clustering work on the victim bots. The absence of any non-vanity victim across two days of observation makes the single-operator hypothesis more economical, but neither is confirmed.

---

## Confirmed contracts (20)

Each confirmation is behavioral — a specific `0xc0ffee` vanity bot tripped the contract exactly once. Every confirmation row has the pattern `Behavioral confirmation: bot 0xc0ffeeXXXX... trapped`.

| Contract | Block | Trapping bot |
|---|---|---|
| `0x97bc14ebfa5f79da142cec92dec6b46b61ef507b` | 44080574 | `0xc0ffeefc06e7d4abc67b8fdba7ef9100b0c85f5b` |
| `0x26be94fc9217d7416502c17fa96b86fd851ea86b` | 44081623 | `0xc0ffee31b1c8c3427a9ed9373bef867fc895d966` |
| `0xfde8f37cbd35bca68af32d9fd9190bf29d03caaf` | 44088595 | `0xc0ffee2af556f31a749146d79bbd2135ceca0e56` |
| `0x8ec0fde7693c4e2ab80d25d8c9ffbb27e00fa164` | 44091692 | `0xc0ffee818c9cd2b6e75754e94e0aad291c4db95e` |
| `0x799cb09d6a3918c8dbe25cd032319643cdc78f38` | 44092316 | `0xc0ffee445a22e6228cfb77ec0483c426fc856161` |
| `0x59e304b18785de49ba58305a715a282d900f65bd` | 44093225 | `0xc0ffee8cae1d4279e42fba7a6cafeb8e1401140f` |
| `0xb3c1481d10125eecd86824cd890d5cf5d9be4fb3` | 44094331 | `0xc0ffee43451db6e0fdd135ed5b06492b35c34e5e` |
| `0xfa757c220736345d52e46344ab443eb1b66f6d0d` | 44107324 | `0xc0ffee445a22e6228cfb77ec0483c426fc856161` |
| `0xd737156b5ee229102f59598cb13eebbd30054dbb` | 44124345 | `0xc0ffee5f66a5546b23f5f430dca69df71e32ebf7` |
| `0x6fec392ee427407586630b416017a3748467d115` | 44126294 | `0xc0ffee17d520056942531cb6d4d6251ff8163bb1` |
| `0xff9c23d9f208813a7857087a024c7a9ed85f522a` | 44134086 | `0xc0ffee26e721b8e0ffeac3176166aad4e599e41d` |
| `0x9b4822d79ebfb7d61e25ee4c131968532be966c0` | 44134926 | `0xc0ffee884426f7c2e6d61b8b6ae8927c4c40ff5e` |
| `0x3076ad28fbc481f843757ad644e5d2aba7a04cae` | 44209456 | `0xc0ffee59f94f54f4f293f01672976408bc1cad7f` |
| `0xc0f2c4ca38a8bd863684d0d4cccd87c11363a23f` | 44293707 | `0xc0ffee884426f7c2e6d61b8b6ae8927c4c40ff5e` |
| `0xb717c140361e500ee5f374e6d46c81b0424b9dc1` | 44298746 | `0xc0ffee1864731c3c33a1967fc8e0fbf454a6a006` |
| `0x5dd5b15323db9168495d9088de90128a33a1e31e` | 44319406 | `0xc0ffee2af556f31a749146d79bbd2135ceca0e56` |
| `0x5dd183d1b0e8bcb9700f2d3790b23b4fd33f3000` | 44320372 | `0xc0ffee818c9cd2b6e75754e94e0aad291c4db95e` |
| `0xcf9dda0e77276ac5a4df307ee6b55b20230e69bd` | 44332793 | `0xc0ffee648f2b70238b827cdd9c3f2c91170ee3b1` |
| `0xb25351a577d9c5d0c1ca83cb4ff77f0eeb656352` | 44345612 | `0xc0ffee2af556f31a749146d79bbd2135ceca0e56` |
| `0x95cdb520c817543d0923cc4eaa7e1a9e6887067a` | 44349036 | `0xc0ffeef2a20f16d8ef3f88206dff938915be44d5` |

**Every confirmed contract has exactly 1 hit and 1 bot.** No contract has been tripped more than once. Two interpretations:
- The scanner fleet rotates fresh probe addresses and each fresh address hits exactly one trap per sweep
- The deployer rotates contracts fast enough that by the time a bot encounters the same trap twice, it's already behind the fleet's leading edge

Several bots appear on multiple contracts (e.g. `0xc0ffee2af556f31a749146d79bbd2135ceca0e56` trapped 3 times; `0xc0ffee818c9cd2b6e75754e94e0aad291c4db95e`, `0xc0ffee884426f7c2e6d61b8b6ae8927c4c40ff5e`, and `0xc0ffee445a22e6228cfb77ec0483c426fc856161` each trapped twice), so the rotation explanation fits bots too — they aren't one-time-use, they just don't remember which traps they've already tripped.

---

## Dormant activation timeline (production)

20 activation events recorded in `dormant_activations`, all on 2026-04-06 through 2026-04-08. Progressive wake from 0 → 20 active contracts inside the fleet:

| Timestamp (UTC) | Fleet | Active before → after | First caller |
|---|---|---|---|
| 2026-04-06 21:55:11 | 50 | 0 → 2 | `0xc0ffeecdffd624...` |
| 2026-04-06 21:55:11 | 50 | 0 → 2 | `0xc0ffee648f2b70...` |
| 2026-04-06 22:10:53 | 50 | 2 → 3 | `0xc0ffeec544131a...` |
| 2026-04-06 22:25:13 | 50 | 3 → 6 | `0xc0ffee17d52005...` |
| 2026-04-06 22:25:13 | 50 | 3 → 6 | `0xc0ffee1864731c...` |
| 2026-04-06 22:25:13 | 50 | 3 → 6 | `0xc0ffee5f66a554...` |
| 2026-04-06 22:40:55 | 50 | 6 → 7 | `0xc0ffeea3f806b3...` |
| 2026-04-06 22:55:15 | 50 | 7 → 8 | `0xc0ffeebd01000e...` |
| 2026-04-06 23:22:28 | 50 | 8 → 9 | `0xc0ffee2af556f3...` |
| 2026-04-06 23:52:30 | 50 | 9 → 10 | `0xc0ffee8cae1d42...` |
| 2026-04-07 00:44:45 | 50 | 10 → 11 | `0xc0ffee884426f7...` |
| 2026-04-07 01:14:47 | 51 | 11 → 12 | `0xc0ffee445a22e6...` |
| 2026-04-07 02:30:35 | 51 | 12 → 13 | `0xc0ffee884426f7...` |
| 2026-04-07 02:44:53 | 51 | 13 → 14 | `0xc0ffeef6724ba7...` |
| 2026-04-07 09:15:19 | 53 | 14 → 15 | `0xc0ffee384d912e...` |
| 2026-04-07 11:15:27 | 53 | 15 → 16 | `0xc0ffeefc06e7d4...` |
| 2026-04-07 17:45:58 | 54 | 16 → 17 | `0xc0ffee7bb4d2ad...` |
| 2026-04-07 20:46:13 | 55 | 17 → 18 | `0xc0ffee43451db6...` |
| 2026-04-07 23:44:34 | 56 | 18 → 19 | `0xc0ffee818c9cd2...` |
| 2026-04-08 02:45:58 | 56 | 19 → 20 | `0xc0ffee467ef760...` |

**Roughly one activation every 1-3 hours, 20 events in 29 hours.** The fleet size itself grew from 50 to 56 during the activation window — the deployer is both deploying new contracts and waking old ones in parallel. As of the last event the fleet is still only 20/56 active, meaning the operation is in an early phase of rollout.

Every first-caller is a `0xc0ffee` vanity. 19 distinct vanity addresses across the 20 activations (one repeat — `0xc0ffee884426f7` appeared twice 2 hours apart).

---

## Bytecode profile

Not re-sampled against production on this rewrite — the earlier sample of 30 suspected contracts on local showed 30 distinct sha256 hashes with sizes ranging 1,609 → 10,199 bytes, which is consistent with same-author-different-compilation pattern. Production has fewer contracts so a future bytecode sweep is cheaper and should be done before closing this case.

Noted from confirmation reasons: the trap mechanism varies contract-to-contract — reasons include `selfdestruct`, `delegatecall_in_token`, and pure bytecode pattern matches. Multiple trap primitives from the same author.

---

## Why the scale is smaller than it looked yesterday

The earlier draft reported 209 contracts / 366 trap_events / 84 victim bots, all of which were local-sync artifacts:

- **Local had 209 contracts** for this deployer because `sync_railway_db.py` uses `INSERT OR REPLACE`/`INSERT OR IGNORE` and never deletes rows that exist locally but no longer exist on Railway. At least one Railway reset (during the 2026-04-05 multiprocessing/spawn debugging) truncated historical data on production, leaving the local sync cache as the only place those rows survived.

- **Local had 366 trap hits** for the same reason — the `trap_events` table is bigger on local than production because historical hits weren't re-recorded after a reset.

- **Production `trap_events` only goes back to 2026-04-06** for this deployer. Anything earlier is gone on production. The 2-day victim window is all the production data we have.

This means the production data is an **undercount** of the real activity. The actual fleet has almost certainly trapped more bots than the 15 we can see — but we can't enumerate them from Railway anymore. The local DB is the only record, but the local DB has data-provenance issues that make it unreliable for case-file claims.

**Working rule going forward:** when writing case files, cite production numbers only, and flag the known undercount explicitly rather than blending local and remote sources.

---

## Open questions

1. **What's the trap mechanism?** The 20 confirmed contracts are not byte-identical. Decompile the top few (`0x97bc14eb`, `0x26be94fc`, `0xfde8f37c`) and characterize the actual fail mode — what calldata triggers the revert, and what happens to the caller's assets?

2. **Why only 1 hit per contract?** Is the mechanism single-use (one-time trap that rearms), or is the scanner fleet just moving fast enough that each bot catches fresh traps on every sweep?

3. **Is this profitable?** If the operator runs both sides, where does value flow? Candidate extraction paths:
   - Gas refund farming (Arbitrum-style)
   - MEV searcher fee capture on the failed txs
   - Reputation manipulation (force competing scorers to rate the contracts as dangerous so third-party MEV bots avoid them, clearing the field)
   - None of the above — it's scanner QA infrastructure for a legitimate defensive operation

4. **Did the funder `0x7c8b9874f7...` fund any other deployers?** If yes, those deployers are siblings of the coffee fleet and worth a direct funder-graph query.

5. **Does the fleet exist on other chains?** Production shows base-only, but coffee-fleet vanity addresses are trivially grindable on any EVM chain. Cross-check arbitrum + optimism for `0xc0ffee*` deployers with the same funder.

6. **Bytecode-family clustering.** Re-sample 20-30 contracts from production, compute sha256 prefixes, see whether any cluster at all or remain fully unique. Anti-fingerprint compilation is suggestive of same-author.

---

## Watchlist actions

- Add `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e` to permanent watchlist as confirmed trap operator; set `entity_type = 'trap_operator'` in deployers.
- Add funder `0x7c8b9874f7be10ba196d3bb6fe1f45556c0bc1b5` to funder watchlist.
- All 60 contracts under this deployer should be minimum-tier `suspected`. The 20 confirmed stay confirmed. The 1 unknown should promote to suspected based on deployer attribution alone.
- Snapshot the 19 distinct `0xc0ffee*` caller addresses from the activation log and mark them as coffee-fleet-cluster for future wallet-clustering analysis.

---

## Related

- Local case file draft (now corrected) — see CORRECTIONS.md `2026-04-08 Correction-of-the-Correction`
- Potentially related: `0xe29a2cbd0c5a300d` — separate dormant fleet (size 11 on local) observed waking 6 contracts in the same 2026-04-07 20:46 window. Not confirmed on production; requires re-query.
- Attack 2 in POTENTIAL_ATTACKS.md (Dormant Fleet Activation + Proxy Upgrade Swap) is the closest template for the activation pattern observed here, minus the proxy-upgrade twist.
