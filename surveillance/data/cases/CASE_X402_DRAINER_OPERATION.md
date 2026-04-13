# Case File — Permit2 Drainer Operation (5 Facilitators)

**Status:** Active, confirmed via on-chain evidence.
**Opened:** 2026-04-09
**Discovery path:** X402Monitor Phase 3/4 follow-up investigation of two Permit2 transferFrom events that didn't fire X402_AGENT_DRAIN because the victims weren't in the pre-existing exposure set.

---

## Summary

Five high-volume EOAs are using `Permit2.transferFrom` to drain token balances from victim wallets across **Arbitrum + Base + Optimism**. Victims have granted **unlimited, never-expiring** Permit2 allowances (`MAX_UINT160` amount, `MAX_UINT48` expiration) to these drainers and been swept. Currently ~$3.9M in stablecoin inflows observed across 1,955 distinct sender addresses in recent 1000-tx windows per the original 4 drainers, with individual victim losses up to $256k confirmed. A 5th facilitator (D270) was identified on 2026-04-12 draining OP tokens on Optimism, expanding the operation beyond stablecoins.

None of the 5 drainers appear in the public `facilitators.x402.watch` registry. They are misusing the x402/Permit2 settlement infrastructure as a drain vector.

---

## The 4 rogue facilitators

| Nickname | Address | Chain | Eth balance | Nonce | Distinct senders (recent 1000) | USD stablecoin inflow |
|---|---|---|---|---|---|---|
| **DRAINER-A7B9** | `0xa7b9874d15742358fb455dd56f97c6d19ad74f5c` | base | **272.29 ETH** | 96,144 | 458 | $229,059 |
| **DRAINER-E3B2** | `0xe3b205da6d47989538f03553bc394d941677ffd3` | base | ? | ? | 663 | $445,115 |
| **DRAINER-E717** | `0xe7176831c898d585cd999bcee9984a7fa9a6be96` | arbitrum | **125.32 ETH** | 80,141 | 512 | $664,177 |
| **DRAINER-CE5E** | `0xce5ec7336f863931fda2ee3e4b9dad99fcc53c91` | arbitrum | 0.94 ETH | 3,052 | 322 | $2,547,480 |
| **DRAINER-D270** | `0xd27047fe310178316b3acc4746e2a30823bb9186` | optimism | ? | 49,006 | ? | ? (OP tokens) |
| **TOTAL (original 4)** | | | | | **1,955** | **$3,885,831** |

Note: the $2.5M to DRAINER-CE5E includes at least one legitimate counterparty (`0xee7ae85f2fe2239e27d9c1e23fffe168d63b4055`, a 175M USDC contract, no Permit2 allowance to CE5E, sent $73k via regular transfers). The real drain portion is lower but still multi-million.

A7B9 and E717 have nonces of 96k and 80k — hundreds of thousands of outbound transactions each. They're industrial-scale automation. CE5E's lower nonce (3k) but much higher drain total ($2.5M) suggests it's a newer and more aggressive operator.

---

## Confirmed drains captured by the live monitor (2026-04-09)

All 6 events are direct calls to `Permit2.transferFrom(from, to, amount, token)` where `tx.from == to` (facilitator submits and receives). All succeeded on-chain. All payers now show zero balance in the drained token.

| When (UTC) | Chain | Drainer | Victim | Token | Amount |
|---|---|---|---|---|---|
| 2026-04-09 00:58:14 | arbitrum | CE5E | `0xd5aa6ac3cf0dc921cc0123bb82dad7ad2acf1f10` | USDC | **9,700.00** |
| 2026-04-09 01:02:35 | base | E3B2 | `0x4ee1f9649cf4c7067ffb76138a6f42f889448e08` | USDC | **2,049.80** |
| 2026-04-09 01:52:14 | arbitrum | E717 | `0x3016dd83a68a71403d94be1e7f872902392a69f7` | USDT | **2,067.73** |
| 2026-04-09 01:58:14 | arbitrum | CE5E | `0xe9ff6c27942882c16048dcd559cc26526238e1a8` | USDC | **10,050.00** |
| 2026-04-09 01:58:15 | base | A7B9 | `0xd28e416cdc3c81027c4881bb9f9402c0558aa38b` | USDC | **748.34** |
| 2026-04-09 02:10:15 | base | A7B9 | `0x0bb06f52737d...` | USDC | **585.36** |
| **Captured total** | | | | | **$25,201** |

---

## The drain fingerprint (forensically confirmed)

Verified on all spot-checked victims:

```
PAYER PROFILE AFTER DRAIN:
  balance_eth:        ~0 (just enough to have sent the approve tx)
  token_balance:      0 (fully drained)
  nonce:              1 to ~500 (often 1 = near-new wallet)

PERMIT2 ALLOWANCE STATE (still active on-chain):
  Permit2.allowance(payer, token, drainer) returns:
    amount      = 1461501637330902918203684832716283019655932542975
                = 2^160 - 1
                = MAX_UINT160 == UNLIMITED
    expiration  = 281474976710655
                = 2^48 - 1
                = MAX_UINT48 == NEVER EXPIRES
    nonce       = 1 (indicates one transferFrom consumed so far)
```

**The allowance is STILL ACTIVE on-chain after the drain.** Any new funds arriving in the victim's wallet will be drained again the moment the drainer sees them. The victims are effectively dead-ended: they cannot safely receive stablecoins to these addresses again without first revoking the Permit2 allowance, which none of them have done (nonce remains 1).

---

## Spot-checked high-value victims

| Victim | Drainer | Amount drained | Nonce | Current token balance | Allowance |
|---|---|---|---|---|---|
| `0x785ce546ed429559b95895cb4a07874bf8ed329c` | E3B2 | **$256,321** | 318 | 0 USDC | UNLIMITED, NEVER |
| `0x303d5773082a740c3040d5763b3d86f84478980f` | E717 | **$179,999** | **1** | 0 USDC | UNLIMITED, NEVER |
| `0x59f13bc19a82e9e67703d865eb96a45692760cd5` | A7B9 | $29,059 | 538 | 0 USDC | UNLIMITED, NEVER |
| `0x303d5773` is the clearest drain example: a wallet that has sent exactly 1 transaction in its lifetime, lost $180k USDC to E717, and still has the drain allowance active. | | | | | |

---

## How the operation differs from real x402

| Attribute | Real x402 facilitator (Coinbase/PayAI/etc.) | This drainer operation |
|---|---|---|
| In `facilitators.x402.watch` | Yes | **No** |
| Average payment size | $0.001 - $0.01 (micropayments) | $100 - $256,000 |
| Payer approval pattern | Short-lived, per-tx sigDeadline | **Unlimited, never-expiring** |
| Post-payment balance | Payer still holds funds | **Payer balance = 0** |
| Number of transfers per payer | Many (ongoing API use) | **One (full sweep)** |
| Nonce pattern on payers | Active wallets | **Often nonce 1 (single-use)** |
| Purpose | API access / resource gating | Wallet drain |

The drainers are using `Permit2.transferFrom` because it's the cleanest programmatic way to sweep approved allowances — the same reason x402 uses it. This is not a bug in x402, it's an approval-phishing pattern that happens to share the settlement selector with x402. Our x402 monitor catches it because the on-chain call is structurally identical to a real x402 settlement until you look at the amount and post-state.

---

## Actions taken

**Production state updates (2026-04-09):**
1. All 4 facilitators reclassified from `unknown` to **`rogue`** in `x402_facilitators`, with forensic source attribution
2. 6 `X402_AGENT_DRAIN` alerts inserted for the monitor-captured events (previously they were only `X402_FACILITATOR_UNKNOWN`, one tier too low)

**Code / monitor updates:** The current `_handle_x402_tx` logic fires `X402_AGENT_DRAIN` only when the payer is in `x402_permit2_exposure` (which is scoped to trap-token approvals). These 6 drains did not fire the drain alert because the victims had approved Permit2 on canonical USDC/USDT, not on trap tokens. The current exposure-tracking definition is too narrow.

**Recommended next-step code change (not shipped yet):** Expand `X402_AGENT_DRAIN` triggering to include any `Permit2.transferFrom` where:
- `tx.from == decoded.to` (facilitator is also the recipient — self-settlement), AND
- amount >= threshold (e.g., 100 stablecoin units), AND
- the facilitator is classified `rogue` OR the payer's current token balance is 0 (sweep pattern)

---

## Outflow trace (2026-04-09, second session)

Traced the top 1000 outbound ERC-20 and external transfers from each
of the 4 drainers via Alchemy asset transfers. Matched destinations
against `fund_tracer.py` registries (CEX hot wallets, bridges, mixers,
DEX routers, ORG_WALLETS).

### Registry match: zero hits

**No outflows to any labeled CEX, bridge, mixer, DEX router, or known
org wallet.** All 693-896 distinct destinations per drainer are
unlabeled addresses. The cashout path does not use any of:
- Top 10 CEX hot wallets (Binance, Coinbase, OKX, Kraken, Bybit)
- Canonical bridges (ArbSys, L2StandardBridge, Stargate)
- Tornado Cash / Railgun
- Uniswap / Sushi / Aerodrome / Camelot / Paraswap / 1inch / 0x
- Existing tracked ORG_WALLETS from org_001/org_002/org_003

This is a **fifth independent operation** — it does not connect to any
previously-catalogued Layer 3 entity.

### Current balances are huge and sitting idle

| Location | Chain | Token | Current balance |
|---|---|---|---|
| **CE5E vanity sink #1** `0xbec87a77...` | arbitrum | USDC | **$3,630,347** |
| A7B9 drainer | base | USDC | $333,266 |
| E717 drainer | arbitrum | USDT | $226,269 |
| E717 drainer | arbitrum | USDC | $115,047 |
| E3B2 drainer | base | USDC | $95,071 |
| CE5E drainer | arbitrum | USDT | $86,204 |
| CE5E drainer | arbitrum | USDC | $27,231 |
| **TOTAL** | | | **~$4,513,437** |

**~$4.5M is currently held on-chain** across these 6 addresses. The
operation is actively accumulating but has not yet cashed out. This
makes the funds **recoverable** if frozen by a compliance-cooperating
exchange OR if the operator is identified before the sweep.

The CE5E vanity sink #1 alone holds **80% of the current balance**.
Freezing that one address protects $3.63M.

### Vanity sink pattern

CE5E concentrates its outflows into a small set of vanity-generated
sinks sharing a distinctive address pattern:

- `0xbec87a77b19797bbe9b920ec521f3716c3725d22` — $4M received, 7 outbound txs, $3.63M held
- `0xbec8721e796b0ce7705d317a73f110693d895d22` — $2.5M received, nonce 621, currently $0 (cycles back to CE5E)

Both start with `0xbec8` and end with `d22` — programmatically
grind-generated vanity addresses. CE5E is a **consolidation operator**:
drain proceeds funnel through a small set of known sinks instead of
the fan-out pattern the other 3 drainers use.

The other 3 drainers (A7B9, E3B2, E717) fan out across 306-896
distinct destinations per drainer, suggesting per-victim forwarding
or sub-wallet rotation rather than concentrated consolidation.

### Cross-drainer destination overlap (ambiguous signal)

34 destinations were hit by 2+ drainers across chains. Most shared
pairs are A7B9 (base) + E717 (arbitrum) — same hex address on two
different chains.

Hop-2 analysis on the top shared destination `0x621db24cdb8c1a2ba1e7b5703d203e9ce0a7abb3`:
- On base: 23-byte contract, nonce 3794, 0.20 ETH — active proxy contract
- On arbitrum: EOA, nonce 3 — barely-used wallet

The same hex address being a contract on one chain and an EOA on the
other could be either (a) an operator CREATE2-deployed a proxy on
base with the same salt as a disposable EOA they also control on arb,
or (b) coincidence. The 34-shared-address count makes coincidence
statistically unlikely but doesn't prove single-operator cross-chain.

### Secondary vanity patterns discovered

Hop-2 trace on hop-1 destinations revealed additional vanity families:
- `0xe888...` prefix on CE5E sink #1's outflows ($243k chunks)
- `0x676d...` prefix on CE5E sink #1's outflows ($120k chunks)
- `0xa219...` prefix on E717 USDT destinations
- `0xda29...` prefix on E717 USDT destinations

The vanity prefixes are consistent within each drainer's outflow
topology. Each drainer uses its own vanity family. This is strong
evidence of automated address-generation infrastructure — not hand-
crafted wallets.

### What hasn't happened yet

- No CEX deposits detected (either the operator is avoiding top-10
  exchanges or using smaller/regional venues not in our registry)
- No bridge crossings detected (operator is keeping funds on L2)
- No mixer usage detected
- The $3.63M in CE5E vanity sink #1 has not moved since accumulation

The absence of a cashout path in 1000+ outbound transfers per drainer
is itself a meaningful finding. Either the cashout path is intentionally
slow/patient, or it uses addresses outside any registry I have. OTC
settlement or un-listed CEX deposit addresses are the remaining
candidates.

---

## Activity update — 2026-04-12

**New facilitator discovered:** DRAINER-D270 (`0xd27047fe310178316b3acc4746e2a30823bb9186`) on **Optimism**. Nonce 49,006. This is the 5th rogue facilitator and the first on Optimism, expanding chain coverage from Arbitrum + Base to **Arbitrum + Base + Optimism**.

**Non-stablecoin expansion:** D270 is draining **OP tokens** via Permit2, not stablecoins. This is the first confirmed non-stablecoin drain in the operation. The drain vector is token-agnostic — any ERC-20 with a Permit2 allowance is vulnerable.

**April 12 activity:** ~$75K drained across 15 victims. Drain pace accelerating to ~1.8 victims/hour.

**Token decimals normalization bug:** The initial D270 alert reported a $3.1 quadrillion drain amount. This was a decimals normalization error — the alert pipeline assumed 6 decimals (USDC standard) but OP has 18 decimals. Real amount was ~3,100 OP (~$4,650-$6,200). See CORRECTIONS.md entry dated 2026-04-12.

**Actions taken:**
1. D270 classified as `rogue` in `x402_facilitators` table (surveillance.db)
2. D270 flagged on production watchlist via `/admin/flag-address` with `priority: CRITICAL` and `entity_type: x402_rogue_facilitator`
3. Case file updated to reflect 5 facilitators across 3 chains

---

## Open questions

1. **Who operates these wallets?** A7B9 + E717 have industrial nonces (80k-96k) and ETH balances (272 + 125 ETH). This is not a hobbyist. Either a single well-funded drain operator running multiple chains, or multiple independent drain operators converging on the same tactic.
2. **How do victims end up with unlimited never-expiring Permit2 allowances?** Most likely path: a phishing site prompts the user to "sign a transaction" which is actually an `approve(Permit2, MAX_UINT256)` call. After that, the drainer has forever to sweep any USDC the victim ever holds. Phishing origin unknown.
3. **What's the full historical drain total?** Our `alchemy_getAssetTransfers` queries are capped at 1000 events per call. The real lifetime inflow to these facilitators is larger. A7B9 with nonce 96k has been active for ~days/weeks — if its rate is ~1000 drains per day, full impact is likely $10M+.
4. **Are the 23 remaining unknown facilitators on production also drainers?** Probably some are. The top 5 off-registry EOAs in Phase 4 (nonces 519k-525k) are a separate cluster — much higher nonce than A7B9/E717 — probably a different service/operator. They haven't been forensically verified as drainers yet.
5. **Coordination with victim addresses?** The 1,955 distinct sender pool across 4 facilitators is large. Spot-checking 4 confirmed drain victims, 4 more with identical fingerprints probably exist for every ~10 senders. A mass-notification or allowance-revocation coordination channel would be valuable but is out of Layer 3 scope.

---

## Linked material

- CORRECTIONS.md — entry dated 2026-04-09 reclassifying the initial "Permit2 self-settlement" finding as confirmed drain
- `surveillance/x402_monitor.py` — X402Monitor class, Phase 3 live detection
- Investigation trail:
  - Phase 3 spec mentioned `X402_AGENT_DRAIN` as a planned alert
  - Phase 4 final report flagged "2 Permit2 transferFrom events where facilitator is also payee" as suspicious but inconclusive
  - This investigation confirmed the pattern and quantified the scale
