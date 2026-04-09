# Case File — Permit2 Drainer Operation (4 Facilitators)

**Status:** Active, confirmed via on-chain evidence.
**Opened:** 2026-04-09
**Discovery path:** X402Monitor Phase 3/4 follow-up investigation of two Permit2 transferFrom events that didn't fire X402_AGENT_DRAIN because the victims weren't in the pre-existing exposure set.

---

## Summary

Four high-volume EOAs are using `Permit2.transferFrom` to drain stablecoin balances from victim wallets. Victims have granted **unlimited, never-expiring** Permit2 allowances (`MAX_UINT160` amount, `MAX_UINT48` expiration) to these drainers and been swept. Currently ~$3.9M in stablecoin inflows observed across 1,955 distinct sender addresses in recent 1000-tx windows per drainer, with individual victim losses up to $256k confirmed.

None of the 4 drainers appear in the public `facilitators.x402.watch` registry. They are misusing the x402/Permit2 settlement infrastructure as a drain vector.

---

## The 4 rogue facilitators

| Nickname | Address | Chain | Eth balance | Nonce | Distinct senders (recent 1000) | USD stablecoin inflow |
|---|---|---|---|---|---|---|
| **DRAINER-A7B9** | `0xa7b9874d15742358fb455dd56f97c6d19ad74f5c` | base | **272.29 ETH** | 96,144 | 458 | $229,059 |
| **DRAINER-E3B2** | `0xe3b205da6d47989538f03553bc394d941677ffd3` | base | ? | ? | 663 | $445,115 |
| **DRAINER-E717** | `0xe7176831c898d585cd999bcee9984a7fa9a6be96` | arbitrum | **125.32 ETH** | 80,141 | 512 | $664,177 |
| **DRAINER-CE5E** | `0xce5ec7336f863931fda2ee3e4b9dad99fcc53c91` | arbitrum | 0.94 ETH | 3,052 | 322 | $2,547,480 |
| **TOTAL** | | | | | **1,955** | **$3,885,831** |

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
