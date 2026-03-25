# CASE FILE: Dragon — 0x2e20b26172a8625c33097288075920a6210a8233
**Classification:** TRAP INVENTORY OPERATOR — PRE-STAGED
**Generated:** 2026-03-25
**Chain:** Base

---

## Executive Summary
A single deployer created **2,077 token contracts** on Base at a rate of one every 15 seconds, then pre-approved **169** of them for unlimited spending by the PancakeSwap V3 Router (`0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24`). The entire inventory is dormant — zero external victim interactions. The operator has not called `addLiquidity()` on any contract. This is the largest pre-staged trap inventory in the corpus, representing a potential mass activation event.

---

## Contract Identity

| Field | Value |
|---|---|
| Deployer | `0x2e20b26172a8625c33097288075920a6210a8233` |
| Chain | Base |
| Total contracts | 2,077 |
| Self-approved to PancakeSwap | 169 |
| External interactions | 15 contracts probed (bot scans, not activation) |
| Deployer balance | ~0 ETH |
| Entity type | `trap_inventory_operator` |
| First deployment | 2026-03-18T22:24:11 UTC |
| Last deployment | 2026-03-19T21:59:09 UTC |
| Deployment rate | ~1 contract every 15 seconds |

## The Inventory

**2,077 contracts deployed in ~24 hours.** All on Base. All ERC-20 tokens. All suspected via deployer_history (velocity detection triggered at 9+ per session).

**169 pre-approved.** The deployer called `approve()` (selector `095ea7b3`) on 169 contracts, granting MAX_UINT256 spending allowance to PancakeSwap V3 Router. This means PancakeSwap can pull tokens from these contracts without further authorization.

**Approval is ongoing.** Transaction events show the deployer continued approving contracts after initial deployment — the 169 count has been growing.

**Spender:** `0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24` — PancakeSwap V3 Router on Base (confirmed legitimate contract).

## The Activation Mechanism

The trap activation requires one action: the deployer calls `addLiquidity()` on PancakeSwap for any pre-approved contract.

```
Current state (DORMANT):
  2,077 contracts deployed
  169 approved to PancakeSwap
  0 liquidity pools created
  0 victims

Activation (ONE TRANSACTION):
  Deployer calls addLiquidity() on PancakeSwap
  -> Creates a trading pair
  -> Bots discover new pair within blocks
  -> Trading begins
  -> Trap mechanism (if present in bytecode) activates
  -> Deployer pulls liquidity = rug pull
```

The pre-approval means the deployer doesn't need to approve each contract individually at activation time — they can create pools for all 169 in rapid succession.

## External Activity

15 of 2,077 contracts have received external interactions:
- **`0xb59c359f8daf13d6...`** — an external scanner hit 14 dragon contracts with selector `f3294c13` (unknown function). This address hits 14 total contracts in the DB — it's scanning, not the operator.
- **The deployer itself** called `approve()` on 169+ contracts (the PancakeSwap pre-approval).
- **`0xa2d348a328090f3d...`** — called `approve()` on 3 contracts. Possible secondary operator wallet.

**No addLiquidity selectors detected.** No `e8e33700`, `f305d719`, or `6a627842` calls. The dragon has not activated.

## Monitoring Status

| Camera | Catches Activation? |
|---|---|
| Liquidity event monitor | YES — addLiquidity on PancakeSwap Router triggers CRITICAL alert |
| Pair creation monitor | YES — PairCreated event from factory triggers CRITICAL alert |
| Deployment monitor | Already captured all 2,077 contracts |
| Daily report | Tracked in Active Watchlist |

## Risk Assessment

| Factor | Rating |
|---|---|
| Inventory size | CRITICAL (2,077 contracts — largest in corpus) |
| Pre-staging | HIGH (169 pre-approved, ready to activate) |
| Current activity | DORMANT (no liquidity, no victims) |
| Activation complexity | LOW (single addLiquidity call per contract) |
| Deployer balance | EMPTY (needs refunding to pay gas for activation) |

**Overall: HIGH (dormant but pre-positioned)**

The empty deployer balance is the only thing preventing activation. If the deployer receives ETH, activation could begin within minutes.

## Timeline

| Date | Event |
|---|---|
| 2026-03-18 22:24 | First contract deployed |
| 2026-03-19 21:59 | Last contract deployed (2,077 total) |
| 2026-03-19 - ongoing | Deployer continues calling approve() on contracts |
| 2026-03-20 09:58 | Deployer sent out 2.71 ETH (wallet emptied) |
| 2026-03-21 - 2026-03-25 | Dormant. External bot probes only. |
