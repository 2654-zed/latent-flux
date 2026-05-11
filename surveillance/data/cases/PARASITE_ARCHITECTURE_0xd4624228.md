# PARASITE ARCHITECTURE: 0xd4624228cce5baa0814c9e7f666a8a2c83b6f159

**Generated:** 2026-03-25
**Chain:** Base
**Bytecode Size:** 14,752 bytes
**Deployer:** `0xe8E0C4883D7196A7De87a6489F6Da58212dbE813`

---

> **Correction #17 addendum (added 2026-04-29):** The "infrastructure parasite" framing and the 14.2× trust amplification figure cited in this file were retired from active claim status by Correction #17 in `reports/correction_log.md` (applied 2026-04-25). Two reconciliation drivers: (1) the 14.2× multiplier was computed against the `T2-eaef6a5d` bytecode family baseline, which was itself dissolved as a NULL-bucket artifact by Correction #3 (2026-04-16) — the comparator was retroactively invalidated; (2) Opus 4.7's reading of the bytecode framed the asymmetric routing pattern as obfuscated fee-on-transfer logic with incidental router exposure rather than deliberate router exploitation. Both readings have empirical merit. The contract self-extinguished on 2026-03-26 and produces no `trust_amplification` row in the live producer (drops below the 50-caller minimum). The 2,910-victim count and ~97% router-delivered traffic are Tier A direct measurements and remain canonical. **This case file is preserved as historical analysis.** The framing tension is documented in the lexicon's [Trust Amplification Factor](../../docs/lexicon.md#trust-amplification-factor) entry and in `reports/correction_log.md` Correction #17.

---

## Executive Summary

This is not a simple fee-on-transfer token. It is a **multi-interface contract** that simultaneously impersonates an ERC-20 token, a Uniswap Universal Router, a Uniswap pool, and an NFT receiver. By implementing selectors from multiple Uniswap components in a single contract, it can intercept swap flows at whichever layer the routing algorithm discovers it. The asymmetric buy/sell logic is gated by 12 CALLER checks and a storage-keyed fee lookup table whose parameters can be changed by the owner post-deployment.

---

## Interface Impersonation

The contract implements **37 function selectors** spanning 5 distinct interfaces:

### ERC-20 Token Interface
| Selector | Function | Purpose |
|---|---|---|
| `0x095ea7b3` | `approve(address,uint256)` | Makes contract look like a token to frontends |
| `0x23b872dd` | `transferFrom(address,address,uint256)` | Enables token movement — **this is where the fee extraction lives** |
| `0x70a08231` | `balanceOf(address)` | Reports balances (can be manipulated) |

### Uniswap Router Interface
| Selector | Function | Purpose |
|---|---|---|
| `0x3593564c` | `execute(bytes,bytes[],uint256)` | **Intercepts Universal Router calls** — 98.7% of traffic uses this |
| `0x04e45aaf` | `exactInputSingle(...)` | Intercepts V3 Router direct swap calls |
| `0x0a1d7c5f` | `exactOutput(...)` | Intercepts reverse-direction V3 swaps |

### Uniswap Pool Interface
| Selector | Function | Purpose |
|---|---|---|
| `0x1698ee82` | `getPool(address,address,uint24)` | Impersonates a pool factory — **tells the router "I am a valid pool"** |
| `0x0c49ccbe` | `decreaseLiquidity(...)` | Accepts liquidity removal calls |

### NFT Position Interface
| Selector | Function | Purpose |
|---|---|---|
| `0x42842e0e` | `safeTransferFrom(address,address,uint256)` | Receives NFT position tokens |
| `0x150b7a02` | `onERC721Received(...)` | Confirms NFT receipt capability |

### Owner Interface (Ownable)
| Selector | Function | Purpose |
|---|---|---|
| `0x715018a6` | `renounceOwnership()` | Standard OpenZeppelin Ownable |
| `0x8da5cb5b` | `owner()` | Returns owner address (slot 0) |
| `0xf2fde38b` | `transferOwnership(address)` | Transfers control |

### Custom Functions (Unknown Signatures)
| Selector | Observed Usage |
|---|---|
| `0x35579f0c` | 13 calls from 13 callers |
| `0x514dd960` | 8 calls from 8 callers |
| `0x600502f6` | 32 calls from 32 callers |
| `0x9f9d8b43` | 8 calls from 8 callers |
| `0xb97b808f` | Unknown |
| `0xc2fed262` | 23 calls from 20 callers |
| `0xd798f86e` | 9 calls from 9 callers |
| `0x251c9422` | Unknown |

---

## Storage Layout

| Slot | Value | Meaning |
|---|---|---|
| 0 | `0xe8E0C4883D7196A7De87a6489F6Da58212dbE813` | Owner address (deployer) |
| 1 | Complex packed data | Likely token name/symbol or configuration bitmap |

The fee parameters are stored in **dynamic storage slots** computed via KECCAK256 hashing (not in fixed slots). This means the fee table is a Solidity `mapping` — the owner can modify fee parameters for specific addresses or conditions by writing to computed storage slots.

---

## The Fee Mechanism (Bytecode Level)

The fee extraction lives in the code around offset `0x3364`:

```
0x3364: SHA3          ← Compute storage slot from address/condition hash
0x3367: JUMPDEST
0x3368: SLOAD         ← Load fee parameter from computed slot
0x336b: JUMPDEST
0x3374: JUMPDEST
0x3375: DUP3
0x3376: LT            ← Compare against threshold
0x3377: ISZERO
0x337b: JUMPI         ← Branch: fee applies or doesn't
0x337f: PUSH1 0x01    ← Increment counter
0x3388: MUL           ← Multiply transfer amount by fee factor
0x3389: ADD           ← Add to accumulated fee
```

**How it works:**
1. `SHA3` computes a storage key from the transfer parameters (sender address, direction, etc.)
2. `SLOAD` reads the fee parameter for this specific transfer context from storage
3. `LT + ISZERO + JUMPI` decides whether to apply the fee (direction-dependent)
4. `MUL + ADD` calculates the fee amount and adjusts the transfer

The fee parameters are in storage, not hardcoded. The owner can change them at any time via `SSTORE`. This means:
- The contract could start with 0% fee (to pass initial audits/scans)
- The fee can be enabled later for specific addresses or directions
- Different callers can have different fee rates
- The fee can be increased or decreased without redeployment

---

## The Asymmetric Logic

The contract contains **12 CALLER opcode instances** — each one checks `msg.sender` to determine which code path to execute. The pattern:

```
CALLER           ← Get msg.sender
[DUP/PUSH]       ← Load comparison address (owner, LP pool, router)
EQ               ← Is caller == known address?
JUMPI            ← If yes, take alternate path
```

This creates **direction-dependent behavior:**

**Buy path** (user sends WETH, gets SXAI):
- CALLER is the Universal Router
- Router sends WETH to the LP pool
- Contract mints/transfers SXAI to user
- No fee applied — transaction completes normally

**Sell path** (user sends SXAI, should get WETH):
- CALLER is the user (or router on behalf of user)
- Contract forwards SXAI to LP pool
- LP pool sends WETH to **the contract** (not the user)
- Contract's CALLER check identifies this as a sell
- Fee mechanism activates: 100% of WETH retained
- User receives nothing

The buy path works because it builds trust. The sell path extracts because by the time the user tries to sell, they already hold SXAI tokens and believe the pair is legitimate.

---

## Opcode Profile

| Opcode | Count | Role in Attack |
|---|---|---|
| JUMPI | 164 | Heavy conditional branching — direction detection, caller gating |
| REVERT | 40 | Error paths (0.3% revert rate = most paths succeed) |
| SLOAD | 13 | Fee parameter reads + owner checks |
| CALLER | 12 | Direction detection: buy vs sell |
| CALL | 12 | External token transfers + pool interactions |
| MUL | 11 | Fee calculations |
| SSTORE | 10 | Balance updates + fee accumulation |
| SHA3 | 7 | Storage slot computation for fee mapping |
| STATICCALL | 5 | Read-only balance checks |

The 164 JUMPI instructions for a 14,752-byte contract is high — approximately one conditional branch every 90 bytes. This density indicates heavily branched logic with many alternate code paths, consistent with a contract that behaves differently depending on caller, direction, and transfer parameters.

---

## The Kill Chain (Technical)

```
1. DEPLOYMENT
   Deployer creates contract with owner=0xe8E0C4...
   Contract registers function selectors for ERC-20 + Router + Pool + NFT
   Fee parameters written to storage via computed SHA3 slots

2. LIQUIDITY SEEDING
   Deployer adds SXAI/WETH liquidity to create a visible trading pair
   The getPool() function makes this contract discoverable by routers

3. ROUTING DISCOVERY
   Uniswap Universal Router's path-finding evaluates this pool
   Contract's visible reserves suggest normal pricing
   Router adds this pool to available routing paths

4. BUY PHASE (building trust)
   User swaps WETH → SXAI via Uniswap
   Router calls execute() (0x3593564c) on the contract
   CALLER check identifies this as router-initiated buy
   Buy path executes: WETH goes to pool, SXAI goes to user
   Transaction succeeds — user sees tokens in wallet

5. SELL PHASE (extraction)
   User swaps SXAI → WETH via Uniswap
   Router calls execute() on the contract
   CALLER check identifies this as a sell
   Contract forwards SXAI to LP pool
   LP pool sends WETH to CONTRACT (not user)
   SHA3→SLOAD→JUMPI→MUL fee mechanism activates
   Fee = 100% — all WETH retained by contract
   Transaction shows "success" — no revert
   User's SXAI is gone, WETH never arrives

6. EXTRACTION
   Owner calls custom functions to withdraw accumulated WETH
   WETH sent to 3 collection addresses:
     0xd462be33c46d84a0ce702103336f2fc290dcf159
     0xe502b1568aba07040a4580717e3399297067c50e
     0x07bd23d6ae11e61450ea74c4d96e21f3946eacb6
   Unicode impersonation tokens (Cyrillic WETH) distributed to obfuscate

7. CLEANUP
   Deployer wallet emptied (balance: 0, nonce: 10)
   Owner could renounceOwnership() to appear abandoned
   Contract continues operating autonomously
```

---

## Why Standard Tools Miss This

| Tool Type | What It Checks | Why It Misses This |
|---|---|---|
| Token scanner (TokenSniffer, GoPlus) | Token contract for honeypot patterns | This contract impersonates a POOL, not just a token. The token interface is secondary. |
| Honeypot detector (honeypot.is) | Can you buy AND sell | Buy works. Sell "succeeds" (no revert). The detector sees a successful sell and reports "safe." |
| Wallet security (Rabby, Pocket Universe) | Transaction simulation | Simulation shows the swap will execute successfully. It DOES execute — just with 100% WETH retained. |
| DEX aggregator safety | Slippage tolerance | Slippage check happens at the router level. The contract satisfies the check on buy-side. |
| Static bytecode scanner | Known malicious patterns | The fee mechanism uses SHA3-keyed storage (dynamic), not hardcoded values. Pattern matchers looking for specific byte sequences miss it. |
| **Layer 3 surveillance** | **Opcode pattern + population flow** | **Catches it.** SHA3→SLOAD→JUMPI→MUL pattern detected at deployment. Token flow imbalance confirmed via population analysis. |

---

## On-Chain Evidence Summary

| Evidence | Source | Value |
|---|---|---|
| Contract | `0xd4624228cce5baa0814c9e7f666a8a2c83b6f159` | Base |
| Deployer | `0xe8E0C4883D7196A7De87a6489F6Da58212dbE813` | Nonce 10, emptied |
| LP Pool | `0x7609153350cd0184c5df525d58490edf3bacef3b` | SXAI/WETH |
| Token | SXAI `0xea6b6bC260ED8241190C277d2fe7718Ea6CbF667` | Fake token |
| Total victims | 2,910+ | Growing ~1,338/day |
| WETH extracted | ~100.56 WETH (~$211,176) | Confirmed via deployer withdrawals |
| Revert rate | 0.3% | Near-invisible |
| Router delivery | 98.7% via `execute()` | Users never see the contract |
| Repeat victims | 71% return 2+ times | Don't know they're being skimmed |
| Bytecode | 14,752 bytes, 37 selectors | Multi-interface impersonation |
| Fee storage | Dynamic (SHA3-keyed mapping) | Modifiable by owner |
