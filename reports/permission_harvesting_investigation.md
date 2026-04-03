# Permission Harvesting Investigation
## Deep Analysis of Undrained Approval Population

---

## Critical Finding: 0x4752ba5d Is PancakeSwap V3 Router

**The centralized spender `0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24` is the PancakeSwap V3 SmartRouter on Base.**

This was already identified in our `live_exposures` table:

> "110 of 1320 Base contracts hold MAX_UINT256 approval to PancakeSwap V3 Router. Drain mechanism ready. Activation pending addLiquidity() call from deployer 0x2e20b2. Pre-staged liquidity trap inventory."

The 1,833 approval events pointing to this address are **normal DeFi behavior** -- tokens being approved for swap on PancakeSwap. The "third-party redirect" pattern for this address is not phishing; it's how DEX token approvals work (you approve the router to spend your tokens, then call swap).

**This deflates the "centralized permission harvester" hypothesis for this specific address.**

However, the Dragon (0x2e20b2...) has pre-staged 110 contracts with MAX_UINT256 approvals to this router, creating a ready-to-fire liquidity trap inventory. That's a legitimate finding -- but it's a staged liquidity trap, not permission harvesting.

---

## Other Third-Party Spenders

| Spender | Approvals | Identity | Assessment |
|---------|-----------|----------|------------|
| `0x4752ba5d...` | 1,833 | **PancakeSwap V3 Router** | Normal DEX infrastructure |
| `0x4dc0234f...` | 782 | Unknown | Needs on-chain verification |
| `0xd4624228...` | 464 | **Confirmed trap** | Trap contract collecting approvals directly |
| `0x000000000022d4...` | 236 | **Uniswap Permit2** | Normal DEX infrastructure |
| `0xec3576c5...` | 202 | Unknown | Needs investigation |
| `0x5b5e4dd7...` | 198 | Unknown | Single token, all unique wallets |
| `0x91a65ef6...` | 121 | **Confirmed trap** | Trap collecting approvals |
| `0x2ea47897...` | 55 | **Also deploys 1 trap** | Permission redirector -- genuine finding |

**Of the top 8 third-party spenders:**
- 2 are known DEX infrastructure (PancakeSwap, Permit2) -- normal
- 2 are confirmed trap contracts collecting approvals directly -- known threat, already tracked
- 1 (`0x2ea47897...`) deploys traps AND collects redirected approvals -- this is the genuine permission harvesting signal
- 3 are unknown and need on-chain verification

---

## The 188 Undrained Collectors

### Pattern A: Approval-Dominant Contracts (approve() > 50% of interactions)

Several top collectors show approve() as the dominant function call:

| Contract | Approvals | approve() % | Tier | Still Active |
|----------|-----------|-------------|------|-------------|
| `0xaa9c0875...` | 122 | **61.2%** | confirmed | YES |
| `0x485c2778...` | 465 | ~high | suspected | YES (Apr 2) |
| `0x16cdc3ac...` | 302 | ~high | suspected | YES (Apr 2) |

These contracts exist primarily to collect approvals. The approve() call IS the product, not a side effect.

### Pattern B: Zero-TX Approval Collectors

Multiple contracts have approvals tracked by the approval monitor but **zero entries in transaction_events**:

| Contract | Approvals | TX Events | Age | Notes |
|----------|-----------|-----------|-----|-------|
| `0xee8ef1ba...` | 403 | 0 | 9 days | Still undrained |
| `0x4a3e2069...` | 390 | 0 | 5 days | approval_trap entity |
| `0xe949de83...` | 89 | 0 | 9 days | confirmed tier, stopped |
| `0x76faaab0...` | 72 | 0 | 5 days | suspected, stopped |
| `0xf34f722f...` | 57 | 0 | 6 days | suspected, stopped |

These have approval activity tracked by the approval monitor (which watches approve() events on-chain) but our selector monitor never saw interactions. This means the approvals happened outside our transaction monitoring window or on contracts we weren't watching when the approvals occurred.

### Pattern C: The 983bc41b Selector

Multiple undrained collectors share the same unknown selector `983bc41b` as their primary interaction:

- `0x938d1699...`: 99 calls of 983bc41b + 92 approve()
- `0x6068b0b9...`: 114 calls of 983bc41b + 93 approve()
- `0x99ff98da...`: 109 calls of 983bc41b + 73 approve()
- `0xb945124e...`: 103 calls of 983bc41b + 60 approve()

This unknown selector appears across multiple contracts from different deployers. It could be:
- A standard function in a shared token template
- A custom function in a trap toolkit
- A legitimate DeFi function we haven't decoded

**This selector cluster deserves investigation** -- multiple contracts using the same unknown function alongside heavy approve() activity is a potential toolkit signature.

---

## What's Real vs What's Normal DeFi

### Deflated Findings
- **"Centralized permission harvester"**: The primary spender (0x4752ba5d) is PancakeSwap V3 Router. Normal DEX behavior.
- **"336 third-party redirects"**: Majority are approvals to known DEX routers. Expected in DeFi.
- **Permit harvesting**: Zero permit() calls detected. Not a vector on these L2s.

### Genuine Findings
- **5,355 undrained approvals (88.3%)**: The sheer volume is notable. 712 have been drained, but 5,355 remain open. Some are to known routers (benign) but many are to suspected contracts.
- **1 permission redirector that also deploys traps** (`0x2ea47897...`): 55 approvals collected, separate from its own trap contract. Small scale but genuine.
- **`0xd4624228...` confirmed trap with 464 undrained approvals**: This is the Uniswap router parasite -- collecting approvals it hasn't exercised yet. Known threat, large approval inventory.
- **983bc41b selector cluster**: Multiple undrained collectors sharing the same unknown function. Potential shared toolkit.
- **Approval-dominant contracts**: Several contracts where approve() is >50% of all interactions. The contract's primary purpose is collecting permissions.
- **Zero-TX approval collectors**: Contracts with hundreds of approvals but zero transaction_events. Operating in a gap in our monitoring.

---

## Conclusion

The permission harvesting hypothesis is **partially confirmed**. The "centralized spender" was a false alarm (PancakeSwap router), but:

1. **88% of tracked approvals are undrained** -- a much larger pending-threat surface than the 712 drains we've documented
2. **Multiple contracts exist primarily to collect approve() calls** -- the approval IS the product
3. **A shared unknown selector (983bc41b)** ties together multiple undrained collectors -- potential toolkit
4. **The Dragon's 110 pre-staged PancakeSwap approvals** are a confirmed staged liquidity trap inventory

This is not the "first documented permission harvesting network" we hoped to find. The scale is smaller and the biggest spender is legitimate infrastructure. But the undrained approval population is real, the approval-dominant contracts are suspicious, and the 983bc41b selector cluster warrants further investigation.

### Recommended Next Steps
1. Decode selector `983bc41b` -- determine if it's a known function or custom trap logic
2. On-chain verify `0x4dc0234f...` (782 approvals, unknown) and `0xec3576c5...` (202 approvals, unknown)
3. Track the 5,355 undrained approvals -- set alerts for any transferFrom from contracts with 10+ pending approvals
4. Monitor the Dragon's 110 PancakeSwap-approved contracts for addLiquidity() activation
