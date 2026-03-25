# CASE FILE: ORG_003 — Ghost Fee-Skimmer Network
**Classification:** CONFIRMED CRIMINAL ORGANIZATION
**Generated:** 2026-03-25
**Chain:** Base

---

## Executive Summary
Six fee-skimming contracts deployed by six disposable wallets with zero traceable funding. All share the identical bytecode pattern (`SHA3 at 0x1508 -> SLOAD at 0x1509 -> JUMPI at 0x151c -> DIV at 0x152e`) with minor offset variations. 81-85% victim overlap between contract pairs proves a single operator. 727 combined victims. The operator's funding is completely invisible to all standard and advanced transfer APIs — the most sophisticated opsec of any organization in the corpus.

---

## Organization Structure

```
UNKNOWN FUNDING SOURCE
  (no external, internal, or ERC-20 inflows detected on ANY deployer)
        |
        +-> Deployer 1 (0x392c56) -> Contract (0x0697a1) [137 victims]
        +-> Deployer 2 (0x9f5db1) -> Contract (0x201b32) [122 victims]
        +-> Deployer 3 (0xadb085) -> Contract (0x7709a1) [122 victims]
        +-> Deployer 4 (0x8f007f) -> Contract (0xf2b2b7) [122 victims]
        +-> Deployer 5 (0x571ba9) -> Contract (0xa80899) [117 victims]
        +-> Deployer 6 (0x888a49) -> Contract (0xc3bc6e) [107 victims]
```

## Deployer Profiles

| Deployer | Contract | Victims | Nonce | Balance | Funding |
|---|---|---|---|---|---|
| `0x392c564a28d6d87d326e8a385c764355e130418d` | `0x0697a11c537829...` | 137 | 6 | 0 | INVISIBLE |
| `0x9f5db1b0436815de051b86a732c3d8ad8877bbef` | `0x201b32f4a87c21...` | 122 | 6 | 0 | INVISIBLE |
| `0xadb085d8279bf7afad7599d3b8685547b19b2798` | `0x7709a1e3fe44a0...` | 122 | 6 | 0 | INVISIBLE |
| `0x8f007f3e4f83a57c34dc4bf19237423665d28dd8` | `0xf2b2b76e439618...` | 122 | 7 | 0 | INVISIBLE |
| `0x571ba99571f588d8238cb56a244de1c44e1d1da7` | `0xa80899d4ce98a4...` | 117 | 5 | 0.41 ETH | INVISIBLE |
| `0x888a491ed0a09c9325507aaccead28a099a2104f` | `0xc3bc6e376159b2...` | 107 | 6 | 0 | INVISIBLE |

All deployers: EOA, nonce 5-7, zero balance (except 0x571ba9 with 0.41 ETH remaining), zero inflows on external/internal/ERC-20 transfer categories.

## Evidence of Single Operator

**81-85% victim overlap.** 98 of 121 victims on `0x201b32` also appear on `0xa80899`. This level of overlap is impossible for independent operators — the same pool of addresses is being routed to both contracts.

**Identical bytecode pattern at identical offsets.** 5 of 6 contracts: `SHA3 at 0x1508 -> SLOAD at 0x1509 -> JUMPI at 0x151c -> DIV at 0x152e`. The 6th (`0xc3bc6e`) has the same logic at different offsets — same compilation, different version.

**Different code hashes.** All 6 have unique bytecodes — parameterized per deployment (different token addresses, fee parameters, or owner addresses). This is a template with per-deployment customization.

**Uniform nonces.** All deployers have nonce 5-7 — the same deployment script creates each deployer, deploys 1 contract, and runs a few setup transactions.

## Anti-Forensics

This is the only organization in the corpus with **zero traceable funding**. The Alchemy `getAssetTransfers` API returns nothing for any transfer category (external, internal, ERC-20) on any of the 6 deployer addresses. Possible funding methods:

1. **Cross-chain bridge deposits** — L1-to-L2 bridge mints don't always surface in standard transfer APIs
2. **Self-destructing funder contracts** — a contract sends ETH then self-destructs, leaving no trace
3. **CREATE2 precomputed funding** — ETH sent to a deterministic address before the deployer EOA exists

## Relationship to Other Orgs

- **org_001:** No connection. Different chain focus, different bytecode, different funding infrastructure.
- **org_002:** No connection. Different pattern (tx.origin vs fee-on-transfer), different opsec.
- **Infrastructure parasite (0xd4624228):** Different operator. The parasite uses Uniswap routing; org_003 does not.

## Risk Assessment

| Factor | Rating |
|---|---|
| Victim count | HIGH (727 across 6 contracts) |
| Trap sophistication | MODERATE (standard fee template) |
| Operational security | **CRITICAL** (invisible funding) |
| Revert camouflage | HIGH (0-8% across all contracts) |
| Active status | ACTIVE (0x571ba9 still has 0.41 ETH) |

**Overall: HIGH** — The opsec makes this organization the hardest to trace. The bytecode is standard but the infrastructure is invisible.
