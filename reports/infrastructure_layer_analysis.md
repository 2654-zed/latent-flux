# Infrastructure-Layer Extraction Analysis

## Correction Notice

Initial analysis flagged `0x93614117...` (stored in proxy slot 1) as a potentially destroyed contract implementation -- evidence of anti-forensic behavior. **Verification via Blockscout and RPC revealed this address is an EOA (externally owned account), not a destroyed contract.** It is actively transacting on Aave V3, Uniswap V3, and ParaSwap as of April 5, 2026.

The proxy watcher has been updated to distinguish EOAs (zero code + nonzero nonce) from destroyed contracts (zero code + creation tx exists) to prevent this class of false positive.

The closed-loop infrastructure ecosystem around `0x08b8b941` is confirmed (12 funded deployers calling back to hub contracts after deploying), but evidence of value extraction is **inconclusive**. The infrastructure-layer threat model remains architecturally valid but lacks a confirmed instance in this corpus.

---

## Phase 1: Proxy Watcher Results

DELEGATECALL contracts checked via RPC:

| Contract | Slot 1 (impl) | Slot 2 (admin) | Status |
|----------|---------------|----------------|--------|
| `0x768a86a8...` | `0x93614117...` | `0xf611aeb5...` | Impl is EOA |
| `0xfb6c1b21...` | `0x93614117...` | `0xf611aeb5...` | Impl is EOA |
| `0xd100d411...` | `0x1d1d4d02...` | `0xf611aeb5...` | 23-byte EOF stub |

- EIP-1967 slots are all zero -- custom proxy pattern using slot 1/2
- Admin contract `0xf611aeb5...` = 2,015 bytes, standard OpenZeppelin ProxyAdmin (upgradeTo, changeAdmin, implementation, admin)
- Admin nonce = 1 (single tx ever sent)
- `0x93614117...` is an EOA doing DeFi (Aave supply/borrow, Uniswap LP, ParaSwap swaps) -- NOT a destroyed implementation
- `0x1d1d4d02...` contains `0xef01...` prefix (INVALID opcode) -- a non-executable stub

## Phase 2: Value Flow Analysis

### Ecosystem Map
Hub: `0x08b8b9410cba313728d98488a84b90c89fd01a17`
Funded deployers: 12
Total downstream contracts: 23

### Selectors called on hub contracts
| Selector | Calls | Callers | Reverts | Note |
|----------|-------|---------|---------|------|
| none (empty calldata) | 131 | 50 | 0 | ETH transfers or fallback calls |
| 51227175 | 34 | 1 | 1 | Custom function |
| 4963d3c4 | 23 | 1 | 1 | Custom function |

### Approval events: 0
No approve/transferFrom extraction mechanism detected.

### Timing: funded deployers call hub AFTER deploying (11/11)
All hub callbacks occur on March 29+, despite deployers being created March 24-29.

### Bytecode families used by funded deployers
- T1-5e2b3e66ae20: DELEGATECALL template (6 deployers, 14 contracts)
- T1-8c0ca6557d14: SELFDESTRUCT template (3 deployers, 5 contracts)
- T2-eaef6a5d7678: Generic no-signature (1 deployer, 1 contract)

## Phase 3: Broader Search -- No Other Instances Found

Searched all 50 funders with 5+ funded deployers. Filtered to those who also deploy 3+ own contracts. Checked if funded deployers call back to the funder's contracts.

**Result: 0 additional closed-loop operators found.** `0x08b8b941` is the only address in the corpus matching the pattern (fund deployers + deploy service contracts + receive callbacks). This pattern is rare -- or our detection criteria are too narrow.

## Phase 4: TaaS Template Distribution

### Highly concentrated bytecode families
| Family | Deployers | Funders | Ratio | Pattern |
|--------|-----------|---------|-------|---------|
| T2-eaef6a5d7678 | 8,240 | 984 | 0.12 | **CONCENTRATED** -- 984 funders distribute to 8,240 wallets |
| T1-d5351e977044 | 435 | 2 | 0.00 | **2 operators** running 435 deployer wallets |
| T1-39b12abd4db3 | 61 | 4 | 0.07 | 4 funders, 61 deployers |
| T1-78d4dfc7ac5f | 59 | 4 | 0.07 | Same pattern |
| T1-fa8c132e5058 | 52 | 4 | 0.08 | Same pattern |

The TaaS evidence remains strong: highly concentrated funding behind shared bytecode templates. Whether the templates contain hidden extraction logic requires bytecode-level analysis of each family's deployment code.

## Assessment

### Confirmed
- Closed-loop ecosystem exists: `0x08b8b941` funds 12 deployers who call back after deploying
- TaaS distribution networks: concentrated funders behind shared templates (T1-d5351e977044: 2 funders for 435 deployers)
- All funded deployers call hub AFTER deploying, not before

### Not confirmed
- Value extraction through the hub contracts (selectors are custom, no approve/transfer patterns)
- Implementation destruction as anti-forensic technique (the "destroyed" address is an EOA)
- Any additional closed-loop operators beyond `0x08b8b941`

### Threat model
The infrastructure-layer extraction concept is architecturally valid:
- Layer 1: Trap contracts extract from victims/bots
- Layer 2: Organizations coordinate trap operators
- Layer 3 (hypothesized): Infrastructure extracts from organizations

We have evidence for Layers 1 and 2. Layer 3 remains a theoretical framework without a confirmed instance. The TaaS distribution networks (Phase 4) are the strongest lead -- if template bytecode contains hardcoded fee collection addresses, that's infrastructure-layer extraction via code distribution rather than contract interaction.
