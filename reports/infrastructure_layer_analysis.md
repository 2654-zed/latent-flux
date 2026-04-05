# Infrastructure-Layer Extraction Analysis

## Phase 1: Proxy Watcher (PENDING — requires RPC)

DELEGATECALL contracts to check: 3
  0x768a86a8c83434623be3c04df80b44b59b2210ef  chain=arbitrum  deployed=2026-03-24
  0xfb6c1b210af8ed99f86d80322a57561fe47d0281  chain=arbitrum  deployed=2026-03-24
  0xd100d4117064bfd8eaba94d9d5ee67a9b81dd181  chain=arbitrum  deployed=2026-03-24

Run on Railway: `python -m surveillance.proxy_check_priority`
Budget: 6-12 RPC calls. Results will determine if implementations changed.

## Phase 2: Value Flow Analysis

### Ecosystem Map
Hub: 0x08b8b9410cba313728d98488a84b90c89fd01a17
Funded deployers: 12
Total downstream contracts: 23

### What funded deployers call on hub contracts
| Selector | Calls | Callers | Reverts | Name |
|----------|-------|---------|---------|------|
| none | 131 | 50 | 0 |  |
| 51227175 | 34 | 1 | 1 |  |
| 4963d3c4 | 23 | 1 | 1 |  |

### Approval events involving hub contracts
Approval events: 0

### Timing: do funded deployers call hub before or after deploying?
  0x04bd50af0325be14... hub_call=2026-03-29T22:50 deploy=2026-03-24T09:13 AFTER
  0x0addabb049aa800d... hub_call=2026-03-29T22:50 deploy=2026-03-26T18:12 AFTER
  0x392260d5d12a9cf0... hub_call=2026-03-29T22:49 deploy=2026-03-24T09:12 AFTER
  0x3a7d5fdac4171705... hub_call=2026-03-29T22:50 deploy=2026-03-24T09:14 AFTER
  0x59181eefd3fd88c7... hub_call=2026-03-29T22:49 deploy=2026-03-24T09:17 AFTER
  0x7469c8a16590c316... hub_call=2026-03-29T22:50 deploy=2026-03-29T00:23 AFTER
  0x89d0764b6acb79b6... hub_call=2026-03-29T22:50 deploy=2026-03-24T09:16 AFTER
  0xaf40c7bb4256ac44... hub_call=2026-03-29T22:50 deploy=2026-03-24T17:01 AFTER
  0xb2a9f0be5f44e798... hub_call=2026-03-29T22:50 deploy=2026-03-28T22:11 AFTER
  0xc933a15dc1ea4914... hub_call=2026-03-29T22:49 deploy=2026-03-24T09:18 AFTER
  0xcd581bf24532d81f... hub_call=2026-03-29T22:50 deploy=2026-03-26T09:54 AFTER

Call hub BEFORE deploying: 0
Call hub AFTER deploying: 11
Same period: 0

### Bytecode families used by funded deployers
  T1-5e2b3e66ae20: Tier1-DELEGATECALL at 0x12e3 in ERC-20 context deployers=6 contracts=14
  T1-8c0ca6557d14: Tier1-SELFDESTRUCT at 0x1071 in ERC-20 transfe deployers=3 contracts=5
  T2-eaef6a5d7678: Tier2-fee=0|asym=0|crev=0 deployers=1 contracts=1

## Phase 3: Broader Infrastructure-Layer Search

### Closed-loop infrastructure operators
(Addresses that fund 5+ deployers, deploy own contracts, and funded deployers call those contracts)


## Phase 4: Template Provider / TaaS Analysis

### Bytecode families with 10+ deployers (potential TaaS templates)

| Family | Deployers | Contracts | Name |
|--------|-----------|-----------|------|
| T2-eaef6a5d7678 | 8,240 | 21,936 | Tier2-fee=0|asym=0|crev=0 |
| T1-d5351e977044 | 435 | 435 | Tier1-ORIGIN at 0x314 -> EQ at 0x31d -> JUMPI  |
| T1-39b12abd4db3 | 61 | 69 | Tier1-CALLER at 0x2ab -> SLOAD at 0x2bf -> JUM |
| T1-78d4dfc7ac5f | 59 | 63 | Tier1-CALLER at 0x2ac -> SLOAD at 0x2c0 -> JUM |
| T1-fa8c132e5058 | 52 | 59 | Tier1-CALLER at 0x2ad -> SLOAD at 0x2c1 -> JUM |
| T1-19295c1373cf | 51 | 52 | Tier1-CALLER at 0x2aa -> SLOAD at 0x2be -> JUM |
| T1-5a5453695c06 | 40 | 41 | Tier1-CALLER at 0x2ae -> SLOAD at 0x2c2 -> JUM |
| T1-eeb31bf0e110 | 37 | 37 | Tier1-CALLER at 0x2a9 -> SLOAD at 0x2bd -> JUM |
| T1-55089c8d41f0 | 29 | 29 | Tier1-DELEGATECALL at 0x1235 in ERC-20 context |
| T1-a4fa11884721 | 23 | 25 | Tier1-CALLER at 0xea -> EQ at 0xee -> JUMPI at |
| T1-503e872128ae | 17 | 19 | Tier1-CALLER at 0x2af -> SLOAD at 0x2c3 -> JUM |
| T1-d81fc8f9ff59 | 17 | 17 | Tier1-CALLER at 0x2a8 -> SLOAD at 0x2bc -> JUM |
| T1-20637bbac346 | 16 | 16 | Tier1-CALLER at 0x2b0 -> SLOAD at 0x2c4 -> JUM |
| T1-2081a9d32218 | 16 | 21 | Tier1-DELEGATECALL at 0x1d33, 0x1ee1, 0x1ff3,  |
| T1-59099351d205 | 12 | 15 | Tier1-DELEGATECALL at 0xc32, 0xe07, 0xe71, 0xe |

### Funder concentration in top families
  T2-eaef6a5d7678: 984 funders / 8240 deployers = 0.12 ** CONCENTRATED **
  T1-d5351e977044: 2 funders / 435 deployers = 0.0 ** CONCENTRATED **
  T1-39b12abd4db3: 4 funders / 61 deployers = 0.07 ** CONCENTRATED **
  T1-78d4dfc7ac5f: 4 funders / 59 deployers = 0.07 ** CONCENTRATED **
  T1-fa8c132e5058: 4 funders / 52 deployers = 0.08 ** CONCENTRATED **