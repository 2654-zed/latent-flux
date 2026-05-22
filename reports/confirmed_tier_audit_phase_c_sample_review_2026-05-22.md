# Phase C STILL_AMBIGUOUS Sample Review

**Filed:** 2026-05-22  
**Sample size:** 50 of 209 STILL_AMBIGUOUS contracts (stratified: 22 verified + 28 unverified)  
**Method:** Deep per-contract review using Blockscout metadata, source code (where verified), transaction_events selector distribution, and trapping-bot serial-FP frequency. Sample is reproducible (random.seed=42).

## Sample verdict distribution

| Verdict | Count | Share |
|---|---|---|
| FP_FROM_SAMPLE | **29** | 58.0% |
| TP_FROM_SAMPLE | **3** | 6.0% |
| TRULY_AMBIGUOUS | **18** | 36.0% |

## Projection to the 209-contract residual

**Sample FP rate:** 58.0% (Wilson 95% CI [44.2%, 70.6%])

**Projected FP count in 209-contract residual:** ~121 (Wilson 95% CI [92, 147])

**If we treat the sample's FP rate as the residual's FP rate:** of the 209 STILL_AMBIGUOUS contracts, an estimated 121 are false-positives. Cumulative post-Phase-C+sample audit downgrade total would be 318 (Phase A+B+C) + ~121 (residual) = ~439, leaving the confirmed tier at ~1211.

## Refined classification rules applied

Rules in priority order; first match wins:

1. **Known-legit infrastructure contract name.** Blockscout `contract_name` matches a curated list of OZ proxies (`ERC1967Proxy`, `TransparentUpgradeableProxy`, `BeaconProxy`, `UUPSUpgradeable`), Chainlink aggregators (`AccessControlledOffchainAggregator`, `OffchainAggregator`), bridge components (`L1StandardBridge`, `L2StandardBridge`, `OptimismPortal`, `TokenVault`, `PublicBridge`), utility contracts (`Multicall3`, `Permit2`, `UniversalRouter`, `EntryPoint`), and Safe/Circle infrastructure. → FP.
2. **Known-legit phrase in verified source.** Verified-source text contains one of the curated phrases (`OpenZeppelin Contracts`, `@chainlink/contracts`, `@uniswap/`, `@safe-global/`, `FiatTokenV`, etc.). → FP.
3. **Serial-FP bot detection.** If the trapping bot has triggered ≥10 confirmed-tier labels (the OFC pre-launch front-runner pattern), the row is FP. The single bot `0x1a1d939b2ee78756…` accounts for hundreds of confirmed-tier rows; its reverts are the Correction-#24 pattern, not real victims.
4. **Standard ERC-20 selector dominance.** A contract with ≥85% of its txs using the four standard ERC-20 selectors is functionally an ERC-20 token. → FP.
5. **Verified source matches ERC-20 interface.** Any verified contract whose source contains `function transferFrom` plus `ERC20` or `IERC20` is a real token. → FP.
6. **Institutional Blockscout tag.** Tags containing `deployer`, `official`, `exchange`, `bridge`, `treasury`, `vault`, `router`, `factory`, `team`. → FP.
7. **Has token metadata.** Blockscout returned a `token` object with name + type — small legit token launch. → FP.
8. **Burst-then-die honeypot signature.** 5-30 interactors + ≤50 txs + ≥40% revert rate matches the honeypot template. → TP.
9. **Textbook honeypot.** ≤5 interactors + ≥90% revert + ≥20 txs on a non-standard selector. → TP.

**Removed from earlier draft:** the 'unique-bot signal = TP' rule that mis-classified OpenZeppelin proxies, Chainlink aggregators, and bridge TokenVaults as adversarial. Bot uniqueness alone is insufficient evidence.

## Per-contract reviews

### FP_FROM_SAMPLE — likely false-positives (29)

#### `0x03f1ec788ec92f66d806c9807337c1b81a68db67` (base)

- **Rationale:** Trapping bot 0x675a38b2a293 is a serial false-positive (231 contracts)
- **Name:** Storage
- **Activity:** 1 interactors, 22 txs, 100% revert
- **Trapping bot:** `0x675a38b2a293…` (triggered 231 confirmed labels total)
- **Top selectors:** [('6e0aacf7', 2), ('mint', 2), ('transferOwnership', 1)]
- **Source excerpt:** `// SPDX-License-Identifier: GPL-3.0  pragma solidity >=0.8.2 <0.9.0;  /**  * @title Storage  * @dev Store & retrieve value in a variable  * @custom:dev-run-scri…`

#### `0x984b9b7f36788cb9be6241a604de5eb18a5f1b50` (base)

- **Rationale:** Has token: Unitas (ERC-20)
- **Token:** Unitas
- **Name:** Token
- **Activity:** 9 interactors, 12 txs, 8% revert
- **Trapping bot:** `0x9c6c9d3a5d7b…` (triggered 1 confirmed labels total)
- **Top selectors:** [('approve', 10), ('transfer', 2)]
- **Source excerpt:** `pragma solidity ^0.8.26;  contract Token {     event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);      event Transfer(add…`

#### `0x7cff972b2227226ff058e9fabbf281e28200f2b9` (base)

- **Rationale:** contract_name='PublicBridge' = Bridge infrastructure
- **Name:** PublicBridge
- **Activity:** 6 interactors, 12 txs, 8% revert
- **Trapping bot:** `0xfa91a7af4152…` (triggered 1 confirmed labels total)
- **Top selectors:** [('3ec77124', 5), ('2f3a2cc7', 4), ('4df59eeb', 2)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT pragma solidity 0.8.18; import {IERC20, Token} from "./Token.sol";  /// @title TokenVault - Holds ERC20 tokens locked during bri…`

#### `0xfed9c78371a2694683eac531b1df5bdb59e929c2` (base)

- **Rationale:** Trapping bot 0x1a1d939b2ee7 is a serial false-positive (13 contracts)
- **Token:** 龙虾 (Lobster)
- **Name:** Token
- **Activity:** 8 interactors, 12 txs, 25% revert
- **Trapping bot:** `0x1a1d939b2ee7…` (triggered 13 confirmed labels total)
- **Top selectors:** [('approve', 8), ('35faa416', 2), ('transfer', 1)]
- **Source excerpt:** `pragma solidity ^0.8.26;  contract Token {     event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);      event Transfer(add…`

#### `0xcce968d27657ad0d80defb48d71867b687656a43` (base)

- **Rationale:** Trapping bot 0x1a1d939b2ee7 is a serial false-positive (13 contracts)
- **Token:** BurnDog
- **Name:** Token
- **Activity:** 7 interactors, 11 txs, 27% revert
- **Trapping bot:** `0x1a1d939b2ee7…` (triggered 13 confirmed labels total)
- **Top selectors:** [('approve', 7), ('transfer', 1), ('8129fc1c', 1)]
- **Source excerpt:** `pragma solidity ^0.8.26;  contract Token {     event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);      event Transfer(add…`

#### `0x3f302248214281d18ddfb4d55b264a1a4eddc4b1` (base)

- **Rationale:** Trapping bot 0x1a1d939b2ee7 is a serial false-positive (13 contracts)
- **Token:** OneFootball Credits
- **Name:** Token
- **Activity:** 8 interactors, 9 txs, 11% revert
- **Trapping bot:** `0x1a1d939b2ee7…` (triggered 13 confirmed labels total)
- **Top selectors:** [('approve', 7), ('transfer', 1), ('2e1a7d4d', 1)]
- **Source excerpt:** `pragma solidity ^0.8.26;  contract Token {     event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);      event Transfer(add…`

#### `0x6a714ae7b5b6ea520f6bca23d2e609c4fd5863f2` (arbitrum)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Name:** ERC1967Proxy
- **Activity:** 16 interactors, 335 txs, 6% revert
- **Trapping bot:** `0xa169574d0d35…` (triggered 1 confirmed labels total)
- **Top selectors:** [('b63c4262', 302), ('683f5efa', 23), ('9646d758', 1)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.2.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.22;  import {Proxy} from …`

#### `0xe6d50ff90532ff1926d32b8b4a757cb44af128e5` (base)

- **Rationale:** contract_name='PublicBridge' = Bridge infrastructure
- **Name:** PublicBridge
- **Activity:** 6 interactors, 20 txs, 10% revert
- **Trapping bot:** `0x2caaaa4ff6a1…` (triggered 1 confirmed labels total)
- **Top selectors:** [('3ec77124', 11), ('2f3a2cc7', 4), ('4df59eeb', 3)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT pragma solidity 0.8.18; import {IERC20, Token} from "./Token.sol";  /// @title TokenVault - Holds ERC20 tokens locked during bri…`

#### `0x144681860389b8925c19b4294dfe7f069d33607e` (optimism)

- **Rationale:** contract_name='AccessControlledOffchainAggregator' = Chainlink price aggregator
- **Name:** AccessControlledOffchainAggregator
- **Activity:** 10 interactors, 24 txs, 4% revert
- **Trapping bot:** `0xa04163e4033a…` (triggered 1 confirmed labels total)
- **Top selectors:** [('c9807539', 18), ('transferOwnership', 2), ('9c849b30', 1)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT pragma solidity 0.7.6;  import "./OffchainAggregator.sol"; import "./SimpleReadAccessController.sol";  /**  * @notice Wrapper of…`

#### `0xf1ed41ad769454bcd457cf84f37fb25d2ec58ef4` (arbitrum)

- **Rationale:** Verified source contains 'OpenZeppelin Contracts'
- **Token:** defimarketplus HIGHER-USD
- **Name:** defimarketplus HIGHER-USD
- **Activity:** 4 interactors, 41 txs, 2% revert
- **Trapping bot:** `0x11888716817f…` (triggered 2 confirmed labels total)
- **Top selectors:** [('8ed955b9', 32), ('6e553f65', 5), ('2f2ff15d', 2)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.1.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.20;  import {Proxy} from …`

#### `0xf0e75412c01fde8344f57e272329e30a3221ffc1` (base)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Name:** ERC1967Proxy
- **Activity:** 2 interactors, 369 txs, 0% revert
- **Trapping bot:** `0x712fb98782f3…` (triggered 1 confirmed labels total)
- **Top selectors:** [('eede8874', 273), ('062f7ed6', 70), ('80cfd1fa', 17)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.0.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.20;  import {Proxy} from …`

#### `0x612efc91d8f7f3c0940e7cba28312a35d52bc953` (arbitrum)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Name:** ERC1967Proxy
- **Activity:** 2 interactors, 51 txs, 12% revert
- **Trapping bot:** `0x01e87caa86c9…` (triggered 1 confirmed labels total)
- **Top selectors:** [('da78a9ad', 50), ('transferOwnership', 1)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.2.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.22;  import {Proxy} from …`

#### `0x2c259d3015dbfa25bfee72baf91f23b3591a593a` (base)

- **Rationale:** Trapping bot 0x1a1d939b2ee7 is a serial false-positive (13 contracts)
- **Token:** iShares MSCI Japan ETF (Derivatives)
- **Name:** Token
- **Activity:** 4 interactors, 5 txs, 20% revert
- **Trapping bot:** `0x1a1d939b2ee7…` (triggered 13 confirmed labels total)
- **Top selectors:** [('approve', 3), ('transfer', 1), ('8129fc1c', 1)]
- **Source excerpt:** `pragma solidity ^0.8.26;  contract Token {     event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);      event Transfer(add…`

#### `0x5960e66f168b441f055e5bacedfd7ad612f87004` (base)

- **Rationale:** Trapping bot 0x675a38b2a293 is a serial false-positive (231 contracts)
- **Name:** Storage
- **Activity:** 1 interactors, 23 txs, 100% revert
- **Trapping bot:** `0x675a38b2a293…` (triggered 231 confirmed labels total)
- **Top selectors:** [('6e0aacf7', 2), ('mint', 2), ('853828b6', 2)]
- **Source excerpt:** `// SPDX-License-Identifier: GPL-3.0  pragma solidity >=0.8.2 <0.9.0;  /**  * @title Storage  * @dev Store & retrieve value in a variable  * @custom:dev-run-scri…`

#### `0xda8f9076f965f8d9d4d0791192d5c100220ed79e` (base)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Token:** Bondingv2test
- **Name:** ERC1967Proxy
- **Activity:** 5 interactors, 17 txs, 6% revert
- **Trapping bot:** `0x3380266ca63a…` (triggered 3 confirmed labels total)
- **Top selectors:** [('3d98ebed', 5), ('213f4ab8', 5), ('9940686e', 4)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.0.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.20;  import {Proxy} from …`

#### `0x1bbda943b2db98a619212147f71669817d9b1a65` (base)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Name:** ERC1967Proxy
- **Activity:** 18 interactors, 50 txs, 8% revert
- **Trapping bot:** `0xc22ee3cf5521…` (triggered 1 confirmed labels total)
- **Top selectors:** [('ebfa7ade', 18), ('8119c065', 18), ('3659cfe6', 5)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v4.7.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.0;  import "../Proxy.sol"…`

#### `0x28d89489d86373006466fdb5e736d21626dcc4b8` (arbitrum)

- **Rationale:** Verified source is ERC-20 token (transferFrom + ERC20 interface)
- **Name:** MEVExecutor
- **Activity:** 3 interactors, 59 txs, 32% revert
- **Trapping bot:** `0x343c3928d659…` (triggered 1 confirmed labels total)
- **Top selectors:** [('6f63677c', 57), ('806ad57e', 2)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // 建议升级到 0.8.4 及以上以支持 Custom Errors pragma solidity ^0.8.4;  interface IERC20 {     function transfer(address to, uint256 a…`

#### `0x765f39e0959ccd8f849937135d607494f967f1a7` (base)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Token:** teststakingPOD
- **Name:** ERC1967Proxy
- **Activity:** 5 interactors, 22 txs, 4% revert
- **Trapping bot:** `0x3380266ca63a…` (triggered 3 confirmed labels total)
- **Top selectors:** [('approve', 13), ('6e553f65', 6), ('2f2ff15d', 2)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.0.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.20;  import {Proxy} from …`

#### `0x49f044ae0ff61b02dde8be7b07455cafefd1d9ea` (arbitrum)

- **Rationale:** contract_name='ERC1967Proxy' = OpenZeppelin ERC1967 upgradeable proxy
- **Name:** ERC1967Proxy
- **Activity:** 10 interactors, 859 txs, 1% revert
- **Trapping bot:** `0xf3a8db4c1655…` (triggered 2 confirmed labels total)
- **Top selectors:** [('88457d08', 756), ('91a24fb3', 37), ('a1f1e137', 30)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT // OpenZeppelin Contracts (last updated v5.2.0) (proxy/ERC1967/ERC1967Proxy.sol)  pragma solidity ^0.8.22;  import {Proxy} from …`

#### `0xdecc8d7de2777ae4f8c37e3758d9eee09dcf38fd` (base)

- **Rationale:** Trapping bot 0x1a1d939b2ee7 is a serial false-positive (13 contracts)
- **Token:** MOONDOGE (moondoge.world)
- **Name:** Token
- **Activity:** 6 interactors, 8 txs, 12% revert
- **Trapping bot:** `0x1a1d939b2ee7…` (triggered 13 confirmed labels total)
- **Top selectors:** [('approve', 6), ('transfer', 1), ('2e1a7d4d', 1)]
- **Source excerpt:** `pragma solidity ^0.8.26;  contract Token {     event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);      event Transfer(add…`

#### `0x359c6981f660d363e6741baef1bda23773706729` (optimism)

- **Rationale:** contract_name='AccessControlledOffchainAggregator' = Chainlink price aggregator
- **Name:** AccessControlledOffchainAggregator
- **Activity:** 4 interactors, 7 txs, 14% revert
- **Trapping bot:** `0xd46acba18e4f…` (triggered 1 confirmed labels total)
- **Top selectors:** [('c9807539', 3), ('9c849b30', 1), ('585aa7de', 1)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT pragma solidity 0.7.6;  import "./OffchainAggregator.sol"; import "./SimpleReadAccessController.sol";  /**  * @notice Wrapper of…`

#### `0xff7af2a2b1adf41fb3121fbe942eb08c2d87d490` (base)

- **Rationale:** Trapping bot 0x675a38b2a293 is a serial false-positive (231 contracts)
- **Activity:** 3 interactors, 272 txs, 1% revert
- **Trapping bot:** `0x675a38b2a293…` (triggered 231 confirmed labels total)
- **Top selectors:** [('45761772', 192), ('4799d5f7', 78), ('transferOwnership', 1)]

#### `0xcf8fabb9e96088de90320f48d9faff65009a99db` (arbitrum)

- **Rationale:** Has token: Apyee USDC Vault (ERC-20)
- **Token:** Apyee USDC Vault
- **Name:** Apyee USDC Vault
- **Activity:** 3 interactors, 196 txs, 0% revert
- **Trapping bot:** `0xeabaad2fb8f2…` (triggered 1 confirmed labels total)
- **Top selectors:** [('77e186b6', 89), ('fa211040', 88), ('6e553f65', 7)]

#### `0x91734696e8164cbf79b666569d2504b0e21218f6` (base)

- **Rationale:** Has token: Ardinal (ERC-721)
- **Token:** Ardinal
- **Name:** Ardinal
- **Activity:** 27 interactors, 91 txs, 22% revert
- **Trapping bot:** `0xa199820ba026…` (triggered 1 confirmed labels total)
- **Top selectors:** [('daedc78e', 61), ('transferFrom', 20), ('42842e0e', 9)]

#### `0x2f805fba291b1e9467643a39ded67228091995f3` (base)

- **Rationale:** Trapping bot 0x1a1d939b2ee7 is a serial false-positive (13 contracts)
- **Activity:** 2 interactors, 13 txs, 92% revert
- **Trapping bot:** `0x1a1d939b2ee7…` (triggered 13 confirmed labels total)
- **Top selectors:** [('a980f971', 11), ('db2e21bc', 1), ('9e281a98', 1)]

#### `0x03b9f640dc3ef49188d94c359344e6ad8620c92b` (base)

- **Rationale:** Trapping bot 0x6fbd620bc264 is a serial false-positive (12 contracts)
- **Activity:** 2 interactors, 11146 txs, 0% revert
- **Trapping bot:** `0x6fbd620bc264…` (triggered 12 confirmed labels total)
- **Top selectors:** [('6f074e32', 11145), ('NULL', 1)]

#### `0x7eb26c03015df1dd7c84913c70b77e5ac8e94eba` (base)

- **Rationale:** Has token: Chimi Strategy USDC SteakhouseHighYieldTurbo (ERC-20)
- **Token:** Chimi Strategy USDC SteakhouseHighYieldTurbo
- **Name:** Chimi Strategy USDC SteakhouseHighYieldTurbo
- **Activity:** 2 interactors, 66 txs, 53% revert
- **Trapping bot:** `0xc272260bf796…` (triggered 1 confirmed labels total)
- **Top selectors:** [('2606a10b', 65), ('57d89c5c', 1)]

#### `0xf79c2dc829cd3a2d8ceec353bdb1b2414ba1eee0` (arbitrum)

- **Rationale:** Has token: The Pool Zap LP Vault V2.1 (ERC-20)
- **Token:** The Pool Zap LP Vault V2.1
- **Name:** The Pool Zap LP Vault V2.1
- **Activity:** 6 interactors, 35 txs, 23% revert
- **Trapping bot:** `0x5cb4d906f046…` (triggered 1 confirmed labels total)
- **Top selectors:** [('1c4a8617', 8), ('38cfc17f', 6), ('6e553f65', 5)]

#### `0xa558533a48f97e145783470ca2fee1d17b230c94` (base)

- **Rationale:** Trapping bot 0x675a38b2a293 is a serial false-positive (231 contracts)
- **Token:** HASH98
- **Name:** HASH98
- **Activity:** 3 interactors, 40 txs, 2% revert
- **Trapping bot:** `0x675a38b2a293…` (triggered 231 confirmed labels total)
- **Top selectors:** [('50fd7367', 15), ('acc8f306', 8), ('d6febde8', 4)]

### TP_FROM_SAMPLE — likely true-positive adversarial (3)

#### `0xbf8766cb657d306bd3fc1d482cbeb422b7530584` (base)

- **Rationale:** Textbook honeypot: 2 interactors, 100% revert, 75 txs
- **Activity:** 2 interactors, 75 txs, 100% revert
- **Trapping bot:** `0xbbe8e731627f…` (triggered 1 confirmed labels total)
- **Top selectors:** [('a7e4008e', 75)]

#### `0x6e0f21e50a8ca1ee8b2c03309c41e86a860c18ad` (base)

- **Rationale:** Textbook honeypot: 1 interactors, 100% revert, 31 txs
- **Activity:** 1 interactors, 31 txs, 100% revert
- **Trapping bot:** `0x644b7154a6e4…` (triggered 1 confirmed labels total)
- **Top selectors:** [('16443568', 31)]

#### `0xf835cc3afd0bc4a8ce3f36a28bbce2302960618d` (base)

- **Rationale:** Textbook honeypot: 1 interactors, 100% revert, 80 txs
- **Activity:** 1 interactors, 80 txs, 100% revert
- **Top selectors:** [('30cf6ea2', 80)]

### TRULY_AMBIGUOUS — cannot decide with available signal (18)

#### `0x75682ac9d721250c2e677bd25ad5cb4133e0a09b` (arbitrum)

- **Rationale:** No decisive signal: interactors=2, txs=33, revert=3%, bot_traps=2, verified=True, top_selectors=[('91955002', 8), ('e91a7ca6', 6), ('f5c89aa7', 6)]
- **Name:** SLIQCore
- **Activity:** 2 interactors, 33 txs, 3% revert
- **Trapping bot:** `0x6b4db669cb16…` (triggered 2 confirmed labels total)
- **Top selectors:** [('91955002', 8), ('e91a7ca6', 6), ('f5c89aa7', 6)]
- **Source excerpt:** `// SPDX-License-Identifier: MIT pragma solidity ^0.8.24;  import {IERC721ReceiverMinimal} from "./interfaces/IERC721ReceiverMinimal.sol"; import {INonfungiblePo…`

#### `0x3bee729955fffa10f5ba7c420241ebd8a10adb8a` (base)

- **Rationale:** No decisive signal: interactors=126, txs=1095, revert=97%, bot_traps=1, verified=False, top_selectors=[('3229befd', 1055), ('42c72e56', 33), ('d8fc063d', 3)]
- **Activity:** 126 interactors, 1095 txs, 97% revert
- **Trapping bot:** `0x5eb8de8fb313…` (triggered 1 confirmed labels total)
- **Top selectors:** [('3229befd', 1055), ('42c72e56', 33), ('d8fc063d', 3)]

#### `0x4486738ec027f0776b25ac2f5e2744fce6f96e1e` (base)

- **Rationale:** No decisive signal: interactors=3, txs=16, revert=6%, bot_traps=1, verified=False, top_selectors=[('6a5bbc1d', 6), ('b6b55f25', 4), ('4a578978', 4)]
- **Activity:** 3 interactors, 16 txs, 6% revert
- **Trapping bot:** `0x11952796edc9…` (triggered 1 confirmed labels total)
- **Top selectors:** [('6a5bbc1d', 6), ('b6b55f25', 4), ('4a578978', 4)]

#### `0x91d7d9460e8ea59c75c092cc28e724e1e03605f5` (base)

- **Rationale:** No decisive signal: interactors=2, txs=43, revert=2%, bot_traps=1, verified=False, top_selectors=[('1878abc2', 14), ('c7925e72', 10), ('93a70c99', 3)]
- **Activity:** 2 interactors, 43 txs, 2% revert
- **Trapping bot:** `0x069d202dbb0a…` (triggered 1 confirmed labels total)
- **Top selectors:** [('1878abc2', 14), ('c7925e72', 10), ('93a70c99', 3)]

#### `0xa4c738f7389c3d7eceb7e000f2f820814591a100` (base)

- **Rationale:** No decisive signal: interactors=2, txs=15, revert=13%, bot_traps=1, verified=False, top_selectors=[('6643adee', 15)]
- **Activity:** 2 interactors, 15 txs, 13% revert
- **Trapping bot:** `0xcffdfadd06b9…` (triggered 1 confirmed labels total)
- **Top selectors:** [('6643adee', 15)]

#### `0xbf5db1bf2cdf8a70a0b73988e856bd90181003af` (base)

- **Rationale:** No decisive signal: interactors=2, txs=6044, revert=14%, bot_traps=1, verified=False, top_selectors=[('7d7fa4f5', 5910), ('f3fef3a3', 134)]
- **Activity:** 2 interactors, 6044 txs, 14% revert
- **Trapping bot:** `0x829bb1ad5bbe…` (triggered 1 confirmed labels total)
- **Top selectors:** [('7d7fa4f5', 5910), ('f3fef3a3', 134)]

#### `0xc3e4cc4039275582684b6ecb7b57945a0501d78a` (base)

- **Rationale:** No decisive signal: interactors=11, txs=78131, revert=36%, bot_traps=1, verified=False, top_selectors=[('19115035', 77599), ('2da1ceaa', 508), ('f80f5dd5', 11)]
- **Activity:** 11 interactors, 78131 txs, 36% revert
- **Trapping bot:** `0x5300cf6102e9…` (triggered 1 confirmed labels total)
- **Top selectors:** [('19115035', 77599), ('2da1ceaa', 508), ('f80f5dd5', 11)]

#### `0xeefa24c1d9a08b075d577694221e9cac941d7b95` (arbitrum)

- **Rationale:** No decisive signal: interactors=5, txs=4286, revert=59%, bot_traps=1, verified=False, top_selectors=[('af56878e', 4285), ('NULL', 1)]
- **Activity:** 5 interactors, 4286 txs, 59% revert
- **Trapping bot:** `0x76aedacce2a7…` (triggered 1 confirmed labels total)
- **Top selectors:** [('af56878e', 4285), ('NULL', 1)]

#### `0xf986c902f4a0adf24115886dd0baa315c250af10` (optimism)

- **Rationale:** No decisive signal: interactors=6, txs=4988, revert=2%, bot_traps=1, verified=False, top_selectors=[('040b2c63', 1354), ('04a75127', 1100), ('310b2c63', 884)]
- **Activity:** 6 interactors, 4988 txs, 2% revert
- **Trapping bot:** `0x1e3073a1de4c…` (triggered 1 confirmed labels total)
- **Top selectors:** [('040b2c63', 1354), ('04a75127', 1100), ('310b2c63', 884)]

#### `0x9836dac3bd474b51a7c428fc0caa14e77fc41024` (arbitrum)

- **Rationale:** No decisive signal: interactors=2, txs=1241, revert=73%, bot_traps=2, verified=False, top_selectors=[('b31d33e9', 643), ('8f9570fc', 334), ('2c754a5c', 250)]
- **Activity:** 2 interactors, 1241 txs, 73% revert
- **Trapping bot:** `0x69ab884ba714…` (triggered 2 confirmed labels total)
- **Top selectors:** [('b31d33e9', 643), ('8f9570fc', 334), ('2c754a5c', 250)]

#### `0x346434dbb1dfd0d941ab0805cfa7b6d65c927ab0` (base)

- **Rationale:** No decisive signal: interactors=11, txs=24, revert=4%, bot_traps=1, verified=False, top_selectors=[('7b0472f0', 18), ('93707a19', 5), ('5312ea8e', 1)]
- **Activity:** 11 interactors, 24 txs, 4% revert
- **Trapping bot:** `0x45af412e42f4…` (triggered 1 confirmed labels total)
- **Top selectors:** [('7b0472f0', 18), ('93707a19', 5), ('5312ea8e', 1)]

#### `0xabc4b078fbc110161b194ad618d0c8ebb6325c34` (arbitrum)

- **Rationale:** No decisive signal: interactors=25, txs=84574, revert=62%, bot_traps=1, verified=False, top_selectors=[('00000001', 41813), ('00000005', 29280), ('00000002', 5322)]
- **Activity:** 25 interactors, 84574 txs, 62% revert
- **Trapping bot:** `0x3034d0499596…` (triggered 1 confirmed labels total)
- **Top selectors:** [('00000001', 41813), ('00000005', 29280), ('00000002', 5322)]

#### `0xbb2d213f794a56ae179797dd144021f35200a319` (base)

- **Rationale:** No decisive signal: interactors=30, txs=76, revert=18%, bot_traps=1, verified=False, top_selectors=[('59840cd4', 76)]
- **Activity:** 30 interactors, 76 txs, 18% revert
- **Trapping bot:** `0x45dbcc9c3fea…` (triggered 1 confirmed labels total)
- **Top selectors:** [('59840cd4', 76)]

#### `0x196a943c9cfc795a1607945e1924a48a113feea7` (base)

- **Rationale:** No decisive signal: interactors=2, txs=143, revert=9%, bot_traps=1, verified=False, top_selectors=[('1de6e2cc', 67), ('5a06dd14', 41), ('066b3cbd', 33)]
- **Activity:** 2 interactors, 143 txs, 9% revert
- **Trapping bot:** `0x9f111112d889…` (triggered 1 confirmed labels total)
- **Top selectors:** [('1de6e2cc', 67), ('5a06dd14', 41), ('066b3cbd', 33)]

#### `0x6b5df69bd79faef13b755254953a13ba4d0fb94a` (base)

- **Rationale:** No decisive signal: interactors=42, txs=101, revert=4%, bot_traps=1, verified=False, top_selectors=[('f6d956df', 97), ('2f2ff15d', 2), ('d547741f', 2)]
- **Activity:** 42 interactors, 101 txs, 4% revert
- **Trapping bot:** `0xaf333f56262d…` (triggered 1 confirmed labels total)
- **Top selectors:** [('f6d956df', 97), ('2f2ff15d', 2), ('d547741f', 2)]

#### `0x8eacdb008568431483f07161a53d7e8d555f58c3` (base)

- **Rationale:** No decisive signal: interactors=3, txs=2364, revert=70%, bot_traps=1, verified=False, top_selectors=[('3fab50da', 2350), ('8acb9299', 5), ('05c1a77e', 5)]
- **Activity:** 3 interactors, 2364 txs, 70% revert
- **Trapping bot:** `0x73a8e582a640…` (triggered 1 confirmed labels total)
- **Top selectors:** [('3fab50da', 2350), ('8acb9299', 5), ('05c1a77e', 5)]

#### `0xc16b1a753308bf909eddcec07bdeded155b0aba7` (base)

- **Rationale:** No decisive signal: interactors=2, txs=31, revert=3%, bot_traps=2, verified=False, top_selectors=[('f0d6e843', 29), ('NULL', 1), ('43686563', 1)]
- **Activity:** 2 interactors, 31 txs, 3% revert
- **Trapping bot:** `0xa57accc582ab…` (triggered 2 confirmed labels total)
- **Top selectors:** [('f0d6e843', 29), ('NULL', 1), ('43686563', 1)]

#### `0xbff5fe7c1565d61aa57ca19192ec4af81a4ab8ee` (arbitrum)

- **Rationale:** No decisive signal: interactors=1, txs=11, revert=73%, bot_traps=1, verified=False, top_selectors=[('24856bc3', 11)]
- **Activity:** 1 interactors, 11 txs, 73% revert
- **Trapping bot:** `0x4bdb8234ad81…` (triggered 1 confirmed labels total)
- **Top selectors:** [('24856bc3', 11)]
