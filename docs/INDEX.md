# Layer 3 — Documentation Index

**Version:** 2026-04-25
**Purpose:** Maintained topic-to-file map. Read this at session start before producing any analytical finding. Every entry resolves to a primary file path; secondary files are listed where they add evidence.
**Discipline:** This index reflects what is *currently documented*. Aspirational entries do not belong here. When a new case file lands in `surveillance/data/cases/` or `reports/`, the index must be updated in the same commit.
**Two correction logs exist** and the index references both: `CORRECTIONS.md` (project root, date-titled, customer-facing claims) and `reports/correction_log.md` (numbered #1–#17, methodology corrections). Always check both.

---

## Section 1 — Organizational Entities

### org_001 — DELEGATECALL Trap Network (CEX-funded, multi-chain)
- **ID:** `org_001`
- **Description:** Professional multi-chain trap operation on Arbitrum + Base, traceable to two CEX origins (Coinbase 2023 seed + Binance 2021 33,333 ETH withdrawal). 7-tier wallet graph, vanity-spoofed shadow wallets (intelligence-layer counter-forensics), simultaneously target of external address-poisoning campaigns.
- **Primary case file:** `surveillance/data/cases/CASE_ORG_001_INFRASTRUCTURE.md` (last updated 2026-04-11)
- **Supporting:** `surveillance/data/cases/CASE_ORG_001_ETHEREUM_DEPTH.md`, `surveillance/data/cases/CROSS_ORG_ANALYSIS_20260322.md`
- **Status:** CONFIRMED — actively expanding
- **Headcount caveat:** `CASE_ORG_001_INFRASTRUCTURE.md` cites "559+ deployers / 7,400+ contracts." `CORRECTIONS.md` 2026-04-02 documents "actual numbers depend entirely on attribution method (16/26/308/324)." Both are consistent if the 559 figure is the union over all attribution methods at a later snapshot — use `CASE_ORG_001_INFRASTRUCTURE.md` as canonical, but quote with the snapshot date.
- **Key wallet roles:**
  - **CEX origins:** Coinbase Hot 1 `0x503828976d22510aad0201ac7ec88293211d23da`, Binance Hot 1 `0x28c6c06298d514db089934071355e5743bf21d60`
  - **Mainnet buffer:** `0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d`, `0x4976a4a02f38326660d17bf34b431dc6e2eb2327`, `0xf3d63166f0ca56c3c1a3508fce03ff0cf3fb691e`, `0x81f91aca8c05b3eefebc00171139afefac17c9a6`
  - **Mainnet operations:** Central Treasury `0x4c968f6beecf1906710b08e8b472b8ba6e75f957`, Whale Trader `0xf70da97812cb96acdf810712aa562db8dfa3dbef`, MEV Bot `0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1`, Revenue Collector `0x391e7c679d29bd940d63be94ad22a25d25b5a604`, WETH Wrapping `0x5e0f8e7337c8955d2124b8e85ca74af884b3e124`, Revenue Source `0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390`
  - **L2 core:** Treasury `0xf186cb00e49e18491db5783ff04fae3818102ff7`, Operator `0xe93d64f3fbc352131e79fc5578cbe44b66697f86`, Operator 2 `0xfd51e33d44b376ef346d24a130a51035db09c1dc`, Cashout `0xc6962004f452be9203591991d15f6b388e09e8d0`, Gas Station `0x8c826f795466e39acbff1bb4eeeb759609377ba1`, Vault Branch `0x360e68faccca8ca495c1b759fd9eee466db9fb32`
  - **L2 exit:** CEX Exit / Shadow 1 `0x01989c93890aed05a63d179b03424997075b6acf` (vanity-spoofed), Laundry `0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb`, LP Staging / Shadow 2 `0x96daa0b8a5499ea9323421ed0cda06b345caab73` (vanity-spoofed), LP Companion `0x27920e8039d2b6e93e36f5d5f53b998e2e631a70`, DeFi Exit Channel `0x51c72848c68a965f66fa7a88855f9f7784502a7f`

### org_002 — tx.origin Trap Campaign (Base)
- **ID:** `org_002`
- **Description:** Single-campaign tx.origin pattern, 350 deployers each deploying 1 contract from a shared template (`tx.origin conditional at 0x314`). 357 wallets total (2 treasuries + 355 disposable deployers).
- **Primary case file:** None — referenced only in `surveillance/data/cases/CROSS_ORG_ANALYSIS_20260322.md` and `surveillance/data/cases/COMPETITION_ANALYSIS_20260322.md`
- **Status:** CONFIRMED (no dedicated case file yet — gap flagged for Phase 5 deliverable)
- **Key wallet roles:** 2 treasuries (addresses not enumerated in current case files), 355 single-use deployers

### org_003 — Ghost Fee-Skimmer Network (Base)
- **ID:** `org_003`
- **Description:** 6 disposable wallets with INVISIBLE FUNDING (no Alchemy traces on any transfer category). 727 combined victims, 81–85% victim overlap proves single operator. Identical bytecode pattern at identical offsets (SHA3-keyed obfuscated fee-on-transfer).
- **Primary case file:** `surveillance/data/cases/CASE_ORG_003_GHOST_NETWORK.md`
- **Status:** CONFIRMED — last activity 2026-03-25
- **Key wallet roles:**
  - 6 deployers: `0x392c564a28d6d87d326e8a385c764355e130418d`, `0x9f5db1b0436815de051b86a732c3d8ad8877bbef`, `0xadb085d8279bf7afad7599d3b8685547b19b2798`, `0x8f007f3e4f83a57c34dc4bf19237423665d28dd8`, `0x571ba99571f588d8238cb56a244de1c44e1d1da7`, `0x888a491ed0a09c9325507aaccead28a099a2104f`
  - 6 contracts: `0x0697a11c537829...`, `0x201b32f4a87c21...`, `0x7709a1e3fe44a0...`, `0xf2b2b76e43961872c3c55823b843c52dd5cb149c`, `0xa80899d4ce98a4...`, `0xc3bc6e376159b2...`

### org_004 — Infrastructure-Scale Operator (referenced, no case file)
- **ID:** `org_004`
- **Description:** Named in `claude.md` Current Priority Items #12 ("Investigate org_004 (0xbaed383e) — next organizational mapping target"). Surfaced in 2026-04-25 by `surveillance/infrastructure_operator_detector.py` as a 210-deployer / 1,492-contract / 63.1% adversarial-ratio infrastructure operator.
- **Primary case file:** None — gap. Only in `claude.md` priority list and `infrastructure_operator_candidates` table.
- **Status:** UNDER_INVESTIGATION — case file pending
- **Key wallet:** `0xbaed383ede0e5d9d72430661f3285daa77e9439f`

### org_005, org_006, org_007 — Tier-1 Cluster Promotions (DB-only)
- **ID:** `org_005`, `org_006`, `org_007`
- **Description:** Three Tier-1 clusters promoted to `org_wallets` on 2026-04-23 via `scripts/tier1_promotion.sql`. 40 wallets total: org_005 (9), org_006 (16), org_007 (15).
- **Primary case file:** None — gap. Only `org_wallets` table entries.
- **Status:** PROMOTED (DB-only) — case files pending

### Entity_005 — "The Architect" (Arbitrum R&D operator)
- **ID:** `Entity_005`
- **Description:** Most sophisticated trap developer in corpus. Arbitrum-exclusive R&D in 7-contract sessions, building toward 4-mechanism weapon (SELFDESTRUCT + DELEGATECALL + TIMESTAMP + CALLER). Pre-production as of last case-file update — no Base deployment yet.
- **Primary case file:** `surveillance/data/cases/CASE_ENTITY_005_THE_ARCHITECT.md`
- **Status:** UNDER_INVESTIGATION — R&D phase
- **Key wallet roles:**
  - Primary: `0x9209c9f7dcb61937f1ec8160c22c0b2365079474`
  - Funder: `0x151b381058f91cf871e7ea1ee83c45326f61e96d`
  - Behavioral matches (watchlist HIGH): `0x4cfe37d2` (0.799), `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` (0.742), `0x30e88ee4c417f126aacf6a4c3cd641424203fe6d` (0.711), `0x7930e1380157596ff82595d502083bf604eda922` (0.710), `0xd28e6a7ad806e85bd0544ed443d25e48f52c06c3` (0.719)

### X402 Drainer Operation — 7 rogue Permit2 facilitators
- **ID:** X402_DRAINER (no canonical org_xxx ID assigned)
- **Description:** 7 EOAs misusing x402/Permit2 settlement infrastructure as a drain vector. Victims granted MAX_UINT160 / MAX_UINT48 Permit2 allowances and were swept via `Permit2.transferFrom(from, to, amount, token)` where `tx.from == to` (self-settlement). ~$2.3M real victim extraction + ~$1.6M pass-through laundering (`CORRECTIONS.md` 2026-04-15 update).
- **Primary case file:** `surveillance/data/cases/CASE_X402_DRAINER_OPERATION.md`
- **Supporting:** `reports/case_CE5E_drainer_operation.md` (CE5E lifetime: 68 drains, $929K USDC, 67 unique victims, Arbitrum)
- **Status:** CONFIRMED — active
- **Facilitators:**
  - **CE5E** `0xce5ec7336f863931fda2ee3e4b9dad99fcc53c91` (Arbitrum)
  - **E717** `0xe7176831c898d585cd999bcee9984a7fa9a6be96` (Arbitrum)
  - **881E** `0x881e7c4c90f2d7f013558caf4feca330c327e476` (Arbitrum)
  - **A7B9** `0xa7b9874d15742358fb455dd56f97c6d19ad74f5c` (Base)
  - **E3B2** `0xe3b205da6d47989538f03553bc394d941677ffd3` (Base)
  - **D270** `0xd27047fe310178316b3acc4746e2a30823bb9186` (Optimism)
  - **F71C** `0xf71c98b3025baa6d1c15148429a9f2f1ce952e8c` (Optimism)
- **Note:** `0x785ce546ed429559b95895cb4a07874bf8ed329c` was originally listed as a $256K victim of E3B2; reclassified 2026-04-13 as a controlled intermediary funded by E717 with 1,406 ETH, distributing $9.8M to address-poisoning collectors.

### Coffee Fleet — `0xc0ffeefeed8b9d27` deployer + 84 c0ffee vanity bots
- **ID:** COFFEE_FLEET (no canonical org_xxx ID assigned)
- **Description:** Confirmed dual-role: trap deployer AND self-scanning bot fleet. 100% c0ffee-fleet-on-c0ffee-fleet at last case-file update (2026-04-08). Single-operator hypothesis leading; closed adversary ecosystem hypothesis not ruled out.
- **Primary case file:** `surveillance/data/cases/CASE_COFFEE_FLEET_0xc0ffeefeed8b.md`
- **Status:** UNDER_INVESTIGATION
- **Key wallet roles:**
  - Deployer: `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e` (60 contracts at case-file date, 251 at 2026-04-25)
  - Funder: `0x7c8b9874f7be10ba196d3bb6fe1f45556c0bc1b5`
  - 84 c0ffee victim bots (all classified `L2_OPTIMIZED_SCANNER`) — see `bot_strategies` table
- **Open question (2026-04-25):** Recent comparison found `0x604be06b9f6b6663f78e755db0c5965eb2337e3d` shares 100% bot-victim overlap with this fleet. Either (a) c0ffee scanners expanded prey list to A's traps (sequential discovery), or (b) `0x604be06b` is the same operator's second deployer wallet (multi-wallet operation). Data does not currently distinguish.

### Dragon — Pre-staged Trap Inventory (`0x2e20b2`)
- **ID:** DRAGON
- **Description:** 2,077 ERC-20 tokens deployed in ~24 hours on Base, 169 pre-approved to PancakeSwap V3 Router. Entire inventory dormant — zero external victim interactions. Largest pre-staged trap inventory in corpus. Activation requires single `addLiquidity()` call per contract.
- **Primary case file:** `surveillance/data/cases/CASE_DRAGON_0x2e20b2.md`
- **Status:** DORMANT — pre-positioned, not activated
- **Key wallet roles:**
  - Deployer: `0x2e20b26172a8625c33097288075920a6210a8233` (currently empty balance — refunding required for activation)
  - PancakeSwap V3 Router (legitimate): `0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24`

### Routing Parasite — `0xd4624228` (SXAI/WETH on Base)
- **ID:** EXTRACTION_003 (per `extraction_events.event_id`)
- **Description:** 14,752-byte multi-interface contract impersonating ERC-20 + Uniswap Universal Router + Uniswap Pool + NFT receiver. 2,910 victims, ~100.56 WETH (~$211K) extracted via 100% sell-side WETH retention. 71% repeat victims. Cantina rejected the bug-bounty submission 2026-03-25 ("the code did what it was supposed to do"). Dormant since 2026-03-26.
- **Primary case file:** `surveillance/data/cases/PARASITE_ARCHITECTURE_0xd4624228.md` (architecture-level, more detailed)
- **Supporting:** `surveillance/data/cases/CASE_0xd4624228_base_20260324_011320.md`, `surveillance/data/cases/UNISWAP_BUGBOUNTY_SUBMISSION.md`, `surveillance/data/cases/TRUST_LAYER_EXPLOITATION_20260324.md`
- **Status:** CONFIRMED — dormant since 2026-03-26
- **Key wallet roles:**
  - Contract: `0xd4624228cce5baa0814c9e7f666a8a2c83b6f159`
  - Deployer: `0xe8e0c4883d7196a7de87a6489f6da58212dbe813` (single-shot burner)
  - LP Pool: `0x7609153350cd0184c5df525d58490edf3bacef3b`
  - Token (SXAI): `0xea6b6bC260ED8241190C277d2fe7718Ea6CbF667`
  - Collection wallets: `0xd462be33c46d84a0ce702103336f2fc290dcf159`, `0xe502b1568aba07040a4580717e3399297067c50e`, `0x07bd23d6ae11e61450ea74c4d96e21f3946eacb6`

### Whale's R&D Lab — Bot_A network (`0x84792c2a` + $5.75M MEV vault)
- **ID:** BOT_A_NETWORK
- **Description:** Bot `0x84792c2a` runs a custom proprietary spray bot (selector `2f139e4f`, used by no other address in the corpus). $4,412 cumulative gas burn / mostly-reverting transactions / $0 visible revenue. The bot's funder network connects to a 2,739.89 ETH (~$5.75M) MEV vault `0xa45b5130...` — the bot is a peripheral experiment of a much larger MEV operation. Anchors the [Tuition Extraction Markets](lexicon.md#tuition-extraction-markets) lexicon entry.
- **Primary case file:** `surveillance/data/cases/BOT_A_NETWORK_INVESTIGATION_20260323.md`
- **Supporting:** `surveillance/data/cases/BOT_INVESTIGATION_0x84792c2a_20260322.md`
- **Status:** UNDER_INVESTIGATION
- **Key wallet roles:**
  - Bot: `0x84792c2a` (subject)
  - MEV Vault: `0xa45b5130f36cdca45667738e2a258ab09f4a5f7f` (2,739.89 ETH, 22,628-byte contract, 2 outbound txs ever)
  - Primary funder: `0x9ebed688...`
  - Secondary funder: `0x260790b1...`

### Zero-Day (`0xa7fea69e`) — first self-test detection catch
- **ID:** ZERODAY_0xa7fea69e
- **Description:** First validated catch by self-test detection system. R&D-to-armed transition in 49 minutes. Same SHA3/SLOAD/JUMPI fee-skimming architecture as org_003, but independent operator (different funding, different deployer).
- **Primary case file:** `surveillance/data/cases/CASE_ZERODAY_0xa7fea69e.md`
- **Status:** CONFIRMED
- **Key wallet roles:**
  - Contract: `0xa7fea69e9cf742ea9e7ef94779752dd451a16af7`
  - Deployer: `0x760640f4aa7309efd3f08c32ead01f0097667c78`
  - Funder: `0x40f4eef12643644cda2a5b9ca5cbba51345045ce` (0.026 ETH)

### Pristine-Solo Operators (4 watchlist HIGH promotions, 2026-04-25)
- **ID:** PRISTINE_SOLO (no canonical org_xxx ID)
- **Description:** Long-mainnet-history wallets surfacing as small-fleet (1–5 contract) trap operators on L2. Lexicon: [Pristine Solo Operator](lexicon.md#pristine-solo-operator). Detector: `surveillance/pristine_solo_detector.py`.
- **Primary case file:** None — DB-only via `pristine_solo_candidates` table. Lexicon entry references the four addresses.
- **Status:** CONFIRMED (4 promoted, 13 candidates)
- **Key wallets (HIGH watchlist):**
  - `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` (gap 2,498d, deployer of `0x752c5a95` harvester)
  - `0xa2a01b4a68575280a2de45178e289da717bedb6f` (gap 2,314d, Arbitrum, fleet 4)
  - `0x147b8869655bc09f226955cc676ff78efe240ca8` (gap 1,777d, Base, fleet 4)
  - `0xf6c99cec5bd639316a19d2f56afc14bd046d3a90` (gap ≈1,748d, Base, fleet 2)

### `0x752c5a95` Pre-Drain Harvester (Base)
- **ID:** No canonical ID — investigation only
- **Description:** Confirmed-tier contract harvesting Permit2 approvals from 1,898+ victims (as of 2026-04-24) without firing a sweep. Deployer is `0x80b12bd0` (pristine-solo, 2019 mainnet vintage). Bytecode flags `has_asymmetric_transfer=1`, `has_unusual_fee_structure=1`. The harvester is the largest active confirmed-tier approval pool in the corpus.
- **Primary case file:** None — gap. Only investigated via session scripts (`scripts/investigate_0x752c5a95.py`).
- **Status:** UNDER_INVESTIGATION — pre-drain accumulation, no sweep yet

### `0xe69f81b8` — High-Volume Bridge User
- **ID:** No canonical org assigned
- **Description:** EOA bridging 49,000 ETH (~$147M) to L1 via the canonical Base bridge over 7 days (April 7–14). One of the most active L2-to-L1 bridge users in the Base ecosystem. Coordinated during western sleep hours.
- **Primary case file:** None — only `CORRECTIONS.md` entries 2026-04-07 (19,000 ETH) and 2026-04-14 (additional 30,000 ETH).
- **Status:** TRACKED (watchlist HIGH, `entity_type: high_value_bridge_user`)
- **Key wallet:** `0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb`

### Infrastructure-Scale Operator candidates (2026-04-25 detector run, 12 candidates)
- **ID:** No canonical IDs — surfaced by detector only
- **Description:** Funder addresses with ≥200 fanout, ≥10% adversarial ratio, ≥50% disposable rate. Lexicon: [Infrastructure-Scale Operator](lexicon.md#infrastructure-scale-operator).
- **Primary case file:** None — DB-only via `infrastructure_operator_candidates` table.
- **Status:** UNDER_INVESTIGATION (12 candidates surfaced, 0 promoted to entity status)
- **Note on attribution:** Two of the 12 candidates are already classified org_001 infrastructure: `0xf70da97812cb96acdf810712aa562db8dfa3dbef` (Whale Trader, mainnet) and `0x8c826f795466e39acbff1bb4eeeb759609377ba1` (L2 Gas Station). A third is `0xbaed383ede0e5d9d72430661f3285daa77e9439f` = org_004. The remaining 9 are not yet attributed to existing organizations.

---

## Section 2 — Address Index

Flat alphabetical (lowercase). Use Ctrl-F. Format: `address  primary_classification  primary_file [+ supporting]`. The "single-file artifact" tag indicates the address only appears in one auto-generated `CASE_0x*` file with no further attribution.

### CEX-related (Tier 1)
- `0x28c6c06298d514db089934071355e5743bf21d60` — Binance Hot Wallet 1 (org_001 origin) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x503828976d22510aad0201ac7ec88293211d23da` — Coinbase Hot Wallet 1 (org_001 origin) — `CASE_ORG_001_INFRASTRUCTURE.md`

### org_001 mainnet hierarchy
- `0x4976a4a02f38326660d17bf34b431dc6e2eb2327` — Binance Buffer (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x4c968f6beecf1906710b08e8b472b8ba6e75f957` — Central Treasury (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1` — MEV Bot (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x5e0f8e7337c8955d2124b8e85ca74af884b3e124` — WETH Wrapping Station (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d` — Buffer Wallet (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x81f91aca8c05b3eefebc00171139afefac17c9a6` — Staging Wallet (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x391e7c679d29bd940d63be94ad22a25d25b5a604` — Revenue Collector (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390` — Revenue Source (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0xf3d63166f0ca56c3c1a3508fce03ff0cf3fb691e` — Intermediate (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0xf70da97812cb96acdf810712aa562db8dfa3dbef` — **Whale Trader (org_001 — primary funding channel, 68% of deployments at 2026-03-28)** — `CASE_ORG_001_INFRASTRUCTURE.md`. Also surfaced by `infrastructure_operator_detector` 2026-04-25 (2,684-deployer fanout).
- `0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae` — LI.FI Diamond (3rd party, used by org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`

### org_001 L2 infrastructure (Arbitrum + Base)
- `0x01989c93890aed05a63d179b03424997075b6acf` — Shadow Wallet 1 / CEX Exit (vanity-spoofed) (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x27920e8039d2b6e93e36f5d5f53b998e2e631a70` — LP Companion (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x360e68faccca8ca495c1b759fd9eee466db9fb32` — Vault / Treasury Branch (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x51c72848c68a965f66fa7a88855f9f7784502a7f` — DeFi Exit Channel (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x8c826f795466e39acbff1bb4eeeb759609377ba1` — **Gas Station (org_001 — Coinbase-funded, 503 deployers, 2,444 contracts)** — `CASE_ORG_001_INFRASTRUCTURE.md`. Also surfaced by `infrastructure_operator_detector` 2026-04-25 (743-deployer fanout).
- `0x96daa0b8a5499ea9323421ed0cda06b345caab73` — Shadow Wallet 2 / LP Staging (vanity-spoofed) (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0xc6962004f452be9203591991d15f6b388e09e8d0` — Cashout (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0xe93d64f3fbc352131e79fc5578cbe44b66697f86` — **Operator (org_001 — 26+ contracts on Arbitrum)** — `CASE_ORG_001_INFRASTRUCTURE.md` + 7 auto-generated `CASE_0x*_arbitrum_*.md` files for individual contracts
- `0xf186cb00e49e18491db5783ff04fae3818102ff7` — **Treasury (org_001 — L2 hub)** — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0xfd51e33d44b376ef346d24a130a51035db09c1dc` — Operator 2 (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb` — Laundry / WBTC swapper (org_001) — `CASE_ORG_001_INFRASTRUCTURE.md`

### org_001 — operator-deployed contracts (auto-generated case files)
- `0x3b8b8e550997541816882c778ae0ac031e86b5a8` — contract by org_001 Operator — `CASE_0x3b8b8e55_arbitrum_20260325_022748.md`
- `0x3e6800980a97038ce5e746cde46c49d45e4966de` — contract by org_001 Operator — `CASE_0x3e680098_arbitrum_20260325_022749.md`
- `0x74b9a8351bd725ca3edd654c9728873b8c6f051e` — Upgradeable Proxy Trap by org_001 Operator — `CASE_0x74b9a835_arbitrum_20260325_022749.md`
- `0x79a2f71187dc9fd9b173781e6dd4ff9960f6f61b` — Upgradeable Proxy Trap by org_001 Operator — `CASE_0x79a2f711_arbitrum_20260325_022748.md`
- `0xc8e6a328d094609a97024978657c29920cabf7c3` — contract by org_001 Operator — `CASE_0xc8e6a328_arbitrum_20260325_022748.md`
- `0xc8f28b043feb244c7e9df76af45a68271013a335` — contract by org_001 Operator — `CASE_0xc8f28b04_arbitrum_20260325_022749.md`

### org_003 — Ghost Fee-Skimmer Network
- `0x392c564a28d6d87d326e8a385c764355e130418d` — Deployer 1 (org_003) — `CASE_ORG_003_GHOST_NETWORK.md`
- `0x571ba99571f588d8238cb56a244de1c44e1d1da7` — Deployer 5 (org_003) — `CASE_ORG_003_GHOST_NETWORK.md`
- `0x888a491ed0a09c9325507aaccead28a099a2104f` — Deployer 6 (org_003) — `CASE_ORG_003_GHOST_NETWORK.md`
- `0x8f007f3e4f83a57c34dc4bf19237423665d28dd8` — Deployer 4 (org_003) — `CASE_ORG_003_GHOST_NETWORK.md`
- `0x9f5db1b0436815de051b86a732c3d8ad8877bbef` — Deployer 2 (org_003) — `CASE_ORG_003_GHOST_NETWORK.md`
- `0xadb085d8279bf7afad7599d3b8685547b19b2798` — Deployer 3 (org_003) — `CASE_ORG_003_GHOST_NETWORK.md`
- `0xf2b2b76e43961872c3c55823b843c52dd5cb149c` — Contract 4 (org_003 — fee-skimmer) — `CASE_ORG_003_GHOST_NETWORK.md` (FP retraction subject in Correction #17)

### org_004
- `0xbaed383ede0e5d9d72430661f3285daa77e9439f` — org_004 (referenced in `claude.md` priority #12) — no case file yet

### Entity_005 — The Architect
- `0x9209c9f7dcb61937f1ec8160c22c0b2365079474` — Architect primary deployer (CRITICAL watchlist) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0x151b381058f91cf871e7ea1ee83c45326f61e96d` — Architect funder (CRITICAL watchlist) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0x4cfe37d2` (full: `0x4cfe37d21a5a8a1d74e4840426d644a1b50dc328`) — Architect alternate (HIGH, 0.799) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` — Architect alternate (HIGH, 0.742) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0x30e88ee4c417f126aacf6a4c3cd641424203fe6d` — Architect candidate (HIGH, 0.711) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0x7930e1380157596ff82595d502083bf604eda922` — Architect candidate (HIGH, 0.710) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0xd28e6a7ad806e85bd0544ed443d25e48f52c06c3` — Architect candidate (HIGH, 0.719) — `CASE_ENTITY_005_THE_ARCHITECT.md`

### X402 Drainer Operation (7 facilitators)
- `0x881e7c4c90f2d7f013558caf4feca330c327e476` — SUSPECT-881E facilitator (Arbitrum) — `CASE_X402_DRAINER_OPERATION.md`
- `0xa7b9874d15742358fb455dd56f97c6d19ad74f5c` — DRAINER-A7B9 (Base) — `CASE_X402_DRAINER_OPERATION.md`
- `0xce5ec7336f863931fda2ee3e4b9dad99fcc53c91` — DRAINER-CE5E (Arbitrum, $929K USDC) — `CASE_X402_DRAINER_OPERATION.md` + `reports/case_CE5E_drainer_operation.md`
- `0xd27047fe310178316b3acc4746e2a30823bb9186` — DRAINER-D270 (Optimism, OP tokens) — `CASE_X402_DRAINER_OPERATION.md`
- `0xe3b205da6d47989538f03553bc394d941677ffd3` — DRAINER-E3B2 (Base) — `CASE_X402_DRAINER_OPERATION.md`
- `0xe7176831c898d585cd999bcee9984a7fa9a6be96` — DRAINER-E717 (Arbitrum, 125 ETH) — `CASE_X402_DRAINER_OPERATION.md`
- `0xf71c98b3025baa6d1c15148429a9f2f1ce952e8c` — DRAINER-F71C (Optimism, funded by D270) — `CASE_X402_DRAINER_OPERATION.md`
- `0x785ce546ed429559b95895cb4a07874bf8ed329c` — Controlled intermediary (NOT a victim — reclassified 2026-04-13) — `CASE_X402_DRAINER_OPERATION.md`

### Coffee Fleet
- `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e` — Coffee Fleet deployer — `CASE_COFFEE_FLEET_0xc0ffeefeed8b.md`
- `0x7c8b9874f7be10ba196d3bb6fe1f45556c0bc1b5` — Coffee Fleet funder — `CASE_COFFEE_FLEET_0xc0ffeefeed8b.md`
- `0x604be06b9f6b6663f78e755db0c5965eb2337e3d` — Open question: independent peer of Coffee Fleet OR same operator's second wallet — watchlist HIGH (added 2026-04-25)

### Routing Parasite (`0xd4624228`)
- `0xd4624228cce5baa0814c9e7f666a8a2c83b6f159` — Routing parasite contract — `PARASITE_ARCHITECTURE_0xd4624228.md` + `CASE_0xd4624228_base_20260324_011320.md` + `UNISWAP_BUGBOUNTY_SUBMISSION.md`
- `0xe8e0c4883d7196a7de87a6489f6da58212dbe813` — Parasite deployer (single-shot burner) — same files
- `0x7609153350cd0184c5df525d58490edf3bacef3b` — SXAI/WETH LP pool — `PARASITE_ARCHITECTURE_0xd4624228.md`
- `0xea6b6bC260ED8241190C277d2fe7718Ea6CbF667` — SXAI token — `PARASITE_ARCHITECTURE_0xd4624228.md`
- `0xd462be33c46d84a0ce702103336f2fc290dcf159` — Collection wallet 1 — `PARASITE_ARCHITECTURE_0xd4624228.md`
- `0xe502b1568aba07040a4580717e3399297067c50e` — Collection wallet 2 — `PARASITE_ARCHITECTURE_0xd4624228.md`
- `0x07bd23d6ae11e61450ea74c4d96e21f3946eacb6` — Collection wallet 3 — `PARASITE_ARCHITECTURE_0xd4624228.md`

### Bot_A network
- `0x84792c2a` — Bot_A spray bot (anchors Tuition Extraction Markets entry) — `BOT_A_NETWORK_INVESTIGATION_20260323.md` + `BOT_INVESTIGATION_0x84792c2a_20260322.md`
- `0xa45b5130f36cdca45667738e2a258ab09f4a5f7f` — MEV Vault (2,739.89 ETH) — `BOT_A_NETWORK_INVESTIGATION_20260323.md`

### Dragon
- `0x2e20b26172a8625c33097288075920a6210a8233` — Dragon deployer (2,077-contract dormant inventory) — `CASE_DRAGON_0x2e20b2.md`
- `0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24` — PancakeSwap V3 Router (legitimate, used by Dragon) — `CASE_DRAGON_0x2e20b2.md`

### Zero-Day
- `0xa7fea69e9cf742ea9e7ef94779752dd451a16af7` — Zero-Day contract — `CASE_ZERODAY_0xa7fea69e.md`
- `0x760640f4aa7309efd3f08c32ead01f0097667c78` — Zero-Day deployer — `CASE_ZERODAY_0xa7fea69e.md`
- `0x40f4eef12643644cda2a5b9ca5cbba51345045ce` — Zero-Day funder — `CASE_ZERODAY_0xa7fea69e.md`

### Pristine-Solo (watchlist HIGH, 2026-04-25)
- `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` — pristine-solo, 2,498d gap, deployer of `0x752c5a95` — lexicon entry, no case file
- `0xa2a01b4a68575280a2de45178e289da717bedb6f` — pristine-solo, 2,314d gap, Arbitrum — lexicon entry, no case file
- `0x147b8869655bc09f226955cc676ff78efe240ca8` — pristine-solo, 1,777d gap, Base — lexicon entry, no case file
- `0xf6c99cec5bd639316a19d2f56afc14bd046d3a90` — pristine-solo, ≈1,748d gap, Base — lexicon entry, no case file
- `0x752c5a95d202972e124390f30a50154409d3c858` — pre-drain harvester (1,898+ approvals, 0 sweeps) — no case file, only `scripts/investigate_0x752c5a95.py`

### Pattern A candidates (reputation-building sacrifices, 4 of 5,810)
- `0x614737b68f3b7fa65a165c4057c2412d4a3a9271` — Pattern A candidate — `reports/reputation_sacrifice_candidates.md`
- `0x5eb7a6583f6386d040f83429a9d46138cb9e8ea3` — Pattern A candidate — `reports/reputation_sacrifice_candidates.md`
- `0x809088835c4cf6a1af0de599da08ff355a9723db` — Pattern A candidate (Arbitrum) — `reports/reputation_sacrifice_candidates.md`
- `0x021868f2e3d49c059ef52b539aaa933b437e0321` — Pattern A candidate — `reports/reputation_sacrifice_candidates.md`

### Pattern C / Pattern D candidates
- `0x4885631c7335290adcdc4b6b95f97549f5a40edd` — flagged by both Pattern C (CEX-laundered) and Pattern D (cross-chain) — `reports/cex_laundered_funding.md`, `reports/cross_chain_import_candidates.md`, `reports/behavioral_laundering_detection_scope.md`
- `0x6dc136bcac04646d8d342599a704fffe9861af56` — Pattern C candidate (CEX-funded, all-suspected fleet) — `reports/cex_laundered_funding.md`
- `0xb87e28fc6086fad8fe228aac3d3e19058e69f828` — Pattern C candidate — `reports/cex_laundered_funding.md`
- `0x561d79e961c4dd7bbaf078e6c9753c764e084d77` — Pattern C candidate — `reports/cex_laundered_funding.md`
- `0x7fd9a5104f1cb261a7215f950c9fa7eac06d60d0` — Pattern D candidate (longest mainnet gap: 8.8 years) — `reports/cross_chain_import_candidates.md`

### Other
- `0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb` — High-volume bridge user (49,000 ETH bridged April 7–14, watchlist HIGH) — `CORRECTIONS.md` 2026-04-07 + 2026-04-14
- `0xb15e7a89e71b8468c23eb330f837caf0f2ff7628` — Multi-Mechanism Trap (independent operator) — `CASE_0xb15e7a89_arbitrum_20260322_040329.md`
- `0x01989c93890aed05cbcda4e62eec1b2eb4c55b1b` — LP_POOL_2 (the **legitimate** address being vanity-spoofed by org_001's Shadow Wallet 1, which has the same first 8 bytes `0x01989c93890aed05`) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x12577cf0d8a07363224d6909c54c056a183e13b3` — Fee-Skimming Token — `CASE_0x12577cf0_base_20260322_040329.md` (single-file artifact)
- `0x08155ec0cf641720d1e1f66fcaeda8b29f9bbb17` — deployer of `0x12577cf0` (3 contracts total) — `CASE_0x12577cf0_base_20260322_040329.md`
- `0x9da33ece6fdf36ecf99e10dbd6ecd0cb529e257e` — Multi-Mechanism Trap — `CASE_0x9da33ece_base_20260322_040303.md` (single-file artifact)
- `0xb5ae0b6cb72dcf5180ac2a4c3b77bebef5b42a81` — deployer of `0x9da33ece` (8-contract Base fleet, top by victim-bot count in 2026-04-25 pairwise probe with 192 bots) — `CASE_0x9da33ece_base_20260322_040303.md`
- `0x1d13a5aefd4d3a0f466c0058526d8bf11d88502a` — deployer of `0xb15e7a89` (Multi-Mechanism Trap, 8 contracts, Arbitrum) — `CASE_0xb15e7a89_arbitrum_20260322_040329.md`
- `0x666521000c595a632fb3e99f392b12e937b77586` — high-productivity solo operator (watchlist HIGH 2026-04-23) — no case file, lexicon-only
- `0xefef185e2c89bbede21a1c41427bdf1332eca392` — high-confirmation-ratio operator (watchlist HIGH 2026-04-23) — no case file, lexicon-only

The remaining ~80 distinct addresses extracted from cases/ + reports/ are either victim/bot addresses without role attribution or appear in single auto-generated `CASE_0x*` files only. They are not enumerated here unless they have a documented role.

---

## Section 3 — Patterns and Hypotheses

### Documented (active framework concepts)
- **Stored Potential** — DOCUMENTED — `docs/lexicon.md#stored-potential`
- **Adversarial Topology Framework** (5 primitives: position, permissions, trust bindings, mutability, observation capability) — DOCUMENTED — `docs/lexicon.md#adversarial-topology` + `claude.md` §Adversarial Topology Framework
- **Compositional Harm** — DOCUMENTED — `docs/lexicon.md#compositional-harm`
- **Cross-Domain Compositional Harm** — DOCUMENTED 2026-04-25 — `docs/lexicon.md#cross-domain-compositional-harm`. Anchor: Vercel/Context.ai breach (no on-chain evidence yet, off-chain case study).
- **Trust Amplification Factor** — DOCUMENTED with methodological caveat — `docs/lexicon.md#trust-amplification-factor`. **Two contradictory retractions of the 14.2× anchor figure exist** (`CORRECTIONS.md` 2026-04-02 vs `reports/correction_log.md` Correction #17 2026-04-25). Resolution open.
- **Camouflage Ratio** — DOCUMENTED with methodological caveat (cluster-dominance impact) — `docs/lexicon.md#camouflage-ratio`. Original 14.2× claim retired (`CORRECTIONS.md`); equilibrium framing requires top-12-excluded re-run.
- **Behavioral Laundering** (Patterns A–F) — DOCUMENTED — `docs/lexicon.md#behavioral-laundering`
  - Pattern A — Reputation-Building Sacrifices — DOCUMENTED — 4 candidates as of 2026-04-18
  - Pattern B — Temporal Pattern Normalization — DOCUMENTED — 0 candidates (corpus too young)
  - Pattern C — Funding Chain Laundering — DOCUMENTED — 4 relaxed candidates
  - Pattern D — Cross-Chain Reputation Import — DOCUMENTED — strongest validated (54 of 100)
  - Pattern E — Fake Legitimate Projects — METHODOLOGY-ONLY — not yet scanned
  - Pattern F — Advisor-Parasite Pattern — DOCUMENTED — 0 candidates (corpus too young)
- **Pristine Solo Operator** — DOCUMENTED 2026-04-25 — `docs/lexicon.md#pristine-solo-operator`. Detector: `surveillance/pristine_solo_detector.py`.
- **Infrastructure-Scale Operator** — DOCUMENTED 2026-04-25 — `docs/lexicon.md#infrastructure-scale-operator`. Detector: `surveillance/infrastructure_operator_detector.py`.
- **Tuition Extraction Markets** — DOCUMENTED 2026-04-25 — `docs/lexicon.md#tuition-extraction-markets`. Anchor: Bot_A.
- **Pooled Custody Amplification** — DOCUMENTED — `docs/lexicon.md#pooled-custody-amplification`
- **Verification-Path Trust Failure** — DOCUMENTED — `docs/lexicon.md#verification-path-trust-failure`
- **Configuration-Level Vulnerability** — DOCUMENTED — `docs/lexicon.md#configuration-level-vulnerability`
- **Operational Layer Attack** — DOCUMENTED — `docs/lexicon.md#operational-layer-attack`
- **Strategy Lifecycle** (EARLY → ARMS_RACE → WEAPONIZED → SATURATED) — DOCUMENTED — `docs/lexicon.md#strategy-lifecycle`
- **Publishing-Induced Recursive Evasion** — DOCUMENTED — `docs/lexicon.md#publishing-induced-recursive-evasion`
- **Static vs Dynamic Behavior** — DOCUMENTED — `docs/lexicon.md#static-vs-dynamic-behavior`
- **Cost-Habituation Asymmetry** / **Micro-Cost Habituation** — DOCUMENTED — `docs/lexicon.md`
- **Cognitive Load Concentration** — DOCUMENTED — `docs/lexicon.md#cognitive-load-concentration`
- **The Proofreading Trap** — DOCUMENTED — `docs/lexicon.md#the-proofreading-trap`
- **The Self-Cannibalizing System** — DOCUMENTED — `docs/lexicon.md#the-self-cannibalizing-system`
- **Victim-to-Predator Pipeline** — DOCUMENTED — `docs/lexicon.md#victim-to-predator-pipeline`
- **Accountability-as-Load-Bearing** — DOCUMENTED — `docs/lexicon.md#accountability-as-load-bearing`
- **External Accountability Infrastructure** — DOCUMENTED — `docs/lexicon.md#external-accountability-infrastructure`
- **The Detection Gap as Product** — DOCUMENTED — `docs/lexicon.md#the-detection-gap-as-product`
- **Observational Edge Non-Convertibility** — DOCUMENTED — `docs/lexicon.md#observational-edge-non-convertibility`
- **Intelligence-as-Compounding-Asset** — DOCUMENTED — `docs/lexicon.md#intelligence-as-compounding-asset`
- **The Bug-Bounty Structural Gap** — DOCUMENTED — `docs/lexicon.md#the-bug-bounty-structural-gap`
- **Epistemic Tier Classification** (Tier A / Tier B / Tier C) — DOCUMENTED — `docs/lexicon.md#epistemic-tier-classification`

### Open hypotheses (not yet documented, not yet retired)
- **Coffee Fleet Single-Operator Hypothesis** — OPEN — `CASE_COFFEE_FLEET_0xc0ffeefeed8b.md`. 100% c0ffee-on-c0ffee victim overlap is consistent with single operator running both sides; data does not currently distinguish from closed adversary ecosystem. Recent 2026-04-25 finding: `0x604be06b` shares 100% bot overlap with c0ffeefeed — adds either expanded-prey (sequential discovery) or multi-wallet (single operator) interpretation.

### Retired or anecdotal (not lexicon-worthy)
- **Adversarial Co-Tenancy** — ANECDOTE — pairwise probe across 903 operator pairs surfaced only the A/B (`0x604be06b` / `0xc0ffeefeed`) pair at meaningful overlap. Held out of lexicon per Correction #16 (`reports/correction_log.md`). **Note 2026-04-25:** the existing `CASE_COFFEE_FLEET_*` documents the single-operator hypothesis the recent investigation reframed as anecdote — there is residual conceptual disagreement between the two readings.
- **Prey-Driven Equilibrium Calibration** — RETIRED — pairwise probe did not generalize. Held out of lexicon per Correction #16.
- **AI-Augmented Adversary Tradecraft** — OPEN BUT NOT LEXICON-READY — surfaced by Vercel/Context.ai disclosure 2026-04-19; no on-chain corpus evidence yet. Held out per Correction #16.
- **Gas Fingerprint Cluster (Architect's 46 deployers)** — RETIRED — debunked 2026-03-30 in `CASE_ENTITY_005_THE_ARCHITECT.md` Corrections section. 0.0200xx gwei is the Arbitrum default (54% of all Arbitrum deployers). Behavioral matches preserved.
- **"Anti-forensic implementation destruction"** — RETIRED — `CORRECTIONS.md` 2026-04-05. The address was an EOA, not a SELFDESTRUCT'd contract.

---

## Section 4 — Extraction Events

| ID | Date | Chain | Value | Primary file | Status |
|---|---|---|---|---|---|
| EXTRACTION_001 | (placeholder — see `extraction_events` table) | — | — | — | DOCUMENTED in DB only |
| EXTRACTION_002 | (placeholder — see `extraction_events` table) | — | — | — | DOCUMENTED in DB only |
| EXTRACTION_003 | 2026-03 | Base | ~$211K (~100.56 WETH) | `surveillance/data/cases/PARASITE_ARCHITECTURE_0xd4624228.md` | CONFIRMED, dormant since 2026-03-26 |
| EXTRACTION_004 | 2026-04-16 | NEAR (off-chain — `monitored_chain=0`) | $18.4M loss / $8.26M recovered | `reports/extraction_event_004_rhea_finance.md` | CONFIRMED |
| EXTRACTION_005 | 2026-04-01 | Solana (off-chain — `monitored_chain=0`) | $285M loss / $247.5M recovered | `reports/drift_prehindsight_simulation.md`, `reports/drift_simulation.md`, `reports/post_drift_impact.md` | CONFIRMED |
| EXTRACTION_006 | 2026-04-09 | BNB Chain (off-chain — `monitored_chain=0`) | (admin compromise drain) | `reports/extraction_event_006_aethir.md` | CONFIRMED |
| EXTRACTION_007 | 2026-04-13 | Ethereum (`monitored_chain=1`) | ~245 ETH + ~1B DOT phase 2 | `reports/extraction_event_007_hyperbridge.md` | CONFIRMED |
| EXTRACTION_008 | 2026-04-18 | Ethereum / Unichain (`monitored_chain=1`) | ~$292M | `reports/extraction_event_008_kelp.md`, `reports/kelp_retrospective_replay.md` | CONFIRMED |

The off-chain events (004, 005, 006) are corpus-expansion case studies — `monitored_chain=0`. EXTRACTION_001 / EXTRACTION_002 are referenced in `claude.md` Database Schema as table rows but have no case-file or report-file documentation in current corpus state. **Gap flagged.**

---

## Section 5 — Bytecode Families

### Active families (top 8 by member count, post-Correction-#3 dissolution of T2-eaef6a5d)
- **`T1-d5351e977044`** — flagship cross-deployer family. ~1,998 members per most-recent count. Pattern: `Tier1-ORIGIN at 0x314 -> EQ at 0x31d -> JUMPI`. **77.13% top-12-funded** (per 2026-04-25 dominance check). Likely closely linked to org_002's tx.origin campaign.
- **`T1-bb7b0ca2e505`** — 832 members. Pattern: `Tier1-SELFDESTRUCT at 0x180b in ERC-20 transfer`. **99.88% top-12-funded** — essentially a single-actor signature mistaken for prevalence.
- **`T1-2179bb01e057`** — 216 members. Pattern: `Tier1-DELEGATECALL at 0x13dd in ERC-20 context`. **94.91% top-12-funded.**
- **`T1-d13505aeb1aa`** — 329 members. Pattern: `Tier1-CALLER at 0x17b -> EQ at 0x17c -> JUMPI`. 11.85% top-12-funded.
- **`T1-78d4dfc7ac5f`** — 227 members. Pattern: `Tier1-CALLER at 0x2ac -> SLOAD at 0x2c0 -> JUMPI`. 47.58% top-12-funded.
- **`T1-39b12abd4db3`** — 222 members. Pattern: `Tier1-CALLER at 0x2ab -> SLOAD at 0x2bf -> JUMPI`. 44.59% top-12-funded.
- **`T1-fa8c132e5058`** — 204 members. Pattern: `Tier1-CALLER at 0x2ad -> SLOAD at 0x2c1 -> JUMPI`. 47.55% top-12-funded.
- **"tx.origin conditional at 0x314"** — 350 members, 350 unique deployers. Documented in `CASE_*COMPETITION_ANALYSIS_20260322.md`. Identified there as **org_002 campaign** (1 contract per deployer).

### Dissolved family
- **`T2-eaef6a5d7678`** — DISSOLVED 2026-04-16 per `reports/correction_log.md` Correction #3. Was a NULL-bucket methodology artifact (clustering on the all-zero bytecode-flag combination). 21,936 member rows + 20,936 contract reclassifications removed. See `reports/family_T2_eaef6a5d_verdict.md`.

### Cross-family observations
- **`SHA3 -> SLOAD -> JUMPI -> MUL/DIV` obfuscated fee-on-transfer** — pattern shared across org_003 (6 contracts) and Zero-Day (`0xa7fea69e`). The technique is propagating beyond its original operator (`CASE_ZERODAY_0xa7fea69e.md` Significance §2).

Per `surveillance/data/cases/COMPETITION_ANALYSIS_20260322.md`: 106 distinct bytecode pattern fingerprints are shared across multiple deployers in the corpus. Most templates are reused; per-deployment customization is parameter-level, not bytecode-level.

---

## Section 6 — Maintenance Protocol

### When to update this index
- **Whenever a new file lands in `surveillance/data/cases/` or `reports/`**, add an entry to the relevant section in the same commit. The case-file commit and the index update commit should be the same commit.
- **Whenever a wallet is reclassified** (e.g., `0x785ce546` from victim → controlled intermediary on 2026-04-13), update the address-index entry and add a note in the relevant entity's Section 1 entry.
- **Whenever a pattern is retired or promoted to lexicon**, update Section 3 status. Anecdotes do not get lexicon entries; OPEN hypotheses do not get retired without a correction-log entry.
- **Whenever an entity gains a case file**, move from "DB-only" or "no case file" to the proper file-cited entry. Currently-pending: org_004, org_005, org_006, org_007, `0x752c5a95` harvester, the 12 infrastructure-scale operator candidates.

### Drift detection
A weekly check should diff Section 2 entries against actual `surveillance/data/cases/` directory contents:
- If addresses appear in case files but not in Section 2 → index is stale; add them.
- If Section 2 entries point to nonexistent files → entries got orphaned; investigate.
- If `surveillance/data/cases/` exceeds the count of indexed entities by more than 5 files, the index is materially stale and Phase 1 of the session-start protocol should flag this before analysis proceeds.

A scripted version of this check (`scripts/check_index.py`) is the recommended stretch-goal automation.

### Known gaps in current index (2026-04-25)
- **org_002** — confirmed entity, no dedicated case file. Only cross-org analyses reference it.
- **org_004, org_005, org_006, org_007** — DB-only; no case files yet.
- **`0x752c5a95` harvester** — 1,898+ approvals on a confirmed contract, no dedicated case file (only investigation scripts).
- **The 12 infrastructure-scale operator candidates** — DB-only via `infrastructure_operator_candidates`; no case files. Two are confirmed org_001 infrastructure (Whale Trader + Gas Station), one is org_004; the other 9 are unattributed.
- **EXTRACTION_001 / EXTRACTION_002** — referenced in `claude.md` but no case-file or report-file documentation surfaced in current corpus.
- **Trust Amplification Factor 14.2× retraction story** — two contradictory documented explanations exist (`CORRECTIONS.md` 2026-04-02 vs `reports/correction_log.md` Correction #17 2026-04-25). The lexicon entry has been updated to caveat the figure but does not explicitly reconcile the two retraction theories. Resolution pending.
- **`0xb5ae0b6cb72dcf5180ac2a4c3b77bebef5b42a81`** — surfaced in pairwise overlap (192 victim bots, top operator by bot-fleet size). Has 8-contract fleet per `CASE_0x9da33ece_*.md` but no organizational attribution.
- **`0x1d13a5aefd4d3a0f466c0058526d8bf11d88502a`** — deployer of `0xb15e7a89` (Multi-Mechanism Trap, 8 contracts, Arbitrum). No organizational attribution.

### Cyclic / ephemeral reports (not individually indexed)

These prefixes produce time-stamped analytical snapshots that accumulate in `surveillance/data/cases/`. They are session outputs, not entity-level case files, and are intentionally excluded from Sections 1–5. Sessions consult them as supplementary context but do not add entries to INDEX.md when generating new ones.

- `DAILY_REPORT_*.md` — daily intelligence briefs
- `FUND_FLOW_TRACE_*.md` — one-off fund-flow analyses
- `ORG_CYCLES_*.md` — organizational activity cycle snapshots
- `INFRA_EVENT_*.md` — infrastructure-event snapshots (e.g., `INFRA_EVENT_FLASHBLOCKS_20260324.md`)
- `ENTITY_CLASSIFICATION_*.md` — corpus-wide classification snapshots

The drift checker (`scripts/check_index.py`) skips these prefixes when looking for unindexed case files.

### Two correction logs — disambiguation
- **`CORRECTIONS.md`** (project root) — date-titled entries, customer-facing claims log, claim/reality/discovery/fix/severity format. Entries from 2026-04-02 onward.
- **`reports/correction_log.md`** — numbered entries (#1–#17), methodology-correction format, multi-section. Entries from 2026-04-16 onward.

Both must be checked at session start. They serve overlapping but distinct purposes: `CORRECTIONS.md` records when external-facing claims got rolled back; `correction_log.md` records when internal methodology had to be revised. A single correction may appear in both if it has both surfaces (Correction #17 was added to `correction_log.md` only; the 14.2× retraction story is partly in each, which is the source of contradiction A flagged in Section 3).
