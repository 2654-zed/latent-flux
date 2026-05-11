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
- **Status:** CONFIRMED — actively expanding. **Drainer-spawn escalation 2026-05-06:** First confirmed direct-org-drainer linkage observed — `0xfbf44e969d4fc5cbad62870207341c976f9e38f9` (Arbitrum, watchlist HIGH `self_deploying_drainer_fbf44e96_org001`) drained 113 victims on 2026-05-06 via contract `0xd6cd943bfc0711125bc01cff7b7dfb87be1d10c8`, funded by org_001 gas station `0x8c826f795466`. **Caveat on framing:** the gas station funds 1,164 lifetime corpus deployers; this is 1 of 1,164 (0.09%) — represents playbook expansion (adding wave-class drain pattern) rather than wholesale class shift from camouflage/infrastructure to drain. Worth tracking whether iter_2 of this drainer pattern emerges from the same gas station downstream.
- **Headcount caveat:** `CASE_ORG_001_INFRASTRUCTURE.md` cites "559+ deployers / 7,400+ contracts." `CORRECTIONS.md` 2026-04-02 documents "actual numbers depend entirely on attribution method (16/26/308/324)." Both are consistent if the 559 figure is the union over all attribution methods at a later snapshot — use `CASE_ORG_001_INFRASTRUCTURE.md` as canonical, but quote with the snapshot date.
- **Key wallet roles:**
  - **CEX origins:** Coinbase Hot 1 `0x503828976d22510aad0201ac7ec88293211d23da`, Binance Hot 1 `0x28c6c06298d514db089934071355e5743bf21d60`
  - **Mainnet buffer:** `0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d`, `0x4976a4a02f38326660d17bf34b431dc6e2eb2327`, `0xf3d63166f0ca56c3c1a3508fce03ff0cf3fb691e`, `0x81f91aca8c05b3eefebc00171139afefac17c9a6`
  - **Mainnet operations:** Central Treasury `0x4c968f6beecf1906710b08e8b472b8ba6e75f957`, ~~Whale Trader `0xf70da97812cb96acdf810712aa562db8dfa3dbef`~~ **[CORRECTION #20 → Relay: Solver — REMOVED from org_001 attribution; this is a cross-chain bridge solver, not an org_001 wallet]**, MEV Bot `0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1`, Revenue Collector `0x391e7c679d29bd940d63be94ad22a25d25b5a604`, WETH Wrapping `0x5e0f8e7337c8955d2124b8e85ca74af884b3e124`, Revenue Source `0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390`. **Note (2026-05-09):** the "68% of org_001 deployments through whale path" claim previously made for `0xf70da978` is retracted — that was Relay's bridge throughput, not org_001 fund flow. The core org_001 case is unchanged; only the whale-trader attribution dissolves. CASE_ORG_001_INFRASTRUCTURE.md needs update.
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

### org_004 — **DISSOLVED 2026-05-09, Correction #20**

> **[CORRECTION #20 — 2026-05-09]** `0xbaed383ede0e5d9d72430661f3285daa77e9439f` is OLI-tagged as **"Bybit: Hot Wallet 6 / DEPOSIT ADDRESS / Exchange / Bybit"**. The "210 deployers / 1,492 contracts / 63.1% adversarial ratio" finding is normal Bybit hot-wallet customer-withdrawal volume; some recipients happened to deploy contracts that classifier flagged as suspected. **org_004 as an organizational entity does not exist.** CLAUDE.md priority #12 ("Investigate org_004") should be removed. Original entry preserved for historical record.

#### Original entry (preserved for historical record):

- **ID:** `org_004`
- **Description:** Named in `claude.md` Current Priority Items #12 ("Investigate org_004 (0xbaed383e) — next organizational mapping target"). Surfaced in 2026-04-25 by `surveillance/infrastructure_operator_detector.py` as a 210-deployer / 1,492-contract / 63.1% adversarial-ratio infrastructure operator.
- **Primary case file:** None — gap. Only in `claude.md` priority list and `infrastructure_operator_candidates` table.
- **Status:** DISSOLVED — see correction note above.
- **Key wallet:** `0xbaed383ede0e5d9d72430661f3285daa77e9439f` **[CORRECTION #20 → Bybit: Hot Wallet 6]**

### org_005, org_006, org_007 — Tier-1 Cluster Promotions (DB-only)
- **ID:** `org_005`, `org_006`, `org_007`
- **Description:** Three Tier-1 clusters promoted to `org_wallets` on 2026-04-23 via `scripts/tier1_promotion.sql`. 40 wallets total: org_005 (9), org_006 (16), org_007 (15).
- **Primary case file:** None — gap. Only `org_wallets` table entries.
- **Status:** PROMOTED (DB-only) — case files pending

### Entity_005 — "The Architect" (Arbitrum R&D operator)

> **[CORRECTION #20 — 2026-05-09 — partial retraction]** The Architect-primary-deployer finding (`0x9209c9f7...`) is NOT retracted by this correction; it stands. Two attribution surfaces around it ARE retracted:
> - **Funder `0x151b381058f9...`** is OLI-tagged as **"MoonPay 4 / Exchange"** — a fiat-onramp address with millions of recipients. The "Architect's sole funder" framing collapses; MoonPay is not exclusive to anyone. The case-file's funding-side narrative needs revision.
> - **Behavioral match `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` (0.742)** is OLI-tagged as **"CryptoCauses: Deployer"** — a separate Web3 project, not an Architect alternate. Behavioral-similarity 0.742 was computed on the deployer-profile dimensions; identity-matching via OLI was never run. Remove from Architect-cluster attribution.
> - The other behavioral-match candidates (0.711, 0.710, 0.719, 0.799) are not yet OLI-cleared.

- **ID:** `Entity_005`
- **Description:** Most sophisticated trap developer in corpus. Arbitrum-exclusive R&D in 7-contract sessions, building toward 4-mechanism weapon (SELFDESTRUCT + DELEGATECALL + TIMESTAMP + CALLER). Pre-production as of last case-file update — no Base deployment yet.
- **Primary case file:** `surveillance/data/cases/CASE_ENTITY_005_THE_ARCHITECT.md`
- **Status:** UNDER_INVESTIGATION — R&D phase
- **Key wallet roles:**
  - Primary: `0x9209c9f7dcb61937f1ec8160c22c0b2365079474`
  - Funder: `0x151b381058f91cf871e7ea1ee83c45326f61e96d` **[CORRECTION #20 → MoonPay 4]**
  - Behavioral matches (watchlist HIGH): `0x4cfe37d2` (0.799), `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` (0.742) **[CORRECTION #20 → CryptoCauses: Deployer]**, `0x30e88ee4c417f126aacf6a4c3cd641424203fe6d` (0.711), `0x7930e1380157596ff82595d502083bf604eda922` (0.710), `0xd28e6a7ad806e85bd0544ed443d25e48f52c06c3` (0.719)

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

### Pristine-Solo Operators (4 watchlist HIGH promotions, 2026-04-25) — **MAJOR REVISION 2026-05-09, Correction #20**

> **[CORRECTION #20 — 2026-05-09]** **3 of 4 promoted entries are OLI-tagged Web3 project deployers**, not adversarial Pristine Solo Operators:
> - `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` → **Animoca: Deployer**
> - `0xa2a01b4a68575280a2de45178e289da717bedb6f` → **Stabilize Finance: Deployer 2**
> - `0x147b8869655bc09f226955cc676ff78efe240ca8` → **Luchadores: Deployer** (NFT project)
> - `0xf6c99cec5bd639316a19d2f56afc14bd046d3a90` → not OLI-tagged; remains under current classification pending further check
>
> The Pristine Solo Operator detector flagged on behavioral signal (long mainnet age + first L2 appearance + small fleet) — the same signature an established institutional or project deployer presents when expanding to a new chain. The detector did not consult OLI tags. Lexicon entry [Pristine Solo Operator](lexicon.md#pristine-solo-operator) updated with FP class section. The `0x752c5a95` Pre-Drain Harvester finding (next entry) is **not retracted** by this correction — the harvester contract's behavior is documented independently of its deployer's identity, and a project-deployer wallet deploying a confirmed-tier approval-harvesting contract is itself a finding worth investigating (compromise? rogue developer? something else?).
>
> **LOW-severity tags require second-source verification** before final action — see Correction #20 open work item #4.

#### Original entry (preserved for historical record):

- **ID:** PRISTINE_SOLO (no canonical org_xxx ID)
- **Description:** Long-mainnet-history wallets surfacing as small-fleet (1–5 contract) trap operators on L2. Lexicon: [Pristine Solo Operator](lexicon.md#pristine-solo-operator). Detector: `surveillance/pristine_solo_detector.py`.
- **Primary case file:** None — DB-only via `pristine_solo_candidates` table. Lexicon entry references the four addresses.
- **Status:** SUPERSEDED for 3 of 4 promoted entries (see correction note above).
- **Key wallets (HIGH watchlist):**
  - `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` (gap 2,498d, deployer of `0x752c5a95` harvester) **[CORRECTION #20 → Animoca: Deployer]**
  - `0xa2a01b4a68575280a2de45178e289da717bedb6f` (gap 2,314d, Arbitrum, fleet 4) **[CORRECTION #20 → Stabilize Finance: Deployer 2]**
  - `0x147b8869655bc09f226955cc676ff78efe240ca8` (gap 1,777d, Base, fleet 4) **[CORRECTION #20 → Luchadores: Deployer]**
  - `0xf6c99cec5bd639316a19d2f56afc14bd046d3a90` (gap ≈1,748d, Base, fleet 2)

### `0x752c5a95` Pre-Drain Harvester (Base)
- **ID:** No canonical ID — investigation only
- **Description:** Confirmed-tier contract harvesting Permit2 approvals from 1,898+ victims (as of 2026-04-24) without firing a sweep. Deployer is `0x80b12bd0` (pristine-solo, 2019 mainnet vintage). Bytecode flags `has_asymmetric_transfer=1`, `has_unusual_fee_structure=1`. The harvester is the largest active confirmed-tier approval pool in the corpus.
- **Primary case file:** None — gap. Only investigated via session scripts (`scripts/investigate_0x752c5a95.py`).
- **Status:** UNDER_INVESTIGATION — pre-drain accumulation, no sweep yet

### Industrial-scale PSO+Single-Purpose hybrid — pulse-burst operator (2026-05-01, **RETRACTED 2026-05-09 — Correction #20**)

> **[CORRECTION #20 — 2026-05-09]** This entry is RETRACTED. Address `0xbb50ce87...` is publicly attributed via Open Labels Initiative as **"Circle: contract deployer"** (`meta.main_entity: "Circle"`) — Circle's institutional contract deployer (the Circle that issues USDC). The 7.7y mainnet history is real but is the history of a publicly-attributed institutional infrastructure address, not Pristine Solo Operator dormancy. The PSO+Single-Purpose+Convergent-Calibration framing below is preserved as historical record per the immutable-corpus-record discipline, but is **not a current Layer 3 finding**. The funder `0x6a9c2449...` is not currently OLI-tagged and may be a Circle-internal hot wallet, an integrator gas-paying for Circle's deployer, or a separate actor; classification deferred. Architecture fix landed: `surveillance/oli_enrichment.py` + `oli_labels` table (`db.py` migration). See `reports/correction_log.md#correction-20` and `reports/blockscout_tag_audit_2026-05-09.csv`.

#### Original entry (preserved for historical record):

- **Deployer:** `0xbb50ce87be3443ed137df1dfdbf2fb0ca8c0a9e0` — Optimism, **38,016 lifetime contracts** (as of 2026-05-02), 7.7y mainnet history (first tx 2018-08-29). Watchlist HIGH (`pristine_solo_industrial_bb50ce87`).
- **Operating pattern (refined 2026-05-03): pulse-burst, not continuous.** Across a 19-day lifespan (2026-04-13 first appearance → 2026-05-01 most recent burst end), bb50 has only **6 active deployment days**. The pattern is: 1 initial deploy 04-13 → **13-day silence** → 4-day burst 04-28→05-01 (33,016 contracts in 4 days, peaking at 13,000 on 05-01) → silence resumed 05-02. Today's quiet is consistent with the operator's historical rhythm; not pivot, not disappearance. Reload-between-bursts posture (capital, opsec rotation, key infrastructure).
- **Funder:** `0x6a9c2449c32779f89d0ccafd746152e237c1bdf2` — pure funding wallet, funds 2 corpus deployers but 99.99% of fleet via the bb50 operator. Watchlist HIGH (`single_purpose_funder_industrial_bb50`).
- **Burst:** 9,333 contracts deployed in 7-hour window 2026-04-30T21:37 → 2026-05-01T04:42 UTC, ~22 contracts/min sustained. Single bytecode template `5439c9995738b4a07047059bd8d43e89` covers 99.97% of the burst. 99.98% suspected-tier.
- **Status:** Case file PENDING. This is structurally novel — confluence of [Pristine Solo Operator](lexicon.md#pristine-solo-operator) (7.7y mainnet age) + [Single-Purpose Infrastructure Funder](lexicon.md#single-purpose-infrastructure-funder) (one funder, one main deployer) + Infrastructure-Scale Operator territory (29K fleet exceeds any documented single-deployer fanout). Case file should propose whether to refine an existing typology or name a new "industrial-scale hybrid" class.
- **Detection note:** Surfaced by post-monitor-restart probe 2026-05-01 — first deployments visible only because Layer 3 ingest came back online at 21:37 UTC. The bb50 deployer's previous ~20K contracts on Optimism happened in our blind window and are visible only as the existing `total_contracts_deployed=29016` baseline.
- **Upstream depth-2 probe (2026-05-01 via Blockscout, Optimism chain):** bb50 funder `0x6a9c2449c327…` first Optimism tx 2025-09-08, current balance ~1,562 ETH (~$3.58M). Top 6 upstream senders (covering ~750 ETH inflow across page 1): `0xe0eace0a4659…` (542 ETH, 71%), `0xe129188380d4…` (126 ETH, 17%), `0xaf8c81b247c9…` (24 ETH), `0x1ebdab5ed9bc…` (23 ETH), `0x8a2ee96c329d…` (18 ETH), `0xbacec9ceb004…` (17 ETH). **All 6 are absent from our corpus** — not in `deployers`, `watchlist`, or `org_wallets`; none fund any corpus deployer. The bb50 operation is structurally independent from org_001-004, the top-12 Infrastructure-Scale Operators, and the 69 Single-Purpose Funder pairs. Reinforces the [Convergent Calibration](lexicon.md#convergent-calibration) pattern at industrial scale: yet another independent operator running the same operational template with no observable coordination linkage.

### Single-Purpose Infrastructure Funder typology (2026-04-28 lexicon entry, 69 Pattern A operators)
- **ID:** No canonical IDs — typology, not a single entity
- **Description:** Funder wallets that fund exactly one corpus deployer with fleet ≥50, where the funder has no deployer record. Opposite calibration of Infrastructure-Scale Operator. Three deployment shapes (burst-mass, slow-drip, active-weaponized). Lexicon: [Single-Purpose Infrastructure Funder](lexicon.md#single-purpose-infrastructure-funder).
- **Primary case file:** `cases/CASE_SINGLE_PURPOSE_INFRASTRUCTURE_FUNDER.md`
- **Status:** ACTIVE — 69 Pattern A operators in corpus; Coffee Fleet retroactively classified as the active-weaponized exemplar; `0xb3c07d462cbc` (3,161 fleet, deployed 2026-04-24 to 28) the largest burst-mass instance to date
- **Key wallets (top-3 burst-mass):** `0xb3c07d462cbcd384636d713aaaa8a841f180e509`, `0x0b701885fbee30213ce8847da8aef1202d13a4e4`, `0xd660fa35cd16f768e41c8e09729e39385b51f55c`

### Cross-Domain Compositional Harm references (anchor cases)
- **Description:** Empirical anchors for lexicon entry [Cross-Domain Compositional Harm](lexicon.md#cross-domain-compositional-harm). Off-chain (Vercel), on-chain (Wasabi), and substrate-bridging (Grok/Bankr) instances are now documented; the framework's claim that the same compositional pattern operates across substrates is empirically anchored on all three sides.
- **Vercel / Context.ai breach (2026-04-19, off-chain):** `cases/CASE_VERCEL_CONTEXT_BREACH_20260419.md`. Eight-domain composition chain (Lumma Stealer → Context.ai AWS → Google Workspace OAuth → Vercel SSO → env-var visibility → bulk customer credential extraction). Status: STRUCTURAL_REFERENCE — primary-source detail to be appended on future disclosure review.
- **Wasabi Protocol admin-key compromise (2026-04-30, on-chain) — EXTRACTION_009:** `cases/CASE_WASABI_EXPLOIT_20260430.md`. ~$5M loss across Ethereum / Base / Berachain / Blast via UUPS proxy upgrade by compromised wasabideployer.eth (`0x5c629f8c0b53`). Same attacker helper deployed at `0x02228b0afcdbEdf8180D96Fc181Da3AF5DD1d1ab` on both mainnet and Base via CREATE2. Status: CONFIRMED. **Layer 3 had zero corpus coverage** — Wasabi predates monitoring window and production ingest was stuck during attack.
- **Grok / Bankr AI-wallet permission chain attack (2026-05-04, substrate-bridging) — `cases/CASE_GROK_BANKR_EXPLOIT_20260504.md`.** Twitter → Grok (LLM translation deputy) → Bankrbot (on-chain execution deputy) → Base blockchain. ~3B DRB tokens (~$150–200K) drained from Grok-controlled wallet to `ilhamrafli.base.eth` via two-stage attack: (1) "Bankr Club Membership NFT" sent to Grok wallet granting "Executive" permissions; (2) Morse-encoded transfer instruction in a Twitter reply asking Grok to translate. Bankrbot treated decoded output as authenticated command and executed. Funds returned in full shortly after. SlowMist labeled "Permission Chain Attack" (2026-05-07). Status: CONFIRMED via multi-source public reporting. **Strongest bridging anchor for the typology** — each substrate's deputy executed correctly under its local rules; only the cross-substrate composition produced harm. **Layer 3 had zero coverage** — entirely off the L2-contract-deployment surveillance surface.
- **Bancor EIP-7702 exploit (2026-04-29):** `cases/CASE_BANCOR_EIP7702_20260429.md`. Status: SKELETON — kept as research target. Wasabi now provides the live structural analog the Bancor file was hypothesizing.

### Private-key drain via Telegram phishing — attacker `0xF7cFFC27` (2026-05-11)
- **Description:** Telegram-phishing-vector private-key compromise drained ~$172K across Base / BSC / Ethereum in a 30-minute window 2026-05-11 00:37–00:56 UTC. Victim wallet `0x62acE10c…` is outside Layer 3's deployment-window corpus (DeFi participant, Sigma trading bot user, not a contract deployer); attacker EOA `0xF7cFFC27732a5C9c4E2D592F3E33435F8dDb019A` is now on watchlist HIGH (local + prod row 95).
- **Forensic anchor:** Sigma tx `0xb81f9f0a1abb2330763d7b9498185404277955a18b3f766a31582c83ba70047e` on Base — EIP-7702 delegation from victim wallet to `0x0…0`, demonstrates attacker's full signing-key control. **First documented L3 case where EIP-7702 delegation is the dispositive forensic anchor.**
- **Current attacker holdings (2026-05-11 ~05:00 UTC):** Base 52.56 ETH (~$122K), BSC 38.80 BNB (~$25K), Ethereum 10.20 ETH (~$24K). All drained tokens (POD on Base, FHE on BSC, SAT1 on Ethereum) already swapped to native via KyberSwap. **Attacker dormant ~4h** at file creation; recovery window open.
- **Mechanism (Tier B per analyst):** Telegram CAPTCHA-bot clipboard injection → browser-stored credential / wallet-key exfiltration → manual cross-chain drain runbook.
- **Framework mapping:** Configuration-Level Vulnerability at the operational-security layer; Cognitive Load Concentration empirical anchor; Forced Deterministic Neutrality at the signature-validation layer. **NOT a Distributed Confused Deputy Chain** — no contract composition failure; the failure was credential custody.
- **Case file:** `cases/CASE_PRIVATE_KEY_DRAIN_F7CFFC27_20260511.md`.
- **Tooling built:** `scripts/monitor_attacker_outflows.py` — polls Blockscout / Etherscan V2 for outbound txs from any EOA, emits stdout line per new tx (suitable for Claude Code Monitor or cron). Two backends; etherscan-v2 backend requires `ETHERSCAN_API_KEY`.
- **Outstanding:** verify on first off-ramp (CEX deposit, bridge, mixer) → escalate to active recovery; BSC visibility (out of Blockscout free coverage); preemptive CEX freeze-intake reports filed by victim.

### Distributed Confused Deputy Chain — Renegade Dark Pool proxy compromise (2026-05-10)
- **Description:** First on-chain anchor for the [Distributed Confused Deputy Chain](lexicon.md#distributed-confused-deputy-chain) lexicon entry (added 2026-05-10 alongside this case). Proxy → vault → approvals architecture in which an unprotected initializer on the Renegade proxy let an attacker reset the implementation pointer; each downstream contract continued to execute correctly against the now-attacker-controlled proxy, draining user assets.
- **Attacker EOA:** `0x777253F28AdC29645152b7b41BE5c772A9657777` (Arbitrum). Pre-positioned 2026-02-03 (first tx), dormant until exploitation burst 2026-05-10 16:51 → 17:14 UTC (10+ token `transfer` sweeps in ~45 seconds). Tier A.
- **Pre-attack implementation:** `0xc038933d0b33359f5C87B4B2f92Ee0DAd11EaDc5` (Arbitrum, created 2025-05-21 by `0x812922c33079c3E2324D25Ef0352a2220686C2Ac`). Tier A.
- **Renegade Darkpool Proxy:** `0x30bD8eAb29181F790D7e495786d4B96d7AfDC518` (Arbitrum, EIP-1967). **Tier A** — confirmed via Blockscout OLI public tag `"Darkpool proxy"` and Renegade deployer attribution (`0x98e4e5C6223bb2Cc945a7c2821E30929dEff3568`), created 2024-09-03.
- **Post-attack implementation:** `0x58f876aAeeCBD5a0fca8F87e1313a9188C155bcC` named **"DarkpoolFrozen"** — protocol-team emergency-freeze swap already in place at file creation. Tier A.
- **Case file:** `cases/CASE_RENEGADE_EXPLOIT_20260510.md`.
- **Status:** CONFIRMED on-chain sweep burst + protocol-team emergency response observed. **Layer 3 had zero corpus coverage** — Renegade contracts predate monitoring window and the protocol sits outside the L2-deployment-burst surveillance surface.
- **Outstanding:** loss-magnitude estimate; identify the caller of the implementation-swap-to-frozen tx; corpus-wide `proxy_initializer_scanner.py` detector proposed in case file.

### `0xe69f81b8` — High-Volume Bridge User — **RETRACTED 2026-05-09, Correction #20**

> **[CORRECTION #20 — 2026-05-09]** OLI-tagged as **"Binance: Internal 2 / Exchange"**. The 49,000 ETH bridged was Binance treasury rebalancing exchange funds. The "coordinated during western sleep hours" framing was Binance internal-transfer scheduling, not adversarial coordination. Original entry preserved below for historical record.

#### Original entry (preserved for historical record):

- **ID:** No canonical org assigned
- **Description:** EOA bridging 49,000 ETH (~$147M) to L1 via the canonical Base bridge over 7 days (April 7–14). One of the most active L2-to-L1 bridge users in the Base ecosystem. Coordinated during western sleep hours.
- **Primary case file:** None — only `CORRECTIONS.md` entries 2026-04-07 (19,000 ETH) and 2026-04-14 (additional 30,000 ETH).
- **Status:** TRACKED (formerly watchlist HIGH, `entity_type: high_value_bridge_user`) — SUPERSEDED by Binance Internal identification.
- **Key wallet:** `0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb`

### Infrastructure-Scale Operator candidates (2026-04-25 detector run, 12 candidates; canonical metrics in `reports/funder_cluster_diagnostic_2026-04-29.md`) — **MAJOR REVISION 2026-05-09, Correction #20**

> **[CORRECTION #20 — 2026-05-09]** **6 of 12 (50%) of the documented Top-12 are confirmed CEX or bridge institutional addresses**, not adversarial operators:
> - `0x3304e22ddaa22bcdc5fca2269b418046ae7b566a` → **Binance 73 / Exchange / Binance** (also retracts the cross-typology drainer-spawn-hub framing in the entry below)
> - `0x39591e7c099a379fd7b349ebfecaeef439c40454` → **OKX 177 / Exchange**
> - `0x4e3ae00e8323558fa5cac04b152238924aa31b60` → **MEXC 15 / Exchange / MEXC**
> - `0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda` → **OKX 137 / Exchange**
> - `0xbaed383ede0e5d9d72430661f3285daa77e9439f` → **Bybit: Hot Wallet 6** (also dissolves the org_004 investigation surface — see Section 1 org_004 entry)
> - `0xf70da97812cb96acdf810712aa562db8dfa3dbef` → **Relay: Solver / Relay Bridge** (also retracts org_001 whale path attribution — see org_001 entry below)
> - `0x80c67432656d59144ceff962e8faf8926599bcf8` → **Orbiter Finance: Bridge**
>
> The detector signal (high deployer-count fanout from a single funding wallet) is the **structural signature of CEX hot wallets and bridge solvers operating at scale** — exactly what these entities do as normal business. The detector did not query public address labels at promotion time.
>
> **The Top-12 cluster is no longer a defensible empirical anchor.** Lexicon entries [Infrastructure-Scale Operator](lexicon.md#infrastructure-scale-operator), [Convergent Calibration](lexicon.md#convergent-calibration), and [Thermodynamic Fundamentalism](lexicon.md#thermodynamic-fundamentalism) (which used `0x3304e22d` as a positive-CER hub example) require revision; the lexicon retraction notes are landed alongside this correction.
>
> **Update 2026-05-09 (post-correction deeper audit):** The remaining 8 of 12 (after subtracting the 4 confirmed CEX/bridge entries above) ARE all OLI-cleared (no public tag returned by metadata service). Of those 8, 4 are pre-attributed within Layer 3 (`0xb0b0b690` Adversarial Vanity Branding, `0xde8eb937` org_002_junior, `0x238d7170` org_002_senior, `0x8c826f79` org_001 L2 Gas Station). The other 4 (`0xc43f317e`, `0x0e6e9177`, `0x8ca70232`, `0xca7ece5e`) carry the topology fingerprint expected of the typology — high disposable-deployer fanout, all-fleet-1 downstream, predominantly L2-only origin (no eth_depth).
>
> **Further update 2026-05-10 (bytecode decompilation of all 4):** Bytecode-level review of one sample from each of the four split them into **two distinct operator classes**:
> - **Vanilla ERC-20 meme-token shops (3 of 4):** `0xc43f317e` (sample: `0xacfdc090` "Kore Agent", OZ v5.0.0 ERC-20), `0x0e6e9177` (sample: `0xcbbd17f9` "X1000XLiquidBGT", OZ v5.0.0), and `0xca7ece5e` (sample: `0x3b6af3e8` "CelestialForge", OZ ERC20.sol). All vanilla, no predatory primitives. **The "Infrastructure-Scale Operator" topology framing applies to these three only in the false-positive sense — they generate the high-fanout-disposable signature because that's what an ERC-20 deployment shop does naturally.** Watchlist downgraded for c43f317e to MEDIUM; the other two remain unwatchlisted.
> - **Honeypot token operator (1 of 4):** `0x8ca70232` (sample: `0xaeac0e69` "Laser Eagle", custom `EVMToken.sol` with hardcoded blacklist + hidden `approev(address)` balance-drain primitive gated to the funder). **The one truly adversarial operator of the four.** Added watchlist HIGH (`honeypot_token_operator_8ca70232`) on local + prod 2026-05-10. Case file: `cases/CASE_HONEYPOT_TOKEN_OPERATOR_0x8ca70232.md`.
>
> **Net result on the original "12 Infrastructure-Scale Operators" cluster:** 7 retracted (CEX/bridge per Correction #20), 3 meme-token shops (false-positive class for this typology), 1 confirmed honeypot operator (`0x8ca70232`), 4 pre-attributed (org_002 senior+junior, org_001 gas station, Adversarial Vanity Branding). **Zero of the original 12 retain the "Infrastructure-Scale Operator" framing as documented.** The detector signal (high disposable-deployer fanout from a single funder) requires bytecode-level disambiguation to be useful.

#### Original entry (preserved for historical record):

- **ID:** No canonical IDs — surfaced by detector only
- **Description:** Funder addresses with ≥200 fanout, ≥10% adversarial ratio, ≥50% disposable rate. Lexicon: [Infrastructure-Scale Operator](lexicon.md#infrastructure-scale-operator).
- **Primary case file:** `reports/funder_cluster_diagnostic_2026-04-29.md` (replaces the missing 2026-04-25 file). Live metrics via `scripts/funder_metrics.py`. DB table: `infrastructure_operator_candidates`.
- **Status:** SUPERSEDED. Cross-funder overlap probe 2026-04-29 resolved single-actor-vs-multi-tenant question (twelve independent operations; 0 of 66 pairs share downstream) — that finding remains correct (independence holds, since CEXes and bridges don't share downstream customers either) but the *adversarial* interpretation is retracted.
- **Note on figures:** Per Correction #18, future references to top-12 cluster scale should cite a dated invocation of `scripts/funder_metrics.py` rather than transcribed numbers. The table below was generated 2026-04-26 and is preserved for historical comparison.

**The twelve funder addresses (counts as of 2026-04-26 sync):**

| Funder | Deployers | Contracts | Confirmed | Notes |
|---|---|---|---|---|
| `0xf70da97812cb96acdf810712aa562db8dfa3dbef` | 2,684 | 6,971 | 109 | ~~org_001 Whale Trader~~ **[CORRECTION #20 → Relay: Solver — bridge solver, not org_001]** |
| `0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda` | 2,187 | 7,876 | 0 | ~~Pure stockpile (zero confirmed)~~ **[CORRECTION #20 → OKX 137 / Exchange]** |
| `0x3304e22ddaa22bcdc5fca2269b418046ae7b566a` | **2,027** (was 1,939; +88/24h) | 6,470+ | 33+ | ~~Funds `0x666521` (HIGH watchlist), `0xbad051ece`, **`0x00169219` (5 confirmed in 24h)**~~ **[CORRECTION #20 → Binance 73 / Exchange / Binance — funding linkage is normal CEX customer-withdrawal volume; downstream confirmed-tier deployers retained on watchlist via their own behavior, but upstream-Binance linkage carries no adversarial signal]** |
| `0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078` | 1,562 | 1,393 | 0 | Pre-stage warehouse — not OLI-cleared, retained for re-audit |
| `0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07` | **6,605** (was 5,775; growing) | 6,000+ | 0 | **L2-only Optimism funder. 7-char vanity prefix `0xb0b0b69*` (~268M generation attempts).** Singular: only `0xb0b0b6*` address in corpus. Downstream fleet is random-prefix (vanity scope = funder only). Created 2026-04-15, dormant for 10 days, then mass-funded 6,598 deployers on 2026-04-25 alone. 99.86% of downstream uses bytecode `476b15536fa9...`. Drove the 8,797 COORDINATED_DEPLOYMENT alert surge 2026-04-26. See lexicon: [Adversarial Vanity Branding](lexicon.md#adversarial-vanity-branding). **OLI-cleared (no public tag) — adversarial framing retained.** |
| `0xde8eb937cb5475eee5ac96dce6ba2d18e439c473` | 938 | 807 | 4 | Watchlist HIGH (`org_002_junior` per live entity_name) — OLI-cleared |
| `0x0e6e91775d24d34b90e0f3d806a90705f0199999` | 878 | 724 | 0 | L2-only — OLI-cleared |
| `0x238d7170f309a55b87a144a341bd6105897082ca` | 861 | 734 | 2 | Watchlist HIGH (`org_002_senior`). 98% suspected ratio. — OLI-cleared |
| `0x8c826f795466e39acbff1bb4eeeb759609377ba1` | 743 | 3,366 | 2 | **org_001 L2 Gas Station** (Coinbase-funded) — OLI-cleared |
| `0x8ca702323c341a8d46ee94a2abeddb08798ca10d` | 737 | 320 | 1 | L2-only — OLI-cleared |
| `0x39591e7c099a379fd7b349ebfecaeef439c40454` | 633 | 2,029 | 5 | ~~L2-only~~ **[CORRECTION #20 → OKX 177 / Exchange]** |
| `0xca7ece5e43ef44de8e430629a5b535eca48e251b` | 484 | 576 | 0 | Pre-stage warehouse — not OLI-cleared, retained for re-audit |

**Attribution notes (revised 2026-05-09 per Correction #20):**
- ~~Three are confirmed pre-existing organization infrastructure~~ **One is** confirmed pre-existing organization infrastructure: `0x8c826f79` (org_001 L2 Gas Station). The other two prior attributions retracted: `0xf70da97812` is **Relay: Solver** (not org_001), and `0xbaed383ede` is **Bybit: Hot Wallet 6** (not org_004; org_004 dissolved per this correction).
- Two more are watchlist-classified org_002 infrastructure per live `watchlist.entity_name`: `0x238d7170` (`org_002_senior`) and `0xde8eb937` (`org_002_junior`).
- Four are L2-native (no mainnet history) — deliberate `eth_depth` evasion: `0xb0b0b690` (Optimism), `0x0e6e9177`, `0x8ca70232`, `0x39591e7c` (Base).
- `0xb0b0b690` (Optimism, L2-native) is currently scaling fastest — 4,371 new deployers in 24 hours, all with the same bytecode hash `476b15536fa9703e2c630e91ac976c514e1868a70e8c996f1bf8bb97a9b9e532` (see Section 5). Campaign-shape resembles **org_002 / Dragon scaled 2.8–16.5×** — pre-stage trap stockpile.
- **Three documented vanity-prefix patterns** in the corpus to date (cross-reference for any future vanity find):
  1. **Operational branding** — `0xc0ffee*` (Coffee Fleet): 84 victim bots + 1 deployer. Vanity at the layer that visibly transacts.
  2. **Anti-forensic spoofing** — `0x01989c93890aed05*` (org_001 Shadow Wallet 1): 7-byte prefix collision targeting truncated-address displays. Vanity at the intelligence layer.
  3. **Funder branding** — `0xb0b0b69*` (this entry): 1 funder, anonymized random-prefix downstream. Vanity scope = funder only. Most plausibly read as wallet/infrastructure-rental service brand.
  See lexicon: [Adversarial Vanity Branding](lexicon.md#adversarial-vanity-branding).

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
- `0xf70da97812cb96acdf810712aa562db8dfa3dbef` — ~~Whale Trader (org_001 — primary funding channel, 68% of deployments at 2026-03-28)~~ **[CORRECTION #20 → Relay: Solver / Relay Bridge]**. Cross-chain bridge solver, not org_001. The 2,684-deployer fanout surfaced by `infrastructure_operator_detector` 2026-04-25 is bridge throughput.
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
- `0xbaed383ede0e5d9d72430661f3285daa77e9439f` — ~~org_004 (referenced in `claude.md` priority #12)~~ **[CORRECTION #20 → Bybit: Hot Wallet 6]**. org_004 dissolved per Correction #20.

### Entity_005 — The Architect
- `0x9209c9f7dcb61937f1ec8160c22c0b2365079474` — Architect primary deployer (CRITICAL watchlist) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0x151b381058f91cf871e7ea1ee83c45326f61e96d` — ~~Architect funder (CRITICAL watchlist)~~ **[CORRECTION #20 → MoonPay 4 / Exchange]** — fiat onramp, not Architect-exclusive. Case file funder-side narrative needs revision.
- `0x4cfe37d2` (full: `0x4cfe37d21a5a8a1d74e4840426d644a1b50dc328`) — Architect alternate (HIGH, 0.799) — `CASE_ENTITY_005_THE_ARCHITECT.md`
- `0xc5d133296e17ba25df0409a6c31607bf3b78e3e3` — ~~Architect alternate (HIGH, 0.742)~~ **[CORRECTION #20 → CryptoCauses: Deployer]** — separate Web3 project, not Architect-cluster. Behavioral similarity 0.742 was profile-shape match, not identity match.
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
- `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` — ~~pristine-solo, 2,498d gap, deployer of `0x752c5a95`~~ **[CORRECTION #20 → Animoca: Deployer]**. The `0x752c5a95` Pre-Drain Harvester finding (1,898+ approval pool) is NOT retracted — investigate why an Animoca-tagged wallet deployed a confirmed-tier approval-harvesting contract. Possibilities: compromise, rogue developer, label staleness, or harvester-contract behavior more nuanced than initial classification.
- `0xa2a01b4a68575280a2de45178e289da717bedb6f` — ~~pristine-solo, 2,314d gap, Arbitrum~~ **[CORRECTION #20 → Stabilize Finance: Deployer 2]**
- `0x147b8869655bc09f226955cc676ff78efe240ca8` — ~~pristine-solo, 1,777d gap, Base~~ **[CORRECTION #20 → Luchadores: Deployer]** (NFT project)
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
- `0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb` — ~~High-volume bridge user (49,000 ETH bridged April 7–14, watchlist HIGH)~~ **[CORRECTION #20 → Binance: Internal 2 / Exchange]** — Binance treasury rebalancing, not adversarial. `CORRECTIONS.md` 2026-04-07 + 2026-04-14 entries should be reviewed.
- `0xb15e7a89e71b8468c23eb330f837caf0f2ff7628` — Multi-Mechanism Trap (independent operator) — `CASE_0xb15e7a89_arbitrum_20260322_040329.md`
- `0x01989c93890aed05cbcda4e62eec1b2eb4c55b1b` — LP_POOL_2 (the **legitimate** address being vanity-spoofed by org_001's Shadow Wallet 1, which has the same first 8 bytes `0x01989c93890aed05`) — `CASE_ORG_001_INFRASTRUCTURE.md`
- `0x12577cf0d8a07363224d6909c54c056a183e13b3` — Fee-Skimming Token — `CASE_0x12577cf0_base_20260322_040329.md` (single-file artifact)
- `0x08155ec0cf641720d1e1f66fcaeda8b29f9bbb17` — deployer of `0x12577cf0` (3 contracts total) — `CASE_0x12577cf0_base_20260322_040329.md`
- `0x9da33ece6fdf36ecf99e10dbd6ecd0cb529e257e` — Multi-Mechanism Trap — `CASE_0x9da33ece_base_20260322_040303.md` (single-file artifact)
- `0xb5ae0b6cb72dcf5180ac2a4c3b77bebef5b42a81` — deployer of `0x9da33ece` (8-contract Base fleet, top by victim-bot count in 2026-04-25 pairwise probe with 192 bots) — `CASE_0x9da33ece_base_20260322_040303.md`
- `0x1d13a5aefd4d3a0f466c0058526d8bf11d88502a` — deployer of `0xb15e7a89` (Multi-Mechanism Trap, 8 contracts, Arbitrum) — `CASE_0xb15e7a89_arbitrum_20260322_040329.md`
- `0x666521000c595a632fb3e99f392b12e937b77586` — high-productivity solo operator (watchlist HIGH 2026-04-23, funder `0x3304e22d` infrastructure-scale candidate) — no case file, lexicon-only
- `0xefef185e2c89bbede21a1c41427bdf1332eca392` — high-confirmation-ratio operator (watchlist HIGH 2026-04-23) — no case file, lexicon-only
- `0x00169219376146760298417404949075285cab72` — high-confirmation-rate operator (5 confirmed traps in 24h on 2026-04-26, fleet 22, mainnet 2024-09-03) — funded by `0x3304e22d` (infrastructure-scale candidate). No case file. Investigated via `scripts/investigate_0x00169219.py`. Bot victims: `0xf2b54380...`, `0x5555553ac295`, `0xffffff35da6e`, `0x999999a4d40f` (vanity-prefix MEV bots, NOT the c0ffee fleet).
- `0x202c8b326ca75bf737fd709b524a1333681f0480` — dual role: self-funded trap operator (fleet 13, 2 confirmed, mainnet 2021-05-02) AND deployer-victim of `0x752c5a95` harvester. No case file; first surfaced in 2026-04-24 task-4 deployer-victim investigation, fired 1 trap event 2026-04-26.

### org_001 drainer escalation (`0xfbf44e96` on Arbitrum, 2026-05-06)
- **Drainer:** `0xfbf44e969d4fc5cbad62870207341c976f9e38f9` — Arbitrum, fleet=1, deploy-once-and-dispose. Watchlist HIGH (`self_deploying_drainer_fbf44e96_org001`).
- **Funder:** `0x8c826f795466e39acbff1bb4eeeb759609377ba1` (org_001 gas station, already HIGH).
- **Event:** 113 victims drained via contract `0xd6cd943bfc0711125bc01cff7b7dfb87be1d10c8` (Arbitrum, confirmed tier). Deploy 07:07 UTC → first drain 10:42 UTC (3.5h bait window) → peak 11-12 UTC (100 of 113 drains in 2h) → tail through 19 UTC.
- **Significance:** First confirmed direct-org-drainer linkage in the corpus. org_001 has been documented as funder/whale + gas-station infrastructure across hundreds of deployers, but the harm has historically been mediated (bot traps, slow extraction, camouflage). This is the first wave-class (deploy → bait → mass-sweep → dispose) drain operation directly funded by the gas station wallet.
- **Framing precision:** 1 of 1,164 lifetime gas-station-funded deployers = 0.09%. Playbook expansion, not wholesale class shift. Watch for iter_2 from same funder.

### Drainer-spawn hub `0x3304e22ddaa2` — cross-typology — **MAJOR REVISION 2026-05-09, Correction #20**

> **[CORRECTION #20 — 2026-05-09]** `0x3304e22ddaa22bcdc5fca2269b418046ae7b566a` is OLI-tagged as **"Binance 73 / Exchange / Binance"** with `meta.main_entity: "Binance"`. The "infrastructure-scale operator" rank is retracted (see Top-12 entry above). What remains real: 4 drainer wallets did receive funds from this Binance hot wallet over the March-May window. **Reframing:** these were CEX-customer withdrawals to drainer wallets (drainers cashing out attacker proceeds, OR victims being drained after withdrawal — the funder-side cannot distinguish). The "single shared funder across both days' drainer wallets" finding is technically correct but uninformative — a CEX hot wallet shares as funder across thousands of unrelated downstream addresses. The cross-typology drainer-spawn-hub finding does NOT survive: the hub is Binance, not a coordinator.
>
> **Preserved as separate finding (still real):** the four drainer wallets `0xbaff5fe29cee…`, `0x7b72595d62b1…` (399 victims!), `0x16cd9c10664b…`, `0x0f5162779f6b…` are still on watchlist HIGH for their own behavior. Drainer-side classification is unchanged. Only the upstream-funder linkage is dissolved.
>
> Original entry preserved below for historical record.

#### Original entry (preserved for historical record):

- **Hub:** `0x3304e22ddaa22bcdc5fca2269b418046ae7b566a` — watchlist HIGH (`infrastructure_scale_drainer_spawn_hub_3304e22d`). Documented top-12 Infrastructure-Scale Operator (rank #4 with 2,134 deployers per `reports/funder_cluster_diagnostic_2026-04-29.md`) AND drainer-spawn hub.
- **4 drainer iterations across March 25 → May 1, 404 total victims:** `0xbaff5fe29cee…` (3-25, 2v), **`0x7b72595d62b1…`** (4-01 → 4-04, **399v — largest single-iteration drain count in corpus**), `0x16cd9c10664b…` (4-27 → 4-28, 2v), `0x0f5162779f6b…` (5-01, 1v). All on watchlist HIGH.
- **Significance:** First documented case of an Infrastructure-Scale Operator running as a drainer-spawn hub. Confirms the lexicon's pre-stage / stockpile / live-extraction sub-types can co-exist with active drainer-spawn within one hub. Surfaced 2026-05-02 via Apr 1 ↔ May 1 cross-link probe (single shared funder across both days' drainer wallets).
- **Lexicon:** [Infrastructure-Scale Operator § Drainer-spawn-hub overlap](lexicon.md#infrastructure-scale-operator) — refinement added 2026-05-02.

### Persistent Arbitrum cluster funder `0xbbfca1dfa3c2` (2026-05-02)
- `0xbbfca1dfa3c2515df653c395a52ac603466bbbab` — watchlist HIGH (`arbitrum_cluster_funder_bbfca1df`). Funds 8 persistent deployers (active across full April 1 → May 1 30-day window) including `0x47b3fb3b0e65…` (188 fleet), `0x1a1c55fe35b5…` (173), `0xcfdef8a8ddbb…` (164), `0x1ce5e3a70d93…` (159). Largest non-watchlisted persistent-funder signal surfaced via the 89-deployer cross-day cohort. Likely Infrastructure-Scale candidate or Architect-cluster adjacent.

### Drainer-spawn operator hub `0xf7883e3fef23` (2026-05-02, automation confirmed 2026-05-03, sub-minute precision verified 2026-05-05, **iter_9 forecast missed 2026-05-07**)
- **Operator hub:** `0xf7883e3fef23c8e645deba4b540549d78028a616` — watchlist HIGH (`drainer_spawn_hub_f7883e3f`).
- **Campaign scale (as of 2026-05-05):** 8 drainer iterations across 28 days, **~1,150 total victims drained on Base.** Largest single-operator-by-victim-count in the corpus.
- **Iterations:** `0x7d34e0a0…` (4-07, 139v) → `0x2ce5ff20…` (4-13, 139v) → idle `0xb07b8e9f…` (4-15) → `0xa370e0f4…` (4-25, 154v) → `0xdf86f7c9…` (4-27, 153v) → `0x1aae146c…` (4-29, 133v) → `0x1ac0dd67…` (5-01, 144v) → `0x72747b31…` (5-03, 143v) → **`0xa8c7ac1c…` (5-05, iter_8)** → **iter_9: NO SPAWN (2026-05-07 — forecast missed)**. All hub-spawned deployers deploy-once-and-dispose, fleet=1.
- **Automation signature — sub-minute precision (verified 2026-05-05 with iter_8):** iter_6 spawned at 09:33:41 → iter_7 at 10:01:41 (+47.7h, 28-min drift) → iter_8 at **10:02:11** (+48.0h, **30-second drift** from iter_7 time-of-day). Three-iteration time-of-day window: 09:33-10:02 UTC. Interval mean 47.85h ± 15min. Time-of-day **converging on 10:02 UTC ± 30 seconds**. This is automation precision tighter than human-tuned cron jobs typically achieve.
- **Forecast iter_9:** 2026-05-07, **~10:02 UTC window** (±30 minutes for safety; precision suggests narrower). Same template, ~140 victims expected.
- **iter_9 outcome (2026-05-07): MISSED.** First observable break in the 28-day automation streak. The 2-day cadence at 09:30-10:02 UTC held for 5 consecutive iterations (Apr 25 → May 5) and produced no May 7 spawn within the forecast window. Cadence prediction track record: iter_4–iter_8 correct (5/5); iter_9 wrong (0/1). Hypotheses (none confirmed): scheduler crash, deliberate pause, operator pivot to a different launch surface, or simple +1-day drift not yet observed at probe time.
- **Cadence prediction track record:** iter_8 prediction made 2026-05-03 with ±90min window confirmed within 1 hour of midpoint (+1 minute drift from iter_7). Three consecutive correct cadence predictions = scheduler-level confidence in operator's automation. Streak broken by iter_9.
- **Spawn-cadence vs drain-cadence are decoupled processes.** iter_8 (`0xa8c7ac1cdc33…`) is **still actively draining** despite the hub's spawn pause: 3 victims at 09:10:03 UTC on 2026-05-07 hitting `0xee80d04303e2…`, cumulative 151 victims as of 2026-05-07. Pattern: scheduler stops; in-flight operator continues. Structural implication — this is not a single automation but two coupled-but-independent processes (a spawner that produces drainer wallets on a 2-day cadence, and the most recent drainer that operates on its own internal rhythm against pre-staged contracts). The pause-while-draining configuration is the diagnostic surface — a single-process automation could not exhibit this divergence.
- **Status update 2026-05-10: 5 days into the spawn pause, no iter_9 or iter_10.** Forecast windows of May 7 ~10:02 UTC and May 9 ~10:02 UTC both missed. The 2-day-cadence framing now has a 0/2 fresh-forecast record vs 5/5 historical record. iter_8 drainer's contract `0xee80d04303e2…` final count: **151 victims drained**, all `approval_watchlist.drain_detected = 1`, full extraction realized (verified 2026-05-10 post-sync against fresh local DB; reconciles the earlier "143v" figure which was an in-flight snapshot). The hub itself shows no on-chain activity in the 5-day window. **If iter_9 does not spawn by 2026-05-13 (8-day mark), the cadence framing should be considered terminated** — the gap will exceed any historical pause precedent and the operator class will be re-characterized from "scheduled drainer-spawn hub" to "concluded drainer-spawn operation."
- **One non-drainer downstream:** `0xb07b8e9f3907…` (4-15, fleet 1, idle). Failed setup or deferred trap.
- **Lexicon:** documented as the scheduler-layer instance of [Convergent Calibration](lexicon.md#convergent-calibration) — automation precision is itself a behavioral signature distinct from human-driven schedules. The 2026-05-07 pause is a candidate empirical anchor for a future "automation pause as signal" lexicon refinement (deferred until 2+ more instances).
- **Framework note:** This refines the drain-wave reading. Yesterday's analysis assumed 6 distinct drainers represented Convergent Calibration without coordination; the linkage probe shows 2+ of N drainers in the wave are spawned by one hub. The wave is **part-coordinated, part-convergent** rather than fully one or the other. The 2026-05-03 automation signature pushes the coordinated portion further — this hub is not just a single operator with rotating wallets but a *scheduled* operator running a drainer-spawn pipeline. The 2026-05-07 pause does not retract that finding (the prior 5 sub-minute spawns are themselves dispositive); it adds a second observation about the *temporal structure* of the operation.

### Self-Deploying Single-Contract Mass-Drain — typology (promoted to lexicon 2026-05-10, three confirmed instances)
**Three structurally identical events across a 5-day window, three distinct funders, zero funder overlap:**
- **2026-05-06 (Iteration I)**: `0xfbf44e969d4f…` drainer drained 113 victims on Arbitrum through one tier=confirmed contract; funder `0x8c826f795466…` (**org_001 L2 Gas Station**, watchlist HIGH `org_001_gas_station`, 1,296 corpus deployers funded — the only iteration linked to a documented organization). First confirmed direct-org-drainer linkage for org_001.
- **2026-05-07 (Iteration II)**: `0x44a2ee1369c3eecf86f8de7c73c3e3602523a198` drainer drained 37 victims on Arbitrum through one tier=confirmed contract `0x955b2c75efffa1ee9ee54e21e9c5c7cf772fdcb0` over a 6.5h window (10:55 → 17:25 UTC); funder `0x68b8b6d48dc6529d7eb4c7943613e04ba2e5b913` (1 lifetime spawn, watchlist HIGH `single_purpose_funder_44a2ee13`, OLI-clean).
- **2026-05-09 → 2026-05-10 (Iteration III)**: `0x72ed7949080a2c57bfe9788a7970fe39629fc6ca` drainer drained **148 victims** (biggest of the three) through contract `0xa68079da060e...` over a 22-hour window (2026-05-09T10:59 → 2026-05-10T09:49 UTC); funder `0x8c8204b8da3defb2a2f525fa35f5026080963579` (1 lifetime spawn, watchlist HIGH `single_purpose_funder_72ed79`, OLI-clean). **Drainer wallet now watchlist HIGH (`self_deploying_drainer_72ed79_pattern_a_clone_iii`) — added 2026-05-10 locally + production.** Active drainage on the contract continues at the time of this writing.
- **Architecture (identical across all three)**: single-purpose funder → self-deploying drainer (fleet=1) → one tier=confirmed contract → many victims drained via approval-spending in a 6h–24h window.
- **Convergent Calibration confirmed at execution layer.** Three distinct funder identities, one of which (May 6) is org_001 infrastructure and two of which (May 7, May 9-10) are unattributed single-purpose wallets. The architecture is the diffusing object, not the operator.
- **Promoted to lexicon 2026-05-10**: [Self-Deploying Single-Contract Mass-Drain](lexicon.md#self-deploying-single-contract-mass-drain). Forward signal: inter-event interval has been 1-2 days; expect Iteration IV within 1-3 days of Iteration III. If absent for >5 days, the cadence framing weakens.

### Meme-token deployment shop — `0xc43f317e` (RECLASSIFIED 2026-05-10 — bytecode decompiled)
> **[RECLASSIFIED 2026-05-10]** Bytecode of dominant hash `49155b60033de73770...` decompiled (sample contract `0xacfdc090ff9f5b160005bdaacb9a2d1025755baf`, "Kore Agent" / KORE). **Result: verified vanilla OpenZeppelin v5.0.0 ERC-20.** No custom transfer logic, no fees, no blacklist, no delegatecall, no selfdestruct. Constructor takes (name, symbol, initialSupply); just mints to deployer. The operator is a **meme-token deployment shop** — sustained-tempo ERC-20 launchpad, same operator class as the Dragon (`0x2e20b261`). The "100% bytecode concentration / 0 realized extraction" signal is exactly what a vanilla ERC-20 factory template produces. **Watchlist downgraded HIGH → MEDIUM (`meme_token_shop_c43f317e`) on local + prod 2026-05-10.** Original "pre-stage trap warehouse" framing retracted; methodology lesson logged in case file. Off-chain harm (rug-pulls, dump schemes) possible but outside Layer 3's L2 contract-layer surface.
- **Funder:** `0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078` — Base, 2,535 downstream deployer wallets (still active 2026-05-10), OLI-clean.
- **Operating pattern:** 49-day sustained tempo (2026-03-23 → 2026-05-10, ongoing), ~95 contracts/day average. Each deployer produces exactly one vanilla ERC-20.
- **Bytecode confirmed:** OpenZeppelin v5.0.0 ERC-20, Solidity 0.8.25, `OpenAI.sol` filename (intentionally provocative naming).
- **Primary case file:** `surveillance/data/cases/CASE_PRESTAGE_WAREHOUSE_0xc43f317e.md` (reclassified header + reframe).
- **Watchlist:** MEDIUM (`meme_token_shop_c43f317e`) since 2026-05-10.

### Honeypot token operator — `0x8ca70232` (case file 2026-05-10, watchlist HIGH)
- **Funder:** `0x8ca702323c341a8d46ee94a2abeddb08798ca10d` — Base, L2-only, 737 downstream deployer wallets in a 5-day compressed burst (2026-04-11 → 2026-04-16T17:37 UTC), then cold stop. OLI-clean.
- **Bytecode signature (confirmed predatory):** Sample contract `0xaeac0e69f6d2f6d88149cdca003c1689c9ed9eb8` ("Laser Eagle" / 🦅LSEG, 220M supply) decompiled 2026-05-10. **Custom `EVMToken` template (NOT vanilla OZ) with two honeypot primitives:**
  1. **Hardcoded blacklist of 5 addresses** in `_transfer` — buyers whose wallets are in the list cannot sell. Error message is `ERC20InvalidReceiver` (deliberately misleading — the check is on `from`, error names `to`).
  2. **Hidden function `approev(address)`** (deliberate misspelling of "approve") that zeroes any holder's balance with NO `Transfer` event emitted. Access-gated by funder's address (via the misleadingly-named `uniswapV2Router02` constructor argument set to the funder itself).
- **Standing exposure:** 258 victim approvals on 177 of the 320 deployed contracts. 25 unique victim wallets exposed to the `approev` drain primitive. Operator dormant since April 16 but primitives remain callable for the lifetime of each contract.
- **Bytecode classifier hit:** 1 contract `confirmed` + 204 `suspected` + 115 `unknown`. Classifier partially detected the pattern but missed the majority (319 of 320 not at `confirmed` despite carrying the same predatory primitives). Recommendation in case file: add `approev` function-name signature to classifier surface.
- **Primary case file:** `surveillance/data/cases/CASE_HONEYPOT_TOKEN_OPERATOR_0x8ca70232.md` (2026-05-10).
- **Watchlist:** HIGH (`honeypot_token_operator_8ca70232`) added 2026-05-10 to local + production.
- **Top-12 ISO context:** The ONE truly adversarial operator among the four OLI-cleared residual Top-12 entries (`0xc43f317e` / `0x0e6e9177` / `0x8ca70232` / `0xca7ece5e`). The other three are vanilla meme-token shops; 8ca70232's custom template with embedded blacklist + `approev` mechanism is the per-token-customized honeypot variant.
- **Forward signal:** Lexicon entry deferred (one instance is thin); promote on a second instance of the `approev`-style hidden-drain pattern.

### X402 facilitator-drain coordinated endpoint (2026-05-02)
- `0xa7b9874d15742358fb455dd56f97c6d19ad74f5c` — Base recipient of 4 of 7 X402_AGENT_DRAIN alerts in 2026-05-01 → 2026-05-02 window. Total ~$285K USD-equivalent received from two payers (escalating $20K/$101K/$102K USDC pattern + 26.9 ETH-worth). Watchlist HIGH (`x402_drain_endpoint_a7b9874d`). See `cases/CASE_X402_DRAINER_OPERATION.md` activity update 2026-05-01/02.

### Industrial-scale PSO operator (case file pending, 2026-05-01) — **RETRACTED 2026-05-09, Correction #20**
- `0xbb50ce87be3443ed137df1dfdbf2fb0ca8c0a9e0` — **[CORRECTION #20 → Circle: contract deployer]** OLI-tagged as Circle's institutional contract deployer. Optimism deployer, 38,016 lifetime contracts (2026-05-02), 7.7y mainnet age, formerly watchlist HIGH (`pristine_solo_industrial_bb50ce87`). **Pulse-burst pattern**: 6 active days across 19-day lifespan, 13-day historical pause precedent, single 4-day burst window 2026-04-28 → 2026-05-01 deployed 33,016 contracts (peak 13,000 on 05-01). — Case file no longer pending; do not author trap-class case file.
- `0x6a9c2449c32779f89d0ccafd746152e237c1bdf2` — formerly funder watchlist HIGH (`single_purpose_funder_industrial_bb50`). Near-Pattern-A (funds 2, but 99.99% of fleet via bb50ce87). **[CORRECTION #20 — partial]**: not OLI-tagged, but funds a known-legitimate institutional address. Pending identification (Circle-internal hot wallet vs. integrator vs. separate actor). Tier reassessment deferred.

### Mass dormant-wallet drain (EXTRACTION_010, 2026-04-30)
- `0xA707034429c8E4E01df056C0CbCf478F0FBeFAd7` — mainnet drain hub, watchlist HIGH (`mass_dormant_drain_hub_a707`). 22h-old EOA receiving from 49+ distinct senders (5 of 8 sampled = 7+ years old), then bridging 324.741 ETH out via Thorchain. — `CASE_DORMANT_WALLET_DRAIN_20260430.md`
- `0xD37BbE5744D730a1d98d8DC97c42F0Ca46aD7146` — Thorchain mainnet deposit router, watchlist HIGH (`thorchain_router_known_offramp`). Off-ramp for the 324.741 ETH at 15:28:59 UTC. Not malicious itself; flagged for cross-chain laundering signal in our L2 corpus. — `CASE_DORMANT_WALLET_DRAIN_20260430.md`
- Confirmed dormant victims (sample): `0xf44087b7e1CCb36019d231C7AD09ba9BF9783F3b` (6.8y), `0x006ac999c96020ba3e54653b2e98e59e92b8b829` (7.0y), `0x166bf677b8d8bb4efce2eab16dc6ba941ed9d3b3` (7.5y), `0x95ca15e460e3c39a1b81e86c90665c4b35052c55` (8.1y), `0x2a2bad8781ded48e4aa5aadad543e43196492575` (7.0y), `0x3a687fade4857dd7840fb04d8dc3dc66cf7f58ee` (7.0y). — `CASE_DORMANT_WALLET_DRAIN_20260430.md`

### Wasabi Protocol exploit (EXTRACTION_009, 2026-04-30)
- `0x5c629f8c0b5368f523c85bfe79d2a8efb64fb0c8` — wasabideployer.eth, compromised admin EOA (mainnet + Base). Sole `ADMIN_ROLE` holder pre-compromise. — `CASE_WASABI_EXPLOIT_20260430.md`
- `0x02228b0afcdbEdf8180D96Fc181Da3AF5DD1d1ab` — attacker helper contract, deployed at the **same address on both Ethereum mainnet and Base** via CREATE2. Funded with 5.09 ETH (mainnet) and 1.172 ETH (Base) by the admin EOA shortly before drains. — `CASE_WASABI_EXPLOIT_20260430.md`
- `0xc0b01a4f4A4459D5A7E13C2E8566CDe93A010e7D` — Wasabi role manager (mainnet). `grantRole`/`revokeRole` target during the attack. — `CASE_WASABI_EXPLOIT_20260430.md`
- `0xeC3e4E0FDB50411F4C5ee9f75436d8b20CF7D70E` — Wasabi role manager (Base). — `CASE_WASABI_EXPLOIT_20260430.md`
- `0xEe5c45DCB0064f9B097edBC5d8adfcE23baaC03b` — Wasabi vault (mainnet, observed in trace; `setWithdrawFeeBips` + `setFeeReceiver`). — `CASE_WASABI_EXPLOIT_20260430.md`
- `0xfAe69F2C82747F878F74C1E57a1AeD945eD8558F` — Wasabi vault (Base, observed in trace). — `CASE_WASABI_EXPLOIT_20260430.md`

### Renegade Dark Pool proxy compromise (2026-05-10)
- `0x777253F28AdC29645152b7b41BE5c772A9657777` — attacker EOA, Arbitrum. First tx 2026-02-03 (3-month pre-positioning). Exploitation burst 2026-05-10 16:51 → 17:14 UTC (10+ token `transfer` sweeps). Tier A. — `CASE_RENEGADE_EXPLOIT_20260510.md`
- `0x30bD8eAb29181F790D7e495786d4B96d7AfDC518` — Renegade Darkpool Proxy (EIP-1967), Arbitrum. OLI public tag `"Darkpool proxy"`. Created 2024-09-03 by Renegade deployer. Tier A. — `CASE_RENEGADE_EXPLOIT_20260510.md`
- `0xc038933d0b33359f5C87B4B2f92Ee0DAd11EaDc5` — pre-attack implementation, Arbitrum. Created 2025-05-21 by `0x812922c33079c3E2324D25Ef0352a2220686C2Ac`. Tier A. — `CASE_RENEGADE_EXPLOIT_20260510.md`
- `0x58f876aAeeCBD5a0fca8F87e1313a9188C155bcC` — post-attack "DarkpoolFrozen" implementation (emergency-freeze swap), Arbitrum. Tier A. — `CASE_RENEGADE_EXPLOIT_20260510.md`
- `0x98e4e5C6223bb2Cc945a7c2821E30929dEff3568` — Renegade deployer (proxy creator), Arbitrum. Tier A. — `CASE_RENEGADE_EXPLOIT_20260510.md`

### Private-key drain — attacker `0xF7cFFC27` (2026-05-11)
- `0xF7cFFC27732a5C9c4E2D592F3E33435F8dDb019A` — attacker EOA, multi-chain (Base / BSC / Ethereum). Watchlist HIGH (`private_key_drain_attacker_F7cFFC27`) row 95 local + prod (2026-05-10). Tier A. — `CASE_PRIVATE_KEY_DRAIN_F7CFFC27_20260511.md`
- `0x62acE10c7f2Aa0e9B5a8e09CbF5D18d0f8a1EE8A` — victim wallet (compromised private key, multi-chain). Tier A. — `CASE_PRIVATE_KEY_DRAIN_F7CFFC27_20260511.md`
- Sigma forensic-anchor tx: `0xb81f9f0a1abb2330763d7b9498185404277955a18b3f766a31582c83ba70047e` (Base) — EIP-7702 delegation to null demonstrating attacker signing control. Tier A. — `CASE_PRIVATE_KEY_DRAIN_F7CFFC27_20260511.md`

### Grok / Bankr AI-wallet permission chain attack (2026-05-04)
- `0xB1058c959987E3513600EB5b4fD82Aeee2a0E4F9` — Grok victim wallet (sender of the 3B DRB transfer), Base. Tier A — verified via tx `0x6fc7eb7da93793…e525739a`. — `CASE_GROK_BANKR_EXPLOIT_20260504.md`
- `0xE8E476bdd78b0aA6669509eC8d3E1c542d5A686B` — attacker recipient on Base (ilhamrafli.base.eth resolved). EIP-7702 Kernel smart account; first tx 2025-04-03. Tier A. — `CASE_GROK_BANKR_EXPLOIT_20260504.md`
- `0x3ec2156D4c0A9CBdAB4a016633b7BcF6a8d68Ea2` — DRB token contract ("DebtReliefBot"), Base, ERC-20. The drained asset. Tier A. — `CASE_GROK_BANKR_EXPLOIT_20260504.md`
- Principal extraction tx: `0x6fc7eb7da9379383efda4253e4f599bbc3a99afed0468eabfe18484ec525739a` (Base block 45543997, 2026-05-04T06:49:01 UTC). Tier A. — `CASE_GROK_BANKR_EXPLOIT_20260504.md`

The remaining ~80 distinct addresses extracted from cases/ + reports/ are either victim/bot addresses without role attribution or appear in single auto-generated `CASE_0x*` files only. They are not enumerated here unless they have a documented role.

---

## Section 3 — Patterns and Hypotheses

### Documented (active framework concepts)
- **Stored Potential** — DOCUMENTED — `docs/lexicon.md#stored-potential`
- **Thermodynamic Fundamentalism** — DOCUMENTED 2026-05-02 — `docs/lexicon.md#thermodynamic-fundamentalism`. Framework-level. Replaces social-consensus value measurements with physical-substrate measurements (CER as ROI replacement). Empirical grounding now includes Layer 3 corpus instances (bb50 stockpile CER ≈ 0; 0xf7883e3fef23 / 0x3304e22ddaa2 hubs CER positive). Cross-refs Stored Potential + Compositional Harm + Neutrality Trap + Forced Deterministic Neutrality (bidirectional refs added).
- **Neutrality Trap** — DOCUMENTED 2026-05-02, refined and authored 2026-05-02 — `docs/lexicon.md#neutrality-trap`. Four-phase structure (Promise → Selection → Predation → Override). Empirical anchor: April 2026 cluster (EXTRACTION_006-010).
- **Forced Deterministic Neutrality** — DOCUMENTED 2026-05-02, refined and authored 2026-05-02 — `docs/lexicon.md#forced-deterministic-neutrality`. Three key characteristics (no context window, no pause/override, no intent parsing). Six empirical examples spanning EVM / ECDSA / UUPS / Permit2 / OAuth / USD wires.
- **Normative Shell Game** — DOCUMENTED 2026-05-02 — `docs/lexicon.md#normative-shell-game`. Two-layer governance posture (public Shell + Emergency Core) that emerges as the structural response to the Neutrality Trap. Empirical anchor: Arbitrum Security Council freezing KelpDAO funds (2026-04-20), The DAO Fork (2016), USDC/USDT freeze authority.
- **Confused Deputy Problem** — DOCUMENTED 2026-05-02 — `docs/lexicon.md#confused-deputy-problem`. Three-role structure (Principal → Deputy → Attacker). The per-program vulnerability that Forced Deterministic Neutrality produces and the Neutrality Trap makes ecosystem-wide. Includes Agentic AI Supercharger sub-section. Empirical anchors: AI coding agents, Permit2 (EXTRACTION_010), Wasabi UUPS (EXTRACTION_009), Vercel/Context.ai OAuth.
- **Distributed Confused Deputy Chain** — DOCUMENTED 2026-05-10 — `docs/lexicon.md#distributed-confused-deputy-chain`. Multi-contract systemic form of the Confused Deputy Problem in modular protocols. Three conditions (fragmented epistemic state, hardcoded trust bindings, single point of syntactic failure). Empirical anchors: Renegade Dark Pool Proxy (2026-05-10, Arbitrum, `0x30bD...DC518`), Wasabi UUPS (EXTRACTION_009, 2026-04-30), Grok/Bankr cross-domain (2026-05-04, Twitter → Grok → Bankr → Base).
- **Adversarial Topology Framework** (5 primitives: position, permissions, trust bindings, mutability, observation capability) — DOCUMENTED — `docs/lexicon.md#adversarial-topology` + `claude.md` §Adversarial Topology Framework
- **Compositional Harm** — DOCUMENTED — `docs/lexicon.md#compositional-harm`
- **Cross-Domain Compositional Harm** — DOCUMENTED 2026-04-25, extended 2026-05-10 — `docs/lexicon.md#cross-domain-compositional-harm`. Anchors: Vercel/Context.ai breach (off-chain, 2026-04-19), Wasabi Protocol admin-key compromise (on-chain, 2026-04-30, EXTRACTION_009 — `cases/CASE_WASABI_EXPLOIT_20260430.md`), and Grok/Bankr permission chain attack (substrate-bridging, 2026-05-04 — `cases/CASE_GROK_BANKR_EXPLOIT_20260504.md`). The Grok/Bankr instance is the strongest *bridging* anchor: Twitter → Grok → Bankrbot → Base, each deputy correct under local rules.
- **Trust Amplification Factor** — DOCUMENTED with methodological caveat — `docs/lexicon.md#trust-amplification-factor`. **Two contradictory retractions of the 14.2× anchor figure exist** (`CORRECTIONS.md` 2026-04-02 vs `reports/correction_log.md` Correction #17 2026-04-25). Resolution open.
- **Camouflage Ratio** — DOCUMENTED with methodological caveat (cluster-dominance impact) — `docs/lexicon.md#camouflage-ratio`. Original 14.2× claim retired (`CORRECTIONS.md`); equilibrium framing requires top-12-excluded re-run.
- **Behavioral Laundering** (Patterns A–F) — DOCUMENTED — `docs/lexicon.md#behavioral-laundering`
  - Pattern A — Reputation-Building Sacrifices — DOCUMENTED — 4 candidates as of 2026-04-18
  - Pattern B — Temporal Pattern Normalization — DOCUMENTED — 0 candidates (corpus too young)
  - Pattern C — Funding Chain Laundering — DOCUMENTED — 4 relaxed candidates
  - Pattern D — Cross-Chain Reputation Import — DOCUMENTED — strongest validated (54 of 100)
  - Pattern E — Fake Legitimate Projects — METHODOLOGY-ONLY — not yet scanned
  - Pattern F — Advisor-Parasite Pattern — DOCUMENTED — 0 candidates (corpus too young)
- **Pristine Solo Operator** — DOCUMENTED 2026-04-25, bidirectional refinement 2026-04-30 — `docs/lexicon.md#pristine-solo-operator`. Detector: `surveillance/pristine_solo_detector.py`. Refinement: same 7+ year aged wallet class is bidirectionally exploited — as **operators** (PSO original framing) and as **victims** (EXTRACTION_010 mass dormant-drain). Detection signal common to both: first activity from a long-dormant wallet.
- **Infrastructure-Scale Operator** — DOCUMENTED 2026-04-25 — `docs/lexicon.md#infrastructure-scale-operator`. Detector: `surveillance/infrastructure_operator_detector.py`.
- **Adversarial Vanity Branding** — DOCUMENTED 2026-04-27 — `docs/lexicon.md#adversarial-vanity-branding`. Three sub-categories (operational / anti-forensic / funder), three corpus instances (Coffee Fleet, org_001 Shadow Wallet 1, `0xb0b0b69*`).
- **Tuition Extraction Markets** — DOCUMENTED 2026-04-25, anchor re-sourced 2026-04-29 — `docs/lexicon.md#tuition-extraction-markets`. Anchor: `0xc0dec76000f6c2d32f23d523748e50ebb5bb34a3` (corpus-derived; replaces the retired `0x84792c2a` external-block-walking anchor per Correction #18). Search log: `reports/tuition_extraction_anchor_search_2026-04-29.md`.
- **Single-Purpose Infrastructure Funder** — DOCUMENTED 2026-04-28 — `docs/lexicon.md#single-purpose-infrastructure-funder`. Case file: `cases/CASE_SINGLE_PURPOSE_INFRASTRUCTURE_FUNDER.md`. 69 Pattern A operators across Base/Arbitrum/Optimism, three deployment shapes, structurally independent of org_001-004.
- **Protocol-Family Specialist Operator** — DOCUMENTED 2026-05-07 — `docs/lexicon.md#protocol-family-specialist-operator`. Within-ecosystem analog of Pattern D (Cross-Chain Reputation Import). Same operator EOA exploits **different vulnerability classes** within a single protocol family / trust-graph over time. Anchor: EXTRACTION_011 — `0xC3EBDdEa4f69df717a8f5c89e7cF20C1c0389100` exploited 1inch Fusion V1 (March 2025) and TrustedVolumes RFQ proxy (2026-05-06), 14-month gap, different bug classes. Off-chain only (mainnet); typology transfers to L2 corpus.
- **Convergent Calibration** — DOCUMENTED 2026-04-29 — `docs/lexicon.md#convergent-calibration`. Meta-observation across three operator scales (funder, operator, execution layers): N independent actors converge on the same template at the same time, no observable coordination signal. Empirical anchors: epistemic test #2 A4 (66/66 zero pair overlap among top-12 funders), `cases/CASE_SINGLE_PURPOSE_INFRASTRUCTURE_FUNDER.md` (69 unrelated Pattern A operators), 2026-04-29 24h drain-burst (6 self-deploying trap operators in 9-hour window, no shared infrastructure).
- **Pooled Custody Amplification** — DOCUMENTED — `docs/lexicon.md#pooled-custody-amplification`
- **Verification-Path Trust Failure** — DOCUMENTED — `docs/lexicon.md#verification-path-trust-failure`
- **Configuration-Level Vulnerability** — DOCUMENTED — `docs/lexicon.md#configuration-level-vulnerability`. April 2026 cluster synthesis: `reports/april_2026_key_management_cluster.md` (5 of 5 EXTRACTION_006-010 incidents in this class). **Extended 2026-05-07** to 6 of 6 with EXTRACTION_011 (TrustedVolumes RFQ proxy, 1inch family) — same configuration-class shape (signed-quote acceptance scope, not bytecode defect).
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
| EXTRACTION_009 | 2026-04-30 | Ethereum + Base + Berachain + Blast (`monitored_chain=1`) | ~$4.5–5.5M | `surveillance/data/cases/CASE_WASABI_EXPLOIT_20260430.md` | CONFIRMED. Wasabi Protocol admin-key compromise. UUPS proxy upgrade attack via compromised wasabideployer.eth. Configuration-Level Vulnerability — same shape as 006/007/008. **Layer 3 had zero corpus coverage** — Wasabi predates 2026-03-17 monitoring window AND production ingest was stuck during attack. |
| EXTRACTION_010 | 2026-04-30 | Ethereum mainnet (`monitored_chain=0`) | ~$733K visible (324.741 ETH off-ramp; full scale across token paths TBD) | `surveillance/data/cases/CASE_DORMANT_WALLET_DRAIN_20260430.md` | CONFIRMED. Mass dormant-wallet drain — single hub `0xA707034429c8…` received from 49+ distinct senders in 3.5-min burst (06:39-06:42 UTC), 5 of 8 sampled senders are 7+ years old, off-ramp via Thorchain at 15:28:59. Probable mass-key-compromise event (vector unconfirmed). Inverts Pristine Solo Operator framing (same aged-wallet class, opposite role: victims not operators). **Layer 3 had zero corpus coverage** — mainnet-only, outside L2 scope. |
| EXTRACTION_011 | 2026-05-06 | Ethereum mainnet (`monitored_chain=0`) | ~$5.87M (1,291.16 WETH + 206,282 USDT + 16.939 WBTC + 1,268,771 USDC) | (off-chain reference — Blockaid disclosure; no Layer 3 case file, see scope-gap note below) | CONFIRMED. TrustedVolumes / 1inch RFQ proxy exploit. Victim contract: TrustedVolumes resolver `0x9bA0CF1588E1DFA905eC948F7FE5104dD40EDa31`. Vulnerable component: TrustedVolumes-controlled custom RFQ swap proxy `0xeEeEEe53033F7227d488ae83a27Bc9A9D5051756`. Exploiter: `0xC3EBDdEa4f69df717a8f5c89e7cF20C1c0389100`. Exploit tx: `0xc5c61b3ac39d854773b9dc34bd0cdbc8b5bbf75f18551802a0b5881fcb990513`. **Same operator as the March-2025 1inch Fusion V1 incident; different vulnerability class** — anchors the new [Protocol-Family Specialist Operator](`docs/lexicon.md#protocol-family-specialist-operator`) lexicon entry (2026-05-07). Configuration-Level Vulnerability cluster grows to **6 of 6** (006-011). **Layer 3 had zero corpus coverage** — mainnet-only, outside L2 scope. |

The off-chain events (004, 005, 006) are corpus-expansion case studies — `monitored_chain=0`. EXTRACTION_001 / EXTRACTION_002 are referenced in `claude.md` Database Schema as table rows but have no case-file or report-file documentation in current corpus state. **Gap flagged.**

### Scope-gap note (added 2026-05-07)

**Mainnet RFQ / resolver / admin-key compromises are structurally outside Layer 3's monitoring scope.** Three of the last four extraction events (009 Wasabi, 010 mass-dormant-drain, 011 TrustedVolumes) carry the marker **"Layer 3 had zero corpus coverage"**. This is not a detection failure — it is a scope statement: Layer 3 monitors Base / Arbitrum / Optimism deployments, and the entire **Configuration-Level Vulnerability** cluster (006-011) overwhelmingly originates on Ethereum mainnet against protocol-tier infrastructure (resolvers, vaults, OFT adapters, multisig admin EOAs) that does not deploy contracts on the L2s we watch.

Implication for forward documentation:
- The Configuration-Level Vulnerability cluster will continue to grow in this off-chain reference column without producing Layer 3 corpus detections.
- Each new event in this class should be logged as an extraction-events-table entry plus (optionally) a case file under `surveillance/data/cases/`, but should NOT be expected to surface in `contracts`, `transaction_events`, or `alerts`.
- Cross-referencing these events into the lexicon (e.g., as anchors for Configuration-Level Vulnerability, Confused Deputy Problem, Forced Deterministic Neutrality, Protocol-Family Specialist Operator) remains valuable — the typology work transfers to the L2 corpus even when the events themselves do not.
- Any future "what is Layer 3 missing on mainnet?" question is answered structurally, not as a gap to close: closing it would require mainnet ingest, which is an architectural decision (Alchemy budget, RPC scope, schema multi-chain implications) deferred indefinitely per `claude.md` infrastructure constraints.

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
- **`476b15536fa9703e2c630e91ac976c514e1868a70e8c996f1bf8bb97a9b9e532`** — bytecode hash, NOT yet a registered family. **5,775 contracts** by **5,775 distinct deployers** (1 contract each, fully disposable), all on **Optimism**, all `tier=unknown`, all `has_*` flags zero (static classifier doesn't recognize the trap pattern). 99.86% (5,767 of 5,775) funded by `0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07`. This is the campaign that drove the 8,797-alert COORDINATED_DEPLOYMENT surge on 2026-04-25/26. Pre-stage trap stockpile shape resembling org_002 scaled 16.5× and ported to Optimism. Status: **mass-deployment in progress**, no detection signal yet, no traffic yet. Worth bytecode_families clustering and ongoing surveillance.

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
