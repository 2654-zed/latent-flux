# Layer 3 Session Memory

**Purpose:** Working memory for Claude across sessions. Read this first on session start. Update at end of each session. Survives context resets.

**Last updated:** 2026-05-10
**Last session focus:** Correction #20 (OLI mass mislabel sweep) + full closure of all 11 follow-up items, including bytecode decompilation of the 4 unattributed Top-12 ISO entries and discovery of honeypot operator `0x8ca70232`.

---

## How to use this file

- Read top-to-bottom at session start. The "Current state snapshot" and "Most recent corrections" sections are the freshest.
- Code-search the corpus instead of trusting memory of file paths — paths change.
- Update at session end: append to "Session log" with date, key findings, and any state transitions. Move stale "open work" items to "closed" once resolved.
- Do not delete historical entries. The file is append-mostly to preserve continuity.

---

## Project overview

**Layer 3 (L3)** is a Base/Arbitrum/Optimism trap-detection surveillance system. It ingests every contract deployment across three L2s via Alchemy WebSockets, classifies bytecode at deployment time, accumulates behavioral data over time, maps organizational structures through fund-flow analysis, scores stored potential via a multi-component risk model, and serves intelligence through an HTTP API.

- **Local dev path:** `C:\Users\jason\Desktop\ai lang\`
- **Git remote:** `origin = https://github.com/2654-zed/latent-flux.git`
- **Production:** Railway, service `stellar-embrace` in project `blockchain`, URL `stellar-embrace-production-2020.up.railway.app`
- **Trading sister project:** `C:\Users\jason\Desktop\Trading\layer3_trading_exp` — reads from L3's local SQLite copy

### Critical project documents

| Doc | Purpose |
|---|---|
| `CLAUDE.md` | Top-level project rules + current corpus state (kept current) |
| `claude.md` | Lowercase symlink/duplicate — Git case-sensitivity quirk on Windows; both exist, both used |
| `docs/INDEX.md` | Master index of all documented entities. The source of truth for "what do we know about address X" |
| `docs/lexicon.md` | Living typology dictionary — Pattern A, Pristine Solo Operator, Infrastructure-Scale Operator, etc. |
| `reports/correction_log.md` | Numbered corrections to prior findings. Correction #20 is the watershed event (2026-05-09 OLI mass mislabel sweep). |
| `surveillance/data/cases/` | Case files per investigated entity. Some are gitignored, some force-tracked. |
| `memory.md` | This file. |

---

## Current corpus state (as of 2026-05-10)

Pulled fresh from production via `scripts/sync_prod_db.py` on 2026-05-10:

| Metric | Value |
|---|---|
| Total contracts | 284,777 |
| Confirmed traps | 1,404 |
| Suspected traps | 115,514 |
| Unique deployers | 67,459 |
| Transaction events | 16,810,247 |
| Bot candidates | 4,244 |
| Funder coverage | 91.9% (61,375 of 66,805 traced) |
| Gas stations identified | 253 |
| Org links found | 4,382 |
| Cross-chain shared deployers | 1,046 |
| Local DB size | 10.0 GB |

**Active monitors:** deployment_monitor running on Base, Arbitrum, Optimism. Heartbeat checked via production `/stats` endpoint.

**API surface (as of 2026-05-10):** stripped read-only slices, NOT the 19-endpoint Tier 1/2/3/A surface from prior architecture. Confirmed endpoints: `/stats`, `/suspected`, `/priority`, `/bots`, `/tx-events`, `/known-selectors`, `/clusters`, `/cluster-events`, `/health`. NO `/dump`, NO `/risk/{addr}`, NO per-address lookup. (CLAUDE.md note on this: section needs reconciliation pass against the new surface.)

---

## Production sync mechanism

**The `/dump` endpoint never existed on Railway** — earlier false premise corrected on 2026-05-09. The actual sync path:

```bash
# Local linkage (one-time per workstation):
railway link --project blockchain
railway service stellar-embrace

# Sync (idempotent, takes 10-30 min for 10 GB):
python scripts/sync_prod_db.py
# Or: python scripts/sync_prod_db.py --dry-run  (validate without replacing local DB)
```

**Mechanism:** `sync_prod_db.py` (local) → reads `sync_prod_db_remote.py`, base64-encodes it → invokes `railway ssh python3 -c "import base64; exec(base64.b64decode('<b64>'))"` → remote runs SQLite `online backup → gzip → base64 → stdout` framed by `L3SYNC_PAYLOAD_START`/`_END` markers → local captures stdout to temp file, mmap-searches markers, chunked base64-decode + gunzip → validates via `PRAGMA integrity_check` + table counts → atomic rename, prior DB retained as `.bak`.

**Empirically verified gotchas:**
- Railway SSH transport does CRLF translation on binary bytes (4 stray byte changes per 1024 random bytes observed). MUST use base64. Cannot stream raw binary.
- Stderr from remote is merged into stdout by railway ssh. Markers must let the local parser trim banner content.
- Windows cmd.exe command-line limit is 8,191 chars. Any base64-bootstrapped script larger than ~6 KB hits the limit; for those, use stdin: `railway ssh "python3 -" < script.py` with `MSYS_NO_PATHCONV=1` set.
- Python `subprocess.Popen` on Windows can't directly execute `.cmd`/`.bat` files; wrap with `cmd.exe /c`. The `sync_prod_db.py` script already handles this.

**Apply-changes-to-prod path:** Use `scripts/apply_correction_20_to_prod.py` template — bash heredoc or short inline scripts via `railway ssh`. For longer scripts use `scripts/apply_correction_20_via_ssh.py` (compact-script variant that fits the cmd.exe limit). See Correction #20 follow-up commits for the pattern.

---

## Architecture (key components)

### Database
- **Local path:** `surveillance/data/surveillance.db` (10.0 GB)
- **Production path:** `/app/surveillance/data/surveillance.db`
- **Schema:** `surveillance/db.py` — `init_db()` is idempotent, runs migrations on startup
- **Backups:** `.bak` retained after each successful sync

### Core modules (in `surveillance/`)
| Module | Purpose |
|---|---|
| `db.py` | Schema + migrations. ~1700 lines. `_log_migration` records each migration outcome. |
| `deployment_monitor.py` | Ingests contract deployments from chain WebSockets |
| `bytecode_classifier.py` | Static analysis on deployed bytecode. ~640 lines. `PATTERN_REGISTRY` is the detector list — 10 detectors as of 2026-05-10 (added `detect_hidden_drain_function` for the `approev` signature). |
| `entity_classifier.py` | Address-level classification with OLI guardrail (added 2026-05-09 per Correction #20). `_OLI_GUARDED_TRAP_SUBTYPES` lists subtypes that auto-redirect to COMMERCIAL/institutional_oli_tagged if address has HIGH-severity OLI tag. |
| `oli_enrichment.py` | Open Labels Initiative tag lookup via Blockscout metadata service. Caches in `oli_labels` table. CLI: `--address X`, `--backfill-flagged`, `--backfill-watchlist`, `--hits`. |
| `auto_funder_tracer.py` | 1 Alchemy call per new deployer for funding trace. Writes to `deployers.funding_trail`. |
| `vanity_detector.py` | Tags vanity addresses on first sight |
| `infrastructure_operator_detector.py` | Surfaces high-fanout funder candidates (the source of the original "Top-12 ISO" cluster — most now retracted via Correction #20) |
| `pristine_solo_detector.py` | Surfaces aged-mainnet wallets first appearing on L2 (false-positive class now documented per Correction #20) |

### Key tables in the DB
| Table | Rows | Purpose |
|---|---|---|
| `contracts` | ~285K | Every deployment, with confidence tier + bytecode flags |
| `deployers` | ~67K | Unique deployer addresses, with `funding_trail` JSON + mainnet age |
| `transaction_events` | 16.8M | Per-tx events |
| `bot_candidates` | 4.2K | Trap-scanner bots |
| `watchlist` | ~91 active | Manually-curated HIGH/CRITICAL/MEDIUM/LOW priority entries |
| `entity_classification` | 1.08K | Category/subtype taxonomy (CRIMINAL, BOT, COMMERCIAL, INFRASTRUCTURE, etc.) |
| `infrastructure_registry` | 12 | Known-legitimate institutional contracts (Circle CCTP v2, etc.) |
| `infrastructure_operator_candidates` | 12 | Top-12 ISO candidates. 7 now marked `rejected_oli_correction_20`. |
| `approval_watchlist` | ~46.7K | Tracked victim approvals to confirmed/suspected contracts |
| `trap_events` | 2,565+ | Confirmed trap firings (bot got drained) |
| `oli_labels` | 142 cached | OLI tag cache — populated by `oli_enrichment.py` |
| `eth_traces` | varies | Mainnet trace data for Pattern D / cross-chain reputation imports |
| `org_wallets` | 53 | Confirmed org_xxx wallets |

### Scripts (in `scripts/`)
| Script | Purpose |
|---|---|
| `sync_prod_db.py` + `sync_prod_db_remote.py` | Production sync (see above) |
| `apply_correction_20_to_prod.py` | Local Python orchestrator for Correction #20 dispositions on prod |
| `apply_correction_20_via_ssh.py` | Compact SSH-dispatched variant (under cmd.exe char limit) |
| `blockscout_tag_audit.py` | One-shot OLI tag audit — produced `reports/blockscout_tag_audit_2026-05-09.csv` |
| `funder_metrics.py` | Canonical funder cluster metrics (live) |
| `delta_sync_from_railway.py` | STALE — pointed at `/dump` endpoint that never existed. Do not use; sync_prod_db.py replaces it. |

---

## Correction #20 — the watershed event (full summary)

**Date:** 2026-05-09 (initial sweep) + 2026-05-10 (items 7-11 follow-up)
**Trigger:** User asked Claude to investigate `0xbb50ce87` "PSO+Single-Purpose hybrid" framing. Manual OLI lookup revealed it's Circle's contract deployer. Authorized full corpus audit.
**Method:** New `scripts/blockscout_tag_audit.py` queried Blockscout metadata service (which aggregates OLI tags) for all 140 unique malicious-flagged addresses (watchlist + entity_classification CRIMINAL subset + infrastructure_operator_candidates).
**Outcome:** 18 of 140 (12.9%) carry public institutional tags incompatible with their adversarial classification.

### What was retracted (HIGH-severity, 10 deactivated)

| L3 framing (PRIOR) | Reality (OLI) |
|---|---|
| `0xbb50ce87` — Pristine Solo Industrial Operator (49K contracts on Optimism) | Circle: contract deployer (issuer of USDC) |
| `0x3304e22d` — Drainer-spawn hub + Top-12 ISO rank #4 | Binance 73 / Exchange |
| `0x39591e7c` — Top-12 ISO rank #10 | OKX 177 / Exchange |
| `0x4e3ae00e` — Top-12 ISO candidate | MEXC 15 / Exchange |
| `0xfd92f4e9` — Top-12 ISO rank #3 stockpile | OKX 137 / Exchange |
| `0xbaed383e` — org_004 / CLAUDE.md priority #12 | Bybit: Hot Wallet 6 |
| `0xf70da978` — org_001 whale trader / 68% of deployments | Relay: Solver bridge |
| `0xe69f81b8` — 49,000 ETH "coordinated" bridge user | Binance Internal 2 |
| `0x151b3810` — Architect's "sole funder" CRITICAL | MoonPay 4 (fiat onramp) |
| `0x45a31827` — Cluster B funder CRITICAL | Owlto Finance Bridge |
| `0xe4edb277` — Cluster A funder CRITICAL | Orbiter Finance Bridge 2 |
| `0x80c67432` — Top-12 ISO candidate | Orbiter Finance Bridge |

### LOW-severity (4 noted, kept active pending second-source verification)

- `0x80b12bd0` — Animoca: Deployer (was Pristine Solo)
- `0xa2a01b4a` — Stabilize Finance: Deployer 2 (was Pristine Solo)
- `0x147b8869` — Luchadores: Deployer (was Pristine Solo)
- `0xc5d133296e` — CryptoCauses: Deployer (was Architect alternate, 0.742 similarity)

OLI provenance: each carries a `tooltipUrl` to a verifiable Web3 project (revvmotorsport.com, stabilize.finance, luchadores.io, crypto4ac.com) but no `tooltipAttribution` (Blockscout-curated, not OLI-consortium-attested) → lower confidence than HIGH-severity entries.

### Items 7-11 follow-up (2026-05-10)

- **Item 7:** Decompiled `0xc43f317e`'s dominant bytecode. Verified vanilla OpenZeppelin v5.0.0 ERC-20. Reclassified from "pre-stage trap warehouse" (HIGH) to "meme-token deployment shop" (MEDIUM). Case file `cases/CASE_PRESTAGE_WAREHOUSE_0xc43f317e.md` reframed with `[RECLASSIFIED]` header.
- **Item 8:** Bytecode-reviewed all 4 unattributed Top-12 entries:
  - `0xc43f317e`, `0x0e6e9177`, `0xca7ece5e` = vanilla OZ ERC-20 meme-token shops
  - `0x8ca70232` = **HONEYPOT TOKEN OPERATOR** with two predatory primitives (see below)
- **Item 9:** April-16 simultaneous stop of `0x8ca70232` and `0xca7ece5e` — coincidence (zero overlap, different burst shapes, different templates, ~9.5h apart on the stop day).
- **Item 10:** Added `detect_hidden_drain_function` to `bytecode_classifier.py`. Detects `approev(address)` selector `0x3ed67ecd` via Solidity dispatch pattern `63 3ed67ecd 14`. Unit-tested. Live for new deploys; backfill against existing fleet deferred.
- **Item 11:** Cross-referenced 5 hardcoded blacklist addresses + 25 unique victim approvers. **Zero matches** across all Layer 3 tables. Honeypot victims are fresh-to-corpus — separate population than drain victims.

### The big finding (Items 7-11 net result)

**Of the original Top-12 Infrastructure-Scale Operator cluster: zero entries retain the documented framing.**

| Disposition | Count |
|---|---|
| Retracted as CEX/bridge | 7 |
| Vanilla meme-token shops (false positives) | 3 |
| **Confirmed honeypot operator (genuine adversarial)** | **1 — `0x8ca70232`** |
| Pre-attributed within L3 | 4 |

The detector signal (high-fanout disposable-deployer topology) cannot, on topology alone, distinguish institutional flow / token shop / honeypot / trap stockpile. **Bytecode-level disambiguation is load-bearing.**

### Honeypot operator `0x8ca70232` — the one real find

- 737 ERC-20 contracts deployed on Base in 5-day burst (2026-04-11 → 2026-04-16T17:37 UTC), cold stop since
- Custom `EVMToken.sol` template, NOT vanilla OZ
- **Primitive 1: hardcoded blacklist of 5 addresses in `_transfer`** — buyers cannot sell. Error message `ERC20InvalidReceiver` is misleading (check is on `from`, error names `to`).
- **Primitive 2: hidden `approev(address)` function** (typo of "approve") — zeroes any holder's balance with no Transfer event emitted. Access-gated by funder's address (passed in as misleadingly-named `uniswapV2Router02` constructor argument).
- 258 standing victim approvals on 177 contracts (25 unique victims). Operator dormant but `approev` still callable for the life of each contract.
- Case file: `surveillance/data/cases/CASE_HONEYPOT_TOKEN_OPERATOR_0x8ca70232.md`
- Watchlist: HIGH `honeypot_token_operator_8ca70232` (local + prod)

---

## Lexicon entries (current state, post-Correction-#20)

Located in `docs/lexicon.md`. Major entries:

| Entry | State |
|---|---|
| Pristine Solo Operator | Refined 2026-05-09: institutional-deployer false-positive class added; OLI cross-check now required at promotion |
| Single-Purpose Infrastructure Funder | Refined 2026-05-08 (Correction #19): both funder-layer AND downstream-layer checks required |
| Infrastructure-Scale Operator | Major revision 2026-05-09: Top-12 anchor 50%+ contaminated by CEX/bridge; detection rule now requires OLI cross-check |
| Convergent Calibration | Funder-layer anchor RETRACTED; operator-, execution-, scheduler-layer anchors retained |
| Thermodynamic Fundamentalism | bb50 stockpile and `0x3304e22d` positive-CER hub examples RETRACTED. Replacement anchors: `0xf7883e3f` (drainer-spawn hub) + Coffee Fleet `0xc0ffeefeed8b` |
| Pattern A Clone Cadence (Self-Deploying Drainer) | 3 confirmed instances (May 6 `0xfbf44e96`, May 7 `0x44a2ee13`, May 9-10 `0x72ed7949080a`). Lexicon-promoted 2026-05-10. |

---

## Watchlist quick reference

### HIGH / CRITICAL (post-Correction-#20)

**Drainer-spawn hubs:**
- `0xf7883e3fef23c8e645deba4b540549d78028a616` — drainer-spawn hub, 6+ iterations, sub-minute automation precision, 859+ victims drained (Base). Watchlist HIGH `drainer_spawn_hub_f7883e3f`.

**Pattern A clone drainers (self-deploying single-contract):**
- `0xfbf44e969d4fc5cbad62870207341c976f9e38f9` (May 6, 113 victims, funded by org_001 gas station)
- `0x44a2ee1369c3eecf86f8de7c73c3e3602523a198` (May 7, 37 victims, funded by single-purpose `0x68b8b6d4`)
- `0x72ed7949080a2c57bfe9788a7970fe39629fc6ca` (May 9-10, 148 victims, funded by single-purpose `0x8c8204b8`)

**Org infrastructure (active):**
- `0x8c826f795466e39acbff1bb4eeeb759609377ba1` — org_001 L2 Gas Station (Coinbase-funded). 1,296 corpus deployers funded. OLI-clean.
- `0x238d7170f309a55b87a144a341bd6105897082ca` — org_002 senior treasury. 98% suspected ratio.
- `0xde8eb937cb5475eee5ac96dce6ba2d18e439c473` — org_002 junior. 938 deployers.

**Architect (still under investigation):**
- `0x9209c9f7dcb61937f1ec8160c22c0b2365079474` — primary deployer, 21 R&D contracts on Arbitrum, 4-mechanism weapon (SD+DC+TS+CL). CRITICAL.
- Funder `0x151b381058f9` RETRACTED as MoonPay per Correction #20.
- Behavioral match `0xc5d133296e` RETRACTED as CryptoCauses per Correction #20.

**Honeypot operators:**
- `0x8ca702323c341a8d46ee94a2abeddb08798ca10d` — `approev`-mechanism honeypot, 737 contracts, dormant since 2026-04-16 (added 2026-05-10).

**Coffee Fleet:**
- `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e` — 322 fleet, 142 confirmed traps, 84 c0ffee vanity bots. Now also the Thermodynamic Fundamentalism positive-CER anchor.

**Adversarial Vanity Branding:**
- `0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07` — `0xb0b0b69*` vanity prefix funder. 6,605+ Optimism deployers. Watchlist HIGH.

### MEDIUM (post-Correction-#20)

- `0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078` — meme-token deployment shop, vanilla OZ ERC-20 (downgraded from HIGH on 2026-05-10).

### Notable deactivated (Correction #20)

These are NO LONGER watchlisted; if you encounter them, check Correction #20:
- `0xbb50ce87` (Circle), `0x3304e22d` (Binance), `0x39591e7c` (OKX), `0xfd92f4e9` (OKX), `0xbaed383e` (Bybit), `0xf70da978` (Relay), `0xe69f81b8` (Binance Internal), `0x151b3810` (MoonPay), `0x45a31827` (Owlto), `0xe4edb277` (Orbiter).

---

## Open work (prioritized)

### Methodology / infrastructure

1. **Wire OLI enrichment more deeply.** Currently `entity_classifier.classify_address` consults it. Should also fire on watchlist promotion paths and on first-flagging by detectors. (Item from Correction #20 — partially done.)
2. **Backfill the `oli_labels` cache** for all 67K deployers, not just the 142 flagged ones. Lets the OLI guardrail block adversarial classifications proactively, not just retroactively.
3. **Semantic detector for hidden drain functions** to replace the brittle selector-name signature in `detect_hidden_drain_function`. Pattern: external function + zero-balance-set + no-Transfer-emit + privileged-caller-check. Bigger lift but durable across naming rotation.
4. **Bulk bytecode pull from Blockscout** for the 320 contracts in `0x8ca70232`'s fleet to extract all 5×N blacklist addresses (~1,600 victim addresses if uniform).

### Investigative

5. **Coffee Fleet ↔ `0x604be06b` overlap** open since 2026-04-25 — `0x604be06b9f6b6663f78e755db0c5965eb2337e3d` shares 100% bot-victim overlap with Coffee Fleet. Either (a) c0ffee scanners expanded prey list, or (b) `0x604be06b` is same operator's second deployer.
7. **Watch for `0x8ca70232` resumption.** Cold stop on 2026-04-16 is unexplained. If they return, deployed contracts' `approev` primitives can fire against the 258 standing approvals.
8. **Watch for `0x72ed7949080a` drainer Pattern A instance 4+.** Three instances in 4 days suggests an active operator class; a fourth would justify a lexicon entry promotion.
9. **bb50 / Circle key-compromise vs. legitimate-rollout reconciliation** — bb50 IS Circle's contract deployer (OLI-tagged) but the 49,764 unverified Optimism deployments are unusual for Circle's normal deployment pattern. Low-probability hypothesis: key compromise. Has not been investigated.

### Methodology gaps / risks

10. **`auto_funder_tracer` doesn't store mainnet-history activity context.** It records `mainnet_first_tx` but not "active or dormant since" — which is the disambiguating signal between Pristine Solo Operator (predatory) and institutional-L2-expansion (legitimate).
11. **`infrastructure_registry` is too narrow.** 12 rows covering Circle CCTP v2 product contracts. Should include CEX deployer wallets, bridge solvers, payment-processor addresses — exactly the addresses that Correction #20 surfaced as false positives.
12. **API surface 19-endpoint documentation in CLAUDE.md is stale** — production /stats /suspected /priority etc. is a different surface. Reconciliation pass needed.

---

## Operational protocols

### Before classifying any address as adversarial

```python
from surveillance.oli_enrichment import is_known_legitimate, enrich_address
import sqlite3
conn = sqlite3.connect(...)
enrich_address(conn, addr)  # populates oli_labels cache
if is_known_legitimate(conn, addr, chain_id=1):
    # Address has HIGH-severity OLI tag — DO NOT promote to adversarial
    ...
```

The `entity_classifier.classify_address` function does this automatically (via `_OLI_GUARDED_TRAP_SUBTYPES` guardrail). Callers don't need to invoke OLI directly unless they bypass `classify_address`.

### Production write-through pattern (Path B from Correction #20)

When making corpus changes that need to propagate to production:

1. Apply locally first
2. Verify locally
3. Author a compact production-side script (under ~6 KB to fit Windows cmd.exe limit when base64-bootstrapped)
4. Dispatch via `railway ssh -p blockchain -s stellar-embrace "python3 -c <base64-bootstrap>"`
5. Verify on prod (print return statements from the remote script)
6. Sync down with `python scripts/sync_prod_db.py`
7. Commit

For longer scripts, use the `< script.py` stdin pattern with `MSYS_NO_PATHCONV=1`.

### Commit message conventions

- One topic per commit
- Past-tense imperative ("Add X", "Fix Y", "Retract Z")
- Cite the Correction # if applicable
- Co-Authored-By trailer for Claude
- Use HEREDOC format for multi-line bodies

---

## Files map

```
ai lang/
├── CLAUDE.md, claude.md                    # Project rules + corpus state (both, case quirk)
├── memory.md                               # THIS FILE
├── README.md                               # Public-facing
├── docs/
│   ├── INDEX.md                            # Entity reference (UPDATE WHEN CORPUS CHANGES)
│   └── lexicon.md                          # Typology dictionary (UPDATE WHEN TYPOLOGY CHANGES)
├── reports/
│   ├── correction_log.md                   # Numbered corrections — APPEND-ONLY
│   ├── blockscout_tag_audit_2026-05-09.csv # OLI audit results from Correction #20
│   └── funder_metrics_*.md                 # Dated funder-cluster snapshots
├── scripts/
│   ├── sync_prod_db.py + _remote.py        # Production sync (use this)
│   ├── apply_correction_20_to_prod.py      # Template for prod-write scripts
│   ├── apply_correction_20_via_ssh.py      # Compact SSH-dispatched variant
│   └── blockscout_tag_audit.py             # OLI audit tool
├── surveillance/
│   ├── data/
│   │   ├── surveillance.db (10 GB)         # The DB
│   │   ├── surveillance.db.bak             # Last good copy
│   │   └── cases/                          # Per-entity case files
│   ├── db.py                               # Schema + migrations
│   ├── bytecode_classifier.py              # 10 detectors in PATTERN_REGISTRY
│   ├── entity_classifier.py                # Has OLI guardrail
│   ├── oli_enrichment.py                   # OLI tag lookup + cache
│   ├── deployment_monitor.py               # Chain WebSocket ingestion
│   └── ... (more modules)
```

---

## Commits log (most recent first)

| Hash | Date | Title |
|---|---|---|
| (next) | 2026-05-10 | Items 10-11 + memory.md journal (this session) |
| 1389cbd | 2026-05-10 | Items 7-9: bytecode decompilation splits residual Top-12 |
| 4ae5e54 | 2026-05-10 | Items 1-3 follow-up: Pattern A clone typology + c43f317e case file |
| 70012a9 | 2026-05-10 | Path B complete: Correction #20 applied to prod + sync'd down |
| 9605653 | 2026-05-09 | Production sync via railway ssh |
| aa93b5e | 2026-05-10 | Sync script Windows path-resolution + linked-context fixes |
| fbf8ee5 | 2026-05-09 | CLAUDE.md /dump premise drop |
| 51deb9a | 2026-05-09 | /dump premise correction in correction_log |
| c52e25d | 2026-05-09 | Correction #20 follow-up: Items 1-4 closed |
| 60787c5 | 2026-05-09 | Correction #20 mass mislabel sweep |
| 4c5b2ec | 2026-05-08 | Lexicon + INDEX + correction log: Correction #19 |

Full log: `git -C "C:/Users/jason/Desktop/ai lang" log --oneline -20`

---

## Session log

### 2026-05-10 — Correction #20 Items 7-11 closure + honeypot discovery

**Starting state:** Correction #20 main sweep landed (commit 60787c5, 2026-05-09). Items 1-6 closed by 2026-05-10 morning. Production sync mechanism built and verified. Local DB freshly synced from prod (10.0 GB, integrity ok).

**Session work:**
1. **Items 7-9 (commit `1389cbd`):** Bytecode decompilation of all 4 unattributed Top-12 entries. Found `0xc43f317e`, `0x0e6e9177`, `0xca7ece5e` = vanilla OZ ERC-20 meme-token shops. Found `0x8ca70232` = HONEYPOT OPERATOR with `approev`-mechanism. Authored case file for `0x8ca70232`. Reframed `0xc43f317e` case file to meme-token shop. Watchlist updates applied to local + prod.

2. **Items 1-3 follow-up (commit `4ae5e54`):** Promoted Pattern A Clone Cadence to lexicon. Added third instance (`0x72ed7949080a`) with funder `0x8c8204b8`. Verified zero funder overlap across 3 instances (different funders for same template) — Convergent Calibration at execution layer.

3. **Items 10-11 (this commit):**
   - **Item 10:** Added `detect_hidden_drain_function` to bytecode classifier. Computed `approev(address)` selector `0x3ed67ecd` via keccak256 (cross-validated pycryptodome + eth_hash). Unit-tested with 5 cases — all pass.
   - **Item 11:** Cross-referenced 5 hardcoded blacklist addresses + 25 unique victim approvers. Zero matches across deployers/bot_candidates/entity_classification/watchlist/org_wallets/approval_watchlist/trap_events. **Finding: honeypot victims are fresh-to-corpus — separate population than drain victims.**

4. **Created this `memory.md` journal** for session continuity across context resets.

**Key takeaway from session:** The 4 "unattributed Top-12 ISO" entries broke into two distinct classes when bytecode-reviewed: 3 vanilla meme-token shops (false positives for the typology) and 1 confirmed honeypot operator with embedded predatory primitives. The Top-12 cluster as originally documented is now fully resolved — zero entries retain the "Infrastructure-Scale Operator" framing.

**Open going into next session:** items 1-12 in the "Open work" section above. Highest priority: backfill `oli_labels` for all 67K deployers (so OLI guardrail blocks classifications proactively, not retroactively).

---

*End of file. Append new sessions to the Session log section above.*
