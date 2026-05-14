# Current State

**Last updated:** 2026-05-13
**Last sync from production:** 2026-05-10 (10.0 GB local DB; production has continued to write since)

---

## Project identity

- **Git remote:** `origin = https://github.com/2654-zed/latent-flux.git`
- **README-declared project name:** **"Layer 3 — On-Chain Behavioral Threat Intel"** (README.md line 1, verified 2026-05-13 via UNK-001 resolution)
- **Repository codename:** Latent Flux (= git remote slug; the DSL is the project's analysis-substrate, not its name)
- **Local dev path:** `C:\Users\jason\Desktop\ai lang\`
- **Repository contains TWO load-bearing subsystems sharing one git tree:**
  - **L3 surveillance** (`surveillance/`, `web/`, `run_surveillance.py`) — deployed worker on Railway. The product.
  - **Latent Flux DSL** (`flux_manifold/`, `stdlib/`, `tsp.lf`) — not deployed; the documented analysis substrate. **10 primitives** (not 8 as I earlier analyzed — ↺ Recursive Flow and ⊗ Attractor Competition are first-class).
- **Adjacent arbitrage applications** (resolved 2026-05-13 via UNK-005/006):
  - `pma/` = **Prediction Market Arbitrage module** (Polymarket-style; consumes flux_manifold via reservoir_tracker.py)
  - `sba/` = **Sports Betting Arbitrage module** (consumes flux_manifold; has account_risk modeling)
- **Documented integration claim is ASPIRATIONAL, not built (UNK-024 RESOLVED 2026-05-13):** README line 34 states *"Latent Flux primitives power Layer 3's analysis layer — AttractorCompetition for contract classification, ReservoirState for deployer behavioral baseline, RecursiveFlow for cluster resolution, FoldReference for data integrity."* Verified 2026-05-13 via 7 independent grep paths (current code, full git history, renamed implementations, indirect-use via pma/sba, surveillance/ARCHITECTURE.md) — ALL returned empty. The integration described in README has never existed in code. Adjacent finding: surveillance/ARCHITECTURE.md has the same stale corpus numbers as README (124K contracts vs current 284K) — both docs written ~2026-04-16 and unmaintained since. **lx-scanner is also independent** of both Latent Flux and L3 surveillance — pure MEV arbitrage scanner sharing nothing but the git tree.

## Deploy surface

| Component | Value |
|---|---|
| Procfile command | `worker: python run_surveillance.py` |
| Production URL (current) | `stellar-embrace-production-2020.up.railway.app` |
| Production URL (README, STALE) | `spypy.up.railway.app` — README needs update; old service |
| Railway project | `blockchain` |
| Railway service | `stellar-embrace` |
| Active environment | `production` |
| Last successful sync | 2026-05-10 (full 10 GB; integrity_check: ok) |
| Sync mechanism | `python scripts/sync_prod_db.py` (railway ssh + base64 framing) |
| Apply-to-prod template | `scripts/apply_correction_20_via_ssh.py` |

## Git hooks (LOCAL — not tracked, manual install required on fresh clone)

Installed in `.git/hooks/` (UNK-002 resolution, 2026-05-13):

| Hook | Behavior |
|---|---|
| `pre-commit` | Runs `python scripts/update_readme.py` to refresh `<!-- AUTOGEN:* -->` README sections; re-stages README if modified |
| `post-commit` | Runs `git push origin HEAD` — auto-pushes every commit to remote (explains why my commits during prior sessions pushed without explicit `git push`) |

**No CI/CD pipeline exists** (no `.github/workflows/`, no `.pre-commit-config.yaml`). Fresh clones do NOT get these hooks. Agents working on a fresh clone need to install them manually, or accept that they'll need to `git push` explicitly and won't get README auto-regen on commit.

## Corpus snapshot (as of 2026-05-10 sync)

| Metric | Value | Source |
|---|---|---|
| Total contracts | 284,777 | production `/stats` |
| Confirmed traps | 1,404 | production `/stats` |
| Suspected traps | 115,514 | production `/stats` |
| Unique deployers | 67,459 | production `/stats` |
| Transaction events | 16,810,247 | production `/stats` |
| Bot candidates | 4,244 | production `/stats` |
| Funder coverage | 91.9% | production `/stats` |
| Gas stations | 253 | production `/stats` |
| Org links | 4,382 | production `/stats` |
| Cross-chain shared deployers | 1,046 | production `/stats` |
| Local DB size | 10.0 GB | sqlite file mtime |
| `oli_labels` rows | 69,870 | local DB query 2026-05-13 |
| oli_labels HIGH severity | 15 | local DB query 2026-05-13 |
| oli_labels LOW severity | 422 | local DB query 2026-05-13 |
| oli_labels self-confirming | 17 | local DB query 2026-05-13 |
| Watchlist active | 103 | local DB query 2026-05-13 |
| Watchlist HIGH | 79+ | per priority breakdown |
| Watchlist CRITICAL | varies | check `watchlist.priority` |

## API surface (production)

Stripped from prior 19-endpoint design. Currently exposed:
- `/stats /suspected /priority /bots /tx-events /known-selectors /clusters /cluster-events /health`
- **NOT exposed:** `/dump`, `/risk/{addr}`, `/check/{addr}`, `/screen/{addr}`, `/feed`, per-address lookups
- Stale documentation in `CLAUDE.md` references the 19-endpoint surface (reconciliation pending — see UNKNOWNS.md)

## Active background tasks

None currently. (Updated on each task start/completion — see `.agent-context/active_work.json` when that file exists; not yet created.)

## Recent commit log (last 6)

```
ed11f10  OLI backfill 69,732 deployers + 14 Etherscan-phishing addresses
a29f589  Semantic detector for hidden balance mutation + OLI backfill scaling fix
ca23c76  Items 10-11 + memory.md journal: Correction #20 fully closed
1389cbd  Items 7-9: bytecode decompilation splits residual Top-12 into meme-shops + 1 honeypot
4ae5e54  Items 1-3 follow-up: Pattern A clone typology + c43f317e case file
70012a9  Path B complete: Correction #20 applied to prod + sync'd down
```

For full log: `git -C "C:/Users/jason/Desktop/ai lang" log --oneline -20`

## Key entity reference (most actionable)

### Honeypot operators (HIGH severity)

- `0x8ca702323c341a8d46ee94a2abeddb08798ca10d` — honeypot token operator, 737 contracts, `approev`-mechanism, dormant since 2026-04-16
- `0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e` — Coffee Fleet, 322 contracts, 142 confirmed traps, dual-role deployer+scanner
- `0xf7883e3fef23c8e645deba4b540549d78028a616` — drainer-spawn hub, 859+ victims, iter_9 missed 2026-05-07

### Etherscan-phishing cluster (added 2026-05-10)

14 addresses tagged `Fake_PhishingXXXXX` operating as L2 deployers in our corpus. Watchlist entity_name pattern: `etherscan_phishing_<8-char-prefix>`. Full list in `docs/INDEX.md`.

### Architect investigation (CRITICAL)

- `0x9209c9f7dcb61937f1ec8160c22c0b2365079474` — primary deployer, 21 R&D contracts on Arbitrum
- `0x4cfe37d2` — 0.799 alternate, **CORROBORATED 2026-05-10 as Etherscan `Fake_Phishing327625`**
- Funder side dissolved (`0x151b3810` = MoonPay per Correction #20)

### Retracted (do not re-classify)

- `0xbb50ce87...` = Circle: contract deployer (was: Pristine Solo Industrial)
- `0x3304e22d...` = Binance 73 (was: drainer-spawn hub)
- `0x39591e7c...` = OKX 177
- `0xfd92f4e9...` = OKX 137
- `0xbaed383e...` = Bybit Hot Wallet 6 (was: org_004)
- `0xf70da978...` = Relay Solver (was: org_001 whale)
- Plus 4 more — see `reports/correction_log.md#correction-20`

## Detector inventory (bytecode_classifier.py PATTERN_REGISTRY)

11 detectors registered (as of 2026-05-13):

1. asymmetric_transfer → has_asymmetric_transfer
2. blacklist_check → has_conditional_revert
3. tx_origin_conditional → has_asymmetric_transfer
4. callback_trap → (None)
5. hidden_fee → has_unusual_fee_structure
6. selfdestruct → (None)
7. delegatecall_in_token → (None)
8. timestamp_activation → (None)
9. origin_eoa_gate → has_asymmetric_transfer
10. obfuscated_fee → has_unusual_fee_structure
11. hidden_drain_function → has_asymmetric_transfer (signature: `0x3ed67ecd` for `approev`)
12. privileged_caller_balance_mutation → has_asymmetric_transfer (semantic; complement to #11)

## Most-recent Correction

**Correction #20 (2026-05-09):** OLI mass mislabel sweep. 18 institutional addresses retracted as misclassified. All 11 follow-up items closed by 2026-05-13. See `reports/correction_log.md#correction-20` for full numbered entry.

## Pointers to deeper context

| Topic | Read |
|---|---|
| Long-form session journal | `memory/JOURNAL.md` |
| Open questions | `memory/UNKNOWNS.md` |
| Architectural decisions | `memory/DECISIONS.md` |
| System invariants | `memory/INVARIANTS.md` |
| Session-end reflection protocol | `memory/LOOP.md` |
| Reflection audit trail | `memory/REFLECTION_LOG.csv` |
| Entity reference (per-address detail) | `docs/INDEX.md` |
| Typology dictionary | `docs/lexicon.md` |
| Numbered corrections (append-only) | `reports/correction_log.md` |
| OLI audit CSV | `reports/blockscout_tag_audit_2026-05-09.csv` |

## Watchlist quick-pulls

For programmatic use:
```bash
# Active watchlist summary
sqlite3 surveillance/data/surveillance.db \
  "SELECT priority, COUNT(*) FROM watchlist WHERE active=1 GROUP BY priority"

# OLI HIGH-severity hits
sqlite3 surveillance/data/surveillance.db \
  "SELECT address, primary_entity, primary_tag_name FROM oli_labels WHERE severity='HIGH'"

# Production stats (live)
curl -s https://stellar-embrace-production-2020.up.railway.app/stats | python -m json.tool
```

## Test coverage state (UNK-008 RESOLVED 2026-05-13)

**Zero surveillance-side tests exist.** The `tests/` directory contains 11 test files, all targeting `flux_manifold/` (parser/repl, core, primitives, interpreter, ontology, baselines, attractor competition, recursive flow, infrastructure, visualize, benchmarks). No `tests/surveillance/` directory. No test file imports from `surveillance/`.

Agents modifying `surveillance/bytecode_classifier.py`, `entity_classifier.py`, `oli_enrichment.py`, etc., have **no verification surface**. The Action 4 from prior Phase 4 output (write `tests/surveillance/test_smoke.py`) is unblocked.

## Open work (canonical list in `memory/UNKNOWNS.md`)

| RESOLVED 2026-05-13 | OPEN (17) |
|---|---|
| UNK-001 (HIGH) README content | UNK-003 ontology §2/§3/§4 spec |
| UNK-002 (HIGH) CI config | UNK-004 active vs archive scripts |
| UNK-005 (MEDIUM) pma/ purpose | UNK-009 private flux_manifold modules |
| UNK-006 (MEDIUM) sba/ purpose | UNK-010 rib_dataset usage |
| UNK-007 (HIGH) lx-scanner | UNK-011 → UNK-023 (DSL theoretical + operational) |
| UNK-008 (HIGH) test coverage | |
| UNK-024 (HIGH) README integration claim | |

**Recommended next session focus:** Move from UNKNOWN resolution → action execution. Top candidates:
- **Action 4** (write `tests/surveillance/test_smoke.py`) — unblocked by UNK-008 resolution; ~90 minute task
- **Action 5** (Integration Path 1, regime-monitor) — unblocked by UNK-024 resolution; multi-session
- **README freshness fix** — update stale URL + corpus numbers + qualify the integration claim; ~20 minute task
