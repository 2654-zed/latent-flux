# Current State

**Last updated:** 2026-05-15
**Last sync from production:** 2026-05-15 (10.8 GB local DB; first sync via v2 chunked protocol after v1 broke at the 11.6 GB raw / 4.4 GB base64 threshold)

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
- **Integration is now PARTIAL — first production-side flux_manifold consumer LIVE (2026-05-13):** `surveillance/regime_monitor.py` imports `BayesianChangePoint` from `flux_manifold.changepoint` and produces real alerts against production corpus (29 alerts on first scan including the Apr-25 b0b0b690 mass-fund event and the May-5 iter_8 confirmed-trap spike). This is the first concrete instantiation of the README's long-standing aspirational claim. The other primitives named in README (AttractorCompetition, ReservoirState, RecursiveFlow, FoldReference) remain unintegrated; the path is established but not yet executed. README has been updated 2026-05-13 to qualify the claim as "Planned integration (not yet wired into production)" with a pointer to `regime_monitor.py` as the first consumer. **lx-scanner is independent** of both Latent Flux and L3 surveillance (pure MEV arbitrage scanner; UNK-007 RESOLVED).

## Deploy surface

| Component | Value |
|---|---|
| Procfile command | `worker: python run_surveillance.py` |
| Production URL (current) | `stellar-embrace-production-2020.up.railway.app` |
| Production URL (README, STALE) | `spypy.up.railway.app` — README needs update; old service |
| Railway project | `blockchain` |
| Railway service | `stellar-embrace` |
| Active environment | `production` |
| Last successful sync | 2026-05-15 [pending verification] — see Session log for v2 protocol migration |
| Sync mechanism | `python scripts/sync_prod_db.py` — **v2 chunked protocol since 2026-05-15** (railway ssh + base64 framing + 100-MB chunked retrieval; see ADR-007). v1 single-stream broke at 11.6 GB DB size with `tungstenite error`. |
| Apply-to-prod template | `scripts/apply_correction_20_via_ssh.py` |

## Git hooks (LOCAL — not tracked, manual install required on fresh clone)

Installed in `.git/hooks/` (UNK-002 resolution, 2026-05-13):

| Hook | Behavior |
|---|---|
| `pre-commit` | Runs `python scripts/update_readme.py` to refresh `<!-- AUTOGEN:* -->` README sections; re-stages README if modified |
| `post-commit` | Runs `git push origin HEAD` — auto-pushes every commit to remote (explains why my commits during prior sessions pushed without explicit `git push`) |

**No CI/CD pipeline exists** (no `.github/workflows/`, no `.pre-commit-config.yaml`). Fresh clones do NOT get these hooks. Agents working on a fresh clone need to install them manually, or accept that they'll need to `git push` explicitly and won't get README auto-regen on commit.

## Corpus snapshot (as of 2026-05-15 sync)

| Metric | 2026-05-10 | **2026-05-15** | Δ |
|---|---|---|---|
| Total contracts | 284,777 | **321,578** | +36,801 (+12.9%) |
| Unique deployers | 67,459 | **73,818** | +6,359 (+9.4%) |
| Transaction events | 16,810,247 | **18,025,924** | +1,215,677 (+7.2%) |
| Local DB size | 10.0 GB | **10.8 GB** | +0.8 GB |
| Latest contract detection | (2026-05-09) | **2026-05-15T23:47:09Z** | 5 days |

Stable since 2026-05-13 (not re-queried; from local sqlite, may drift from prod):
- `oli_labels` rows: 69,870 (15 HIGH, 422 LOW, 17 self-confirming)
- Watchlist active: 103 (79+ HIGH)

**Note: `regime_alerts` table is empty in the fresh DB.** The 29 alerts from the 2026-05-13 manual `regime_monitor.py` run were local-only writes — `regime_monitor` is not yet wired into `run_surveillance.py` as a scheduled job. Re-run `python -m surveillance.regime_monitor` against the fresh corpus before Phase A of NEXT_SESSION_PLAN.

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

## Test coverage state

**Latent Flux DSL:** 11 test files in `tests/` covering parser/repl, core, primitives, interpreter, ontology, baselines, attractor competition, recursive flow, infrastructure, visualize, benchmarks.

**L3 surveillance (added 2026-05-13):** `tests/surveillance/` directory now exists with 14 tests in 2 files:
- `test_smoke.py` (9 tests): OLI guardrail redirect + pass-through, hidden-drain signature detector (positive + negative), KNOWN_HIDDEN_DRAIN_SELECTORS registry, PATTERN_REGISTRY contents, confidence rank-protection (down + up), migration idempotency principle
- `test_regime_monitor.py` (5 tests): obvious changepoint detection, stationary-series silence, table writes, idempotency, missing-source-table graceful skip

All 14 surveillance tests pass in ~270ms. Run via `python -m pytest tests/surveillance/`. Verification surface for surveillance modifications is now established.

**Known gap (INV-016):** `init_db()` on a truly fresh path crashes on the `extraction_events.chain` migration because `extraction_events` is never created in code (only in pre-existing binary DBs). Documented in INVARIANTS.md INV-016; fix path documented; production unaffected.

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
