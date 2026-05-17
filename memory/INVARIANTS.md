# System Invariants

Invariants are statements that must remain true for the system to function as designed. Each entry includes the statement, why it must hold, where it's enforced in code, how it's tested, and when it was surfaced.

A violation becomes a numbered Correction in `reports/correction_log.md`. A discovery / refinement updates the entry here.

---

## Latent Flux DSL invariants

### INV-001 — Every flux_flow call must have an attached ConvergenceContract

- **Rationale:** The halting problem for continuous dynamical systems is unresolved. Without an explicit contract, "did the flow converge" is undefined. Contracts force the caller to declare which mathematical regime they're operating in.
- **Enforcement:** `flux_manifold/core.py:flux_flow` — defaults to `TIER_2_DEFAULT` if `contract=None`. Tier 1 raises ValueError if `lipschitz_bound is None or >= 1.0` (see `convergence.py:ConvergenceContract.__post_init__`).
- **How tested:** `tests/test_convergence_reservoir.py` (assumed — UNK-related; verify when tests are next run)
- **Surfaced:** 2026-05-13 (Phase 1-5 analysis session)

### INV-002 — State vectors must have dimension d ≤ 1024

- **Rationale:** Safety cap against pathological input. Hardcoded in validator to prevent OOM / unbounded computation.
- **Enforcement:** `flux_manifold/core.py:_validate_inputs` — `raise ValueError(f"Dimension {d} exceeds safety cap of 1024")`
- **How tested:** untested (audit candidate)
- **Surfaced:** 2026-05-13

### INV-003 — Pipeline evaluation is strictly left-to-right

- **Rationale:** Pipe operator `|` is sequential application semantics. Reordering changes results since each primitive has side effects on shape/state.
- **Enforcement:** `flux_manifold/parser.py:Parser` — recursive descent with no operator override
- **How tested:** `tests/test_parser_repl.py` (assumed)
- **Surfaced:** 2026-05-13

### INV-004 — NaN / inf deltas in flow are replaced with zeros, not propagated

- **Rationale:** Once a NaN enters the state vector, all subsequent operations are poisoned. Replacing with zero halts the flow at that step rather than corrupting downstream.
- **Enforcement:** `flux_manifold/core.py:flux_flow` — `if np.any(np.isnan(delta)) or np.any(np.isinf(delta)): delta = np.zeros_like(delta)`. Reinforced by `FoldReference.no_nan_critique`.
- **How tested:** likely covered in `tests/test_core.py` or `tests/test_primitives.py` (verify)
- **Surfaced:** 2026-05-13

### INV-005 — All state arrays cast to float32 on entry to flux_flow

- **Rationale:** Deterministic precision behavior. Mixing float32 and float64 produces silent dtype coercion.
- **Enforcement:** `flux_manifold/core.py:flux_flow` — `s = s0.astype(np.float32, copy=True); q = q.astype(np.float32, copy=False)`
- **How tested:** untested for dtype specifically
- **Surfaced:** 2026-05-13

### INV-006 — Vector shape contract: state dim equals attractor dim

- **Rationale:** Flow operates element-wise; shape mismatch is undefined.
- **Enforcement:** `flux_manifold/core.py:_validate_inputs` — raises ValueError on mismatch (handles both 1-D and 2-D batch cases)
- **How tested:** likely in `tests/test_core.py`
- **Surfaced:** 2026-05-13

---

## L3 surveillance invariants

### INV-007 — OLI guardrail runs BEFORE adversarial typology is committed

- **Rationale:** Correction #20 established that behavioral/topology classification produces systematic false positives on institutional addresses (CEX hot wallets, bridges, payment processors, protocol deployers). Identity disambiguation via OLI must precede classification, not follow it.
- **Enforcement:** `surveillance/entity_classifier.py:classify_address` — early check against `_OLI_GUARDED_TRAP_SUBTYPES` set; redirects to `COMMERCIAL/institutional_oli_tagged` if HIGH-severity OLI tag exists.
- **How tested:** smoke-tested 2026-05-10 (3 scenarios: HIGH+trap→redirect; HIGH+non-guarded→pass-through; no-OLI+trap→pass-through). No persistent test file yet — see UNKNOWNS.md for surveillance-test-coverage gap.
- **Surfaced:** 2026-05-09 (Correction #20)
- **Refined:** 2026-05-13 (made explicit in INVARIANTS.md)

### INV-008 — `entity_classification.confidence` is rank-protected (never downgrades)

- **Rationale:** A LOW-confidence classification arriving after a HIGH-confidence one should not erase the prior finding. The system upgrades only.
- **Enforcement:** `surveillance/entity_classifier.py:classify_address` — checks `CONFIDENCE_RANK[existing.confidence] vs new`; returns False without writing if new < existing.
- **How tested:** untested at the moment (Action 4 candidate)
- **Surfaced:** 2026-05-13

### INV-009 — Production SQLite is in WAL mode

- **Rationale:** Allows concurrent reader/writer access. Required for the deployed worker (writes) + HTTP API (reads) to coexist.
- **Enforcement:** `surveillance/db.py:init_db` — `PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=500`
- **How tested:** verified by `sync_prod_db.py` integrity check showing healthy DB after sync
- **Surfaced:** 2026-05-13

### INV-010 — Schema migrations in db.py are idempotent

- **Rationale:** `init_db()` runs on every worker start (including restarts). Re-applying migrations must be a no-op.
- **Enforcement:** Every migration in `surveillance/db.py:init_db` is guarded by a "does the column/table already exist?" check before applying. `_log_migration` records `applied` or `skip`.
- **How tested:** implicit via repeated production restarts. No explicit test.
- **Surfaced:** 2026-05-13

### INV-011 — Production sync uses base64 framing, never raw binary

- **Rationale:** Railway SSH transport does CRLF translation on raw binary bytes (verified empirically: 4 stray byte changes per 1024 random bytes) and merges stderr into stdout. Raw binary streams are corrupted.
- **Enforcement:** `scripts/sync_prod_db_remote.py` — gzips backup file then base64-encodes; output framed by `L3SYNC_PAYLOAD_START` / `L3SYNC_PAYLOAD_END` markers. `scripts/sync_prod_db.py` strips banner content via marker search.
- **How tested:** verified end-to-end against 10 GB production DB on 2026-05-10
- **Surfaced:** 2026-05-09 (during sync mechanism development; see ADR-001)

### INV-011a — Production sync is two-phase (prepare on container, chunked retrieval) for large DBs

- **Rationale:** A single `railway ssh` invocation cannot stream the full base64 payload of a multi-GB DB to stdout — fails with `Error: WebSocket error: tungstenite error` at some unknown threshold between 50 MB and 4.4 GB. The WebSocket transport tolerates long-idle sessions (verified: 7.4 min backup+gzip with no streaming succeeded) and small streams (verified: 50 MB streamed cleanly). What kills it is total streamed-volume-per-SSH-invocation. Chunking across multiple SSH sessions (each its own WebSocket) sidesteps the limit; SHA-256 from prepare phase catches any chunk-stitching error.
- **Enforcement:** `scripts/sync_prod_db_remote.py` exposes 4 modes (prepare, chunk, cleanup, sha256). `scripts/sync_prod_db.py` runs phase 1 (prepare → READY:size:sha256), phase 2 loop (chunk by 100 MB), phase 3 (decompress + integrity), phase 4 (remote cleanup). Per-chunk retries: 2.
- **How tested:** verified 2026-05-15 against 11.6 GB production DB after v1 (single-stream) protocol failed three times in a row with tungstenite error.
- **Surfaced:** 2026-05-15 (DB size crossed the tungstenite-streaming threshold sometime between 2026-05-10 — last successful v1 sync at 10.0 GB — and 2026-05-15 at 11.6 GB)

### INV-012 — Watchlist deactivations preserve history (active=0, never DELETE)

- **Rationale:** Corrections must be reversible. The full historical record of every watchlist entry — including ones since deactivated — supports audit trails and "why did we previously think X" investigations.
- **Enforcement:** `scripts/apply_correction_20_to_prod.py` and equivalents — `UPDATE watchlist SET active=0`. No `DELETE FROM watchlist` exists in any active script.
- **How tested:** preserved across multiple sync cycles
- **Surfaced:** 2026-05-13

---

## Repository-level invariants

### INV-013 — Procfile deploys only `python run_surveillance.py`

- **Rationale:** Production deploys the surveillance worker, not the DSL runtime. Confusion between subsystems gets caught at this boundary.
- **Enforcement:** `Procfile` — single line: `worker: python run_surveillance.py`. Also: `nixpacks.toml` has `[start] cmd = "python run_surveillance.py"`.
- **How tested:** verified by current Railway deployment
- **Surfaced:** 2026-05-13

### INV-014 — `reports/correction_log.md` is append-only

- **Rationale:** Numbered corrections form an immutable audit trail. A previously-asserted claim retracted in Correction #N can only be re-retracted in Correction #M (later); the original entry stays.
- **Enforcement:** Convention. No mechanical enforcement.
- **How tested:** untested mechanically. Future enforcement: pre-commit hook checking that existing Correction sections are unchanged.
- **Surfaced:** 2026-05-13 (existed implicitly since first correction)

### INV-015 — Bayesian changepoint consumers use `is_changepoint()`, not raw `update()` return value

- **Rationale:** `BayesianChangePoint.update(x)` returns `P(run_length < 10)` which is naturally ≥ 0.5 during the first ~20 observations (the burn-in period) because the run-length distribution starts with all mass at r=0. Treating this raw probability as a changepoint signal produces spurious alerts on every observation in the burn-in window. The `is_changepoint()` method has the proper gating: `MAP_run_length < 10 AND t > 20 AND cp_prob > threshold`.
- **Enforcement:** `surveillance/regime_monitor.py:_scan_one_signal` — uses `detector.is_changepoint(threshold=...)` after `detector.update(value)`. Comment in code documents the reasoning.
- **How tested:** `tests/surveillance/test_regime_monitor.py::test_regime_monitor_silent_on_stationary_series` asserts ≤ 1 alert over 60 stationary observations (would fail if raw `update()` value were used — currently 10 spurious alerts in pre-fix run).
- **Surfaced:** 2026-05-13 (caught by smoke test on stationary series)

### INV-017 — Every Layer 3 capability, data path, or pipeline must map back to a question in `memory/questions.yaml`

- **Rationale:** Per ADR-008 (SAI substrate). The system's primary abstraction is the question, not the data or the model. If a capability does not answer a structured question in the store, it is orphan work — likely unnecessary or untrackable. The discipline forces every build to declare its decision_output.
- **Enforcement:** New modules in `surveillance/` must reference a question id in their docstring (e.g., `surveillance/analytics/approval_spike_detector.py` declares it answers Q-002). `question_runner.check_wiring()` audits the inverse: every active question's `implementation_target` must exist or be a skeleton with TODO markers.
- **How tested:** `tests/surveillance/test_sai.py::test_question_loads_18_questions` validates the store is loadable. Capability liveness check (Q-008, partial implementation in `surveillance/sai/capability_liveness.py`) audits orphan capabilities at runtime.
- **Surfaced:** 2026-05-16 (SAI cycle session, ADR-008 landing)

### INV-016 — `extraction_events` table is NOT in `schema.sql` — known latent bug

- **Rationale:** A migration in `surveillance/db.py` (line ~545-561) does `ALTER TABLE extraction_events ADD COLUMN chain TEXT` but `CREATE TABLE extraction_events` exists nowhere in code (only in binary DB files). Running `init_db()` against a truly fresh path fails on this migration. In production this never surfaces because the table has existed since unrecorded manual creation.
- **Status:** OPEN BUG, not yet fixed.
- **Where surfaced:** `tests/surveillance/test_smoke.py::test_migration_idempotency_principle` — original version of this test attempted `init_db(tmp_path / "test.db")` and failed with `sqlite3.OperationalError: no such table: extraction_events`. The test was reduced to verify the idempotency *pattern* directly without depending on full schema bootstrap.
- **Fix path:** Either (a) add `CREATE TABLE extraction_events` to schema.sql, or (b) guard the `chain` column migration with a `SELECT name FROM sqlite_master WHERE name='extraction_events'` existence check. Option (b) is safer (zero-side-effect if missing); option (a) is more correct (a real schema needs to declare its tables).
- **Surfaced:** 2026-05-13 (via smoke-test execution against fresh tmp_path)
