# Open Unknowns

Canonical schema (per `memory/LOOP.md` UNKNOWNS section):

```
## UNK-XXX — <One-line question>
- Surfaced: <iso>, by <session-id>
- Category: Architecture | Subsystem | Theoretical | Operational
- Why it matters: <consequence of leaving open>
- Resolution plan: <files/code paths>
- Status: OPEN | IN_PROGRESS | RESOLVED | SUPERSEDED
- Confidence (if RESOLVED): HIGH | MEDIUM | LOW
- Resolved at: <iso, if RESOLVED>
- Resolved by: <one-line answer + citation>
```

Confidence calibration (from LOOP.md):
- **HIGH:** primary-source verified + independent cross-reference
- **MEDIUM:** single primary-source verification
- **LOW:** single secondary source — flagged for revisit

---

## Architecture Unknowns

### UNK-001 — README.md content (40 KB at repo root, never opened)

- **Surfaced:** 2026-05-13, by 2026-05-13-phase-analysis
- **Category:** Architecture
- **Why it matters:** My "Active Purpose" framing of the system is currently inferred. The README likely contains the explicit project positioning, the intended relationship between Latent Flux and surveillance, and the primary use case. Every downstream framing carries this inference as a load-bearing assumption.
- **Resolution plan:** `Read C:/Users/jason/Desktop/ai lang/README.md` end-to-end. Map explicit statements to STATE.md "Project identity" section. Update if README contradicts.
- **Status:** RESOLVED
- **Confidence:** HIGH (primary-source verified — README.md lines 1-200, 560-680 read directly)
- **Resolved at:** 2026-05-13
- **Resolved by:** Project framing is "Layer 3 — On-Chain Behavioral Threat Intel" with **Latent Flux DSL as the documented analysis substrate**. README line 34: *"Latent Flux primitives power Layer 3's analysis layer — AttractorCompetition for contract classification, ReservoirState for deployer behavioral baseline, RecursiveFlow for cluster resolution, FoldReference for data integrity."* HOWEVER: this integration claim does NOT match code reality (see UNK-024 below). README also documents **10 primitives, not 8** as I'd analyzed (↺ Recursive Flow and ⊗ Attractor Competition are first-class primitives, not "less common operators"). README live-URL is `spypy.up.railway.app` (STALE — current production is `stellar-embrace-production-2020.up.railway.app`). README corpus numbers (124,341 contracts) are STALE (current: 284,777). README documents `/api/v1/agent/screen/...` and `/api/v1/agent/facilitator/...` endpoints that are NOT in my probed production surface — possibly stale OR my probe missed them.

### UNK-002 — Whether `.github/workflows/` or any CI config exists

- **Surfaced:** 2026-05-13
- **Category:** Architecture
- **Why it matters:** Test discipline cannot be assessed without it. Agents don't know if tests must pass before commits or if production deploys are gated.
- **Resolution plan:** `Bash: find .github .pre-commit-config.yaml -type f 2>/dev/null`. Also check `.git/hooks/`.
- **Status:** RESOLVED
- **Confidence:** HIGH (primary-source verified — `find` returned no `.github/` directory and no `.pre-commit-config.yaml`; `ls .git/hooks/` showed `pre-commit` and `post-commit` are LOCAL hooks)
- **Resolved at:** 2026-05-13
- **Resolved by:** **No CI/CD pipeline.** No `.github/workflows/`. No `.pre-commit-config.yaml`. Local-only git hooks installed in `.git/hooks/` (not tracked, not auto-installed on fresh clone): (1) `pre-commit` runs `python scripts/update_readme.py` to auto-update README dynamic sections marked `<!-- AUTOGEN:* -->` then re-stages README if modified; (2) `post-commit` runs `git push origin HEAD` — explains the auto-push behavior on every commit during prior sessions. **Implication:** test runs require manual invocation. No gate on production deploys. New agents working on a fresh clone won't have auto-push or auto-README behavior until they install the hooks manually. → Candidate for ADR-006: "Local-only git hooks; install instructions belong in STATE.md."

### UNK-003 — The "ontology §2/§3/§4" theoretical specification document

- **Surfaced:** 2026-05-13
- **Category:** Architecture / Theoretical
- **Why it matters:** `flux_manifold/__init__.py` references "Hamiltonian Flows (§2 ontology)", "Quantum Interference (§3 ontology)", "Topological Squeeze (§4 ontology)" — implying a written specification with numbered sections. If it exists outside the repo, design-level extensions are anchored. If it lives only in `tests/test_ontology.py`, the proof obligations there ARE the spec.
- **Resolution plan:**
  1. `Grep -r "§" --include="*.py" --include="*.md"` for cross-refs
  2. `Read tests/test_ontology.py` and `tests/proof/test_meta.py` + 5 numbered proofs
  3. Check `docs/` for any `.md` with section symbols
  4. If nothing surfaces, flag for explicit documentation as a new ADR
- **Status:** OPEN

### UNK-004 — Which of the 110+ root-level scripts are active vs archive

- **Surfaced:** 2026-05-13
- **Category:** Architecture / Operational
- **Why it matters:** The repo root has 110+ Python files. An agent cannot distinguish "current investigation" from "completed one-off probe." An agent extending an investigation may modify a stale script.
- **Resolution plan:**
  1. `Bash: git log --since="30 days ago" --name-only --format= | sort -u | grep -E "^[a-z]" | grep -v "/"` — recently-touched
  2. Compare against `Bash: ls *.py` for the full list
  3. Triage non-recent into `scripts/archive/<date>/` with `git mv` preserving history
  4. Write `scripts/MANIFEST.md` enumerating active scripts with 1-line purpose
- **Status:** OPEN

---

## Subsystem Unknowns

### UNK-005 — `pma/` subsystem purpose

- **Surfaced:** 2026-05-13
- **Category:** Subsystem
- **Why it matters:** Imports `flux_manifold`. Has `reservoir_tracker.py`, `market_feed.py`, `pma_searcher.py`, `run_pma_backtest.py`. Naming suggests "Predictive Market Analysis" but no README confirms. Integration plans require knowing if pma is canonical Latent-Flux-applied-to-markets or deprecated prototype.
- **Resolution plan:** `Read pma/__init__.py pma/run_pma_backtest.py pma/pma_searcher.py` (first 100 lines each). Document in a new `pma/README.md`.
- **Status:** RESOLVED
- **Confidence:** MEDIUM (primary-source verified for purpose; implementation surface not yet inspected)
- **Resolved at:** 2026-05-13
- **Resolved by:** `pma/__init__.py` single-line docstring: **"Prediction Market Arbitrage module"**. So pma = **Prediction Market Arbitrage** (e.g., Polymarket-style binary outcome markets), NOT "Predictive Market Analysis" as I'd guessed. Different domain: pma is a searcher/scanner for arbitrage opportunities across prediction markets, using flux_manifold's reservoir-state primitives via `reservoir_tracker.py`. **Confidence is MEDIUM** because the implementation surface (`pma_searcher.py`, `market_feed.py`, `run_pma_backtest.py`) has not been read yet. Promote to HIGH after reading those. → revisit-LOW candidate.

### UNK-006 — `sba/` subsystem purpose

- **Surfaced:** 2026-05-13
- **Category:** Subsystem
- **Why it matters:** Same shape as pma (`reservoir_tracker.py`, `account_risk.py`, `odds_feed.py`, `run_sba_backtest.py`). "Sports Betting Adjacent"? Unclear application surface.
- **Resolution plan:** `Read sba/__init__.py sba/run_sba_backtest.py sba/odds_feed.py`. Presence of `odds_feed.py` strongly suggests sports betting odds; verify.
- **Status:** RESOLVED
- **Confidence:** MEDIUM (primary-source verified for purpose; implementation surface not yet inspected)
- **Resolved at:** 2026-05-13
- **Resolved by:** `sba/__init__.py` single-line docstring: **"Sports Betting Arbitrage module"**. Confirms "sports betting arbitrage" (vs. my earlier "sports betting adjacent" guess). Mirror structure to pma: a searcher/scanner for arbitrage opportunities across sportsbooks, using flux_manifold reservoir primitives via `reservoir_tracker.py`. `account_risk.py` suggests this also models account-level constraints (max stake per book, exposure limits). MEDIUM confidence — implementation files unread. → revisit-LOW candidate.

### UNK-007 — `lx-scanner/` integration with flux_manifold

- **Surfaced:** 2026-05-13
- **Category:** Subsystem
- **Why it matters:** I claimed grep showed no direct integration. But lx-scanner has `mev_arbitration_router.py` + DEX executors and might use reservoir-tracking indirectly via pma/sba.
- **Resolution plan:**
  1. `Grep -r "reservoir\|flux_flow\|SuperpositionTensor" lx-scanner/`
  2. `Read lx-scanner/live_feed_scanner.py` and `mev_arbitration_router.py` headers
- **Status:** OPEN

### UNK-008 — Surveillance-side test coverage

- **Surfaced:** 2026-05-13
- **Category:** Subsystem / Operational
- **Why it matters:** `tests/` at repo root contains flux_manifold tests. UNKNOWN whether surveillance-specific tests exist. Without them, agents modifying `bytecode_classifier.py`, `entity_classifier.py`, `oli_enrichment.py` have no verification surface.
- **Resolution plan:** `Glob tests/**/*surveillance* tests/**/*oli* tests/**/*classifier*`. If empty, the gap is real → Action 4 of the previous Phase 4 output applies.
- **Status:** OPEN

### UNK-009 — `flux_manifold/cex_feed.py`, `kalman_reservoir.py`, `multi_scale_reservoir.py`, `changepoint.py`, `pheno_log.py`, `monitor.py` roles

- **Surfaced:** 2026-05-13
- **Category:** Subsystem
- **Why it matters:** These files exist in `flux_manifold/` but are NOT exported in `__init__.py`. Either private utilities, deprecated, or used by adjacent subsystems via direct import. Agents may delete them assuming unused.
- **Resolution plan:**
  1. `Grep -r "from flux_manifold.changepoint\|from flux_manifold.kalman_reservoir\|from flux_manifold.cex_feed\|from flux_manifold.pheno_log\|from flux_manifold.monitor"` for consumers
  2. For each consumer, check if in `pma/`, `sba/`, `lx-scanner/`, scripts
  3. Add a "private utilities (consumed by X)" section to the package docstring
- **Status:** OPEN

### UNK-010 — `rib_dataset/` usage

- **Surfaced:** 2026-05-13
- **Category:** Subsystem
- **Why it matters:** Contains graph data (`edges.csv`, `address_map.json`, `ground_truth.json`, `metadata.json`). Ground-truth labels suggest an eval dataset. If it's canonical, agents adding classifiers should run against it.
- **Resolution plan:**
  1. `Read rib_dataset/README.md` (exists)
  2. `Read rib_dataset/metadata.json` for description
  3. `Grep -r "rib_dataset" --include="*.py"` for consumers
- **Status:** OPEN

---

## Theoretical Unknowns

### UNK-011 — Convergence Contract Tier 1's Lipschitz bound verification

- **Surfaced:** 2026-05-13
- **Category:** Theoretical
- **Why it matters:** Tier 1 PROVABLE convergence is the strongest mathematical claim in the system. UNKNOWN whether the supplied `lipschitz_bound` is verified against the actual flow function or just trusted as declared. If unverified, Tier 1 is a self-declaration with no enforcement.
- **Resolution plan:**
  1. `Read flux_manifold/core.py` lines 47-110 for operational uses of `contract.lipschitz_bound`
  2. `Grep "lipschitz_bound" flux_manifold/` for all references
  3. If unverified, document explicitly and consider adding `docs/CONVERGENCE_GUARANTEES.md`
- **Status:** OPEN

### UNK-012 — Tier 3 NON_CONVERGENT runtime behavior

- **Surfaced:** 2026-05-13
- **Category:** Theoretical / Operational
- **Why it matters:** `convergence.py` docstring says Tier 3 "explicitly signals failure rather than silently diverging." UNKNOWN what "explicit signal" means in code — exception? sentinel return?
- **Resolution plan:**
  1. `Read flux_manifold/core.py` for branches on `contract.tier == ConvergenceTier.NON_CONVERGENT`
  2. `Grep "NON_CONVERGENT" flux_manifold/` for usages
  3. `Read tests/test_convergence_reservoir.py` for behavioral assertions
- **Status:** OPEN

### UNK-013 — `↺ recurse` and `⊗ compete` operator semantics

- **Surfaced:** 2026-05-13
- **Category:** Theoretical
- **Why it matters:** Both operators are in `OPERATORS` map with implementation files but don't appear in the canonical pipeline. Agents writing `.lf` programs don't know when to use them. No `stdlib/` examples reference them.
- **Resolution plan:**
  1. `Read flux_manifold/recursive_flow.py` class docstring
  2. `Read flux_manifold/attractor_competition.py` class docstring
  3. `Read tests/test_recursive_flow.py` and `tests/test_attractor_competition.py` for usage
  4. Promote findings to `stdlib/` as example `.lf` programs
- **Status:** OPEN

### UNK-014 — `emit(value)` built-in semantics

- **Surfaced:** 2026-05-13
- **Category:** Theoretical / Operational
- **Why it matters:** Used in `tsp.lf` to report results. Behavior (logs only / mutates context / returns) determines how `.lf` programs surface outputs.
- **Resolution plan:** `Grep "def.*emit\|\"emit\"\|emit =" flux_manifold/parser.py`. 5-minute task.
- **Status:** OPEN

### UNK-015 — Convergence-contract propagation through `.lf` pipelines

- **Surfaced:** 2026-05-13
- **Category:** Theoretical
- **Why it matters:** Tier declarations attach to `flux_flow` calls in Python. UNKNOWN whether every `⟼` in a `.lf` pipeline defaults to Tier 2 or whether a global default can be set per-program.
- **Resolution plan:**
  1. `Grep "ConvergenceContract\|TIER_2_DEFAULT\|contract=" flux_manifold/parser.py`
  2. Find where the parser-side evaluator invokes `flux_flow` — inspect the contract passed
- **Status:** OPEN

---

## Operational Unknowns

### UNK-016 — FlowTrace memory bound

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** `FlowTrace` records every convergence event. A long-running flow (`max_steps=1000` × N candidates) could accumulate ~16K entries per pipeline. UNKNOWN if bounded.
- **Resolution plan:** `Read flux_manifold/flow_trace.py` — look for ring-buffer, cap, or unbounded list. If unbounded, add a default cap.
- **Status:** OPEN

### UNK-017 — Import binding-collision behavior

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** If `geometry.lf` exports `origin_2d` and the user's script ALSO defines `origin_2d`, what wins? Affects how stdlib imports compose with user code.
- **Resolution plan:**
  1. `Read flux_manifold/parser.py` — `LFImport` handler in `evaluate_program`
  2. Write test fixture: import a module + re-bind one of its exports; assert which value `ctx.variables` holds
- **Status:** OPEN

### UNK-018 — Critique function failure semantics

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** If `norm_bound_critique` rejects every state for 100 fold-intervals (flow diverges immediately), does the pipeline terminate or loop forever?
- **Resolution plan:** `Read flux_manifold/fold_reference.py` — look for failure-counter, max-rejection cap. Write adversarial test case.
- **Status:** OPEN

### UNK-019 — Pipeline branching support

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** Grammar shows strictly linear pipelines. UNKNOWN whether `let x = pipeline_A; let y = pipeline_B; let z = combine(x, y)` works. Affects how complex programs are composed.
- **Resolution plan:** `Read parser.py:parse_program` to confirm let-statements allow this. Write a test fixture demonstrating combine semantics.
- **Status:** OPEN

### UNK-020 — Reservoir state isolation between consecutive pipelines

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** `tsp.lf` has multiple pipeline statements. If reservoir state from pipeline #1 leaks into pipeline #2, results aren't independent.
- **Resolution plan:**
  1. `Read flux_manifold/reservoir_state.py` for any module-level singletons
  2. `Read parser.py` — find `SuperpositionReservoir` instantiation; check if per-statement or per-program
- **Status:** OPEN

### UNK-021 — EvalContext initialization API for testing

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** Tests that want to inject initial variables (`cities = <pre-computed>`) need a way to do so before `run_file`. Determines test-writing ergonomics.
- **Resolution plan:** `Read parser.py:EvalContext` — check `__init__` for initial-bindings parameter; check if `variables` is mutable post-construction.
- **Status:** OPEN

### UNK-022 — OLI cache behavior on Blockscout outage

- **Surfaced:** 2026-05-13
- **Category:** Operational
- **Why it matters:** OLI guardrail consults local `oli_labels` cache. If new deployer has no row, `is_known_legitimate` returns None and adversarial classification proceeds. UNKNOWN if there's "fetch on miss" or strict cache-only.
- **Resolution plan:** `Read surveillance/oli_enrichment.py:is_known_legitimate` — should be ~10 lines. Document the policy explicitly.
- **Status:** OPEN

### UNK-023 — Should LOW-severity OLI tags also gate certain typologies?

- **Surfaced:** 2026-05-13
- **Category:** Operational (policy)
- **Why it matters:** Current guardrail catches HIGH severity. LOW severity (Web3 brand deployers like Animoca, Stabilize) is noted but not gated. Correction #20's FP class came from exactly this category. Whether intentional or omission unclear.
- **Resolution plan:** Policy decision, not pure-code question. Options:
  1. Block `pristine_solo_operator` promotion on any non-empty OLI tag (HIGH or LOW)
  2. Block only HIGH-severity (current behavior)
  3. Selective: LOW-severity blocks PSO but not ISO
  Update `_OLI_GUARDED_TRAP_SUBTYPES` × severity matrix in `entity_classifier.py` accordingly.
- **Status:** OPEN

---

### UNK-024 — README's "Latent Flux primitives power Layer 3's analysis layer" claim vs. zero imports in surveillance/

- **Surfaced:** 2026-05-13 (resolving UNK-001)
- **Category:** Architecture
- **Why it matters:** README line 34 makes an explicit integration claim: AttractorCompetition / ReservoirState / RecursiveFlow / FoldReference "power Layer 3's analysis layer." But `grep -r "from flux_manifold\|AttractorCompetition\|ReservoirState\|RecursiveFlow\|FoldReference" surveillance/` returns ZERO matches. The integration the README advertises does NOT exist in code (as of 2026-05-13 sync state).
- **Why this is load-bearing:** (a) STATE.md and any external claims about the project may inherit this aspirational-as-real framing. (b) Phase 3 integration paths I proposed earlier ("propose integration") are partly already-proposed in the README. (c) Whether the integration was never built, rolled back, or implemented under different names changes which of the 3 paths to pursue.
- **Possible explanations:**
  1. **Aspirational README** — claim was written before integration; integration deferred; README never updated
  2. **Integration done in pma/sba/** — flux_manifold is imported there; if surveillance EXECUTES pma/sba modules (e.g., via subprocess), the README claim is technically true but indirect
  3. **Implementation re-named** — algorithm copied into surveillance under different names (no `from flux_manifold` but conceptually same)
  4. **Integration rolled back** — was built, removed, README never updated
- **Resolution plan:**
  1. `Bash: git log -p --follow -S "AttractorCompetition" -- "surveillance/*"` to see if it was ever imported and reverted
  2. `Grep -r "reservoir\|attractor\|fold_reference\|recursive_flow" --include="*.py" surveillance/` for renamed implementations
  3. `Grep -r "import pma\|import sba\|from pma\|from sba" surveillance/` for indirect-use via adjacent subsystems
  4. `Bash: ls surveillance/ARCHITECTURE.md && head -200 surveillance/ARCHITECTURE.md` — README points here for end-to-end system; might describe the actual integration
  5. Reconcile finding in next session
- **Status:** OPEN
- **Priority:** HIGH for accurate STATE.md; LOW for unblocking concrete work

---

## Resolved (with confidence)

UNKNOWNs marked RESOLVED can be browsed inline above by status field. Index of RESOLVED entries (2026-05-13 first pass):

| ID | Confidence | Resolved at |
|---|---|---|
| UNK-001 | HIGH | 2026-05-13 |
| UNK-002 | HIGH | 2026-05-13 |
| UNK-005 | MEDIUM | 2026-05-13 (revisit-LOW candidate) |
| UNK-006 | MEDIUM | 2026-05-13 (revisit-LOW candidate) |

---

## Revisit-LOW queue

*Will populate as RESOLVED+LOW entries accumulate. The discipline: scan this section at session start and consider promoting LOW resolutions to MEDIUM or HIGH via primary-source verification.*
