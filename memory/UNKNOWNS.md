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
- **Status:** RESOLVED
- **Confidence:** HIGH (primary-source grep + docstring inspection)
- **Resolved at:** 2026-05-13
- **Resolved by:** **lx-scanner is independent of both Latent Flux and L3 surveillance.** `grep -r "from flux_manifold|reservoir|SuperpositionTensor|flux_flow" lx-scanner/` returns ZERO matches. `live_feed_scanner.py` docstring: *"Live Feed Scanner — watches 5 DEXs on Arbitrum, logs price gaps. Polls all 5 DEXs every 5 seconds for WETH/USDC quotes."* `mev_arbitration_router.py` docstring: *"MEV arbitration router — collects quotes from all 5 DEXs and picks the best. Phase 1: quote comparison only. No execution."* — implementation is plain Python dict comparison. lx-scanner is a self-contained MEV arbitrage scanner that happens to live in the same git tree but shares no code with the other subsystems.

### UNK-008 — Surveillance-side test coverage

- **Surfaced:** 2026-05-13
- **Category:** Subsystem / Operational
- **Why it matters:** `tests/` at repo root contains flux_manifold tests. UNKNOWN whether surveillance-specific tests exist. Without them, agents modifying `bytecode_classifier.py`, `entity_classifier.py`, `oli_enrichment.py` have no verification surface.
- **Resolution plan:** `Glob tests/**/*surveillance* tests/**/*oli* tests/**/*classifier*`. If empty, the gap is real → Action 4 of the previous Phase 4 output applies.
- **Status:** RESOLVED
- **Confidence:** HIGH (primary-source verified — `ls tests/surveillance/` returns "No such file or directory"; all 11 test files in `tests/` are Latent Flux DSL tests; `grep -r "from surveillance|import surveillance" tests/` returns empty)
- **Resolved at:** 2026-05-13
- **Resolved by:** **Zero surveillance-side test coverage.** The full test suite (`tests/test_attractor_competition.py`, `test_baselines.py`, `test_benchmarks.py`, `test_convergence_reservoir.py`, `test_core.py`, `test_infrastructure.py`, `test_interpreter.py`, `test_ontology.py`, `test_parser_repl.py`, `test_primitives.py`, `test_recursive_flow.py`, `test_visualize.py`) targets `flux_manifold/` exclusively. No file imports anything from `surveillance/`. The gap is real and confirmed. → Unblocks Action 4 (write `tests/surveillance/test_smoke.py`) from the prior Phase 4 list.

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
- **Resolution plan:** *(see prior — git log -S, alt-name grep, pma/sba indirect, ARCHITECTURE.md)*
- **Status:** RESOLVED
- **Confidence:** HIGH (4 hypothesis tests all ran; all returned definitive empty/null)
- **Resolved at:** 2026-05-13
- **Resolved by:** **Hypothesis 1 (aspirational README) is correct.** All other hypotheses ruled out by primary-source verification:
  - `git log --all -S "AttractorCompetition" -- surveillance/` → EMPTY (never imported, never rolled back)
  - `git log --all -S "ReservoirState" -- surveillance/` → EMPTY
  - `git log --all -S "RecursiveFlow" -- surveillance/` → EMPTY
  - `git log --all -S "FoldReference" -- surveillance/` → EMPTY
  - `git log --all -S "from flux_manifold" -- surveillance/` → EMPTY
  - `grep -r "reservoir\|attractor\|fold_reference\|recursive_flow" surveillance/ --include="*.py"` → EMPTY (no renamed implementations)
  - `grep -r "import pma\|import sba\|from pma\|from sba" surveillance/ --include="*.py"` → EMPTY (no indirect use)
  - `surveillance/ARCHITECTURE.md` does NOT mention flux_manifold (the only matches for "latent.flux" are incidental repo-URL slug references in the `git clone` command and GitHub footer)
  - The README claim is unique to README and has never been backed by code.
- **Implications:**
  - STATE.md "Project identity" should mark the integration as aspirational (DONE 2026-05-13 — already captured in the "Documented integration claim NOT verified in code" line)
  - Phase 3 Integration Hypotheses (regime-monitor / detector-as-DSL / behavioral-classifier) are net-new work, NOT duplicating existing functionality
  - README needs an update: either remove the claim or build the integration. **Decision deferred — see ADR-006 candidate**
- **Adjacent finding:** ARCHITECTURE.md has the same stale corpus numbers as README ("124,341 contracts | 1.17M transaction events | 36,115 deployers" vs. current 284K / 16.8M / 67K). Both docs written together; neither updated since. Documentation-freshness is a separate pattern worth tracking.

---

### UNK-025 — Bytecode classification of Apr-25 Optimism mass-deploy templates `0x476b1553` and `0xc3314989`

- **Surfaced:** 2026-05-15, by 2026-05-15-phase-a
- **Category:** Subsystem
- **Why it matters:** The Apr-25 Optimism deployer surge produced 6,606 contracts from these two bytecode hashes (87% + 12.5% of the cohort). 838 already classify as `suspected`; the rest are `unknown`. If the templates are benign (e.g., AA wallet factories), the entire cohort is corpus noise. If malicious (trap variant), this is a single-day 6K+ adversarial deployment event — an enormous spike in confirmed-bad volume.
- **Resolution plan:** Pull a sample contract for each hash; decompile or use `surveillance/bytecode_classifier.py` PATTERN_REGISTRY to evaluate. Cross-reference against known AA wallet templates (Safe, Biconomy, ZeroDev, Alchemy AA SDK).
- **Status:** OPEN

### UNK-026 — Does `approval_watchlist` filter out confirmed-tier contracts?

- **Surfaced:** 2026-05-15, by 2026-05-15-phase-a
- **Category:** Subsystem
- **Why it matters:** Phase A3 found Coffee Fleet (`0xc0ffeefeed`) contributes ZERO approvals across Apr-22..27 despite 416 deployed contracts and 48 new deployments in window. The likely explanation is that `approval_watchlist` filters by `contract_tier` and excludes confirmed-tier contracts. If true, the regime-monitor's `approval_events_per_day` signal is fundamentally "new-victim signal on suspected-tier", NOT "any approval anywhere". This changes how to interpret regime alerts on that signal.
- **Resolution plan:** Read `surveillance/approval_monitor.py` and `surveillance/approval_drain_monitor.py` to find the tier filter (or its absence). Confirm by querying: do ANY confirmed-tier contracts appear in approval_watchlist? If zero, hypothesis confirmed.
- **Status:** OPEN

### UNK-027 — Identity of `0xb0b0b690` referenced in prior session memory as "Apr-25 vanity-funder mass-fund operator"

- **Surfaced:** 2026-05-15, by 2026-05-15-phase-a
- **Category:** Operational
- **Why it matters:** NEXT_SESSION_PLAN.md attributed the Apr-25 deployer mass (8,052) to `0xb0b0b690` as a "vanity-funder mass-fund event". Phase A1 found ZERO matches for `0xb0b0b6` anywhere in `deployers.funding_sources`, `deployers.known_associated_deployers`, or any other column. The address may have been (a) a prior-session hallucination, (b) a real entity that was retracted/deleted, (c) referenced under a different prefix. If (a), every other prior-session attribution chain should be re-verified.
- **Resolution plan:** `git log -p` search for `b0b0b690` and `b0b0b6` across all branches and history. Also grep memory/, docs/, reports/, surveillance/data/cases/. If still nothing, mark as prior-session hallucination and add to a "named-entity verification" discipline note.
- **Status:** OPEN

### UNK-028 — INDEX.md "May-5 = iter_8 spawn day" claim vs zero iter_8-traceable contracts on May-5

- **Surfaced:** 2026-05-15, by 2026-05-15-phase-a
- **Category:** Operational
- **Why it matters:** NEXT_SESSION_PLAN.md cited INDEX.md for "May-5 = iter_8 spawn day of drainer-spawn hub 0xf7883e3f, wallet 0xa8c7ac1cdc33". Phase A2 found zero contracts deployed by either address on May-5 (or anywhere in the corpus). Either (a) the iter_8 wallet prefix is wrong, (b) the May-5 mapping is wrong, (c) the entire iter_8 claim is stale or fabricated. Worth verifying against INDEX.md's actual text.
- **Resolution plan:** Grep INDEX.md for `iter_8`, `iter 8`, `0xf7883e3f`, `0xa8c7ac1cdc33`, and `drainer-spawn`. Verify what INDEX.md actually says vs what NEXT_SESSION_PLAN.md inferred. Update or retract.
- **Status:** OPEN

### UNK-029 — THORChain (2026-05-15) exploit mechanism for Attack 11/9/10/15 mapping

- **Surfaced:** 2026-05-15, by 2026-05-15-attacks-v3
- **Category:** Theoretical (threat-modeling)
- **Why it matters:** Determines whether POTENTIAL_ATTACKS_V3.md's tentative Attack 15 candidacy gets promoted to a standalone category or absorbed into Attack 9/10/11. Affects detection-hook priorities (validator-set monitoring vs cross-chain proof verification vs admin compromise).
- **Resolution plan:** Watch for THORChain post-mortem from THORChain team, Halborn, Blockaid, or Chainalysis. Map mechanism to (a) validator-attestation forgery → Attack 9; (b) cryptographic proof bypass → Attack 10; (c) validator-key compromise → Attack 11a; (d) novel native-swap economic exploit → Attack 15.
- **Status:** OPEN

### UNK-030 — What caused the May-5 confirmed-trap re-classification pulse (53.8% backfill)?

- **Surfaced:** 2026-05-15, by 2026-05-15-phase-a
- **Category:** Subsystem
- **Why it matters:** A2 showed 113 of 210 May-5 confirmed-tier contracts had deployer.first_seen BEFORE May-5 — they were deployed earlier and only newly classified on May-5. This is either a classifier rule change or a backfill/re-scan job. Knowing which informs how to interpret similar pulses in the future and whether the regime monitor needs to mask out classifier-pulse vs deployment-pulse signal.
- **Resolution plan:** `git log --since="2026-05-04" --until="2026-05-06" -- surveillance/bytecode_classifier.py surveillance/pattern_*.py surveillance/db.py`. Also check `surveillance/scripts/` for any re-classification job around that date.
- **Status:** OPEN

---

## Resolved (with confidence)

UNKNOWNs marked RESOLVED can be browsed inline above by status field. Index of RESOLVED entries:

| ID | Confidence | Resolved at |
|---|---|---|
| UNK-001 | HIGH | 2026-05-13 |
| UNK-002 | HIGH | 2026-05-13 |
| UNK-005 | MEDIUM | 2026-05-13 (revisit-LOW candidate) |
| UNK-006 | MEDIUM | 2026-05-13 (revisit-LOW candidate) |
| UNK-007 | HIGH | 2026-05-13 |
| UNK-008 | HIGH | 2026-05-13 |
| UNK-024 | HIGH | 2026-05-13 |

**RESOLVED count: 7 of 30 total (23%)**
**OPEN count: 23**
**revisit-LOW queue: 2 (UNK-005, UNK-006)**

---

## Revisit-LOW queue

*Will populate as RESOLVED+LOW entries accumulate. The discipline: scan this section at session start and consider promoting LOW resolutions to MEDIUM or HIGH via primary-source verification.*
