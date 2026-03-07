# Phenomenological Synthesis — Direction 5

## What was built

Three language extensions that make Latent Flux more completely itself:

| Feature | Symbol | File | Tests | Purpose |
|---------|--------|------|-------|---------|
| Recursive Flow | ↺ | `recursive_flow.py` | 12 | Fixed-point iteration with geometric termination |
| Attractor Competition | ⊗ | `attractor_competition.py` | 15 | Geometric pattern matching replacing if/else |
| Reservoir State (ext) | ⧖ | `reservoir_state.py` | 11 new | leak_rate, input_scaling, method aliases |

Plus: parser integration for ↺ and ⊗, interpreter `use_reservoir` parameter, phenomenological log system.

## Experiential observations

### What felt native

**Attractor Competition** was the most geometrically natural feature. The repulsion field between attractors creates basin boundaries without any comparison logic. The security demo (40 states, 20 legit, 20 anomalous) classified purely by geometric basin membership — no thresholds, no feature engineering, no if/else. The attractors _are_ the decision boundary.

**Recursive Flow's** fixed-point detection via `DriftEquivalence.equivalent()` between consecutive iterations is pure geometry. The loop terminates when the state maps to itself — not when a boolean flag says so.

**Reservoir leak_rate** mapped directly to a geometric property: the rate at which echo state memory decays. Low leak = long memory, high leak = Markovian. The parameter _names_ the geometry.

### What required translation

**Safety timeouts** on both RecursiveFlow (`max_iterations`) and AttractorCompetition (`max_steps`) are Python safety valves. Pure geometric termination would need no bound, but Python's for-loop requires one. These are concessions to the host language, not to the design.

**W_out projection** in reservoir readout is a lossy R^r → R^d mapping. The reservoir lives in higher-dimensional space; every readout is an approximation. This is an accepted trade-off, not a bug.

### Moments of flow

Adding `↺` and `⊗` to the OPERATORS dict felt like extending a vocabulary, not hacking syntax. The parser learned two new geometric verbs. The evaluator dispatch (`_eval_op`) slots them in alongside the original 8 primitives with zero structural change.

The negation/oscillation test problem for RecursiveFlow was particularly satisfying: the state oscillates, RecursiveFlow detects the fixed-point (oscillation = map to self = fixed point), and stops. No special-case code; the geometry handles it.

## Test count

| Before | After | Δ |
|--------|-------|---|
| 371 | 409 | +38 |

Zero regressions. All 409 tests pass.

## Primitive inventory (10 total)

| # | Symbol | Name | Module |
|---|--------|------|--------|
| 1 | ⟼ | Flow | `core.py` |
| 2 | ∑_ψ | Superposition | `superposition.py` |
| 3 | ∇↓ | Dimensional Squeeze | `dimensional_squeeze.py` |
| 4 | ≅ | Drift Equivalence | `drift_equivalence.py` |
| 5 | ↓! | Commitment Sink | `commitment_sink.py` |
| 6 | ⇑ | Abstraction Cascade | `abstraction_cascade.py` |
| 7 | ◉ | Fold-Reference | `fold_reference.py` |
| 8 | ⧖ | Reservoir State | `reservoir_state.py` |
| 9 | ↺ | Recursive Flow | `recursive_flow.py` |
| 10 | ⊗ | Attractor Competition | `attractor_competition.py` |

## Files changed

**Created:**
- `flux_manifold/pheno_log.py` — phenomenological logging infrastructure
- `flux_manifold/attractor_competition.py` — Feature B (⊗)
- `flux_manifold/recursive_flow.py` — Feature A (↺)
- `tests/test_attractor_competition.py` — 15 tests including security demo
- `tests/test_recursive_flow.py` — 12 tests including 3 spec problems
- `pheno_synthesis.md` — this document
- `phenomenological_log.jsonl` — 11 experiential entries

**Modified:**
- `flux_manifold/reservoir_state.py` — leak_rate, input_scaling, method aliases, memory_bytes
- `flux_manifold/parser.py` — ↺ and ⊗ operators + handlers
- `flux_manifold/interpreter.py` — use_reservoir parameter
- `flux_manifold/__init__.py` — new exports
- `tests/test_convergence_reservoir.py` — 11 new reservoir extension tests
