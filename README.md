# Latent Flux

**A continuous-flow programming language for latent-space computation.**

Latent Flux replaces discrete symbolic reasoning with continuous vector field dynamics. States flow toward semantic attractors through differentiable manifolds. There are no conditionals, no loops, no call stacks — only flow, superposition, and collapse.

```
ds/dt = f(s, q)       s ∈ ℝ^d,  q ∈ ℝ^d (attractor),  f: ℝ^d × ℝ^d → ℝ^d
```

---

## Table of Contents

1. [Language Primitives](#language-primitives)
2. [Execution Model](#execution-model)
3. [Expression Syntax](#expression-syntax)
4. [Pipeline Semantics](#pipeline-semantics)
5. [REPL](#repl)
6. [Quick Start](#quick-start)
7. [Full API Reference](#full-api-reference)
8. [TSP Proof-of-Concept](#tsp-proof-of-concept)
9. [Visualization](#visualization)
10. [Benchmarks & Kill Tests](#benchmarks--kill-tests)
11. [Project Structure](#project-structure)
12. [Tests](#tests)
13. [Design Principles](#design-principles)

---

## Language Primitives

Latent Flux has exactly **7 primitives**. Each operates on continuous states in ℝ^d. Together they form a complete compute cycle: explore → flow → critique → accept → commit → abstract.

| Symbol | Name | Signature | Semantics |
|--------|------|-----------|-----------|
| **⟼** | FluxManifold | `s₀, q, f → s*` | Continuous flow: iterate `s ← s + ε·f(s, q)` until `‖s − q‖ < tol` |
| **∑_ψ** | Superposition Tensor | `{sᵢ, wᵢ}ⁿ → batch` | Weighted parallel exploration of N candidate states |
| **∇↓** | Dimensional Squeeze | `ℝ^d → ℝ^k` where `k < d` | Lossy projection (PCA or random) to reduce dimensionality early |
| **≅** | Drift Equivalence | `(a, b, τ) → bool` | Approximate equality: `‖a − b‖ ≤ τ` (L2 or cosine) |
| **↓!** | Commitment Sink | `s → s` (irreversible) | Lock state when entropy is low or drift stabilizes. Cannot be undone. |
| **◉** | Fold-Reference | `(s, step) → s'` | Mid-flow self-critique: detect NaN, norm blowup, edge crossings; apply correction |
| **⇑** | Abstraction Cascade | `ℝ^d → [ℝ^k₁, ℝ^k₂, …]` | Hierarchical PCA reduction into decreasing-dimensional summaries |

### Formal Definitions

**⟼ FluxManifold (flow convergence)**
```
Given s₀ ∈ ℝ^d, q ∈ ℝ^d, f: ℝ^d × ℝ^d → ℝ^d, ε ∈ (0,1), τ > 0:
  δ = ε · f(s, q)
  δ ← clip(δ, ‖δ‖ ≤ 1)              # overflow guard
  δ ← δ · min(1, ‖q−s‖/‖δ‖)         # overshoot prevention
  s ← s + δ
  converged ⟺ ‖s − q‖ < τ
```
Operates on single states `(d,)` or batches `(N, d)`. Batch mode shares the attractor and evaluates all N states in a single vectorized loop.

**∑_ψ Superposition Tensor**
```
Given N states {s₁, …, sₙ} ∈ ℝ^(N×d) with weights w ∈ Δ^N (probability simplex):
  flow_all(q, f) → batch ⟼ for all N states simultaneously
  reweight(q)    → wᵢ ∝ softmax(−‖sᵢ − q‖)
  collapse_best  → argmin_i ‖sᵢ − q‖
  collapse_mean  → Σ wᵢ · sᵢ
  entropy        → −Σ wᵢ log₂ wᵢ   (high = spread, low = collapsed)
  prune(k)       → keep top-k by weight
```

**∇↓ Dimensional Squeeze**
```
fit(X) → learn projection P ∈ ℝ^(k×d)    (PCA eigenvectors or random Gaussian)
squeeze(s)   → P · s ∈ ℝ^k
unsqueeze(z) → P^T · z ∈ ℝ^d             (lossy reconstruction)
```

**≅ Drift Equivalence**
```
L2:     equivalent(a, b) ⟺ ‖a − b‖₂ ≤ τ
Cosine: equivalent(a, b) ⟺ 1 − cos(a, b) ≤ τ
quality(s, t) = max(0, 1 − distance(s,t)/τ)     ∈ [0, 1]
```

**↓! Commitment Sink**
```
Triggers when EITHER:
  entropy(∑_ψ) < threshold                        (weight collapse)
  mean(drift_trace[−window:]) < drift_threshold    (convergence plateau)
commit(s) → frozen copy. Second commit → error.
```

**◉ Fold-Reference**
```
At every `interval` steps during flow:
  (ok, diagnosis, correction) = critique_fn(s)
  if ¬ok: s ← correction; log(diagnosis)
Built-in critiques:
  no_nan_critique    → reject NaN/Inf, replace with zeros
  norm_bound(M)      → clip if ‖s‖ > M
  crossing_repulsion → geometric gradient to uncross TSP edges
```

**⇑ Abstraction Cascade**
```
Given states S ∈ ℝ^(N×d):
  level_0: S projected to ℝ^k₁ via PCA    (k₁ = d/2)
  level_1: level_0 projected to ℝ^k₂      (k₂ = k₁/2)
  …
  level_L: ℝ^(max(k_L, min_dim))
Each level returns: {dim, states, components, explained_variance_ratio}
```

---

## Execution Model

The interpreter orchestrates all 7 primitives in a fixed pipeline:

```
Input: attractor q ∈ ℝ^d

  ┌─────────────────────────────────────────────────────┐
  │ 1. ∇↓  Dimensional Squeeze   (compress if high-d)  │
  │ 2. ∑_ψ Superposition Tensor  (N random candidates) │
  │ 3. ⟼   FluxManifold          (batch flow → q)      │
  │ 4. ◉   Fold-Reference        (self-critique)       │
  │ 5. ≅   DriftEquivalence      (accept if close)     │
  │ 6. ↓!  Commitment Sink       (irreversible lock)   │
  │ 7. ⇑   Abstraction Cascade   (hierarchical view)   │
  └─────────────────────────────────────────────────────┘

Output: {committed_state, equivalence_quality, abstraction_levels, ...}
```

**Batch vectorization.** Step 3 (⟼) runs all N candidates in a single NumPy loop iteration — no Python for-loop over states. The flow function `f(S, q)` receives `S` as `(N, d)` and returns `(N, d)` gradients. Per-state convergence is tracked via a boolean mask; converged states are frozen in-place.

**No sequential contamination.** The superposition's `flow_all()` delegates to `flux_flow_traced_batch()`, which processes the full `(N, d)` matrix at each timestep. There is no serial dependency between candidates.

---

## Expression Syntax

Latent Flux has a dedicated expression parser that compiles source text into an AST and evaluates it as a pipeline.

### Grammar

```ebnf
pipeline  ::= atom (OP atom?)*
atom      ::= vector | number | symbol | func_call | '(' pipeline ')'
vector    ::= '[' number (',' number)* (';' number (',' number)*)* ']'
func_call ::= IDENT '(' args ')'
OP        ::= '⟼' | '∇↓' | '≅' | '↓!' | '⇑' | '◉' | '∑_ψ'
            | '->' | 'squeeze' | '~=' | 'commit' | 'cascade' | 'fold' | 'sum_psi'
            | '|'
```

### Operators

| Unicode | ASCII Alias | Operand | Description |
|---------|-------------|---------|-------------|
| `⟼` | `->`, `flow` | target vector | Flow current state toward target |
| `∑_ψ` | `sum_psi`, `superpose` | state(s) | Create or tag as superposition |
| `∇↓` | `squeeze` | integer `k` | Squeeze to `k` dimensions |
| `≅` | `~=`, `equiv` | float tolerance | Check drift equivalence |
| `↓!` | `commit` | — | Commit (irreversible) |
| `⇑` | `cascade` | integer levels | Build abstraction cascade |
| `◉` | `fold` | — | Apply fold-reference critique |
| `\|` | `\|` | — | Generic pipe (pass value through) |

### Built-in Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `random(n, d)` | int, int | SuperpositionTensor with N random states in ℝ^d |
| `zeros(d)` | int | Zero vector ∈ ℝ^d |
| `ones(d)` | int | Ones vector ∈ ℝ^d |
| `randn(d)` | int | Standard normal vector ∈ ℝ^d |
| `linspace(a, b, n)` | float, float, int | Evenly spaced 1D vector |
| `nearest_neighbor(cities)` | (n,2) array | NN tour encoded as continuous state |

### Examples

```
# Flow a single state to an attractor
[0.5, 0.5] ⟼ [1, 1]

# Superpose 10 random candidates in ℝ^32, flow, commit
∑_ψ random(10, 32) ⟼ zeros(32) | ↓!

# Full pipeline with squeeze + cascade
random(10, 64) | superpose | squeeze 16 | -> zeros(16) | fold | ~= 0.05 | commit | cascade 3

# Variable assignment in REPL
x = random(5, 8)
x ⟼ ones(8) | ↓!
```

---

## Pipeline Semantics

Evaluation proceeds **left-to-right**. Each operator receives the result of the previous stage as implicit input:

```
∑_ψ random(5, 8)  ⟼  zeros(8)  |  ↓!
│                  │            │   │
│ Create 5×8       │ Flow all   │   │ Commit best
│ superposition    │ toward 0   │   │ (irreversible)
│                  │            │   │
└──────────────────┴────────────┴───┘
      pipeline result: committed ndarray (8,)
```

**Type propagation rules:**

| Input Type | Operator | Output Type |
|------------|----------|-------------|
| `ndarray (d,)` | `⟼ target` | `ndarray (d,)` (flowed state) |
| `SuperpositionTensor` | `⟼ target` | `SuperpositionTensor` (all states flowed) |
| `SuperpositionTensor` | `↓!` | `ndarray (d,)` (collapsed best) |
| `any` | `∑_ψ` | `SuperpositionTensor` |
| `ndarray (d,)` | `∇↓ k` | `ndarray (k,)` |
| `SuperpositionTensor` | `∇↓ k` | `SuperpositionTensor` (squeezed) |
| `any` | `≅ τ` | `dict {equivalent, quality, distance}` |
| `any` | `⇑ L` | `list[dict]` (cascade levels) |
| `any` | `◉` | same type (possibly corrected) |

---

## REPL

Interactive read-eval-print loop with `LF>` prompt.

```bash
python -m flux_manifold.repl
```

### Syntax

```
LF> [0.5, 0.5] ⟼ [1, 1]
LF> x = random(5, 8)
LF> x ⟼ ones(8) | ↓!
  ↓! committed (entropy)
  → array(...)
LF> :vars
LF> :set epsilon 0.05
LF> :help
LF> :quit
```

### Commands

| Command | Effect |
|---------|--------|
| `:help` | Show syntax reference |
| `:vars` | List all bound variables |
| `:reset` | Clear variables and commitment state |
| `:set epsilon <v>` | Set flow step size |
| `:set tol <v>` | Set convergence tolerance |
| `:set maxsteps <v>` | Set iteration cap |
| `:set flow <name>` | Switch flow function (`normalize`, `sin`, `damped`, `adaptive`) |
| `:set seed <v>` | Set RNG seed |
| `:quit` / `:q` / `:exit` | Exit |

Assignment: `x = <expr>` or `let x = <expr>` binds the result to `x`.

Commitment messages (e.g. `↓! committed (entropy)`) are printed automatically when `↓!` fires.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt    # numpy, scipy, matplotlib

# Run all benchmarks + TSP demos
python run_benchmarks.py

# Run tests (172 tests)
python -m pytest tests/ -v

# Launch REPL
python -m flux_manifold.repl
```

### Minimal Python Example

```python
import numpy as np
from flux_manifold import (
    flux_flow, flux_flow_traced, flux_flow_batch,
    normalize_flow, SuperpositionTensor, LatentFluxInterpreter,
)

# Single flow
s0 = np.zeros(8)
q  = np.ones(8)
result = flux_flow(s0, q, normalize_flow, epsilon=0.1)
# result ≈ [1, 1, 1, 1, 1, 1, 1, 1]

# Traced flow (returns dict with drift_trace, path, steps, converged)
trace = flux_flow_traced(s0, q, normalize_flow)
print(f"Converged in {trace['steps']} steps, drift: {trace['drift_trace'][-1]:.6f}")

# Batch flow (N states at once)
S = np.random.default_rng(42).standard_normal((16, 8))
converged = flux_flow_batch(S, q, normalize_flow)  # (16, 8)

# Full interpreter pipeline (all 7 primitives)
interp = LatentFluxInterpreter(d=32, n_candidates=10)
result = interp.evaluate(q=np.ones(32))
print(f"Quality: {result['equivalence_quality']:.3f}, Steps: {result['total_steps']}")
```

### Expression Parser

```python
from flux_manifold.parser import run

# Parse and evaluate a Latent Flux expression
result = run("∑_ψ random(5, 8) ⟼ zeros(8) | ↓!")
print(result)  # committed ndarray (8,)

# With ASCII aliases
result = run("random(10, 32) | superpose | -> zeros(32) | commit")
```

---

## Full API Reference

### Core Engine — `flux_manifold.core`

```python
FlowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]   # f(s, q) → delta

flux_flow(s0, q, f, epsilon=0.1, tol=1e-3, max_steps=1000) → np.ndarray
    # Single state (d,) → converged state (d,)

flux_flow_traced(s0, q, f, epsilon=0.1, tol=1e-3, max_steps=1000) → dict
    # → {converged_state, steps, converged, drift_trace, path}

flux_flow_batch(S, q, f, epsilon=0.1, tol=1e-3, max_steps=1000) → np.ndarray
    # Batch (N,d) → converged states (N,d). Vectorized, no Python loop over N.

flux_flow_traced_batch(S, q, f, epsilon=0.1, tol=1e-3, max_steps=1000) → dict
    # → {converged_states (N,d), steps (N,), converged (N,),
    #    total_steps (int), drift_traces (max_iters, N)}
```

**Constraints:** `d ≤ 1024`, `ε ∈ (0, 1)`, `tol > 0`, `max_steps ≥ 1`.

### Flow Functions — `flux_manifold.flows`

All flows accept `(d,)` or `(N, d)` inputs via broadcasting.

| Function | Formula | Behavior |
|----------|---------|----------|
| `normalize_flow(s, q)` | `(q − s) / ‖q − s‖` | Unit step toward q |
| `sin_flow(s, q)` | `sin(‖q − s‖) · (q − s) / ‖q − s‖` | Curved approach |
| `damped_flow(s, q)` | `(q − s) · ‖q − s‖ / (1 + ‖q − s‖)` | Proportional damping |
| `adaptive_flow(s, q)` | `(q − s) / (1 + ‖q − s‖²)` | Gentle near attractor |
| `repulsive_flow(s, q)` | `−(q − s) / ‖q − s‖` | Push away (adversarial/testing) |

### SuperpositionTensor — `flux_manifold.superposition`

```python
st = SuperpositionTensor(states: np.ndarray, weights: np.ndarray | None = None)
st = SuperpositionTensor.from_random(n=16, d=64, seed=42, scale=1.0)

st.flow_all(q, f, epsilon, tol, max_steps) → dict     # batch trace
st.reweight_by_drift(q)                    → None      # softmax re-weight
st.collapse_to_best(q)                     → ndarray   # nearest to q
st.collapse_to_mean()                      → ndarray   # weighted centroid
st.prune(keep=5)                           → None      # top-k by weight
st.entropy()                               → float     # Shannon entropy (bits)
st.mean_state()                            → ndarray   # weighted average
```

**Attributes:** `states (N,d)`, `weights (N,)`, `n`, `d`

### DriftEquivalence — `flux_manifold.drift_equivalence`

```python
de = DriftEquivalence(tolerance=0.05, metric="l2")  # or "cosine"

de.equivalent(a, b) → bool
de.distance(a, b)   → float
de.quality(s, t)     → float          # ∈ [0, 1]
de.best_equivalent(candidates, target) → (index, distance)
```

### CommitmentSink — `flux_manifold.commitment_sink`

```python
cs = CommitmentSink(entropy_threshold=0.5, drift_window=10, drift_threshold=0.01)

cs.should_commit_entropy(superposition) → bool
cs.should_commit_drift(drift_trace)     → bool
cs.commit(state, reason="manual")       → ndarray   # irreversible
cs.try_commit(superposition, q, drift_trace) → ndarray | None
```

**Attributes:** `committed (bool)`, `committed_state`, `commit_reason`, `state (property)`

### AbstractionCascade — `flux_manifold.abstraction_cascade`

```python
ac = AbstractionCascade(levels=3, min_dim=2)

ac.cascade(states)        → list[dict]    # {level, dim, states, components, explained_variance_ratio}
ac.cascade_single(state)  → list[dict]    # {level, dim, state}  (truncation-based)
```

### FoldReference — `flux_manifold.fold_reference`

```python
fr = FoldReference(critique_fn, interval=10, max_corrections=50)

fr.check(state, step)     → (corrected_state, was_corrected: bool)
fr.reset()                → None
fr.corrections_count      → int
fr.history                → list[dict]

# Built-in critique functions:
no_nan_critique(state)         → (ok, diagnosis, correction | None)
norm_bound_critique(max_norm)  → CritiqueFn   # factory
```

### DimensionalSqueeze — `flux_manifold.dimensional_squeeze`

```python
ds = DimensionalSqueeze(target_dim=16, method="pca")  # or "random_projection"

ds.fit(data, seed=42)   → self
ds.squeeze(state)        → ndarray   # (d,) → (k,)  or  (N,d) → (N,k)
ds.unsqueeze(compressed) → ndarray   # (k,) → (d,)  lossy
ds.compression_ratio     → float | None
```

### LatentFluxInterpreter — `flux_manifold.interpreter`

```python
interp = LatentFluxInterpreter(
    d=64, n_candidates=16, flow_fn=normalize_flow,
    epsilon=0.1, tol=1e-3, max_steps=500,
    equiv_tolerance=0.05, entropy_threshold=0.5,
    drift_window=10, drift_commit_threshold=0.01,
    cascade_levels=3, critique_fn=None, critique_interval=10,
    squeeze_dim=None, seed=42,
)

result = interp.evaluate(q, initial_states=None) → dict
# Returns:
#   committed_state        ndarray (d,)
#   equivalence_quality    float ∈ [0, 1]
#   is_equivalent          bool
#   abstraction_levels     list[dict]
#   total_steps            int
#   converged_count        int
#   n_candidates           int
#   fold_corrections       int
#   superposition_entropy  float
#   commit_reason          str
```

### Parser & Evaluator — `flux_manifold.parser`

```python
from flux_manifold.parser import parse, evaluate, run, EvalContext

ast    = parse("∑_ψ random(5, 8) ⟼ zeros(8)")     # → LFPipeline AST
result = evaluate(ast, ctx)                           # → value
result = run("∑_ψ random(5, 8) ⟼ zeros(8)")         # parse + eval

ctx = EvalContext(seed=42, epsilon=0.1, tol=1e-3, max_steps=500,
                  flow_name="normalize", on_message=print)
ctx.set("x", value)
ctx.get("x")
```

**AST node types:** `LFVector`, `LFNumber`, `LFSymbol`, `LFFuncCall`, `LFPipeline`, `LFOp`

### TSP Solver — `flux_manifold.tsp_solver`

```python
from flux_manifold import LatentFluxTSP, solve_tsp

# Class-based
solver = LatentFluxTSP(cities, n_candidates=20, epsilon=0.15, seed=42)
result = solver.solve()

# One-liner
result = solve_tsp(cities, n_candidates=20, seed=42)

# Result dict:
#   best_tour         list[int]
#   best_length       float
#   nn_length          float (nearest-neighbor baseline)
#   improvement_pct    float
#   total_steps        int
#   converged_count    int
#   fold_corrections   int
#   all_lengths        list[float]

# Utilities
tour_length(cities, order)         → float
order_to_state(order, n_cities)    → ndarray
state_to_order(state)              → ndarray
nearest_neighbor_tour(cities)      → ndarray
make_tsp_crossing_flow(cities, repulsion_strength=0.3)  → FlowFn
make_crossing_critique(cities, repulsion_strength=0.3)  → CritiqueFn
```

**Crossing repulsion** replaces discrete 2-opt swaps with a continuous geometric gradient. For each pair of crossing edges, it computes a repulsion vector that pushes the state toward an uncrossed configuration — no discrete combinatorial search.

### Visualization — `flux_manifold.visualize`

All functions return `matplotlib.figure.Figure` and accept an optional `save_path`.

```python
from flux_manifold.visualize import (
    plot_flow_2d,                # 2D flow trajectory with arrows
    plot_convergence,            # Log-scale drift vs steps
    plot_convergence_comparison, # Multi-trace overlay
    plot_superposition_2d,       # Scatter plot sized by weight
    plot_tsp_tour,               # Single TSP tour
    plot_tsp_comparison,         # Side-by-side tour grid
    plot_cascade,                # Bar chart per abstraction level
    plot_commitment_timeline,    # Dual-axis drift + entropy timeline
)
```

---

## TSP Proof-of-Concept

The TSP solver demonstrates all 7 primitives working together on a real combinatorial problem:

```
Cities ∈ ℝ^(n×2)
  → encode tour as continuous vector ∈ ℝ^n   (order_to_state)
  → nearest_neighbor_tour → attractor q
  → ∑_ψ: 20 random tour candidates
  → ⟼: flow all candidates toward q via tsp_crossing_flow
  → ◉: geometric crossing repulsion (detect + resolve edge crossings)
  → ≅: accept tours within tolerance of best
  → ↓!: commit when entropy < threshold
  → ⇑: build abstraction hierarchy of final tour
  → decode best state back to permutation     (state_to_order)
```

```python
import numpy as np
from flux_manifold import solve_tsp

cities = np.random.default_rng(42).uniform(0, 100, (15, 2))
result = solve_tsp(cities, n_candidates=20)
print(f"Tour length: {result['best_length']:.1f} (NN baseline: {result['nn_length']:.1f})")
print(f"Improvement: {result['improvement_pct']:.1f}%")
```

---

## Benchmarks & Kill Tests

### Benchmarks

| Tier | Domain | Acceptance Criteria |
|------|--------|---------------------|
| **A** (micro) | 2D: s₀=[0,0] → q=[1,1] | mean steps < 10, variance < 2, 100% convergence |
| **B** (simulation) | 128D random manifolds | t-test p < 0.05 vs random walk, Cohen's d > 0.5 |
| **C** (toy ARC) | Grid embeddings | Convergence on small puzzles (placeholder) |

### Kill Tests (fast disproval)

| Test | Condition | Kills If |
|------|-----------|----------|
| Convergence | 128D, 100 runs | > 20% fail to converge |
| Drift | 128D, 50 runs | Any final drift > tol |
| vs Random Walk | 128D, 50 runs | Flux wins < 50% of matchups |
| Scalability | d=1024, 10 runs | Mean time > 5ms per run |
| Adversarial | repulsive_flow | System fails to detect divergence |

```bash
python run_benchmarks.py       # Runs all tiers + kill tests + TSP demos
                               # Outputs to results/*.json, results/*.csv
```

---

## Project Structure

```
flux_manifold/
  __init__.py                 # Public API + __all__ exports
  core.py                     # ⟼  flux_flow, flux_flow_traced, _batch variants
  flows.py                    # Vector fields: normalize, sin, damped, adaptive, repulsive
  superposition.py            # ∑_ψ SuperpositionTensor
  drift_equivalence.py        # ≅  DriftEquivalence
  commitment_sink.py          # ↓! CommitmentSink
  abstraction_cascade.py      # ⇑  AbstractionCascade
  fold_reference.py           # ◉  FoldReference + built-in critiques
  dimensional_squeeze.py      # ∇↓ DimensionalSqueeze
  interpreter.py              # Full 7-primitive pipeline orchestrator
  tsp_solver.py               # TSP proof-of-concept (crossing repulsion)
  parser.py                   # Expression parser + AST + evaluator
  repl.py                     # Interactive REPL (LF> prompt)
  visualize.py                # 8 matplotlib plot functions
  baselines.py                # Random walk, gradient descent, static
  benchmarks.py               # Tier A/B/C benchmarks
  kill_tests.py               # 5 kill tests
  monitor.py                  # JSON trace logging

tests/
  test_core.py                # Core engine + batch flow + safety
  test_primitives.py          # All 6 additional primitives
  test_interpreter.py         # Interpreter + TSP + crossing repulsion
  test_parser_repl.py         # Tokenizer + parser + evaluator + REPL
  test_benchmarks.py          # Benchmark pass/fail assertions
  test_baselines.py           # Baseline correctness
  test_visualize.py           # Plot generation

run_benchmarks.py             # Entry point: all benchmarks + TSP demos
requirements.txt              # numpy>=1.24, scipy>=1.10, matplotlib>=3.7
```

---

## Tests

172 tests across 7 test files.

```bash
python -m pytest tests/ -v
```

| File | Focus | Count |
|------|-------|-------|
| `test_core.py` | Core flow, batch ops, input validation, NaN/overflow safety | ~26 |
| `test_primitives.py` | ∑_ψ, ≅, ↓!, ⇑, ◉, ∇↓ | ~40 |
| `test_interpreter.py` | Full pipeline, TSP encoding, geometric crossing | ~24 |
| `test_parser_repl.py` | Tokenizer, parser, AST evaluation, REPL commands | ~41 |
| `test_benchmarks.py` | Tier A/B/C, kill tests | 12 |
| `test_baselines.py` | Random walk, gradient descent, static | 5 |
| `test_visualize.py` | All 8 plot functions | 12 |

---

## Design Principles

1. **No sequential contamination.** Batch operations are fully vectorized via NumPy broadcasting. `flow_all()` calls `flux_flow_traced_batch()` — one matrix operation per timestep, not a Python loop over candidates.

2. **No human heuristics.** The TSP fold-reference uses continuous geometric crossing repulsion, not discrete 2-opt swaps. Every correction is a differentiable gradient, not a combinatorial search.

3. **Irreversibility is a feature.** `↓!` (commitment) cannot be undone. This prevents infinite reconsideration and forces the system to converge to a decision.

4. **Everything is a flow.** There are no conditionals or branches. Control is replaced by continuous dynamics: flow strength, drift thresholds, entropy decay, and geometric gradients.

5. **Composable primitives.** Each primitive is a standalone class with no hidden dependencies. The interpreter is one possible wiring; the parser lets you compose them in any order.

6. **Observable.** Every operation produces traceable output: drift traces, step counts, convergence flags, entropy values, correction histories. Nothing is opaque.

---

## Constraints

- Dimension cap: d ≤ 1024
- Local-only: no external API calls
- Deterministic: `seed=42` default for reproducibility
- Proof-of-concept: not production-optimized

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
Python ≥ 3.10
```

## License

Unlicensed / experimental.
