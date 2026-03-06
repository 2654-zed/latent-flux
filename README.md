# FluxManifold Interpreter

**Version 1.0 – Proof-of-Concept**

Minimal interpreter for the FluxManifold (⟼) primitive from the Latent Flux language. Simulates continuous flow toward attractors in a high-dimensional latent space.

## Core Idea

Computation as continuous, gradient-like flow toward semantic attractors in latent space — no discrete steps, no human control flow. The state `s` flows toward attractor `q` via a vector field `f(s, q)`:

```
ds/dt = f(s, q)
s_{t+1} = s_t + ε * f(s_t, q)
```

```
s0 (initial state)
   |
   v
FluxManifold ⟼ q
   | (flow f, ε steps)
   v
Converged s (attractor)
   | (if drift < tol)
   v
Output state
```

## Quick Start

```bash
pip install -r requirements.txt
python run_benchmarks.py
```

## Project Structure

```
flux_manifold/
  __init__.py          # Public API
  core.py              # flux_flow / flux_flow_traced engine
  flows.py             # Flow functions: normalize, sin, damped, adaptive, repulsive
  baselines.py         # Baselines: random walk, gradient descent, static
  benchmarks.py        # Tier A/B/C benchmarks
  kill_tests.py        # Kill tests (fast disproval gates)
  monitor.py           # JSON logging / export
tests/
  test_core.py         # Unit tests for core + flows + safety
  test_benchmarks.py   # Benchmark & kill test assertions
  test_baselines.py    # Baseline tests
run_benchmarks.py      # Main entry point – runs all tiers + kill tests
requirements.txt       # numpy, scipy
```

## API

### `flux_flow(s0, q, f, epsilon=0.1, tol=1e-3, max_steps=1000) → np.ndarray`

Run flow from `s0` toward attractor `q` using flow function `f`. Returns converged state.

### `flux_flow_traced(...) → dict`

Same as above but returns full trace: `{converged_state, steps, converged, drift_trace, path}`.

### Flow Functions

| Function | Description |
|---|---|
| `normalize_flow(s, q)` | Normalized difference — straight pull toward `q` |
| `sin_flow(s, q)` | Sin-modulated magnitude — curved approach |
| `damped_flow(s, q)` | Damping proportional to distance |
| `adaptive_flow(s, q)` | Reduces step near attractor |
| `repulsive_flow(s, q)` | Pushes away (adversarial, for testing) |

## Benchmarks

| Tier | Description | Acceptance |
|---|---|---|
| **A** (micro) | 2D toy, s0=[0,0], q=[1,1] | mean steps <10, var <2 |
| **B** (sim) | 128D random manifolds vs baselines | t-test p<0.05, Cohen's d >0.5 |
| **C** (toy ARC) | Grid embeddings (placeholder) | Convergence on small puzzles |

## Kill Tests

1. **Convergence** – >20% non-converging runs → kill
2. **Drift** – Final drift > tol → kill
3. **vs Random** – Random walk beats flux → kill
4. **Scalability** – d=1024 time >5ms → kill
5. **Adversarial** – Repulsive f must diverge (expected)

## Running Tests

```bash
python -m pytest tests/ -v
```

## Outputs

Results are saved to `results/` as JSON and CSV files. Trace logs go to `logs/`.

## Constraints

- Dimension cap: d ≤ 1024
- No external APIs (local only)
- Deterministic: seed=42 for all benchmarks
- Toy prototype, not production

## Open Questions

- Optimal `f`? (Next: MLP-learned flow)
- Adaptive ε? (Gradient-based stepping)
- High-d scaling? (d=4096 → GPU path)
- Multi-attractor flow?
- Integration with ∑_ψ superposition?
