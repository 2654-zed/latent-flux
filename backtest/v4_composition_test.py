"""V4 Composition Test — first-ever composition of RecursiveFlow + AttractorCompetition + ReservoirState.

Standalone diagnostic script (not pytest). Runs a scenario mirroring
the intended ETH arbitrage backtest use case:

  1. ReservoirState processes 100 sequential synthetic "pool price" vectors
     with leak_rate=0.1 (fast decay, ~12s block time analog).
  2. The reservoir readout becomes the initial state for path selection.
  3. AttractorCompetition selects between 3 candidate "arbitrage paths"
     (modeled as attractors) based on basin membership.
  4. RecursiveFlow iterates the winning path toward its attractor until
     fixed-point convergence.

Records: convergence behavior, iteration counts, basin selection,
edge cases (non-convergence, contested basins, reservoir drift).

Deterministic: seed=42 throughout, identical output on every run.

Usage:
    python backtest/v4_composition_test.py
"""

from __future__ import annotations

import sys
import os

# Ensure repo root is on sys.path so flux_manifold is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flux_manifold.reservoir_state import ReservoirState
from flux_manifold.recursive_flow import RecursiveFlow
from flux_manifold.attractor_competition import AttractorCompetition
from flux_manifold.flows import normalize_flow


# ── Configuration ─────────────────────────────────────────────────

SEED = 42
D = 6           # state dimensionality (e.g., 6 token-pair price dimensions)
N_PRICES = 100  # number of sequential price observations
LEAK_RATE = 0.1 # fast decay for ~12s block time
NUM_ATTRACTORS = 3
NUM_SCENARIOS = 5  # different starting conditions to test


def generate_synthetic_prices(rng: np.random.Generator, d: int, n: int) -> np.ndarray:
    """Generate 100 synthetic pool price vectors with realistic drift.

    Simulates a random walk with mean-reversion and occasional jumps,
    mimicking real AMM pool price evolution at block granularity.

    Returns (n, d) array of price vectors.
    """
    prices = np.zeros((n, d), dtype=np.float32)
    # Start near a "fair value"
    fair_value = rng.uniform(0.5, 2.0, size=d).astype(np.float32)
    prices[0] = fair_value + rng.standard_normal(d).astype(np.float32) * 0.1

    for t in range(1, n):
        # Mean-reverting random walk
        noise = rng.standard_normal(d).astype(np.float32) * 0.05
        reversion = 0.02 * (fair_value - prices[t - 1])
        # Occasional jump (~5% chance)
        jump = np.zeros(d, dtype=np.float32)
        if rng.random() < 0.05:
            jump_idx = rng.integers(0, d)
            jump[jump_idx] = rng.standard_normal() * 0.3
        prices[t] = prices[t - 1] + noise + reversion + jump

    return prices


def define_attractors(rng: np.random.Generator, d: int) -> tuple[np.ndarray, list[str]]:
    """Define 3 candidate arbitrage path attractors in R^d.

    Each attractor represents a "profitable path endpoint" — a state
    the system should converge to if that path is optimal.

    Attractors are spread out in R^d to create distinct basins.
    """
    attractors = np.array([
        rng.uniform(-1.0, 0.0, size=d),   # Path A: low-price regime
        rng.uniform(0.5, 1.5, size=d),     # Path B: mid-price regime
        rng.uniform(2.0, 3.0, size=d),     # Path C: high-price regime
    ], dtype=np.float32)
    labels = ["path_A_low", "path_B_mid", "path_C_high"]
    return attractors, labels


def run_reservoir_phase(prices: np.ndarray, d: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Phase 1: Feed price sequence through ReservoirState.

    Returns:
        final_readout: The reservoir's ℝ^d readout after processing all prices.
        readout_history: List of readouts at each step (for diagnostics).
    """
    reservoir = ReservoirState(
        d=d,
        reservoir_scale=4,
        spectral_radius=0.9,
        input_scaling=0.1,
        leak_rate=LEAK_RATE,
        seed=SEED,
    )

    readout_history = []
    for t in range(prices.shape[0]):
        readout = reservoir.step(prices[t])
        readout_history.append(readout.copy())

    return readout_history[-1], readout_history


def run_competition_phase(
    state: np.ndarray,
    attractors: np.ndarray,
    labels: list[str],
) -> dict:
    """Phase 2: AttractorCompetition selects the winning basin.

    Returns the compete() result dict.
    """
    competition = AttractorCompetition(
        attractors=attractors,
        labels=labels,
        flow_fn=normalize_flow,
        epsilon=0.1,
        tol=1e-2,
        max_steps=500,
        repulsion=0.05,
        seed=SEED,
    )
    return competition.compete(state)


def run_convergence_phase(
    state: np.ndarray,
    attractor: np.ndarray,
) -> dict:
    """Phase 3: RecursiveFlow converges toward the winning attractor.

    Returns the run() result dict.
    """
    flow = RecursiveFlow(
        flow_fn=normalize_flow,
        attractor=attractor,
        epsilon=0.1,
        tol=1e-3,
        max_iterations=100,
        inner_steps=50,
        fixed_point_tol=1e-4,
        seed=SEED,
    )
    return flow.run(state)


def run_full_pipeline(
    scenario_id: int,
    prices: np.ndarray,
    attractors: np.ndarray,
    labels: list[str],
) -> dict:
    """Run the full three-stage composition pipeline for one scenario.

    Returns a dict with all diagnostics.
    """
    d = prices.shape[1]
    result = {"scenario_id": scenario_id}

    # ── Phase 1: Reservoir ────────────────────────────────────
    final_readout, readout_history = run_reservoir_phase(prices, d)
    result["reservoir"] = {
        "steps_processed": len(readout_history),
        "final_readout_norm": float(np.linalg.norm(final_readout)),
        "readout_drift": float(np.linalg.norm(readout_history[-1] - readout_history[0])),
        "final_readout": final_readout.copy(),
    }

    # ── Phase 2: Competition ──────────────────────────────────
    comp_result = run_competition_phase(final_readout, attractors, labels)
    result["competition"] = {
        "winner": comp_result["winner"],
        "winner_idx": comp_result["winner_idx"],
        "certainty": comp_result["certainty"],
        "margin": comp_result["margin"],
        "contested": comp_result["contested"],
        "trajectory_steps": len(comp_result["trajectory"]),
    }

    # ── Phase 3: Convergence ──────────────────────────────────
    winning_attractor = attractors[comp_result["winner_idx"]]
    # Start convergence from the competition's final position
    competition_final = comp_result["trajectory"][-1]
    conv_result = run_convergence_phase(competition_final, winning_attractor)
    result["convergence"] = {
        "iterations": conv_result["iterations"],
        "converged": conv_result["converged"],
        "termination": conv_result["termination"],
        "final_drift": float(np.linalg.norm(
            conv_result["final_state"] - winning_attractor
        )),
        "final_state_norm": float(np.linalg.norm(conv_result["final_state"])),
    }

    return result


def print_result(result: dict) -> None:
    """Pretty-print a single scenario result."""
    sid = result["scenario_id"]
    r = result["reservoir"]
    c = result["competition"]
    v = result["convergence"]

    print(f"\n{'='*60}")
    print(f"  Scenario {sid}")
    print(f"{'='*60}")

    print(f"\n  [Reservoir] steps={r['steps_processed']}  "
          f"readout_norm={r['final_readout_norm']:.4f}  "
          f"drift={r['readout_drift']:.4f}")

    contested_flag = " **CONTESTED**" if c["contested"] else ""
    print(f"  [Competition] winner={c['winner']}  "
          f"certainty={c['certainty']:.4f}  "
          f"margin={c['margin']:.4f}  "
          f"steps={c['trajectory_steps']}{contested_flag}")

    conv_status = "CONVERGED" if v["converged"] else "**DID NOT CONVERGE**"
    print(f"  [Convergence] {conv_status}  "
          f"termination={v['termination']}  "
          f"iterations={v['iterations']}  "
          f"final_drift={v['final_drift']:.6f}")


def main() -> int:
    """Run V4 composition test across multiple scenarios."""
    print("V4 Composition Test: RecursiveFlow + AttractorCompetition + ReservoirState")
    print("=" * 70)
    print(f"Config: d={D}, N_prices={N_PRICES}, leak_rate={LEAK_RATE}, "
          f"attractors={NUM_ATTRACTORS}, scenarios={NUM_SCENARIOS}")
    print(f"Seed: {SEED} (deterministic)")

    rng = np.random.default_rng(SEED)

    # Define attractors (shared across scenarios)
    attractors, labels = define_attractors(rng, D)
    print(f"\nAttractors:")
    for i, (lab, att) in enumerate(zip(labels, attractors)):
        print(f"  [{i}] {lab}: {att}")

    # Run scenarios with different price trajectories
    results = []
    convergence_failures = 0
    contested_count = 0

    for scenario_id in range(NUM_SCENARIOS):
        prices = generate_synthetic_prices(rng, D, N_PRICES)
        result = run_full_pipeline(scenario_id, prices, attractors, labels)
        results.append(result)
        print_result(result)

        if not result["convergence"]["converged"]:
            convergence_failures += 1
        if result["competition"]["contested"]:
            contested_count += 1

    # ── Edge case: start state exactly at an attractor ────────
    print(f"\n{'='*60}")
    print("  Edge Case: state at attractor[0]")
    print(f"{'='*60}")
    edge_state = attractors[0].copy()
    comp_edge = run_competition_phase(edge_state, attractors, labels)
    print(f"  [Competition] winner={comp_edge['winner']}  "
          f"certainty={comp_edge['certainty']:.4f}  "
          f"steps={len(comp_edge['trajectory'])}")
    conv_edge = run_convergence_phase(edge_state, attractors[comp_edge["winner_idx"]])
    print(f"  [Convergence] converged={conv_edge['converged']}  "
          f"termination={conv_edge['termination']}  "
          f"iterations={conv_edge['iterations']}  "
          f"drift={float(np.linalg.norm(conv_edge['final_state'] - attractors[comp_edge['winner_idx']])):.6f}")

    # ── Edge case: state equidistant from two attractors ──────
    print(f"\n{'='*60}")
    print("  Edge Case: state equidistant from attractor[0] and attractor[1]")
    print(f"{'='*60}")
    midpoint = (attractors[0] + attractors[1]) / 2.0
    comp_mid = run_competition_phase(midpoint, attractors, labels)
    contested_tag = " **CONTESTED**" if comp_mid["contested"] else ""
    print(f"  [Competition] winner={comp_mid['winner']}  "
          f"certainty={comp_mid['certainty']:.4f}  "
          f"margin={comp_mid['margin']:.4f}{contested_tag}")
    conv_mid = run_convergence_phase(
        comp_mid["trajectory"][-1],
        attractors[comp_mid["winner_idx"]],
    )
    print(f"  [Convergence] converged={conv_mid['converged']}  "
          f"termination={conv_mid['termination']}  "
          f"iterations={conv_mid['iterations']}")

    # ── Edge case: very large initial state ───────────────────
    print(f"\n{'='*60}")
    print("  Edge Case: large-magnitude initial state (norm ~100)")
    print(f"{'='*60}")
    large_state = rng.standard_normal(D).astype(np.float32) * 100
    comp_large = run_competition_phase(large_state, attractors, labels)
    print(f"  [Competition] winner={comp_large['winner']}  "
          f"certainty={comp_large['certainty']:.4f}  "
          f"steps={len(comp_large['trajectory'])}")
    conv_large = run_convergence_phase(
        comp_large["trajectory"][-1],
        attractors[comp_large["winner_idx"]],
    )
    conv_large_drift = float(np.linalg.norm(
        conv_large["final_state"] - attractors[comp_large["winner_idx"]]
    ))
    conv_tag = "CONVERGED" if conv_large["converged"] else "**DID NOT CONVERGE**"
    print(f"  [Convergence] {conv_tag}  "
          f"termination={conv_large['termination']}  "
          f"iterations={conv_large['iterations']}  "
          f"drift={conv_large_drift:.6f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Scenarios run:          {NUM_SCENARIOS}")
    print(f"  Convergence failures:   {convergence_failures} / {NUM_SCENARIOS}")
    print(f"  Contested competitions: {contested_count} / {NUM_SCENARIOS}")

    winners = [r["competition"]["winner"] for r in results]
    from collections import Counter
    winner_counts = Counter(winners)
    print(f"  Basin distribution:     {dict(winner_counts)}")

    avg_iterations = np.mean([r["convergence"]["iterations"] for r in results])
    avg_certainty = np.mean([r["competition"]["certainty"] for r in results])
    print(f"  Avg iterations:         {avg_iterations:.1f}")
    print(f"  Avg certainty:          {avg_certainty:.4f}")

    all_converged = all(r["convergence"]["converged"] for r in results)
    print(f"  All converged:          {all_converged}")

    # Failure threshold check per spec
    failure_rate = convergence_failures / NUM_SCENARIOS
    if failure_rate > 0.10:
        print(f"\n  ** STOP: Non-convergence rate {failure_rate:.0%} exceeds 10% threshold. **")
        print(f"  ** Report to Jason before proceeding. **")
        return 1

    print(f"\n  V4 Composition Test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
