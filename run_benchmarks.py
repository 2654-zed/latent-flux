#!/usr/bin/env python3
"""Run all Latent Flux benchmarks, kill tests, and TSP demo."""

from __future__ import annotations

import json
import sys

import numpy as np

from flux_manifold.benchmarks import tier_a, tier_b, tier_c_placeholder, save_results_csv
from flux_manifold.kill_tests import run_all_kill_tests
from flux_manifold.monitor import export_benchmark
from flux_manifold.tsp_solver import solve_tsp, tour_length, nearest_neighbor_tour


def _banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main() -> int:
    all_pass = True

    # ── Tier A ──────────────────────────────────────────────────
    _banner("Tier A – 2D Toy Attractor (100 runs)")
    a = tier_a(n_runs=100, seed=42)
    print(f"  Mean steps : {a['mean_steps']:.2f}  (target <10)")
    print(f"  Variance   : {a['var_steps']:.4f}  (target <2)")
    print(f"  Converged  : {a['converged_pct']:.0f}%")
    print(f"  PASS mean  : {a['pass_mean']}")
    print(f"  PASS var   : {a['pass_var']}")
    if not (a["pass_mean"] and a["pass_var"]):
        all_pass = False
    save_results_csv(a, "results/tier_a.csv")
    export_benchmark(a, "results/tier_a.json")

    # ── Tier B ──────────────────────────────────────────────────
    _banner("Tier B – 128D Random Manifolds (50 runs)")
    b = tier_b(d=128, n_runs=50, seed=42)
    print(f"  Flux mean steps : {b['flux_mean_steps']:.2f}")
    print(f"  RW mean steps   : {b['rw_mean_steps']:.2f}")
    print(f"  GD mean steps   : {b['gd_mean_steps']:.2f}")
    print(f"  Flux mean drift : {b['flux_mean_drift']:.6f}")
    print(f"  RW mean drift   : {b['rw_mean_drift']:.6f}")
    print(f"  p-value vs RW   : {b['p_value_vs_rw']:.6f}  (target <0.05)")
    print(f"  Cohen's d vs RW : {b['cohens_d_vs_rw']:.4f}  (target >0.5)")
    print(f"  PASS p-value    : {b['pass_p_rw']}")
    print(f"  PASS Cohen's d  : {b['pass_cohens_d']}")
    if not (b["pass_p_rw"] and b["pass_cohens_d"]):
        all_pass = False
    save_results_csv(b, "results/tier_b.csv")
    export_benchmark(b, "results/tier_b.json")

    # ── Tier C (placeholder) ───────────────────────────────────
    _banner("Tier C – ARC Toy Grid (placeholder, 5 puzzles)")
    c = tier_c_placeholder(n_puzzles=5, grid_size=4, d=16, seed=42)
    print(f"  Converged pct : {c['converged_pct']:.0f}%")
    for r in c["results"]:
        print(f"    Puzzle {r['puzzle']}: steps={r['steps']}, converged={r['converged']}, drift={r['final_drift']:.6f}")
    export_benchmark(c, "results/tier_c.json")

    # ── Kill Tests ─────────────────────────────────────────────
    _banner("Kill Tests")
    kills = run_all_kill_tests()
    for k in kills:
        status = "PASS" if k["pass"] else "FAIL"
        details = {key: val for key, val in k.items() if key not in ("test", "pass")}
        print(f"  [{status}] {k['test']}: {details}")
        if not k["pass"]:
            all_pass = False
    export_benchmark({"kill_tests": kills}, "results/kill_tests.json")

    # ── Latent Flux TSP Demo ───────────────────────────────────
    for n_cities, label in [(6, "6-city"), (10, "10-city"), (15, "15-city")]:
        _banner(f"Latent Flux TSP – {label}")
        rng = np.random.default_rng(42)
        cities = rng.uniform(0, 100, (n_cities, 2)).astype(np.float32)
        result = solve_tsp(cities, n_candidates=20, seed=42)

        # Compare to random tours
        random_lengths = []
        for i in range(50):
            perm = np.random.default_rng(i).permutation(n_cities).astype(np.int32)
            random_lengths.append(tour_length(cities, perm))
        avg_random = float(np.mean(random_lengths))

        print(f"  Best tour length   : {result['best_length']:.2f}")
        print(f"  NN heuristic       : {result['nn_length']:.2f}")
        print(f"  Avg random tour    : {avg_random:.2f}")
        print(f"  Improvement vs NN  : {result['improvement_pct']:.1f}%")
        print(f"  Converged          : {result['converged_count']}/{result['n_candidates']}")
        print(f"  Fold corrections   : {result['fold_corrections']}")
        print(f"  Superpos. entropy  : {result['superposition_entropy']:.3f}")
        print(f"  Commit reason      : {result['commit_reason']}")
        print(f"  Abstraction levels : {result['abstraction_levels']}")
        print(f"  Best tour          : {result['best_tour']}")
        export_benchmark(result, f"results/tsp_{label}.json")

    # ── Summary ────────────────────────────────────────────────
    _banner("SUMMARY")
    if all_pass:
        print("  ✓ All acceptance criteria met.")
    else:
        print("  ✗ Some tests FAILED – review results above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
