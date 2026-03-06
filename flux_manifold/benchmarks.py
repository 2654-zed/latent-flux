"""Benchmarks for FluxManifold – Tier A (micro) and Tier B (simulation)."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from flux_manifold.core import flux_flow_traced
from flux_manifold.flows import normalize_flow, sin_flow, damped_flow
from flux_manifold.baselines import random_walk, gradient_descent, static_baseline


# ---------------------------------------------------------------------------
# Tier A – 2D toy attractor micro-benchmark
# ---------------------------------------------------------------------------

def tier_a(
    n_runs: int = 100,
    seed: int = 42,
    epsilon: float = 0.2,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Tier A: 2D toy attractor.  s0=[0,0], q=[1,1], f=normalize.

    Acceptance: mean steps < 10, variance < 2.
    """
    rng = np.random.default_rng(seed)
    steps_list: list[int] = []
    converged_count = 0

    for _ in range(n_runs):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        result = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        steps_list.append(result["steps"])
        if result["converged"]:
            converged_count += 1

    steps_arr = np.array(steps_list)
    return {
        "tier": "A",
        "n_runs": n_runs,
        "mean_steps": float(steps_arr.mean()),
        "var_steps": float(steps_arr.var()),
        "std_steps": float(steps_arr.std()),
        "converged_pct": converged_count / n_runs * 100,
        "pass_mean": bool(steps_arr.mean() < 10),
        "pass_var": bool(steps_arr.var() < 2),
        "raw_steps": steps_list,
    }


# ---------------------------------------------------------------------------
# Tier B – 128D random manifolds simulation
# ---------------------------------------------------------------------------

def tier_b(
    d: int = 128,
    n_runs: int = 50,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Tier B: 128D random manifolds, goal=origin.

    Compares FluxManifold (normalize_flow) vs baselines.
    Acceptance: t-test p<0.05 vs random walk, Cohen's d > 0.5.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)

    flux_steps: list[int] = []
    rw_steps: list[int] = []
    gd_steps: list[int] = []

    flux_drifts: list[float] = []
    rw_drifts: list[float] = []
    gd_drifts: list[float] = []

    for i in range(n_runs):
        s0 = rng.standard_normal(d).astype(np.float32)
        s0 = np.clip(s0, -1, 1)

        # FluxManifold
        fm = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        flux_steps.append(fm["steps"])
        flux_drifts.append(fm["drift_trace"][-1] if fm["drift_trace"] else float("inf"))

        # Random walk
        rw_rng = np.random.default_rng(seed + i)
        rw = random_walk(s0, q, epsilon=epsilon, tol=tol, max_steps=max_steps, rng=rw_rng)
        rw_steps.append(rw["steps"])
        rw_drifts.append(rw["drift_trace"][-1] if rw["drift_trace"] else float("inf"))

        # Gradient descent
        gd = gradient_descent(s0, q, epsilon=epsilon, tol=tol, max_steps=max_steps)
        gd_steps.append(gd["steps"])
        gd_drifts.append(gd["drift_trace"][-1] if gd["drift_trace"] else float("inf"))

    flux_steps_arr = np.array(flux_steps, dtype=float)
    rw_steps_arr = np.array(rw_steps, dtype=float)
    gd_steps_arr = np.array(gd_steps, dtype=float)

    # t-test: flux vs random walk (steps)
    t_rw, p_rw = sp_stats.ttest_ind(flux_steps_arr, rw_steps_arr)
    # Cohen's d: flux vs random walk
    pooled_std = np.sqrt((flux_steps_arr.var() + rw_steps_arr.var()) / 2)
    cohens_d_rw = float((rw_steps_arr.mean() - flux_steps_arr.mean()) / pooled_std) if pooled_std > 0 else 0.0

    # t-test: flux vs gradient descent (steps)
    t_gd, p_gd = sp_stats.ttest_ind(flux_steps_arr, gd_steps_arr)

    return {
        "tier": "B",
        "d": d,
        "n_runs": n_runs,
        "flux_mean_steps": float(flux_steps_arr.mean()),
        "rw_mean_steps": float(rw_steps_arr.mean()),
        "gd_mean_steps": float(gd_steps_arr.mean()),
        "flux_mean_drift": float(np.mean(flux_drifts)),
        "rw_mean_drift": float(np.mean(rw_drifts)),
        "gd_mean_drift": float(np.mean(gd_drifts)),
        "t_stat_vs_rw": float(t_rw),
        "p_value_vs_rw": float(p_rw),
        "cohens_d_vs_rw": cohens_d_rw,
        "t_stat_vs_gd": float(t_gd),
        "p_value_vs_gd": float(p_gd),
        "pass_p_rw": bool(p_rw < 0.05),
        "pass_cohens_d": bool(cohens_d_rw > 0.5),
    }


# ---------------------------------------------------------------------------
# Tier C – ARC toy grid solver (placeholder)
# ---------------------------------------------------------------------------

def tier_c_placeholder(
    n_puzzles: int = 5,
    grid_size: int = 4,
    d: int = 16,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 500,
) -> dict:
    """Tier C stub: embed small grids as flat vectors, flow toward a known pattern.

    This is a simplified proof-of-concept – real ARC integration would need
    proper grid-to-embedding and evaluation.
    """
    rng = np.random.default_rng(seed)
    results: list[dict] = []

    for i in range(n_puzzles):
        # Target pattern: identity-like pattern in flattened grid
        q = np.eye(grid_size, dtype=np.float32).flatten()[:d]
        if q.shape[0] < d:
            q = np.pad(q, (0, d - q.shape[0]))
        q = np.clip(q, -1, 1).astype(np.float32)

        # Noisy start
        s0 = (q + rng.standard_normal(d).astype(np.float32) * 0.5).astype(np.float32)
        s0 = np.clip(s0, -1, 1)

        fm = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        results.append({
            "puzzle": i,
            "steps": fm["steps"],
            "converged": fm["converged"],
            "final_drift": fm["drift_trace"][-1] if fm["drift_trace"] else float("inf"),
        })

    return {
        "tier": "C",
        "n_puzzles": n_puzzles,
        "results": results,
        "converged_pct": sum(r["converged"] for r in results) / n_puzzles * 100,
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(results: dict, path: str | Path) -> None:
    """Save flat benchmark results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in results.items() if not isinstance(v, (list, dict))}
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
