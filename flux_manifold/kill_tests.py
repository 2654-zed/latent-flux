"""Kill tests – fast disproval gates for FluxManifold."""

from __future__ import annotations

import time

import numpy as np

from flux_manifold.core import flux_flow_traced
from flux_manifold.flows import normalize_flow, repulsive_flow


def kill_test_convergence(
    n_runs: int = 100,
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    fail_pct: float = 20.0,
) -> dict:
    """Kill Test 1: If steps > max_steps in >20% runs → kill."""
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    not_converged = 0

    for _ in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        r = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        if not r["converged"]:
            not_converged += 1

    pct = not_converged / n_runs * 100
    return {"test": "convergence", "not_converged_pct": pct, "pass": pct <= fail_pct}


def kill_test_drift(
    n_runs: int = 50,
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Kill Test 2: If final drift > tol in tier B → kill."""
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    bad_drift = 0

    for _ in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        r = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        final = r["drift_trace"][-1] if r["drift_trace"] else float("inf")
        if final > tol:
            bad_drift += 1

    pct = bad_drift / n_runs * 100
    return {"test": "drift", "bad_drift_pct": pct, "pass": pct == 0}


def kill_test_vs_random(
    n_runs: int = 50,
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Kill Test 3: If random walk baseline beats flux → kill."""
    from flux_manifold.baselines import random_walk

    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    flux_wins = 0

    for i in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        fm = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        rw = random_walk(s0, q, epsilon=epsilon, tol=tol, max_steps=max_steps, rng=np.random.default_rng(seed + i))
        fm_drift = fm["drift_trace"][-1] if fm["drift_trace"] else float("inf")
        rw_drift = rw["drift_trace"][-1] if rw["drift_trace"] else float("inf")
        if fm_drift <= rw_drift:
            flux_wins += 1

    return {"test": "vs_random", "flux_win_pct": flux_wins / n_runs * 100, "pass": flux_wins / n_runs > 0.5}


def kill_test_scalability(
    d: int = 1024,
    n_runs: int = 10,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    time_limit_ms: float = 5.0,
) -> dict:
    """Kill Test 4: d=1024 – if time > 5ms per run → kill."""
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    times: list[float] = []

    for _ in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        t0 = time.perf_counter()
        flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    mean_ms = float(np.mean(times))
    return {"test": "scalability", "d": d, "mean_ms": mean_ms, "pass": mean_ms <= time_limit_ms}


def kill_test_adversarial(
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 100,
) -> dict:
    """Kill Test 5: Adversarial repulsive f – should diverge (expected behavior)."""
    rng = np.random.default_rng(seed)
    s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
    q = np.zeros(d, dtype=np.float32)

    r = flux_flow_traced(s0, q, repulsive_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
    final_drift = r["drift_trace"][-1] if r["drift_trace"] else float("inf")
    initial_drift = float(np.linalg.norm(s0 - q))

    # Repulsive flow should increase drift – verify system detects divergence
    diverged = final_drift > initial_drift
    return {
        "test": "adversarial",
        "initial_drift": initial_drift,
        "final_drift": final_drift,
        "diverged": diverged,
        "pass": diverged,  # We WANT it to diverge with repulsive f
    }


def run_all_kill_tests() -> list[dict]:
    """Run all kill tests and return results."""
    return [
        kill_test_convergence(),
        kill_test_drift(),
        kill_test_vs_random(),
        kill_test_scalability(),
        kill_test_adversarial(),
    ]
