"""Baseline methods for fair comparison against FluxManifold."""

from __future__ import annotations

import numpy as np


def random_walk(
    s0: np.ndarray,
    q: np.ndarray,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Random walk baseline: delta is a random unit vector each step."""
    rng = rng or np.random.default_rng(42)
    s = s0.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)
    d = s.shape[0]
    drift_trace: list[float] = []

    for step in range(max_steps):
        direction = rng.standard_normal(d).astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        s = s + epsilon * direction

        drift = float(np.linalg.norm(s - q))
        drift_trace.append(drift)
        if drift < tol:
            return {"converged_state": s, "steps": step + 1, "converged": True, "drift_trace": drift_trace}

    return {"converged_state": s, "steps": max_steps, "converged": False, "drift_trace": drift_trace}


def gradient_descent(
    s0: np.ndarray,
    q: np.ndarray,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Gradient descent baseline: delta = q - s (unnormalized)."""
    s = s0.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)
    drift_trace: list[float] = []

    for step in range(max_steps):
        delta = epsilon * (q - s)
        s = s + delta
        drift = float(np.linalg.norm(s - q))
        drift_trace.append(drift)
        if drift < tol:
            return {"converged_state": s, "steps": step + 1, "converged": True, "drift_trace": drift_trace}

    return {"converged_state": s, "steps": max_steps, "converged": False, "drift_trace": drift_trace}


def static_baseline(
    s0: np.ndarray,
    q: np.ndarray,
) -> dict:
    """No-flow baseline: returns s0 unchanged."""
    drift = float(np.linalg.norm(s0 - q))
    return {"converged_state": s0.copy(), "steps": 0, "converged": False, "drift_trace": [drift]}
