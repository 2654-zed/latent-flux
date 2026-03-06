"""Core FluxManifold engine – simulates continuous flow toward attractors."""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

FlowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _validate_inputs(
    s0: np.ndarray, q: np.ndarray, epsilon: float, tol: float, max_steps: int
) -> None:
    if s0.shape != q.shape:
        raise ValueError(f"s0 shape {s0.shape} != q shape {q.shape}")
    if s0.ndim != 1:
        raise ValueError(f"s0 must be 1-D, got ndim={s0.ndim}")
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon}")
    if tol <= 0:
        raise ValueError(f"tol must be >0, got {tol}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >=1, got {max_steps}")
    d = s0.shape[0]
    if d > 1024:
        raise ValueError(f"Dimension {d} exceeds safety cap of 1024")


def flux_flow(
    s0: np.ndarray,
    q: np.ndarray,
    f: FlowFn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> np.ndarray:
    """Run FluxManifold flow from s0 toward attractor q.

    Returns the converged state.
    """
    _validate_inputs(s0, q, epsilon, tol, max_steps)
    s = s0.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)

    for _ in range(max_steps):
        delta = epsilon * f(s, q)
        # Guard overflow / NaN
        if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
            delta = np.zeros_like(delta)
        # Clip large deltas
        norm = np.linalg.norm(delta)
        if norm > 1.0:
            delta = delta / norm
        # Prevent overshoot: cap step at distance to target
        dist = np.linalg.norm(q - s)
        if norm > 0 and norm > dist:
            delta = delta * (dist / norm)
        s = s + delta
        if np.linalg.norm(s - q) < tol:
            return s
    return s


def flux_flow_traced(
    s0: np.ndarray,
    q: np.ndarray,
    f: FlowFn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Run FluxManifold flow and return full trace for analysis/logging.

    Returns dict with keys:
        converged_state, steps, converged (bool), drift_trace (list[float]),
        path (list of states).
    """
    _validate_inputs(s0, q, epsilon, tol, max_steps)
    s = s0.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)

    drift_trace: list[float] = []
    path: list[np.ndarray] = [s.copy()]
    converged = False

    for step in range(max_steps):
        delta = epsilon * f(s, q)
        if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
            delta = np.zeros_like(delta)
        norm = np.linalg.norm(delta)
        if norm > 1.0:
            delta = delta / norm
        # Prevent overshoot: cap step at distance to target
        dist = np.linalg.norm(q - s)
        if norm > 0 and norm > dist:
            delta = delta * (dist / norm)
        s = s + delta

        drift = float(np.linalg.norm(s - q))
        drift_trace.append(drift)
        path.append(s.copy())

        if drift < tol:
            converged = True
            break

    return {
        "converged_state": s,
        "steps": len(drift_trace),
        "converged": converged,
        "drift_trace": drift_trace,
        "path": path,
    }
