"""Core FluxManifold engine – simulates continuous flow toward attractors.

Supports both single-state flow (d,) and batch flow (N, d) for O(1)
parallel evaluation across a Superposition Tensor.

Every flow invocation accepts an optional ConvergenceContract declaring
its convergence tier (1=Provable, 2=Empirical, 3=Non-convergent).
If no contract is provided, Tier 2 (EMPIRICAL) is assumed.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from flux_manifold.convergence import (
    ConvergenceContract, ConvergenceResult, ConvergenceTier,
    TIER_2_DEFAULT,
)

FlowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _validate_inputs(
    s0: np.ndarray, q: np.ndarray, epsilon: float, tol: float, max_steps: int
) -> None:
    if s0.ndim == 1:
        if s0.shape != q.shape:
            raise ValueError(f"s0 shape {s0.shape} != q shape {q.shape}")
        d = s0.shape[0]
    elif s0.ndim == 2:
        if s0.shape[1] != q.shape[0]:
            raise ValueError(f"s0 dim {s0.shape[1]} != q dim {q.shape[0]}")
        d = s0.shape[1]
    else:
        raise ValueError(f"s0 must be 1-D or 2-D, got ndim={s0.ndim}")
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon}")
    if tol <= 0:
        raise ValueError(f"tol must be >0, got {tol}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >=1, got {max_steps}")
    if d > 1024:
        raise ValueError(f"Dimension {d} exceeds safety cap of 1024")


def flux_flow(
    s0: np.ndarray,
    q: np.ndarray,
    f: FlowFn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    contract: ConvergenceContract | None = None,
) -> np.ndarray:
    """Run FluxManifold flow from s0 toward attractor q.

    Args:
        contract: Optional convergence contract. Defaults to Tier 2 (EMPIRICAL).
                  Tier 1 requires a Lipschitz bound and will raise if flow
                  fails to converge. Tier 3 expects non-convergence.

    Returns the converged state.
    """
    _validate_inputs(s0, q, epsilon, tol, max_steps)
    if contract is None:
        contract = TIER_2_DEFAULT
    s = s0.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)

    converged = False
    steps_used = 0
    for steps_used in range(1, max_steps + 1):
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
            converged = True
            break

    # Check convergence contract
    result = ConvergenceResult(
        contract=contract,
        converged=converged,
        steps_used=steps_used,
        max_steps=max_steps,
        final_drift=float(np.linalg.norm(s - q)),
    )
    result.check()
    if result.failure_signal and contract.tier == ConvergenceTier.PROVABLE:
        raise RuntimeError(result.failure_signal)

    return s


def flux_flow_traced(
    s0: np.ndarray,
    q: np.ndarray,
    f: FlowFn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    contract: ConvergenceContract | None = None,
) -> dict:
    """Run FluxManifold flow and return full trace for analysis/logging.

    Args:
        contract: Optional convergence contract. Defaults to Tier 2.

    Returns dict with keys:
        converged_state, steps, converged (bool), drift_trace (list[float]),
        path (list of states), convergence_result (ConvergenceResult).
    """
    _validate_inputs(s0, q, epsilon, tol, max_steps)
    if contract is None:
        contract = TIER_2_DEFAULT
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

    conv_result = ConvergenceResult(
        contract=contract,
        converged=converged,
        steps_used=len(drift_trace),
        max_steps=max_steps,
        final_drift=drift_trace[-1] if drift_trace else float(np.linalg.norm(s - q)),
    )
    conv_result.check()

    return {
        "converged_state": s,
        "steps": len(drift_trace),
        "converged": converged,
        "drift_trace": drift_trace,
        "path": path,
        "convergence_result": conv_result,
    }


# ── Batch (N, d) flow – O(1) parallel evaluation ──────────────────

def flux_flow_batch(
    S: np.ndarray,
    q: np.ndarray,
    f: FlowFn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> np.ndarray:
    """Flow N states toward attractor q simultaneously using vectorized ops.

    Args:
        S: (N, d) matrix of initial states.
        q: (d,) attractor.
        f: Flow function that accepts both (d,) and (N, d) inputs.

    Returns:
        (N, d) converged states.
    """
    _validate_inputs(S, q, epsilon, tol, max_steps)
    S = S.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)
    N = S.shape[0]

    active = np.ones(N, dtype=bool)

    for _ in range(max_steps):
        n_active = active.sum()
        if n_active == 0:
            break

        Sa = S[active]
        delta = epsilon * f(Sa, q)

        # NaN/inf safety
        bad = np.any(np.isnan(delta) | np.isinf(delta), axis=1)
        delta[bad] = 0.0

        # Clip norms > 1
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        big = norms > 1.0
        delta = np.where(big, delta / np.maximum(norms, 1e-12), delta)

        # Prevent overshoot
        dists = np.linalg.norm(q - Sa, axis=1, keepdims=True)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        overshoot = (norms > 1e-12) & (norms > dists)
        delta = np.where(overshoot, delta * (dists / np.maximum(norms, 1e-12)), delta)

        S[active] += delta

        # Check convergence
        drifts = np.linalg.norm(S[active] - q, axis=1)
        converged_mask = drifts < tol
        active_idx = np.where(active)[0]
        active[active_idx[converged_mask]] = False

    return S


def flux_flow_traced_batch(
    S: np.ndarray,
    q: np.ndarray,
    f: FlowFn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Batch flow with full trace — vectorized equivalent of N × flux_flow_traced.

    Returns dict:
        converged_states: (N, d) final states
        steps: (N,) per-state step counts
        converged: (N,) per-state convergence flags
        total_steps: int, total iterations executed
        drift_traces: (max_iters, N) drift matrix (padded with NaN after convergence)
    """
    _validate_inputs(S, q, epsilon, tol, max_steps)
    S = S.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)
    N = S.shape[0]

    active = np.ones(N, dtype=bool)
    steps = np.zeros(N, dtype=np.int32)
    converged_flags = np.zeros(N, dtype=bool)
    drift_history: list[np.ndarray] = []

    for iteration in range(max_steps):
        n_active = active.sum()
        if n_active == 0:
            break

        Sa = S[active]
        delta = epsilon * f(Sa, q)

        # NaN/inf safety
        bad = np.any(np.isnan(delta) | np.isinf(delta), axis=1)
        delta[bad] = 0.0

        # Clip norms > 1
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        big = norms > 1.0
        delta = np.where(big, delta / np.maximum(norms, 1e-12), delta)

        # Prevent overshoot
        dists = np.linalg.norm(q - Sa, axis=1, keepdims=True)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        overshoot = (norms > 1e-12) & (norms > dists)
        delta = np.where(overshoot, delta * (dists / np.maximum(norms, 1e-12)), delta)

        S[active] += delta

        # Record drift for all states (NaN for already-converged)
        all_drifts = np.full(N, np.nan)
        all_drifts[active] = np.linalg.norm(S[active] - q, axis=1)
        drift_history.append(all_drifts)

        # Update step counts for active states
        steps[active] += 1

        # Check convergence
        drifts = all_drifts[active]
        converged_mask = drifts < tol
        active_idx = np.where(active)[0]
        newly_converged = active_idx[converged_mask]
        converged_flags[newly_converged] = True
        active[newly_converged] = False

    total_iters = len(drift_history)
    drift_matrix = np.array(drift_history) if drift_history else np.empty((0, N))

    return {
        "converged_states": S,
        "steps": steps,
        "converged": converged_flags,
        "total_steps": int(steps.sum()),
        "drift_traces": drift_matrix,  # (iters, N)
    }
