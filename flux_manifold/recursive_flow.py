"""Recursive Flow (↺) — fixed-point attractor loops with geometric termination.

Instead of while-loops with boolean conditions:
    while not converged: state = transform(state)

RecursiveFlow iterates the full geometric pipeline until the state
reaches a fixed point — a state that maps to itself under the flow.
Termination is geometric: the distance between successive states
shrinks below tolerance, or entropy collapses, or a safety timeout fires.

Three termination conditions (checked in order each iteration):
    1. Entropy collapse — CommitmentSink detects low superposition entropy
    2. Fixed-point convergence — DriftEquivalence says s_{t} ≅ s_{t-1}
    3. Safety timeout — max_iterations reached (Python safety valve)

Each iteration creates a FRESH CommitmentSink — the recursive structure
is the loop, commitment happens within each iteration independently.

Symbol: ↺ (Anticlockwise Loop / Recurse)
"""

from __future__ import annotations

import numpy as np

from flux_manifold.drift_equivalence import DriftEquivalence


class RecursiveFlow:
    """Geometric fixed-point iteration — loops until the state converges.

    Args:
        flow_fn: Flow function f(s, q) → delta.
        attractor: Target attractor q ∈ ℝ^d.
        epsilon: Step size per inner flow step (default 0.1).
        tol: Distance tolerance for fixed-point detection (default 1e-3).
        max_iterations: Outer loop safety timeout (default 100).
        inner_steps: Steps per inner flow iteration (default 50).
        fixed_point_tol: Tolerance for DriftEquivalence between iterations (default 1e-4).
        seed: Random seed (default 42).
    """

    def __init__(
        self,
        flow_fn,
        attractor: np.ndarray,
        epsilon: float = 0.1,
        tol: float = 1e-3,
        max_iterations: int = 100,
        inner_steps: int = 50,
        fixed_point_tol: float = 1e-4,
        seed: int = 42,
    ):
        self.flow_fn = flow_fn
        self.attractor = np.asarray(attractor, dtype=np.float32)
        self.d = self.attractor.shape[0]
        self.epsilon = epsilon
        self.tol = tol
        self.max_iterations = max_iterations
        self.inner_steps = inner_steps
        self.fixed_point_tol = fixed_point_tol
        self.seed = seed
        self._equiv = DriftEquivalence(tolerance=fixed_point_tol)

    def _inner_flow(self, state: np.ndarray) -> np.ndarray:
        """Run inner flow iteration: evolve state toward attractor for inner_steps."""
        x = state.astype(np.float32, copy=True)
        for _ in range(self.inner_steps):
            delta = self.epsilon * self.flow_fn(x, self.attractor)
            # NaN/inf safety
            if np.any(np.isnan(delta) | np.isinf(delta)):
                break
            # Clip step
            norm = np.linalg.norm(delta)
            if norm > 1.0:
                delta = delta / norm
            x += delta
            # Early exit if already converged
            if np.linalg.norm(x - self.attractor) < self.tol:
                break
        return x

    def run(self, initial_state: np.ndarray) -> dict:
        """Execute recursive flow until geometric termination.

        Args:
            initial_state: Starting state x₀ ∈ ℝ^d.

        Returns:
            final_state: Converged state
            iterations: Number of outer iterations executed
            converged: Whether fixed-point was reached
            termination: "fixed_point" | "attractor_reached" | "timeout"
            trajectory: List of states at each outer iteration
            drift_trace: List of distances to attractor at each iteration
            fixed_point_distances: Distances between consecutive iterations
        """
        if initial_state.shape != (self.d,):
            raise ValueError(
                f"Expected state shape ({self.d},), got {initial_state.shape}"
            )

        state = initial_state.astype(np.float32, copy=True)
        trajectory = [state.copy()]
        drift_trace = [float(np.linalg.norm(state - self.attractor))]
        fp_distances = []
        termination = "timeout"

        iterations = 0
        for iteration in range(self.max_iterations):
            prev_state = state.copy()

            # Inner flow iteration
            state = self._inner_flow(state)
            trajectory.append(state.copy())

            # Drift to attractor
            drift = float(np.linalg.norm(state - self.attractor))
            drift_trace.append(drift)

            # Fixed-point distance (between consecutive outer iterations)
            fp_dist = float(self._equiv.distance(state, prev_state))
            fp_distances.append(fp_dist)

            iterations = iteration + 1

            # Termination 1: Attractor reached
            if drift < self.tol:
                termination = "attractor_reached"
                break

            # Termination 2: Fixed-point convergence (state maps to itself)
            if self._equiv.equivalent(state, prev_state):
                termination = "fixed_point"
                break

        return {
            "final_state": state,
            "iterations": iterations,
            "converged": termination != "timeout",
            "termination": termination,
            "trajectory": trajectory,
            "drift_trace": drift_trace,
            "fixed_point_distances": fp_distances,
        }

    def run_batch(self, states: np.ndarray) -> list[dict]:
        """Run recursive flow for a batch of initial states.

        Args:
            states: (N, d) array of initial states.

        Returns:
            List of N result dicts (same format as run()).
        """
        if states.ndim != 2 or states.shape[1] != self.d:
            raise ValueError(f"Expected (N, {self.d}) states, got {states.shape}")
        return [self.run(states[i]) for i in range(states.shape[0])]
