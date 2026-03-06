"""Superposition Tensor (∑_ψ) – parallel exploration of multiple states.

Represents a weighted superposition of latent states. Instead of flowing
a single s0 → q, we flow N candidate states simultaneously and aggregate.
This enables parallel exploration of solution space (e.g., multiple tour
permutations in TSP).

Notation:  ∑_ψ [s1, s2, ..., sN] ⟼ q  →  collapse to best
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from flux_manifold.core import FlowFn, flux_flow_traced, flux_flow_traced_batch


class SuperpositionTensor:
    """A weighted superposition of N latent states in ℝ^d."""

    def __init__(self, states: np.ndarray, weights: np.ndarray | None = None):
        """
        Args:
            states: (N, d) array of candidate states.
            weights: (N,) probability weights (uniform if None).
        """
        if states.ndim != 2:
            raise ValueError(f"states must be 2-D (N, d), got ndim={states.ndim}")
        self.states = states.astype(np.float32, copy=True)
        self.n, self.d = self.states.shape
        if weights is None:
            self.weights = np.ones(self.n, dtype=np.float32) / self.n
        else:
            w = np.asarray(weights, dtype=np.float32)
            if w.shape != (self.n,):
                raise ValueError(f"weights shape {w.shape} != ({self.n},)")
            self.weights = w / w.sum()  # normalize
        # Optional coupled reservoir (⧖)
        self._reservoir = None  # SuperpositionReservoir or None

    def attach_reservoir(self, reservoir) -> None:
        """Couple a SuperpositionReservoir to this tensor.

        Once attached, reweight_by_drift and prune will keep
        reservoir histories coupled with candidate states.
        """
        self._reservoir = reservoir

    @classmethod
    def from_random(
        cls, n: int, d: int, seed: int = 42, scale: float = 1.0
    ) -> "SuperpositionTensor":
        """Create N random states in [-scale, scale]^d."""
        rng = np.random.default_rng(seed)
        states = np.clip(
            rng.standard_normal((n, d)).astype(np.float32) * scale, -1, 1
        )
        return cls(states)

    def mean_state(self) -> np.ndarray:
        """Weighted centroid of the superposition."""
        return np.average(self.states, axis=0, weights=self.weights)

    def entropy(self) -> float:
        """Shannon entropy of weights (log2). High = spread, low = collapsed."""
        w = self.weights[self.weights > 0]
        return float(-np.sum(w * np.log2(w)))

    def flow_all(
        self,
        q: np.ndarray,
        f: FlowFn,
        epsilon: float = 0.1,
        tol: float = 1e-3,
        max_steps: int = 1000,
    ) -> dict:
        """Flow every state toward q simultaneously using vectorized batch ops.

        Returns the batch trace dict from flux_flow_traced_batch:
            converged_states, steps, converged, total_steps, drift_traces
        """
        trace = flux_flow_traced_batch(
            self.states, q, f,
            epsilon=epsilon, tol=tol, max_steps=max_steps,
        )
        self.states = trace["converged_states"]
        return trace

    def reweight_by_drift(self, q: np.ndarray) -> None:
        """Reweight states: closer to q gets higher weight (softmax of -dist).

        If a SuperpositionReservoir is attached, reservoir histories are
        reordered to stay coupled with their candidate states.
        """
        dists = np.linalg.norm(self.states - q, axis=1)
        # Softmax of negative distances → closer = higher weight
        logits = -dists
        logits -= logits.max()  # numerical stability
        w = np.exp(logits)
        self.weights = (w / w.sum()).astype(np.float32)
        # Couple reservoir: reorder by descending weight
        if self._reservoir is not None:
            order = np.argsort(self.weights)[::-1]
            self._reservoir.reorder(order)

    def collapse_to_best(self, q: np.ndarray) -> np.ndarray:
        """Collapse superposition: return the state closest to q."""
        dists = np.linalg.norm(self.states - q, axis=1)
        best = int(np.argmin(dists))
        return self.states[best].copy()

    def collapse_to_mean(self) -> np.ndarray:
        """Collapse to weighted mean (centroid)."""
        return self.mean_state()

    def prune(self, keep: int) -> None:
        """Keep only the top-k states by weight.

        If a SuperpositionReservoir is attached, pruned reservoir
        histories are discarded alongside their candidate states.
        """
        if keep >= self.n:
            return
        idx = np.argsort(self.weights)[::-1][:keep]
        self.states = self.states[idx]
        self.weights = self.weights[idx]
        self.weights /= self.weights.sum()
        self.n = keep
        if self._reservoir is not None:
            self._reservoir.prune(idx)
