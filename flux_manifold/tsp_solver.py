"""Latent Flux TSP Solver – solves Travelling Salesman via all 7 primitives.

Encodes TSP as continuous flow in latent space:
  - Cities → points in ℝ^2 (or ℝ^k)
  - Tour → flattened permutation vector (city visit order as positions)
  - Attractor q → embedding of "minimal tour" (nearest-neighbor heuristic seed)
  - Flow → continuous relaxation toward shorter tours
  - Superposition → multiple initial tour candidates
  - Fold-Reference → detect and uncross edge crossings
  - Commitment → lock in when tour quality stabilizes

This is a proof-of-concept demonstrating all primitives working together,
not a competitive TSP solver.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from flux_manifold.core import FlowFn, flux_flow_traced
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.drift_equivalence import DriftEquivalence
from flux_manifold.commitment_sink import CommitmentSink
from flux_manifold.abstraction_cascade import AbstractionCascade
from flux_manifold.fold_reference import FoldReference, CritiqueFn
from flux_manifold.dimensional_squeeze import DimensionalSqueeze
from flux_manifold.flows import normalize_flow


# ── TSP Encoding ────────────────────────────────────────────────────

def tour_length(cities: np.ndarray, order: np.ndarray) -> float:
    """Total tour distance for a given visit order.

    Args:
        cities: (n, 2) city coordinates.
        order: (n,) integer permutation.
    """
    n = len(order)
    total = 0.0
    for i in range(n):
        a, b = int(order[i]), int(order[(i + 1) % n])
        total += float(np.linalg.norm(cities[a] - cities[b]))
    return total


def order_to_state(order: np.ndarray, n_cities: int) -> np.ndarray:
    """Encode a tour permutation as a continuous state vector.

    Maps each position to a value in [0, 1] based on rank.
    """
    state = np.zeros(n_cities, dtype=np.float32)
    for rank, city in enumerate(order):
        state[int(city)] = rank / max(n_cities - 1, 1)
    return state


def state_to_order(state: np.ndarray) -> np.ndarray:
    """Decode a continuous state vector back to a tour permutation."""
    return np.argsort(state).astype(np.int32)


def nearest_neighbor_tour(cities: np.ndarray, start: int = 0) -> np.ndarray:
    """Greedy nearest-neighbor heuristic for TSP."""
    n = len(cities)
    visited = {start}
    order = [start]
    current = start
    for _ in range(n - 1):
        dists = np.linalg.norm(cities - cities[current], axis=1)
        dists[list(visited)] = np.inf
        nxt = int(np.argmin(dists))
        visited.add(nxt)
        order.append(nxt)
        current = nxt
    return np.array(order, dtype=np.int32)


# ── TSP-specific flow function with geometric crossing repulsion ──

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects p3-p4 (2D only)."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    return 0 < t < 1 and 0 < u < 1


def _crossing_repulsion(state: np.ndarray, cities: np.ndarray,
                         strength: float = 0.3) -> np.ndarray:
    """Compute repulsive gradient that pushes crossing edges apart in latent space.

    Instead of a discrete 2-opt swap, we compute the geometric direction toward
    the uncrossed configuration and return a continuous gradient. The topology
    untangles dynamically over multiple flow steps.
    """
    order = state_to_order(state)
    n = len(order)
    gradient = np.zeros_like(state)

    if cities.shape[1] != 2:
        return gradient

    for i in range(n):
        for j in range(i + 2, n):
            if j == (i + n - 1) % n:
                continue
            a, b = int(order[i]), int(order[(i + 1) % n])
            c, d = int(order[j]), int(order[(j + 1) % n])
            if _segments_intersect(cities[a], cities[b], cities[c], cities[d]):
                # Compute geometric target: the uncrossed configuration
                uncrossed = order.copy()
                uncrossed[i + 1:j + 1] = uncrossed[i + 1:j + 1][::-1]
                target = order_to_state(uncrossed, n)
                # Continuous gradient toward uncrossed geometry
                gradient += strength * (target - state)

    return gradient


def _crossing_repulsion_batch(states: np.ndarray, cities: np.ndarray,
                              strength: float = 0.3) -> np.ndarray:
    """Compute crossing repulsion gradients for an (N, d) batch of states.

    Each state is a continuous tour encoding. The gradient pushes crossing
    edges apart via geometric repulsion. This is inherently per-tour
    (each tour has its own crossing structure), but we avoid Python-level
    row iteration by using numpy operations where possible.
    """
    N, d = states.shape
    gradients = np.zeros_like(states)
    if cities.shape[1] != 2:
        return gradients

    # Pre-decode all orders at once
    orders = np.argsort(states, axis=1).astype(np.int32)  # (N, d)
    n = d

    for k in range(N):
        order = orders[k]
        gradient = np.zeros(d, dtype=states.dtype)
        for i in range(n):
            for j in range(i + 2, n):
                if j == (i + n - 1) % n:
                    continue
                a, b = int(order[i]), int(order[(i + 1) % n])
                c, dd = int(order[j]), int(order[(j + 1) % n])
                if _segments_intersect(cities[a], cities[b], cities[c], cities[dd]):
                    uncrossed = order.copy()
                    uncrossed[i + 1:j + 1] = uncrossed[i + 1:j + 1][::-1]
                    target = order_to_state(uncrossed, n)
                    gradient += strength * (target - states[k])
        gradients[k] = gradient

    return gradients


def make_tsp_crossing_flow(
    cities: np.ndarray,
    repulsion_strength: float = 0.3,
) -> FlowFn:
    """Create a flow function with built-in geometric crossing repulsion.

    The vector field combines:
      1. Attractor pull (normalized difference toward q)
      2. Crossing repulsion (continuous gradient that pushes crossing edges apart)

    Edge crossings are resolved dynamically during flow, not by discrete swaps.
    """
    def flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
        diff = q - s
        if s.ndim == 1:
            norm = np.linalg.norm(diff)
            base = diff / max(norm, 1e-12) if norm > 1e-12 else np.zeros_like(diff)
            repulsion = _crossing_repulsion(s, cities, repulsion_strength)
            return base + repulsion
        else:
            # Batch: vectorized attractor pull + batch repulsion
            norms = np.linalg.norm(diff, axis=1, keepdims=True)
            base = np.where(norms > 1e-12, diff / np.maximum(norms, 1e-12), 0.0)
            base += _crossing_repulsion_batch(s, cities, repulsion_strength)
            return base

    return flow


def tsp_flow_fn(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Basic TSP flow that pulls toward the nearest-neighbor attractor."""
    diff = q - s
    if s.ndim == 1:
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            return np.zeros_like(diff)
        return diff / norm
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    return np.where(norms > 1e-12, diff / np.maximum(norms, 1e-12), 0.0)


# ── TSP-specific critique (Fold-Reference) — geometric gradient ──

def make_crossing_critique(cities: np.ndarray, repulsion_strength: float = 0.3) -> CritiqueFn:
    """Build a critique that applies geometric repulsion for crossings.

    Instead of a discrete 2-opt swap, the critique computes a continuous
    repulsive gradient and applies it to the state, nudging the topology
    toward an uncrossed configuration.
    """
    def critique(state: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
        gradient = _crossing_repulsion(state, cities, repulsion_strength)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-8:
            corrected = state + gradient
            return False, f"crossing repulsion applied (‖∇‖={grad_norm:.4f})", corrected
        return True, "no crossings", None

    return critique


# ── Full Latent Flux TSP Solver ─────────────────────────────────────

class LatentFluxTSP:
    """TSP solver using all 7 Latent Flux primitives."""

    def __init__(
        self,
        cities: np.ndarray,
        n_candidates: int = 20,
        epsilon: float = 0.15,
        tol: float = 1e-3,
        max_steps: int = 300,
        equiv_tolerance: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            cities: (n, 2) array of city coordinates.
            n_candidates: Number of initial tour candidates (superposition size).
            epsilon: Flow step size.
            tol: Convergence tolerance.
            max_steps: Max flow steps per candidate.
            equiv_tolerance: DriftEquivalence tolerance.
            seed: Random seed.
        """
        self.cities = cities.astype(np.float32)
        self.n_cities = len(cities)
        self.n_candidates = n_candidates
        self.epsilon = epsilon
        self.tol = tol
        self.max_steps = max_steps
        self.seed = seed

        # Build attractor from nearest-neighbor heuristic
        nn_tour = nearest_neighbor_tour(cities)
        self.nn_length = tour_length(cities, nn_tour)
        self.attractor = order_to_state(nn_tour, self.n_cities)

        # Use geometric crossing flow instead of plain attractor pull
        self.flow_fn = make_tsp_crossing_flow(cities, repulsion_strength=0.3)

        # Primitives
        self.equivalence = DriftEquivalence(tolerance=equiv_tolerance)
        self.commitment = CommitmentSink(
            entropy_threshold=1.0,
            drift_window=5,
            drift_threshold=0.005,
        )
        self.cascade = AbstractionCascade(levels=3, min_dim=2)
        self.fold_ref = FoldReference(
            critique_fn=make_crossing_critique(cities),
            interval=1,  # Check every candidate
            max_corrections=100,
        )

        # Squeeze only if n_cities is large
        self.squeeze: DimensionalSqueeze | None = None
        if self.n_cities > 50:
            self.squeeze = DimensionalSqueeze(target_dim=min(32, self.n_cities // 2))

    def _generate_candidates(self, rng: np.random.Generator) -> np.ndarray:
        """Generate N initial tour candidates as state vectors (vectorized)."""
        # Generate N random permutations and encode as continuous states
        n = self.n_cities
        candidates = np.zeros((self.n_candidates, n), dtype=np.float32)
        for k in range(self.n_candidates):
            perm = rng.permutation(n).astype(np.int32)
            # Vectorized encoding: rank / (n-1)
            candidates[k, perm] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
        return candidates

    def solve(self) -> dict:
        """Run the full Latent Flux TSP pipeline.

        Returns dict with:
            best_tour, best_length, nn_length, improvement_pct,
            total_steps, converged_count, fold_corrections,
            superposition_entropy, commit_reason, abstraction_levels,
            all_lengths
        """
        rng = np.random.default_rng(self.seed)

        # ── ∇↓ Dimensional Squeeze ──────────────────────────────
        q = self.attractor.copy()
        if self.squeeze is not None:
            candidates_raw = self._generate_candidates(rng)
            self.squeeze.fit(candidates_raw, seed=self.seed)
            q = self.squeeze.squeeze(q)
            candidates_squeezed = self.squeeze.squeeze(candidates_raw)
            superposition = SuperpositionTensor(candidates_squeezed)
        else:
            candidates_raw = self._generate_candidates(rng)
            superposition = SuperpositionTensor(candidates_raw)

        # ── ∑_ψ + ⟼ Flow all candidates toward attractor (batch) ──
        trace = superposition.flow_all(
            q, self.flow_fn,
            epsilon=self.epsilon, tol=self.tol, max_steps=self.max_steps,
        )

        total_steps = int(trace["total_steps"])
        converged_count = int(trace["converged"].sum())

        # ── ◉ Fold-Reference: fix crossings (batch) ────────────
        if self.squeeze is not None:
            unsqueezed = self.squeeze.unsqueeze(superposition.states)
            unsqueezed, fold_corrections = self.fold_ref.check_batch(unsqueezed)
            superposition.states = self.squeeze.squeeze(unsqueezed)
        else:
            superposition.states, fold_corrections = self.fold_ref.check_batch(
                superposition.states
            )

        # ── ≅ DriftEquivalence reweight ─────────────────────────
        superposition.reweight_by_drift(q)
        entropy = superposition.entropy()

        # ── Evaluate all tour lengths (vectorized decode) ───────
        final_states = superposition.states.copy()
        if self.squeeze is not None:
            final_states = self.squeeze.unsqueeze(final_states)

        # Batch decode: argsort each row to get orders, compute tour lengths
        all_orders = np.argsort(final_states, axis=1).astype(np.int32)  # (N, n_cities)
        lengths = [tour_length(self.cities, all_orders[i]) for i in range(superposition.n)]

        best_candidate = int(np.argmin(lengths))
        best_length = lengths[best_candidate]
        best_order = state_to_order(final_states[best_candidate])

        # ── ↓! Commitment Sink ──────────────────────────────────
        best_state = final_states[best_candidate]
        # Extract per-state drift trace for the best candidate
        drift_matrix = trace["drift_traces"]  # (iters, N)
        best_drift = []
        if drift_matrix.size > 0:
            col = drift_matrix[:, best_candidate]
            best_drift = col[~np.isnan(col)].tolist()
        committed = self.commitment.try_commit(
            superposition, q, best_drift
        )
        commit_reason = self.commitment.commit_reason
        if committed is None:
            self.commitment.commit(
                superposition.states[best_candidate], reason="best_tour"
            )
            commit_reason = "best_tour"

        # ── ⇑ Abstraction Cascade ──────────────────────────────
        levels = self.cascade.cascade_single(best_state)

        # ── Results ─────────────────────────────────────────────
        improvement = (1.0 - best_length / self.nn_length) * 100 if self.nn_length > 0 else 0.0

        return {
            "best_tour": best_order.tolist(),
            "best_length": best_length,
            "nn_length": self.nn_length,
            "improvement_pct": improvement,
            "total_steps": total_steps,
            "converged_count": converged_count,
            "n_candidates": superposition.n,
            "fold_corrections": fold_corrections,
            "superposition_entropy": entropy,
            "commit_reason": commit_reason,
            "abstraction_levels": len(levels),
            "all_lengths": lengths,
        }


# ── Convenience runner ──────────────────────────────────────────────

def solve_tsp(
    cities: np.ndarray,
    n_candidates: int = 20,
    seed: int = 42,
    **kwargs,
) -> dict:
    """One-liner TSP solve using Latent Flux."""
    solver = LatentFluxTSP(cities, n_candidates=n_candidates, seed=seed, **kwargs)
    return solver.solve()
