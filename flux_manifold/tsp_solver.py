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


# ── TSP-specific flow function ──────────────────────────────────────

def tsp_flow_fn(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Flow that pulls toward the nearest-neighbor attractor with smoothing."""
    diff = q - s
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    return diff / norm


# ── TSP-specific critique (Fold-Reference) ─────────────────────────

def make_crossing_critique(cities: np.ndarray) -> CritiqueFn:
    """Build a critique function that detects and fixes edge crossings."""

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

    def critique(state: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
        order = state_to_order(state)
        n = len(order)
        if cities.shape[1] != 2:
            return True, "skip (not 2D)", None

        # Find first crossing and fix via 2-opt swap
        for i in range(n):
            for j in range(i + 2, n):
                if j == (i + n - 1) % n:
                    continue
                a, b = int(order[i]), int(order[(i + 1) % n])
                c, d = int(order[j]), int(order[(j + 1) % n])
                if _segments_intersect(cities[a], cities[b], cities[c], cities[d]):
                    # 2-opt: reverse the segment between i+1 and j
                    new_order = order.copy()
                    new_order[i + 1:j + 1] = new_order[i + 1:j + 1][::-1]
                    corrected = order_to_state(new_order, n)
                    return False, f"crossing at edges {i}-{j}", corrected

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
        """Generate N initial tour candidates as state vectors."""
        candidates = []
        for _ in range(self.n_candidates):
            perm = rng.permutation(self.n_cities).astype(np.int32)
            candidates.append(order_to_state(perm, self.n_cities))
        return np.array(candidates, dtype=np.float32)

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

        # ── ∑_ψ + ⟼ Flow all candidates toward attractor ───────
        traces = superposition.flow_all(
            q, tsp_flow_fn,
            epsilon=self.epsilon, tol=self.tol, max_steps=self.max_steps,
        )

        total_steps = sum(t["steps"] for t in traces)
        converged_count = sum(1 for t in traces if t["converged"])

        # ── ◉ Fold-Reference: fix crossings ─────────────────────
        fold_corrections = 0
        for i in range(superposition.n):
            state = superposition.states[i]
            if self.squeeze is not None:
                state = self.squeeze.unsqueeze(state)
            corrected, was_corrected = self.fold_ref.check(state, step=i)
            if was_corrected:
                if self.squeeze is not None:
                    superposition.states[i] = self.squeeze.squeeze(corrected)
                else:
                    superposition.states[i] = corrected
                fold_corrections += 1

        # ── ≅ DriftEquivalence reweight ─────────────────────────
        superposition.reweight_by_drift(q)
        entropy = superposition.entropy()

        # ── Evaluate all tour lengths ───────────────────────────
        lengths: list[float] = []
        final_states = superposition.states.copy()
        if self.squeeze is not None:
            final_states = np.array([
                self.squeeze.unsqueeze(s) for s in final_states
            ])

        for i in range(superposition.n):
            order = state_to_order(final_states[i])
            length = tour_length(self.cities, order)
            lengths.append(length)

        best_candidate = int(np.argmin(lengths))
        best_length = lengths[best_candidate]
        best_order = state_to_order(final_states[best_candidate])

        # ── ↓! Commitment Sink ──────────────────────────────────
        best_state = final_states[best_candidate]
        best_drift = traces[best_candidate]["drift_trace"]
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
