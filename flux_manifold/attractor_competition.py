"""Attractor Competition (⊗) — geometric replacement for pattern matching.

Instead of discrete if/else branching:
    if distance(x, A) < distance(x, B): ...

Competition evolves the state under SIMULTANEOUS attraction to every
attractor plus geometric repulsion between attractors. The winner is
the basin that captures the state — a purely geometric outcome.

For each state x ∈ ℝ^d and attractors {a_k}, the dynamics are:

    pull_k  = flow(x, a_k)
    repel_k = -repulsion × Σ_{j≠k} (a_k - a_j) / ||a_k - a_j||²
    net     = Σ_k  (pull_k + repel_k) / K

The state flows until it commits to a single basin (distance to one
attractor dominates all others by a margin).

Symbol: ⊗ (Circled Times / Competition)
"""

from __future__ import annotations

import numpy as np


class AttractorCompetition:
    """Geometric attractor competition — states flow to their natural basin.

    Args:
        attractors: (K, d) array of attractor positions.
        labels: Length-K list of labels for each attractor.
        flow_fn: Flow function f(s, q) → delta, same signature as flows.py.
        epsilon: Step size (default 0.1).
        tol: Distance threshold to declare capture (default 1e-2).
        max_steps: Safety timeout (default 500).
        repulsion: Inter-attractor repulsion strength (default 0.05).
        seed: Random seed for any stochastic components (default 42).
    """

    def __init__(
        self,
        attractors: np.ndarray,
        labels: list[str],
        flow_fn,
        epsilon: float = 0.1,
        tol: float = 1e-2,
        max_steps: int = 500,
        repulsion: float = 0.05,
        seed: int = 42,
    ):
        self.attractors = np.asarray(attractors, dtype=np.float32)
        if self.attractors.ndim != 2:
            raise ValueError(f"attractors must be 2D (K, d), got shape {self.attractors.shape}")
        self.K, self.d = self.attractors.shape
        if len(labels) != self.K:
            raise ValueError(f"Expected {self.K} labels, got {len(labels)}")
        self.labels = list(labels)
        self.flow_fn = flow_fn
        self.epsilon = epsilon
        self.tol = tol
        self.max_steps = max_steps
        self.repulsion = repulsion
        self.seed = seed

        # Precompute inter-attractor repulsion directions (K, K, d)
        # repel[i, j] = direction pushing attractor i away from attractor j
        self._repulsion_field = self._compute_repulsion_field()

    def _compute_repulsion_field(self) -> np.ndarray:
        """Precompute repulsion vectors between all attractor pairs.

        Returns (K, d) net repulsion direction for each attractor.
        """
        net = np.zeros((self.K, self.d), dtype=np.float32)
        for k in range(self.K):
            for j in range(self.K):
                if k == j:
                    continue
                diff = self.attractors[k] - self.attractors[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq > 1e-12:
                    net[k] += diff / dist_sq
        return net

    def compete(self, state: np.ndarray) -> dict:
        """Compete a single state x ∈ ℝ^d against all attractors.

        Returns:
            winner: label of the winning attractor
            winner_idx: index of the winning attractor
            margin: distance gap between closest and second-closest attractor
            certainty: 1 - (dist_winner / dist_second), in [0, 1]
            contested: True if margin < tol (state in overlap zone)
            trajectory: list of (d,) states along the competition path
        """
        if state.shape != (self.d,):
            raise ValueError(f"Expected state shape ({self.d},), got {state.shape}")

        x = state.astype(np.float32, copy=True)
        trajectory = [x.copy()]

        for _ in range(self.max_steps):
            # Compute distances to all attractors
            dists = np.linalg.norm(self.attractors - x, axis=1)

            # Check capture: closest attractor within tolerance
            closest = int(np.argmin(dists))
            if dists[closest] < self.tol:
                break

            # Net force: sum of pulls toward each attractor + repulsion
            net_force = np.zeros(self.d, dtype=np.float32)
            for k in range(self.K):
                pull = self.flow_fn(x, self.attractors[k])
                repel = self.repulsion * self._repulsion_field[k]
                net_force += pull + repel
            net_force /= self.K

            # Step
            step = self.epsilon * net_force

            # NaN/inf safety
            if np.any(np.isnan(step) | np.isinf(step)):
                break

            # Clip step norm
            step_norm = np.linalg.norm(step)
            if step_norm > 1.0:
                step = step / step_norm

            x += step
            trajectory.append(x.copy())

        # Final distances and result
        dists = np.linalg.norm(self.attractors - x, axis=1)
        sorted_idx = np.argsort(dists)
        winner_idx = int(sorted_idx[0])
        dist_winner = float(dists[winner_idx])
        dist_second = float(dists[sorted_idx[1]]) if self.K > 1 else float('inf')
        margin = dist_second - dist_winner

        certainty = 1.0 - (dist_winner / max(dist_second, 1e-12))
        certainty = float(np.clip(certainty, 0.0, 1.0))

        return {
            "winner": self.labels[winner_idx],
            "winner_idx": winner_idx,
            "margin": float(margin),
            "certainty": certainty,
            "contested": margin < self.tol,
            "trajectory": trajectory,
        }

    def compete_batch(self, states: np.ndarray) -> list[dict]:
        """Compete a batch of states against all attractors.

        Args:
            states: (N, d) array of states.

        Returns:
            List of N result dicts (same format as compete()).
        """
        if states.ndim != 2 or states.shape[1] != self.d:
            raise ValueError(f"Expected (N, {self.d}) states, got {states.shape}")

        return [self.compete(states[i]) for i in range(states.shape[0])]

    def summary(self, results: list[dict]) -> dict:
        """Summarize batch competition results.

        Returns dict with counts per label, mean certainty, contested count.
        """
        counts = {label: 0 for label in self.labels}
        certainties = []
        contested = 0
        for r in results:
            counts[r["winner"]] += 1
            certainties.append(r["certainty"])
            if r["contested"]:
                contested += 1

        return {
            "counts": counts,
            "mean_certainty": float(np.mean(certainties)) if certainties else 0.0,
            "contested": contested,
            "total": len(results),
        }
