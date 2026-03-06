"""DriftEquivalence (≅) – approximate equality with semantic tolerance.

Instead of exact convergence (drift < tol), DriftEquivalence accepts
"close enough" solutions. For optimization problems like TSP there is
no single correct answer — we accept any tour within tolerance of the
best known quality.

Notation:  result ≅ target | tolerance=0.05
"""

from __future__ import annotations

import numpy as np


class DriftEquivalence:
    """Approximate equality checker for latent states."""

    def __init__(self, tolerance: float = 0.05, metric: str = "l2"):
        """
        Args:
            tolerance: Maximum acceptable distance to consider equivalent.
            metric: Distance metric — "l2" (Euclidean) or "cosine".
        """
        if tolerance <= 0:
            raise ValueError(f"tolerance must be >0, got {tolerance}")
        if metric not in ("l2", "cosine"):
            raise ValueError(f"metric must be 'l2' or 'cosine', got '{metric}'")
        self.tolerance = tolerance
        self.metric = metric

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two states."""
        if self.metric == "l2":
            return float(np.linalg.norm(a - b))
        else:  # cosine
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                return 1.0
            return float(1.0 - np.dot(a, b) / (na * nb))

    def equivalent(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if a ≅ b within tolerance."""
        return self.distance(a, b) <= self.tolerance

    def quality(self, state: np.ndarray, target: np.ndarray) -> float:
        """Quality score: 1.0 = perfect match, 0.0 = at tolerance boundary, <0 = beyond."""
        d = self.distance(state, target)
        if self.tolerance == 0:
            return 1.0 if d == 0 else 0.0
        return max(0.0, 1.0 - d / self.tolerance)

    def best_equivalent(
        self, candidates: np.ndarray, target: np.ndarray
    ) -> tuple[int, float]:
        """Find the best candidate that is ≅ target.

        Returns (index, distance). Index is -1 if none are equivalent.
        """
        dists = np.array([self.distance(c, target) for c in candidates])
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= self.tolerance:
            return best_idx, float(dists[best_idx])
        return -1, float(dists[best_idx])
