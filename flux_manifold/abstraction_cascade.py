"""Abstraction Cascade (⇑) – hierarchical view of converged states.

After flow convergence, build layered abstractions via dimensionality
reduction (PCA). Each level captures coarser structure:
  Level 0 = full state
  Level 1 = top-k principal components
  Level 2 = further reduced
  ...

Notation:  ⇑ state levels=3
"""

from __future__ import annotations

import numpy as np


class AbstractionCascade:
    """Build hierarchical abstractions of latent states via PCA."""

    def __init__(self, levels: int = 3, min_dim: int = 2):
        """
        Args:
            levels: Number of abstraction levels to produce.
            min_dim: Minimum dimensionality at the coarsest level.
        """
        if levels < 1:
            raise ValueError(f"levels must be >=1, got {levels}")
        self.levels = levels
        self.min_dim = min_dim

    def cascade(self, states: np.ndarray) -> list[dict]:
        """Build abstraction cascade from a set of states (N, d).

        Returns a list of dicts per level:
            {level, dim, states, components, explained_variance_ratio}
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)

        n, d = states.shape
        result: list[dict] = []

        # Level 0: full resolution
        result.append({
            "level": 0,
            "dim": d,
            "states": states.copy(),
            "components": None,
            "explained_variance_ratio": None,
        })

        current = states.copy()
        remaining = d

        for level in range(1, self.levels):
            target_dim = max(self.min_dim, remaining // 2)
            if target_dim >= remaining:
                # Can't reduce further
                result.append({
                    "level": level,
                    "dim": remaining,
                    "states": current.copy(),
                    "components": None,
                    "explained_variance_ratio": None,
                })
                continue

            reduced, components, var_ratio = self._pca_reduce(current, target_dim)
            result.append({
                "level": level,
                "dim": target_dim,
                "states": reduced,
                "components": components,
                "explained_variance_ratio": var_ratio,
            })
            current = reduced
            remaining = target_dim

        return result

    def cascade_single(self, state: np.ndarray) -> list[dict]:
        """Cascade a single state vector — wraps it as (1, d) and cascades.

        Since PCA on a single point is degenerate, this uses truncation
        instead (keeps first k dimensions per level).
        """
        if state.ndim != 1:
            raise ValueError(f"Expected 1-D state, got ndim={state.ndim}")
        d = state.shape[0]
        result: list[dict] = []

        result.append({"level": 0, "dim": d, "state": state.copy()})

        current = state.copy()
        remaining = d

        for level in range(1, self.levels):
            target_dim = max(self.min_dim, remaining // 2)
            if target_dim >= remaining:
                result.append({"level": level, "dim": remaining, "state": current.copy()})
                continue
            # Simple truncation for single vectors
            current = current[:target_dim].copy()
            remaining = target_dim
            result.append({"level": level, "dim": target_dim, "state": current})

        return result

    @staticmethod
    def _pca_reduce(
        X: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce X (N, d) to (N, k) via PCA. Returns (reduced, components, variance_ratio)."""
        mean = X.mean(axis=0)
        centered = X - mean
        n = centered.shape[0]

        if n < 2:
            # Can't do PCA with 1 sample — truncate
            return X[:, :k].copy(), np.eye(k, X.shape[1], dtype=np.float32), np.ones(k, dtype=np.float32) / k

        # Covariance and eigen decomposition
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Top k
        components = eigenvectors[:, :k].T  # (k, d)
        reduced = centered @ components.T   # (N, k)

        total_var = eigenvalues.sum()
        if total_var > 0:
            var_ratio = eigenvalues[:k] / total_var
        else:
            var_ratio = np.ones(k, dtype=np.float32) / k

        return (
            reduced.astype(np.float32),
            components.astype(np.float32),
            var_ratio.astype(np.float32),
        )
