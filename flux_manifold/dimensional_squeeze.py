"""Dimensional Squeeze (∇↓) – compress high-d manifolds before flow.

Reduces dimensionality of the state space early in the pipeline to
make flow cheaper and more focused. Uses PCA or random projection.

Notation:  ∇↓ state target_dim=32
"""

from __future__ import annotations

import numpy as np


class DimensionalSqueeze:
    """Compress high-dimensional states to a lower-dimensional manifold."""

    def __init__(self, target_dim: int, method: str = "pca"):
        """
        Args:
            target_dim: Output dimensionality.
            method: "pca" or "random_projection".
        """
        if target_dim < 1:
            raise ValueError(f"target_dim must be >=1, got {target_dim}")
        if method not in ("pca", "random_projection"):
            raise ValueError(f"method must be 'pca' or 'random_projection', got '{method}'")
        self.target_dim = target_dim
        self.method = method
        self._projection: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._fitted = False

    def fit(self, data: np.ndarray, seed: int = 42) -> "DimensionalSqueeze":
        """Fit the squeeze transform on data (N, d)."""
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got ndim={data.ndim}")
        _, d = data.shape
        if self.target_dim >= d:
            # No squeeze needed
            self._projection = np.eye(d, dtype=np.float32)
            self._mean = np.zeros(d, dtype=np.float32)
            self._fitted = True
            return self

        if self.method == "pca":
            self._mean = data.mean(axis=0).astype(np.float32)
            centered = data - self._mean
            cov = np.cov(centered, rowvar=False)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            self._projection = eigenvectors[:, idx[:self.target_dim]].T.astype(np.float32)  # (target, d)
        else:
            rng = np.random.default_rng(seed)
            # Gaussian random projection (Johnson-Lindenstrauss)
            self._projection = (
                rng.standard_normal((self.target_dim, d)).astype(np.float32)
                / np.sqrt(self.target_dim)
            )
            self._mean = np.zeros(d, dtype=np.float32)

        self._fitted = True
        return self

    def squeeze(self, state: np.ndarray) -> np.ndarray:
        """Squeeze a state (d,) or batch (N, d) to target_dim."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before squeeze()")
        if state.ndim == 1:
            return ((state - self._mean) @ self._projection.T).astype(np.float32)
        return ((state - self._mean) @ self._projection.T).astype(np.float32)

    def unsqueeze(self, compressed: np.ndarray) -> np.ndarray:
        """Approximate reconstruction via pseudo-inverse (lossy)."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before unsqueeze()")
        pinv = np.linalg.pinv(self._projection)  # (d, target)
        if compressed.ndim == 1:
            return (compressed @ pinv.T + self._mean).astype(np.float32)
        return (compressed @ pinv.T + self._mean).astype(np.float32)

    @property
    def compression_ratio(self) -> float | None:
        """Ratio of original dim to target dim."""
        if self._projection is None:
            return None
        return self._projection.shape[1] / self._projection.shape[0]
