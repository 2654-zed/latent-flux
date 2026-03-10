"""Kalman Reservoir — UKF-based state estimator replacing EMA in ReservoirState.

Uses an Unscented Kalman Filter to estimate the posterior distribution of
log-price, drift, and volatility for each edge dimension. Provides both
posterior mean (like ReservoirState.step()) and posterior covariance (new).

State vector per edge dimension: [log_price, drift, log_volatility]
Observation: AMM spot log_price (noisy, averaged over 1-hour window)

Adaptive gain: when innovation > 3σ, inflates process noise Q temporarily
to avoid rejecting genuine large moves as outliers.

Pure numpy implementation — no filterpy or other external deps.
"""

from __future__ import annotations

import math
import numpy as np

from flux_manifold.changepoint import BayesianChangePoint


# ── Unscented Transform parameters ───────────────────────────────

# Merwe's scaled sigma point parameters
_ALPHA = 1e-3      # spread of sigma points (small → tight around mean)
_BETA = 2.0        # prior knowledge of distribution (2 = Gaussian optimal)
_KAPPA = 0.0       # secondary scaling parameter


def _sigma_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Van der Merwe's sigma point weights for n-dim state.

    Returns (Wm, Wc) weight vectors of length 2n+1.
    """
    lam = _ALPHA ** 2 * (n + _KAPPA) - n

    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))

    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - _ALPHA ** 2 + _BETA)

    return Wm, Wc


def _sigma_points(x: np.ndarray, P: np.ndarray, n: int) -> np.ndarray:
    """Generate 2n+1 sigma points from mean x and covariance P.

    Returns (2n+1, n) array.
    """
    lam = _ALPHA ** 2 * (n + _KAPPA) - n
    sqrt_factor = (n + lam)

    # Cholesky decomposition of scaled covariance
    # Add small jitter for numerical stability
    scaled_P = sqrt_factor * P
    try:
        L = np.linalg.cholesky(scaled_P)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigendecomposition fallback
        eigvals, eigvecs = np.linalg.eigh(scaled_P)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i + 1] = x + L[i]
        sigmas[n + i + 1] = x - L[i]

    return sigmas


# ── Per-dimension scalar UKF ─────────────────────────────────────

class _EdgeUKF:
    """3-state UKF for a single edge dimension.

    State: [log_price, drift, log_volatility]
    Observation: log_price from AMM
    """

    _N = 3  # state dimension

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        dt: float = 1.0,
    ):
        self.dt = dt

        # State: [log_price, drift, log_volatility]
        self.x = np.zeros(self._N)
        self.P = np.eye(self._N) * 0.1  # initial covariance (uninformative)

        # Noise
        self._base_Q = np.diag([process_noise, process_noise * 0.1, process_noise * 0.01])
        self.Q = self._base_Q.copy()
        self.R = np.array([[measurement_noise]])  # 1×1 measurement noise

        # Weights
        self._Wm, self._Wc = _sigma_weights(self._N)

        # Adaptive gain state
        self._inflated_steps = 0
        self._initialized = False

    def predict(self) -> None:
        """Predict step: propagate state through process model."""
        sigmas = _sigma_points(self.x, self.P, self._N)

        # Process model: x_{t+1} = f(x_t)
        #   log_price += drift * dt
        #   drift     += 0 (random walk)
        #   log_vol   += 0 (random walk, mean-reverting handled by Q)
        sigmas_pred = sigmas.copy()
        sigmas_pred[:, 0] += sigmas[:, 1] * self.dt

        # Predicted mean and covariance
        x_pred = self._Wm @ sigmas_pred
        diff = sigmas_pred - x_pred
        P_pred = (self._Wc[:, None] * diff).T @ diff + self.Q

        self.x = x_pred
        self.P = P_pred
        self._sigmas_pred = sigmas_pred

    def update(self, z: float) -> float:
        """Update step with observation z (log_price).

        Returns the innovation (residual).
        """
        sigmas = _sigma_points(self.x, self.P, self._N)

        # Observation model: h(x) = x[0] (log_price)
        z_sigmas = sigmas[:, 0]  # (2n+1,)

        # Predicted observation mean and covariance
        z_pred = self._Wm @ z_sigmas
        dz = z_sigmas - z_pred
        S = float((self._Wc * dz) @ dz) + self.R[0, 0]

        # Cross-covariance
        dx = sigmas - self.x
        Pxz = (self._Wc[:, None] * dx).T @ dz  # (3,)

        # Kalman gain
        K = Pxz / S  # (3,)

        # Innovation
        innovation = z - z_pred

        # Adaptive gain: if innovation > 3σ, inflate Q
        sigma_innov = math.sqrt(max(S, 1e-12))
        if abs(innovation) > 3.0 * sigma_innov:
            self.Q = self._base_Q * 10.0
            self._inflated_steps = 3
        elif self._inflated_steps > 0:
            self._inflated_steps -= 1
            if self._inflated_steps == 0:
                self.Q = self._base_Q.copy()

        # Update state
        self.x = self.x + K * innovation
        self.P = self.P - np.outer(K, K) * S

        # Ensure P stays symmetric positive definite
        self.P = 0.5 * (self.P + self.P.T)
        np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-10))
        # Clamp covariance entries to prevent overflow
        np.clip(self.P, -1e6, 1e6, out=self.P)

        return innovation

    def step(self, z: float) -> float:
        """Combined predict + update. Returns posterior log_price estimate."""
        if not self._initialized:
            # First observation: initialize state
            self.x[0] = z
            self.x[1] = 0.0   # zero drift
            self.x[2] = -5.0  # log(vol) ≈ exp(-5) ≈ 0.007
            self._initialized = True
            return z

        self.predict()
        self.update(z)
        # Guard against NaN/Inf blowup
        if not np.all(np.isfinite(self.x)):
            self.reset()
            self.x[0] = z
            self._initialized = True
        return self.x[0]  # posterior log_price

    def reset(self) -> None:
        """Reset to uninformative prior."""
        self.x = np.zeros(self._N)
        self.P = np.eye(self._N) * 0.1
        self.Q = self._base_Q.copy()
        self._inflated_steps = 0
        self._initialized = False

    @property
    def variance(self) -> float:
        """Posterior variance of log_price."""
        return float(self.P[0, 0])

    @property
    def drift(self) -> float:
        """Current drift estimate."""
        return float(self.x[1])

    @property
    def log_volatility(self) -> float:
        """Current log-volatility estimate."""
        return float(self.x[2])


# ── Main KalmanReservoir class ────────────────────────────────────

class KalmanReservoir:
    """UKF-based reservoir replacing EMA for price manifold estimation.

    Maintains one 3-state UKF per edge dimension. Provides the same
    step(x) → np.ndarray interface as ReservoirState, plus posterior
    covariance for uncertainty-aware downstream processing.
    """

    def __init__(
        self,
        d: int,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        seed: int = 42,
        # Ignored but accepted for API compatibility with ReservoirState
        reservoir_scale: int = 4,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 0.05,
    ):
        self.d = d
        self._filters: list[_EdgeUKF] = [
            _EdgeUKF(process_noise=process_noise, measurement_noise=measurement_noise)
            for _ in range(d)
        ]
        self._changepoint_detectors: list[BayesianChangePoint] = [
            BayesianChangePoint(hazard_rate=1 / 300, threshold=0.5)
            for _ in range(d)
        ]
        self._regime_changes: list[bool] = [False] * d
        self._changepoint_events: list[tuple[int, int]] = []  # (step, dim)
        self._step_count = 0

    def step(self, x: np.ndarray) -> np.ndarray:
        """Process one observation vector, return posterior estimates.

        Same interface as ReservoirState.step(): takes (d,) input,
        returns (d,) output. Resets UKF on detected changepoints.
        """
        out = np.zeros(self.d, dtype=np.float32)
        self._regime_changes = [False] * self.d
        for i in range(self.d):
            # Changepoint detection on the raw observation
            self._changepoint_detectors[i].update(float(x[i]))
            if self._changepoint_detectors[i].is_changepoint():
                # Reset this dimension's filter to adapt to new regime
                self._filters[i].reset()
                self._regime_changes[i] = True
                self._changepoint_events.append((self._step_count, i))
            out[i] = self._filters[i].step(float(x[i]))
        self._step_count += 1
        return out

    @property
    def regime_change_detected(self) -> bool:
        """True if any dimension had a changepoint this step."""
        return any(self._regime_changes)

    @property
    def regime_change_dims(self) -> list[int]:
        """Dimensions that had a changepoint this step."""
        return [i for i, v in enumerate(self._regime_changes) if v]

    @property
    def changepoint_events(self) -> list[tuple[int, int]]:
        """All (step, dimension) changepoint events."""
        return self._changepoint_events

    def get_covariance(self) -> np.ndarray:
        """Return posterior variance for each dimension (d,)."""
        return np.array([f.variance for f in self._filters], dtype=np.float32)

    def get_drifts(self) -> np.ndarray:
        """Return drift estimates for each dimension (d,)."""
        return np.array([f.drift for f in self._filters], dtype=np.float32)

    def get_log_volatilities(self) -> np.ndarray:
        """Return log-volatility estimates for each dimension (d,)."""
        return np.array([f.log_volatility for f in self._filters], dtype=np.float32)

    def reset(self, dimensions: list[int] | None = None) -> None:
        """Reset specific dimensions or all to uninformative prior."""
        if dimensions is None:
            for f in self._filters:
                f.reset()
            for cp in self._changepoint_detectors:
                cp.reset()
            self._changepoint_events.clear()
        else:
            for i in dimensions:
                if 0 <= i < self.d:
                    self._filters[i].reset()
                    self._changepoint_detectors[i].reset()

    @property
    def step_count(self) -> int:
        return self._step_count

    # Compatibility aliases
    def update(self, x: np.ndarray) -> np.ndarray:
        return self.step(x)

    def readout(self) -> np.ndarray:
        return np.array([f.x[0] for f in self._filters], dtype=np.float32)
