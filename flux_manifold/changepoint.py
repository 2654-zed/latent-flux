"""Bayesian Online Change Point Detection (Adams & MacKay, 2007).

Implements a streaming changepoint detector using Normal-Gamma conjugate
prior for Gaussian observations with unknown mean and variance. Tracks
run-length distribution to compute P(changepoint) at each step.

Usage:
    from flux_manifold.changepoint import BayesianChangePoint

    cp = BayesianChangePoint(hazard_rate=1/300)
    for obs in observations:
        prob = cp.update(obs)
        if cp.is_changepoint():
            print("Regime change detected")
"""

from __future__ import annotations

import numpy as np


class BayesianChangePoint:
    """Online Bayesian changepoint detector with Normal-Gamma conjugate prior.

    Parameters:
        hazard_rate: Constant hazard rate (1/expected_run_length).
                     Default 1/300 ≈ 300-hour expected run for crypto.
        threshold:   P(changepoint) above which we declare a change.
        prior_mu:    Prior mean for Normal-Gamma.
        prior_kappa: Prior precision scaling (pseudo-observations for mean).
        prior_alpha: Prior shape of Gamma on precision.
        prior_beta:  Prior rate of Gamma on precision.
        max_run:     Maximum run length to track (prune beyond this).
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 300,
        threshold: float = 0.5,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        max_run: int = 500,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        self.max_run = max_run

        # Prior hyperparameters (stored for reset)
        self._prior_mu = prior_mu
        self._prior_kappa = prior_kappa
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Sufficient statistics arrays — one entry per run length
        # Index r corresponds to run length r
        self._mu = np.array([prior_mu], dtype=np.float64)
        self._kappa = np.array([prior_kappa], dtype=np.float64)
        self._alpha = np.array([prior_alpha], dtype=np.float64)
        self._beta = np.array([prior_beta], dtype=np.float64)

        # Run-length distribution (log space for numerical stability)
        self._log_R = np.array([0.0], dtype=np.float64)  # P(r=0) = 1

        self._t = 0
        self._cp_prob = 0.0
        self._map_run_length = 0
        self._changepoint_log: list[int] = []  # timestamps of detected CPs

    def update(self, x: float) -> float:
        """Process one observation. Returns P(changepoint at this step).

        Implements the message-passing algorithm from Adams & MacKay (2007):
        1. Evaluate predictive probability under each run length
        2. Compute growth and changepoint probabilities
        3. Update sufficient statistics
        4. Normalize
        """
        self._t += 1

        # 1. Predictive probability: Student-t with 2*alpha df
        #    p(x | r) = Student_t(x; mu_r, beta_r*(kappa_r+1)/(alpha_r*kappa_r), 2*alpha_r)
        df = 2.0 * self._alpha
        scale = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa)
        # Guard against non-positive scale
        scale = np.maximum(scale, 1e-20)

        # Log Student-t density
        log_pred = self._log_student_t(x, self._mu, scale, df)

        # 2. Growth probabilities: P(r_t = r_{t-1}+1, x_{1:t})
        log_H = np.log(self.hazard_rate)
        log_1mH = np.log(1.0 - self.hazard_rate)
        log_growth = self._log_R + log_pred + log_1mH

        # Changepoint probability: P(r_t = 0, x_{1:t})
        # Sum over all run lengths contributing to changepoint
        log_cp = self._logsumexp(self._log_R + log_pred + log_H)

        # 3. Combine into new run-length distribution
        new_log_R = np.empty(len(log_growth) + 1, dtype=np.float64)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # Normalize
        log_evidence = self._logsumexp(new_log_R)
        new_log_R -= log_evidence

        # 4. Update sufficient statistics for each run length
        # New run (r=0): reset to prior
        new_mu = np.empty(len(self._mu) + 1, dtype=np.float64)
        new_kappa = np.empty_like(new_mu)
        new_alpha = np.empty_like(new_mu)
        new_beta = np.empty_like(new_mu)

        new_mu[0] = self._prior_mu
        new_kappa[0] = self._prior_kappa
        new_alpha[0] = self._prior_alpha
        new_beta[0] = self._prior_beta

        # Existing runs: Bayesian update of Normal-Gamma
        kappa_new = self._kappa + 1.0
        new_mu[1:] = (self._kappa * self._mu + x) / kappa_new
        new_kappa[1:] = kappa_new
        new_alpha[1:] = self._alpha + 0.5
        new_beta[1:] = (
            self._beta
            + 0.5 * self._kappa * (x - self._mu) ** 2 / kappa_new
        )

        # Prune: keep only top max_run entries
        if len(new_log_R) > self.max_run:
            new_log_R = new_log_R[: self.max_run]
            new_mu = new_mu[: self.max_run]
            new_kappa = new_kappa[: self.max_run]
            new_alpha = new_alpha[: self.max_run]
            new_beta = new_beta[: self.max_run]
            # Re-normalize after pruning
            log_evidence = self._logsumexp(new_log_R)
            new_log_R -= log_evidence

        self._log_R = new_log_R
        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        # Changepoint detection via MAP run length drop.
        # In BOCPD, P(r=0) ≡ hazard_rate always. The real signal is when
        # the MAP run length drops — fresh run lengths accumulate more mass
        # because they predict the new regime's data better.
        R_probs = np.exp(new_log_R)
        self._map_run_length = int(np.argmax(R_probs))
        # P(r < window) captures whether we're likely in a new regime
        window = min(10, len(R_probs))
        self._cp_prob = float(np.sum(R_probs[:window]))

        if self._map_run_length < 10 and self._t > 20 and self._cp_prob > self.threshold:
            self._changepoint_log.append(self._t)

        return self._cp_prob

    def is_changepoint(self, threshold: float | None = None) -> bool:
        """Check if a regime change was recently detected."""
        t = threshold if threshold is not None else self.threshold
        return (
            self._map_run_length < 10
            and self._t > 20
            and self._cp_prob > t
        )

    @property
    def map_run_length(self) -> int:
        """MAP estimate of current run length."""
        return self._map_run_length

    @property
    def changepoint_probability(self) -> float:
        """P(r < 10) — probability we're in a young regime."""
        return self._cp_prob

    @property
    def changepoint_log(self) -> list[int]:
        """List of timesteps where changepoints were detected."""
        return self._changepoint_log

    @property
    def step_count(self) -> int:
        return self._t

    def reset(self) -> None:
        """Reset to initial state."""
        self._mu = np.array([self._prior_mu], dtype=np.float64)
        self._kappa = np.array([self._prior_kappa], dtype=np.float64)
        self._alpha = np.array([self._prior_alpha], dtype=np.float64)
        self._beta = np.array([self._prior_beta], dtype=np.float64)
        self._log_R = np.array([0.0], dtype=np.float64)
        self._t = 0
        self._cp_prob = 0.0
        self._changepoint_log = []

    @staticmethod
    def _log_student_t(
        x: float,
        mu: np.ndarray,
        scale: np.ndarray,
        df: np.ndarray,
    ) -> np.ndarray:
        """Log density of Student-t distribution (vectorized over parameters)."""
        from math import lgamma as _lgamma

        # Student-t: p(x|mu,s,v) ∝ (1 + (x-mu)^2/(v*s))^(-(v+1)/2) / sqrt(v*s)
        half_df = df / 2.0
        half_dfp1 = (df + 1.0) / 2.0

        # Use scipy-free lgamma via numpy loop (avoid scipy dependency)
        # lgamma vectorized
        log_num = np.array([_lgamma(float(v)) for v in half_dfp1])
        log_den = np.array([_lgamma(float(v)) for v in half_df])

        z = (x - mu) ** 2 / (df * scale)
        log_p = (
            log_num
            - log_den
            - 0.5 * np.log(df * scale * np.pi)
            - half_dfp1 * np.log1p(z)
        )
        return log_p

    @staticmethod
    def _logsumexp(x: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        if len(x) == 0:
            return -np.inf
        c = float(np.max(x))
        if not np.isfinite(c):
            return -np.inf
        return c + float(np.log(np.sum(np.exp(x - c))))
