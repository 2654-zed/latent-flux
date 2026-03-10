"""Multi-Scale Reservoir — parallel smoothing at 3 timescales.

Runs micro (3h), meso (20h), and macro (100h) reservoirs in parallel.
When all three agree on deviation direction, scale_agreement → 1.0.
Disagreement → scale_agreement → 0.0.

Usage:
    from flux_manifold.multi_scale_reservoir import MultiScaleReservoir

    ms = MultiScaleReservoir(d=20)
    smoothed = ms.step(market_state)
    agreement = ms.get_scale_agreement(market_state)
"""

from __future__ import annotations

import numpy as np

from flux_manifold.reservoir_state import ReservoirState


# Scale definitions: name → (leak_rate, window_hours)
SCALES = {
    "micro": {"leak_rate": 0.3, "window_hours": 3},
    "meso":  {"leak_rate": 0.05, "window_hours": 20},
    "macro": {"leak_rate": 0.01, "window_hours": 100},
}


class MultiScaleReservoir:
    """Three ReservoirState instances at different timescales."""

    def __init__(
        self,
        d: int,
        reservoir_scale: int = 4,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.d = d
        self._reservoirs: dict[str, ReservoirState] = {}
        for name, params in SCALES.items():
            self._reservoirs[name] = ReservoirState(
                d=d,
                reservoir_scale=reservoir_scale,
                spectral_radius=spectral_radius,
                input_scaling=input_scaling,
                leak_rate=params["leak_rate"],
                seed=seed,
            )
        self._step_count = 0

    def step(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Feed observation through all 3 scales. Returns dict of smoothed states."""
        result = {}
        for name, res in self._reservoirs.items():
            result[name] = res.step(x)
        self._step_count += 1
        return result

    def get_scale_agreement(self, market_state: np.ndarray) -> float:
        """Compute agreement between scales on deviation direction.

        For each dimension, compute deviation = market - smoothed at each scale.
        Agreement = fraction of dimensions where all 3 scales agree on sign.
        Returns value in [0, 1].
        """
        if self._step_count < 2:
            return 0.0

        deviations = {}
        for name, res in self._reservoirs.items():
            # Get the reservoir's current internal state (last smoothed output)
            smoothed = res.step(market_state)
            deviations[name] = market_state - smoothed

        # For each dimension, check if all 3 agree on sign
        micro = deviations["micro"]
        meso = deviations["meso"]
        macro = deviations["macro"]

        # Sign agreement: all positive or all negative
        signs_agree = (
            ((micro > 0) & (meso > 0) & (macro > 0))
            | ((micro < 0) & (meso < 0) & (macro < 0))
        )
        return float(signs_agree.mean())

    def get_scale_agreement_from_cached(
        self,
        market_state: np.ndarray,
        smoothed_outputs: dict[str, np.ndarray],
    ) -> float:
        """Compute agreement using already-computed smoothed outputs.

        Avoids double-stepping the reservoirs.
        """
        if self._step_count < 2:
            return 0.0

        micro = market_state - smoothed_outputs["micro"]
        meso = market_state - smoothed_outputs["meso"]
        macro = market_state - smoothed_outputs["macro"]

        signs_agree = (
            ((micro > 0) & (meso > 0) & (macro > 0))
            | ((micro < 0) & (meso < 0) & (macro < 0))
        )
        return float(signs_agree.mean())

    def reset(self) -> None:
        """Reset all scales."""
        for res in self._reservoirs.values():
            res.reset()
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count
