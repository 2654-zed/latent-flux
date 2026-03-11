"""PMA Reservoir Tracker — wraps ReservoirState for probability tracking.

One ReservoirState per market pair, with dimensionality d=4:
    [poly_yes, kalshi_yes, spread, volume_ratio]

Uses the same ESN dynamics as latent_flux_searcher.py to track
temporal patterns in cross-platform prediction market spreads.
"""

from __future__ import annotations

import numpy as np

from flux_manifold.reservoir_state import ReservoirState


# Reservoir configuration (mirrors latent_flux_searcher.py settings)
_RESERVOIR_LEAK_RATE = 0.05
_RESERVOIR_SCALE = 4
_SPECTRAL_RADIUS = 0.9
_INPUT_SCALING = 0.1
_DIM = 4  # poly_yes, kalshi_yes, spread, volume_ratio


class PMAReservoirTracker:
    """Track prediction market dynamics using ReservoirState.

    One reservoir per market_id. The hidden state captures temporal
    patterns in cross-platform spreads.
    """

    def __init__(self):
        self._reservoirs: dict[str, ReservoirState] = {}
        # Running stats for manifold deviation: {market_id: [spread_history]}
        self._spread_history: dict[str, list[float]] = {}

    def _get_reservoir(self, market_id: str) -> ReservoirState:
        if market_id not in self._reservoirs:
            self._reservoirs[market_id] = ReservoirState(
                d=_DIM,
                reservoir_scale=_RESERVOIR_SCALE,
                spectral_radius=_SPECTRAL_RADIUS,
                input_scaling=_INPUT_SCALING,
                leak_rate=_RESERVOIR_LEAK_RATE,
                seed=hash(market_id) % (2**31),
            )
        return self._reservoirs[market_id]

    def update(
        self,
        market_id: str,
        poly_yes: float,
        kalshi_yes: float,
        spread: float,
        volume_ratio: float,
    ) -> np.ndarray:
        """Update reservoir for this market, return hidden state."""
        res = self._get_reservoir(market_id)
        x = np.array(
            [poly_yes, kalshi_yes, spread, volume_ratio],
            dtype=np.float32,
        )
        res.step(x)
        self._spread_history.setdefault(market_id, []).append(spread)
        return res.hidden_state

    def get_manifold_deviation(self, market_id: str) -> float:
        """Current spread vs historical manifold mean.

        Returns deviation as fraction: 0.05 = 5% above manifold mean spread.
        Equivalent to cex_deviation in latent_flux_searcher.py.
        """
        history = self._spread_history.get(market_id, [])
        if len(history) < 2:
            return 0.0

        current = history[-1]
        mean = np.mean(history[:-1])
        if abs(mean) < 1e-8:
            return 0.0
        return float((current - mean) / max(abs(mean), 1e-8))

    def get_s_score(self, market_id: str) -> float:
        """Reservoir-based signal strength for this market.

        Uses the readout magnitude as proxy for signal strength,
        analogous to s_score in latent_flux_searcher.py.
        """
        res = self._reservoirs.get(market_id)
        if res is None:
            return 0.0
        readout = res.readout()
        return float(np.linalg.norm(readout))

    def reset(self) -> None:
        """Clear all reservoir state."""
        self._reservoirs.clear()
        self._spread_history.clear()
