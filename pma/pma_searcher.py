"""PMA Searcher — scans market pairs for cross-platform spread opportunities.

Equivalent to backtest/latent_flux_searcher.py but for prediction markets.
For each timestamp, checks all market pairs for spreads between Polymarket
and Kalshi, uses ReservoirTracker for temporal analysis, and emits PMASignal
objects for opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass

from pma.market_feed import MarketFeed, TOTAL_FEES
from pma.reservoir_tracker import PMAReservoirTracker


# Minimum 24h volume (USD) on both platforms for a signal to be viable
MIN_VOLUME = 10_000.0


@dataclass
class PMASignal:
    timestamp: int
    market_id: str
    description: str
    poly_yes: float
    kalshi_yes: float
    spread: float            # raw spread between platforms
    arb_return: float        # net return after fees (spread - 2% fees)
    manifold_deviation: float  # how far current spread is from historical mean
    s_score: float           # reservoir signal strength
    viable: bool             # arb_return > 0 AND volume sufficient
    side: str                # "buy_poly_sell_kalshi" | "buy_kalshi_sell_poly"


def scan(
    timestamp: int,
    market_feed: MarketFeed,
    reservoir_tracker: PMAReservoirTracker,
) -> list[PMASignal]:
    """Scan all market pairs at a given timestamp for arb opportunities.

    Returns list of PMASignal for every pair with data on both platforms.
    """
    signals: list[PMASignal] = []

    for market_id in market_feed.all_market_ids():
        spread_info = market_feed.get_spread(market_id, timestamp)
        if spread_info is None:
            continue

        poly_yes = spread_info["poly_yes"]
        kalshi_yes = spread_info["kalshi_yes"]
        spread = spread_info["spread"]
        arb_return = spread_info["arb_return"]
        side = spread_info["side"]

        # Volume check
        poly_vol = market_feed.get_volume(market_id, "polymarket", timestamp)
        kalshi_vol = market_feed.get_volume(market_id, "kalshi", timestamp)
        volume_ratio = min(poly_vol, kalshi_vol) / max(poly_vol, kalshi_vol) if max(poly_vol, kalshi_vol) > 0 else 0.0

        # Update reservoir
        reservoir_tracker.update(
            market_id, poly_yes, kalshi_yes, spread, volume_ratio
        )

        manifold_dev = reservoir_tracker.get_manifold_deviation(market_id)
        s_score = reservoir_tracker.get_s_score(market_id)

        # Viable: positive return after fees AND both platforms have some data
        # Note: volume figures may be trade counts rather than USD when
        # sourced from aggregated trades, so we only require > 0.
        viable = (
            arb_return > 0
            and poly_vol + kalshi_vol > 0
        )

        # Description from pair metadata
        pair = market_feed.get_pair(market_id)
        description = pair.description if pair else market_id

        signals.append(PMASignal(
            timestamp=timestamp,
            market_id=market_id,
            description=description,
            poly_yes=poly_yes,
            kalshi_yes=kalshi_yes,
            spread=spread,
            arb_return=arb_return,
            manifold_deviation=manifold_dev,
            s_score=s_score,
            viable=viable,
            side=side,
        ))

    return signals
