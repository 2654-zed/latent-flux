"""Cross-platform prediction market price feed.

Provides historical and real-time prices from Polymarket and Kalshi.
Mirrors the architecture of flux_manifold/cex_feed.py but for prediction
markets instead of CEXs.

Three modes:
    proxy  — synthetic prices from sample data
    file   — load from pma/data/pma_prices.json
    live   — Polymarket CLOB + Kalshi REST (stub)

A MarketPrice has:
    timestamp: int
    market_id: str        # canonical event identifier
    platform: str         # "polymarket" | "kalshi"
    yes_price: float      # probability of YES (0.0-1.0)
    volume_24h: float     # USD volume last 24h

A MarketPair has:
    market_id: str
    polymarket_slug: str
    kalshi_ticker: str
    description: str
    resolution_source: str
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path


_PMA_DATA_DIR = Path(__file__).parent / "data"
_PMA_PRICES_PATH = _PMA_DATA_DIR / "pma_prices.json"
_SAMPLE_PRICES_PATH = _PMA_DATA_DIR / "sample_pma_prices.json"
_MARKET_PAIRS_PATH = _PMA_DATA_DIR / "market_pairs.json"

# Fees: ~1% per side on both Polymarket and Kalshi
FEE_PER_SIDE = 0.01
TOTAL_FEES = 2 * FEE_PER_SIDE


@dataclass
class MarketPrice:
    timestamp: int
    market_id: str
    platform: str       # "polymarket" | "kalshi"
    yes_price: float    # 0.0-1.0
    volume_24h: float


@dataclass
class MarketPair:
    market_id: str
    polymarket_slug: str
    kalshi_ticker: str
    description: str
    resolution_source: str


class MarketFeed:
    """Cross-platform prediction market price feed.

    Args:
        mode: "proxy" (sample data), "file" (load JSON), "live" (stub).
        data_path: Path to pma_prices.json. Auto-upgrades from proxy to file
                   if the file exists.
    """

    def __init__(self, mode: str = "proxy", data_path: Path | None = None):
        if mode not in ("proxy", "file", "live"):
            raise ValueError(f"Unknown MarketFeed mode: {mode!r}")

        self._data_path = data_path or _PMA_PRICES_PATH
        self._sample_path = _SAMPLE_PRICES_PATH
        self._mode = mode

        # {(market_id, platform): [(timestamp, yes_price, volume_24h), ...]}
        self._prices: dict[tuple[str, str], list[tuple[int, float, float]]] = {}

        # Market pair metadata
        self._pairs: dict[str, MarketPair] = {}

        # Auto-detect: if pma_prices.json exists, upgrade from proxy
        if mode == "proxy" and self._data_path.exists():
            self._mode = "file"

        if self._mode == "file":
            self._load_file(self._data_path)
        elif self._mode == "proxy":
            if self._sample_path.exists():
                self._load_file(self._sample_path)
        elif self._mode == "live":
            # Stub: fall back to file/proxy
            if self._data_path.exists():
                self._load_file(self._data_path)
            elif self._sample_path.exists():
                self._load_file(self._sample_path)

        # Load market pair metadata if available
        if _MARKET_PAIRS_PATH.exists():
            self._load_pairs()

    def _load_file(self, path: Path) -> None:
        """Load prices from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for rec in records:
            market_id = rec["market_id"]
            platform = rec["platform"]
            ts = int(rec["timestamp"])
            yes_price = float(rec["yes_price"])
            volume = float(rec.get("volume_24h", 0.0))
            key = (market_id, platform)
            self._prices.setdefault(key, []).append((ts, yes_price, volume))

        # Sort by timestamp
        for key in self._prices:
            self._prices[key].sort(key=lambda x: x[0])

    def _load_pairs(self) -> None:
        """Load market pair metadata."""
        with open(_MARKET_PAIRS_PATH, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        for p in pairs:
            self._pairs[p["market_id"]] = MarketPair(**p)

    def get_price(
        self, market_id: str, platform: str, timestamp: int
    ) -> float | None:
        """Return YES price for market_id on platform at timestamp.

        Returns None if data unavailable.
        """
        key = (market_id, platform)
        entries = self._prices.get(key)
        if not entries:
            return None

        ts_list = [e[0] for e in entries]
        idx = bisect.bisect_right(ts_list, timestamp)

        if idx == 0:
            if abs(timestamp - entries[0][0]) <= 7200:
                return entries[0][1]
            return None

        if idx >= len(entries):
            if abs(timestamp - entries[-1][0]) <= 7200:
                return entries[-1][1]
            return None

        # Interpolate
        t0, p0, _ = entries[idx - 1]
        t1, p1, _ = entries[idx]
        if t1 == t0:
            return p0
        frac = (timestamp - t0) / (t1 - t0)
        return p0 + frac * (p1 - p0)

    def get_volume(
        self, market_id: str, platform: str, timestamp: int
    ) -> float:
        """Return 24h volume for market_id on platform at timestamp."""
        key = (market_id, platform)
        entries = self._prices.get(key)
        if not entries:
            return 0.0

        ts_list = [e[0] for e in entries]
        idx = bisect.bisect_right(ts_list, timestamp)
        if idx == 0:
            return entries[0][2] if abs(timestamp - entries[0][0]) <= 7200 else 0.0
        if idx >= len(entries):
            return entries[-1][2] if abs(timestamp - entries[-1][0]) <= 7200 else 0.0
        return entries[idx - 1][2]

    def get_spread(self, market_id: str, timestamp: int) -> dict | None:
        """Return spread analysis between Polymarket and Kalshi.

        Returns dict with keys: poly_yes, kalshi_yes, spread, arb_return,
        viable (True if arb_return > 0.02 after fees), side.
        Returns None if either platform lacks data.
        """
        poly = self.get_price(market_id, "polymarket", timestamp)
        kalshi = self.get_price(market_id, "kalshi", timestamp)
        if poly is None or kalshi is None:
            return None

        spread = abs(poly - kalshi)
        arb_return = spread - TOTAL_FEES

        if poly < kalshi:
            side = "buy_poly_sell_kalshi"
        else:
            side = "buy_kalshi_sell_poly"

        return {
            "poly_yes": poly,
            "kalshi_yes": kalshi,
            "spread": spread,
            "arb_return": arb_return,
            "viable": arb_return > 0.02,
            "side": side,
        }

    def all_market_ids(self) -> list[str]:
        """Return unique market IDs with data loaded."""
        return sorted({mid for mid, _ in self._prices})

    def all_timestamps(self) -> list[int]:
        """Return sorted unique timestamps across all data."""
        ts_set: set[int] = set()
        for entries in self._prices.values():
            for ts, _, _ in entries:
                ts_set.add(ts)
        return sorted(ts_set)

    def get_pair(self, market_id: str) -> MarketPair | None:
        """Return MarketPair metadata, or None."""
        return self._pairs.get(market_id)

    @property
    def mode(self) -> str:
        return self._mode
