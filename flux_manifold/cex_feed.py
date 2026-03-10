"""CEX Reference Price Feed.

Provides historical CEX mid-prices for token pairs.
In production: connects to Binance/Coinbase WebSocket.
In backtest: loads from JSON file or falls back to interpolation
             from the AMM data itself (CEX proxy mode).

Three modes:
    proxy  — derive CEX reference from volume-weighted AMM prices
             across all fee tiers (best available without real CEX data)
    file   — load from backtest/data/cex_prices.json if present
    live   — Binance REST API (stub; only when BINANCE_API_KEY env var set)

Supported pairs: ETH/USDC, ETH/USDT, BTC/USDC, BTC/USDT, DAI/USDC, etc.
"""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path


# Token normalization: strip "W" prefix for matching
_TOKEN_ALIASES = {
    "WETH": "ETH",
    "WBTC": "BTC",
}

# Reverse map for lookup
_ALIAS_REVERSE = {v: k for k, v in _TOKEN_ALIASES.items()}

# Default path for CEX price data
_CEX_DATA_PATH = Path(__file__).parent.parent / "backtest" / "data" / "cex_prices.json"


def _normalize_token(token: str) -> str:
    """Normalize token name: WETH→ETH, WBTC→BTC, others unchanged."""
    return _TOKEN_ALIASES.get(token, token)


def _pair_key(base: str, quote: str) -> str:
    """Canonical pair key from normalized token names."""
    return f"{_normalize_token(base)}/{_normalize_token(quote)}"


class CexFeed:
    """CEX reference price feed with proxy, file, and live modes.

    Args:
        mode: "proxy" (VW-average from AMM), "file" (load JSON), "live" (stub).
        data_path: Path to cex_prices.json for file mode. If the file exists
                   and mode="proxy", auto-upgrades to file mode.
    """

    def __init__(self, mode: str = "proxy", data_path: Path | None = None):
        if mode not in ("proxy", "file", "live"):
            raise ValueError(f"Unknown CexFeed mode: {mode!r}")

        self._data_path = data_path or _CEX_DATA_PATH
        self._mode = mode

        # Proxy state: {pair_key: [(timestamp, vwap_price), ...]} sorted by ts
        self._proxy_prices: dict[str, list[tuple[int, float]]] = {}

        # File state: same structure, pre-loaded
        self._file_prices: dict[str, list[tuple[int, float]]] = {}

        # Auto-detect: if cex_prices.json exists and mode is proxy, upgrade
        if mode == "proxy" and self._data_path.exists():
            self._mode = "file"

        if self._mode == "file":
            self._load_file()
        elif self._mode == "live":
            if "BINANCE_API_KEY" not in os.environ:
                self._mode = "proxy"

    def _load_file(self) -> None:
        """Load CEX prices from JSON file."""
        if not self._data_path.exists():
            self._mode = "proxy"
            return

        with open(self._data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # Group by pair
        by_pair: dict[str, list[tuple[int, float]]] = {}
        for rec in records:
            pair = rec["pair"]
            ts = int(rec["timestamp"])
            price = float(rec["price"])
            by_pair.setdefault(pair, []).append((ts, price))

        # Sort by timestamp and store
        for pair, entries in by_pair.items():
            entries.sort(key=lambda x: x[0])
            self._file_prices[pair] = entries
            # Also store reciprocal
            parts = pair.split("/")
            if len(parts) == 2:
                recip_pair = f"{parts[1]}/{parts[0]}"
                self._file_prices[recip_pair] = [
                    (ts, 1.0 / p) for ts, p in entries if p > 0
                ]

    def update(self, pool_states: list, timestamp: int) -> None:
        """Feed AMM pool data for proxy mode. Call once per block.

        Computes volume-weighted average price across fee tiers for each
        token pair, then derives cross-rates through ETH.
        """
        if self._mode != "proxy":
            return

        # Collect prices per normalized pair: {pair_key: [(price, volume), ...]}
        pair_data: dict[str, list[tuple[float, float]]] = {}

        for ps in pool_states:
            t0 = _normalize_token(ps.token0)
            t1 = _normalize_token(ps.token1)
            vol = max(ps.volume_usd, 0.0)

            # token0_price = units of token1 per token0
            if ps.token0_price > 0:
                key_01 = f"{t0}/{t1}"
                pair_data.setdefault(key_01, []).append((ps.token0_price, vol))

            # token1_price = units of token0 per token1
            if ps.token1_price > 0:
                key_10 = f"{t1}/{t0}"
                pair_data.setdefault(key_10, []).append((ps.token1_price, vol))

        # Compute VWAP for direct pairs
        direct_prices: dict[str, float] = {}
        for pair, entries in pair_data.items():
            total_vol = sum(v for _, v in entries)
            if total_vol > 0:
                vwap = sum(p * v for p, v in entries) / total_vol
            else:
                # Equal-weight average if no volume data
                vwap = sum(p for p, _ in entries) / len(entries)
            direct_prices[pair] = vwap

            # Store in timeline
            self._proxy_prices.setdefault(pair, []).append((timestamp, vwap))

        # Derive cross-rates through ETH for pairs not directly observed
        # e.g., BTC/USDC = BTC/ETH × ETH/USDC
        eth_pairs_buy: dict[str, float] = {}   # token/ETH prices
        eth_pairs_sell: dict[str, float] = {}  # ETH/token prices
        for pair, price in direct_prices.items():
            parts = pair.split("/")
            if len(parts) != 2:
                continue
            base, quote = parts
            if quote == "ETH":
                eth_pairs_buy[base] = price    # base/ETH
            if base == "ETH":
                eth_pairs_sell[quote] = price  # ETH/quote

        # For each (base, quote) not in direct_prices, try base→ETH→quote
        for base, base_eth in eth_pairs_buy.items():
            for quote, eth_quote in eth_pairs_sell.items():
                if base == quote:
                    continue
                cross_pair = f"{base}/{quote}"
                if cross_pair not in direct_prices:
                    cross_price = base_eth * eth_quote
                    direct_prices[cross_pair] = cross_price
                    self._proxy_prices.setdefault(cross_pair, []).append(
                        (timestamp, cross_price)
                    )
                # Also the reciprocal
                recip_pair = f"{quote}/{base}"
                if recip_pair not in direct_prices and cross_price > 0:
                    recip_price = 1.0 / (base_eth * eth_quote)
                    direct_prices[recip_pair] = recip_price
                    self._proxy_prices.setdefault(recip_pair, []).append(
                        (timestamp, recip_price)
                    )

    def get_price(self, base: str, quote: str, timestamp: int) -> float | None:
        """Return CEX reference price for base/quote at timestamp.

        Returns None if the pair is unavailable.
        """
        pair = _pair_key(base, quote)

        if self._mode == "file":
            return self._interpolate(self._file_prices, pair, timestamp)
        else:
            return self._interpolate(self._proxy_prices, pair, timestamp)

    def get_deviation(
        self, base: str, quote: str, amm_price: float, timestamp: int
    ) -> float | None:
        """Return signed deviation: (amm_price - cex_price) / cex_price.

        Positive = AMM price is higher than CEX (AMM is expensive).
        Negative = AMM price is lower than CEX (AMM is cheap → buy on AMM).
        Returns None if CEX price unavailable.
        """
        cex_price = self.get_price(base, quote, timestamp)
        if cex_price is None or cex_price <= 0:
            return None
        return (amm_price - cex_price) / cex_price

    def available_pairs(self) -> list[str]:
        """Return list of pair keys with data available."""
        if self._mode == "file":
            return list(self._file_prices.keys())
        return list(self._proxy_prices.keys())

    @staticmethod
    def _interpolate(
        store: dict[str, list[tuple[int, float]]],
        pair: str,
        timestamp: int,
    ) -> float | None:
        """Linear interpolation between stored timestamps for a pair."""
        entries = store.get(pair)
        if not entries:
            return None

        # Extract timestamps for bisect
        ts_list = [e[0] for e in entries]
        idx = bisect.bisect_right(ts_list, timestamp)

        if idx == 0:
            # Before first data point — use first price if within 2 hours
            if abs(timestamp - entries[0][0]) <= 7200:
                return entries[0][1]
            return None

        if idx >= len(entries):
            # After last data point — use last price if within 2 hours
            if abs(timestamp - entries[-1][0]) <= 7200:
                return entries[-1][1]
            return None

        # Interpolate between entries[idx-1] and entries[idx]
        t0, p0 = entries[idx - 1]
        t1, p1 = entries[idx]
        if t1 == t0:
            return p0
        frac = (timestamp - t0) / (t1 - t0)
        return p0 + frac * (p1 - p0)
