"""Fetch historical CEX hourly prices from Binance for backtest comparison.

Downloads 1-hour OHLCV candles for ETH/USDC, ETH/USDT, BTC/USDC, BTC/USDT
over the same 30-day window as uniswap_v3_30d.json (Feb 8 – Mar 10, 2026).
Saves to backtest/data/cex_prices.json.

Binance klines endpoint is public — no API key required.
30 days × 24 hours = 720 candles per pair (under 1000 limit per request).

Usage:
    python scripts/fetch_cex_prices.py
    python scripts/fetch_cex_prices.py --help
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# Binance public REST endpoint
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Time window (matching uniswap_v3_30d.json)
START_TS = 1770508800   # 2026-02-08 00:00:00 UTC
END_TS   = 1773100800   # 2026-03-10 00:00:00 UTC

# Pairs to fetch (Binance symbol, canonical pair name)
PAIRS = [
    ("ETHUSDC", "ETH/USDC"),
    ("ETHUSDT", "ETH/USDT"),
    ("BTCUSDC", "BTC/USDC"),
    ("BTCUSDT", "BTC/USDT"),
]

OUTPUT_PATH = Path(__file__).parent.parent / "backtest" / "data" / "cex_prices.json"


def fetch_binance_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1h",
    limit: int = 1000,
) -> list[list]:
    """Fetch OHLCV klines from Binance REST API.

    Returns raw kline arrays as returned by Binance:
        [open_time, open, high, low, close, volume, close_time, ...]
    """
    params = (
        f"symbol={symbol}&interval={interval}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    )
    url = f"{BINANCE_KLINES_URL}?{params}"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "latent-flux-backtest/1.0")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except urllib.error.URLError as e:
        print(f"  ERROR: Could not reach Binance API: {e}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON response: {e}", file=sys.stderr)
        return []


def klines_to_records(klines: list[list], pair_name: str) -> list[dict]:
    """Convert raw Binance klines to CexFeed-compatible records.

    Mid-price = (high + low) / 2 for each hourly candle.
    """
    records = []
    for k in klines:
        # k[0] = open_time_ms, k[2] = high, k[3] = low
        ts = int(k[0]) // 1000  # ms → seconds
        high = float(k[2])
        low = float(k[3])
        mid = (high + low) / 2.0
        records.append({
            "timestamp": ts,
            "pair": pair_name,
            "price": round(mid, 6),
        })
    return records


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(f"Output: {OUTPUT_PATH}")
        print(f"Window: {START_TS} – {END_TS} (Feb 8 – Mar 10, 2026)")
        print(f"Pairs:  {', '.join(p[1] for p in PAIRS)}")
        return 0

    start_ms = START_TS * 1000
    end_ms = END_TS * 1000

    all_records: list[dict] = []
    failed: list[str] = []

    for symbol, pair_name in PAIRS:
        print(f"Fetching {pair_name} ({symbol})...", end=" ", flush=True)
        klines = fetch_binance_klines(symbol, start_ms, end_ms)
        if klines:
            records = klines_to_records(klines, pair_name)
            all_records.extend(records)
            print(f"{len(records)} candles")
        else:
            failed.append(pair_name)
            print("FAILED")
        # Respect Binance rate limits
        time.sleep(0.5)

    if not all_records:
        print("\nNo data fetched. Check network connectivity.", file=sys.stderr)
        print("Binance API may be unreachable from this environment.")
        return 1

    # Sort by timestamp then pair
    all_records.sort(key=lambda r: (r["timestamp"], r["pair"]))

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nSaved {len(all_records)} records to {OUTPUT_PATH}")
    if failed:
        print(f"Failed pairs: {', '.join(failed)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
