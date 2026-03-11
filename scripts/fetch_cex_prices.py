"""Fetch historical CEX hourly prices from Coinbase for backtest comparison.

Downloads 1-hour OHLCV candles for ETH-USDC, ETH-USDT, BTC-USDC, BTC-USDT
over the same 30-day window as uniswap_v3_30d.json (Feb 8 – Mar 10, 2026).
Saves to backtest/data/cex_prices.json.

Coinbase Advanced Trade candles endpoint is public — no API key required.
Max 300 candles per request, so 30 days (720h) requires pagination.

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


# Coinbase Advanced Trade public candles endpoint
COINBASE_CANDLES_URL = (
    "https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
)

# Time window (matching uniswap_v3_30d.json)
START_TS = 1770508800   # 2026-02-08 00:00:00 UTC
END_TS   = 1773100800   # 2026-03-10 00:00:00 UTC

# Pairs to fetch (Coinbase product_id, canonical pair name)
PAIRS = [
    ("ETH-USDC", "ETH/USDC"),
    ("ETH-USDT", "ETH/USDT"),
    ("BTC-USDC", "BTC/USDC"),
    ("BTC-USDT", "BTC/USDT"),
]

# Fixed chunk boundaries for the 30-day window (240 hours each)
_CHUNKS = [
    (1770508800, 1771372800),   # Chunk 1: Feb 08 → Feb 18
    (1771372800, 1772236800),   # Chunk 2: Feb 18 → Feb 28
    (1772236800, 1773100800),   # Chunk 3: Feb 28 → Mar 10
]

OUTPUT_PATH = Path(__file__).parent.parent / "backtest" / "data" / "cex_prices.json"


def _fetch_chunk(product_id: str, start: int, end: int) -> list[dict]:
    """Fetch one chunk of candles from Coinbase Advanced Trade API."""
    url = COINBASE_CANDLES_URL.format(product_id=product_id)
    url += f"?start={start}&end={end}&granularity=ONE_HOUR"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "latent-flux-backtest/1.0")

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("candles", [])


def fetch_coinbase_candles(product_id: str) -> list[dict]:
    """Fetch all hourly candles for a product using 3 fixed chunks."""
    all_candles: list[dict] = []

    for i, (chunk_start, chunk_end) in enumerate(_CHUNKS, 1):
        try:
            page = _fetch_chunk(product_id, chunk_start, chunk_end)
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"\n  ERROR on chunk {i}: {e}", file=sys.stderr)
            return []
        print(f"chunk{i}={len(page)}", end=" ", flush=True)
        all_candles.extend(page)
        time.sleep(0.3)

    return all_candles


def candles_to_records(candles: list[dict], pair_name: str) -> list[dict]:
    """Convert Coinbase candles to CexFeed-compatible records.

    Mid-price = (high + low) / 2 for each hourly candle.
    """
    seen: set[int] = set()
    records = []
    for c in candles:
        ts = int(c["start"])
        if ts in seen:
            continue
        seen.add(ts)
        high = float(c["high"])
        low = float(c["low"])
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

    all_records: list[dict] = []
    failed: list[str] = []

    for product_id, pair_name in PAIRS:
        print(f"Fetching {pair_name} ({product_id})... ", end="", flush=True)
        candles = fetch_coinbase_candles(product_id)
        if not candles:
            failed.append(pair_name)
            print("FAILED")
            continue
        records = candles_to_records(candles, pair_name)
        count = len(records)
        print(f"→ {count} candles")
        if count < 700:
            print(f"  ABORT: {pair_name} has only {count} candles (need ≥700)",
                  file=sys.stderr)
            return 1
        all_records.extend(records)

    if not all_records:
        print("\nNo data fetched. Check network connectivity.", file=sys.stderr)
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
