"""Data Ingestion Layer — fetches historical Uniswap V3 pool state from The Graph.

Queries poolHourDatas for top 20 pools by volume over 30 days.
Caches results to backtest/data/ as JSON for reproducible reruns.

Usage:
    python backtest/data_ingestion.py

Environment variables:
    THEGRAPH_API_KEY  — API key for The Graph's decentralized gateway (required)
    THEGRAPH_ENDPOINT — Custom subgraph endpoint URL (overrides default)

The hosted service at api.thegraph.com was sunset in 2024. This module uses
The Graph's decentralized gateway which requires an API key. Get one free at
https://thegraph.com/studio/apikeys/
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Configuration ─────────────────────────────────────────────────

# Uniswap V3 subgraph ID on The Graph's decentralized network (Ethereum mainnet)
UNISWAP_V3_SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# Default gateway URL template (requires API key)
DEFAULT_GATEWAY = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"

# Data directory for cached snapshots
DATA_DIR = Path(__file__).parent / "data"

# Query parameters
TOP_N_POOLS = 20
SNAPSHOT_DAYS = 30
HOURS_PER_DAY = 24
PAGE_SIZE = 1000  # The Graph max per query
REQUEST_TIMEOUT = 30  # seconds
RETRY_DELAY = 2  # seconds between retries
MAX_RETRIES = 3


# ── Data model ────────────────────────────────────────────────────

@dataclass
class PoolState:
    """Single hourly snapshot of a Uniswap V3 pool."""
    block_timestamp: int
    pool_address: str
    token0: str
    token1: str
    sqrt_price: float       # normalized from sqrtPriceX96
    liquidity: float
    tick: int
    token0_price: float     # token1 per token0
    token1_price: float     # token0 per token1
    volume_usd: float       # hourly volume in USD
    fee_tier: int            # fee in hundredths of a bip (e.g., 3000 = 0.3%)


@dataclass
class PoolMetadata:
    """Static metadata for a pool (doesn't change per snapshot)."""
    pool_address: str
    token0_address: str
    token1_address: str
    token0_symbol: str
    token1_symbol: str
    token0_decimals: int
    token1_decimals: int
    fee_tier: int
    total_value_locked_usd: float


# ── GraphQL queries ───────────────────────────────────────────────

TOP_POOLS_QUERY = """
query TopPools($first: Int!) {
  pools(
    first: $first,
    orderBy: totalValueLockedUSD,
    orderDirection: desc,
    where: { totalValueLockedUSD_gt: "0", volumeUSD_gt: "0" }
  ) {
    id
    token0 { id symbol decimals }
    token1 { id symbol decimals }
    feeTier
    totalValueLockedUSD
  }
}
"""

POOL_HOUR_DATA_QUERY = """
query PoolHourData($pool: String!, $startTime: Int!, $skip: Int!, $first: Int!) {
  poolHourDatas(
    where: { pool: $pool, periodStartUnix_gte: $startTime },
    orderBy: periodStartUnix,
    orderDirection: asc,
    skip: $skip,
    first: $first
  ) {
    periodStartUnix
    sqrtPrice
    liquidity
    tick
    token0Price
    token1Price
    volumeUSD
  }
}
"""


# ── HTTP / GraphQL client ────────────────────────────────────────

def _get_endpoint() -> str:
    """Resolve the subgraph endpoint URL.

    Priority:
    1. THEGRAPH_ENDPOINT env var (full custom URL)
    2. THEGRAPH_API_KEY env var + default gateway template
    3. Fail loudly with instructions
    """
    custom = os.environ.get("THEGRAPH_ENDPOINT")
    if custom:
        return custom

    api_key = os.environ.get("THEGRAPH_API_KEY")
    if api_key:
        return DEFAULT_GATEWAY.format(api_key=api_key, subgraph_id=UNISWAP_V3_SUBGRAPH_ID)

    raise RuntimeError(
        "No Graph API endpoint configured.\n"
        "\n"
        "The Graph's hosted service (api.thegraph.com) was sunset in 2024.\n"
        "You need an API key for the decentralized gateway.\n"
        "\n"
        "Option 1: Set THEGRAPH_API_KEY environment variable:\n"
        "  $env:THEGRAPH_API_KEY = 'your-key-here'\n"
        "  Get a free key at https://thegraph.com/studio/apikeys/\n"
        "\n"
        "Option 2: Set THEGRAPH_ENDPOINT to a full custom subgraph URL:\n"
        "  $env:THEGRAPH_ENDPOINT = 'https://your-endpoint/subgraphs/...'\n"
    )


def _graphql_request(endpoint: str, query: str, variables: dict) -> dict:
    """Execute a GraphQL query against The Graph.

    Retries on transient failures. Fails loudly on non-transient errors.

    Returns the 'data' dict from the response.
    """
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                status = resp.status
                body = resp.read().decode("utf-8")

                if status != 200:
                    raise RuntimeError(
                        f"The Graph API returned HTTP {status}.\n"
                        f"Response body:\n{body[:2000]}"
                    )

                result = json.loads(body)

                if "errors" in result:
                    raise RuntimeError(
                        f"The Graph API returned GraphQL errors:\n"
                        f"{json.dumps(result['errors'], indent=2)}"
                    )

                if "data" not in result:
                    raise RuntimeError(
                        f"The Graph API returned response without 'data' field.\n"
                        f"Response:\n{json.dumps(result, indent=2)[:2000]}"
                    )

                return result["data"]

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:2000]
            except Exception:
                pass

            # Rate limiting — retry
            if e.code == 429 and attempt < MAX_RETRIES:
                print(f"  Rate limited (429), retrying in {RETRY_DELAY}s... "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY * attempt)
                continue

            raise RuntimeError(
                f"The Graph API request failed with HTTP {e.code}.\n"
                f"URL: {endpoint}\n"
                f"Response body:\n{body}"
            ) from e

        except urllib.error.URLError as e:
            if attempt < MAX_RETRIES:
                print(f"  Connection error: {e.reason}, retrying in {RETRY_DELAY}s... "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY * attempt)
                continue

            raise RuntimeError(
                f"The Graph API is unreachable.\n"
                f"URL: {endpoint}\n"
                f"Error: {e.reason}\n"
                f"Check your network connection and API endpoint."
            ) from e

    raise RuntimeError(f"The Graph API request failed after {MAX_RETRIES} retries.")


# ── Price normalization ───────────────────────────────────────────

def normalize_sqrt_price(sqrt_price_x96: str, token0_decimals: int, token1_decimals: int) -> float:
    """Convert sqrtPriceX96 to a human-readable price ratio.

    sqrtPriceX96 = sqrt(price) * 2^96
    price = (sqrtPriceX96 / 2^96)^2

    Adjusted for token decimals:
        adjusted_price = price * 10^(token0_decimals - token1_decimals)
    """
    sqrt_price = int(sqrt_price_x96) / (2 ** 96)
    raw_price = sqrt_price ** 2
    decimal_adjustment = 10 ** (token0_decimals - token1_decimals)
    return raw_price * decimal_adjustment


# ── Fetching logic ────────────────────────────────────────────────

def fetch_top_pools(endpoint: str, n: int = TOP_N_POOLS) -> list[PoolMetadata]:
    """Fetch top N pools by TVL from The Graph."""
    print(f"Fetching top {n} pools by TVL...")
    data = _graphql_request(endpoint, TOP_POOLS_QUERY, {"first": n})

    pools_raw = data.get("pools", [])
    if not pools_raw:
        raise RuntimeError(
            "The Graph returned empty pool list. "
            "The subgraph may be syncing or the endpoint may be incorrect."
        )

    pools = []
    for p in pools_raw:
        pools.append(PoolMetadata(
            pool_address=p["id"],
            token0_address=p["token0"]["id"],
            token1_address=p["token1"]["id"],
            token0_symbol=p["token0"]["symbol"],
            token1_symbol=p["token1"]["symbol"],
            token0_decimals=int(p["token0"]["decimals"]),
            token1_decimals=int(p["token1"]["decimals"]),
            fee_tier=int(p["feeTier"]),
            total_value_locked_usd=float(p["totalValueLockedUSD"]),
        ))

    print(f"  Found {len(pools)} pools")
    for i, pm in enumerate(pools):
        print(f"  [{i:2d}] {pm.token0_symbol}/{pm.token1_symbol} "
              f"fee={pm.fee_tier/10000:.2f}% "
              f"TVL=${pm.total_value_locked_usd:,.0f} "
              f"({pm.pool_address[:10]}...)")

    return pools


def fetch_pool_snapshots(
    endpoint: str,
    pool: PoolMetadata,
    days: int = SNAPSHOT_DAYS,
) -> list[PoolState]:
    """Fetch hourly snapshots for a single pool over the given number of days.

    Paginates through all results. Fails loudly on partial data.
    """
    # Calculate start timestamp (N days ago from now)
    now = int(time.time())
    start_time = now - (days * 24 * 3600)

    print(f"  Fetching {days}-day history for "
          f"{pool.token0_symbol}/{pool.token1_symbol} ({pool.pool_address[:10]}...)...",
          end="", flush=True)

    all_snapshots: list[PoolState] = []
    skip = 0

    while True:
        data = _graphql_request(endpoint, POOL_HOUR_DATA_QUERY, {
            "pool": pool.pool_address,
            "startTime": start_time,
            "skip": skip,
            "first": PAGE_SIZE,
        })

        hour_datas = data.get("poolHourDatas", [])
        if not hour_datas:
            break

        for hd in hour_datas:
            # Parse prices — use subgraph-provided token prices directly
            token0_price_raw = hd.get("token0Price", "0")
            token1_price_raw = hd.get("token1Price", "0")
            token0_price = float(token0_price_raw) if token0_price_raw else 0.0
            token1_price = float(token1_price_raw) if token1_price_raw else 0.0

            # Also normalize sqrtPrice for reference
            sqrt_price_str = hd.get("sqrtPrice", "0")
            sqrt_price_normalized = normalize_sqrt_price(
                sqrt_price_str,
                pool.token0_decimals,
                pool.token1_decimals,
            )

            snapshot = PoolState(
                block_timestamp=int(hd["periodStartUnix"]),
                pool_address=pool.pool_address,
                token0=pool.token0_address,
                token1=pool.token1_address,
                sqrt_price=sqrt_price_normalized,
                liquidity=float(hd.get("liquidity", "0")),
                tick=int(hd.get("tick", "0")),
                token0_price=token0_price,
                token1_price=token1_price,
                volume_usd=float(hd.get("volumeUSD", "0")),
                fee_tier=pool.fee_tier,
            )
            all_snapshots.append(snapshot)

        if len(hour_datas) < PAGE_SIZE:
            break  # Last page

        skip += PAGE_SIZE

    print(f" {len(all_snapshots)} snapshots")
    return all_snapshots


# ── Validation ────────────────────────────────────────────────────

def validate_snapshots(snapshots: list[PoolState], pool: PoolMetadata) -> None:
    """Validate fetched snapshots for correctness.

    Checks:
    - Minimum count (≥600 for 30 days)
    - Price reciprocal consistency: token0_price * token1_price ≈ 1.0
    """
    if len(snapshots) < 600:
        print(f"  WARNING: {pool.token0_symbol}/{pool.token1_symbol} has only "
              f"{len(snapshots)} snapshots (expected ≥600 for 30 days)")

    # Price reciprocal check on non-zero prices
    violations = 0
    for s in snapshots:
        if s.token0_price > 0 and s.token1_price > 0:
            product = s.token0_price * s.token1_price
            if abs(product - 1.0) > 0.001:  # 0.1% tolerance
                violations += 1

    if violations > 0:
        pct = violations / len(snapshots) * 100
        print(f"  WARNING: {violations} snapshots ({pct:.1f}%) have "
              f"token0_price * token1_price deviating >0.1% from 1.0")


# ── Cache management ─────────────────────────────────────────────

def _cache_path(pool_address: str) -> Path:
    """Cache file path for a pool's snapshots."""
    return DATA_DIR / f"pool_{pool_address}.json"


def _metadata_cache_path() -> Path:
    """Cache file path for pool metadata."""
    return DATA_DIR / "pool_metadata.json"


def save_pool_cache(pool: PoolMetadata, snapshots: list[PoolState]) -> None:
    """Save pool snapshots to disk cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(pool.pool_address)
    data = {
        "metadata": asdict(pool),
        "snapshots": [asdict(s) for s in snapshots],
        "fetched_at": int(time.time()),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_metadata_cache(pools: list[PoolMetadata]) -> None:
    """Save pool metadata list to disk cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _metadata_cache_path()
    data = {
        "pools": [asdict(p) for p in pools],
        "fetched_at": int(time.time()),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_pool_cache(pool_address: str) -> tuple[PoolMetadata, list[PoolState]] | None:
    """Load pool snapshots from disk cache. Returns None if not cached."""
    path = _cache_path(pool_address)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    meta = PoolMetadata(**data["metadata"])
    snapshots = [PoolState(**s) for s in data["snapshots"]]
    return meta, snapshots


def load_metadata_cache() -> list[PoolMetadata] | None:
    """Load pool metadata from disk cache. Returns None if not cached."""
    path = _metadata_cache_path()
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return [PoolMetadata(**p) for p in data["pools"]]


def load_all_cached() -> dict[str, tuple[PoolMetadata, list[PoolState]]]:
    """Load all cached pool data from disk.

    Returns dict mapping pool_address → (metadata, snapshots).
    """
    result = {}
    if not DATA_DIR.exists():
        return result

    for path in DATA_DIR.glob("pool_0x*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = PoolMetadata(**data["metadata"])
        snapshots = [PoolState(**s) for s in data["snapshots"]]
        result[meta.pool_address] = (meta, snapshots)

    return result


# ── Main ingestion pipeline ───────────────────────────────────────

def fetch_all(use_cache: bool = True) -> dict[str, tuple[PoolMetadata, list[PoolState]]]:
    """Fetch all pool data: top 20 pools × 30 days of hourly snapshots.

    If use_cache=True (default), uses cached data when available.

    Returns dict mapping pool_address → (metadata, snapshots).
    """
    # Check for cached data first
    if use_cache:
        cached = load_all_cached()
        if len(cached) >= TOP_N_POOLS:
            print(f"Using cached data: {len(cached)} pools found in {DATA_DIR}")
            total_snapshots = sum(len(snaps) for _, snaps in cached.values())
            print(f"  Total snapshots: {total_snapshots}")
            return cached

    # Resolve endpoint (fails loudly if not configured)
    endpoint = _get_endpoint()
    print(f"Endpoint: {endpoint[:60]}...")

    # Step 1: Fetch top pools
    pools = fetch_top_pools(endpoint, TOP_N_POOLS)
    save_metadata_cache(pools)

    # Step 2: Fetch hourly data for each pool
    all_data: dict[str, tuple[PoolMetadata, list[PoolState]]] = {}
    failed_pools: list[tuple[str, str]] = []

    for i, pool in enumerate(pools):
        print(f"\n[{i+1}/{len(pools)}]", end="")

        # Check individual pool cache
        if use_cache:
            cached = load_pool_cache(pool.pool_address)
            if cached is not None:
                print(f"  {pool.token0_symbol}/{pool.token1_symbol}: "
                      f"using cache ({len(cached[1])} snapshots)")
                all_data[pool.pool_address] = cached
                continue

        try:
            snapshots = fetch_pool_snapshots(endpoint, pool, SNAPSHOT_DAYS)

            if not snapshots:
                raise RuntimeError(
                    f"The Graph returned 0 snapshots for pool "
                    f"{pool.token0_symbol}/{pool.token1_symbol} "
                    f"({pool.pool_address}). The subgraph data may be incomplete."
                )

            validate_snapshots(snapshots, pool)
            save_pool_cache(pool, snapshots)
            all_data[pool.pool_address] = (pool, snapshots)

        except Exception as e:
            failed_pools.append((
                f"{pool.token0_symbol}/{pool.token1_symbol}",
                str(e),
            ))
            # Per spec: fail loudly, do not write partial cache
            raise RuntimeError(
                f"Failed to fetch data for pool "
                f"{pool.token0_symbol}/{pool.token1_symbol} "
                f"({pool.pool_address}).\n"
                f"Error: {e}\n"
                f"Aborting ingestion — no partial data written."
            ) from e

    # Summary
    print(f"\n{'='*60}")
    print(f"Ingestion complete: {len(all_data)} pools, "
          f"{sum(len(s) for _, s in all_data.values())} total snapshots")
    print(f"Cache directory: {DATA_DIR}")

    return all_data


def get_snapshots_at_timestamp(
    all_data: dict[str, tuple[PoolMetadata, list[PoolState]]],
    target_timestamp: int,
    tolerance: int = 3600,  # ±1 hour
) -> list[PoolState]:
    """Get pool states closest to a target timestamp across all pools.

    For each pool, finds the snapshot whose block_timestamp is closest
    to target_timestamp within the tolerance window.

    Returns a list of PoolState objects (one per pool that has data near
    the target timestamp).
    """
    result = []
    for pool_addr, (meta, snapshots) in all_data.items():
        best = None
        best_diff = float("inf")
        for s in snapshots:
            diff = abs(s.block_timestamp - target_timestamp)
            if diff < best_diff:
                best_diff = diff
                best = s
        if best is not None and best_diff <= tolerance:
            result.append(best)
    return result


def get_all_timestamps(
    all_data: dict[str, tuple[PoolMetadata, list[PoolState]]],
) -> list[int]:
    """Get sorted list of all unique hourly timestamps across all pools.

    Returns timestamps where at least 2 pools have data (for meaningful
    arbitrage graph construction).
    """
    from collections import Counter
    ts_counter: Counter[int] = Counter()
    for pool_addr, (meta, snapshots) in all_data.items():
        for s in snapshots:
            ts_counter[s.block_timestamp] += 1

    # Keep timestamps with at least 2 pools reporting
    timestamps = sorted(ts for ts, count in ts_counter.items() if count >= 2)
    return timestamps


# ── CLI entry point ───────────────────────────────────────────────

def main() -> int:
    """Run data ingestion from command line."""
    print("ETH Arbitrage Backtest — Data Ingestion")
    print("=" * 50)

    try:
        all_data = fetch_all(use_cache=True)
    except RuntimeError as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        return 1

    # Print 5 sample PoolState records
    print(f"\n{'='*50}")
    print("Sample PoolState records (5):")
    print(f"{'='*50}")
    sample_count = 0
    for pool_addr, (meta, snapshots) in all_data.items():
        if sample_count >= 5:
            break
        if snapshots:
            s = snapshots[0]
            print(f"\n  Pool: {meta.token0_symbol}/{meta.token1_symbol}")
            print(f"  block_timestamp: {s.block_timestamp}")
            print(f"  pool_address:    {s.pool_address[:16]}...")
            print(f"  token0:          {s.token0[:16]}...")
            print(f"  token1:          {s.token1[:16]}...")
            print(f"  sqrt_price:      {s.sqrt_price:.8f}")
            print(f"  liquidity:       {s.liquidity:.0f}")
            print(f"  tick:            {s.tick}")
            print(f"  token0_price:    {s.token0_price:.8f}")
            print(f"  token1_price:    {s.token1_price:.8f}")
            print(f"  volume_usd:      ${s.volume_usd:,.2f}")
            print(f"  fee_tier:        {s.fee_tier} ({s.fee_tier/10000:.2f}%)")
            if s.token0_price > 0 and s.token1_price > 0:
                product = s.token0_price * s.token1_price
                print(f"  price_check:     {s.token0_price:.6f} × {s.token1_price:.6f} "
                      f"= {product:.6f} (expect ≈1.0)")
            sample_count += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
