"""Backtest Harness — orchestrates side-by-side Bellman-Ford vs Latent Flux comparison.

Iterates over all cached hourly pool snapshots, runs both arbitrage
detectors on each timestamp, records every opportunity found, and
writes results to backtest/results/.

Can run on either:
  - Live data fetched via data_ingestion.py (requires THEGRAPH_API_KEY)
  - Synthetic data for deterministic, offline testing

Usage:
    python backtest/harness.py               # live data (requires cache or API key)
    python backtest/harness.py --synthetic   # synthetic data (no API needed)
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

# Ensure repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.bellman_ford import ArbOpportunity, find_arbitrage_opportunities as bf_find
from backtest.latent_flux_searcher import (
    find_arbitrage_opportunities as lf_find,
    reset_reservoir,
)
from backtest.data_ingestion import PoolState

# ── Configuration ─────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
INPUT_USD = 10000.0


# ── CSV output ────────────────────────────────────────────────────

_CSV_FIELDS = [f.name for f in fields(ArbOpportunity)]


def _write_csv_header(path: Path) -> None:
    """Write CSV header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDS)


def _append_csv_rows(path: Path, opportunities: list[ArbOpportunity]) -> None:
    """Append opportunity rows to CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for opp in opportunities:
            row = [
                opp.method,
                " → ".join(opp.path),
                opp.hop_count,
                opp.gross_profit_usd,
                opp.input_size_usd,
                opp.iterations,
                opp.block_timestamp,
            ]
            writer.writerow(row)


# ── Synthetic data generator ─────────────────────────────────────

def _build_synthetic_blocks() -> list[tuple[int, list[PoolState]]]:
    """Build 10 synthetic hourly blocks with known arbitrage profiles.

    Block layout:
      0-2: Fair prices, no arbitrage
      3:   Small triangle arb (~$5)
      4:   Larger triangle arb (~$10)
      5:   Two arbs: triangle + 2-hop
      6-7: Fair prices, no arbitrage
      8:   Large 4-hop arb (~$15)
      9:   Fair prices, no arbitrage

    Returns list of (timestamp, pool_states) tuples.
    """
    blocks: list[tuple[int, list[PoolState]]] = []

    # Token addresses (synthetic)
    WETH = "0xweth"
    USDC = "0xusdc"
    DAI = "0xdai"
    USDT = "0xusdt"
    WBTC = "0xwbtc"

    base_ts = 1700000000  # fixed epoch for determinism

    for block_idx in range(10):
        ts = base_ts + block_idx * 3600
        pools = []

        if block_idx in (0, 1, 2, 6, 7, 9):
            # Fair prices — no arbitrage
            pools = [
                PoolState(ts, "pool_weth_usdc", WETH, USDC, 0, 0, 0,
                          3200.0, 1/3200.0, 1_000_000, 3000),
                PoolState(ts, "pool_usdc_dai", USDC, DAI, 0, 0, 0,
                          1.0, 1.0, 500_000, 500),
                PoolState(ts, "pool_dai_weth", DAI, WETH, 0, 0, 0,
                          1/3200.0, 3200.0, 300_000, 3000),
                PoolState(ts, "pool_usdt_usdc", USDT, USDC, 0, 0, 0,
                          1.0, 1.0, 400_000, 100),
            ]
        elif block_idx == 3:
            # Small triangle arb: WETH→USDC slightly rich
            pools = [
                PoolState(ts, "pool_weth_usdc", WETH, USDC, 0, 0, 0,
                          3210.0, 1/3210.0, 1_000_000, 3000),
                PoolState(ts, "pool_usdc_dai", USDC, DAI, 0, 0, 0,
                          1.002, 1/1.002, 500_000, 500),
                PoolState(ts, "pool_dai_weth", DAI, WETH, 0, 0, 0,
                          1/3200.0, 3200.0, 300_000, 3000),
                PoolState(ts, "pool_usdt_usdc", USDT, USDC, 0, 0, 0,
                          1.0, 1.0, 400_000, 100),
            ]
        elif block_idx == 4:
            # Larger triangle arb
            pools = [
                PoolState(ts, "pool_weth_usdc", WETH, USDC, 0, 0, 0,
                          3220.0, 1/3220.0, 1_000_000, 3000),
                PoolState(ts, "pool_usdc_dai", USDC, DAI, 0, 0, 0,
                          1.005, 1/1.005, 500_000, 500),
                PoolState(ts, "pool_dai_weth", DAI, WETH, 0, 0, 0,
                          1/3200.0, 3200.0, 300_000, 3000),
                PoolState(ts, "pool_usdt_usdc", USDT, USDC, 0, 0, 0,
                          1.0, 1.0, 400_000, 100),
            ]
        elif block_idx == 5:
            # Two arbs: triangle + 2-hop stablecoin
            pools = [
                PoolState(ts, "pool_weth_usdc", WETH, USDC, 0, 0, 0,
                          3215.0, 1/3215.0, 1_000_000, 3000),
                PoolState(ts, "pool_usdc_dai", USDC, DAI, 0, 0, 0,
                          1.008, 1/1.008, 500_000, 500),
                PoolState(ts, "pool_dai_weth", DAI, WETH, 0, 0, 0,
                          1/3200.0, 3200.0, 300_000, 3000),
                PoolState(ts, "pool_usdt_usdc", USDT, USDC, 0, 0, 0,
                          1.007, 1/1.007, 400_000, 100),
                PoolState(ts, "pool_dai_usdt", DAI, USDT, 0, 0, 0,
                          1.001, 1/1.001, 200_000, 500),
            ]
        elif block_idx == 8:
            # 4-hop arb via WBTC
            pools = [
                PoolState(ts, "pool_weth_usdc", WETH, USDC, 0, 0, 0,
                          3200.0, 1/3200.0, 1_000_000, 3000),
                PoolState(ts, "pool_usdc_wbtc", USDC, WBTC, 0, 0, 0,
                          1/62000.0, 62000.0, 800_000, 3000),
                PoolState(ts, "pool_wbtc_dai", WBTC, DAI, 0, 0, 0,
                          62500.0, 1/62500.0, 600_000, 3000),
                PoolState(ts, "pool_dai_weth", DAI, WETH, 0, 0, 0,
                          1/3200.0, 3200.0, 300_000, 3000),
                PoolState(ts, "pool_usdt_usdc", USDT, USDC, 0, 0, 0,
                          1.0, 1.0, 400_000, 100),
            ]

        blocks.append((ts, pools))

    return blocks


# ── Core harness logic ────────────────────────────────────────────

def run_harness(
    blocks: list[tuple[int, list[PoolState]]],
    csv_path: Path = RESULTS_CSV,
    verbose: bool = True,
) -> list[ArbOpportunity]:
    """Run both searchers on every block and record results.

    Args:
        blocks: List of (timestamp, pool_states) tuples.
        csv_path: Path for CSV output.
        verbose: Print progress to stdout.

    Returns:
        All ArbOpportunity objects found across all blocks.
    """
    # Reset Latent Flux reservoir for clean start
    reset_reservoir()

    # Prepare CSV
    _write_csv_header(csv_path)

    all_results: list[ArbOpportunity] = []
    bf_total = 0
    lf_total = 0
    blocks_with_arb = 0

    if verbose:
        print(f"Running harness: {len(blocks)} blocks, input=${INPUT_USD:.0f}")
        print(f"Results → {csv_path}")
        print("=" * 70)

    for i, (ts, pool_states) in enumerate(blocks):
        # Run Bellman-Ford
        bf_opps = bf_find(pool_states, block_timestamp=ts, input_usd=INPUT_USD)

        # Run Latent Flux (reservoir maintains state across blocks)
        lf_opps = lf_find(pool_states, block_timestamp=ts, input_usd=INPUT_USD)

        # Record
        combined = bf_opps + lf_opps
        if combined:
            _append_csv_rows(csv_path, combined)
            blocks_with_arb += 1

        all_results.extend(combined)
        bf_total += len(bf_opps)
        lf_total += len(lf_opps)

        if verbose:
            bf_str = f"BF={len(bf_opps)}" if bf_opps else "BF=0"
            lf_str = f"LF={len(lf_opps)}" if lf_opps else "LF=0"
            top_bf = f"${max(o.gross_profit_usd for o in bf_opps):.2f}" if bf_opps else "-"
            top_lf = f"${max(o.gross_profit_usd for o in lf_opps):.2f}" if lf_opps else "-"
            print(f"  Block {i:>3d} (ts={ts}): {bf_str} {lf_str}  "
                  f"top: BF={top_bf} LF={top_lf}")

    if verbose:
        print("=" * 70)
        print(f"Done: {len(all_results)} total opportunities "
              f"(BF={bf_total}, LF={lf_total}) across {blocks_with_arb}/{len(blocks)} blocks")
        print(f"CSV written: {csv_path}")

    return all_results


# ── Live data harness ─────────────────────────────────────────────

def run_live(verbose: bool = True) -> list[ArbOpportunity]:
    """Run harness on live cached data from The Graph.

    Requires data_ingestion.py to have been run first (cached data in backtest/data/).
    Fails loudly if no cached data exists.
    """
    from backtest.data_ingestion import fetch_all, get_all_timestamps, get_snapshots_at_timestamp

    if verbose:
        print("ETH Arbitrage Backtest Harness — Live Data Mode")
        print("=" * 70)

    # Load cached data (fails loudly if not available)
    all_data = fetch_all(use_cache=True)

    if not all_data:
        raise RuntimeError(
            "No pool data available. Run `python backtest/data_ingestion.py` first "
            "to fetch and cache pool snapshots from The Graph."
        )

    # Get all timestamps and build blocks
    timestamps = get_all_timestamps(all_data)
    if not timestamps:
        raise RuntimeError("No valid timestamps found in cached data.")

    if verbose:
        print(f"Loaded {len(all_data)} pools, {len(timestamps)} timestamps")
        print(f"Time range: {timestamps[0]} → {timestamps[-1]}")

    blocks: list[tuple[int, list[PoolState]]] = []
    for ts in timestamps:
        pool_states = get_snapshots_at_timestamp(all_data, ts)
        if len(pool_states) >= 2:  # Need at least 2 pools for arbitrage graph
            blocks.append((ts, pool_states))

    if verbose:
        print(f"Blocks with ≥2 pools: {len(blocks)}")

    return run_harness(blocks, verbose=verbose)


# ── CLI entry point ───────────────────────────────────────────────

def main() -> int:
    """Run backtest harness from command line."""
    synthetic = "--synthetic" in sys.argv

    if synthetic:
        print("ETH Arbitrage Backtest Harness — Synthetic Data Mode")
        print("=" * 70)
        blocks = _build_synthetic_blocks()
        results = run_harness(blocks, verbose=True)
    else:
        try:
            results = run_live(verbose=True)
        except RuntimeError as e:
            print(f"\nFATAL: {e}", file=sys.stderr)
            print("\nTip: Run with --synthetic for offline testing.", file=sys.stderr)
            return 1

    # Generate report
    from backtest.report import generate_report
    generate_report(results, RESULTS_CSV, total_blocks=len(blocks))

    return 0


if __name__ == "__main__":
    sys.exit(main())
