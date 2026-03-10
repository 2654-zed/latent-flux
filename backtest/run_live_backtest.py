"""Live Backtest Harness — runs BF vs Latent Flux on historical Uniswap V3 data.

Loads uniswap_v3_30d.json (hourly pool snapshots), converts to PoolState
objects, groups by timestamp, and runs both arbitrage detectors side-by-side.

Usage:
    python backtest/run_live_backtest.py                # EMA reservoir (baseline)
    python backtest/run_live_backtest.py --use-kalman   # UKF reservoir (Phase 1)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

# Ensure repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.bellman_ford import ArbOpportunity, find_arbitrage_opportunities as bf_find
from backtest.latent_flux_searcher import (
    find_arbitrage_opportunities as lf_find,
    get_signal_metadata,
    reset_reservoir,
)
from backtest.data_ingestion import PoolState
from backtest.report import generate_report

# ── Configuration ─────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent / "data" / "uniswap_v3_30d.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
INPUT_USD = 10_000.0


# ── JSON → PoolState conversion ──────────────────────────────────

def load_json_data(path: Path = DATA_FILE) -> tuple[dict, list[dict]]:
    """Load and return (meta, hour_data) from the JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Place uniswap_v3_30d.json in backtest/data/"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["meta"], raw["hour_data"]


def _build_pool_metadata(meta: dict) -> dict[str, dict]:
    """Build pool_id → {token0, token1, fee_tier} lookup from JSON meta."""
    lookup = {}
    for pool in meta["pools"]:
        lookup[pool["id"]] = {
            "token0": pool["token0"],
            "token1": pool["token1"],
            "fee_tier": int(pool["fee_tier"]),
        }
    return lookup


def parse_hour_data(meta: dict, hour_data: list[dict]) -> list[PoolState]:
    """Convert raw JSON hour_data records into PoolState objects."""
    pool_meta = _build_pool_metadata(meta)
    states = []

    for rec in hour_data:
        pool_id = rec["pool_id"]
        pm = pool_meta.get(pool_id)
        if pm is None:
            continue

        token0_price = float(rec["token0Price"])
        token1_price = float(rec["token1Price"])

        states.append(PoolState(
            block_timestamp=int(rec["periodStartUnix"]),
            pool_address=pool_id,
            token0=pm["token0"],
            token1=pm["token1"],
            sqrt_price=float(rec.get("sqrtPrice", 0)),
            liquidity=float(rec["liquidity"]),
            tick=int(rec["tick"]),
            token0_price=token0_price,
            token1_price=token1_price,
            volume_usd=float(rec["volumeUSD"]),
            fee_tier=pm["fee_tier"],
        ))

    return states


def group_by_timestamp(states: list[PoolState]) -> list[tuple[int, list[PoolState]]]:
    """Group pool states by block_timestamp, sorted chronologically."""
    by_ts: dict[int, list[PoolState]] = defaultdict(list)
    for s in states:
        by_ts[s.block_timestamp].append(s)

    return sorted(by_ts.items(), key=lambda x: x[0])


# ── CSV output ────────────────────────────────────────────────────

_CSV_FIELDS = [f.name for f in fields(ArbOpportunity)]


def _write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDS)


def _append_csv_rows(path: Path, opportunities: list[ArbOpportunity]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for opp in opportunities:
            writer.writerow([
                opp.method,
                " → ".join(opp.path),
                opp.hop_count,
                opp.gross_profit_usd,
                opp.input_size_usd,
                opp.iterations,
                opp.block_timestamp,
            ])


# ── Lead/lag analysis ─────────────────────────────────────────────

def _analyze_lead_lag(all_results: list[ArbOpportunity]) -> dict:
    """Compute which method detects opportunities first at each timestamp.

    For each cycle (normalized), find the earliest timestamp each method
    detected it. If LF detects it earlier → LF leads. If BF → BF leads.
    """
    from backtest.report import _cycle_key

    # cycle_key → {method → earliest_timestamp}
    first_seen: dict[str, dict[str, int]] = defaultdict(dict)
    for opp in all_results:
        key = _cycle_key(opp)
        method = opp.method
        ts = opp.block_timestamp
        if method not in first_seen[key] or ts < first_seen[key][method]:
            first_seen[key][method] = ts

    leads = {"lf_leads": 0, "bf_leads": 0, "simultaneous": 0, "lf_exclusive": 0, "bf_exclusive": 0}
    for key, methods in first_seen.items():
        bf_ts = methods.get("bellman_ford")
        lf_ts = methods.get("latent_flux")
        if bf_ts is not None and lf_ts is not None:
            if lf_ts < bf_ts:
                leads["lf_leads"] += 1
            elif bf_ts < lf_ts:
                leads["bf_leads"] += 1
            else:
                leads["simultaneous"] += 1
        elif lf_ts is not None:
            leads["lf_exclusive"] += 1
        elif bf_ts is not None:
            leads["bf_exclusive"] += 1

    total = sum(leads.values())
    leads["total_unique_cycles"] = total
    if total > 0:
        leads["lf_lead_rate"] = (leads["lf_leads"] + leads["lf_exclusive"]) / total
    else:
        leads["lf_lead_rate"] = 0.0

    return leads


# ── Core backtest runner ──────────────────────────────────────────

def run_backtest(
    blocks: list[tuple[int, list[PoolState]]],
    csv_path: Path = RESULTS_CSV,
    use_kalman: bool = False,
    verbose: bool = True,
) -> list[ArbOpportunity]:
    """Run both searchers on every hourly block.

    Args:
        blocks: List of (timestamp, pool_states) tuples.
        csv_path: Path for CSV output.
        use_kalman: If True, use KalmanReservoir instead of EMA.
        verbose: Print progress.

    Returns:
        All ArbOpportunity objects found.
    """
    reset_reservoir()
    _write_csv_header(csv_path)

    all_results: list[ArbOpportunity] = []
    bf_total = 0
    lf_total = 0
    blocks_with_arb = 0

    t0 = time.time()

    if verbose:
        mode = "KalmanReservoir (UKF)" if use_kalman else "EMA Reservoir (baseline)"
        print(f"Running backtest: {len(blocks)} blocks, input=${INPUT_USD:.0f}, mode={mode}")
        print(f"Results → {csv_path}")
        print("=" * 70)

    # Every 50 blocks, print a progress line
    for i, (ts, pool_states) in enumerate(blocks):
        bf_opps = bf_find(pool_states, block_timestamp=ts, input_usd=INPUT_USD)
        lf_opps = lf_find(pool_states, block_timestamp=ts, input_usd=INPUT_USD,
                          use_kalman=use_kalman)

        combined = bf_opps + lf_opps
        if combined:
            _append_csv_rows(csv_path, combined)
            blocks_with_arb += 1

        all_results.extend(combined)
        bf_total += len(bf_opps)
        lf_total += len(lf_opps)

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Block {i+1:>4d}/{len(blocks)}  "
                  f"BF={bf_total} LF={lf_total}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0

    if verbose:
        print("=" * 70)
        print(f"Done: {len(all_results)} total opportunities "
              f"(BF={bf_total}, LF={lf_total}) "
              f"across {blocks_with_arb}/{len(blocks)} blocks "
              f"in {elapsed:.1f}s")
        print(f"CSV: {csv_path}")

    return all_results


# ── Extended report ───────────────────────────────────────────────

def print_extended_report(
    results: list[ArbOpportunity],
    total_blocks: int,
    csv_path: Path = RESULTS_CSV,
) -> None:
    """Generate standard report + lead/lag analysis."""
    # Standard report
    report_text = generate_report(
        results, csv_path=csv_path, total_blocks=total_blocks, verbose=True
    )

    # Lead/lag analysis
    leads = _analyze_lead_lag(results)
    lines = [
        "",
        "=" * 70,
        "LEAD/LAG ANALYSIS",
        "=" * 70,
        f"  Unique cycles detected:     {leads['total_unique_cycles']}",
        f"  LF leads (detected first):  {leads['lf_leads']}",
        f"  BF leads (detected first):  {leads['bf_leads']}",
        f"  Simultaneous:               {leads['simultaneous']}",
        f"  LF exclusive:               {leads['lf_exclusive']}",
        f"  BF exclusive:               {leads['bf_exclusive']}",
        f"  LF lead rate:               {leads['lf_lead_rate']:.1%}",
        "",
    ]
    lead_text = "\n".join(lines)
    print(lead_text)

    # Append to summary file
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(lead_text)

    # Deviation distribution
    _print_deviation_distribution(results)

    # S-score distribution (only populated when --use-kalman)
    _print_s_score_distribution(results)

    # Executable opportunities (net profit > 0)
    _print_executable_opportunities(results)

    # High-confidence lead rate
    _print_high_confidence_lead_rate(results)


def _print_deviation_distribution(results: list[ArbOpportunity]) -> None:
    """Print histogram of signal deviations (gross_profit as % of input)."""
    lf_opps = [o for o in results if o.method == "latent_flux"]
    if not lf_opps:
        return

    deviations = [o.gross_profit_usd / o.input_size_usd * 100.0 for o in lf_opps]

    buckets = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, float("inf")]
    counts = [0] * len(buckets)
    for d in deviations:
        for j, b in enumerate(buckets):
            if d < b:
                counts[j] += 1
                break

    print("LF SIGNAL DEVIATION DISTRIBUTION")
    print("-" * 40)
    labels = ["<0.01%", "<0.02%", "<0.05%", "<0.10%", "<0.15%", "<0.20%", "<0.50%", "<1.00%", "≥1.00%"]
    for label, count in zip(labels, counts):
        bar = "█" * min(count, 60)
        print(f"  {label:>8s}: {count:>4d}  {bar}")
    print(f"  Total: {len(deviations)}")
    print()


def _print_s_score_distribution(results: list[ArbOpportunity]) -> None:
    """Print S-score distribution for LF signals (only meaningful with --use-kalman)."""
    meta = get_signal_metadata()
    if not meta:
        return

    lf_opps = [o for o in results if o.method == "latent_flux"]
    if not lf_opps:
        return

    s_scores = []
    kappas = []
    valid_count = 0
    for opp in lf_opps:
        path_str = "→".join(opp.path)
        key = (opp.block_timestamp, path_str)
        sm = meta.get(key)
        if sm is None:
            continue
        s_scores.append(sm.s_score)
        kappas.append(sm.mean_reversion_speed)
        if sm.s_score_valid:
            valid_count += 1

    if not s_scores:
        return

    import numpy as np
    s_arr = np.array(s_scores)
    k_arr = np.array(kappas)
    abs_s = np.abs(s_arr)

    # S-score buckets
    s_buckets = [0.5, 1.0, 1.5, 2.0, 3.0, float("inf")]
    s_labels = ["|S|<0.5", "|S|<1.0", "|S|<1.5", "|S|<2.0", "|S|<3.0", "|S|≥3.0"]
    s_counts = [0] * len(s_buckets)
    for s in abs_s:
        for j, b in enumerate(s_buckets):
            if s < b:
                s_counts[j] += 1
                break

    print("S-SCORE DISTRIBUTION (UKF + OU estimation)")
    print("-" * 50)
    for label, count in zip(s_labels, s_counts):
        bar = "█" * min(count, 60)
        print(f"  {label:>10s}: {count:>4d}  {bar}")
    print(f"  Valid S-scores: {valid_count}/{len(s_scores)}")
    print(f"  Mean |S|: {float(abs_s.mean()):.4f}")
    print(f"  Max  |S|: {float(abs_s.max()):.4f}")
    print(f"  Mean κ:   {float(k_arr[k_arr > 0].mean()) if (k_arr > 0).any() else 0:.4f}")
    print(f"  Signals with |S|>1: {int((abs_s > 1.0).sum())}")
    print(f"  Signals with |S|>2: {int((abs_s > 2.0).sum())}")
    print()

    # Regime change summary
    regime_count = sum(1 for sm in meta.values() if sm.regime_change)
    if regime_count > 0:
        print(f"  Signals during regime change: {regime_count}/{len(meta)}")
        print()


def _print_executable_opportunities(results: list[ArbOpportunity]) -> None:
    """Print summary of net-profitable signals (gross - gas - tip > 0)."""
    meta = get_signal_metadata()
    if not meta:
        return

    lf_opps = [o for o in results if o.method == "latent_flux"]
    if not lf_opps:
        return

    net_positive = []
    for opp in lf_opps:
        path_str = "→".join(opp.path)
        key = (opp.block_timestamp, path_str)
        sm = meta.get(key)
        if sm and sm.net_profit_usd > 0:
            net_positive.append((opp, sm))

    print("EXECUTABLE OPPORTUNITIES (net_profit > 0)")
    print("-" * 50)
    print(f"  Net-positive signals: {len(net_positive)}/{len(lf_opps)}")
    if net_positive:
        profits = [sm.net_profit_usd for _, sm in net_positive]
        print(f"  Mean net profit:  ${sum(profits)/len(profits):.2f}")
        print(f"  Max net profit:   ${max(profits):.2f}")
        print(f"  Total net profit: ${sum(profits):.2f}")
        # Show fee thresholds
        thresholds = [sm.fee_weighted_threshold * 100 for _, sm in net_positive]
        print(f"  Mean fee threshold: {sum(thresholds)/len(thresholds):.3f}%")
        # Show top 5
        net_positive.sort(key=lambda x: x[1].net_profit_usd, reverse=True)
        print(f"  Top opportunities:")
        for opp, sm in net_positive[:5]:
            print(f"    ts={opp.block_timestamp}: gross=${opp.gross_profit_usd:.2f} "
                  f"net=${sm.net_profit_usd:.2f} fee_thr={sm.fee_weighted_threshold*100:.3f}% "
                  f"{'->'.join(opp.path)}")
    print()


def _print_high_confidence_lead_rate(results: list[ArbOpportunity]) -> None:
    """Separate lead rate for high-confidence signals (scale_agreement > 0.7)."""
    meta = get_signal_metadata()
    if not meta:
        return

    # Filter to high-confidence LF signals
    lf_opps = [o for o in results if o.method == "latent_flux"]
    hc_timestamps: set[int] = set()
    for opp in lf_opps:
        path_str = "→".join(opp.path)
        key = (opp.block_timestamp, path_str)
        sm = meta.get(key)
        if sm and sm.scale_agreement > 0.7:
            hc_timestamps.add(opp.block_timestamp)

    if not hc_timestamps:
        print("HIGH-CONFIDENCE SIGNALS (scale_agreement > 0.7)")
        print("-" * 50)
        print("  No high-confidence signals found.")
        # Show distribution of scale_agreement
        agreements = [sm.scale_agreement for sm in meta.values()]
        if agreements:
            import numpy as np
            arr = np.array(agreements)
            print(f"  Scale agreement: mean={arr.mean():.3f} max={arr.max():.3f}")
        print()
        return

    # Among high-confidence LF signals, compute lead rate vs BF
    bf_timestamps: dict[int, set[str]] = {}
    lf_timestamps: dict[int, set[str]] = {}
    for opp in results:
        key = "->".join(sorted(opp.path))
        if opp.method == "bellman_ford":
            bf_timestamps.setdefault(opp.block_timestamp, set()).add(key)
        elif opp.method == "latent_flux" and opp.block_timestamp in hc_timestamps:
            lf_timestamps.setdefault(opp.block_timestamp, set()).add(key)

    # Count blocks where HC-LF found something BF didn't (at same timestamp)
    hc_lf_only = 0
    hc_both = 0
    for ts in sorted(hc_timestamps):
        lf_cycles = lf_timestamps.get(ts, set())
        bf_cycles = bf_timestamps.get(ts, set())
        if lf_cycles - bf_cycles:
            hc_lf_only += 1
        if lf_cycles & bf_cycles:
            hc_both += 1

    total_hc = len(hc_timestamps)
    hc_lead_rate = hc_lf_only / total_hc if total_hc > 0 else 0

    print("HIGH-CONFIDENCE SIGNALS (scale_agreement > 0.7)")
    print("-" * 50)
    print(f"  High-confidence blocks: {total_hc}")
    print(f"  HC-LF exclusive blocks: {hc_lf_only}")
    print(f"  HC-LF + BF overlap:     {hc_both}")
    print(f"  HC lead rate:           {hc_lead_rate:.1%}")
    # Show agreement distribution
    agreements = [sm.scale_agreement for sm in meta.values()]
    if agreements:
        import numpy as np
        arr = np.array(agreements)
        print(f"  Scale agreement: mean={arr.mean():.3f} max={arr.max():.3f}")
    print()


# ── CLI entry point ───────────────────────────────────────────────

def main() -> int:
    # Force UTF-8 output on Windows to handle special chars
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    use_kalman = "--use-kalman" in sys.argv

    print("ETH Arbitrage Backtest — Live Data")
    print("=" * 70)

    # Load data
    try:
        meta, hour_data = load_json_data()
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    print(f"Loaded: {meta['total_records']} records, "
          f"{len(meta['pools'])} pools, "
          f"{meta['start']} -> {meta['end']}")

    # Parse and group
    all_states = parse_hour_data(meta, hour_data)
    blocks = group_by_timestamp(all_states)

    print(f"Parsed: {len(all_states)} PoolState objects, {len(blocks)} hourly blocks")

    # Sanity check: pools per typical block
    if blocks:
        sample_ts, sample_pools = blocks[len(blocks) // 2]
        print(f"Sample block (ts={sample_ts}): {len(sample_pools)} pools")

    # Run backtest
    results = run_backtest(blocks, use_kalman=use_kalman, verbose=True)

    # Extended report
    print_extended_report(results, total_blocks=len(blocks))

    return 0


if __name__ == "__main__":
    sys.exit(main())
