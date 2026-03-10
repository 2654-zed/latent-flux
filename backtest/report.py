"""Backtest Report — summary statistics and verdict for BF vs Latent Flux comparison.

Reads ArbOpportunity results from the harness and computes:
  - Hit rate per method (timestamps with ≥1 opportunity)
  - Exclusive hits (opportunities found by one method but not the other)
  - Profit distribution stats (mean, median, max, std)
  - Overlap analysis (same cycle found by both methods)
  - Verdict: does Latent Flux surface anything Bellman-Ford misses?

Writes summary to backtest/results/summary.txt.

Usage:
    from backtest.report import generate_report
    generate_report(results, csv_path)
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.bellman_ford import ArbOpportunity

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.txt"


# ── Analysis helpers ──────────────────────────────────────────────

def _median(values: list[float]) -> float:
    """Compute median of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _std(values: list[float], mean: float) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _cycle_key(opp: ArbOpportunity) -> str:
    """Canonical key for a cycle: sorted pool path, direction-invariant."""
    # Normalize: use the token sequence but canonicalize rotation
    # "A → B → C → A" and "B → C → A → B" are the same cycle
    if len(opp.path) < 2:
        return " → ".join(opp.path)
    # Remove the closing token (same as start)
    tokens = opp.path[:-1] if opp.path[-1] == opp.path[0] else opp.path
    # Canonical rotation: start from lexicographically smallest token
    if tokens:
        min_idx = tokens.index(min(tokens))
        tokens = tokens[min_idx:] + tokens[:min_idx]
    return " → ".join(tokens)


# ── Main report generation ────────────────────────────────────────

@dataclass
class MethodStats:
    """Aggregate stats for one method."""
    name: str
    total_opps: int
    unique_cycles: int
    blocks_with_arb: int
    total_blocks: int
    hit_rate: float
    profits: list[float]
    mean_profit: float
    median_profit: float
    max_profit: float
    std_profit: float
    exclusive_opps: int        # found by this method but not the other
    exclusive_cycles: set       # cycles exclusive to this method


def _compute_method_stats(
    name: str,
    opps: list[ArbOpportunity],
    total_blocks: int,
    other_cycle_keys: set[str],
) -> MethodStats:
    """Compute aggregate statistics for one method's results."""
    profits = [o.gross_profit_usd for o in opps]
    blocks = set(o.block_timestamp for o in opps)
    cycles = set(_cycle_key(o) for o in opps)
    exclusive = cycles - other_cycle_keys

    mean_p = sum(profits) / len(profits) if profits else 0.0
    return MethodStats(
        name=name,
        total_opps=len(opps),
        unique_cycles=len(cycles),
        blocks_with_arb=len(blocks),
        total_blocks=total_blocks,
        hit_rate=len(blocks) / total_blocks if total_blocks > 0 else 0.0,
        profits=profits,
        mean_profit=mean_p,
        median_profit=_median(profits),
        max_profit=max(profits) if profits else 0.0,
        std_profit=_std(profits, mean_p),
        exclusive_opps=sum(1 for o in opps if _cycle_key(o) in exclusive),
        exclusive_cycles=exclusive,
    )


def generate_report(
    results: list[ArbOpportunity],
    csv_path: Path | None = None,
    output_path: Path = SUMMARY_PATH,
    verbose: bool = True,
    total_blocks: int | None = None,
) -> str:
    """Generate summary report from harness results.

    Args:
        results: All ArbOpportunity objects from the harness.
        csv_path: Path to CSV file (for reference in report).
        output_path: Where to write the summary text.
        verbose: Print report to stdout.
        total_blocks: Total number of blocks analyzed (including those with 0 opps).

    Returns:
        The report text.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split by method
    bf_opps = [o for o in results if o.method == "bellman_ford"]
    lf_opps = [o for o in results if o.method == "latent_flux"]

    # Total blocks analyzed
    all_timestamps = set(o.block_timestamp for o in results)
    if total_blocks is None:
        if results:
            min_ts = min(o.block_timestamp for o in results)
            max_ts = max(o.block_timestamp for o in results)
            total_blocks = max(len(all_timestamps), (max_ts - min_ts) // 3600 + 1)
        else:
            total_blocks = 0

    # Compute cycle keys for overlap analysis
    bf_cycle_keys = set(_cycle_key(o) for o in bf_opps)
    lf_cycle_keys = set(_cycle_key(o) for o in lf_opps)
    overlap_cycles = bf_cycle_keys & lf_cycle_keys

    bf_stats = _compute_method_stats("Bellman-Ford", bf_opps, total_blocks, lf_cycle_keys)
    lf_stats = _compute_method_stats("Latent Flux", lf_opps, total_blocks, bf_cycle_keys)

    # ── Build report text ─────────────────────────────────────────
    lines: list[str] = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("ETH DEX ARBITRAGE BACKTEST — SUMMARY REPORT")
    lines.append(sep)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total blocks analyzed:    {total_blocks}")
    lines.append(f"  Blocks with any arb:      {len(all_timestamps)}")
    lines.append(f"  Total opportunities:      {len(results)}")
    lines.append(f"  Input size per trade:     ${10000.0:.0f}")
    if csv_path:
        lines.append(f"  Results CSV:              {csv_path}")
    lines.append("")

    # Per-method stats
    for stats in (bf_stats, lf_stats):
        lines.append(f"{stats.name.upper()}")
        lines.append("-" * 40)
        lines.append(f"  Opportunities found:      {stats.total_opps}")
        lines.append(f"  Unique cycles:            {stats.unique_cycles}")
        lines.append(f"  Blocks with arb:          {stats.blocks_with_arb} / {stats.total_blocks}")
        lines.append(f"  Hit rate:                 {stats.hit_rate:.1%}")
        if stats.profits:
            lines.append(f"  Mean profit:              ${stats.mean_profit:.2f}")
            lines.append(f"  Median profit:            ${stats.median_profit:.2f}")
            lines.append(f"  Max profit:               ${stats.max_profit:.2f}")
            lines.append(f"  Std profit:               ${stats.std_profit:.2f}")
        else:
            lines.append(f"  (no profitable opportunities found)")
        lines.append(f"  Exclusive opportunities:  {stats.exclusive_opps}")
        lines.append(f"  Exclusive cycles:         {len(stats.exclusive_cycles)}")
        if stats.exclusive_cycles:
            for cyc in sorted(stats.exclusive_cycles):
                lines.append(f"    • {cyc}")
        lines.append("")

    # Overlap analysis
    lines.append("OVERLAP ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"  Cycles found by both:     {len(overlap_cycles)}")
    lines.append(f"  BF-only cycles:           {len(bf_stats.exclusive_cycles)}")
    lines.append(f"  LF-only cycles:           {len(lf_stats.exclusive_cycles)}")
    if overlap_cycles:
        lines.append(f"  Shared cycles:")
        for cyc in sorted(overlap_cycles):
            # Compare profits on shared cycles
            bf_profit = max(
                (o.gross_profit_usd for o in bf_opps if _cycle_key(o) == cyc),
                default=0.0,
            )
            lf_profit = max(
                (o.gross_profit_usd for o in lf_opps if _cycle_key(o) == cyc),
                default=0.0,
            )
            lines.append(f"    • {cyc}: BF=${bf_profit:.2f} LF=${lf_profit:.2f}")
    lines.append("")

    # Per-block detail
    lines.append("PER-BLOCK DETAIL")
    lines.append("-" * 40)
    block_opps: dict[int, list[ArbOpportunity]] = defaultdict(list)
    for o in results:
        block_opps[o.block_timestamp].append(o)

    for ts in sorted(block_opps.keys()):
        opps = block_opps[ts]
        bf_count = sum(1 for o in opps if o.method == "bellman_ford")
        lf_count = sum(1 for o in opps if o.method == "latent_flux")
        bf_max = max((o.gross_profit_usd for o in opps if o.method == "bellman_ford"), default=0)
        lf_max = max((o.gross_profit_usd for o in opps if o.method == "latent_flux"), default=0)
        flag = ""
        if lf_count > bf_count:
            flag = " ← LF found more"
        elif bf_count > lf_count:
            flag = " ← BF found more"
        lines.append(f"  ts={ts}: BF={bf_count} (${bf_max:.2f})  "
                     f"LF={lf_count} (${lf_max:.2f}){flag}")
    lines.append("")

    # Verdict
    lines.append(sep)
    lines.append("VERDICT")
    lines.append(sep)

    if not results:
        verdict = ("NO DATA: No opportunities found by either method. "
                   "This could indicate fair market prices or insufficient data.")
    elif len(lf_stats.exclusive_cycles) > 0 and len(bf_stats.exclusive_cycles) == 0:
        verdict = (
            f"LATENT FLUX WINS: Found {len(lf_stats.exclusive_cycles)} exclusive cycle(s) "
            f"that Bellman-Ford missed, while BF found nothing exclusive. "
            f"The geometric approach surfaces structurally different arbitrage paths."
        )
    elif len(lf_stats.exclusive_cycles) > len(bf_stats.exclusive_cycles):
        verdict = (
            f"LATENT FLUX ADVANTAGE: Found {len(lf_stats.exclusive_cycles)} exclusive cycles "
            f"vs BF's {len(bf_stats.exclusive_cycles)}. The geometric search discovers "
            f"paths that classical negative-cycle detection misses."
        )
    elif len(lf_stats.exclusive_cycles) == len(bf_stats.exclusive_cycles) and len(lf_stats.exclusive_cycles) > 0:
        verdict = (
            f"MIXED: Both methods found {len(lf_stats.exclusive_cycles)} exclusive cycle(s). "
            f"They are complementary — each surfaces paths the other misses."
        )
    elif len(bf_stats.exclusive_cycles) > len(lf_stats.exclusive_cycles):
        verdict = (
            f"BELLMAN-FORD ADVANTAGE: BF found {len(bf_stats.exclusive_cycles)} exclusive cycles "
            f"vs LF's {len(lf_stats.exclusive_cycles)}. Classical search outperforms geometric on this data."
        )
    else:
        verdict = (
            "EQUIVALENT: Both methods found the same cycles. No evidence that "
            "the geometric approach surfaces additional arbitrage paths."
        )

    lines.append(f"  {verdict}")
    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    if verbose:
        print()
        print(report_text)

    return report_text
