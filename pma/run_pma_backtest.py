"""PMA Backtest — run prediction market arbitrage scanner on historical data.

Equivalent to backtest/run_live_backtest.py. Loads price data, runs
pma_searcher.scan() across all timestamps, prints results.

Usage:
    python pma/run_pma_backtest.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pma.market_feed import MarketFeed
from pma.reservoir_tracker import PMAReservoirTracker
from pma.pma_searcher import PMASignal, scan


def _ts_to_str(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def run_backtest() -> list[PMASignal]:
    """Run PMA backtest, return all signals."""
    feed = MarketFeed(mode="proxy")
    tracker = PMAReservoirTracker()

    timestamps = feed.all_timestamps()
    if not timestamps:
        print("No data loaded. Place pma_prices.json or sample_pma_prices.json in pma/data/")
        return []

    all_signals: list[PMASignal] = []

    for ts in timestamps:
        signals = scan(ts, feed, tracker)
        all_signals.extend(signals)

    return all_signals


def print_report(signals: list[PMASignal]) -> None:
    """Print full PMA backtest report."""
    total_hours = len({s.timestamp for s in signals})
    viable = [s for s in signals if s.viable]

    # Spread distribution
    spread_lt1 = sum(1 for s in signals if s.spread < 0.01)
    spread_1_3 = sum(1 for s in signals if 0.01 <= s.spread < 0.03)
    spread_3_5 = sum(1 for s in signals if 0.03 <= s.spread < 0.05)
    spread_gt5 = sum(1 for s in signals if s.spread >= 0.05)

    print("=" * 70)
    print("PMA BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Total market-hours scanned:        {len(signals)}")
    print(f"  Unique timestamps:                 {total_hours}")
    print(f"  Total signals detected:            {len(signals)}")
    print(f"  Viable arb opportunities:          {len(viable)}  "
          f"(arb_return > 0 after fees)")
    print()
    print("  Spread distribution:")
    print(f"    < 1%:   {spread_lt1}")
    print(f"    1-3%:   {spread_1_3}")
    print(f"    3-5%:   {spread_3_5}")
    print(f"    > 5%:   {spread_gt5}")
    print()

    # Top 10 opportunities by arb_return
    top = sorted(signals, key=lambda s: s.arb_return, reverse=True)[:10]
    print("  Top 10 opportunities:")
    print(f"  {'Market':<30s} {'Poly':>6s} {'Kalshi':>7s} {'Spread':>7s} "
          f"{'Net Ret':>8s} {'mDev':>7s} {'S-scr':>6s}")
    print("  " + "-" * 73)
    for s in top:
        desc = s.description[:28] if len(s.description) > 28 else s.description
        print(f"  {desc:<30s} {s.poly_yes:>6.1%} {s.kalshi_yes:>6.1%} "
              f"{s.spread:>6.1%} {s.arb_return:>+7.1%} "
              f"{s.manifold_deviation:>+6.1%} {s.s_score:>6.2f}")
    print()

    # Viable by market
    print("  Viable opportunities by market:")
    by_market: dict[str, int] = {}
    for s in viable:
        by_market[s.market_id] = by_market.get(s.market_id, 0) + 1
    for mid, count in sorted(by_market.items(), key=lambda x: -x[1]):
        print(f"    {mid:<35s} {count:>4d}")
    print()

    # Side distribution
    buy_poly = sum(1 for s in viable if s.side == "buy_poly_sell_kalshi")
    buy_kalshi = sum(1 for s in viable if s.side == "buy_kalshi_sell_poly")
    print(f"  Side split (viable):  buy_poly={buy_poly}  buy_kalshi={buy_kalshi}")
    print()

    _print_temporal_lead_lag(signals, viable)


def _print_temporal_lead_lag(
    all_signals: list[PMASignal],
    viable: list[PMASignal],
) -> None:
    """Does reservoir deviation precede viable spreads by 1-3 hours?

    A 'qualifying deviation' is |manifold_deviation| > 0.5 (spread is
    50%+ above historical mean) at a timestamp where the spread is NOT
    yet viable. If a viable spread appears 1-3 hours later for the same
    market, the deviation "led" the opportunity.
    """
    if not viable:
        print("  (No viable signals — skipping temporal analysis)")
        return

    HOUR = 3600
    lag_hours = [1, 2, 3]

    # Index: (market_id, timestamp) → signal
    sig_index: dict[tuple[str, int], PMASignal] = {}
    for s in all_signals:
        sig_index[(s.market_id, s.timestamp)] = s

    # Viable timestamps per market
    viable_by_market: dict[str, set[int]] = {}
    for s in viable:
        viable_by_market.setdefault(s.market_id, set()).add(s.timestamp)

    # Qualifying deviations: high manifold_deviation at non-viable timestamps
    qualifying: list[PMASignal] = [
        s for s in all_signals
        if abs(s.manifold_deviation) > 0.5 and not s.viable
    ]

    print("=" * 70)
    print("TEMPORAL LEAD/LAG ANALYSIS")
    print("=" * 70)
    print(f"  Viable signals:                    {len(viable)}")
    print(f"  Qualifying deviations (|mDev|>0.5, non-viable): {len(qualifying)}")
    print()

    # For each lag: how many viable signals were preceded by a qualifying deviation?
    lag_stats: dict[int, dict] = {}
    examples: list[tuple[int, PMASignal, PMASignal]] = []  # (lag_h, deviation_sig, viable_sig)

    for lag_h in lag_hours:
        preceded = 0
        for s in viable:
            lookback_ts = s.timestamp - lag_h * HOUR
            prior = sig_index.get((s.market_id, lookback_ts))
            if prior and abs(prior.manifold_deviation) > 0.5 and not prior.viable:
                preceded += 1
                examples.append((lag_h, prior, s))
        rate = preceded / len(viable) if viable else 0
        lag_stats[lag_h] = {"preceded": preceded, "total": len(viable), "rate": rate}

    # Cumulative
    viable_with_prior: set[tuple[str, int]] = set()
    for lag_h in lag_hours:
        for s in viable:
            lookback_ts = s.timestamp - lag_h * HOUR
            prior = sig_index.get((s.market_id, lookback_ts))
            if prior and abs(prior.manifold_deviation) > 0.5 and not prior.viable:
                viable_with_prior.add((s.market_id, s.timestamp))
    cum_rate = len(viable_with_prior) / len(viable) if viable else 0

    # False lead rate: qualifying deviations NOT followed by viable in 1-3h
    false_leads = 0
    for s in qualifying:
        followed = False
        mkt_viable = viable_by_market.get(s.market_id, set())
        for lag_h in lag_hours:
            if (s.timestamp + lag_h * HOUR) in mkt_viable:
                followed = True
                break
        if not followed:
            false_leads += 1
    false_rate = false_leads / len(qualifying) if qualifying else 0

    print("  Per-lag window:")
    for lag_h in lag_hours:
        st = lag_stats[lag_h]
        marker = " <-" if st["rate"] > 0.6 else ""
        print(f"    {lag_h}h lookback: {st['preceded']:>3d}/{st['total']} viable preceded "
              f"(lead rate: {st['rate']:.1%}){marker}")
    print(f"  Cumulative (any 1-3h):   {len(viable_with_prior)}/{len(viable)} "
          f"({cum_rate:.1%})")
    print(f"  False lead rate:         {false_leads}/{len(qualifying)} "
          f"({false_rate:.1%})")
    print()

    # Top 10 examples
    if examples:
        examples.sort(key=lambda x: x[2].arb_return, reverse=True)
        top = examples[:10]
        print("  Top 10 examples (deviation -> viable spread):")
        print(f"  {'Lag':>3s}  {'Dev ts':<17s} {'mDev':>7s} {'S-scr':>6s}  "
              f"{'Viable ts':<17s} {'Spread':>7s} {'Net':>7s}  Market")
        print("  " + "-" * 80)
        for lag_h, dev_sig, via_sig in top:
            print(f"  {lag_h:>2d}h  {_ts_to_str(dev_sig.timestamp):<17s} "
                  f"{dev_sig.manifold_deviation:>+6.1%} {dev_sig.s_score:>6.2f}  "
                  f"{_ts_to_str(via_sig.timestamp):<17s} {via_sig.spread:>6.1%} "
                  f"{via_sig.arb_return:>+6.1%}  {via_sig.market_id}")
    print()


def main() -> int:
    signals = run_backtest()
    if not signals:
        return 1
    print_report(signals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
