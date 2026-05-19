"""Q-002 Z-score backtest calibration.

Converts the arbitrary 'Z >= 10 = T1_IMMINENT' threshold into an
evidence-based threshold with explicit precision and 95% Wilson CIs.

Methodology
-----------
For each historical (contract, day) pair where:
  - contract is in Q-002's candidate set (suspected/confirmed-tier with a
    watchlisted deployer)
  - that day had >= 10 approvals (matching production min_value)
  - the contract's history allows a 14-day baseline
  - at least 24h of post-day observation exists in the corpus

Compute:
  - Z-score against the 14-day trailing baseline (same formula as production)
  - Outcome: any drain_detected=1 within the 24h window starting at the
    eval-day midnight (= same-calendar-day-after-spike OR following day)

Aggregate (Z, outcome) pairs into bins and compute precision with Wilson
95% CIs. Also report cumulative precision at each candidate threshold K.

Caveats
-------
- Backtest is retrospective against the current snapshot. Future data may
  shift the curves.
- "Drain within 24h" includes same-day drains after the spike. The 0x80b12bd0
  case (May-9 spike + May-9 11:28 drain) is therefore a positive example.
- The candidate set is "as of NOW" — contracts added to the watchlist
  recently are evaluated as if they were always on the watchlist. This
  introduces look-ahead bias (we know which contracts ended up watchlisted).
  A future version should reconstruct the watchlist at each evaluation
  point in time.

CLI:
    python -m surveillance.analytics.backtest_q002_calibration
    python -m surveillance.analytics.backtest_q002_calibration --min-value 5
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from surveillance.analytics.approval_spike_detector import (
    compute_baseline, z_score, fetch_watchlisted_baits,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


def parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for proportion k/n. Robust to k=0 and k=n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_daily_counts(conn: sqlite3.Connection, contract: str) -> dict[str, int]:
    """Return {YYYY-MM-DD: approval_count} for a contract."""
    out: dict[str, int] = {}
    for r in conn.execute(
        "SELECT substr(approve_timestamp, 1, 10), COUNT(*) FROM approval_watchlist "
        "WHERE contract_address=? GROUP BY 1",
        (contract,)
    ):
        out[r[0]] = r[1]
    return out


def load_drain_times(conn: sqlite3.Connection, contract: str) -> list[datetime]:
    """Return sorted drain_timestamps for a contract."""
    out: list[datetime] = []
    for r in conn.execute(
        "SELECT drain_timestamp FROM approval_watchlist "
        "WHERE contract_address=? AND drain_detected=1 ORDER BY drain_timestamp",
        (contract,)
    ):
        dt = parse_ts(r[0])
        if dt:
            out.append(dt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-value", type=int, default=10,
                    help="minimum same-day approvals to evaluate (default 10)")
    ap.add_argument("--window-hours", type=int, default=24)
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    # Build candidate set
    print("  loading watchlisted-deployer contracts...", file=sys.stderr)
    baits = fetch_watchlisted_baits(conn)
    print(f"  {len(baits)} contracts in Q-002 candidate set", file=sys.stderr)

    # Determine corpus cutoff (drop trials with insufficient post-window data)
    max_ts = conn.execute(
        "SELECT MAX(approve_timestamp) FROM approval_watchlist"
    ).fetchone()[0]
    max_dt = parse_ts(max_ts) if max_ts else None
    if max_dt is None:
        print("No approval data found.", file=sys.stderr)
        return 1
    # Need at least `window_hours` of post-eval observation
    cutoff = max_dt - timedelta(hours=args.window_hours)
    print(f"  corpus latest = {max_dt.isoformat()}; cutoff for eval = {cutoff.isoformat()}",
          file=sys.stderr)

    # Iterate (contract, day) pairs
    trials: list[tuple[float, bool, str, str]] = []  # (z, drained, contract, date)
    n_skipped_too_recent = 0
    n_skipped_low_volume = 0
    n_skipped_no_baseline = 0

    for contract, chain, deployer, wl_label, tier in baits:
        daily = load_daily_counts(conn, contract)
        if not daily:
            continue
        drain_times = load_drain_times(conn, contract)

        for date_str, count in daily.items():
            if count < args.min_value:
                n_skipped_low_volume += 1
                continue
            eval_date = parse_ts(date_str + "T00:00:00+00:00")
            if eval_date is None:
                continue
            window_end = eval_date + timedelta(hours=args.window_hours)
            if window_end > cutoff:
                n_skipped_too_recent += 1
                continue

            mean, stddev, n_obs = compute_baseline(daily, eval_date, args.lookback_days)
            if n_obs == 0 and mean == 0:
                # No baseline at all — synthetic Z from first-ever event
                # smoothing. Still emits a positive Z. Keep it.
                pass

            z = z_score(count, mean, stddev, n_obs)
            # Outcome: any drain in [eval_date, eval_date + window_hours)
            drained = any(eval_date <= t < window_end for t in drain_times)
            trials.append((z, drained, contract, date_str))

    print(f"  trials evaluated: {len(trials)}", file=sys.stderr)
    print(f"  skipped (too recent for 24h obs): {n_skipped_too_recent}", file=sys.stderr)
    print(f"  skipped (below min_value): {n_skipped_low_volume}", file=sys.stderr)
    print()

    # Bin and compute P(drain | Z in bin)
    print("=" * 78)
    print("CALIBRATION CURVE: P(drain within 24h | Z in bin)")
    print("=" * 78)
    bins = [
        (0.0, 3.0, "Z < 3"),
        (3.0, 5.0, "3 <= Z < 5"),
        (5.0, 10.0, "5 <= Z < 10"),
        (10.0, 20.0, "10 <= Z < 20"),
        (20.0, 50.0, "20 <= Z < 50"),
        (50.0, 100.0, "50 <= Z < 100"),
        (100.0, 1e12, "Z >= 100"),
    ]
    print(f"  {'Z bin':14s}  {'trials':>7s}  {'drained':>7s}  {'P(drain)':>9s}  {'95% Wilson CI':>20s}")
    print("  " + "-" * 70)
    for low, high, lbl in bins:
        n = sum(1 for z, _, _, _ in trials if low <= z < high)
        k = sum(1 for z, d, _, _ in trials if low <= z < high and d)
        if n == 0:
            print(f"  {lbl:14s}  {n:>7}  {k:>7}  {'(no data)':>9s}")
            continue
        p = k / n
        lo, hi = wilson_ci(k, n)
        print(f"  {lbl:14s}  {n:>7}  {k:>7}  {p:>9.4f}  [{lo:.4f}, {hi:.4f}]")

    # Cumulative thresholds
    print()
    print("=" * 78)
    print("THRESHOLD ANALYSIS: P(drain within 24h | Z >= K)")
    print("=" * 78)
    print(f"  {'threshold K':>12s}  {'n_trials':>9s}  {'drained':>8s}  {'precision':>10s}  {'95% Wilson CI':>20s}")
    print("  " + "-" * 70)
    for K in [3, 5, 10, 15, 20, 30, 50, 100]:
        n = sum(1 for z, _, _, _ in trials if z >= K)
        k = sum(1 for z, d, _, _ in trials if z >= K and d)
        if n == 0:
            print(f"  Z >= {K:>4}     {n:>9}  {k:>8}  (no data)")
            continue
        p = k / n
        lo, hi = wilson_ci(k, n)
        # Format with a star if it crosses the natural "useful" threshold
        marker = "*" if lo >= 0.5 else " "
        print(f"  Z >= {K:>4}     {n:>9}  {k:>8}  {p:>10.4f}  [{lo:.4f}, {hi:.4f}] {marker}")
    print()
    print("  (* = lower bound of 95% CI is >= 50%, i.e., 'mostly precedes drain')")

    # Baseline rate
    n_all = len(trials)
    k_all = sum(1 for _, d, _, _ in trials if d)
    base_rate = k_all / n_all if n_all else 0
    bl_lo, bl_hi = wilson_ci(k_all, n_all)
    print()
    print(f"  Baseline drain rate (any Z, any contract, any day) = "
          f"{base_rate:.4f} [{bl_lo:.4f}, {bl_hi:.4f}]")

    # Current threshold implication
    K = 10
    n10 = sum(1 for z, _, _, _ in trials if z >= K)
    k10 = sum(1 for z, d, _, _ in trials if z >= K and d)
    if n10 > 0:
        print()
        print("=" * 78)
        print("CURRENT THRESHOLD (Z >= 10, the 'T1_IMMINENT' label)")
        print("=" * 78)
        p10 = k10 / n10
        lo, hi = wilson_ci(k10, n10)
        print(f"  Fires on:     {n10} (contract, day) trials")
        print(f"  True positives: {k10}  (drain occurred within 24h)")
        print(f"  Precision:    {p10:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}])")
        lift = p10 / base_rate if base_rate > 0 else float("inf")
        print(f"  Lift over base rate: {lift:.1f}x")

    # Show some specific high-Z trials
    print()
    print("=" * 78)
    print("HIGHEST-Z TRIALS (top 10 — sanity check)")
    print("=" * 78)
    trials_sorted = sorted(trials, key=lambda t: -t[0])[:10]
    print(f"  {'Z':>8s}  drained?  {'contract':44s}  date")
    for z, d, contract, date in trials_sorted:
        flag = "YES" if d else "no "
        print(f"  {z:>8.2f}  {flag}      {contract}  {date}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
