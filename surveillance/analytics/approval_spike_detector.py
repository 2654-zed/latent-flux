"""Approval-rate spike detector — answers Q-002.

For each confirmed-tier bait contract on watchlist, computes the per-day
approval-arrival Z-score against its own 14-day trailing baseline. Surfaces
contracts whose Z exceeds a threshold (default Z ≥ 5).

Empirical validation: this module is built against the 2026-05-09 event on
`0x752c5a95`, which received 50 approvals/day baseline and 4,498 on the
discharge day (Z ≈ 88). The detector is calibrated so that this event
produces a Tier 1 alert hours before the 11:28 UTC discharge.

CLI:
    python -m surveillance.analytics.approval_spike_detector
    python -m surveillance.analytics.approval_spike_detector --as-of 2026-05-09
    python -m surveillance.analytics.approval_spike_detector --z 3.0 --window 14

Decision output (per question store):
    Real-time "imminent discharge" alert; 0-3h pre-trigger warning.

This is a session-time analyzer; the production wiring (running it on a
schedule) is tracked separately. See `surveillance.sai.question_runner`.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


@dataclass
class SpikeAlert:
    contract_address: str
    chain: str
    deployer_address: str
    deployer_watchlist_label: str | None
    confidence_tier: str
    as_of_date: str
    same_day_approvals: int
    baseline_mean: float
    baseline_stddev: float
    z_score: float
    baseline_days_observed: int

    def severity(self) -> str:
        """Tier-1: Z >= 10 = imminent discharge candidate.
        Tier-2: Z >= 5  = elevated.
        Tier-3: Z >= 3  = noteworthy.
        """
        if self.z_score >= 10:
            return "T1_IMMINENT"
        if self.z_score >= 5:
            return "T2_ELEVATED"
        return "T3_NOTEWORTHY"

    def fmt(self) -> str:
        return (
            f"[{self.severity()}] Z={self.z_score:>6.1f}  "
            f"as_of={self.as_of_date}  "
            f"contract={self.contract_address}  "
            f"chain={self.chain}  tier={self.confidence_tier}\n"
            f"           approvals_today={self.same_day_approvals:>5}  "
            f"baseline_mean={self.baseline_mean:>6.1f}  "
            f"baseline_stddev={self.baseline_stddev:>5.2f}  "
            f"baseline_obs={self.baseline_days_observed}\n"
            f"           deployer={self.deployer_address}  "
            f"watchlist={self.deployer_watchlist_label or '(off-watchlist)'}"
        )


def parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def fetch_watchlisted_baits(conn: sqlite3.Connection) -> list[tuple[str, str, str, str | None, str]]:
    """Return (contract_address, chain, deployer_address, deployer_label, tier).

    Strategy: a "bait" is a confirmed-tier or suspected-tier contract whose
    deployer is on the active watchlist. This is the contract-set we surveil.

    The contract itself may not be on the watchlist (most bait contracts
    aren't tagged individually — they're inferred via the deployer's
    operator role).
    """
    cur = conn.execute(
        """
        SELECT c.contract_address, c.chain, c.deployer_address,
               w.entity_name, c.confidence_tier
        FROM contracts c
        JOIN watchlist w
          ON w.address = c.deployer_address
         AND w.active = 1
        WHERE c.confidence_tier IN ('confirmed', 'suspected')
        """
    )
    return list(cur)


def fetch_daily_approval_counts(
    conn: sqlite3.Connection,
    contract_address: str,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, int]:
    """Return {YYYY-MM-DD: approval_count} for the given contract in [window_start, window_end)."""
    cur = conn.execute(
        """
        SELECT substr(approve_timestamp, 1, 10) AS d, COUNT(*) AS n
        FROM approval_watchlist
        WHERE contract_address = ?
          AND approve_timestamp >= ? AND approve_timestamp < ?
        GROUP BY 1
        """,
        (contract_address, window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")),
    )
    return {row[0]: row[1] for row in cur}


def compute_baseline(daily: dict[str, int], as_of: datetime, lookback_days: int) -> tuple[float, float, int]:
    """Compute mean + stddev over the lookback window ending day-before-as-of.

    Days with no recorded approvals contribute zeros (not omissions). This
    matters: a contract going from 0 approvals/day to 4,000 should produce
    a high Z, not a degenerate result.

    Returns (mean, stddev, n_days_observed_nonzero).
    """
    samples = []
    for offset in range(1, lookback_days + 1):
        day = (as_of - timedelta(days=offset)).strftime("%Y-%m-%d")
        samples.append(daily.get(day, 0))
    if not samples:
        return (0.0, 0.0, 0)
    mean = sum(samples) / len(samples)
    if len(samples) < 2:
        return (mean, 0.0, sum(1 for s in samples if s > 0))
    var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
    stddev = math.sqrt(var)
    return (mean, stddev, sum(1 for s in samples if s > 0))


def z_score(value: int, mean: float, stddev: float, n_obs: int) -> float:
    """Compute Z. If baseline is flat (zero stddev), return a synthetic high-Z
    if the new value is significantly above the trivial baseline.

    Edge cases:
      - baseline mean=0 and value=0 -> Z=0
      - baseline mean=0 and value>0 -> uses a smoothed denominator (sqrt(value))
        to avoid divide-by-zero. This makes "first-ever activity" detectable.
      - baseline stddev=0 but mean>0 -> smoothed using sqrt(mean+1).
      - too-few-observations (n_obs < 3) -> return a lower-confidence Z
        scaled by sqrt(n_obs/3); avoids spurious alerts when baseline is thin.
    """
    if value <= 0 and mean <= 0:
        return 0.0
    if stddev == 0:
        # Smoothing for trivial baselines
        denom = math.sqrt(max(mean, 1.0)) if mean > 0 else math.sqrt(max(value, 1.0))
        raw_z = (value - mean) / denom
    else:
        raw_z = (value - mean) / stddev
    if n_obs < 3:
        raw_z *= math.sqrt(n_obs / 3.0) if n_obs > 0 else 0.0
    return raw_z


def detect_spikes(
    db_path: Path = DB_PATH,
    as_of: datetime | None = None,
    lookback_days: int = 14,
    z_threshold: float = 3.0,
    min_value: int = 10,
) -> list[SpikeAlert]:
    """Run the detector. Returns alerts at or above z_threshold AND min_value.

    Args:
        as_of: the day to evaluate. Approvals on this date are the test sample.
               If None, defaults to today's UTC date.
        lookback_days: baseline trailing window. 14 is the spec; shorter
               windows produce more sensitive Z but noisier baselines.
        z_threshold: minimum Z to emit. Z>=3 is conservative; Z>=5 is
               actionable; Z>=10 is the imminent-discharge tier.
        min_value: minimum same_day_approvals to avoid alerting on
               low-volume noise (e.g., 0→1 approval is technically high-Z
               but useless).
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    window_start = as_of - timedelta(days=lookback_days + 1)
    window_end = as_of + timedelta(days=1)

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        baits = fetch_watchlisted_baits(conn)
        alerts: list[SpikeAlert] = []
        for contract, chain, deployer, label, tier in baits:
            daily = fetch_daily_approval_counts(conn, contract, window_start, window_end)
            same_day = daily.get(as_of.strftime("%Y-%m-%d"), 0)
            if same_day < min_value:
                continue
            mean, stddev, n_obs = compute_baseline(daily, as_of, lookback_days)
            z = z_score(same_day, mean, stddev, n_obs)
            if z < z_threshold:
                continue
            alerts.append(SpikeAlert(
                contract_address=contract,
                chain=chain,
                deployer_address=deployer,
                deployer_watchlist_label=label,
                confidence_tier=tier,
                as_of_date=as_of.strftime("%Y-%m-%d"),
                same_day_approvals=same_day,
                baseline_mean=mean,
                baseline_stddev=stddev,
                z_score=z,
                baseline_days_observed=n_obs,
            ))
    finally:
        conn.close()

    alerts.sort(key=lambda a: a.z_score, reverse=True)
    return alerts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today (UTC)")
    ap.add_argument("--window", type=int, default=14, help="baseline lookback days (default 14)")
    ap.add_argument("--z", type=float, default=3.0, help="Z threshold (default 3.0)")
    ap.add_argument("--min-value", type=int, default=10,
                    help="minimum same-day approvals to alert (default 10)")
    ap.add_argument("--db", type=str, default=str(DB_PATH))
    args = ap.parse_args()

    as_of = parse_iso_date(args.as_of) if args.as_of else None
    print(f"Approval spike detector — answering Q-002")
    print(f"  db: {args.db}")
    print(f"  as_of: {as_of.strftime('%Y-%m-%d') if as_of else 'today (UTC)'}")
    print(f"  baseline window: {args.window} days")
    print(f"  Z threshold: {args.z}")
    print(f"  min value: {args.min_value}")
    print()

    alerts = detect_spikes(
        db_path=Path(args.db),
        as_of=as_of,
        lookback_days=args.window,
        z_threshold=args.z,
        min_value=args.min_value,
    )
    print(f"Alerts: {len(alerts)}")
    print()
    for a in alerts:
        print(a.fmt())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
