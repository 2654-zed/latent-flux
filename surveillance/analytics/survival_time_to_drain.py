"""Survival analysis: time-from-detection-to-first-drain, stratified by
confidence_tier. Kaplan-Meier estimator + log-rank test.

Answers the canonical inferential question from the 2026-05-18 statistical
research program:

    "Is Kelp's 56.7-day pre-exploit lead time statistically unusual, or do
    CRITICAL-tier contracts systematically survive for a similarly long
    period before discharge?"

Population:
    Contracts that (a) have detection_timestamp AND
                   (b) have at least one approval in approval_watchlist
    (rationale: a contract with zero approvals is not "at risk" of being
    drained — including all 327k contracts would dilute the analysis with
    contracts that no victim ever interacted with).

Event: first drain_timestamp where drain_detected=1 for that contract.
Censoring: contracts that received approvals but never drained — censored
    at MAX(approve_timestamp) for that contract (= last observation date
    on which it was still alive).

Output:
    - Per-tier KM survival curve (printable table)
    - Median survival time per tier with 95% CI
    - Log-rank test for between-tier difference
    - Where 56.7 days (Kelp) falls in the CRITICAL/confirmed distribution
    - Comparison to 0x80b12bd0's 44-day lead time (2026-03-26 deploy →
      2026-05-09 discharge)

Uses scipy.stats only (already in requirements.txt). Kaplan-Meier and
log-rank implemented from scratch — both are short and well-defined.

CLI:
    python -m surveillance.analytics.survival_time_to_drain
    python -m surveillance.analytics.survival_time_to_drain --bins-days 14
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


@dataclass
class Subject:
    """One contract in the survival population."""
    contract_address: str
    chain: str
    confidence_tier: str
    detection_ts: datetime
    last_obs_ts: datetime          # = drain_ts if event=True, else MAX(approve_ts)
    event: bool                    # True if drained
    days_observed: float           # = (last_obs_ts - detection_ts).days


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


def load_population(conn: sqlite3.Connection) -> list[Subject]:
    """Build the at-risk population: contracts with >=1 approval.

    For each, compute time-from-detection to either first-drain (event) or
    last-observed-approval (censoring).
    """
    print("  loading at-risk population...", file=sys.stderr)
    rows = conn.execute(
        """
        SELECT
            c.contract_address,
            c.chain,
            c.confidence_tier,
            c.detection_timestamp,
            MIN(CASE WHEN aw.drain_detected = 1 THEN aw.drain_timestamp END) AS first_drain,
            MAX(aw.approve_timestamp) AS last_appr
        FROM contracts c
        JOIN approval_watchlist aw ON aw.contract_address = c.contract_address
        WHERE c.detection_timestamp IS NOT NULL
        GROUP BY c.contract_address
        """
    ).fetchall()
    out: list[Subject] = []
    skipped = 0
    for row in rows:
        addr, chain, tier, det_s, drain_s, appr_s = row
        det = parse_ts(det_s)
        if det is None:
            skipped += 1
            continue
        if drain_s:
            last = parse_ts(drain_s)
            event = True
        else:
            last = parse_ts(appr_s)
            event = False
        if last is None or last < det:
            skipped += 1
            continue
        days = (last - det).total_seconds() / 86400.0
        out.append(Subject(
            contract_address=addr,
            chain=chain or "?",
            confidence_tier=tier or "unknown",
            detection_ts=det,
            last_obs_ts=last,
            event=event,
            days_observed=days,
        ))
    print(f"  {len(out)} subjects loaded; {skipped} skipped (parse / time-ordering)", file=sys.stderr)
    return out


# ============================================================
# Kaplan-Meier estimator
# ============================================================

@dataclass
class KMPoint:
    t: float                # time (days)
    n_at_risk: int          # subjects still at risk just before t
    d: int                  # events at t
    S: float                # survival probability at t (cumulative)
    se: float               # Greenwood standard error


def kaplan_meier(subjects: list[Subject]) -> list[KMPoint]:
    """Compute KM curve from a list of subjects.

    Implements the standard product-limit estimator with Greenwood variance.
    Both event and censored observations contribute to the at-risk set.
    """
    if not subjects:
        return []
    # Sort by time
    sorted_subj = sorted(subjects, key=lambda s: s.days_observed)
    # Find unique event times only (KM jumps at event times; censoring
    # only removes subjects from the at-risk set)
    event_times = sorted({s.days_observed for s in sorted_subj if s.event})
    out: list[KMPoint] = [KMPoint(t=0.0, n_at_risk=len(sorted_subj), d=0, S=1.0, se=0.0)]
    S = 1.0
    var_sum = 0.0
    for t in sorted(event_times):
        n_at_risk = sum(1 for s in sorted_subj if s.days_observed >= t)
        d = sum(1 for s in sorted_subj if s.event and s.days_observed == t)
        if n_at_risk == 0:
            break
        S *= (1.0 - d / n_at_risk)
        # Greenwood: SE^2 = S^2 * sum( d_i / (n_i * (n_i - d_i)) )
        if n_at_risk > d:
            var_sum += d / (n_at_risk * (n_at_risk - d))
        se = S * math.sqrt(var_sum) if var_sum > 0 else 0.0
        out.append(KMPoint(t=t, n_at_risk=n_at_risk, d=d, S=S, se=se))
    return out


def median_survival(km: list[KMPoint]) -> float | None:
    """Return median survival time (first t at which S(t) <= 0.5).

    Returns None if S never crosses 0.5 (right-censored above the median).
    """
    for p in km:
        if p.S <= 0.5:
            return p.t
    return None


def quantile_survival(km: list[KMPoint], q: float) -> float | None:
    """Return t at which S(t) <= 1-q (e.g., q=0.10 -> first 10% to fail)."""
    target = 1.0 - q
    for p in km:
        if p.S <= target:
            return p.t
    return None


def survival_at_time(km: list[KMPoint], t: float) -> tuple[float, float]:
    """Return (S(t), SE(t)) at the given time."""
    last = km[0]
    for p in km:
        if p.t > t:
            break
        last = p
    return last.S, last.se


# ============================================================
# Log-rank test (Mantel-Haenszel) for two groups
# ============================================================

def log_rank_two_groups(s1: list[Subject], s2: list[Subject]) -> dict:
    """Two-sample log-rank test.

    Returns dict with O1, E1, V (variance), chi2 statistic, p-value.
    """
    # Combine, sort by time
    combined = [(s.days_observed, s.event, 1) for s in s1] + \
               [(s.days_observed, s.event, 2) for s in s2]
    combined.sort()
    event_times = sorted({t for t, e, _ in combined if e})
    O1 = 0
    E1 = 0.0
    V = 0.0
    for t in event_times:
        n1 = sum(1 for ts, _, g in combined if ts >= t and g == 1)
        n2 = sum(1 for ts, _, g in combined if ts >= t and g == 2)
        n = n1 + n2
        if n == 0:
            continue
        d1 = sum(1 for ts, e, g in combined if ts == t and e and g == 1)
        d2 = sum(1 for ts, e, g in combined if ts == t and e and g == 2)
        d = d1 + d2
        O1 += d1
        E1 += d * n1 / n
        # Variance term (hypergeometric)
        if n > 1:
            V += d * (n - d) * n1 * n2 / (n * n * (n - 1))
    if V == 0:
        return {"O1": O1, "E1": E1, "V": 0.0, "chi2": 0.0, "p": 1.0}
    chi2 = (O1 - E1) ** 2 / V
    # Approximate p from chi-squared with 1 df
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore
        p = 1.0 - chi2_dist.cdf(chi2, df=1)
    except ImportError:
        # Fallback: rough approximation for chi2 with 1 df
        # P(X > x) ≈ 2 * (1 - Φ(sqrt(x))) for x > 0
        from math import erf, sqrt
        p = 1.0 - erf(math.sqrt(chi2 / 2))
    return {"O1": O1, "E1": E1, "V": V, "chi2": chi2, "p": p}


def fmt_km_table(km: list[KMPoint], bins_days: int = 14, max_t: int = 90) -> str:
    """Format KM curve at binned timepoints up to max_t days."""
    lines = [f"  {'t_days':>7s}  {'n_risk':>7s}  {'d':>5s}  {'S(t)':>7s}  {'SE':>6s}  {'95%_CI':>15s}"]
    bin_times = list(range(0, max_t + 1, bins_days))
    for bt in bin_times:
        # Find KM point just before bt
        p = km[0]
        for q in km:
            if q.t > bt:
                break
            p = q
        ci_low = max(0.0, p.S - 1.96 * p.se)
        ci_high = min(1.0, p.S + 1.96 * p.se)
        lines.append(
            f"  {bt:>7}  {p.n_at_risk:>7}  {p.d:>5}  {p.S:>7.4f}  {p.se:>6.4f}  [{ci_low:.4f},{ci_high:.4f}]"
        )
    return "\n".join(lines)


# ============================================================
# Main analysis
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bins-days", type=int, default=7)
    ap.add_argument("--max-t", type=int, default=90)
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        all_subjects = load_population(conn)
    finally:
        conn.close()

    if not all_subjects:
        print("No subjects loaded — aborting.")
        return 1

    print()
    print("=" * 76)
    print("SURVIVAL ANALYSIS: time-from-detection-to-first-drain")
    print("=" * 76)
    print(f"Population: contracts with >=1 approval in approval_watchlist")
    print(f"  N = {len(all_subjects):,} subjects")
    n_events = sum(1 for s in all_subjects if s.event)
    print(f"  Events (drained): {n_events:,} ({100*n_events/len(all_subjects):.1f}%)")
    print(f"  Censored: {len(all_subjects)-n_events:,} ({100*(len(all_subjects)-n_events)/len(all_subjects):.1f}%)")

    # Stratify by confidence_tier
    tiers = sorted({s.confidence_tier for s in all_subjects})
    by_tier: dict[str, list[Subject]] = {t: [] for t in tiers}
    for s in all_subjects:
        by_tier[s.confidence_tier].append(s)

    print()
    print(f"{'tier':15s}  {'N':>7s}  {'events':>7s}  {'event_rate':>10s}  {'median_survival':>18s}")
    print("-" * 76)
    km_by_tier = {}
    for tier in tiers:
        subs = by_tier[tier]
        if len(subs) < 10:
            print(f"  {tier:15s}  {len(subs):>7}  (too small, skipping)")
            continue
        km = kaplan_meier(subs)
        km_by_tier[tier] = km
        n_e = sum(1 for s in subs if s.event)
        ms = median_survival(km)
        ms_s = f"{ms:.1f} days" if ms is not None else ">window"
        print(f"  {tier:15s}  {len(subs):>7}  {n_e:>7}  {100*n_e/len(subs):>9.2f}%  {ms_s:>18s}")

    # KM tables for the two most interesting tiers
    print()
    for tier in ("confirmed", "suspected"):
        if tier not in km_by_tier:
            continue
        print(f"\n=== Kaplan-Meier curve: {tier} ===")
        print(fmt_km_table(km_by_tier[tier], bins_days=args.bins_days, max_t=args.max_t))

    # Log-rank test: confirmed vs suspected
    if "confirmed" in km_by_tier and "suspected" in km_by_tier:
        print()
        print("=" * 76)
        print("LOG-RANK TEST: confirmed vs suspected")
        print("=" * 76)
        result = log_rank_two_groups(by_tier["confirmed"], by_tier["suspected"])
        print(f"  O1 (observed events, confirmed): {result['O1']}")
        print(f"  E1 (expected if equal hazard):   {result['E1']:.2f}")
        print(f"  Variance:                         {result['V']:.2f}")
        print(f"  chi-squared (1 df):               {result['chi2']:.3f}")
        print(f"  p-value:                          {result['p']:.6f}")
        if result['p'] < 0.001:
            verdict = "REJECT H0 with high confidence (p < 0.001)"
        elif result['p'] < 0.05:
            verdict = "REJECT H0 (p < 0.05)"
        else:
            verdict = "FAIL TO REJECT H0 (p >= 0.05)"
        print(f"  Verdict: {verdict}")

    # The Kelp question
    print()
    print("=" * 76)
    print("THE KELP QUESTION: where does 56.7 days fall?")
    print("=" * 76)
    if "confirmed" in km_by_tier:
        S_56, SE_56 = survival_at_time(km_by_tier["confirmed"], 56.7)
        print(f"  Among confirmed-tier contracts:")
        print(f"    S(56.7) = {S_56:.4f}  (95% CI: [{max(0,S_56-1.96*SE_56):.4f}, {min(1,S_56+1.96*SE_56):.4f}])")
        print(f"    Interpretation: {100*S_56:.1f}% of confirmed-tier contracts survive past 56.7 days "
              f"without first drain.")
        med = median_survival(km_by_tier["confirmed"])
        if med:
            print(f"    Median survival time: {med:.1f} days")
            if 56.7 > med:
                ratio = 56.7 / med
                print(f"    Kelp's 56.7 days is {ratio:.1f}x the median.")
            else:
                print(f"    Kelp's 56.7 days is below the median — well within typical range.")

    # 0x80b12bd0 comparison: deployed 2026-03-26, drained 2026-05-09 = 44 days
    print()
    print("=" * 76)
    print("0x80b12bd0 comparison (the empirical Pattern A case)")
    print("=" * 76)
    print("  Deploy: 2026-03-26;  first drain: 2026-05-09  =>  44 days")
    if "confirmed" in km_by_tier:
        S_44, SE_44 = survival_at_time(km_by_tier["confirmed"], 44.0)
        print(f"  S(44) for confirmed-tier: {S_44:.4f}")
        print(f"    => 0x80b12bd0 sits at the {100*(1-S_44):.1f}th percentile of failure-time distribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
