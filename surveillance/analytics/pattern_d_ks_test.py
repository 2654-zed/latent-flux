"""Kolmogorov-Smirnov test on Pattern D mainnet-L2 gap distribution.

Tests whether suspected/confirmed-tier deployers have a *distributionally
different* mainnet-L2 first-tx gap than unanalyzed/unknown-tier deployers.

The lexicon's Pattern D entry claims cross-chain reputation import is a
behavioral marker of adversarial operators. The corresponding empirical
claim is: "54/100 high-risk L2 deployers have mainnet first-tx predating
L2 first-seen."

The Cox PH model (commit e6ecc7c) found that mainnet_l2_gap_days has no
HAZARD effect (p=0.82). But that tests a different question. Cox asks
"does the gap predict when a contract drains?" KS asks "is the gap
DISTRIBUTION different between predator and non-predator deployers?"

If KS rejects equality, Pattern D is a real behavioral marker — just not
a hazard predictor.
If KS fails to reject, Pattern D may have been overstated entirely.

Methodology
-----------
H0: The CDFs of mainnet_l2_gap_days are equal between the two populations.
H1: The CDFs differ (two-sided test).

Populations:
  Test A — by contract tier:
    Group "predator":  deployers with >=1 contract in suspected OR confirmed
    Group "control":   deployers whose contracts are ALL unanalyzed/unknown

  Test B — by drain outcome:
    Group "predator":  deployers with >=1 contract that has drain_detected=1
    Group "control":   deployers with >=1 contract that received approvals
                       but NEVER drained

Exclusion: deployers with no mainnet_first_tx are excluded entirely (they
have no defined gap; mixing them with gap=0 conflates "no mainnet identity"
with "synchronous mainnet identity").

The KS test uses scipy.stats.ks_2samp. Bonferroni correction is applied
when reporting since we run two tests (alpha_each = 0.025 for joint
alpha = 0.05).

CLI:
    python -m surveillance.analytics.pattern_d_ks_test
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import ks_2samp
except ImportError as e:
    sys.stderr.write(f"numpy + scipy required: {e}\n")
    raise

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


def parse_ts(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class DeployerRow:
    address: str
    chain: str
    first_seen: datetime
    mainnet_first_tx: datetime
    gap_days: float
    has_suspected_or_confirmed: bool
    has_drain: bool
    contract_count: int


def load_deployers(conn: sqlite3.Connection) -> list[DeployerRow]:
    """Load deployers with mainnet history + their contract-level outcome flags."""
    sys.stderr.write("  loading deployers + outcome flags...\n")
    rows = conn.execute(
        """
        SELECT d.deployer_address, d.chain, d.first_seen, d.mainnet_first_tx,
               COUNT(c.contract_address) AS n_contracts,
               SUM(CASE WHEN c.confidence_tier IN ('suspected', 'confirmed') THEN 1 ELSE 0 END) AS n_sc,
               (
                 SELECT COUNT(DISTINCT aw.contract_address)
                 FROM approval_watchlist aw
                 JOIN contracts c2 ON c2.contract_address = aw.contract_address
                 WHERE c2.deployer_address = d.deployer_address
                   AND aw.drain_detected = 1
               ) AS n_drains
        FROM deployers d
        LEFT JOIN contracts c ON c.deployer_address = d.deployer_address
        WHERE d.mainnet_first_tx IS NOT NULL
          AND d.first_seen IS NOT NULL
        GROUP BY d.deployer_address
        """
    ).fetchall()

    out: list[DeployerRow] = []
    for r in rows:
        addr, chain, fs_s, mn_s, n_c, n_sc, n_drains = r
        fs = parse_ts(fs_s)
        mn = parse_ts(mn_s)
        if fs is None or mn is None:
            continue
        gap_seconds = (fs - mn).total_seconds()
        if gap_seconds < 0:
            # L2 first tx before mainnet first tx — shouldn't happen in normal cases.
            # Either parsing issue or an address that started L2-first and back-bridged
            # to mainnet. Skip — invalid Pattern D candidate.
            continue
        gap_days = gap_seconds / 86400.0
        out.append(DeployerRow(
            address=addr, chain=chain or "?",
            first_seen=fs, mainnet_first_tx=mn,
            gap_days=gap_days,
            has_suspected_or_confirmed=(n_sc or 0) > 0,
            has_drain=(n_drains or 0) > 0,
            contract_count=int(n_c or 0),
        ))
    sys.stderr.write(f"  {len(out)} deployers with mainnet history loaded\n")
    return out


def quantiles(arr: np.ndarray, qs: list[float]) -> list[float]:
    return [float(np.quantile(arr, q)) for q in qs]


def fmt_pct(p: float) -> str:
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def cdf_at(arr: np.ndarray, threshold: float) -> float:
    """Empirical CDF: P(X <= threshold)."""
    return float(np.mean(arr <= threshold))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = load_deployers(conn)
    finally:
        conn.close()

    if not rows:
        sys.stderr.write("No deployers with mainnet history.\n")
        return 1

    print()
    print("=" * 78)
    print("PATTERN D KS TEST — mainnet-L2 first-tx gap distribution")
    print("=" * 78)
    print(f"Total deployers with mainnet_first_tx populated: {len(rows):,}")
    print(f"Excluded (no mainnet history): see deployers table; not loaded.")
    print()

    # ============================================================
    # TEST A: by contract tier
    # ============================================================
    group_a_pred = [r.gap_days for r in rows if r.has_suspected_or_confirmed]
    group_a_ctrl = [r.gap_days for r in rows if not r.has_suspected_or_confirmed]

    if not group_a_pred or not group_a_ctrl:
        print("Test A skipped — one group is empty.")
    else:
        gap_a_p = np.array(group_a_pred)
        gap_a_c = np.array(group_a_ctrl)
        print("=" * 78)
        print("TEST A — by contract tier (predator: any suspected/confirmed contract)")
        print("=" * 78)
        print(f"  Group 'predator':  N = {len(gap_a_p):,}")
        print(f"  Group 'control':   N = {len(gap_a_c):,}")
        print()
        print(f"  {'statistic':30s}  {'predator':>14s}  {'control':>14s}")
        for label, q in [("min", 0.0), ("10th pct", 0.10), ("median", 0.5),
                         ("90th pct", 0.90), ("max", 1.0)]:
            qp = float(np.quantile(gap_a_p, q))
            qc = float(np.quantile(gap_a_c, q))
            print(f"  {label:30s}  {qp:>14.1f}  {qc:>14.1f}")
        print(f"  {'mean':30s}  {gap_a_p.mean():>14.1f}  {gap_a_c.mean():>14.1f}")
        print(f"  {'stddev':30s}  {gap_a_p.std():>14.1f}  {gap_a_c.std():>14.1f}")

        # The lexicon claim: 54% of high-risk have mainnet predating L2 (i.e., gap > 0)
        # We've already filtered to gap >= 0 (those with mainnet history at all),
        # so the relevant cross-check is what fraction have gap > 60 days
        # (the Pattern D Scanner threshold per the lexicon entry).
        print()
        print(f"  P(gap > 60 days)        (Pattern D threshold)")
        p_pred_60 = float(np.mean(gap_a_p > 60))
        p_ctrl_60 = float(np.mean(gap_a_c > 60))
        print(f"    predator: {p_pred_60:.4f}  ({100*p_pred_60:.1f}%)")
        print(f"    control:  {p_ctrl_60:.4f}  ({100*p_ctrl_60:.1f}%)")
        print(f"    delta:    {p_pred_60 - p_ctrl_60:+.4f}")

        ks_result_a = ks_2samp(gap_a_p, gap_a_c, alternative="two-sided")
        print()
        print(f"  KS test (two-sided):")
        print(f"    D statistic:      {ks_result_a.statistic:.4f}")
        print(f"    p-value:          {fmt_pct(ks_result_a.pvalue)}")
        bonferroni_alpha = 0.025
        verdict_a = "REJECT H0" if ks_result_a.pvalue < bonferroni_alpha else "FAIL TO REJECT H0"
        print(f"    Verdict (Bonferroni alpha=0.025): {verdict_a}")

    print()

    # ============================================================
    # TEST B: by drain outcome
    # ============================================================
    group_b_pred = [r.gap_days for r in rows if r.has_drain]
    group_b_ctrl = [r.gap_days for r in rows
                    if not r.has_drain and r.has_suspected_or_confirmed]
    # control: has a flagged contract but no drain observed yet

    if not group_b_pred or not group_b_ctrl:
        print("Test B skipped — one group is empty.")
    else:
        gap_b_p = np.array(group_b_pred)
        gap_b_c = np.array(group_b_ctrl)
        print("=" * 78)
        print("TEST B — by drain outcome (predator: any contract drained)")
        print("       Note: control restricted to deployers with flagged contracts")
        print("       but no drain (i.e., 'flagged but quiet').")
        print("=" * 78)
        print(f"  Group 'drained':       N = {len(gap_b_p):,}")
        print(f"  Group 'flagged-quiet': N = {len(gap_b_c):,}")
        print()
        print(f"  {'statistic':30s}  {'drained':>14s}  {'flagged-quiet':>16s}")
        for label, q in [("min", 0.0), ("10th pct", 0.10), ("median", 0.5),
                         ("90th pct", 0.90), ("max", 1.0)]:
            qp = float(np.quantile(gap_b_p, q))
            qc = float(np.quantile(gap_b_c, q))
            print(f"  {label:30s}  {qp:>14.1f}  {qc:>16.1f}")
        print(f"  {'mean':30s}  {gap_b_p.mean():>14.1f}  {gap_b_c.mean():>16.1f}")
        print(f"  {'stddev':30s}  {gap_b_p.std():>14.1f}  {gap_b_c.std():>16.1f}")

        print()
        print(f"  P(gap > 60 days):")
        p_pred_60_b = float(np.mean(gap_b_p > 60))
        p_ctrl_60_b = float(np.mean(gap_b_c > 60))
        print(f"    drained:       {p_pred_60_b:.4f}  ({100*p_pred_60_b:.1f}%)")
        print(f"    flagged-quiet: {p_ctrl_60_b:.4f}  ({100*p_ctrl_60_b:.1f}%)")
        print(f"    delta:         {p_pred_60_b - p_ctrl_60_b:+.4f}")

        ks_result_b = ks_2samp(gap_b_p, gap_b_c, alternative="two-sided")
        print()
        print(f"  KS test (two-sided):")
        print(f"    D statistic:      {ks_result_b.statistic:.4f}")
        print(f"    p-value:          {fmt_pct(ks_result_b.pvalue)}")
        verdict_b = "REJECT H0" if ks_result_b.pvalue < bonferroni_alpha else "FAIL TO REJECT H0"
        print(f"    Verdict (Bonferroni alpha=0.025): {verdict_b}")

    # ============================================================
    # CROSS-CHECK: the lexicon claim "54/100 high-risk L2 deployers have
    # mainnet first-tx predating L2 first-seen"
    # ============================================================
    print()
    print("=" * 78)
    print("LEXICON CROSS-CHECK: '54/100 high-risk L2 deployers have mainnet predating L2'")
    print("=" * 78)
    # All "high-risk" deployers (= any suspected/confirmed contract) including
    # those WITHOUT mainnet_first_tx (which would be 0% of the population with
    # gap, but most of the overall population)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        n_hr_total = conn.execute(
            """SELECT COUNT(DISTINCT d.deployer_address)
               FROM deployers d JOIN contracts c ON c.deployer_address = d.deployer_address
               WHERE c.confidence_tier IN ('suspected', 'confirmed')"""
        ).fetchone()[0]
        n_hr_with_mainnet = conn.execute(
            """SELECT COUNT(DISTINCT d.deployer_address)
               FROM deployers d JOIN contracts c ON c.deployer_address = d.deployer_address
               WHERE c.confidence_tier IN ('suspected', 'confirmed')
                 AND d.mainnet_first_tx IS NOT NULL"""
        ).fetchone()[0]
        n_hr_predating = conn.execute(
            """SELECT COUNT(DISTINCT d.deployer_address)
               FROM deployers d JOIN contracts c ON c.deployer_address = d.deployer_address
               WHERE c.confidence_tier IN ('suspected', 'confirmed')
                 AND d.mainnet_first_tx IS NOT NULL
                 AND d.mainnet_first_tx < d.first_seen"""
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"  High-risk deployers (any suspected/confirmed contract): {n_hr_total:,}")
    print(f"  Of those, with mainnet_first_tx populated: {n_hr_with_mainnet:,} "
          f"({100*n_hr_with_mainnet/max(n_hr_total,1):.1f}%)")
    print(f"  Of those, with mainnet predating L2: {n_hr_predating:,} "
          f"({100*n_hr_predating/max(n_hr_with_mainnet,1):.1f}% of mainnet-enriched)")
    print(f"  Of all high-risk: {100*n_hr_predating/max(n_hr_total,1):.1f}% have mainnet predating L2")
    print()
    print(f"  Lexicon claim: 54% (54/100). Refresh: see corpus values above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
