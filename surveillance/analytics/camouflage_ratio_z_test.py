"""Camouflage Ratio two-proportion z-test.

Tests the lexicon's claim that "dangerous contracts maintain low revert
rates (under 10%) to evade standard detection. Stable at 70-79% across
chains, organizations, and time."

Per CLAUDE.md retirement list, "Camouflage ratio 68%" was retired
2026-04-02 with current value stated as 70-79% range.

Hypothesis
----------
H0: P(revert_rate < 0.10 | dangerous-tier) == P(revert_rate < 0.10 | non-dangerous-tier)
H1: They differ.

If H0 is rejected with the predator group SIGNIFICANTLY HIGHER, that
confirms the camouflage equilibrium (predators systematically calibrate
to low revert rates). If equal or reversed, the equilibrium claim needs
revision.

Methodology
-----------
- "Dangerous" = confirmed-tier (canonical interpretation per Pattern A
                 framework). Also reports suspected-tier separately.
- "Non-dangerous" = unanalyzed-tier (the most-populous baseline).
- Per-contract revert rate = SUM(is_reverted) / COUNT(*) from
  transaction_events, contract_address grouped.
- Inclusion criterion: at least 5 transactions. Below that, revert_rate
  estimates are too noisy.
- Cross-tabulate by chain.
- Two-proportion z-test (large N - no continuity correction needed).
- Wilson 95% CIs on the per-group proportions.

CLI:
    python -m surveillance.analytics.camouflage_ratio_z_test
    python -m surveillance.analytics.camouflage_ratio_z_test --min-tx 10
    python -m surveillance.analytics.camouflage_ratio_z_test --threshold 0.05
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import norm
except ImportError as e:
    sys.stderr.write(f"numpy + scipy required: {e}\n")
    raise

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_prop_z_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-proportion z-test (unpooled). Returns z, p, and effect size."""
    if n1 == 0 or n2 == 0:
        return {"z": float("nan"), "p": 1.0, "diff": float("nan")}
    p1 = k1 / n1
    p2 = k2 / n2
    # Pooled proportion for the null-hypothesis variance estimate
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": float("nan"), "p": 1.0, "diff": p1 - p2}
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return {"z": z_stat, "p": p_value, "diff": p1 - p2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--min-tx", type=int, default=5,
                    help="minimum transactions per contract to include (default 5)")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="revert-rate threshold for 'low revert' (default 0.10)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    sys.stderr.write("  computing per-contract revert rates...\n")
    rows = conn.execute(
        """
        SELECT c.contract_address, c.chain, c.confidence_tier,
               COUNT(t.id) AS n_tx,
               SUM(CASE WHEN t.is_reverted = 1 THEN 1 ELSE 0 END) AS n_revert
        FROM contracts c
        JOIN transaction_events t ON t.contract_address = c.contract_address
        GROUP BY c.contract_address
        HAVING n_tx >= ?
        """,
        (args.min_tx,)
    ).fetchall()
    conn.close()

    sys.stderr.write(f"  {len(rows):,} contracts with >= {args.min_tx} txs\n")
    if not rows:
        sys.stderr.write("No data\n")
        return 1

    # Compute per-contract revert rate + classify low_revert
    by_tier_chain: dict[tuple[str, str], list[bool]] = {}
    by_tier: dict[str, list[bool]] = {}
    for addr, chain, tier, n_tx, n_revert in rows:
        rate = n_revert / n_tx
        low = rate < args.threshold
        key = (tier or "unknown", chain or "?")
        by_tier_chain.setdefault(key, []).append(low)
        by_tier.setdefault(tier or "unknown", []).append(low)

    # Per-tier global ratio + 95% CI
    print()
    print("=" * 78)
    print(f"CAMOUFLAGE RATIO by tier (revert_rate < {args.threshold} = 'low'; min_tx >= {args.min_tx})")
    print("=" * 78)
    print(f"  {'tier':15s}  {'N':>8s}  {'low-revert':>12s}  {'ratio':>7s}  {'95% Wilson CI':>20s}")
    print("  " + "-" * 70)
    tier_stats = {}
    for tier in sorted(by_tier.keys()):
        lows = by_tier[tier]
        n = len(lows)
        k = sum(lows)
        ratio = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        tier_stats[tier] = {"n": n, "k": k, "ratio": ratio, "ci": (lo, hi)}
        print(f"  {tier:15s}  {n:>8,}  {k:>12,}  {ratio:>7.4f}  [{lo:.4f}, {hi:.4f}]")

    # Two-proportion z-tests: confirmed vs unanalyzed; suspected vs unanalyzed
    print()
    print("=" * 78)
    print("TWO-PROPORTION Z-TEST: predator vs control (control = unanalyzed)")
    print("=" * 78)

    baseline_tier = "unanalyzed"
    if baseline_tier not in tier_stats:
        # Fall back to "unknown" if unanalyzed not present
        baseline_tier = "unknown"
    if baseline_tier not in tier_stats:
        print(f"No baseline tier ({baseline_tier}) present.")
        return 1
    base = tier_stats[baseline_tier]
    print(f"  baseline ({baseline_tier}):  ratio = {base['ratio']:.4f}  (n={base['n']:,})")
    print()

    for predator_tier in ("confirmed", "suspected"):
        if predator_tier not in tier_stats:
            continue
        pred = tier_stats[predator_tier]
        z = two_prop_z_test(pred["k"], pred["n"], base["k"], base["n"])
        diff_pp = (pred["ratio"] - base["ratio"]) * 100
        sig = ""
        if z["p"] < 0.001:
            sig = " ***"
        elif z["p"] < 0.01:
            sig = " **"
        elif z["p"] < 0.05:
            sig = " *"
        print(f"  {predator_tier:15s}  vs {baseline_tier}:")
        print(f"    predator ratio = {pred['ratio']:.4f}  (n={pred['n']:,})")
        print(f"    delta:           {diff_pp:+.2f} pp")
        print(f"    z statistic:     {z['z']:>7.3f}")
        print(f"    p-value:         {z['p']:.6f}{sig}")
        if z["p"] < 0.05:
            direction = "HIGHER" if diff_pp > 0 else "LOWER"
            print(f"    Verdict: predator group has {direction} low-revert rate "
                  f"than baseline (Reject H0)")
        else:
            print(f"    Verdict: Fail to reject H0 at alpha=0.05")
        print()

    # Per-chain breakdown
    print()
    print("=" * 78)
    print("CAMOUFLAGE RATIO by chain × tier")
    print("=" * 78)
    chains = sorted({c for _, c in by_tier_chain.keys()})
    print(f"  {'chain':10s}  {'tier':15s}  {'N':>7s}  {'low-revert':>12s}  {'ratio':>7s}  {'95% CI':>18s}")
    print("  " + "-" * 75)
    for chain in chains:
        for tier in ("confirmed", "suspected", "unanalyzed", "unknown"):
            key = (tier, chain)
            if key not in by_tier_chain:
                continue
            lows = by_tier_chain[key]
            n = len(lows)
            k = sum(lows)
            if n < 10:
                continue  # too small to report
            ratio = k / n
            lo, hi = wilson_ci(k, n)
            print(f"  {chain:10s}  {tier:15s}  {n:>7,}  {k:>12,}  {ratio:>7.4f}  [{lo:.3f}, {hi:.3f}]")

    # Lexicon claim refresh
    print()
    print("=" * 78)
    print("LEXICON CLAIM REFRESH")
    print("=" * 78)
    print(f"  Lexicon (2026-04-02 onward): 'Camouflage Ratio 70-79% across chains'")
    print(f"  Lexicon Section A7 (2026-04-29 robustness): full-corpus 67.1%, top-12-excluded 68.1%")
    print()
    if "confirmed" in tier_stats:
        c = tier_stats["confirmed"]
        in_range = "WITHIN 70-79% band" if 0.70 <= c["ratio"] <= 0.79 else "OUTSIDE 70-79% band"
        print(f"  Corpus refresh (2026-05-19):")
        print(f"    confirmed-tier camouflage ratio: {c['ratio']:.4f} (95% CI [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}])")
        print(f"    {in_range}")
    if "suspected" in tier_stats:
        s = tier_stats["suspected"]
        print(f"    suspected-tier camouflage ratio: {s['ratio']:.4f} (95% CI [{s['ci'][0]:.4f}, {s['ci'][1]:.4f}])")
    if baseline_tier in tier_stats:
        b = tier_stats[baseline_tier]
        print(f"    {baseline_tier}-tier (baseline):       {b['ratio']:.4f} (95% CI [{b['ci'][0]:.4f}, {b['ci'][1]:.4f}])")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
