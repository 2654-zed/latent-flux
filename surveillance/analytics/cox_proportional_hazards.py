"""Cox proportional hazards model for time-to-first-drain.

Extends the univariate survival analysis (survival_time_to_drain.py) into
a multi-covariate model. For each contract, we now regress instantaneous
hazard on:

    - chain (one-hot: base / arbitrum / optimism)
    - confidence_tier (one-hot: confirmed / suspected / unanalyzed)
    - deployer_age_days (numeric — days from deployer.first_seen to contract deploy)
    - has_mainnet_history (binary: deployer.mainnet_first_tx IS NOT NULL)
    - mainnet_l2_gap_days (numeric, 0 if no mainnet history — Pattern D primitive)
    - funder_on_watchlist (binary: deployer's funding_trail.funder is on the
                          active watchlist — Q-009 ancestor-linkage primitive)
    - deployer_total_contracts (numeric — fleet-size primitive)

Identifies WHICH lexicon primitive carries the strongest discriminating
signal once others are controlled. The Q-002 calibration backtest showed
that Z alone is statistically underdetermined — this analysis asks whether
the deeper topological features (chain, tier, funder graph) carry that
signal once we make them explicit.

The Cox PH model is fit via maximum partial likelihood (Breslow tie
handling) using scipy.optimize. Implemented from scratch — no lifelines
dependency. SEs derived from the Hessian inverse.

Output:
    - Per-covariate β_hat, SE, exp(β) (hazard ratio), 95% CI, Wald z, p-value
    - Likelihood-ratio test vs. tier-only model (proves multi-covariate
      adds discriminating power)
    - Top-ranked predictors by absolute coefficient magnitude (standardized)

CLI:
    python -m surveillance.analytics.cox_proportional_hazards
    python -m surveillance.analytics.cox_proportional_hazards --min-events 30
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.stats import norm, chi2 as chi2_dist
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
class Row:
    contract: str
    chain: str
    tier: str
    detection_ts: datetime
    last_obs_ts: datetime
    event: bool
    days_observed: float
    deployer_age_days: float | None
    has_mainnet: int  # 0/1
    mainnet_l2_gap_days: float
    funder_on_watchlist: int  # 0/1
    deployer_total_contracts: int


def load_data(conn: sqlite3.Connection) -> list[Row]:
    """Load the at-risk population with full covariates."""
    sys.stderr.write("  loading covariates...\n")
    # Active watchlist addresses (for funder lookup)
    wl_addrs = {r[0].lower() for r in conn.execute(
        "SELECT address FROM watchlist WHERE active=1 AND address IS NOT NULL"
    ) if r[0]}

    rows_raw = conn.execute(
        """
        SELECT c.contract_address,
               c.chain,
               c.confidence_tier,
               c.detection_timestamp,
               MIN(CASE WHEN aw.drain_detected=1 THEN aw.drain_timestamp END) AS first_drain,
               MAX(aw.approve_timestamp) AS last_appr,
               d.first_seen AS deployer_first_seen,
               d.mainnet_first_tx,
               d.total_contracts_deployed,
               d.funding_trail
        FROM contracts c
        JOIN approval_watchlist aw ON aw.contract_address = c.contract_address
        LEFT JOIN deployers d ON d.deployer_address = c.deployer_address
        WHERE c.detection_timestamp IS NOT NULL
        GROUP BY c.contract_address
        """
    ).fetchall()

    out: list[Row] = []
    for r in rows_raw:
        addr, chain, tier, det_s, drain_s, appr_s, dep_seen, mn_first, total_d, ft = r
        det = parse_ts(det_s)
        if det is None:
            continue
        if drain_s:
            last = parse_ts(drain_s)
            event = True
        else:
            last = parse_ts(appr_s)
            event = False
        if last is None or last < det:
            continue
        days = (last - det).total_seconds() / 86400.0
        if days <= 0:
            days = 0.01  # avoid zero observations

        dep_seen_dt = parse_ts(dep_seen) if dep_seen else None
        if dep_seen_dt:
            deployer_age_days = max(0.0, (det - dep_seen_dt).total_seconds() / 86400.0)
        else:
            deployer_age_days = None

        mn_first_dt = parse_ts(mn_first) if mn_first else None
        has_mainnet = 1 if mn_first_dt else 0
        if mn_first_dt:
            mainnet_l2_gap_days = max(0.0, (det - mn_first_dt).total_seconds() / 86400.0)
        else:
            mainnet_l2_gap_days = 0.0

        funder_wl = 0
        if ft:
            try:
                ftj = json.loads(ft)
                f = ftj.get("funder") if isinstance(ftj, dict) else None
                if f and f.lower() in wl_addrs:
                    funder_wl = 1
            except (json.JSONDecodeError, TypeError):
                pass

        out.append(Row(
            contract=addr, chain=chain or "?", tier=tier or "unknown",
            detection_ts=det, last_obs_ts=last, event=event, days_observed=days,
            deployer_age_days=deployer_age_days,
            has_mainnet=has_mainnet,
            mainnet_l2_gap_days=mainnet_l2_gap_days,
            funder_on_watchlist=funder_wl,
            deployer_total_contracts=int(total_d or 0),
        ))
    sys.stderr.write(f"  {len(out)} subjects loaded\n")
    return out


def build_design_matrix(rows: list[Row]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Build (X, feature_names, times, events) for Cox fitting.

    Reference categories (omitted from one-hot to avoid collinearity):
      - chain: 'base' as reference (most common)
      - tier:  'unanalyzed' as reference (most common; suspected/confirmed
               are the alternatives we want hazard ratios for)

    Continuous features are STANDARDIZED (mean=0, sd=1) so coefficients are
    interpretable in standard-deviation units.
    """
    feature_names: list[str] = []
    cols: list[np.ndarray] = []

    n = len(rows)
    if n == 0:
        return np.zeros((0, 0)), [], np.zeros(0), np.zeros(0)

    # Chain one-hot (reference: base)
    for ch_val in ("arbitrum", "optimism"):
        col = np.array([1.0 if r.chain == ch_val else 0.0 for r in rows])
        cols.append(col)
        feature_names.append(f"chain={ch_val}")

    # Tier one-hot (reference: unanalyzed)
    for t_val in ("suspected", "confirmed", "unknown"):
        col = np.array([1.0 if r.tier == t_val else 0.0 for r in rows])
        cols.append(col)
        feature_names.append(f"tier={t_val}")

    # Numeric: deployer_age_days (impute missing with median)
    raw_age = np.array([
        r.deployer_age_days if r.deployer_age_days is not None else np.nan
        for r in rows
    ])
    med_age = float(np.nanmedian(raw_age)) if not np.all(np.isnan(raw_age)) else 0.0
    raw_age = np.where(np.isnan(raw_age), med_age, raw_age)
    cols.append(_standardize(raw_age))
    feature_names.append("deployer_age_days (std)")

    # Binary: has_mainnet
    cols.append(np.array([float(r.has_mainnet) for r in rows]))
    feature_names.append("has_mainnet_history")

    # Numeric: mainnet_l2_gap_days (standardized among rows with mainnet history)
    raw_gap = np.array([r.mainnet_l2_gap_days for r in rows])
    cols.append(_standardize(raw_gap))
    feature_names.append("mainnet_l2_gap_days (std)")

    # Binary: funder_on_watchlist (the Q-009 primitive)
    cols.append(np.array([float(r.funder_on_watchlist) for r in rows]))
    feature_names.append("funder_on_watchlist")

    # Numeric: deployer_total_contracts (fleet-size; log-transform first)
    raw_tot = np.array([float(r.deployer_total_contracts) for r in rows])
    log_tot = np.log1p(raw_tot)
    cols.append(_standardize(log_tot))
    feature_names.append("log(1 + deployer_total_contracts) (std)")

    X = np.column_stack(cols)
    times = np.array([r.days_observed for r in rows])
    events = np.array([1 if r.event else 0 for r in rows], dtype=int)
    return X, feature_names, times, events


def _standardize(arr: np.ndarray) -> np.ndarray:
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < 1e-12:
        return arr - mu  # constant column
    return (arr - mu) / sd


def cox_nll_and_grad(beta: np.ndarray, X: np.ndarray, times: np.ndarray,
                     events: np.ndarray) -> tuple[float, np.ndarray]:
    """Negative log partial likelihood and gradient for Cox PH (Breslow ties).

    The risk-set sum is computed via descending-time sort and cumulative
    backward sums — O(n log n) total.
    """
    n, p = X.shape
    eta = X @ beta  # (n,)

    # Sort by descending time so that as we walk forward, we accumulate
    # exp(eta) for the risk set (subjects with time >= current).
    order = np.argsort(-times)
    eta_s = eta[order]
    times_s = times[order]
    events_s = events[order]
    X_s = X[order]

    # Cumulative sums (running from i=0 onward = risk set for time_s[i]
    # since we sorted descending, subjects 0..i have time >= times_s[i])
    exp_eta_s = np.exp(eta_s)
    cum_exp_eta = np.cumsum(exp_eta_s)              # (n,)
    cum_X_exp = np.cumsum(X_s * exp_eta_s[:, None], axis=0)  # (n, p)

    nll = 0.0
    grad = np.zeros(p)

    # For ties at a given time t, Breslow approximation says: each event at t
    # uses the SAME risk set (subjects with time >= t). With descending sort,
    # ties have contiguous indices. We collapse them.
    i = 0
    while i < n:
        j = i
        while j < n and times_s[j] == times_s[i]:
            j += 1
        # subjects [i, j) have the same time. The risk set for this time is
        # subjects [0, j) (all with time >= times_s[i]).
        # Events in this tied block: events_s[i:j]
        d = int(events_s[i:j].sum())
        if d > 0:
            sum_exp = float(cum_exp_eta[j - 1])
            if sum_exp <= 0:
                i = j
                continue
            log_sum = math.log(sum_exp)
            # Contribution to NLL: for each event, eta - log_sum
            event_mask = events_s[i:j].astype(bool)
            sum_eta_events = float(eta_s[i:j][event_mask].sum())
            nll -= sum_eta_events - d * log_sum
            # Gradient: -sum_X_events + d * (sum_X_exp_eta / sum_exp_eta)
            sum_X_events = X_s[i:j][event_mask].sum(axis=0)
            X_weighted_mean = cum_X_exp[j - 1] / sum_exp
            grad -= sum_X_events - d * X_weighted_mean
        i = j

    return nll, grad


def cox_hessian(beta: np.ndarray, X: np.ndarray, times: np.ndarray,
                events: np.ndarray) -> np.ndarray:
    """Observed information matrix (negative second derivative of partial
    log-likelihood). Used to derive standard errors via inverse.

    For Cox with Breslow ties: H = sum over events of [V_i - μ_i μ_i^T]
    where μ_i is the X-weighted mean over risk set, V_i is the X X^T
    weighted second moment.
    """
    n, p = X.shape
    eta = X @ beta
    order = np.argsort(-times)
    eta_s = eta[order]
    times_s = times[order]
    events_s = events[order]
    X_s = X[order]
    exp_eta_s = np.exp(eta_s)
    cum_exp_eta = np.cumsum(exp_eta_s)
    cum_X_exp = np.cumsum(X_s * exp_eta_s[:, None], axis=0)  # (n, p)
    # cum_XX_exp[i] = sum_{k<=i} X_k X_k^T exp(eta_k)
    cum_XX_exp = np.zeros((n, p, p))
    running = np.zeros((p, p))
    for k in range(n):
        running = running + np.outer(X_s[k], X_s[k]) * exp_eta_s[k]
        cum_XX_exp[k] = running

    H = np.zeros((p, p))
    i = 0
    while i < n:
        j = i
        while j < n and times_s[j] == times_s[i]:
            j += 1
        d = int(events_s[i:j].sum())
        if d > 0:
            sum_exp = float(cum_exp_eta[j - 1])
            if sum_exp <= 0:
                i = j
                continue
            mu = cum_X_exp[j - 1] / sum_exp
            V = cum_XX_exp[j - 1] / sum_exp
            H += d * (V - np.outer(mu, mu))
        i = j
    return H


def fit_cox(X: np.ndarray, times: np.ndarray, events: np.ndarray,
            verbose: bool = True) -> dict:
    """Fit Cox PH via L-BFGS-B with analytical gradient.

    Returns dict with beta, se, hr (exp(beta)), ci_lo, ci_hi, z, p, ll0, ll1.
    """
    n, p = X.shape
    beta0 = np.zeros(p)

    def fun(b):
        nll, grad = cox_nll_and_grad(b, X, times, events)
        return nll, grad

    result = minimize(
        fun, beta0, jac=True, method="L-BFGS-B",
        options={"gtol": 1e-6, "maxiter": 200, "disp": False},
    )
    beta_hat = result.x
    ll_final = -result.fun
    ll_null = -cox_nll_and_grad(np.zeros(p), X, times, events)[0]

    H = cox_hessian(beta_hat, X, times, events)
    try:
        # Add tiny ridge for numerical stability
        cov = np.linalg.inv(H + 1e-8 * np.eye(p))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        if verbose:
            sys.stderr.write("  WARNING: Hessian singular, SEs unreliable\n")
        se = np.full(p, np.nan)

    z = beta_hat / np.where(se > 0, se, 1.0)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    hr = np.exp(beta_hat)
    ci_lo = np.exp(beta_hat - 1.96 * se)
    ci_hi = np.exp(beta_hat + 1.96 * se)

    return {
        "beta": beta_hat,
        "se": se,
        "hr": hr,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "z": z,
        "p": p_value,
        "ll_null": ll_null,
        "ll_final": ll_final,
        "n_events": int(events.sum()),
        "n_subjects": n,
        "converged": result.success,
    }


def likelihood_ratio_test(ll_null: float, ll_full: float, df: int) -> tuple[float, float]:
    """Likelihood-ratio chi-squared test."""
    lr = 2 * (ll_full - ll_null)
    if lr < 0:
        lr = 0.0
    p = 1.0 - chi2_dist.cdf(lr, df=df)
    return lr, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--min-events", type=int, default=20,
                    help="abort if N events < this (default 20)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = load_data(conn)
    finally:
        conn.close()

    if not rows:
        sys.stderr.write("No subjects loaded.\n")
        return 1

    X, feat_names, times, events = build_design_matrix(rows)
    n_events = int(events.sum())
    if n_events < args.min_events:
        sys.stderr.write(f"Only {n_events} events — too few for reliable fit (need >= {args.min_events}).\n")
        return 1

    print(f"\n  N subjects: {X.shape[0]}")
    print(f"  N events:   {n_events}  ({100*n_events/X.shape[0]:.2f}%)")
    print(f"  N features: {X.shape[1]}")
    print(f"  Features:   {feat_names}")
    print()

    print("Fitting Cox PH (Breslow ties, L-BFGS-B with analytical gradient)...")
    fit = fit_cox(X, times, events)
    converged = "yes" if fit["converged"] else "NO (warnings)"
    print(f"  converged: {converged}")
    print(f"  log-likelihood (null):  {fit['ll_null']:.4f}")
    print(f"  log-likelihood (final): {fit['ll_final']:.4f}")

    lr, lr_p = likelihood_ratio_test(fit["ll_null"], fit["ll_final"], df=X.shape[1])
    print(f"  Likelihood-ratio test: chi2={lr:.3f}, df={X.shape[1]}, p={lr_p:.6f}")

    print()
    print("=" * 92)
    print("COVARIATE TABLE")
    print("=" * 92)
    print(f"  {'feature':40s}  {'beta':>8s}  {'SE':>8s}  {'HR':>8s}  {'95% CI':>16s}  {'z':>6s}  {'p':>9s}")
    print("  " + "-" * 88)
    # Sort by absolute beta (since features are standardized this is meaningful)
    order = sorted(range(len(feat_names)), key=lambda i: -abs(fit["beta"][i]))
    for i in order:
        ci = f"[{fit['ci_lo'][i]:.3f},{fit['ci_hi'][i]:.3f}]"
        sig = ""
        if fit["p"][i] < 0.001:
            sig = "***"
        elif fit["p"][i] < 0.01:
            sig = "**"
        elif fit["p"][i] < 0.05:
            sig = "*"
        print(f"  {feat_names[i]:40s}  {fit['beta'][i]:>+8.3f}  {fit['se'][i]:>8.3f}  "
              f"{fit['hr'][i]:>8.3f}  {ci:>16s}  {fit['z'][i]:>+6.2f}  {fit['p'][i]:>9.4f} {sig}")
    print()
    print("  * p<0.05  ** p<0.01  *** p<0.001")

    # Reference category summary
    print()
    print("Reference categories (omitted from design matrix):")
    print("  chain reference = base")
    print("  tier reference  = unanalyzed")

    # Identify top predictors
    print()
    print("=" * 92)
    print("TOP-RANKED PREDICTORS (by |beta| since continuous features are standardized)")
    print("=" * 92)
    for rank, idx in enumerate(order[:5], 1):
        direction = "increases" if fit["beta"][idx] > 0 else "decreases"
        magnitude = math.exp(abs(fit["beta"][idx]))
        sig = ""
        if fit["p"][idx] < 0.05:
            sig = f" (p={fit['p'][idx]:.4f})"
        else:
            sig = f" (p={fit['p'][idx]:.4f}; not sig)"
        print(f"  {rank}. {feat_names[idx]:40s}  {direction} hazard by {magnitude:.2f}x{sig}")

    # Univariate tier-only model — to quantify multi-covariate uplift
    print()
    print("=" * 92)
    print("COMPARISON: tier-only model (the prior univariate analysis)")
    print("=" * 92)
    # Re-build design matrix with ONLY tier indicators
    tier_cols = []
    tier_names = []
    for t_val in ("suspected", "confirmed", "unknown"):
        col = np.array([1.0 if r.tier == t_val else 0.0 for r in rows])
        tier_cols.append(col)
        tier_names.append(f"tier={t_val}")
    X_tier = np.column_stack(tier_cols)
    fit_tier = fit_cox(X_tier, times, events, verbose=False)
    print(f"  tier-only log-likelihood:  {fit_tier['ll_final']:.4f}")
    print(f"  full-model log-likelihood: {fit['ll_final']:.4f}")
    delta_ll = fit["ll_final"] - fit_tier["ll_final"]
    df_diff = X.shape[1] - X_tier.shape[1]
    lr_uplift, lr_uplift_p = likelihood_ratio_test(fit_tier["ll_final"], fit["ll_final"], df=df_diff)
    print(f"  LR uplift (full vs tier-only): chi2={lr_uplift:.3f}, df={df_diff}, p={lr_uplift_p:.6f}")
    if lr_uplift_p < 0.05:
        verdict = "REJECT null; multi-covariate model adds significant discriminating power"
    else:
        verdict = "Fail to reject null; multi-covariate model does NOT add significant power"
    print(f"  Verdict: {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
