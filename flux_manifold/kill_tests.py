"""Kill tests – fast disproval gates for FluxManifold."""

from __future__ import annotations

import time

import numpy as np

from flux_manifold.core import flux_flow_traced
from flux_manifold.flows import normalize_flow, repulsive_flow


def kill_test_convergence(
    n_runs: int = 100,
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    fail_pct: float = 20.0,
) -> dict:
    """Kill Test 1: If steps > max_steps in >20% runs → kill."""
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    not_converged = 0

    for _ in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        r = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        if not r["converged"]:
            not_converged += 1

    pct = not_converged / n_runs * 100
    return {"test": "convergence", "not_converged_pct": pct, "pass": pct <= fail_pct}


def kill_test_drift(
    n_runs: int = 50,
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Kill Test 2: If final drift > tol in tier B → kill."""
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    bad_drift = 0

    for _ in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        r = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        final = r["drift_trace"][-1] if r["drift_trace"] else float("inf")
        if final > tol:
            bad_drift += 1

    pct = bad_drift / n_runs * 100
    return {"test": "drift", "bad_drift_pct": pct, "pass": pct == 0}


def kill_test_vs_random(
    n_runs: int = 50,
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
) -> dict:
    """Kill Test 3: If random walk baseline beats flux → kill."""
    from flux_manifold.baselines import random_walk

    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    flux_wins = 0

    for i in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        fm = flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        rw = random_walk(s0, q, epsilon=epsilon, tol=tol, max_steps=max_steps, rng=np.random.default_rng(seed + i))
        fm_drift = fm["drift_trace"][-1] if fm["drift_trace"] else float("inf")
        rw_drift = rw["drift_trace"][-1] if rw["drift_trace"] else float("inf")
        if fm_drift <= rw_drift:
            flux_wins += 1

    return {"test": "vs_random", "flux_win_pct": flux_wins / n_runs * 100, "pass": flux_wins / n_runs > 0.5}


def kill_test_scalability(
    d: int = 1024,
    n_runs: int = 10,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 1000,
    time_limit_ms: float = 5.0,
) -> dict:
    """Kill Test 4: d=1024 – if time > 5ms per run → kill."""
    rng = np.random.default_rng(seed)
    q = np.zeros(d, dtype=np.float32)
    times: list[float] = []

    for _ in range(n_runs):
        s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
        t0 = time.perf_counter()
        flux_flow_traced(s0, q, normalize_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    mean_ms = float(np.mean(times))
    return {"test": "scalability", "d": d, "mean_ms": mean_ms, "pass": mean_ms <= time_limit_ms}


def kill_test_adversarial(
    d: int = 128,
    seed: int = 42,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 100,
) -> dict:
    """Kill Test 5: Adversarial repulsive f – should diverge (expected behavior)."""
    rng = np.random.default_rng(seed)
    s0 = np.clip(rng.standard_normal(d).astype(np.float32), -1, 1)
    q = np.zeros(d, dtype=np.float32)

    r = flux_flow_traced(s0, q, repulsive_flow, epsilon=epsilon, tol=tol, max_steps=max_steps)
    final_drift = r["drift_trace"][-1] if r["drift_trace"] else float("inf")
    initial_drift = float(np.linalg.norm(s0 - q))

    # Repulsive flow should increase drift – verify system detects divergence
    diverged = final_drift > initial_drift
    return {
        "test": "adversarial",
        "initial_drift": initial_drift,
        "final_drift": final_drift,
        "diverged": diverged,
        "pass": diverged,  # We WANT it to diverge with repulsive f
    }


def kill_test_parser_roundtrip() -> dict:
    """Kill Test 6: Parser round-trip — known expressions must produce correct types."""
    from flux_manifold.parser import run as lf_run
    from flux_manifold.superposition import SuperpositionTensor

    cases = [
        ("[1, 2, 3]", np.ndarray, (3,)),
        ("[5, 5] ⟼ [0, 0]", np.ndarray, (2,)),
        ("∑_ψ [1, 0; 0, 1]", SuperpositionTensor, None),
        ("[1, 2, 3, 4] | squeeze 2", np.ndarray, (2,)),
    ]

    failures = []
    for expr, expected_type, expected_shape in cases:
        try:
            result = lf_run(expr)
            if not isinstance(result, expected_type):
                failures.append(f"{expr!r}: got {type(result).__name__}, expected {expected_type.__name__}")
            elif expected_shape is not None and hasattr(result, "shape") and result.shape != expected_shape:
                failures.append(f"{expr!r}: shape {result.shape} != {expected_shape}")
        except Exception as e:
            failures.append(f"{expr!r}: raised {type(e).__name__}: {e}")

    return {"test": "parser_roundtrip", "failures": failures, "pass": len(failures) == 0}


def kill_test_near_identity_rates() -> dict:
    """Kill Test 7: Near-identity rates — no profitable cycles when rates ≈ 1."""
    from backtest.data_ingestion import PoolState
    from backtest.latent_flux_searcher import find_arbitrage_opportunities, reset_reservoir

    reset_reservoir()

    ts = 1_700_000_000
    # Three stablecoins with rates within fee tier (0.05% = 500 bips)
    pool_states = [
        PoolState(block_timestamp=ts, pool_address="0xAB01", token0="USDC", token1="USDT",
                  sqrt_price=1.0, liquidity=1e9, tick=0,
                  token0_price=1.00005, token1_price=0.99995, volume_usd=1e6, fee_tier=500),
        PoolState(block_timestamp=ts, pool_address="0xAB02", token0="USDT", token1="DAI",
                  sqrt_price=1.0, liquidity=1e9, tick=0,
                  token0_price=1.00003, token1_price=0.99997, volume_usd=1e6, fee_tier=500),
        PoolState(block_timestamp=ts, pool_address="0xAB03", token0="DAI", token1="USDC",
                  sqrt_price=1.0, liquidity=1e9, tick=0,
                  token0_price=1.00002, token1_price=0.99998, volume_usd=1e6, fee_tier=500),
    ]

    opps = find_arbitrage_opportunities(pool_states, block_timestamp=ts, input_usd=10000.0)
    # With fees dominating, no cycle should be net-profitable
    net_profitable = [o for o in opps if o.net_profit_usd > 0]

    reset_reservoir()
    return {"test": "near_identity_rates", "signals": len(opps), "net_profitable": len(net_profitable),
            "pass": len(net_profitable) == 0}


def kill_test_disconnected_graph() -> dict:
    """Kill Test 8: Disconnected graph — no crash, returns empty."""
    from backtest.data_ingestion import PoolState
    from backtest.latent_flux_searcher import find_arbitrage_opportunities, reset_reservoir

    reset_reservoir()

    ts = 1_700_000_000
    # Two pairs sharing no tokens — graph is disconnected, no cycle possible
    pool_states = [
        PoolState(block_timestamp=ts, pool_address="0xCD01", token0="WETH", token1="USDC",
                  sqrt_price=1.0, liquidity=1e8, tick=0,
                  token0_price=2500.0, token1_price=0.0004, volume_usd=5e5, fee_tier=3000),
        PoolState(block_timestamp=ts, pool_address="0xCD02", token0="MATIC", token1="DAI",
                  sqrt_price=1.0, liquidity=1e8, tick=0,
                  token0_price=0.80, token1_price=1.25, volume_usd=2e5, fee_tier=3000),
    ]

    try:
        opps = find_arbitrage_opportunities(pool_states, block_timestamp=ts, input_usd=10000.0)
        crashed = False
    except Exception:
        opps = []
        crashed = True

    reset_reservoir()
    return {"test": "disconnected_graph", "signals": len(opps), "crashed": crashed,
            "pass": not crashed and len(opps) == 0}


def kill_test_reservoir_nan() -> dict:
    """Kill Test 9: Reservoir NaN propagation — NaN must not contaminate all dims forever."""
    from flux_manifold.reservoir_state import ReservoirState

    d = 8
    rs = ReservoirState(d=d, seed=42)

    # Feed one clean step to initialize
    rs.step(np.ones(d, dtype=np.float32))

    # Inject NaN in one dimension
    nan_input = np.ones(d, dtype=np.float32)
    nan_input[0] = float("nan")
    rs.step(nan_input)

    # Feed 10 clean steps
    for _ in range(10):
        rs.step(np.ones(d, dtype=np.float32) * 0.5)

    final = rs.readout()
    nan_dims = int(np.sum(np.isnan(final)))
    nan_pct = nan_dims / d * 100

    return {"test": "reservoir_nan", "nan_dims": nan_dims, "total_dims": d,
            "nan_pct": nan_pct, "pass": nan_pct < 10}


def kill_test_timestamp_gap() -> dict:
    """Kill Test 10: Timestamp gap resilience — 6-hour gaps must not create phantom spikes."""
    from backtest.data_ingestion import PoolState
    from backtest.latent_flux_searcher import find_arbitrage_opportunities, reset_reservoir

    reset_reservoir()

    base_ts = 1_700_000_000

    def _make_states(ts: int) -> list:
        # Small triangle with slight rate variation seeded by timestamp
        rng = np.random.default_rng(ts % 10000)
        drift = rng.uniform(-0.001, 0.001)
        return [
            PoolState(block_timestamp=ts, pool_address="0xEF01", token0="WETH", token1="USDC",
                      sqrt_price=1.0, liquidity=1e8, tick=0,
                      token0_price=2500.0 + drift * 100, token1_price=1 / (2500.0 + drift * 100),
                      volume_usd=5e5, fee_tier=3000),
            PoolState(block_timestamp=ts, pool_address="0xEF02", token0="USDC", token1="DAI",
                      sqrt_price=1.0, liquidity=1e9, tick=0,
                      token0_price=1.0001 + drift, token1_price=1 / (1.0001 + drift),
                      volume_usd=1e6, fee_tier=500),
            PoolState(block_timestamp=ts, pool_address="0xEF03", token0="DAI", token1="WETH",
                      sqrt_price=1.0, liquidity=1e8, tick=0,
                      token0_price=0.0004 + drift * 1e-5, token1_price=2500.0 - drift * 100,
                      volume_usd=3e5, fee_tier=3000),
        ]

    # Phase 1: 10 hourly snapshots (warm up reservoir)
    pre_gap_signals = 0
    for i in range(10):
        ts = base_ts + i * 3600
        opps = find_arbitrage_opportunities(_make_states(ts), block_timestamp=ts)
        pre_gap_signals += len(opps)

    # Phase 2: 6-hour gap (skip timestamps 10-15)
    # Phase 3: 5 snapshots after gap
    post_gap_signals = 0
    for i in range(5):
        ts = base_ts + (16 + i) * 3600  # resume after 6-hour gap
        opps = find_arbitrage_opportunities(_make_states(ts), block_timestamp=ts)
        post_gap_signals += len(opps)

    # Normalize to per-snapshot rate
    pre_rate = pre_gap_signals / 10 if pre_gap_signals > 0 else 0
    post_rate = post_gap_signals / 5

    # Post-gap signal rate should not spike more than 5x pre-gap
    spike_ok = post_rate <= max(pre_rate * 5, 1)  # allow at least 1 per snapshot

    reset_reservoir()
    return {"test": "timestamp_gap", "pre_gap_signals": pre_gap_signals,
            "post_gap_signals": post_gap_signals, "pre_rate": pre_rate,
            "post_rate": post_rate, "pass": spike_ok}


def run_all_kill_tests() -> list[dict]:
    """Run all kill tests and return results."""
    return [
        kill_test_convergence(),
        kill_test_drift(),
        kill_test_vs_random(),
        kill_test_scalability(),
        kill_test_adversarial(),
        kill_test_parser_roundtrip(),
        kill_test_near_identity_rates(),
        kill_test_disconnected_graph(),
        kill_test_reservoir_nan(),
        kill_test_timestamp_gap(),
    ]
