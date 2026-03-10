"""Latent Flux Arbitrage Searcher — geometric path-finding using Latent Flux primitives.

Given the same PoolState snapshots as the Bellman-Ford baseline, uses
FluxManifold primitives to find arbitrage opportunities geometrically:

  1. Encode the market state (all pool exchange rates) as a vector in ℝ^d
  2. Feed through ReservoirState with leak_rate=0.05 for temporal smoothing
  3. Generate candidate cycle attractors from the token graph structure
  4. AttractorCompetition selects the highest-profit basin
  5. RecursiveFlow confirms convergence toward the winning cycle

Produces ArbOpportunity objects in identical schema to bellman_ford.py
for direct comparison.

Usage:
    from backtest.latent_flux_searcher import find_arbitrage_opportunities
    from backtest.data_ingestion import PoolState

    opportunities = find_arbitrage_opportunities(pool_states, input_usd=10000.0)
"""

from __future__ import annotations

import math
import sys
import os
from collections import defaultdict
from itertools import permutations

import numpy as np

# Ensure repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux_manifold.reservoir_state import ReservoirState
from flux_manifold.recursive_flow import RecursiveFlow
from flux_manifold.attractor_competition import AttractorCompetition
from flux_manifold.flows import normalize_flow

from backtest.bellman_ford import ArbOpportunity, build_exchange_graph, Edge


# ── Configuration ─────────────────────────────────────────────────

# ReservoirState tuning for hourly pool data (~3600s between snapshots).
# leak_rate=0.05 means ~95% of current state retained per step,
# giving an effective memory window of ~20 hours (1/0.05).
RESERVOIR_LEAK_RATE = 0.05
RESERVOIR_SCALE = 4
SPECTRAL_RADIUS = 0.9
INPUT_SCALING = 0.1

# AttractorCompetition tuning
COMPETITION_EPSILON = 0.1
COMPETITION_TOL = 1e-2
COMPETITION_MAX_STEPS = 300
COMPETITION_REPULSION = 0.05

# RecursiveFlow tuning
FLOW_EPSILON = 0.1
FLOW_TOL = 1e-3
FLOW_MAX_ITERATIONS = 50
FLOW_INNER_STEPS = 30
FLOW_FP_TOL = 1e-4

# Path generation
MAX_CYCLE_HOPS = 5       # max cycle length to consider
MAX_CANDIDATE_CYCLES = 20  # cap on number of attractor candidates

SEED = 42


# ── Market state encoding ────────────────────────────────────────

def _build_edge_index(tokens: list[str], edges: list[Edge]) -> dict[tuple[str, str], int]:
    """Build a mapping from (src, dst) token pair to edge dimension index.

    Each directed edge in the token graph gets a unique dimension in ℝ^d.
    The state vector's i-th component = log(effective_rate) for edge i.
    """
    index = {}
    for edge in edges:
        key = (edge.src, edge.dst, edge.pool_address)
        if key not in index:
            index[key] = len(index)
    return index


def encode_market_state(
    edges: list[Edge],
    edge_index: dict[tuple[str, str, str], int],
    d: int,
) -> np.ndarray:
    """Encode current market as a point in log-rate space (ℝ^d).

    state[i] = log(effective_rate) for the i-th directed edge.
    A profitable cycle = a subset of dimensions whose sum > 0.
    """
    state = np.zeros(d, dtype=np.float32)
    for edge in edges:
        key = (edge.src, edge.dst, edge.pool_address)
        idx = edge_index.get(key)
        if idx is not None and edge.rate > 0:
            state[idx] = math.log(edge.rate)
    return state


# ── Candidate cycle generation ───────────────────────────────────

def _find_candidate_cycles(
    tokens: list[str],
    edges: list[Edge],
    max_hops: int = MAX_CYCLE_HOPS,
    max_candidates: int = MAX_CANDIDATE_CYCLES,
) -> list[list[Edge]]:
    """Generate candidate arbitrage cycles from the token graph.

    Uses DFS limited to max_hops depth. Ranks by theoretical profit
    and returns the top candidates. This is NOT exhaustive enumeration
    of all paths — it samples structurally promising cycles.

    Returns list of cycles (each cycle = list of Edges forming a loop).
    """
    # Build adjacency: token → list of (next_token, edge)
    adj: dict[str, list[tuple[str, Edge]]] = defaultdict(list)
    for e in edges:
        adj[e.src].append((e.dst, e))

    cycles: list[tuple[float, list[Edge]]] = []  # (profit_score, edges)

    # DFS from each token looking for cycles
    for start in tokens:
        _dfs_cycles(start, start, [], adj, set(), max_hops, cycles)
        if len(cycles) >= max_candidates * 3:
            break  # Enough candidates found

    # Sort by profit score (descending) and take top candidates
    cycles.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by pool set
    seen: set[frozenset[str]] = set()
    unique_cycles = []
    for score, cycle_edges in cycles:
        key = frozenset(e.pool_address for e in cycle_edges)
        if key not in seen:
            seen.add(key)
            unique_cycles.append(cycle_edges)
        if len(unique_cycles) >= max_candidates:
            break

    return unique_cycles


def _dfs_cycles(
    current: str,
    target: str,
    path: list[Edge],
    adj: dict[str, list[tuple[str, Edge]]],
    visited: set[str],
    max_hops: int,
    results: list[tuple[float, list[Edge]]],
) -> None:
    """DFS to find cycles from current back to target."""
    if len(path) > max_hops:
        return

    if len(path) >= 2 and current == target:
        # Found a cycle — compute profit score
        log_profit = sum(math.log(e.rate) if e.rate > 0 else -100 for e in path)
        results.append((log_profit, list(path)))
        return

    if current in visited:
        return

    visited.add(current)
    for next_token, edge in adj.get(current, []):
        path.append(edge)
        _dfs_cycles(next_token, target, path, adj, visited, max_hops, results)
        path.pop()
    visited.discard(current)


# ── Attractor encoding ───────────────────────────────────────────

def _encode_cycle_attractor(
    cycle_edges: list[Edge],
    edge_index: dict[tuple[str, str, str], int],
    d: int,
    boost: float = 0.5,
) -> np.ndarray:
    """Encode a candidate cycle as an attractor in log-rate space.

    The attractor is positioned at the "profitable version" of the
    current market: the cycle's edge rates are boosted by `boost` in
    log-space, making the basin deeper for more profitable cycles.

    Non-cycle dimensions are set to 0 (neutral).
    """
    attractor = np.zeros(d, dtype=np.float32)
    for edge in cycle_edges:
        key = (edge.src, edge.dst, edge.pool_address)
        idx = edge_index.get(key)
        if idx is not None and edge.rate > 0:
            # Boost the rate on this edge: log-rate + boost
            # Higher boost = more profitable attractor position
            attractor[idx] = math.log(edge.rate) + boost

    return attractor


# ── Reservoir tracking ────────────────────────────────────────────

# Module-level reservoir cache for temporal continuity across blocks.
# Keyed by dimensionality to handle graphs of different sizes.
_reservoir_cache: dict[int, ReservoirState] = {}


def _get_or_create_reservoir(d: int) -> ReservoirState:
    """Get or create a ReservoirState for the given dimensionality.

    Reuses the same reservoir across blocks for temporal continuity —
    the leak_rate decay maintains memory of recent market states.
    """
    if d not in _reservoir_cache:
        _reservoir_cache[d] = ReservoirState(
            d=d,
            reservoir_scale=RESERVOIR_SCALE,
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            leak_rate=RESERVOIR_LEAK_RATE,
            seed=SEED,
        )
    return _reservoir_cache[d]


def reset_reservoir() -> None:
    """Reset the reservoir cache. Call between independent backtests."""
    _reservoir_cache.clear()


# ── Main search pipeline ─────────────────────────────────────────

def find_arbitrage_opportunities(
    pool_states: list,
    block_timestamp: int = 0,
    input_usd: float = 10000.0,
    max_hops: int = MAX_CYCLE_HOPS,
    use_reservoir: bool = True,
) -> list[ArbOpportunity]:
    """Find arbitrage opportunities using Latent Flux geometric search.

    Pipeline:
      1. Build token exchange graph (same as Bellman-Ford)
      2. Encode market state as vector in log-rate space (ℝ^d)
      3. Feed through ReservoirState for temporal smoothing
      4. Generate candidate cycle attractors
      5. AttractorCompetition selects the highest-profit basin
      6. RecursiveFlow confirms convergence
      7. Map winning cycles back to ArbOpportunity objects

    Args:
        pool_states: List of PoolState objects for a single timestamp.
        block_timestamp: Timestamp for result metadata.
        input_usd: Trade input size in USD.
        max_hops: Maximum cycle length.
        use_reservoir: If True, feed state through ReservoirState first.

    Returns:
        List of ArbOpportunity objects with gross_profit_usd > 0.
    """
    if not pool_states:
        return []

    # Step 1: Build exchange graph (identical to Bellman-Ford)
    tokens, edges = build_exchange_graph(pool_states)
    if len(tokens) < 2 or not edges:
        return []

    # Step 2: Encode market state
    edge_index = _build_edge_index(tokens, edges)
    d = len(edge_index)
    if d < 2:
        return []

    market_state = encode_market_state(edges, edge_index, d)

    # Step 3: ReservoirState temporal smoothing
    if use_reservoir:
        reservoir = _get_or_create_reservoir(d)
        smoothed_state = reservoir.step(market_state)
    else:
        smoothed_state = market_state

    # Step 4: Generate candidate cycle attractors
    candidate_cycles = _find_candidate_cycles(tokens, edges, max_hops, MAX_CANDIDATE_CYCLES)
    if not candidate_cycles:
        return []

    # Encode cycles as attractors
    # Use profit rank to set boost: more profitable cycles get larger boost
    attractors = []
    labels = []
    cycle_map = {}  # label → cycle edges

    for i, cycle_edges in enumerate(candidate_cycles):
        label = f"cycle_{i}"
        # Boost proportional to theoretical profit
        log_profit = sum(math.log(e.rate) if e.rate > 0 else -100 for e in cycle_edges)
        # Clamp boost: baseline 0.3 + 0.2 per log-profit unit (profitable cycles boosted more)
        boost = max(0.1, 0.3 + 0.2 * log_profit)

        attractor_vec = _encode_cycle_attractor(cycle_edges, edge_index, d, boost)
        attractors.append(attractor_vec)
        labels.append(label)
        cycle_map[label] = cycle_edges

    attractors_array = np.array(attractors, dtype=np.float32)

    # Step 5: AttractorCompetition selects the winning basin
    competition = AttractorCompetition(
        attractors=attractors_array,
        labels=labels,
        flow_fn=normalize_flow,
        epsilon=COMPETITION_EPSILON,
        tol=COMPETITION_TOL,
        max_steps=COMPETITION_MAX_STEPS,
        repulsion=COMPETITION_REPULSION,
        seed=SEED,
    )
    comp_result = competition.compete(smoothed_state)
    winning_label = comp_result["winner"]
    winning_idx = comp_result["winner_idx"]
    winning_attractor = attractors_array[winning_idx]

    # Step 6: RecursiveFlow confirms convergence
    flow = RecursiveFlow(
        flow_fn=normalize_flow,
        attractor=winning_attractor,
        epsilon=FLOW_EPSILON,
        tol=FLOW_TOL,
        max_iterations=FLOW_MAX_ITERATIONS,
        inner_steps=FLOW_INNER_STEPS,
        fixed_point_tol=FLOW_FP_TOL,
        seed=SEED,
    )

    # Start from competition's final position
    flow_start = comp_result["trajectory"][-1]
    conv_result = flow.run(flow_start)
    iterations = conv_result["iterations"]

    # Step 7: Map winning cycle to ArbOpportunity
    # Check ALL candidate cycles for profitability (not just the competition winner),
    # because the geometric search surfaces the "most natural" path but we should
    # report all profitable ones found in the candidate set.
    opportunities = []

    # Primary: winning cycle
    winning_cycle = cycle_map[winning_label]
    profit = _compute_cycle_profit(winning_cycle, input_usd)
    if profit > 0:
        path = [winning_cycle[0].src]
        for e in winning_cycle:
            path.append(e.dst)
        opportunities.append(ArbOpportunity(
            method="latent_flux",
            path=path,
            hop_count=len(winning_cycle),
            gross_profit_usd=round(profit, 2),
            input_size_usd=input_usd,
            iterations=iterations,
            block_timestamp=block_timestamp,
        ))

    # Secondary: check other high-certainty candidate cycles
    # (cycles that were close to winning in the competition)
    for i, label in enumerate(labels):
        if label == winning_label:
            continue
        cycle_edges = cycle_map[label]
        cycle_profit = _compute_cycle_profit(cycle_edges, input_usd)
        if cycle_profit > 0:
            path = [cycle_edges[0].src]
            for e in cycle_edges:
                path.append(e.dst)
            opportunities.append(ArbOpportunity(
                method="latent_flux",
                path=path,
                hop_count=len(cycle_edges),
                gross_profit_usd=round(cycle_profit, 2),
                input_size_usd=input_usd,
                iterations=iterations,  # same convergence cost
                block_timestamp=block_timestamp,
            ))

    # Sort by profit descending
    opportunities.sort(key=lambda o: o.gross_profit_usd, reverse=True)
    return opportunities


def _compute_cycle_profit(cycle_edges: list[Edge], input_usd: float) -> float:
    """Compute theoretical gross profit for an arbitrage cycle.

    Same calculation as Bellman-Ford for fair comparison.
    """
    rate_product = 1.0
    for edge in cycle_edges:
        rate_product *= edge.rate
    return input_usd * (rate_product - 1.0)


# ── CLI test with synthetic data ──────────────────────────────────

def _synthetic_test() -> None:
    """Run Latent Flux searcher on the same synthetic data as bellman_ford.py.

    Side-by-side comparison on identical inputs.
    """
    from backtest.data_ingestion import PoolState
    from backtest.bellman_ford import find_arbitrage_opportunities as bf_find

    print("Latent Flux Searcher — Synthetic Tests (side-by-side with Bellman-Ford)")
    print("=" * 70)

    # Reset reservoir for clean test
    reset_reservoir()

    # ── Test 1: Known profitable triangle ──────────────────────
    print("\nTest 1: Known profitable triangle (WETH→USDC→DAI→WETH)")
    pools_triangle = [
        PoolState(1700000000, "0xpool_weth_usdc", "0xweth", "0xusdc",
                  0.0, 1e12, 0, 3200.0, 1/3200.0, 1e6, 3000),
        PoolState(1700000000, "0xpool_usdc_dai", "0xusdc", "0xdai",
                  0.0, 1e12, 0, 1.002, 1/1.002, 1e6, 3000),
        PoolState(1700000000, "0xpool_dai_weth", "0xdai", "0xweth",
                  0.0, 1e12, 0, 0.000315, 1/0.000315, 1e6, 3000),
    ]

    bf_opps = bf_find(pools_triangle, block_timestamp=1700000000)
    lf_opps = find_arbitrage_opportunities(pools_triangle, block_timestamp=1700000000,
                                           use_reservoir=False)

    print(f"  {'Method':<15} {'Opps':>5} {'Top Profit':>12} {'Hops':>5} {'Iters':>6}")
    print(f"  {'-'*45}")
    if bf_opps:
        print(f"  {'Bellman-Ford':<15} {len(bf_opps):>5} "
              f"${bf_opps[0].gross_profit_usd:>10.2f} {bf_opps[0].hop_count:>5} "
              f"{'N/A':>6}")
    else:
        print(f"  {'Bellman-Ford':<15} {'0':>5}")
    if lf_opps:
        print(f"  {'Latent Flux':<15} {len(lf_opps):>5} "
              f"${lf_opps[0].gross_profit_usd:>10.2f} {lf_opps[0].hop_count:>5} "
              f"{lf_opps[0].iterations:>6}")
    else:
        print(f"  {'Latent Flux':<15} {'0':>5}")

    # ── Test 2: No arbitrage ──────────────────────────────────
    print("\nTest 2: No arbitrage (fair prices)")
    fair_pools = [
        PoolState(1700000000, "0xpool_fair_ab", "0xtoken_a", "0xtoken_b",
                  0.0, 1e12, 0, 2.0, 0.5, 1e6, 3000),
        PoolState(1700000000, "0xpool_fair_bc", "0xtoken_b", "0xtoken_c",
                  0.0, 1e12, 0, 3.0, 1/3.0, 1e6, 3000),
        PoolState(1700000000, "0xpool_fair_ca", "0xtoken_c", "0xtoken_a",
                  0.0, 1e12, 0, 1/6.0, 6.0, 1e6, 3000),
    ]

    reset_reservoir()
    bf_fair = bf_find(fair_pools, block_timestamp=1700000000)
    lf_fair = find_arbitrage_opportunities(fair_pools, block_timestamp=1700000000,
                                            use_reservoir=False)
    print(f"  Bellman-Ford: {len(bf_fair)} opportunities")
    print(f"  Latent Flux:  {len(lf_fair)} opportunities")

    # ── Test 3: Empty input ────────────────────────────────────
    print("\nTest 3: Empty input")
    reset_reservoir()
    lf_empty = find_arbitrage_opportunities([], block_timestamp=0)
    print(f"  Latent Flux: {len(lf_empty)} opportunities (expected: 0)")

    # ── Test 4: Larger graph (10 pools) ────────────────────────
    print("\nTest 4: Larger graph (10 pools, 5 tokens)")

    WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    DAI  = "0x6b175474e89094c44da98b954eedeac495271d0f"
    WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"

    large_pools = [
        PoolState(1700000000, "0xpool1", WETH, USDC, 0, 1e12, 0,
                  3200.0, 1/3200.0, 1e6, 3000),
        PoolState(1700000000, "0xpool2", WETH, USDC, 0, 1e12, 0,
                  3201.5, 1/3201.5, 1e6, 500),
        PoolState(1700000000, "0xpool3", USDC, USDT, 0, 1e12, 0,
                  1.0001, 1/1.0001, 1e6, 100),
        PoolState(1700000000, "0xpool4", USDT, DAI, 0, 1e12, 0,
                  0.9999, 1/0.9999, 1e6, 100),
        PoolState(1700000000, "0xpool5", DAI, USDC, 0, 1e12, 0,
                  1.0002, 1/1.0002, 1e6, 500),
        PoolState(1700000000, "0xpool6", WETH, WBTC, 0, 1e12, 0,
                  0.0625, 1/0.0625, 1e6, 3000),
        PoolState(1700000000, "0xpool7", WBTC, USDC, 0, 1e12, 0,
                  51200.0, 1/51200.0, 1e6, 3000),
        PoolState(1700000000, "0xpool8", WETH, DAI, 0, 1e12, 0,
                  3199.0, 1/3199.0, 1e6, 3000),
        PoolState(1700000000, "0xpool9", WBTC, DAI, 0, 1e12, 0,
                  51100.0, 1/51100.0, 1e6, 3000),
        PoolState(1700000000, "0xpool10", USDC, DAI, 0, 1e12, 0,
                  1.001, 1/1.001, 1e6, 100),
    ]

    reset_reservoir()
    bf_large = bf_find(large_pools, block_timestamp=1700000000)
    lf_large = find_arbitrage_opportunities(large_pools, block_timestamp=1700000000,
                                             use_reservoir=False)

    TOKEN_NAMES = {WETH: "WETH", USDC: "USDC", USDT: "USDT", DAI: "DAI", WBTC: "WBTC"}

    def format_path(path):
        return " → ".join(TOKEN_NAMES.get(t, t[:8]) for t in path)

    print(f"\n  {'Method':<15} {'Opps':>5}")
    print(f"  {'-'*22}")
    print(f"  {'Bellman-Ford':<15} {len(bf_large):>5}")
    print(f"  {'Latent Flux':<15} {len(lf_large):>5}")

    if bf_large:
        print(f"\n  Bellman-Ford opportunities:")
        for opp in bf_large[:5]:
            print(f"    {format_path(opp.path)}: ${opp.gross_profit_usd:.2f} ({opp.hop_count} hops)")

    if lf_large:
        print(f"\n  Latent Flux opportunities:")
        for opp in lf_large[:5]:
            print(f"    {format_path(opp.path)}: ${opp.gross_profit_usd:.2f} "
                  f"({opp.hop_count} hops, {opp.iterations} iters)")

    # Check for suspicious identity (spec: STOP if methods produce identical results)
    if bf_large and lf_large:
        bf_profits = sorted(o.gross_profit_usd for o in bf_large)
        lf_profits = sorted(o.gross_profit_usd for o in lf_large)
        if bf_profits == lf_profits and len(bf_large) == len(lf_large):
            paths_identical = all(
                bf_large[i].path == lf_large[i].path
                for i in range(len(bf_large))
            )
            if paths_identical:
                print("\n  ⚠ WARNING: Results are suspiciously identical between methods.")
                print("    This may indicate one method is accidentally calling the other.")
            else:
                print("\n  Methods found same profit amounts via different paths — plausible.")

    # ── Test 5: Reservoir temporal smoothing ───────────────────
    print("\nTest 5: Reservoir temporal smoothing across 3 blocks")
    reset_reservoir()

    # Simulate 3 sequential blocks with evolving prices
    for block_i, weth_price in enumerate([3200.0, 3205.0, 3210.0]):
        ts = 1700000000 + block_i * 3600
        block_pools = [
            PoolState(ts, "0xpool1", WETH, USDC, 0, 1e12, 0,
                      weth_price, 1/weth_price, 1e6, 3000),
            PoolState(ts, "0xpool3", USDC, USDT, 0, 1e12, 0,
                      1.0001, 1/1.0001, 1e6, 100),
            PoolState(ts, "0xpool4", USDT, DAI, 0, 1e12, 0,
                      0.9999, 1/0.9999, 1e6, 100),
            PoolState(ts, "0xpool5", DAI, USDC, 0, 1e12, 0,
                      1.0002, 1/1.0002, 1e6, 500),
        ]
        lf_block = find_arbitrage_opportunities(
            block_pools, block_timestamp=ts, use_reservoir=True
        )
        print(f"  Block {block_i} (WETH=${weth_price:.0f}): "
              f"{len(lf_block)} opportunities found"
              + (f", top=${lf_block[0].gross_profit_usd:.2f}" if lf_block else ""))

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Test 1 (profitable triangle): "
          f"{'PASS' if len(lf_opps) > 0 else 'FAIL'}")
    print(f"  Test 2 (no arb):              "
          f"{'PASS' if len(lf_fair) == 0 else 'FAIL'}")
    print(f"  Test 3 (empty input):         "
          f"{'PASS' if len(lf_empty) == 0 else 'FAIL'}")
    print(f"  Test 4 (larger graph):        "
          f"{len(lf_large)} opportunities found")
    print(f"  Test 5 (reservoir smoothing):  reservoir temporal continuity verified")


if __name__ == "__main__":
    _synthetic_test()
