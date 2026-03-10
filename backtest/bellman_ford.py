"""Bellman-Ford Arbitrage Detector — classical MEV path-finding baseline.

Given a set of PoolState snapshots representing a single hourly block,
constructs a directed graph of token exchange rates and uses Bellman-Ford
to detect negative-weight cycles (= profitable arbitrage loops).

No numpy. Uses Python stdlib only: math, heapq, collections.

Usage:
    from backtest.bellman_ford import find_arbitrage_opportunities
    from backtest.data_ingestion import PoolState

    opportunities = find_arbitrage_opportunities(pool_states, input_usd=10000.0)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


# Re-use PoolState from data_ingestion without importing it at module level
# to keep this file self-contained for testing. The harness passes PoolState objects.


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""
    method: str             # "bellman_ford" or "latent_flux"
    path: list[str]         # token addresses in order
    hop_count: int
    gross_profit_usd: float
    input_size_usd: float   # always 10000.0
    iterations: int          # convergence cost (N/A for BF, set to relaxation count)
    block_timestamp: int


# ── Fee tier mapping ──────────────────────────────────────────────

# Uniswap V3 fee tiers in hundredths of a bip → decimal fee rate
FEE_TIER_TO_RATE = {
    100: 0.0001,     # 0.01%
    500: 0.00050,    # 0.05%
    3000: 0.003,     # 0.3%
    10000: 0.01,     # 1.0%
}


def _fee_rate(fee_tier: int) -> float:
    """Convert fee tier to decimal fee rate. Unknown tiers default to 0.3%."""
    return FEE_TIER_TO_RATE.get(fee_tier, 0.003)


# ── Graph construction ───────────────────────────────────────────

@dataclass
class Edge:
    """A directed edge in the token exchange graph."""
    src: str            # source token address
    dst: str            # destination token address
    rate: float         # exchange rate (units of dst per unit of src, after fee)
    weight: float       # -log(rate) for Bellman-Ford
    pool_address: str   # source pool
    fee_rate: float     # decimal fee


def build_exchange_graph(pool_states: list) -> tuple[list[str], list[Edge]]:
    """Build a directed graph from pool states.

    Each pool creates two directed edges (one per swap direction).
    Edge weight = -log(exchange_rate × (1 - fee)) for Bellman-Ford.

    A negative-weight cycle in this graph corresponds to a profitable
    arbitrage loop: the product of exchange rates around the cycle > 1.

    Args:
        pool_states: List of PoolState objects for a single timestamp.

    Returns:
        (tokens, edges) — list of unique token addresses and directed edges.
    """
    edges: list[Edge] = []
    token_set: set[str] = set()

    for ps in pool_states:
        fee = _fee_rate(ps.fee_tier)

        # Direction 1: token0 → token1
        # Rate: how many token1 you get per token0 (after fee)
        if ps.token0_price > 0:
            # token0_price = token1 per token0 (the price of token0 in token1 terms)
            rate_0_to_1 = ps.token0_price * (1.0 - fee)
            if rate_0_to_1 > 0:
                weight = -math.log(rate_0_to_1)
                edges.append(Edge(
                    src=ps.token0,
                    dst=ps.token1,
                    rate=rate_0_to_1,
                    weight=weight,
                    pool_address=ps.pool_address,
                    fee_rate=fee,
                ))
                token_set.add(ps.token0)
                token_set.add(ps.token1)

        # Direction 2: token1 → token0
        if ps.token1_price > 0:
            # token1_price = token0 per token1 (the price of token1 in token0 terms)
            rate_1_to_0 = ps.token1_price * (1.0 - fee)
            if rate_1_to_0 > 0:
                weight = -math.log(rate_1_to_0)
                edges.append(Edge(
                    src=ps.token1,
                    dst=ps.token0,
                    rate=rate_1_to_0,
                    weight=weight,
                    pool_address=ps.pool_address,
                    fee_rate=fee,
                ))
                token_set.add(ps.token0)
                token_set.add(ps.token1)

    tokens = sorted(token_set)
    return tokens, edges


# ── Bellman-Ford negative cycle detection ─────────────────────────

def _bellman_ford_detect_cycles(
    tokens: list[str],
    edges: list[Edge],
    max_hops: int = 6,
) -> list[list[Edge]]:
    """Run Bellman-Ford from every token to detect negative-weight cycles.

    Standard Bellman-Ford: relax all edges |V|-1 times. If any edge can
    still be relaxed on the |V|th pass, a negative cycle exists.

    We limit cycle length to max_hops for practical arbitrage paths.

    Returns list of cycles, where each cycle is a list of edges.
    """
    if not tokens or not edges:
        return []

    n = len(tokens)
    token_idx = {t: i for i, t in enumerate(tokens)}

    # Build adjacency for predecessor tracking
    adj: dict[int, list[tuple[int, Edge]]] = defaultdict(list)
    for e in edges:
        src_i = token_idx.get(e.src)
        dst_i = token_idx.get(e.dst)
        if src_i is not None and dst_i is not None:
            adj[src_i].append((dst_i, e))

    found_cycles: list[list[Edge]] = []
    seen_cycle_keys: set[frozenset[str]] = set()

    # Run Bellman-Ford from each source token
    for source in range(n):
        dist = [math.inf] * n
        pred: list[tuple[int, Edge | None]] = [(-1, None)] * n
        dist[source] = 0.0

        # Relax edges min(n-1, max_hops) times
        relaxation_rounds = min(n - 1, max_hops)
        for _ in range(relaxation_rounds):
            updated = False
            for u in range(n):
                if dist[u] == math.inf:
                    continue
                for v, edge in adj[u]:
                    new_dist = dist[u] + edge.weight
                    if new_dist < dist[v] - 1e-10:
                        dist[v] = new_dist
                        pred[v] = (u, edge)
                        updated = True
            if not updated:
                break

        # Check for negative cycles (one more relaxation pass)
        for u in range(n):
            if dist[u] == math.inf:
                continue
            for v, edge in adj[u]:
                if dist[u] + edge.weight < dist[v] - 1e-10:
                    # Negative cycle detected. Trace it back.
                    cycle_edges = _trace_cycle(v, pred, tokens, max_hops)
                    if cycle_edges:
                        # Deduplicate: same set of pools = same cycle
                        cycle_key = frozenset(e.pool_address for e in cycle_edges)
                        if cycle_key not in seen_cycle_keys:
                            seen_cycle_keys.add(cycle_key)
                            found_cycles.append(cycle_edges)

    return found_cycles


def _trace_cycle(
    start: int,
    pred: list[tuple[int, "Edge | None"]],
    tokens: list[str],
    max_hops: int,
) -> list["Edge"] | None:
    """Trace a negative-weight cycle from a node involved in one.

    Walks predecessors to find the cycle. Returns the cycle edges
    or None if we can't reconstruct a valid cycle.
    """
    n = len(tokens)

    # Walk back max_hops steps to make sure we're on the cycle
    node = start
    for _ in range(n):
        prev_node, _ = pred[node]
        if prev_node < 0:
            return None
        node = prev_node

    # Now node is on the cycle. Walk until we return to node.
    cycle_start = node
    cycle_edges: list[Edge] = []
    current = node

    while True:
        prev_node, edge = pred[current]
        if edge is None or prev_node < 0:
            return None
        cycle_edges.append(edge)
        current = prev_node
        if current == cycle_start:
            break
        if len(cycle_edges) > max_hops:
            return None  # Cycle too long

    cycle_edges.reverse()  # Put in forward order

    # Validate: cycle must be at least 2 hops and form a proper loop
    if len(cycle_edges) < 2:
        return None

    # Verify the cycle is actually closed
    if cycle_edges[0].src != cycle_edges[-1].dst:
        return None

    return cycle_edges


# ── Profit calculation ────────────────────────────────────────────

def _compute_cycle_profit(
    cycle_edges: list[Edge],
    input_usd: float,
) -> float:
    """Compute theoretical gross profit for an arbitrage cycle.

    Multiplies exchange rates around the cycle. If the product > 1,
    the cycle is profitable.

    Args:
        cycle_edges: Edges forming a complete cycle.
        input_usd: Input trade size in USD.

    Returns:
        Gross profit in USD (positive = profitable, negative = loss).
    """
    # Product of rates around the cycle
    rate_product = 1.0
    for edge in cycle_edges:
        rate_product *= edge.rate

    # Profit = input × (rate_product - 1)
    return input_usd * (rate_product - 1.0)


# ── Public API ────────────────────────────────────────────────────

def find_arbitrage_opportunities(
    pool_states: list,
    block_timestamp: int = 0,
    input_usd: float = 10000.0,
    max_hops: int = 6,
) -> list[ArbOpportunity]:
    """Find arbitrage opportunities using Bellman-Ford negative cycle detection.

    Args:
        pool_states: List of PoolState objects for a single timestamp.
        block_timestamp: Timestamp for this block (for result metadata).
        input_usd: Trade input size in USD.
        max_hops: Maximum cycle length to consider.

    Returns:
        List of ArbOpportunity objects with gross_profit_usd > 0.
        Returns empty list (not error) when no arbitrage exists.
    """
    if not pool_states:
        return []

    # Build exchange graph
    tokens, edges = build_exchange_graph(pool_states)

    if len(tokens) < 2 or not edges:
        return []

    # Run Bellman-Ford
    cycles = _bellman_ford_detect_cycles(tokens, edges, max_hops)

    # Count total relaxations for the iterations field
    n = len(tokens)
    total_relaxations = n * len(edges)  # upper bound on work done

    # Convert cycles to ArbOpportunity objects
    opportunities = []
    for cycle_edges in cycles:
        profit = _compute_cycle_profit(cycle_edges, input_usd)

        if profit > 0:
            # Build path: token addresses in order
            path = [cycle_edges[0].src]
            for e in cycle_edges:
                path.append(e.dst)

            opportunities.append(ArbOpportunity(
                method="bellman_ford",
                path=path,
                hop_count=len(cycle_edges),
                gross_profit_usd=round(profit, 2),
                input_size_usd=input_usd,
                iterations=total_relaxations,
                block_timestamp=block_timestamp,
            ))

    # Sort by profit descending
    opportunities.sort(key=lambda o: o.gross_profit_usd, reverse=True)

    return opportunities


# ── CLI test with synthetic data ──────────────────────────────────

def _synthetic_test() -> None:
    """Run Bellman-Ford on synthetic pool data to verify correctness.

    Creates a triangle of pools with a known arbitrage opportunity:
      WETH → USDC → DAI → WETH with a 0.5% edge (should be profitable
      after 0.3% fees on each leg).

    Also tests: no-arb scenario, disconnected graph, empty input.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backtest.data_ingestion import PoolState

    print("Bellman-Ford Baseline — Synthetic Tests")
    print("=" * 50)

    # ── Test 1: Known profitable triangle ──────────────────────
    print("\nTest 1: Known profitable triangle (WETH→USDC→DAI→WETH)")

    # Create a profitable triangle:
    # WETH → USDC: 1 WETH = 3200 USDC (token0_price)
    # USDC → DAI:  1 USDC = 1.002 DAI
    # DAI → WETH:  1 DAI = 0.000315 WETH (= 1/3174.6)
    #
    # Product: 3200 * 1.002 * 0.000315 = 1.01008
    # After 0.3% fee each leg: 1.01008 * (0.997)^3 ≈ 1.00101 → profitable
    pools = [
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_weth_usdc",
            token0="0xweth",
            token1="0xusdc",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=3200.0,      # 1 WETH = 3200 USDC
            token1_price=1/3200.0,    # 1 USDC = 0.0003125 WETH
            volume_usd=1e6,
            fee_tier=3000,
        ),
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_usdc_dai",
            token0="0xusdc",
            token1="0xdai",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=1.002,       # 1 USDC = 1.002 DAI
            token1_price=1/1.002,     # 1 DAI = 0.998 USDC
            volume_usd=1e6,
            fee_tier=3000,
        ),
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_dai_weth",
            token0="0xdai",
            token1="0xweth",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=0.000315,    # 1 DAI = 0.000315 WETH
            token1_price=1/0.000315,  # 1 WETH = 3174.6 DAI
            volume_usd=1e6,
            fee_tier=3000,
        ),
    ]

    opps = find_arbitrage_opportunities(pools, block_timestamp=1700000000)
    if opps:
        for opp in opps:
            path_str = " → ".join(t[:8] + "..." for t in opp.path)
            print(f"  FOUND: {path_str}")
            print(f"    Hops: {opp.hop_count}, Profit: ${opp.gross_profit_usd:.2f}, "
                  f"Iterations: {opp.iterations}")
    else:
        print("  No opportunities found")

    # Verify the math manually
    rate_product = (3200.0 * 0.997) * (1.002 * 0.997) * (0.000315 * 0.997)
    expected_profit = 10000.0 * (rate_product - 1.0)
    print(f"  Expected rate product: {rate_product:.6f}")
    print(f"  Expected profit: ${expected_profit:.2f}")

    # ── Test 2: No arbitrage (fair prices) ─────────────────────
    print("\nTest 2: No arbitrage (fair prices)")
    fair_pools = [
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_fair_ab",
            token0="0xtoken_a",
            token1="0xtoken_b",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=2.0,
            token1_price=0.5,
            volume_usd=1e6,
            fee_tier=3000,
        ),
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_fair_bc",
            token0="0xtoken_b",
            token1="0xtoken_c",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=3.0,
            token1_price=1/3.0,
            volume_usd=1e6,
            fee_tier=3000,
        ),
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_fair_ca",
            token0="0xtoken_c",
            token1="0xtoken_a",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=1/6.0,  # C→A: 1/6 (fair: A=2B, B=3C, so A=6C, C=A/6)
            token1_price=6.0,
            volume_usd=1e6,
            fee_tier=3000,
        ),
    ]

    opps_fair = find_arbitrage_opportunities(fair_pools, block_timestamp=1700000000)
    print(f"  Opportunities found: {len(opps_fair)} (expected: 0)")

    # ── Test 3: Empty input ────────────────────────────────────
    print("\nTest 3: Empty input")
    opps_empty = find_arbitrage_opportunities([], block_timestamp=0)
    print(f"  Opportunities found: {len(opps_empty)} (expected: 0)")

    # ── Test 4: Disconnected graph (no path between tokens) ───
    print("\nTest 4: Disconnected graph")
    disconnected = [
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_island1",
            token0="0xisland_a",
            token1="0xisland_b",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=1.5,
            token1_price=1/1.5,
            volume_usd=1e6,
            fee_tier=3000,
        ),
        PoolState(
            block_timestamp=1700000000,
            pool_address="0xpool_island2",
            token0="0xisland_c",
            token1="0xisland_d",
            sqrt_price=0.0,
            liquidity=1e12,
            tick=0,
            token0_price=2.5,
            token1_price=1/2.5,
            volume_usd=1e6,
            fee_tier=3000,
        ),
    ]
    opps_disc = find_arbitrage_opportunities(disconnected, block_timestamp=1700000000)
    print(f"  Opportunities found: {len(opps_disc)} (expected: 0)")

    # ── Test 5: Larger graph with 10 pools ─────────────────────
    print("\nTest 5: Larger graph (10 pools, realistic token set)")

    # Realistic token set with known addresses (abbreviated)
    WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    DAI  = "0x6b175474e89094c44da98b954eedeac495271d0f"
    WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"

    large_pools = [
        # WETH/USDC pools at different fee tiers
        PoolState(1700000000, "0xpool1", WETH, USDC, 0, 1e12, 0,
                  3200.0, 1/3200.0, 1e6, 3000),
        PoolState(1700000000, "0xpool2", WETH, USDC, 0, 1e12, 0,
                  3201.5, 1/3201.5, 1e6, 500),
        # USDC/USDT
        PoolState(1700000000, "0xpool3", USDC, USDT, 0, 1e12, 0,
                  1.0001, 1/1.0001, 1e6, 100),
        # USDT/DAI
        PoolState(1700000000, "0xpool4", USDT, DAI, 0, 1e12, 0,
                  0.9999, 1/0.9999, 1e6, 100),
        # DAI/USDC
        PoolState(1700000000, "0xpool5", DAI, USDC, 0, 1e12, 0,
                  1.0002, 1/1.0002, 1e6, 500),
        # WETH/WBTC
        PoolState(1700000000, "0xpool6", WETH, WBTC, 0, 1e12, 0,
                  0.0625, 1/0.0625, 1e6, 3000),
        # WBTC/USDC
        PoolState(1700000000, "0xpool7", WBTC, USDC, 0, 1e12, 0,
                  51200.0, 1/51200.0, 1e6, 3000),
        # WETH/DAI
        PoolState(1700000000, "0xpool8", WETH, DAI, 0, 1e12, 0,
                  3199.0, 1/3199.0, 1e6, 3000),
        # WBTC/DAI
        PoolState(1700000000, "0xpool9", WBTC, DAI, 0, 1e12, 0,
                  51100.0, 1/51100.0, 1e6, 3000),
        # USDC/DAI (another fee tier)
        PoolState(1700000000, "0xpool10", USDC, DAI, 0, 1e12, 0,
                  1.001, 1/1.001, 1e6, 100),
    ]

    opps_large = find_arbitrage_opportunities(large_pools, block_timestamp=1700000000)
    print(f"  Tokens: {len(set(p.token0 for p in large_pools) | set(p.token1 for p in large_pools))}")
    print(f"  Edges: {len(build_exchange_graph(large_pools)[1])}")
    print(f"  Opportunities found: {len(opps_large)}")
    for opp in opps_large[:5]:  # Show top 5
        symbols = []
        for t in opp.path:
            if t == WETH: symbols.append("WETH")
            elif t == USDC: symbols.append("USDC")
            elif t == USDT: symbols.append("USDT")
            elif t == DAI: symbols.append("DAI")
            elif t == WBTC: symbols.append("WBTC")
            else: symbols.append(t[:8])
        print(f"    {' → '.join(symbols)}: ${opp.gross_profit_usd:.2f} "
              f"({opp.hop_count} hops)")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Test 1 (profitable triangle): {'PASS' if len(opps) > 0 else 'FAIL'}")
    print(f"  Test 2 (no arb):              {'PASS' if len(opps_fair) == 0 else 'FAIL'}")
    print(f"  Test 3 (empty input):         {'PASS' if len(opps_empty) == 0 else 'FAIL'}")
    print(f"  Test 4 (disconnected):        {'PASS' if len(opps_disc) == 0 else 'FAIL'}")
    print(f"  Test 5 (larger graph):        {len(opps_large)} opportunities found")


if __name__ == "__main__":
    _synthetic_test()
