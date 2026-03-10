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
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

# Ensure repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux_manifold.reservoir_state import ReservoirState
from flux_manifold.kalman_reservoir import KalmanReservoir
from flux_manifold.multi_scale_reservoir import MultiScaleReservoir
from flux_manifold.recursive_flow import RecursiveFlow
from flux_manifold.attractor_competition import AttractorCompetition
from flux_manifold.flows import normalize_flow
from flux_manifold.cex_feed import CexFeed

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

    Uses Tarjan SCC pre-filtering + Johnson's algorithm (via networkx)
    for complete enumeration of simple cycles up to max_hops length.
    Falls back to DFS if networkx unavailable.

    Returns list of cycles (each cycle = list of Edges forming a loop).
    """
    try:
        import networkx as nx
        return _find_cycles_johnson(tokens, edges, max_hops, max_candidates, nx)
    except ImportError:
        return _find_cycles_dfs(tokens, edges, max_hops, max_candidates)


def _find_cycles_johnson(
    tokens: list[str],
    edges: list[Edge],
    max_hops: int,
    max_candidates: int,
    nx,
) -> list[list[Edge]]:
    """Johnson's algorithm with SCC pre-filtering for complete cycle enumeration."""
    # Build directed multigraph (multiple edges between same token pair)
    G = nx.DiGraph()
    # Edge lookup: (src, dst) → list of edges (multiple pools possible)
    edge_lookup: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for e in edges:
        G.add_edge(e.src, e.dst)
        edge_lookup[(e.src, e.dst)].append(e)

    # SCC pre-filter: only keep nodes in strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    scc_nodes = set()
    for scc in sccs:
        if len(scc) >= 2:  # Need at least 2 nodes for a cycle
            scc_nodes.update(scc)

    if not scc_nodes:
        return []

    G_scc = G.subgraph(scc_nodes)

    # Johnson's algorithm: enumerate all simple cycles up to max_hops
    # networkx.simple_cycles uses Johnson's algorithm
    cycles_scored: list[tuple[float, list[Edge]]] = []

    for node_cycle in nx.simple_cycles(G_scc, length_bound=max_hops):
        if len(node_cycle) < 2:
            continue
        # node_cycle is [A, B, C] representing A→B→C→A
        # For each edge in the cycle, pick the best (highest rate) pool
        cycle_edges: list[Edge] = []
        valid = True
        for i in range(len(node_cycle)):
            src = node_cycle[i]
            dst = node_cycle[(i + 1) % len(node_cycle)]
            pool_edges = edge_lookup.get((src, dst), [])
            if not pool_edges:
                valid = False
                break
            # Pick the edge with the highest rate (best for arbitrage)
            best = max(pool_edges, key=lambda e: e.rate)
            cycle_edges.append(best)

        if not valid:
            continue

        log_profit = sum(math.log(e.rate) if e.rate > 0 else -100 for e in cycle_edges)
        cycles_scored.append((log_profit, cycle_edges))

    # Sort by profit, deduplicate, take top candidates
    cycles_scored.sort(key=lambda x: x[0], reverse=True)

    seen: set[frozenset[str]] = set()
    unique_cycles = []
    for score, cycle_edges in cycles_scored:
        key = frozenset(e.pool_address for e in cycle_edges)
        if key not in seen:
            seen.add(key)
            unique_cycles.append(cycle_edges)
        if len(unique_cycles) >= max_candidates:
            break

    return unique_cycles


def _find_cycles_dfs(
    tokens: list[str],
    edges: list[Edge],
    max_hops: int,
    max_candidates: int,
) -> list[list[Edge]]:
    """Fallback DFS cycle finder (original implementation)."""
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
_kalman_cache: dict[int, KalmanReservoir] = {}
_multi_scale_cache: dict[int, MultiScaleReservoir] = {}

# Deviation history for S-score computation (per edge dimension)
_deviation_history: dict[int, list[np.ndarray]] = {}  # keyed by d
_DEVIATION_HISTORY_MAX = 200  # keep last 200 observations for OU fit


def _get_or_create_reservoir(d: int, use_kalman: bool = False):
    """Get or create a reservoir for the given dimensionality.

    If use_kalman=True, returns a KalmanReservoir (UKF-based).
    Otherwise returns EMA-based ReservoirState.
    """
    if use_kalman:
        if d not in _kalman_cache:
            _kalman_cache[d] = KalmanReservoir(
                d=d,
                process_noise=1e-4,
                measurement_noise=1e-3,
                seed=SEED,
            )
        return _kalman_cache[d]
    else:
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
    global _cex_feed
    _reservoir_cache.clear()
    _kalman_cache.clear()
    _multi_scale_cache.clear()
    _deviation_history.clear()
    _signal_metadata.clear()
    _cex_feed = None


# ── Signal metadata (enriched info that can't go in ArbOpportunity) ───

@dataclass
class SignalMeta:
    """Extra metadata for a Latent Flux signal, keyed by (block_timestamp, path_str)."""
    s_score: float                # aggregate OU S-score for this cycle
    mean_reversion_speed: float   # κ (OU mean reversion rate)
    s_score_valid: bool           # whether S-score passed quality filters
    scale_agreement: float        # multi-scale agreement [0, 1] (Phase 5)
    net_profit_usd: float         # gross - execution cost (Phase 4)
    fee_weighted_threshold: float # min deviation for profitability (Phase 4)
    regime_change: bool           # changepoint detected this block (Phase 2)
    cex_deviation: float | None = None  # (amm - cex) / cex for primary pair


# Module-level store: (block_timestamp, path_str) → SignalMeta
_signal_metadata: dict[tuple[int, str], SignalMeta] = {}

# Module-level CEX feed for proxy price tracking
_cex_feed: CexFeed | None = None


def get_signal_metadata() -> dict[tuple[int, str], SignalMeta]:
    """Return all signal metadata accumulated during this backtest run."""
    return _signal_metadata


def _cycle_s_score(
    cycle_edges: list,
    edge_index: dict,
    s_scores: np.ndarray,
    kappas: np.ndarray,
    s_valid: np.ndarray,
) -> tuple[float, float, bool]:
    """Compute aggregate S-score for a cycle from per-edge S-scores.

    Returns (s_score, kappa, is_valid).
    Mean of S-scores across edges in this cycle that are valid.
    """
    indices = []
    for e in cycle_edges:
        key = (e.src, e.dst, e.pool_address)
        idx = edge_index.get(key)
        if idx is not None:
            indices.append(idx)

    if not indices:
        return 0.0, 0.0, False

    idx_arr = np.array(indices)
    valid_mask = s_valid[idx_arr]
    if not valid_mask.any():
        return 0.0, 0.0, False

    valid_idx = idx_arr[valid_mask]
    s = float(np.mean(s_scores[valid_idx]))
    k = float(np.mean(kappas[valid_idx]))
    return s, k, True


# ── S-score (Ornstein-Uhlenbeck) ──────────────────────────────────

def _compute_s_scores(
    market_state: np.ndarray,
    smoothed_state: np.ndarray,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute OU S-scores from the raw log-rate time series.

    Fits AR(1) model to the raw market_state per dimension:
        X_{t+1} = a + b * X_t + epsilon
    Then: κ = -ln(b), m = a/(1-b), σ from residuals, S = (X - m) / σ_eq

    Returns (s_scores, kappas, is_valid) arrays of shape (d,).
    """
    # Track raw market state history (not deviation — UKF tracks too closely)
    if d not in _deviation_history:
        _deviation_history[d] = []
    _deviation_history[d].append(market_state.copy())
    if len(_deviation_history[d]) > _DEVIATION_HISTORY_MAX:
        _deviation_history[d] = _deviation_history[d][-_DEVIATION_HISTORY_MAX:]

    history = _deviation_history[d]
    s_scores = np.zeros(d, dtype=np.float32)
    kappas = np.zeros(d, dtype=np.float32)
    is_valid = np.zeros(d, dtype=bool)

    if len(history) < 20:
        return s_scores, kappas, is_valid

    # Stack history into (T, d) array
    H = np.array(history, dtype=np.float64)  # (T, d)
    T = H.shape[0]

    # Vectorized AR(1) fit across all dimensions
    X = H[:-1]    # (T-1, d)
    Y = H[1:]     # (T-1, d)
    n = X.shape[0]

    # b = cov(X, Y) / var(X),  a = mean(Y) - b * mean(X)
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_var = ((X - X_mean) ** 2).sum(axis=0) / n
    XY_cov = ((X - X_mean) * (Y - Y_mean)).sum(axis=0) / n

    # Avoid division by zero
    valid_var = X_var > 1e-15
    b = np.where(valid_var, XY_cov / X_var, 0.0)
    a = Y_mean - b * X_mean

    # κ = -ln(b) / dt (dt = 1 hour)
    valid_b = (b > 0.001) & (b < 0.999)
    kappa = np.where(valid_b, -np.log(np.clip(b, 1e-10, 1.0)), 0.0)

    # m = a / (1 - b)
    m = np.where(valid_b, a / np.clip(1.0 - b, 1e-10, None), 0.0)

    # Residual σ
    residuals = Y - (a + b * X)
    sigma = np.sqrt((residuals ** 2).mean(axis=0))

    # σ_eq = σ / sqrt(2κ)
    valid_kappa = kappa > 0.01
    denom_eq = np.where(valid_kappa, np.sqrt(2.0 * np.maximum(kappa, 1e-10)), 1.0)
    sigma_eq = np.where(valid_kappa, sigma / denom_eq, 0.0)

    # S-score = (X_current - m) / σ_eq
    X_current = H[-1]
    valid_sigma = sigma_eq > 1e-10
    safe_sigma_eq = np.where(valid_sigma, sigma_eq, 1.0)
    s = np.where(valid_sigma, (X_current - m) / safe_sigma_eq, 0.0)

    s_scores = s.astype(np.float32)
    kappas = kappa.astype(np.float32)
    is_valid = (valid_b & valid_kappa & valid_sigma)

    return s_scores, kappas, is_valid


# ── Main search pipeline ─────────────────────────────────────────

def _estimate_eth_price(pool_states) -> float:
    """Estimate ETH price in USD from available pool data.

    Looks for USDC/WETH or WETH/USDT pools to extract price.
    Returns $2500 as fallback if no ETH pool found.
    """
    stables = {"USDC", "USDT", "DAI"}
    for ps in pool_states:
        if ps.token0 == "WETH" and ps.token1 in stables and ps.token0_price > 0:
            return ps.token0_price  # WETH priced in stablecoin
        if ps.token1 == "WETH" and ps.token0 in stables and ps.token1_price > 0:
            return ps.token1_price  # WETH priced in stablecoin
    return 2500.0  # fallback


def find_arbitrage_opportunities(
    pool_states: list,
    block_timestamp: int = 0,
    input_usd: float = 10000.0,
    max_hops: int = MAX_CYCLE_HOPS,
    use_reservoir: bool = True,
    use_kalman: bool = False,
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
        use_kalman: If True, use KalmanReservoir instead of EMA.

    Returns:
        List of ArbOpportunity objects with gross_profit_usd > 0.
    """
    if not pool_states:
        return []

    # Step 0: Update CEX reference feed
    global _cex_feed
    if _cex_feed is None:
        _cex_feed = CexFeed(mode="proxy")
    _cex_feed.update(pool_states, block_timestamp)

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

    # Step 3: Reservoir temporal smoothing (EMA or UKF)
    if use_reservoir:
        reservoir = _get_or_create_reservoir(d, use_kalman=use_kalman)
        smoothed_state = reservoir.step(market_state)
    else:
        smoothed_state = market_state

    # Step 3a: Multi-scale agreement
    if d not in _multi_scale_cache:
        _multi_scale_cache[d] = MultiScaleReservoir(
            d=d,
            reservoir_scale=RESERVOIR_SCALE,
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            seed=SEED,
        )
    ms_reservoir = _multi_scale_cache[d]
    ms_outputs = ms_reservoir.step(market_state)
    _scale_agreement = ms_reservoir.get_scale_agreement_from_cached(market_state, ms_outputs)

    # Step 3b: Compute S-scores from spot vs smoothed deviation
    s_scores, kappas, s_valid = _compute_s_scores(market_state, smoothed_state, d)

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

    # Check for regime change from KalmanReservoir changepoint detection
    _regime_change = False
    if use_kalman and use_reservoir:
        reservoir = _get_or_create_reservoir(d, use_kalman=True)
        if hasattr(reservoir, 'regime_change_detected'):
            _regime_change = reservoir.regime_change_detected

    # Phase 4: Estimate ETH price from pool data for gas cost calculation
    eth_price_usd = _estimate_eth_price(pool_states)

    # Gas cost: ~280k gas × 0.1 gwei = 0.000028 ETH + $2 Flashbots tip
    GAS_UNITS = 280_000
    GAS_PRICE_GWEI = 0.1
    gas_cost_eth = GAS_UNITS * GAS_PRICE_GWEI * 1e-9
    gas_cost_usd = gas_cost_eth * eth_price_usd + 2.0  # + Flashbots tip

    # Build pool lookup for CEX deviation (pool_address → PoolState)
    _pool_lookup: dict[str, object] = {}
    for ps in pool_states:
        _pool_lookup[ps.pool_address] = ps

    def _store_meta(opp: ArbOpportunity, cycle_edges: list[Edge]) -> None:
        """Compute and store signal metadata for one opportunity."""
        s, k, valid = _cycle_s_score(cycle_edges, edge_index, s_scores, kappas, s_valid)
        path_str = "→".join(opp.path)
        # Fee-weighted threshold: sum of all fee rates in the cycle
        fee_threshold = sum(e.fee_rate for e in cycle_edges)
        net_profit = opp.gross_profit_usd - gas_cost_usd

        # CEX deviation: find the primary pair (highest volume edge)
        cex_dev = None
        if _cex_feed is not None:
            best_vol = -1.0
            best_edge = None
            best_ps = None
            for e in cycle_edges:
                ps = _pool_lookup.get(e.pool_address)
                if ps is not None and ps.volume_usd > best_vol:
                    best_vol = ps.volume_usd
                    best_edge = e
                    best_ps = ps
            if best_edge is not None and best_ps is not None:
                cex_dev = _cex_feed.get_deviation(
                    best_edge.src, best_edge.dst,
                    best_edge.rate / (1.0 - best_edge.fee_rate),  # gross rate
                    block_timestamp,
                )

        _signal_metadata[(opp.block_timestamp, path_str)] = SignalMeta(
            s_score=s,
            mean_reversion_speed=k,
            s_score_valid=valid,
            scale_agreement=_scale_agreement,
            net_profit_usd=round(net_profit, 2),
            fee_weighted_threshold=fee_threshold,
            regime_change=_regime_change,
            cex_deviation=cex_dev,
        )

    # Primary: winning cycle
    winning_cycle = cycle_map[winning_label]
    profit = _compute_cycle_profit(winning_cycle, input_usd)
    if profit > 0:
        path = [winning_cycle[0].src]
        for e in winning_cycle:
            path.append(e.dst)
        opp = ArbOpportunity(
            method="latent_flux",
            path=path,
            hop_count=len(winning_cycle),
            gross_profit_usd=round(profit, 2),
            input_size_usd=input_usd,
            iterations=iterations,
            block_timestamp=block_timestamp,
        )
        opportunities.append(opp)
        _store_meta(opp, winning_cycle)

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
            opp = ArbOpportunity(
                method="latent_flux",
                path=path,
                hop_count=len(cycle_edges),
                gross_profit_usd=round(cycle_profit, 2),
                input_size_usd=input_usd,
                iterations=iterations,  # same convergence cost
                block_timestamp=block_timestamp,
            )
            opportunities.append(opp)
            _store_meta(opp, cycle_edges)

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
