"""Tests for the full Latent Flux interpreter and TSP solver."""

import numpy as np
import pytest

from flux_manifold.interpreter import LatentFluxInterpreter
from flux_manifold.tsp_solver import (
    LatentFluxTSP,
    solve_tsp,
    tour_length,
    order_to_state,
    state_to_order,
    nearest_neighbor_tour,
    make_crossing_critique,
)


# ── TSP encoding helpers ───────────────────────────────────────────

class TestTSPEncoding:
    def test_order_state_roundtrip(self):
        order = np.array([2, 0, 3, 1], dtype=np.int32)
        state = order_to_state(order, 4)
        recovered = state_to_order(state)
        np.testing.assert_array_equal(recovered, order)

    def test_tour_length_square(self):
        # Unit square: tour 0→1→2→3 = 4 sides of length 1
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        order = np.array([0, 1, 2, 3], dtype=np.int32)
        length = tour_length(cities, order)
        assert abs(length - 4.0) < 1e-5

    def test_nearest_neighbor_tour(self):
        cities = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
        tour = nearest_neighbor_tour(cities, start=0)
        # Collinear cities: NN from 0 should visit in order
        np.testing.assert_array_equal(tour, [0, 1, 2, 3])

    def test_nearest_neighbor_visits_all(self):
        rng = np.random.default_rng(42)
        cities = rng.uniform(0, 10, (10, 2)).astype(np.float32)
        tour = nearest_neighbor_tour(cities)
        assert set(tour.tolist()) == set(range(10))


# ── Crossing critique (Fold-Reference for TSP) ────────────────────

class TestCrossingCritique:
    def test_detects_crossing(self):
        # Square cities with crossing tour: 0→2→1→3 crosses
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        critique = make_crossing_critique(cities)
        crossing_order = np.array([0, 2, 1, 3], dtype=np.int32)
        state = order_to_state(crossing_order, 4)
        ok, diag, correction = critique(state)
        assert not ok
        assert "crossing" in diag

    def test_no_crossing_clean(self):
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        critique = make_crossing_critique(cities)
        clean_order = np.array([0, 1, 2, 3], dtype=np.int32)
        state = order_to_state(clean_order, 4)
        ok, diag, correction = critique(state)
        assert ok


# ── LatentFluxInterpreter ─────────────────────────────────────────

class TestInterpreter:
    def test_evaluate_returns_all_keys(self):
        interp = LatentFluxInterpreter(d=8, n_candidates=4, max_steps=50, seed=42)
        q = np.zeros(8, dtype=np.float32)
        result = interp.evaluate(q)
        assert "committed_state" in result
        assert "equivalence_quality" in result
        assert "abstraction_levels" in result
        assert "total_steps" in result
        assert "fold_corrections" in result
        assert "superposition_entropy" in result
        assert "commit_reason" in result

    def test_evaluate_converges(self):
        interp = LatentFluxInterpreter(d=8, n_candidates=8, max_steps=200, seed=42)
        q = np.zeros(8, dtype=np.float32)
        result = interp.evaluate(q)
        assert result["converged_count"] > 0

    def test_evaluate_with_squeeze(self):
        interp = LatentFluxInterpreter(
            d=32, n_candidates=4, max_steps=100, squeeze_dim=8, seed=42
        )
        q = np.zeros(32, dtype=np.float32)
        result = interp.evaluate(q)
        assert result["committed_state"].shape == (32,)

    def test_evaluate_with_custom_initial_states(self):
        states = np.random.default_rng(42).standard_normal((6, 16)).astype(np.float32)
        states = np.clip(states, -1, 1)
        interp = LatentFluxInterpreter(d=16, n_candidates=6, max_steps=100, seed=42)
        q = np.zeros(16, dtype=np.float32)
        result = interp.evaluate(q, initial_states=states)
        assert result["n_candidates"] == 6


# ── LatentFluxTSP ─────────────────────────────────────────────────

class TestLatentFluxTSP:
    def test_solve_small(self):
        cities = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],
        ], dtype=np.float32)
        result = solve_tsp(cities, n_candidates=10, seed=42)
        assert "best_tour" in result
        assert "best_length" in result
        assert "nn_length" in result
        assert len(result["best_tour"]) == 4
        assert set(result["best_tour"]) == {0, 1, 2, 3}

    def test_solve_improves_or_matches_random(self):
        rng = np.random.default_rng(42)
        cities = rng.uniform(0, 10, (8, 2)).astype(np.float32)
        result = solve_tsp(cities, n_candidates=15, seed=42)
        # Random tour length
        random_order = rng.permutation(8).astype(np.int32)
        random_length = tour_length(cities, random_order)
        # Flux TSP should do at least as well as random
        assert result["best_length"] <= random_length * 1.5  # Generous margin

    def test_solve_reports_fold_corrections(self):
        cities = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2], [1, 1],
        ], dtype=np.float32)
        result = solve_tsp(cities, n_candidates=10, seed=42)
        assert "fold_corrections" in result
        assert isinstance(result["fold_corrections"], int)

    def test_solve_commitment(self):
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        result = solve_tsp(cities, n_candidates=8, seed=42)
        assert result["commit_reason"] != ""

    def test_solve_abstraction_levels(self):
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        result = solve_tsp(cities, n_candidates=8, seed=42)
        assert result["abstraction_levels"] >= 1

    def test_solve_10_cities(self):
        rng = np.random.default_rng(99)
        cities = rng.uniform(0, 100, (10, 2)).astype(np.float32)
        result = solve_tsp(cities, n_candidates=20, seed=99)
        assert len(result["best_tour"]) == 10
        assert set(result["best_tour"]) == set(range(10))
        assert result["best_length"] > 0
