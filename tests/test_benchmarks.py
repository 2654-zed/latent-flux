"""Tests for benchmarks and kill tests."""

import numpy as np
import pytest

from flux_manifold.benchmarks import tier_a, tier_b, tier_c_placeholder
from flux_manifold.kill_tests import (
    kill_test_convergence,
    kill_test_drift,
    kill_test_vs_random,
    kill_test_scalability,
    kill_test_adversarial,
    run_all_kill_tests,
)


class TestTierA:
    def test_tier_a_mean_steps_under_10(self):
        r = tier_a(n_runs=50)
        assert r["mean_steps"] < 10, f"mean_steps={r['mean_steps']}"

    def test_tier_a_variance_under_2(self):
        r = tier_a(n_runs=50)
        assert r["var_steps"] < 2, f"var_steps={r['var_steps']}"

    def test_tier_a_all_converge(self):
        r = tier_a(n_runs=50)
        assert r["converged_pct"] == 100.0


class TestTierB:
    def test_tier_b_flux_beats_random(self):
        r = tier_b(d=64, n_runs=20)
        assert r["flux_mean_steps"] < r["rw_mean_steps"]

    def test_tier_b_p_value(self):
        r = tier_b(d=64, n_runs=30)
        assert r["pass_p_rw"], f"p_value_vs_rw={r['p_value_vs_rw']}"


class TestTierC:
    def test_tier_c_runs(self):
        r = tier_c_placeholder(n_puzzles=3, grid_size=4, d=16)
        assert r["tier"] == "C"
        assert len(r["results"]) == 3


class TestKillTests:
    def test_convergence(self):
        r = kill_test_convergence(n_runs=20, d=64)
        assert r["pass"], f"not_converged_pct={r['not_converged_pct']}"

    def test_drift(self):
        r = kill_test_drift(n_runs=20, d=64)
        assert r["pass"], f"bad_drift_pct={r['bad_drift_pct']}"

    def test_vs_random(self):
        r = kill_test_vs_random(n_runs=20, d=64)
        assert r["pass"], f"flux_win_pct={r['flux_win_pct']}"

    def test_scalability(self):
        r = kill_test_scalability(d=256, n_runs=5, time_limit_ms=50.0)
        assert r["pass"], f"mean_ms={r['mean_ms']}"

    def test_adversarial(self):
        r = kill_test_adversarial(d=64)
        assert r["pass"], "Repulsive flow should diverge"

    def test_run_all(self):
        results = run_all_kill_tests()
        assert len(results) == 5
        for r in results:
            assert "test" in r
            assert "pass" in r
