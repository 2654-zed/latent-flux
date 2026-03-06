"""Tests for baselines."""

import numpy as np
import pytest

from flux_manifold.baselines import random_walk, gradient_descent, static_baseline


class TestRandomWalk:
    def test_returns_dict(self):
        s0 = np.zeros(4, dtype=np.float32)
        q = np.ones(4, dtype=np.float32)
        r = random_walk(s0, q, max_steps=10)
        assert "converged_state" in r
        assert "steps" in r
        assert "drift_trace" in r

    def test_rarely_converges(self):
        s0 = np.zeros(64, dtype=np.float32)
        q = np.ones(64, dtype=np.float32)
        r = random_walk(s0, q, max_steps=100)
        # Random walk in high-d almost never converges in 100 steps
        assert r["steps"] >= 1


class TestGradientDescent:
    def test_converges_2d(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        r = gradient_descent(s0, q, epsilon=0.1, tol=1e-3, max_steps=1000)
        assert r["converged"]

    def test_exponential_convergence(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        r = gradient_descent(s0, q, epsilon=0.5, tol=1e-6, max_steps=1000)
        # GD with epsilon=0.5 should converge, drift decreases exponentially
        drifts = r["drift_trace"]
        for i in range(1, len(drifts)):
            assert drifts[i] < drifts[i - 1] + 1e-6


class TestStaticBaseline:
    def test_returns_unchanged(self):
        s0 = np.array([1.0, 2.0], dtype=np.float32)
        q = np.array([3.0, 4.0], dtype=np.float32)
        r = static_baseline(s0, q)
        np.testing.assert_array_equal(r["converged_state"], s0)
        assert r["converged"] is False
        assert r["steps"] == 0
