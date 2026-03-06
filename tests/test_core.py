"""Unit tests for FluxManifold core engine."""

import numpy as np
import pytest

from flux_manifold.core import flux_flow, flux_flow_traced
from flux_manifold.flows import normalize_flow, sin_flow, damped_flow, adaptive_flow, repulsive_flow


# ── Basic convergence ──────────────────────────────────────────────

class TestFluxFlow:
    def test_2d_converges(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        result = flux_flow(s0, q, normalize_flow, epsilon=0.1, tol=1e-3)
        assert np.linalg.norm(result - q) < 1e-3

    def test_10d_converges(self):
        rng = np.random.default_rng(42)
        s0 = rng.standard_normal(10).astype(np.float32)
        q = np.zeros(10, dtype=np.float32)
        result = flux_flow(s0, q, normalize_flow, epsilon=0.1, tol=1e-3)
        assert np.linalg.norm(result - q) < 1e-3

    def test_already_at_attractor(self):
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        s0 = q.copy()
        result = flux_flow(s0, q, normalize_flow, epsilon=0.1, tol=1e-3)
        np.testing.assert_allclose(result, q, atol=1e-3)

    def test_does_not_mutate_s0(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        s0_orig = s0.copy()
        q = np.array([1.0, 1.0], dtype=np.float32)
        flux_flow(s0, q, normalize_flow)
        np.testing.assert_array_equal(s0, s0_orig)

    def test_max_steps_reached(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([100.0, 100.0], dtype=np.float32)
        result = flux_flow(s0, q, normalize_flow, epsilon=0.01, tol=1e-10, max_steps=5)
        # Should return something, not crash
        assert result.shape == (2,)


# ── Traced version ─────────────────────────────────────────────────

class TestFluxFlowTraced:
    def test_returns_trace_dict(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        r = flux_flow_traced(s0, q, normalize_flow)
        assert "converged_state" in r
        assert "steps" in r
        assert "converged" in r
        assert "drift_trace" in r
        assert "path" in r

    def test_drift_monotonically_decreases_normalize(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        r = flux_flow_traced(s0, q, normalize_flow, epsilon=0.05)
        drifts = r["drift_trace"]
        for i in range(1, len(drifts)):
            assert drifts[i] <= drifts[i - 1] + 1e-6

    def test_path_length_matches_steps(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        r = flux_flow_traced(s0, q, normalize_flow)
        # path includes initial state + each step
        assert len(r["path"]) == r["steps"] + 1

    def test_converged_flag(self):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        r = flux_flow_traced(s0, q, normalize_flow, tol=1e-3, max_steps=1000)
        assert r["converged"] is True


# ── Input validation ───────────────────────────────────────────────

class TestValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            flux_flow(np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32), normalize_flow)

    def test_2d_array_accepted_for_batch(self):
        """2D arrays are now accepted for batch flow."""
        from flux_manifold.core import flux_flow_batch
        S = np.zeros((2, 2), dtype=np.float32)
        q = np.ones(2, dtype=np.float32)
        result = flux_flow_batch(S, q, normalize_flow)
        assert result.shape == (2, 2)

    def test_bad_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            flux_flow(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), normalize_flow, epsilon=0)

    def test_bad_tol_raises(self):
        with pytest.raises(ValueError, match="tol"):
            flux_flow(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), normalize_flow, tol=-1)

    def test_dimension_cap_raises(self):
        with pytest.raises(ValueError, match="1024"):
            flux_flow(np.zeros(2000, dtype=np.float32), np.zeros(2000, dtype=np.float32), normalize_flow)


# ── Flow functions ─────────────────────────────────────────────────

class TestFlows:
    @pytest.mark.parametrize("flow_fn", [normalize_flow, sin_flow, damped_flow, adaptive_flow])
    def test_convergence_with_flow(self, flow_fn):
        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        result = flux_flow(s0, q, flow_fn, epsilon=0.1, tol=1e-2, max_steps=2000)
        assert np.linalg.norm(result - q) < 0.1  # Loose bound for sin_flow

    def test_normalize_at_target_returns_zero(self):
        q = np.array([1.0, 1.0], dtype=np.float32)
        delta = normalize_flow(q, q)
        np.testing.assert_array_equal(delta, np.zeros(2, dtype=np.float32))

    def test_repulsive_increases_distance(self):
        s0 = np.array([0.5, 0.5], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        delta = repulsive_flow(s0, q)
        new_s = s0 + 0.1 * delta
        assert np.linalg.norm(new_s - q) > np.linalg.norm(s0 - q)


# ── NaN / overflow safety ─────────────────────────────────────────

class TestSafety:
    def test_nan_flow_handled(self):
        def bad_flow(s, q):
            return np.full_like(s, np.nan)

        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        result = flux_flow(s0, q, bad_flow, max_steps=10)
        assert not np.any(np.isnan(result))

    def test_inf_flow_handled(self):
        def inf_flow(s, q):
            return np.full_like(s, np.inf)

        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        result = flux_flow(s0, q, inf_flow, max_steps=10)
        assert not np.any(np.isinf(result))

    def test_large_delta_clipped(self):
        def huge_flow(s, q):
            return (q - s) * 1000

        s0 = np.array([0.0, 0.0], dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        result = flux_flow(s0, q, huge_flow, epsilon=0.5, max_steps=100)
        # Should still produce finite values
        assert np.all(np.isfinite(result))


# ── Batch flow (N, d) ─────────────────────────────────────────────

class TestBatchFlow:
    def test_batch_converges(self):
        from flux_manifold.core import flux_flow_batch
        S = np.array([[1, 1], [2, 2], [-1, -1]], dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        result = flux_flow_batch(S, q, normalize_flow, epsilon=0.1, tol=1e-3)
        assert result.shape == (3, 2)
        for i in range(3):
            assert np.linalg.norm(result[i] - q) < 1e-2

    def test_batch_traced_returns_dict(self):
        from flux_manifold.core import flux_flow_traced_batch
        S = np.array([[1, 0], [0, 1]], dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        trace = flux_flow_traced_batch(S, q, normalize_flow)
        assert "converged_states" in trace
        assert "steps" in trace
        assert "converged" in trace
        assert "total_steps" in trace
        assert "drift_traces" in trace

    def test_batch_converged_flags(self):
        from flux_manifold.core import flux_flow_traced_batch
        S = np.array([[0.001, 0], [100, 100]], dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        trace = flux_flow_traced_batch(S, q, normalize_flow, tol=1e-2, max_steps=5)
        # First state is already near q, second needs many steps
        assert trace["converged"][0] == True
        assert trace["steps"][0] < trace["steps"][1]

    def test_batch_matches_serial(self):
        """Batch results should match serial flow for each state."""
        from flux_manifold.core import flux_flow_batch
        rng = np.random.default_rng(99)
        S = rng.standard_normal((5, 4)).astype(np.float32) * 2
        q = np.ones(4, dtype=np.float32)
        batch_result = flux_flow_batch(S.copy(), q, normalize_flow, epsilon=0.1, tol=1e-3)
        for i in range(5):
            serial_result = flux_flow(S[i], q, normalize_flow, epsilon=0.1, tol=1e-3)
            np.testing.assert_allclose(batch_result[i], serial_result, atol=1e-5)

    def test_batch_nan_safety(self):
        from flux_manifold.core import flux_flow_batch
        def bad_flow(s, q):
            return np.full_like(s, np.nan)
        S = np.ones((3, 2), dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        result = flux_flow_batch(S, q, bad_flow, max_steps=5)
        assert not np.any(np.isnan(result))

    def test_batch_speed_advantage(self):
        """Batch flow should be faster than serial for many states."""
        import time
        from flux_manifold.core import flux_flow_batch
        rng = np.random.default_rng(123)
        N, d = 50, 16
        S = rng.standard_normal((N, d)).astype(np.float32)
        q = np.zeros(d, dtype=np.float32)

        # Batch
        t0 = time.perf_counter()
        flux_flow_batch(S.copy(), q, normalize_flow, epsilon=0.1, tol=1e-3)
        t_batch = time.perf_counter() - t0

        # Serial
        t0 = time.perf_counter()
        for i in range(N):
            flux_flow(S[i].copy(), q, normalize_flow, epsilon=0.1, tol=1e-3)
        t_serial = time.perf_counter() - t0

        # Batch should not be dramatically slower (allow 3x margin for small N)
        assert t_batch < t_serial * 3
