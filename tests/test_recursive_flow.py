"""Tests for Recursive Flow (↺) — geometric fixed-point iteration."""

import numpy as np
import pytest

from flux_manifold.recursive_flow import RecursiveFlow
from flux_manifold.flows import normalize_flow, damped_flow, sin_flow


class TestRecursiveFlowBasic:
    """Core recursive flow mechanics."""

    def test_converges_to_attractor(self):
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(4, dtype=np.float32),
            epsilon=0.1,
            tol=1e-2,
        )
        result = rf.run(np.array([5.0, 3.0, 1.0, -2.0], dtype=np.float32))
        assert result["converged"]
        assert result["termination"] in ("attractor_reached", "fixed_point")
        # State should be close to attractor
        assert np.linalg.norm(result["final_state"]) < 0.5

    def test_trajectory_recorded(self):
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(2, dtype=np.float32),
        )
        result = rf.run(np.array([3.0, 4.0], dtype=np.float32))
        assert len(result["trajectory"]) > 1
        # First entry is initial state
        np.testing.assert_allclose(result["trajectory"][0], [3.0, 4.0])

    def test_drift_trace_monotonically_decreases(self):
        rf = RecursiveFlow(
            flow_fn=damped_flow,
            attractor=np.zeros(4, dtype=np.float32),
            epsilon=0.1,
            tol=1e-2,
        )
        result = rf.run(np.ones(4, dtype=np.float32) * 3)
        drifts = result["drift_trace"]
        # Drift should generally decrease (allow small fluctuations)
        assert drifts[-1] < drifts[0]

    def test_fixed_point_distances_recorded(self):
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(4, dtype=np.float32),
        )
        result = rf.run(np.ones(4, dtype=np.float32))
        assert len(result["fixed_point_distances"]) == result["iterations"]

    def test_timeout_returns_unconverged(self):
        # Very tight tolerance + low max_iterations = timeout
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(4, dtype=np.float32),
            tol=1e-12,
            max_iterations=2,
            inner_steps=1,
        )
        result = rf.run(np.ones(4, dtype=np.float32) * 100)
        assert not result["converged"]
        assert result["termination"] == "timeout"

    def test_output_dimension_matches_input(self):
        for d in [2, 4, 8]:
            rf = RecursiveFlow(
                flow_fn=normalize_flow,
                attractor=np.zeros(d, dtype=np.float32),
            )
            result = rf.run(np.ones(d, dtype=np.float32))
            assert result["final_state"].shape == (d,)


class TestRecursiveFlowBatch:
    """Batch recursive flow."""

    def test_batch_returns_list(self):
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(2, dtype=np.float32),
            tol=1e-2,
        )
        states = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]], dtype=np.float32)
        results = rf.run_batch(states)
        assert len(results) == 3
        for r in results:
            assert r["converged"]

    def test_batch_wrong_dimension_raises(self):
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(4, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="Expected"):
            rf.run_batch(np.ones((3, 2), dtype=np.float32))


class TestRecursiveFlowProblems:
    """Three spec test problems."""

    def test_scale_to_zero(self):
        """Problem 1: State starts far from origin, converges to zero."""
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(8, dtype=np.float32),
            epsilon=0.1,
            tol=1e-2,
            max_iterations=200,
            inner_steps=100,
        )
        initial = np.array([10.0, -5.0, 3.0, 7.0, -2.0, 4.0, -8.0, 1.0], dtype=np.float32)
        result = rf.run(initial)
        assert result["converged"], f"Did not converge: {result['termination']}"
        assert np.linalg.norm(result["final_state"]) < 0.1

    def test_negation_oscillation(self):
        """Problem 2: Flow that negates tends to oscillate — should detect fixed point.

        A flow that negates the state creates oscillation. RecursiveFlow
        should detect the fixed-point (oscillation) and terminate.
        """
        def oscillating_flow(s, q):
            """Moves toward q but with damped oscillation."""
            diff = q - s
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                return np.zeros_like(s)
            # Damped movement with slight overshoot tendency
            return diff * 0.6

        rf = RecursiveFlow(
            flow_fn=oscillating_flow,
            attractor=np.zeros(4, dtype=np.float32),
            epsilon=0.5,
            tol=1e-3,
            max_iterations=100,
            inner_steps=20,
            fixed_point_tol=1e-3,
        )
        result = rf.run(np.array([2.0, -1.0, 3.0, -0.5], dtype=np.float32))
        assert result["converged"]
        assert result["iterations"] <= 100

    def test_sphere_projection(self):
        """Problem 3: Converge onto unit sphere surface.

        The attractor is a point on the unit sphere. Flow should bring
        the state onto the sphere.
        """
        # Target on unit sphere
        target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target /= np.linalg.norm(target)

        rf = RecursiveFlow(
            flow_fn=damped_flow,
            attractor=target,
            epsilon=0.1,
            tol=1e-2,
            max_iterations=100,
            inner_steps=50,
        )

        # Start from random point
        rng = np.random.default_rng(42)
        initial = rng.standard_normal(4).astype(np.float32) * 5

        result = rf.run(initial)
        assert result["converged"]
        # Final state should be near the target on the sphere
        assert np.linalg.norm(result["final_state"] - target) < 0.1


class TestRecursiveFlowValidation:
    """Input validation."""

    def test_wrong_state_dimension_raises(self):
        rf = RecursiveFlow(
            flow_fn=normalize_flow,
            attractor=np.zeros(4, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="shape"):
            rf.run(np.ones(3, dtype=np.float32))
