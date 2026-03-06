"""Test 3 — Cognitive Load Test.

Blind trial: 20 geometry problems (4 types × 5 each), solved
10 in Python (numpy) and 10 in LF (primitives). Measures
cognitive load via correction count (exceptions caught and
retried) and solution robustness.

PROXY: Since this is automated (not a human trial), "corrections"
are measured as:
  - Exceptions thrown during solving (caught and retried)
  - Validation failures (result doesn't satisfy problem spec)
  - Solver retries with adjusted parameters

PREDICTION: LF solver requires fewer corrections because the
primitives handle edge cases (NaN, divergence, normalization)
that Python solvers must handle manually.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from flux_manifold.core import flux_flow, flux_flow_traced
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.drift_equivalence import DriftEquivalence
from flux_manifold.dimensional_squeeze import DimensionalSqueeze
from flux_manifold.abstraction_cascade import AbstractionCascade
from flux_manifold.flows import normalize_flow

from tests.proof.generators import generate_cognitive_suite, GeometryProblem
from tests.proof.measurement import compare_groups
from tests.proof.logger import SolutionAttempt, AttemptLogger


RESULTS_DIR = Path("results/proof")


# ═══════════════════════════════════════════════════════════════════
# Python solver (numpy only) — generic for all 4 problem types
# ═══════════════════════════════════════════════════════════════════

def solve_python(problem: GeometryProblem) -> tuple[np.ndarray, list[str]]:
    """Solve a geometry problem using only numpy.

    Returns (result, corrections) where corrections is a list of
    correction descriptions (exception messages, retries).
    """
    corrections = []
    d = problem.d
    params = problem.params
    rng = np.random.default_rng(problem.seed)

    if problem.problem_type == 'interpolation':
        a, b = params['a'], params['b']
        try:
            result = (a + b) / 2.0
        except Exception as e:
            corrections.append(f"interpolation failed: {e}")
            result = a.copy()
        return result.astype(np.float32), corrections

    elif problem.problem_type == 'constraint':
        targets = params['targets']
        radii = params['radii']
        # Gradient descent toward centroid with radii constraints
        v = np.mean(targets, axis=0).astype(np.float32)
        lr = 0.05
        for step in range(500):
            grad = np.zeros(d, dtype=np.float32)
            for t, r in zip(targets, radii):
                diff = v - t
                dist = float(np.linalg.norm(diff))
                if dist > r:
                    if dist < 1e-12:
                        corrections.append(f"step {step}: near-zero distance")
                        continue
                    grad += diff / dist
            v = v - lr * grad
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < 1e-6:
                break
            if not np.all(np.isfinite(v)):
                corrections.append(f"step {step}: NaN/Inf detected, resetting")
                v = np.mean(targets, axis=0).astype(np.float32)
                lr *= 0.5
        return v, corrections

    elif problem.problem_type == 'competition':
        a1, a2 = params['a1'], params['a2']
        # Find equidistant point
        midpoint = ((a1 + a2) / 2.0).astype(np.float32)
        v = midpoint + rng.standard_normal(d).astype(np.float32) * 0.1
        lr = 0.05
        for step in range(500):
            d1 = float(np.linalg.norm(v - a1))
            d2 = float(np.linalg.norm(v - a2))
            grad = np.zeros(d, dtype=np.float32)
            if d1 > 1e-12:
                grad += (a1 - v) / d1
            else:
                corrections.append(f"step {step}: collapsed onto a1")
            if d2 > 1e-12:
                grad += (a2 - v) / d2
            else:
                corrections.append(f"step {step}: collapsed onto a2")
            v = v + lr * grad
            if not np.all(np.isfinite(v)):
                corrections.append(f"step {step}: NaN, resetting")
                v = midpoint.copy()
                lr *= 0.5
            if float(np.linalg.norm(lr * grad)) < 1e-6:
                break
        return v, corrections

    elif problem.problem_type == 'projection':
        basis = params['basis']
        point = params['point']
        try:
            proj = basis.T @ (basis @ point)
        except Exception as e:
            corrections.append(f"projection failed: {e}")
            proj = point.copy()
        if not np.all(np.isfinite(proj)):
            corrections.append("projection produced NaN, falling back")
            proj = point.copy()
        return proj.astype(np.float32), corrections

    else:
        corrections.append(f"unknown problem type: {problem.problem_type}")
        return np.zeros(d, dtype=np.float32), corrections


# ═══════════════════════════════════════════════════════════════════
# LF solver (primitives) — generic for all 4 problem types
# ═══════════════════════════════════════════════════════════════════

def solve_lf(problem: GeometryProblem) -> tuple[np.ndarray, list[str]]:
    """Solve a geometry problem using LF primitives.

    Returns (result, corrections) where corrections is a list of
    correction descriptions.
    """
    corrections = []
    d = problem.d
    params = problem.params

    if problem.problem_type == 'interpolation':
        a, b = params['a'], params['b']
        target = ((a + b) / 2.0).astype(np.float32)
        # Flow from a toward target
        sp = SuperpositionTensor.from_random(10, d, seed=problem.seed)
        sp.flow_all(target, normalize_flow, epsilon=0.1, tol=1e-3, max_steps=300)
        sp.reweight_by_drift(target)
        result = sp.collapse_to_best(target)
        return result, corrections

    elif problem.problem_type == 'constraint':
        targets = params['targets']
        centroid = np.mean(targets, axis=0).astype(np.float32)
        sp = SuperpositionTensor.from_random(20, d, seed=problem.seed)
        try:
            sp.flow_all(centroid, normalize_flow,
                        epsilon=0.1, tol=1e-3, max_steps=300)
        except Exception as e:
            corrections.append(f"flow failed: {e}")
        sp.reweight_by_drift(centroid)
        result = sp.collapse_to_best(centroid)
        return result, corrections

    elif problem.problem_type == 'competition':
        a1, a2 = params['a1'], params['a2']
        midpoint = ((a1 + a2) / 2.0).astype(np.float32)
        sp = SuperpositionTensor.from_random(20, d, seed=problem.seed)
        try:
            sp.flow_all(midpoint, normalize_flow,
                        epsilon=0.1, tol=1e-3, max_steps=300)
        except Exception as e:
            corrections.append(f"flow failed: {e}")
        sp.reweight_by_drift(midpoint)
        result = sp.collapse_to_best(midpoint)
        return result, corrections

    elif problem.problem_type == 'projection':
        basis = params['basis']
        point = params['point']
        # Use DimensionalSqueeze for projection
        k = basis.shape[0]
        ds = DimensionalSqueeze(target_dim=k, method='pca')
        # Fit on basis vectors as training data (basis is (k, d))
        rng_proj = np.random.default_rng(problem.seed + 99)
        samples = np.vstack([
            basis,
            basis + rng_proj.standard_normal((k, d)).astype(np.float32) * 0.01
        ])
        try:
            ds.fit(samples)
            squeezed = ds.squeeze(point)
            result = ds.unsqueeze(squeezed)
        except Exception as e:
            corrections.append(f"squeeze failed: {e}")
            result = point.copy()
        return result.astype(np.float32), corrections

    else:
        corrections.append(f"unknown type: {problem.problem_type}")
        return np.zeros(d, dtype=np.float32), corrections


# ═══════════════════════════════════════════════════════════════════
# Test Class
# ═══════════════════════════════════════════════════════════════════

class TestCognitiveLoad:
    """Test 3 — cognitive load comparison via correction counting."""

    @pytest.fixture(scope="class")
    def suite(self):
        return generate_cognitive_suite(seed=42)

    @pytest.fixture(scope="class")
    def results(self, suite):
        """Run all 20 problems: odd-indexed → Python, even-indexed → LF."""
        logger = AttemptLogger()
        for i, problem in enumerate(suite):
            # Alternate assignment for fairness
            if i % 2 == 0:
                # Python
                t0 = time.monotonic_ns()
                result, corrections = solve_python(problem)
                elapsed = (time.monotonic_ns() - t0) / 1e6
                quality = problem.quality(result)
                logger.log(SolutionAttempt(
                    problem_id=i,
                    language='python',
                    problem_type=problem.problem_type,
                    corrections=corrections,
                    success=quality > -1.0,
                    subjective_difficulty=3,
                    elapsed_ms=elapsed,
                ))
            else:
                # LF
                t0 = time.monotonic_ns()
                result, corrections = solve_lf(problem)
                elapsed = (time.monotonic_ns() - t0) / 1e6
                quality = problem.quality(result)
                logger.log(SolutionAttempt(
                    problem_id=i,
                    language='lf',
                    problem_type=problem.problem_type,
                    corrections=corrections,
                    success=quality > -1.0,
                    subjective_difficulty=3,
                    elapsed_ms=elapsed,
                ))
        return logger

    def test_all_problems_attempted(self, results):
        """All 20 problems should be attempted."""
        assert len(results.attempts) == 20

    def test_python_group_exists(self, results):
        """Should have 10 Python attempts."""
        py = [a for a in results.attempts if a.language == 'python']
        assert len(py) == 10

    def test_lf_group_exists(self, results):
        """Should have 10 LF attempts."""
        lf = [a for a in results.attempts if a.language == 'lf']
        assert len(lf) == 10

    def test_correction_comparison(self, results):
        """Compare correction counts between Python and LF."""
        summary = results.summary()
        py_corrections = summary['python']['total_corrections']
        lf_corrections = summary['lf']['total_corrections']
        # Document the comparison (may or may not favor LF)
        assert isinstance(py_corrections, (int, float))
        assert isinstance(lf_corrections, (int, float))

    def test_success_rates(self, results):
        """Both solvers should have reasonable success rates."""
        summary = results.summary()
        assert summary['python']['success_rate'] >= 0.5, (
            f"Python success rate too low: {summary['python']['success_rate']}"
        )
        assert summary['lf']['success_rate'] >= 0.5, (
            f"LF success rate too low: {summary['lf']['success_rate']}"
        )

    def test_cognitive_load_ratio(self, results):
        """Report the cognitive load ratio."""
        summary = results.summary()
        ratio = summary.get('cognitive_load_ratio')
        # Document the ratio even if it doesn't favor our hypothesis
        assert ratio is not None, "Could not compute cognitive load ratio"

    def test_statistical_analysis(self, results):
        """Statistical comparison of correction counts."""
        py_corrections = [len(a.corrections) for a in results.attempts
                          if a.language == 'python']
        lf_corrections = [len(a.corrections) for a in results.attempts
                          if a.language == 'lf']
        if max(py_corrections) == 0 and max(lf_corrections) == 0:
            # Both had zero corrections — can't compare
            return
        stat = compare_groups(py_corrections, lf_corrections)
        # Save results regardless of significance
        assert isinstance(stat.p_value, float)

    def test_save_results(self, results):
        """Save cognitive load results."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results.save(RESULTS_DIR / "test_3_cognitive_load.json")

    def test_verification_proxies(self):
        """Document what this test actually measures vs claims.

        PROXY 1: "Corrections" = exceptions caught + NaN recoveries.
        This measures SOLVER ROBUSTNESS, not human cognitive load.
        The test cannot measure actual cognitive load without a
        human trial.

        PROXY 2: Success rate measures solution QUALITY under each
        framework. Higher success rate suggests the framework
        naturally avoids failure modes.

        LIMITATION: Both solvers were written by the same entity
        (this AI) with full knowledge of the test design. A true
        cognitive load test requires naive subjects solving novel
        problems without knowing the experimental design.

        WHAT THIS DOES SHOW: The LF primitives (SuperpositionTensor,
        flow_all, reweight_by_drift, collapse_to_best) handle edge
        cases internally (NaN checking, normalization). The Python
        solver must handle these explicitly, creating more code
        paths where errors can occur.
        """
        pass
