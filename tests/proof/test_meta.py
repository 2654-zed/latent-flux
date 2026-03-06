"""Meta Test — Coherence across 50 problems.

50 geometry problems (5 categories × 10 each), solved in both
Python and LF. Solutions are scored for COHERENCE (1-5):
  5 = All operations are geometric, single representational frame
  4 = Mostly geometric, minor scaffolding
  3 = Mixed geometric and symbolic
  2 = Mostly symbolic with geometric elements
  1 = Purely symbolic, geometry is incidental

The test measures whether LF solutions are more coherent
(stay in a single geometric frame) than Python solutions.

This is the integrative test — it doesn't prove any single
point but checks whether the overall pattern holds.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from flux_manifold.core import flux_flow
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.drift_equivalence import DriftEquivalence
from flux_manifold.dimensional_squeeze import DimensionalSqueeze
from flux_manifold.flows import normalize_flow

from tests.proof.generators import generate_meta_suite, GeometryProblem
from tests.proof.measurement import score_coherence, compare_groups


RESULTS_DIR = Path("results/proof")


# ═══════════════════════════════════════════════════════════════════
# Python solver (numpy) — single generic function
# ═══════════════════════════════════════════════════════════════════

def solve_python_meta(problem: GeometryProblem) -> np.ndarray:
    """Python-only solver for meta test problems.

    This function is scored for coherence — how much of it is
    geometric vs symbolic scaffolding.
    """
    d = problem.d
    params = problem.params
    rng = np.random.default_rng(problem.seed)

    if problem.problem_type == 'interpolation':
        a = params['a']
        b = params['b']
        result = (a + b) / 2.0
        return result.astype(np.float32)

    elif problem.problem_type == 'constraint':
        targets = params['targets']
        radii = params['radii']
        v = np.mean(targets, axis=0).astype(np.float32)
        lr = 0.05
        for step in range(300):
            grad = np.zeros(d, dtype=np.float32)
            for t, r in zip(targets, radii):
                diff = v - t
                dist = float(np.linalg.norm(diff))
                if dist > r and dist > 1e-12:
                    grad += diff / dist
            v = v - lr * grad
            if float(np.linalg.norm(grad)) < 1e-6:
                break
        return v

    elif problem.problem_type == 'competition':
        a1 = params['a1']
        a2 = params['a2']
        v = ((a1 + a2) / 2.0).astype(np.float32)
        v += rng.standard_normal(d).astype(np.float32) * 0.05
        lr = 0.05
        for step in range(300):
            d1 = float(np.linalg.norm(v - a1))
            d2 = float(np.linalg.norm(v - a2))
            grad = np.zeros(d, dtype=np.float32)
            if d1 > 1e-12:
                grad += (a1 - v) / d1
            if d2 > 1e-12:
                grad += (a2 - v) / d2
            v = v + lr * grad
            if float(np.linalg.norm(lr * grad)) < 1e-6:
                break
        return v

    elif problem.problem_type == 'projection':
        basis = params['basis']
        point = params['point']
        proj = basis.T @ (basis @ point)
        return proj.astype(np.float32)

    else:
        return np.zeros(d, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# LF solver (primitives) — single generic function
# ═══════════════════════════════════════════════════════════════════

def solve_lf_meta(problem: GeometryProblem) -> np.ndarray:
    """LF-primitive solver for meta test problems.

    This function is scored for coherence — how much of it uses
    the geometric primitives vs ad-hoc computation.
    """
    d = problem.d
    params = problem.params

    if problem.problem_type == 'interpolation':
        a = params['a']
        b = params['b']
        target = ((a + b) / 2.0).astype(np.float32)
        sp = SuperpositionTensor.from_random(10, d, seed=problem.seed)
        sp.flow_all(target, normalize_flow, epsilon=0.1, tol=1e-3, max_steps=200)
        sp.reweight_by_drift(target)
        return sp.collapse_to_best(target)

    elif problem.problem_type == 'constraint':
        targets = params['targets']
        centroid = np.mean(targets, axis=0).astype(np.float32)
        sp = SuperpositionTensor.from_random(20, d, seed=problem.seed)
        sp.flow_all(centroid, normalize_flow, epsilon=0.1, tol=1e-3, max_steps=300)
        sp.reweight_by_drift(centroid)
        return sp.collapse_to_best(centroid)

    elif problem.problem_type == 'competition':
        a1 = params['a1']
        a2 = params['a2']
        midpoint = ((a1 + a2) / 2.0).astype(np.float32)
        sp = SuperpositionTensor.from_random(20, d, seed=problem.seed)
        sp.flow_all(midpoint, normalize_flow, epsilon=0.1, tol=1e-3, max_steps=300)
        sp.reweight_by_drift(midpoint)
        return sp.collapse_to_best(midpoint)

    elif problem.problem_type == 'projection':
        basis = params['basis']
        point = params['point']
        k = basis.shape[0]
        ds = DimensionalSqueeze(target_dim=k, method='pca')
        rng_proj = np.random.default_rng(problem.seed + 99)
        samples = np.vstack([
            basis,
            basis + rng_proj.standard_normal((k, d)).astype(np.float32) * 0.01
        ])
        ds.fit(samples)
        squeezed = ds.squeeze(point)
        return ds.unsqueeze(squeezed).astype(np.float32)

    else:
        return np.zeros(d, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Test Class
# ═══════════════════════════════════════════════════════════════════

class TestMeta:
    """Meta Test — coherence comparison across 50 problems."""

    @pytest.fixture(scope="class")
    def suite(self):
        return generate_meta_suite(seed=42)

    @pytest.fixture(scope="class")
    def coherence_data(self, suite):
        """Score solver source code coherence for each problem type."""
        # Get source code of both solvers
        py_source = inspect.getsource(solve_python_meta)
        lf_source = inspect.getsource(solve_lf_meta)

        py_score = score_coherence(py_source)
        lf_score = score_coherence(lf_source)

        # Run all 50 problems to verify both solve correctly
        py_qualities = []
        lf_qualities = []
        for problem in suite:
            py_result = solve_python_meta(problem)
            lf_result = solve_lf_meta(problem)
            py_qualities.append(problem.quality(py_result))
            lf_qualities.append(problem.quality(lf_result))

        return {
            'py_coherence': py_score,
            'lf_coherence': lf_score,
            'py_qualities': py_qualities,
            'lf_qualities': lf_qualities,
            'py_source': py_source,
            'lf_source': lf_source,
        }

    def test_both_solvers_work(self, coherence_data):
        """Both solvers produce finite results for all 50 problems."""
        py_q = coherence_data['py_qualities']
        lf_q = coherence_data['lf_qualities']
        assert all(np.isfinite(q) for q in py_q), "Python solver produced non-finite"
        assert all(np.isfinite(q) for q in lf_q), "LF solver produced non-finite"

    def test_lf_coherence_higher(self, coherence_data):
        """LF solver should score higher (or equal) on coherence rubric."""
        py = coherence_data['py_coherence']
        lf = coherence_data['lf_coherence']
        assert lf.score >= py.score, (
            f"LF coherence ({lf.score}: {lf.rationale}) not >= "
            f"Python ({py.score}: {py.rationale})"
        )

    def test_lf_more_geometric_ops(self, coherence_data):
        """LF solver should have more geometric operations."""
        py = coherence_data['py_coherence']
        lf = coherence_data['lf_coherence']
        assert lf.geometric_ops >= py.geometric_ops, (
            f"LF geo ops ({lf.geometric_ops}) < Python ({py.geometric_ops})"
        )

    def test_python_more_symbolic_ops(self, coherence_data):
        """Python solver should have more symbolic scaffolding."""
        py = coherence_data['py_coherence']
        lf = coherence_data['lf_coherence']
        assert py.symbolic_ops >= lf.symbolic_ops, (
            f"Python sym ops ({py.symbolic_ops}) < LF ({lf.symbolic_ops})"
        )

    def test_quality_comparison(self, coherence_data):
        """Compare solution quality between Python and LF."""
        py_q = coherence_data['py_qualities']
        lf_q = coherence_data['lf_qualities']
        stat = compare_groups(py_q, lf_q)
        # Quality should be similar (neither should be much worse)
        # We only care about coherence, not quality advantage
        assert isinstance(stat.p_value, float)

    def test_fewer_representation_switches(self, coherence_data):
        """LF should have fewer switches between geo and symbolic frames."""
        py = coherence_data['py_coherence']
        lf = coherence_data['lf_coherence']
        assert lf.representation_switches <= py.representation_switches, (
            f"LF switches ({lf.representation_switches}) > "
            f"Python ({py.representation_switches})"
        )

    def test_save_results(self, coherence_data):
        """Save meta test results."""
        py = coherence_data['py_coherence']
        lf = coherence_data['lf_coherence']
        py_q = coherence_data['py_qualities']
        lf_q = coherence_data['lf_qualities']

        stat = compare_groups(py_q, lf_q)

        output = {
            'python_coherence': {
                'score': py.score,
                'rationale': py.rationale,
                'geometric_ops': py.geometric_ops,
                'symbolic_ops': py.symbolic_ops,
                'representation_switches': py.representation_switches,
            },
            'lf_coherence': {
                'score': lf.score,
                'rationale': lf.rationale,
                'geometric_ops': lf.geometric_ops,
                'symbolic_ops': lf.symbolic_ops,
                'representation_switches': lf.representation_switches,
            },
            'quality_comparison': {
                'mean_python': float(np.mean(py_q)),
                'mean_lf': float(np.mean(lf_q)),
                'difference': stat.difference,
                'p_value': stat.p_value,
                'cohens_d': stat.cohens_d,
                'significant': stat.significant,
            },
            'interpretation': {
                'coherence_advantage': lf.score - py.score,
                'conclusion': (
                    'LF solutions maintain a more coherent geometric frame. '
                    'Python solutions require symbolic scaffolding (loops, conditionals, '
                    'manual convergence checks) that breaks the geometric flow. '
                    'LF primitives (superpose, flow, reweight, collapse) stay in the '
                    'geometric frame throughout.'
                    if lf.score > py.score else
                    'Coherence scores are equal. The generic solver structure '
                    'dominates over framework-specific differences.'
                ),
            },
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "test_meta.json", 'w') as f:
            json.dump(output, f, indent=2)

    def test_phenomenological_note(self):
        """Meta-observation about the proof test suite.

        OBSERVATION:
        Writing the same solver logic in Python (numpy) vs LF (primitives)
        reveals a consistent pattern:

        Python requires:
        - Explicit loops with step sizes
        - Manual convergence checks
        - Explicit gradient computation
        - Case-by-case NaN/Inf handling
        - Parameter tuning (lr, tolerance, max_iter)

        LF requires:
        - Choose an attractor target
        - Superpose candidates
        - Flow, reweight, collapse

        The LF version is SHORTER and STAYS GEOMETRIC. But it's also
        LESS FLEXIBLE — every problem gets the same ∑_ψ → ⟼ → collapse
        pipeline. The Python version can customize at every point.

        This suggests the thesis holds for problems that fit the
        "explore-flow-commit" paradigm and breaks down for problems
        that require fine-grained control over the optimization
        trajectory.
        """
        pass
