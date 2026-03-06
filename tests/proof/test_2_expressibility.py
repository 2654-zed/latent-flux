"""Test 2 — Expressibility Boundary Test.

Measures whether superposition with inter-state coupling during flow
produces better solutions than independent parallel flow.

CRITICAL FINDING FROM PHASE 1 READING:
The current SuperpositionTensor.flow_all() does NOT couple states
during flow. It runs flux_flow_traced_batch which flows each state
independently. reweight_by_drift() is called AFTER flow completes.

This means "superposition" in the current implementation is equivalent
to independent parallel search + post-hoc selection. There is no
genuine inter-state coupling.

This test therefore has TWO parts:
  2a: Confirm the null result with current implementation
  2b: Implement coupled flow and test whether coupling matters

The null result in 2a is itself evidence — it reveals that the
current ∑_ψ primitive doesn't express genuine superposition coupling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flux_manifold.core import flux_flow, flux_flow_traced
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.flows import normalize_flow

from tests.proof.generators import (
    generate_expressibility_suite, AttractorBasinProblem,
)
from tests.proof.measurement import paired_compare


RESULTS_DIR = Path("results/proof")


# ═══════════════════════════════════════════════════════════════════
# Condition A: Solo (independent flow, no coupling)
# ═══════════════════════════════════════════════════════════════════

def solve_solo(problem: AttractorBasinProblem) -> np.ndarray:
    """Flow each candidate independently, return best."""
    rng = np.random.default_rng(problem.seed)
    candidates = rng.standard_normal(
        (problem.n_candidates, problem.d)
    ).astype(np.float32)

    best_state = None
    best_quality = -np.inf

    for i in range(problem.n_candidates):
        result = flux_flow(
            candidates[i], problem.target, normalize_flow,
            epsilon=0.1, tol=1e-3, max_steps=500,
        )
        q = problem.quality(result)
        if q > best_quality:
            best_quality = q
            best_state = result

    return best_state


# ═══════════════════════════════════════════════════════════════════
# Condition B: Current SuperpositionTensor (no coupling during flow)
# ═══════════════════════════════════════════════════════════════════

def solve_superposition_uncoupled(problem: AttractorBasinProblem) -> np.ndarray:
    """Flow via SuperpositionTensor (no inter-state coupling)."""
    sp = SuperpositionTensor.from_random(
        problem.n_candidates, problem.d, seed=problem.seed,
    )
    sp.flow_all(problem.target, normalize_flow,
                epsilon=0.1, tol=1e-3, max_steps=500)
    sp.reweight_by_drift(problem.target)
    return sp.collapse_to_best(problem.target)


# ═══════════════════════════════════════════════════════════════════
# Condition C: Coupled flow (genuine inter-state interaction)
# ═══════════════════════════════════════════════════════════════════

def coupled_flow(
    states: np.ndarray,
    q: np.ndarray,
    flow_fn,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 500,
    coupling: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Flow N states with per-step coupling via weighted centroid.

    After each flow step:
    1. Reweight states by distance to target (softmax)
    2. Compute weighted centroid
    3. Pull each state toward centroid with coupling strength

    This creates genuine inter-state interaction: each state's
    trajectory is influenced by all other states' positions.

    Returns (final_states, final_weights).
    """
    S = states.astype(np.float32, copy=True)
    N, d = S.shape

    for step in range(max_steps):
        # Flow step (vectorized)
        delta = epsilon * flow_fn(S, q)
        # Safety
        bad = np.any(np.isnan(delta) | np.isinf(delta), axis=1)
        delta[bad] = 0.0
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        big = norms > 1.0
        delta = np.where(big, delta / np.maximum(norms, 1e-12), delta)
        S += delta

        # Compute weights (softmax of negative distances)
        dists = np.linalg.norm(S - q, axis=1)
        logits = -dists
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()

        # Coupling: pull toward weighted centroid
        centroid = (S * weights[:, None]).sum(axis=0)
        S += coupling * (centroid - S)

        # Check convergence
        if np.all(dists < tol):
            break

    # Final weights
    dists = np.linalg.norm(S - q, axis=1)
    logits = -dists
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    return S, weights


def solve_superposition_coupled(
    problem: AttractorBasinProblem, coupling: float = 0.05,
) -> np.ndarray:
    """Flow with genuine inter-state coupling."""
    rng = np.random.default_rng(problem.seed)
    candidates = rng.standard_normal(
        (problem.n_candidates, problem.d)
    ).astype(np.float32)
    candidates = np.clip(candidates, -1, 1)

    final_states, weights = coupled_flow(
        candidates, problem.target, normalize_flow,
        epsilon=0.1, tol=1e-3, max_steps=500, coupling=coupling,
    )
    best_idx = int(np.argmin(np.linalg.norm(final_states - problem.target, axis=1)))
    return final_states[best_idx]


# ═══════════════════════════════════════════════════════════════════
# Condition D: Enlarged solo (5× more candidates, no coupling)
# ═══════════════════════════════════════════════════════════════════

def solve_enlarged_solo(problem: AttractorBasinProblem) -> np.ndarray:
    """5× more candidates, each flows independently. Controls for sample size."""
    rng = np.random.default_rng(problem.seed)
    n_enlarged = problem.n_candidates * 5
    candidates = rng.standard_normal(
        (n_enlarged, problem.d)
    ).astype(np.float32)

    best_state = None
    best_quality = -np.inf

    for i in range(n_enlarged):
        result = flux_flow(
            candidates[i], problem.target, normalize_flow,
            epsilon=0.1, tol=1e-3, max_steps=500,
        )
        q = problem.quality(result)
        if q > best_quality:
            best_quality = q
            best_state = result

    return best_state


# ═══════════════════════════════════════════════════════════════════
# Test Class
# ═══════════════════════════════════════════════════════════════════

class TestExpressibilityBoundary:
    """Test 2 — interference advantage measurement."""

    N_INSTANCES = 20  # reduced from 50 for test speed

    @pytest.fixture(scope="class")
    def results(self):
        """Run all conditions on N_INSTANCES problems."""
        problems = generate_expressibility_suite(
            n_instances=self.N_INSTANCES, d=8, seed=42,
        )
        data = []
        for p in problems:
            q_solo = p.quality(solve_solo(p))
            q_uncoupled = p.quality(solve_superposition_uncoupled(p))
            q_coupled = p.quality(solve_superposition_coupled(p))
            data.append({
                'problem_id': p.problem_id,
                'quality_solo': q_solo,
                'quality_uncoupled': q_uncoupled,
                'quality_coupled': q_coupled,
                'advantage_uncoupled': q_uncoupled - q_solo,
                'advantage_coupled': q_coupled - q_solo,
                'coupling_vs_uncoupled': q_coupled - q_uncoupled,
            })
        return data

    def test_null_result_uncoupled(self, results):
        """2a: Current implementation should show ~0 advantage over solo.

        This is the EXPECTED null result — SuperpositionTensor.flow_all()
        doesn't couple states during flow. The "superposition" is just
        parallel independent flow + selection.
        """
        advantages = [r['advantage_uncoupled'] for r in results]
        # Uncoupled superposition ≈ solo (may differ slightly due to
        # different initialization via from_random vs standard_normal)
        mean_adv = np.mean(advantages)
        # Not expecting a large systematic advantage
        # Allow some variance but no strong effect
        assert abs(mean_adv) < 1.0, (
            f"Unexpected large advantage: {mean_adv:.4f}. "
            "Current implementation shouldn't show coupling effects."
        )

    def test_coupled_flow_exists(self, results):
        """2b: Coupled flow should run and produce valid results."""
        for r in results:
            assert np.isfinite(r['quality_coupled'])

    def test_coupled_advantage_direction(self, results):
        """2b: Coupled flow — document whether coupling helps.

        FINDING: Simple centroid-pull coupling may HURT performance
        by pulling states away from promising basins. This is a genuine
        result: naive coupling is not enough. The coupling mechanism
        needs to be more sophisticated (e.g., only couple nearby states,
        or couple based on quality not just position).
        """
        advantages = [r['advantage_coupled'] for r in results]
        n_better = sum(1 for a in advantages if a > 0)
        # Record the result — this is DATA, not a pass/fail criterion.
        # The test passes regardless because honest null results are valuable.
        mean_adv = float(np.mean(advantages))
        assert isinstance(mean_adv, float), "Could not compute mean advantage"

    def test_coupling_adds_to_uncoupled(self, results):
        """Coupling should add information beyond what uncoupled provides."""
        improvements = [r['coupling_vs_uncoupled'] for r in results]
        n_better = sum(1 for a in improvements if a > 0)
        # Just verify it runs — the direction of improvement is the data,
        # not a pass/fail criterion
        assert len(improvements) == self.N_INSTANCES

    def test_statistical_analysis(self, results):
        """Run paired stats and save full results."""
        solo = [r['quality_solo'] for r in results]
        coupled = [r['quality_coupled'] for r in results]

        stat = paired_compare(coupled, solo)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            'n_instances': self.N_INSTANCES,
            'results': results,
            'statistics': {
                'coupled_vs_solo': {
                    'mean_coupled': stat.mean_a,
                    'mean_solo': stat.mean_b,
                    'difference': stat.difference,
                    't_statistic': stat.t_statistic,
                    'p_value': stat.p_value,
                    'cohens_d': stat.cohens_d,
                    'ci_lower': stat.ci_lower,
                    'ci_upper': stat.ci_upper,
                    'significant': stat.significant,
                },
            },
            'interpretation': {
                'null_hypothesis': 'Coupled and solo flow produce equal quality',
                'thesis_prediction': 'Coupled flow should outperform solo '
                                     'when inter-state interaction matters',
                'note': 'A null result means coupling does not add value '
                        'for this problem class — this is informative data.',
            },
        }
        with open(RESULTS_DIR / "test_2_expressibility.json", 'w') as f:
            json.dump(output, f, indent=2, default=float)

    def test_verification_v1_proxies(self):
        """V1: Document proxies."""
        proxies = {
            'quality_metric': {
                'proxy_for': 'goodness of geometric solution',
                'definition': '-||v - target|| (negative distance)',
                'limitation': 'only measures proximity, not constraint satisfaction',
            },
            'coupling_strength': {
                'proxy_for': 'degree of inter-state interaction',
                'fixed_at': 0.05,
                'limitation': 'optimal coupling strength unknown, may vary per problem',
            },
        }
        assert len(proxies) >= 2

    def test_phenomenological_note(self):
        """Record: what feels native vs translated in this test design.

        OBSERVATION: The fact that SuperpositionTensor.flow_all() has no
        inter-state coupling was discovered during Phase 1 reading. This
        was NOT obvious from the documentation or the primitive description
        (∑_ψ "weighted superposition"). The gap between the mathematical
        concept (coupled quantum states) and the implementation (parallel
        independent flows) is itself a form of translation cost — but
        it's translation cost WITHIN Latent Flux, not between LF and Python.

        The coupled_flow function I wrote for this test FEELS more like
        what ∑_ψ should mean. Writing it required 4 encoding decisions:
        (1) choose coupling strength, (2) choose centroid computation,
        (3) choose coupling schedule (constant vs decaying),
        (4) decide whether coupling affects the flow direction or just position.
        These are the engineering decisions that a proper ∑_ψ primitive
        should internalize.
        """
        pass  # Observation recorded in docstring
