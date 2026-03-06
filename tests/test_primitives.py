"""Tests for the 6 new Latent Flux primitives."""

import numpy as np
import pytest

from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.drift_equivalence import DriftEquivalence
from flux_manifold.commitment_sink import CommitmentSink
from flux_manifold.abstraction_cascade import AbstractionCascade
from flux_manifold.fold_reference import FoldReference, no_nan_critique, norm_bound_critique
from flux_manifold.dimensional_squeeze import DimensionalSqueeze
from flux_manifold.flows import normalize_flow


# ── SuperpositionTensor (∑_ψ) ──────────────────────────────────────

class TestSuperposition:
    def test_create_uniform_weights(self):
        states = np.random.default_rng(42).standard_normal((8, 16)).astype(np.float32)
        sp = SuperpositionTensor(states)
        assert sp.n == 8
        assert sp.d == 16
        np.testing.assert_allclose(sp.weights.sum(), 1.0, atol=1e-6)

    def test_from_random(self):
        sp = SuperpositionTensor.from_random(10, 32, seed=42)
        assert sp.n == 10
        assert sp.d == 32

    def test_mean_state(self):
        states = np.array([[1, 0], [0, 1]], dtype=np.float32)
        sp = SuperpositionTensor(states)
        mean = sp.mean_state()
        np.testing.assert_allclose(mean, [0.5, 0.5], atol=1e-6)

    def test_entropy_uniform(self):
        sp = SuperpositionTensor.from_random(8, 4, seed=42)
        # Uniform over 8 → entropy = log2(8) = 3.0
        assert abs(sp.entropy() - 3.0) < 0.01

    def test_entropy_collapsed(self):
        states = np.ones((4, 3), dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sp = SuperpositionTensor(states, weights=weights)
        assert sp.entropy() == 0.0

    def test_flow_all_converges(self):
        sp = SuperpositionTensor.from_random(4, 8, seed=42, scale=0.5)
        q = np.zeros(8, dtype=np.float32)
        traces = sp.flow_all(q, normalize_flow, epsilon=0.1, tol=1e-2, max_steps=500)
        assert len(traces) == 4
        assert all(t["converged"] for t in traces)

    def test_reweight_by_drift(self):
        states = np.array([[0.1, 0], [10, 0]], dtype=np.float32)
        sp = SuperpositionTensor(states)
        q = np.zeros(2, dtype=np.float32)
        sp.reweight_by_drift(q)
        # State [0.1, 0] is closer → higher weight
        assert sp.weights[0] > sp.weights[1]

    def test_collapse_to_best(self):
        states = np.array([[0.1, 0], [5.0, 5.0]], dtype=np.float32)
        sp = SuperpositionTensor(states)
        q = np.zeros(2, dtype=np.float32)
        best = sp.collapse_to_best(q)
        np.testing.assert_array_equal(best, states[0])

    def test_prune(self):
        sp = SuperpositionTensor.from_random(10, 4, seed=42)
        q = np.zeros(4, dtype=np.float32)
        sp.reweight_by_drift(q)
        sp.prune(keep=3)
        assert sp.n == 3
        np.testing.assert_allclose(sp.weights.sum(), 1.0, atol=1e-6)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            SuperpositionTensor(np.zeros(5, dtype=np.float32))


# ── DriftEquivalence (≅) ───────────────────────────────────────────

class TestDriftEquivalence:
    def test_equivalent_close(self):
        eq = DriftEquivalence(tolerance=0.1)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.05, 0.0], dtype=np.float32)
        assert eq.equivalent(a, b)

    def test_not_equivalent_far(self):
        eq = DriftEquivalence(tolerance=0.01)
        a = np.zeros(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        assert not eq.equivalent(a, b)

    def test_cosine_metric(self):
        eq = DriftEquivalence(tolerance=0.1, metric="cosine")
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.99, 0.01], dtype=np.float32)
        assert eq.equivalent(a, b)

    def test_quality_perfect(self):
        eq = DriftEquivalence(tolerance=1.0)
        a = np.array([1.0, 1.0], dtype=np.float32)
        assert eq.quality(a, a) == 1.0

    def test_quality_at_boundary(self):
        eq = DriftEquivalence(tolerance=1.0)
        a = np.array([0.0], dtype=np.float32)
        b = np.array([1.0], dtype=np.float32)
        assert abs(eq.quality(a, b)) < 0.01

    def test_best_equivalent(self):
        eq = DriftEquivalence(tolerance=0.5)
        candidates = np.array([[0.0], [0.3], [0.9]], dtype=np.float32)
        target = np.array([0.25], dtype=np.float32)
        idx, dist = eq.best_equivalent(candidates, target)
        assert idx == 1

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance"):
            DriftEquivalence(tolerance=-1)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            DriftEquivalence(metric="manhattan")


# ── CommitmentSink (↓!) ────────────────────────────────────────────

class TestCommitmentSink:
    def test_commit_is_irreversible(self):
        cs = CommitmentSink()
        state = np.array([1.0, 2.0], dtype=np.float32)
        cs.commit(state, reason="test")
        assert cs.committed
        with pytest.raises(RuntimeError, match="irreversible"):
            cs.commit(state)

    def test_commit_returns_copy(self):
        cs = CommitmentSink()
        state = np.array([1.0, 2.0], dtype=np.float32)
        result = cs.commit(state)
        state[0] = 999
        assert result[0] == 1.0  # Not mutated

    def test_auto_commit_entropy(self):
        cs = CommitmentSink(entropy_threshold=1.0)
        # Create low-entropy superposition (1 dominant weight)
        states = np.array([[0.1, 0], [5, 5], [9, 9]], dtype=np.float32)
        weights = np.array([0.98, 0.01, 0.01], dtype=np.float32)
        sp = SuperpositionTensor(states, weights=weights)
        q = np.zeros(2, dtype=np.float32)

        result = cs.try_commit(sp, q)
        assert result is not None
        assert cs.commit_reason == "entropy_low"

    def test_auto_commit_drift(self):
        cs = CommitmentSink(drift_window=3, drift_threshold=0.05)
        sp = SuperpositionTensor.from_random(4, 4, seed=42)
        q = np.zeros(4, dtype=np.float32)

        # Simulate stable drift
        drift_trace = [0.01, 0.008, 0.009, 0.007]
        result = cs.try_commit(sp, q, drift_trace)
        assert result is not None
        assert cs.commit_reason == "drift_stable"

    def test_no_commit_high_entropy(self):
        cs = CommitmentSink(entropy_threshold=0.1)
        sp = SuperpositionTensor.from_random(8, 4, seed=42)  # Uniform = high entropy
        q = np.zeros(4, dtype=np.float32)
        result = cs.try_commit(sp, q)
        assert result is None
        assert not cs.committed


# ── AbstractionCascade (⇑) ─────────────────────────────────────────

class TestAbstractionCascade:
    def test_single_state_cascade(self):
        ac = AbstractionCascade(levels=3, min_dim=2)
        state = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        levels = ac.cascade_single(state)
        assert len(levels) == 3
        assert levels[0]["dim"] == 32
        assert levels[1]["dim"] == 16
        assert levels[2]["dim"] == 8

    def test_multi_state_cascade(self):
        ac = AbstractionCascade(levels=2)
        states = np.random.default_rng(42).standard_normal((10, 16)).astype(np.float32)
        levels = ac.cascade(states)
        assert len(levels) == 2
        assert levels[0]["dim"] == 16
        assert levels[1]["dim"] == 8

    def test_min_dim_respected(self):
        ac = AbstractionCascade(levels=10, min_dim=4)
        state = np.random.default_rng(42).standard_normal(8).astype(np.float32)
        levels = ac.cascade_single(state)
        for lvl in levels:
            assert lvl["dim"] >= 4

    def test_pca_variance_ratio(self):
        ac = AbstractionCascade(levels=2)
        rng = np.random.default_rng(42)
        states = rng.standard_normal((20, 16)).astype(np.float32)
        levels = ac.cascade(states)
        if levels[1]["explained_variance_ratio"] is not None:
            assert levels[1]["explained_variance_ratio"].sum() <= 1.0 + 1e-6


# ── FoldReference (◉) ──────────────────────────────────────────────

class TestFoldReference:
    def test_no_nan_critique_clean(self):
        ok, diag, correction = no_nan_critique(np.array([1.0, 2.0], dtype=np.float32))
        assert ok
        assert correction is None

    def test_no_nan_critique_catches_nan(self):
        ok, diag, correction = no_nan_critique(np.array([1.0, np.nan], dtype=np.float32))
        assert not ok
        assert correction is not None
        assert not np.any(np.isnan(correction))

    def test_norm_bound_critique(self):
        critique = norm_bound_critique(max_norm=5.0)
        big = np.ones(100, dtype=np.float32) * 10  # norm >> 5
        ok, diag, correction = critique(big)
        assert not ok
        assert correction is not None
        assert np.linalg.norm(correction) <= 5.0 + 1e-6

    def test_fold_reference_applies_corrections(self):
        def always_fix(s):
            return False, "always bad", np.zeros_like(s)

        fr = FoldReference(always_fix, interval=1, max_corrections=5)
        state = np.ones(4, dtype=np.float32)
        corrected, was = fr.check(state, step=0)
        assert was
        np.testing.assert_array_equal(corrected, np.zeros(4))
        assert fr.corrections_count == 1

    def test_fold_reference_respects_interval(self):
        calls = []
        def tracking_critique(s):
            calls.append(1)
            return True, "ok", None

        fr = FoldReference(tracking_critique, interval=5)
        for step in range(10):
            fr.check(np.zeros(2, dtype=np.float32), step=step)
        # Only steps 0, 5 trigger critique
        assert len(calls) == 2

    def test_max_corrections_limit(self):
        def always_fix(s):
            return False, "bad", np.zeros_like(s)

        fr = FoldReference(always_fix, interval=1, max_corrections=2)
        for step in range(5):
            fr.check(np.ones(2, dtype=np.float32), step=step)
        assert fr.corrections_count == 2  # Capped at 2


# ── DimensionalSqueeze (∇↓) ───────────────────────────────────────

class TestDimensionalSqueeze:
    def test_pca_squeeze(self):
        ds = DimensionalSqueeze(target_dim=4, method="pca")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 16)).astype(np.float32)
        ds.fit(data)
        squeezed = ds.squeeze(data[0])
        assert squeezed.shape == (4,)

    def test_random_projection_squeeze(self):
        ds = DimensionalSqueeze(target_dim=8, method="random_projection")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 32)).astype(np.float32)
        ds.fit(data)
        squeezed = ds.squeeze(data[0])
        assert squeezed.shape == (8,)

    def test_batch_squeeze(self):
        ds = DimensionalSqueeze(target_dim=4, method="pca")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 16)).astype(np.float32)
        ds.fit(data)
        batch = ds.squeeze(data[:5])
        assert batch.shape == (5, 4)

    def test_unsqueeze_reconstruction(self):
        ds = DimensionalSqueeze(target_dim=8, method="pca")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 16)).astype(np.float32)
        ds.fit(data)
        squeezed = ds.squeeze(data[0])
        reconstructed = ds.unsqueeze(squeezed)
        assert reconstructed.shape == (16,)
        # Lossy – but should preserve some structure
        assert np.isfinite(reconstructed).all()

    def test_compression_ratio(self):
        ds = DimensionalSqueeze(target_dim=4, method="pca")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 16)).astype(np.float32)
        ds.fit(data)
        assert ds.compression_ratio == 16 / 4

    def test_no_squeeze_needed(self):
        ds = DimensionalSqueeze(target_dim=16)
        data = np.random.default_rng(42).standard_normal((10, 8)).astype(np.float32)
        ds.fit(data)
        squeezed = ds.squeeze(data[0])
        assert squeezed.shape == (8,)  # target > d, no reduction

    def test_squeeze_before_fit_raises(self):
        ds = DimensionalSqueeze(target_dim=4)
        with pytest.raises(RuntimeError, match="fit"):
            ds.squeeze(np.zeros(8, dtype=np.float32))

    def test_invalid_target_dim_raises(self):
        with pytest.raises(ValueError, match="target_dim"):
            DimensionalSqueeze(target_dim=0)
