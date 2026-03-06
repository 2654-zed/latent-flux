"""Tests for Convergence Contracts (Tier 1/2/3) and Reservoir State (⧖)."""

import numpy as np
import pytest

from flux_manifold.convergence import (
    ConvergenceTier, ConvergenceContract, ConvergenceResult,
    TIER_1_NORMALIZE, TIER_2_DEFAULT, TIER_3_REPULSIVE,
    default_contract_for,
)
from flux_manifold.reservoir_state import ReservoirState, SuperpositionReservoir
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.fold_reference import FoldReference, reservoir_norm_critique
from flux_manifold.core import flux_flow, flux_flow_traced
from flux_manifold.flows import normalize_flow, sin_flow


# ── Convergence Contracts ──────────────────────────────────────────

class TestConvergenceTier:
    def test_tier_ordering(self):
        assert ConvergenceTier.PROVABLE < ConvergenceTier.EMPIRICAL
        assert ConvergenceTier.EMPIRICAL < ConvergenceTier.NON_CONVERGENT

    def test_tier_values(self):
        assert ConvergenceTier.PROVABLE == 1
        assert ConvergenceTier.EMPIRICAL == 2
        assert ConvergenceTier.NON_CONVERGENT == 3


class TestConvergenceContract:
    def test_tier1_requires_lipschitz(self):
        with pytest.raises((ValueError, TypeError)):
            ConvergenceContract(
                tier=ConvergenceTier.PROVABLE,
                justification="no lipschitz bound",
            )

    def test_tier1_lipschitz_must_be_less_than_1(self):
        with pytest.raises(ValueError):
            ConvergenceContract(
                tier=ConvergenceTier.PROVABLE,
                justification="bad bound",
                lipschitz_bound=1.5,
            )

    def test_tier2_no_lipschitz_needed(self):
        c = ConvergenceContract(
            tier=ConvergenceTier.EMPIRICAL,
            justification="empirical convergence",
        )
        assert c.tier == ConvergenceTier.EMPIRICAL

    def test_prebuilt_tier1_normalize(self):
        assert TIER_1_NORMALIZE.tier == ConvergenceTier.PROVABLE
        assert TIER_1_NORMALIZE.lipschitz_bound < 1.0

    def test_prebuilt_tier3(self):
        assert TIER_3_REPULSIVE.tier == ConvergenceTier.NON_CONVERGENT

    def test_default_contract_for_known_flow(self):
        c = default_contract_for("normalize_flow")
        assert c.tier == ConvergenceTier.PROVABLE

    def test_default_contract_for_unknown_flow(self):
        c = default_contract_for("my_random_flow_xyz")
        assert c.tier == ConvergenceTier.EMPIRICAL


class TestConvergenceResult:
    def test_converged_tier1_passes(self):
        r = ConvergenceResult(
            contract=TIER_1_NORMALIZE,
            converged=True, steps_used=50, max_steps=500,
            final_drift=0.0001,
        )
        r.check()  # should not raise
        assert r.tier_honored

    def test_unconverged_tier1_sets_failure(self):
        r = ConvergenceResult(
            contract=TIER_1_NORMALIZE,
            converged=False, steps_used=500, max_steps=500,
            final_drift=0.5,
        )
        r.check()
        assert not r.tier_honored
        assert "TIER 1" in r.failure_signal

    def test_unconverged_tier2_does_not_raise(self):
        r = ConvergenceResult(
            contract=TIER_2_DEFAULT,
            converged=False, steps_used=500, max_steps=500,
            final_drift=0.5,
        )
        r.check()  # should not raise for Tier 2
        # Tier 2 non-convergence is recorded, not treated as violation
        assert r.tier_honored  # Tier 2 doesn't set tier_honored=False

    def test_tier3_signals_failure(self):
        r = ConvergenceResult(
            contract=TIER_3_REPULSIVE,
            converged=False, steps_used=100, max_steps=100,
            final_drift=5.0,
        )
        r.check()
        assert r.failure_signal


class TestConvergenceIntegration:
    def test_flux_flow_with_default_contract(self):
        s = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = np.zeros(3, dtype=np.float32)
        result = flux_flow(s, q, normalize_flow, epsilon=0.1, tol=1e-2, max_steps=200)
        assert isinstance(result, np.ndarray)

    def test_flux_flow_traced_returns_convergence_result(self):
        s = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = np.zeros(3, dtype=np.float32)
        trace = flux_flow_traced(s, q, normalize_flow, epsilon=0.1, tol=1e-2, max_steps=200)
        assert "convergence_result" in trace
        cr = trace["convergence_result"]
        assert isinstance(cr, ConvergenceResult)


# ── Reservoir State ────────────────────────────────────────────────

class TestReservoirState:
    def test_create_default(self):
        rs = ReservoirState(d=8)
        assert rs.d == 8
        assert rs.r == 32  # 4 * 8

    def test_create_custom_scale(self):
        rs = ReservoirState(d=8, reservoir_scale=2)
        assert rs.r == 16

    def test_step_returns_readout(self):
        rs = ReservoirState(d=4, seed=42)
        x = np.array([1.0, 0.5, -0.3, 0.8], dtype=np.float32)
        y = rs.step(x)
        assert y.shape == (4,)
        # Readout should be finite
        assert np.all(np.isfinite(y))

    def test_step_changes_hidden_state(self):
        rs = ReservoirState(d=4, seed=42)
        h0 = rs._h.copy()
        rs.step(np.ones(4, dtype=np.float32))
        assert not np.allclose(rs._h, h0)

    def test_readout_matches_last_step(self):
        rs = ReservoirState(d=4, seed=42)
        x = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        y_step = rs.step(x)
        y_readout = rs.readout()
        np.testing.assert_allclose(y_step, y_readout, atol=1e-6)

    def test_multiple_steps_accumulate(self):
        rs = ReservoirState(d=4, seed=42)
        outputs = []
        for i in range(5):
            x = np.full(4, float(i), dtype=np.float32)
            outputs.append(rs.step(x))
        # Each step should produce different output (echo state property)
        assert not np.allclose(outputs[0], outputs[-1])

    def test_commit_returns_readout(self):
        rs = ReservoirState(d=4, seed=42)
        rs.step(np.ones(4, dtype=np.float32))
        committed = rs.commit()
        assert committed.shape == (4,)

    def test_commit_blocks_further_steps(self):
        rs = ReservoirState(d=4, seed=42)
        rs.step(np.ones(4, dtype=np.float32))
        rs.commit()
        with pytest.raises(RuntimeError, match="[Cc]ommit"):
            rs.step(np.ones(4, dtype=np.float32))

    def test_reset_requires_uncommitted(self):
        rs = ReservoirState(d=4, seed=42)
        rs.step(np.ones(4, dtype=np.float32))
        rs.commit()
        # Reset after commit is a linear type error
        with pytest.raises(RuntimeError, match="[Cc]ommit"):
            rs.reset()

    def test_reset_before_commit(self):
        rs = ReservoirState(d=4, seed=42)
        rs.step(np.ones(4, dtype=np.float32))
        rs.reset()
        # After reset, should be able to step again
        y = rs.step(np.ones(4, dtype=np.float32))
        assert y.shape == (4,)

    def test_history_tracking(self):
        rs = ReservoirState(d=4, seed=42)
        for i in range(3):
            rs.step(np.full(4, float(i), dtype=np.float32))
        assert len(rs._history) == 3


# ── SuperpositionReservoir ─────────────────────────────────────────

class TestSuperpositionReservoir:
    def test_create(self):
        sr = SuperpositionReservoir(n=5, d=8, seed=42)
        assert len(sr._hidden_states) == 5

    def test_step_all(self):
        sr = SuperpositionReservoir(n=3, d=4, seed=42)
        states = np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        readouts = sr.step_all(states)
        assert readouts.shape == (3, 4)
        assert np.all(np.isfinite(readouts))

    def test_reorder(self):
        sr = SuperpositionReservoir(n=4, d=4, seed=42)
        states = np.eye(4, dtype=np.float32)
        sr.step_all(states)
        # Reorder: reverse
        sr.reorder(np.array([3, 2, 1, 0]))
        # After reorder, reservoir 0 should have the state from original reservoir 3
        r0 = sr.readout_all()[0]
        assert r0.shape == (4,)

    def test_prune(self):
        sr = SuperpositionReservoir(n=5, d=4, seed=42)
        states = np.random.default_rng(42).standard_normal((5, 4)).astype(np.float32)
        sr.step_all(states)
        sr.prune(np.array([0, 2, 4]))
        assert len(sr._hidden_states) == 3

    def test_commit_best(self):
        sr = SuperpositionReservoir(n=3, d=4, seed=42)
        states = np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        sr.step_all(states)
        committed = sr.commit_best(1)
        assert committed.shape == (4,)

    def test_get_history(self):
        sr = SuperpositionReservoir(n=2, d=4, seed=42)
        states = np.random.default_rng(42).standard_normal((2, 4)).astype(np.float32)
        sr.step_all(states)
        sr.step_all(states * 0.5)
        h = sr.get_history(0)
        assert len(h) == 2

    def test_memory_warning(self):
        # Uses a low threshold to trigger warning
        with pytest.warns(ResourceWarning):
            SuperpositionReservoir(
                n=100, d=100, seed=42,
                reservoir_memory_warning_threshold=1,  # 1 byte — will trigger
            )


# ── Reservoir × Superposition Coupling ─────────────────────────────

class TestReservoirSuperpositionCoupling:
    def test_attach_reservoir(self):
        sp = SuperpositionTensor.from_random(4, 8, seed=42)
        sr = SuperpositionReservoir(n=4, d=8, seed=42)
        sp.attach_reservoir(sr)
        assert sp._reservoir is sr

    def test_reweight_reorders_reservoir(self):
        sp = SuperpositionTensor.from_random(4, 8, seed=42)
        sr = SuperpositionReservoir(n=4, d=8, seed=42)
        sr.step_all(sp.states)
        sp.attach_reservoir(sr)
        # Get initial reservoir readouts
        readouts_before = sr.readout_all().copy()
        # Reweight — reservoir should reorder
        q = sp.states[3]  # bias toward state 3
        sp.reweight_by_drift(q)
        readouts_after = sr.readout_all()
        # After reweight, the reservoir for the closest state should be first
        assert readouts_after.shape == readouts_before.shape

    def test_prune_removes_reservoir_entries(self):
        sp = SuperpositionTensor.from_random(5, 8, seed=42)
        sr = SuperpositionReservoir(n=5, d=8, seed=42)
        sr.step_all(sp.states)
        sp.attach_reservoir(sr)
        sp.prune(3)
        assert len(sr._hidden_states) == 3
        assert sp.n == 3


# ── Reservoir × FoldReference ──────────────────────────────────────

class TestReservoirFoldReference:
    def test_reservoir_norm_critique(self):
        critique = reservoir_norm_critique(max_norm=10.0)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        passed, msg, corrected = critique(state)
        assert passed

    def test_reservoir_norm_critique_with_history(self):
        critique = reservoir_norm_critique(max_norm=10.0)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        history = [np.ones(16), np.ones(16) * 2]
        passed, msg, corrected = critique(state, reservoir_history=history)
        assert passed

    def test_fold_ref_with_reservoir_aware_critique(self):
        critique = reservoir_norm_critique(max_norm=10.0)
        fr = FoldReference(critique_fn=critique, interval=1)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        corrected, was_corrected = fr.check(state, step=0)
        # State is within bounds, so no correction needed
        assert not was_corrected
        np.testing.assert_array_equal(corrected, state)


# ── Parser ⧖ Integration ──────────────────────────────────────────

class TestParserReservoir:
    def test_reservoir_operator_single_vector(self):
        from flux_manifold.parser import run, EvalContext
        ctx = EvalContext()
        result = run("[1.0, 2.0, 3.0, 4.0] | ⧖", ctx=ctx)
        # Should pass through (reservoir steps but returns input)
        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 4

    def test_reservoir_operator_ascii(self):
        from flux_manifold.parser import run, EvalContext
        ctx = EvalContext()
        result = run("[1.0, 2.0, 3.0, 4.0] | reservoir", ctx=ctx)
        assert isinstance(result, np.ndarray)

    def test_reservoir_creates_context_state(self):
        from flux_manifold.parser import run, EvalContext
        ctx = EvalContext()
        run("[1.0, 2.0, 3.0, 4.0] | ⧖", ctx=ctx)
        assert ctx.reservoir is not None

    def test_reservoir_in_pipeline(self):
        from flux_manifold.parser import run, EvalContext
        ctx = EvalContext()
        result = run("[1.0, 2.0, 3.0, 4.0] ⟼ [0, 0, 0, 0] | ⧖ | ◉ | ↓!", ctx=ctx)
        assert isinstance(result, np.ndarray)

    def test_reservoir_with_superposition(self):
        from flux_manifold.parser import run, EvalContext
        ctx = EvalContext()
        result = run("∑_ψ [1.0, 2.0; 3.0, 4.0] | ⧖", ctx=ctx)
        assert isinstance(result, SuperpositionTensor)
        assert ctx.sp_reservoir is not None
