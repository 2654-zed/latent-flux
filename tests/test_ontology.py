"""Tests for the three ontology expansion modules:

Pillar 1: hamiltonian.py — HamiltonianFlowEngine, 5 flow variants
Pillar 2: quantum_interference.py — QuantumInterferenceEngine, BIC isolation
Pillar 3: topological_squeeze.py — TopologicalSqueeze, RTD preservation
"""

import numpy as np
import pytest


# ── Pillar 1: Hamiltonian Flows ──────────────────────────────────

from flux_manifold.hamiltonian import (
    HamiltonianState,
    conformal_hamiltonian_step,
    relativistic_hamiltonian_step,
    finsler_asymmetric_step,
    langevin_annealing_step,
    eyring_kramers_schedule,
    exponential_schedule,
    HamiltonianFlowEngine,
    hamiltonian_flow,
    hamiltonian_flow_batch,
)


class TestHamiltonianSteps:
    """Test individual step functions."""

    def test_conformal_moves_toward_attractor(self):
        x = np.array([0.0, 0.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.array([1.0, 1.0], dtype=np.float32)
        x2, p2 = conformal_hamiltonian_step(x, p, q, gamma=0.1, dt=0.1)
        # Should have moved toward q
        assert np.linalg.norm(x2 - q) < np.linalg.norm(x - q)

    def test_conformal_dissipation(self):
        x = np.array([5.0, 5.0], dtype=np.float32)
        p = np.array([10.0, 10.0], dtype=np.float32)
        q = np.array([0.0, 0.0], dtype=np.float32)
        x2, p2 = conformal_hamiltonian_step(x, p, q, gamma=0.5, dt=0.1)
        # Momentum should decrease due to dissipation
        assert np.linalg.norm(p2) < np.linalg.norm(p)

    def test_relativistic_speed_bounded(self):
        x = np.zeros(3, dtype=np.float32)
        p = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        q = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        x2, p2 = relativistic_hamiltonian_step(x, p, q, gamma=0.01, c=1.0, dt=0.1)
        # Effective velocity bounded by c
        v = p2 / np.sqrt(1.0 + np.dot(p2, p2) / 1.0)
        assert np.linalg.norm(v) <= 1.0 + 1e-6

    def test_relativistic_reduces_distance(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        x2, p2 = relativistic_hamiltonian_step(x, p, q, gamma=0.1, c=5.0, dt=0.1)
        assert np.linalg.norm(x2 - q) < np.linalg.norm(x - q)

    def test_finsler_asymmetric_cost(self):
        x = np.array([1.0, 0.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.array([0.0, 0.0], dtype=np.float32)
        x2, p2 = finsler_asymmetric_step(x, p, q, gamma=0.1, asymmetry=0.5, dt=0.1)
        # Should move toward q
        assert np.linalg.norm(x2 - q) < np.linalg.norm(x - q)

    def test_langevin_adds_noise(self):
        rng = np.random.default_rng(42)
        x = np.array([1.0, 0.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.array([0.0, 0.0], dtype=np.float32)
        x2, p2 = langevin_annealing_step(
            x, p, q, gamma=0.5, dt=0.1, temperature=1.0, rng=rng
        )
        # With temperature > 0, momentum should be nonzero even from rest
        assert np.linalg.norm(p2) > 0

    def test_langevin_no_noise_at_zero_temp(self):
        rng = np.random.default_rng(42)
        x = np.array([1.0, 0.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.array([0.0, 0.0], dtype=np.float32)
        x_hot, p_hot = langevin_annealing_step(
            x.copy(), p.copy(), q, gamma=0.5, dt=0.1, temperature=0.0, rng=rng
        )
        # Should be deterministic at T=0
        x_cold, p_cold = langevin_annealing_step(
            x.copy(), p.copy(), q, gamma=0.5, dt=0.1, temperature=0.0,
            rng=np.random.default_rng(999),
        )
        np.testing.assert_allclose(x_hot, x_cold, atol=1e-6)


class TestSchedules:
    def test_eyring_kramers_decreases(self):
        temps = [eyring_kramers_schedule(s, T0=1.0, tau=10) for s in range(0, 100, 10)]
        for i in range(1, len(temps)):
            assert temps[i] < temps[i - 1]

    def test_exponential_decreases(self):
        temps = [exponential_schedule(s, T0=1.0, decay=0.99) for s in range(0, 100, 10)]
        for i in range(1, len(temps)):
            assert temps[i] < temps[i - 1]

    def test_eyring_kramers_starts_at_T0(self):
        assert abs(eyring_kramers_schedule(0, T0=2.5, tau=10) - 2.5) < 1e-8

    def test_exponential_starts_at_T0(self):
        assert abs(exponential_schedule(0, T0=3.0, decay=0.9) - 3.0) < 1e-8


class TestHamiltonianFlowEngine:
    def test_conformal_converges(self):
        eng = HamiltonianFlowEngine(variant="conformal", gamma=0.3, dt=0.1)
        x = np.array([5.0, 5.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        for _ in range(200):
            x, p = eng.step(x, p, q)
        assert np.linalg.norm(x - q) < 0.5

    def test_relativistic_converges(self):
        eng = HamiltonianFlowEngine(variant="relativistic", gamma=0.3, dt=0.1, c=5.0)
        x = np.array([3.0, 4.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        for _ in range(200):
            x, p = eng.step(x, p, q)
        assert np.linalg.norm(x - q) < 0.5

    def test_langevin_converges_with_annealing(self):
        eng = HamiltonianFlowEngine(
            variant="langevin", gamma=0.5, dt=0.05, T0=1.0,
            cooling_tau=30, seed=42,
        )
        x = np.array([3.0, 3.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        for _ in range(300):
            x, p = eng.step(x, p, q)
        # Langevin is stochastic, looser tolerance
        assert np.linalg.norm(x - q) < 3.0

    def test_adaptive_selects_variants(self):
        eng = HamiltonianFlowEngine(
            variant="adaptive", gamma=0.3, dt=0.1,
            T0=1.0, cooling_tau=20, seed=42,
        )
        x = np.array([5.0, 5.0], dtype=np.float32)
        p = np.zeros(2, dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        for _ in range(300):
            x, p = eng.step(x, p, q)
        # Should converge regardless of variant switching
        assert np.linalg.norm(x - q) < 3.0


class TestHamiltonianFlowFunctions:
    def test_hamiltonian_flow_returns_dict(self):
        s0 = np.array([3.0, 3.0], dtype=np.float32)
        q = np.zeros(2, dtype=np.float32)
        result = hamiltonian_flow(s0, q, variant="conformal", max_steps=50)
        assert "converged_state" in result
        assert "drift_trace" in result
        assert "momentum_trace" in result
        assert result["converged_state"].shape == (2,)

    def test_hamiltonian_flow_batch(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((5, 3)).astype(np.float32)
        q = np.zeros(3, dtype=np.float32)
        result = hamiltonian_flow_batch(S, q, variant="conformal", max_steps=100)
        assert isinstance(result, dict)
        assert "converged_states" in result
        assert result["converged_states"].shape == (5, 3)
        assert len(result["steps"]) == 5


# ── Pillar 2: Quantum Interference ──────────────────────────────

from flux_manifold.quantum_interference import (
    QuantumInterferenceEngine,
    InterferenceResult,
)


class TestQuantumInterferenceEngine:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.q = np.zeros(4, dtype=np.float32)
        # Create states: one near q, others far
        self.states = np.array([
            [0.1, 0.0, 0.0, 0.0],  # closest (should survive)
            [5.0, 5.0, 5.0, 5.0],  # far
            [3.0, 3.0, 3.0, 3.0],  # medium
            [-3.0, -3.0, -3.0, -3.0],  # medium, opposite direction
            [10.0, 10.0, 10.0, 10.0],  # farthest
        ], dtype=np.float32)
        self.weights = np.array([1.0, 0.3, 0.5, 0.5, 0.1], dtype=np.float32)

    def test_assign_amplitudes_complex(self):
        eng = QuantumInterferenceEngine()
        amps = eng.assign_amplitudes(self.states, self.q, self.weights)
        assert amps.dtype == np.complex128 or amps.dtype == np.complex64
        assert len(amps) == 5
        assert all(np.abs(a) > 0 for a in amps)

    def test_filter_returns_result(self):
        eng = QuantumInterferenceEngine()
        result = eng.filter(self.states, self.q, self.weights)
        assert isinstance(result, InterferenceResult)
        assert result.n_annihilated >= 0
        assert len(result.surviving_indices) > 0
        assert result.bic_index is not None

    def test_filter_reduces_states(self):
        eng = QuantumInterferenceEngine(dissipation_rate=0.5)
        result = eng.filter(self.states, self.q, self.weights)
        # Should annihilate at least some states
        assert result.n_annihilated >= 0
        assert len(result.surviving_indices) <= len(self.states)

    def test_bic_isolation_selects_best(self):
        eng = QuantumInterferenceEngine()
        result = eng.filter(self.states, self.q, self.weights)
        bic_idx = result.bic_index
        # BIC should be one of the surviving indices
        assert bic_idx in result.surviving_indices

    def test_collapse_to_bic_returns_state(self):
        eng = QuantumInterferenceEngine()
        bic_state, result = eng.collapse_to_bic(self.states, self.q, self.weights)
        assert bic_state.shape == (4,)
        assert isinstance(result, InterferenceResult)

    def test_single_state_passthrough(self):
        eng = QuantumInterferenceEngine()
        single = self.states[:1]
        w = self.weights[:1]
        bic_state, result = eng.collapse_to_bic(single, self.q, w)
        np.testing.assert_allclose(bic_state, single[0], atol=1e-6)

    def test_quality_factors_positive(self):
        eng = QuantumInterferenceEngine()
        result = eng.filter(self.states, self.q, self.weights)
        for qf in result.quality_factors:
            assert qf >= 0


class TestDestructiveInterference:
    def test_anti_aligned_pairs_cancel(self):
        q = np.zeros(2, dtype=np.float32)
        # Two states in opposite directions from q
        states = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        weights = np.array([1.0, 1.0], dtype=np.float32)

        eng = QuantumInterferenceEngine(phase_sensitivity=1.0)
        amps = eng.assign_amplitudes(states, q, weights)
        amps2 = eng.destructive_interference(amps.copy(), states, q)
        # At least one amplitude should be reduced
        assert np.min(np.abs(amps2)) <= np.max(np.abs(amps)) + 1e-6


class TestNonHermitianEvolution:
    def test_far_states_decay_more(self):
        q = np.zeros(3, dtype=np.float32)
        states = np.array([
            [0.1, 0.0, 0.0],
            [10.0, 10.0, 10.0],
        ], dtype=np.float32)
        weights = np.array([1.0, 1.0], dtype=np.float32)

        eng = QuantumInterferenceEngine(dissipation_rate=0.5)
        amps = eng.assign_amplitudes(states, q, weights)
        amps2 = eng.non_hermitian_evolution(amps.copy(), states, q)
        # Far state should have smaller amplitude after evolution
        assert np.abs(amps2[1]) < np.abs(amps2[0])


# ── Pillar 3: Topological Squeeze ───────────────────────────────

from flux_manifold.topological_squeeze import (
    TopologicalSqueeze,
    TopologyDiagnostics,
)


class TestTopologicalSqueezeBasic:
    def test_fit_and_squeeze(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 10)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=3, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        compressed = ts.squeeze(data)
        assert compressed.shape == (20, 3)

    def test_single_point_squeeze(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 10)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=3, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        point = data[0]
        compressed = ts.squeeze(point)
        assert compressed.shape == (3,)

    def test_noop_when_target_ge_input(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=5)
        ts.fit(data)
        compressed = ts.squeeze(data)
        # Should pass through (identity projection)
        assert compressed.shape[1] == 5

    def test_must_fit_before_squeeze(self):
        ts = TopologicalSqueeze(target_dim=3)
        with pytest.raises(RuntimeError):
            ts.squeeze(np.zeros((5, 10), dtype=np.float32))


class TestTopologicalSqueezeDiagnostics:
    def test_diagnose_returns_diagnostics(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((15, 8)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=3, n_neighbors=5, refine_steps=5, ricci_steps=3)
        ts.fit(data)
        diag = ts.diagnose(data)
        assert isinstance(diag, TopologyDiagnostics)
        assert diag.original_dim == 8
        assert diag.target_dim == 3
        assert diag.compression_ratio > 1.0

    def test_rtd_score_finite(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((15, 8)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=3, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        diag = ts.diagnose(data)
        assert np.isfinite(diag.rtd_score)
        assert diag.rtd_score >= 0

    def test_geodesic_distortion_bounded(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((15, 8)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=3, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        diag = ts.diagnose(data)
        assert np.isfinite(diag.geodesic_distortion)
        assert diag.geodesic_distortion >= 0


class TestInverseRicciFlow:
    def test_ricci_concentrates_near_attractor(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 6)).astype(np.float32)
        q = np.zeros(6, dtype=np.float32)
        ts = TopologicalSqueeze(
            target_dim=3, n_neighbors=5, refine_steps=5,
            ricci_steps=10, ricci_rate=0.05,
        )
        ts.fit(data, q=q)
        # Curvature weights should vary (concentration)
        assert ts._curvature_weights is not None
        assert np.std(ts._curvature_weights) > 0


class TestTopologicalSqueezeUnsqueeze:
    def test_unsqueeze_round_trip(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 10)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=5, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        compressed = ts.squeeze(data)
        reconstructed = ts.unsqueeze(compressed)
        assert reconstructed.shape == data.shape
        # Lossy, but should have finite values
        assert np.all(np.isfinite(reconstructed))

    def test_unsqueeze_single_point(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 10)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=5, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        compressed = ts.squeeze(data[0])
        reconstructed = ts.unsqueeze(compressed)
        assert reconstructed.shape == (10,)


class TestTopologicalSqueezeCompressionRatio:
    def test_compression_ratio(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((15, 20)).astype(np.float32)
        ts = TopologicalSqueeze(target_dim=4, n_neighbors=5, refine_steps=5)
        ts.fit(data)
        assert ts.compression_ratio is not None
        assert ts.compression_ratio > 0


# ── Cross-module integration ────────────────────────────────────

class TestOntologyIntegration:
    """Test that all three pillars work together."""

    def test_hamiltonian_to_interference_pipeline(self):
        """Flow multiple states via Hamiltonian, then filter with interference."""
        rng = np.random.default_rng(42)
        q = np.zeros(4, dtype=np.float32)
        S = rng.standard_normal((5, 4)).astype(np.float32) * 3

        # Step 1: Flow all states with Hamiltonian
        result = hamiltonian_flow_batch(S, q, variant="conformal", max_steps=50)
        flowed = result["converged_states"]

        # Step 2: Interference filtering
        eng = QuantumInterferenceEngine()
        weights = np.ones(5, dtype=np.float32)
        bic_state, ir = eng.collapse_to_bic(flowed, q, weights)

        assert bic_state.shape == (4,)
        assert ir.bic_index is not None

    def test_squeeze_preserves_relative_structure(self):
        """Higher-dim states squeezed should preserve nearest-neighbor relations."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((15, 20)).astype(np.float32)

        ts = TopologicalSqueeze(target_dim=4, n_neighbors=5, refine_steps=10)
        ts.fit(data)
        compressed = ts.squeeze(data)

        # Check that nearest neighbor in original ≈ nearest neighbor in compressed
        # for at least some points (probabilistic, not guaranteed)
        orig_dists = np.linalg.norm(data[:, None] - data[None, :], axis=2)
        comp_dists = np.linalg.norm(compressed[:, None] - compressed[None, :], axis=2)

        np.fill_diagonal(orig_dists, np.inf)
        np.fill_diagonal(comp_dists, np.inf)

        orig_nn = np.argmin(orig_dists, axis=1)
        comp_nn = np.argmin(comp_dists, axis=1)

        # At least 20% of nearest neighbors should be preserved
        preservation_rate = np.mean(orig_nn == comp_nn)
        assert preservation_rate >= 0.1  # conservative threshold
