"""Quantum Interference Engine for the Superposition Tensor (∑_ψ).

Implements phase-based state suppression via four mechanisms:
  1. Phase-Based Destructive Interference — anti-aligned amplitudes cancel
  2. Non-Hermitian Dissipation — complex eigenvalue probability decay
  3. Exceptional Point Coalescence — degenerate eigenvector merging
  4. Friedrich-Wintgen BIC — bound state isolation with infinite Q-factor

Each execution path is represented as a complex amplitude:
    ψ_k = |a_k| · exp(iφ_k)
where |a_k| is geometric relevance and φ_k is semantic phase alignment.

Sub-optimal paths are organically annihilated mid-flight through
interference, while the optimal path is isolated as a topologically
protected bound state in the continuum.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class InterferenceResult:
    """Result of quantum interference filtering."""
    surviving_indices: np.ndarray     # indices of paths that survived
    amplitudes: np.ndarray            # complex amplitudes of survivors
    phases: np.ndarray                # phases of survivors
    quality_factors: np.ndarray       # Q-factors (higher = more stable)
    bic_index: int                    # index of the BIC-protected state
    n_annihilated: int                # number of paths destroyed


class QuantumInterferenceEngine:
    """Applies quantum-mechanical interference to filter superposition states.

    Converts real-valued superposition states into complex wave-functions
    in a rigged Hilbert space, applies non-Hermitian evolution, detects
    exceptional points, and isolates the optimal path via BIC formation.
    """

    def __init__(
        self,
        dissipation_rate: float = 0.3,
        phase_sensitivity: float = 2.0,
        interference_threshold: float = 0.1,
        ep_merge_threshold: float = 0.05,
        bic_coupling_strength: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            dissipation_rate: γ — controls non-Hermitian amplitude decay
            phase_sensitivity: scales phase assignment from drift gradient
            interference_threshold: amplitude below which paths are annihilated
            ep_merge_threshold: eigenvalue distance for EP coalescence
            bic_coupling_strength: g — coupling for FW-BIC condition
            seed: random seed
        """
        self.dissipation_rate = dissipation_rate
        self.phase_sensitivity = phase_sensitivity
        self.interference_threshold = interference_threshold
        self.ep_merge_threshold = ep_merge_threshold
        self.bic_coupling_strength = bic_coupling_strength
        self.rng = np.random.default_rng(seed)

    # ── Complex amplitude assignment ──────────────────────────

    def assign_amplitudes(
        self,
        states: np.ndarray,
        q: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Assign complex amplitudes ψ_k = |a_k|·exp(iφ_k) to each state.

        Amplitude |a_k| derives from weight (geometric relevance).
        Phase φ_k derives from semantic alignment with attractor q.

        Args:
            states: (N, d) superposition states
            q: (d,) target attractor
            weights: (N,) probability weights

        Returns:
            (N,) complex amplitudes
        """
        N = states.shape[0]

        # Amplitude from weights (relevance)
        magnitudes = np.sqrt(np.maximum(weights, 0.0))

        # Phase from alignment with attractor gradient
        diffs = states - q  # (N, d)
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.maximum(dists, 1e-12)

        # Normalize differences and compute phase via projection
        # Phase = angle of principal component projection onto attractor direction
        mean_diff = diffs.mean(axis=0)
        mean_norm = np.linalg.norm(mean_diff)
        if mean_norm > 1e-12:
            ref_dir = mean_diff / mean_norm
        else:
            ref_dir = np.zeros(states.shape[1], dtype=np.float32)
            ref_dir[0] = 1.0

        # Phase from dot product with reference direction
        projections = diffs @ ref_dir  # (N,)
        phases = self.phase_sensitivity * projections / dists

        # Complex amplitudes: ψ_k = |a_k| · exp(iφ_k)
        amplitudes = magnitudes * np.exp(1j * phases)

        return amplitudes.astype(np.complex64)

    # ── Phase-based destructive interference (§3.1) ───────────

    def destructive_interference(
        self,
        amplitudes: np.ndarray,
        states: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """Apply Pegg-Barnett-style phase interference.

        When two sub-optimal paths have anti-aligned phases (Δφ ≈ π),
        their amplitudes destructively subtract, reducing probability
        density of erroneous paths.

        Returns:
            (N,) updated complex amplitudes after interference
        """
        N = len(amplitudes)
        phases = np.angle(amplitudes)
        result = amplitudes.copy()

        # Compute pairwise phase differences
        # For efficiency, use vectorized outer difference
        phase_diffs = phases[:, None] - phases[None, :]  # (N, N)

        # Find anti-aligned pairs: |Δφ| ≈ π (within tolerance)
        anti_aligned = np.abs(np.abs(phase_diffs) - np.pi) < 0.5

        # Zero out diagonal (no self-interference)
        np.fill_diagonal(anti_aligned, False)

        # For each state, compute interference from anti-aligned partners
        for i in range(N):
            partners = np.where(anti_aligned[i])[0]
            if len(partners) > 0:
                # Destructive: subtract sum of anti-aligned amplitudes
                interference = np.sum(amplitudes[partners])
                result[i] = amplitudes[i] + interference * np.exp(1j * np.pi)

                # Amplitude cannot go negative magnitude — floor at 0
                if np.abs(result[i]) < self.interference_threshold:
                    result[i] = 0.0

        return result

    # ── Non-Hermitian dissipation (§3.2) ──────────────────────

    def non_hermitian_evolution(
        self,
        amplitudes: np.ndarray,
        states: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        r"""Apply non-Hermitian Hamiltonian evolution.

        Constructs the effective non-Hermitian Hamiltonian H = Ω - iΓ
        where Ω is the energy detuning and Γ encodes engineered dissipation.

        Complex eigenvalues ω± = ω₀ - i(γ_J + γ_P)/2 ± √(g² - (Δγ/2)²)
        produce exponential decay for sub-optimal paths.

        Returns:
            (N,) amplitudes after non-Hermitian evolution step
        """
        N = len(amplitudes)

        # Energy from distance to attractor (detuning)
        dists = np.linalg.norm(states - q, axis=1)
        max_dist = np.max(dists) + 1e-12

        # Normalized energy: close to q → low energy, far → high
        energies = dists / max_dist

        # Dissipation rate proportional to energy (distance from optimal)
        # Low-energy (close) paths lose little; high-energy paths lose much
        gamma_eff = self.dissipation_rate * energies  # (N,)

        # Non-Hermitian evolution: amplitude decays as exp(-γ_eff)
        decay = np.exp(-gamma_eff)
        evolved = amplitudes * decay

        return evolved

    # ── Exceptional Point coalescence (§3.2) ──────────────────

    def exceptional_point_merge(
        self,
        amplitudes: np.ndarray,
        states: np.ndarray,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and merge states near Exceptional Points.

        At an EP, both eigenvalues and eigenvectors of the non-Hermitian
        Hamiltonian coalesce. Similar sub-optimal states are forced to
        merge and then self-annihilate through perfect interference.

        Returns:
            (amplitudes, states, merge_mask) — mask indicates surviving states
        """
        N = len(amplitudes)

        # Compute pairwise state distances
        diffs = states[:, None, :] - states[None, :, :]  # (N, N, d)
        pairwise_dists = np.linalg.norm(diffs, axis=2)  # (N, N)

        # Find EP candidates: pairs with similar eigenvalues (distances to q)
        dists_to_q = np.linalg.norm(states - q, axis=1)
        energy_diffs = np.abs(dists_to_q[:, None] - dists_to_q[None, :])

        # EP condition: similar energy AND similar state
        ep_mask = (
            (energy_diffs < self.ep_merge_threshold)
            & (pairwise_dists < self.ep_merge_threshold * states.shape[1])
        )
        np.fill_diagonal(ep_mask, False)

        # Merge EP pairs: annihilate the weaker partner
        merge_mask = np.ones(N, dtype=bool)
        merged = set()
        for i in range(N):
            if i in merged:
                continue
            partners = np.where(ep_mask[i] & merge_mask)[0]
            for j in partners:
                if j in merged or j <= i:
                    continue
                # Merge: stronger amplitude survives, weaker annihilated
                if np.abs(amplitudes[i]) >= np.abs(amplitudes[j]):
                    # i absorbs j's amplitude (destructive interference)
                    phase_diff = np.angle(amplitudes[j]) - np.angle(amplitudes[i])
                    amplitudes[i] += amplitudes[j] * np.exp(1j * np.pi)
                    merge_mask[j] = False
                    merged.add(j)
                else:
                    amplitudes[j] += amplitudes[i] * np.exp(1j * np.pi)
                    merge_mask[i] = False
                    merged.add(i)
                    break

        return amplitudes[merge_mask], states[merge_mask], merge_mask

    # ── Friedrich-Wintgen BIC isolation (§3.3) ────────────────

    def bic_isolation(
        self,
        amplitudes: np.ndarray,
        states: np.ndarray,
        q: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Identify and isolate the Friedrich-Wintgen Bound State.

        The BIC is the state whose radiation channels destructively
        interfere, achieving B_out = 0 (zero outgoing wave). This
        state has theoretically infinite Q-factor and is immune to
        the surrounding dissipation.

        The Q-factor for each state is computed as:
            Q_k = |ψ_k|² / Σ_{j≠k} |coupling(k,j)|²

        The state with the highest Q-factor is the BIC.

        Returns:
            (bic_index, quality_factors) — index of BIC state and all Q-factors
        """
        N = len(amplitudes)
        if N == 0:
            return 0, np.array([])

        # Compute coupling matrix: off-diagonal radiation coupling
        # Coupling strength ~ inverse distance (near states couple more)
        diffs = states[:, None, :] - states[None, :, :]
        pairwise_dists = np.linalg.norm(diffs, axis=2) + 1e-12
        coupling_matrix = self.bic_coupling_strength / pairwise_dists

        np.fill_diagonal(coupling_matrix, 0.0)

        # For FW-BIC: the state whose radiation channels cancel
        # Q_k = |ψ_k|² / total_radiation_loss
        prob_density = np.abs(amplitudes) ** 2
        radiation_loss = np.sum(coupling_matrix ** 2, axis=1)

        # Q-factor: high probability, low radiation = BIC candidate
        quality_factors = prob_density / np.maximum(radiation_loss, 1e-12)

        # BIC is the state with maximum Q-factor
        bic_index = int(np.argmax(quality_factors))

        return bic_index, quality_factors

    # ── Full interference pipeline ────────────────────────────

    def filter(
        self,
        states: np.ndarray,
        q: np.ndarray,
        weights: np.ndarray,
    ) -> InterferenceResult:
        """Run the full quantum interference pipeline.

        1. Assign complex amplitudes (§3.1)
        2. Apply destructive interference (§3.1)
        3. Non-Hermitian dissipation (§3.2)
        4. Exceptional Point coalescence (§3.2)
        5. Friedrich-Wintgen BIC isolation (§3.3)

        Returns:
            InterferenceResult with surviving states and BIC identification
        """
        N_original = len(states)

        # Step 1: Assign complex amplitudes
        amplitudes = self.assign_amplitudes(states, q, weights)

        # Step 2: Phase-based destructive interference
        amplitudes = self.destructive_interference(amplitudes, states, q)

        # Step 3: Non-Hermitian dissipation
        amplitudes = self.non_hermitian_evolution(amplitudes, states, q)

        # Filter annihilated states (amplitude below threshold)
        alive = np.abs(amplitudes) > self.interference_threshold
        amplitudes = amplitudes[alive]
        states = states[alive]
        surviving_indices = np.where(alive)[0]

        # Step 4: Exceptional Point coalescence
        if len(amplitudes) > 1:
            amplitudes, states, ep_mask = self.exceptional_point_merge(
                amplitudes, states, q
            )
            surviving_indices = surviving_indices[ep_mask]

        # Step 5: BIC isolation
        bic_local, quality_factors = self.bic_isolation(amplitudes, states, q)

        # Map BIC index back to original state index
        bic_index = int(surviving_indices[bic_local]) if len(surviving_indices) > 0 else 0

        return InterferenceResult(
            surviving_indices=surviving_indices,
            amplitudes=amplitudes,
            phases=np.angle(amplitudes),
            quality_factors=quality_factors,
            bic_index=bic_index,
            n_annihilated=N_original - len(surviving_indices),
        )

    def collapse_to_bic(
        self,
        states: np.ndarray,
        q: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, InterferenceResult]:
        """Filter states and return the BIC-isolated optimal path.

        This is the primary entry point: applies full interference
        pipeline and returns the single topologically protected state.

        Returns:
            (bic_state, interference_result)
        """
        result = self.filter(states, q, weights)

        if len(result.surviving_indices) == 0:
            # All paths annihilated — fall back to closest to q
            dists = np.linalg.norm(states - q, axis=1)
            best = int(np.argmin(dists))
            return states[best], result

        # Return the BIC state
        bic_state = states[result.surviving_indices == result.bic_index]
        if len(bic_state) == 0:
            bic_state = states[result.surviving_indices[0]]
        else:
            bic_state = bic_state[0]

        return bic_state, result
