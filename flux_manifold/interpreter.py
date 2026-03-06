"""Latent Flux Interpreter – full pipeline wiring all 7 primitives.

Pipeline for a single evaluation:

  1. ∇↓  Dimensional Squeeze   – compress input if high-d
  2. ∑_ψ Superposition Tensor  – create N candidate states
  3. ⟼   FluxManifold          – flow each candidate toward attractor q
  4. ◉   Fold-Reference        – self-critique mid/post-flow
  5. ≅   DriftEquivalence      – accept "close enough" solutions
  6. ↓!  Commitment Sink       – irreversible collapse when confident
  7. ⇑   Abstraction Cascade   – build hierarchical view of result

Usage:
    interpreter = LatentFluxInterpreter(d=64, n_candidates=16)
    result = interpreter.evaluate(q=target_attractor)
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from flux_manifold.core import FlowFn, flux_flow_traced
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.drift_equivalence import DriftEquivalence
from flux_manifold.commitment_sink import CommitmentSink
from flux_manifold.abstraction_cascade import AbstractionCascade
from flux_manifold.fold_reference import FoldReference, CritiqueFn, no_nan_critique
from flux_manifold.dimensional_squeeze import DimensionalSqueeze
from flux_manifold.flows import normalize_flow


class LatentFluxInterpreter:
    """Full Latent Flux interpreter orchestrating all 7 primitives."""

    def __init__(
        self,
        d: int = 64,
        n_candidates: int = 16,
        flow_fn: FlowFn = normalize_flow,
        epsilon: float = 0.1,
        tol: float = 1e-3,
        max_steps: int = 500,
        equiv_tolerance: float = 0.05,
        entropy_threshold: float = 0.5,
        drift_window: int = 10,
        drift_commit_threshold: float = 0.01,
        cascade_levels: int = 3,
        critique_fn: CritiqueFn | None = None,
        critique_interval: int = 10,
        squeeze_dim: int | None = None,
        seed: int = 42,
    ):
        self.d = d
        self.n_candidates = n_candidates
        self.flow_fn = flow_fn
        self.epsilon = epsilon
        self.tol = tol
        self.max_steps = max_steps
        self.seed = seed

        # Primitives
        self.equivalence = DriftEquivalence(tolerance=equiv_tolerance)
        self.commitment = CommitmentSink(
            entropy_threshold=entropy_threshold,
            drift_window=drift_window,
            drift_threshold=drift_commit_threshold,
        )
        self.cascade = AbstractionCascade(levels=cascade_levels)
        self.fold_ref = FoldReference(
            critique_fn=critique_fn or no_nan_critique,
            interval=critique_interval,
        )
        self.squeeze: DimensionalSqueeze | None = None
        if squeeze_dim is not None and squeeze_dim < d:
            self.squeeze = DimensionalSqueeze(target_dim=squeeze_dim)

        # State
        self._superposition: SuperpositionTensor | None = None
        self._trace_log: list[dict] = []

    def evaluate(
        self,
        q: np.ndarray,
        initial_states: np.ndarray | None = None,
    ) -> dict:
        """Run the full Latent Flux pipeline.

        Args:
            q: Target attractor (d,).
            initial_states: Optional (N, d) initial candidates.
                            If None, generates random candidates.

        Returns dict with:
            committed_state, equivalence_quality, abstraction_levels,
            total_steps, fold_corrections, converged_count, superposition_entropy
        """
        working_d = self.d
        q_work = q.astype(np.float32).copy()

        # ── Step 1: ∇↓ Dimensional Squeeze ──────────────────────
        if self.squeeze is not None:
            if initial_states is not None:
                self.squeeze.fit(initial_states, seed=self.seed)
            else:
                # Fit on random data to establish projection
                rng = np.random.default_rng(self.seed)
                dummy = rng.standard_normal((max(self.n_candidates, 10), self.d)).astype(np.float32)
                self.squeeze.fit(dummy, seed=self.seed)
            q_work = self.squeeze.squeeze(q_work)
            working_d = q_work.shape[0]
            if initial_states is not None:
                initial_states = self.squeeze.squeeze(initial_states)

        # ── Step 2: ∑_ψ Superposition Tensor ────────────────────
        if initial_states is not None:
            if initial_states.ndim == 1:
                initial_states = initial_states.reshape(1, -1)
            self._superposition = SuperpositionTensor(initial_states)
        else:
            self._superposition = SuperpositionTensor.from_random(
                self.n_candidates, working_d, seed=self.seed
            )

        # ── Step 3: ⟼ FluxManifold flow (all candidates, batch) ─
        batch_trace = self._superposition.flow_all(
            q_work, self.flow_fn,
            epsilon=self.epsilon, tol=self.tol, max_steps=self.max_steps,
        )

        total_steps = int(batch_trace["total_steps"])
        converged_count = int(batch_trace["converged"].sum())

        # Aggregate drift trace (from best candidate)
        best_idx = int(np.argmin([
            np.linalg.norm(self._superposition.states[i] - q_work)
            for i in range(self._superposition.n)
        ]))
        drift_matrix = batch_trace["drift_traces"]  # (iters, N)
        combined_drift = []
        if drift_matrix.size > 0:
            col = drift_matrix[:, best_idx]
            combined_drift = col[~np.isnan(col)].tolist()

        # ── Step 4: ◉ Fold-Reference (post-flow critique) ──────
        fold_corrections = 0
        for i in range(self._superposition.n):
            corrected, was_corrected = self.fold_ref.check(
                self._superposition.states[i], step=i
            )
            if was_corrected:
                self._superposition.states[i] = corrected
                fold_corrections += 1

        # ── Step 5: ≅ DriftEquivalence (reweight by quality) ────
        self._superposition.reweight_by_drift(q_work)
        entropy_after = self._superposition.entropy()

        # ── Step 6: ↓! Commitment Sink ──────────────────────────
        committed = self.commitment.try_commit(
            self._superposition, q_work, combined_drift
        )
        if committed is None:
            # Force commit to best if auto-commit didn't trigger
            committed = self._superposition.collapse_to_best(q_work)
            self.commitment.commit(committed, reason="forced_best")

        # Un-squeeze if needed
        if self.squeeze is not None:
            committed = self.squeeze.unsqueeze(committed)

        # Quality score
        q_final = q if self.squeeze is None else q
        eq_quality = self.equivalence.quality(committed, q.astype(np.float32))
        eq_equivalent = self.equivalence.equivalent(committed, q.astype(np.float32))

        # ── Step 7: ⇑ Abstraction Cascade ───────────────────────
        levels = self.cascade.cascade_single(committed)

        return {
            "committed_state": committed,
            "equivalence_quality": eq_quality,
            "is_equivalent": eq_equivalent,
            "abstraction_levels": levels,
            "total_steps": total_steps,
            "converged_count": converged_count,
            "n_candidates": self._superposition.n,
            "fold_corrections": fold_corrections,
            "superposition_entropy": entropy_after,
            "commit_reason": self.commitment.commit_reason,
        }
