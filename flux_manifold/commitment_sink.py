"""Commitment Sink (↓!) – irreversible collapse when confidence is high.

Monitors flow entropy / drift history. When the system is confident
(low entropy, stable drift), it commits irrevocably — destroying
alternative branches and locking in the result.

Notation:  ↓! state  (collapses superposition, no going back)
"""

from __future__ import annotations

import numpy as np

from flux_manifold.superposition import SuperpositionTensor


class CommitmentSink:
    """Monitors flow and triggers irreversible commitment."""

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        drift_window: int = 10,
        drift_threshold: float = 0.01,
    ):
        """
        Args:
            entropy_threshold: Commit when superposition entropy drops below this.
            drift_window: Number of recent drift values to average.
            drift_threshold: Commit when average recent drift < this.
        """
        self.entropy_threshold = entropy_threshold
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.committed = False
        self.committed_state: np.ndarray | None = None
        self.commit_reason: str = ""

    def should_commit_entropy(self, superposition: SuperpositionTensor) -> bool:
        """Check if superposition entropy is low enough to commit."""
        return superposition.entropy() < self.entropy_threshold

    def should_commit_drift(self, drift_trace: list[float]) -> bool:
        """Check if recent drift is stable enough to commit."""
        if len(drift_trace) < self.drift_window:
            return False
        recent = drift_trace[-self.drift_window:]
        return float(np.mean(recent)) < self.drift_threshold

    def commit(self, state: np.ndarray, reason: str = "manual") -> np.ndarray:
        """Irreversibly commit to a state. No going back."""
        if self.committed:
            raise RuntimeError("Already committed — ↓! is irreversible")
        self.committed = True
        self.committed_state = state.copy()
        self.commit_reason = reason
        return self.committed_state

    def try_commit(
        self,
        superposition: SuperpositionTensor,
        q: np.ndarray,
        drift_trace: list[float] | None = None,
    ) -> np.ndarray | None:
        """Try to commit based on entropy or drift. Returns state if committed, else None."""
        if self.committed:
            return self.committed_state

        if self.should_commit_entropy(superposition):
            best = superposition.collapse_to_best(q)
            return self.commit(best, reason="entropy_low")

        if drift_trace and self.should_commit_drift(drift_trace):
            best = superposition.collapse_to_best(q)
            return self.commit(best, reason="drift_stable")

        return None

    @property
    def state(self) -> np.ndarray | None:
        """The committed state, or None if not yet committed."""
        return self.committed_state
