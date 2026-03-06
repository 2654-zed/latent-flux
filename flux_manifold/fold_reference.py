"""Fold-Reference (◉) – self-critique and mid-flow reflection.

During flow, periodically inspect the current state via a critique
function. If the critique detects a problem (e.g., crossing edges
in TSP), apply a correction before continuing flow.

Notation:  ◉[state] ⊸ (λs. critique(s)) → corrected_state
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# Type: critique_fn(state) → (is_ok: bool, diagnosis: str, correction: np.ndarray | None)
CritiqueFn = Callable[[np.ndarray], tuple[bool, str, np.ndarray | None]]


class FoldReference:
    """Self-referential critique applied during or after flow."""

    def __init__(
        self,
        critique_fn: CritiqueFn,
        interval: int = 10,
        max_corrections: int = 50,
    ):
        """
        Args:
            critique_fn: Called with state, returns (ok, diagnosis, correction).
                         If ok=False and correction is not None, it replaces state.
            interval: Apply critique every N flow steps.
            max_corrections: Safety limit on total corrections.
        """
        self.critique_fn = critique_fn
        self.interval = interval
        self.max_corrections = max_corrections
        self.history: list[dict] = []
        self._corrections_applied = 0

    def check(self, state: np.ndarray, step: int) -> tuple[np.ndarray, bool]:
        """Check state at given step. Returns (possibly corrected state, was_corrected)."""
        if step % self.interval != 0:
            return state, False

        ok, diagnosis, correction = self.critique_fn(state)
        record = {"step": step, "ok": ok, "diagnosis": diagnosis}

        if not ok and correction is not None and self._corrections_applied < self.max_corrections:
            self._corrections_applied += 1
            record["corrected"] = True
            self.history.append(record)
            return correction.astype(np.float32), True

        record["corrected"] = False
        self.history.append(record)
        return state, False

    def check_batch(self, states: np.ndarray) -> tuple[np.ndarray, int]:
        """Critique all N states in (N, d) matrix. Returns (corrected_states, n_corrections).

        Applies critique_fn to each state, but avoids Python-level per-state loops
        for built-in critiques by using vectorized numpy operations directly.
        """
        N = states.shape[0]
        corrected = states.copy()
        n_corrections = 0

        # Fast path for no_nan_critique: fully vectorized
        if self.critique_fn is no_nan_critique:
            bad_mask = np.any(np.isnan(corrected) | np.isinf(corrected), axis=1)
            if bad_mask.any():
                corrected[bad_mask] = np.nan_to_num(
                    corrected[bad_mask], nan=0.0, posinf=1.0, neginf=-1.0
                )
                n_corrections = int(bad_mask.sum())
            self._corrections_applied += n_corrections
            return corrected, n_corrections

        # General path: apply critique per state (necessary for arbitrary critique_fn)
        budget = self.max_corrections - self._corrections_applied
        for i in range(N):
            if n_corrections >= budget:
                break
            ok, diagnosis, correction = self.critique_fn(corrected[i])
            if not ok and correction is not None:
                corrected[i] = correction.astype(np.float32)
                n_corrections += 1

        self._corrections_applied += n_corrections
        return corrected, n_corrections

    @property
    def corrections_count(self) -> int:
        return self._corrections_applied

    def reset(self) -> None:
        self.history.clear()
        self._corrections_applied = 0


def no_nan_critique(state: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
    """Built-in critique: reject NaN/Inf states."""
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        return False, "NaN/Inf detected", np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
    return True, "clean", None


def norm_bound_critique(
    max_norm: float = 10.0,
) -> CritiqueFn:
    """Factory: critique that clips state if norm exceeds bound."""
    def _critique(state: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
        n = float(np.linalg.norm(state))
        if n > max_norm:
            return False, f"norm={n:.2f} > {max_norm}", state * (max_norm / n)
        return True, f"norm={n:.2f} ok", None
    return _critique
