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
