"""Convergence Contracts — three-tier convergence classification for FluxManifold.

The halting problem for continuous dynamical systems is unresolved.
Every FluxManifold invocation must declare its convergence tier:

  Tier 1 — Provable convergence: convex manifolds with Lipschitz-continuous
            flow functions. Guaranteed to converge in finite steps.

  Tier 2 — Empirically bounded convergence: non-convex manifolds where
            convergence is observed but not provable. FoldReference monitors
            and corrects. max_steps acts as a computational timeout.
            This is the DEFAULT tier.

  Tier 3 — Acknowledged non-convergence: chaotic or adversarial manifolds
            where the system explicitly signals failure rather than
            silently diverging.

Unknown tier defaults to Tier 2.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ConvergenceTier(enum.IntEnum):
    """Three-tier convergence classification."""
    PROVABLE = 1       # Convex + Lipschitz → guaranteed convergence
    EMPIRICAL = 2      # Observed convergence, not provable → monitored
    NON_CONVERGENT = 3 # Chaotic / adversarial → explicit failure signal


@dataclass(frozen=True)
class ConvergenceContract:
    """A convergence contract attached to a flow invocation.

    Attributes:
        tier: The convergence tier (1, 2, or 3).
        justification: Why this tier was declared.
        lipschitz_bound: For Tier 1 only — the Lipschitz constant L of the
                         flow function. If L < 1 the flow is a contraction
                         mapping and convergence is provable by Banach.
        max_steps_binding: For Tier 2 — whether max_steps is a hard timeout
                           (True) or a soft suggestion (False).
    """
    tier: ConvergenceTier
    justification: str = ""
    lipschitz_bound: float | None = None
    max_steps_binding: bool = True

    def __post_init__(self) -> None:
        if self.tier == ConvergenceTier.PROVABLE and self.lipschitz_bound is None:
            raise ValueError(
                "Tier 1 (PROVABLE) requires a Lipschitz bound. "
                "If you cannot provide one, use Tier 2 (EMPIRICAL)."
            )
        if (self.lipschitz_bound is not None
                and self.tier == ConvergenceTier.PROVABLE
                and self.lipschitz_bound >= 1.0):
            raise ValueError(
                f"Tier 1 requires Lipschitz bound < 1.0 for contraction "
                f"mapping guarantee, got L={self.lipschitz_bound}. "
                f"Use Tier 2 if convergence is empirical only."
            )


@dataclass
class ConvergenceResult:
    """Outcome of a flow invocation, annotated with its contract.

    The contract is declared BEFORE flow. The result is populated AFTER.
    """
    contract: ConvergenceContract
    converged: bool
    steps_used: int
    max_steps: int
    final_drift: float
    tier_honored: bool = True
    failure_signal: str = ""

    @property
    def tier(self) -> ConvergenceTier:
        return self.contract.tier

    def check(self) -> None:
        """Validate the result against its contract.

        Tier 1: convergence MUST have occurred. If not, the contract was wrong.
        Tier 2: non-convergence is recorded, not an error.
        Tier 3: non-convergence is EXPECTED. Convergence is a bonus.
        """
        if self.contract.tier == ConvergenceTier.PROVABLE and not self.converged:
            self.tier_honored = False
            self.failure_signal = (
                f"TIER 1 CONTRACT VIOLATION: flow declared PROVABLE convergence "
                f"but did not converge in {self.steps_used} steps "
                f"(final drift={self.final_drift:.6f}). "
                f"Lipschitz bound L={self.contract.lipschitz_bound} may be wrong."
            )
        elif self.contract.tier == ConvergenceTier.NON_CONVERGENT and not self.converged:
            self.failure_signal = (
                f"Tier 3: acknowledged non-convergence after {self.steps_used} steps "
                f"(final drift={self.final_drift:.6f}). This is expected behavior."
            )


# ── Convenience constructors ────────────────────────────────────────

# Pre-built contracts for the built-in flow functions

TIER_1_NORMALIZE = ConvergenceContract(
    tier=ConvergenceTier.PROVABLE,
    justification=(
        "normalize_flow computes (q-s)/||q-s||, a contraction mapping with "
        "effective Lipschitz constant L = epsilon < 1.0 when epsilon ∈ (0,1). "
        "On a convex domain with a single attractor, convergence is guaranteed."
    ),
    lipschitz_bound=0.1,  # matches default epsilon
)

TIER_1_DAMPED = ConvergenceContract(
    tier=ConvergenceTier.PROVABLE,
    justification=(
        "damped_flow clips magnitude to min(||q-s||, 1.0), making it a "
        "contraction mapping with L = epsilon on convex domains."
    ),
    lipschitz_bound=0.1,
)

TIER_2_DEFAULT = ConvergenceContract(
    tier=ConvergenceTier.EMPIRICAL,
    justification=(
        "Default tier for flows without provable convergence guarantees. "
        "FoldReference monitors for divergence. max_steps is a hard timeout."
    ),
    max_steps_binding=True,
)

TIER_2_SIN = ConvergenceContract(
    tier=ConvergenceTier.EMPIRICAL,
    justification=(
        "sin_flow adds curvature via sin(||q-s||), which is non-monotonic. "
        "Convergence is empirically observed but the non-monotonic magnitude "
        "means it cannot be a strict contraction mapping."
    ),
)

TIER_2_ADAPTIVE = ConvergenceContract(
    tier=ConvergenceTier.EMPIRICAL,
    justification=(
        "adaptive_flow scales steps by distance, empirically convergent "
        "but not provably contractive on non-convex domains."
    ),
)

TIER_3_REPULSIVE = ConvergenceContract(
    tier=ConvergenceTier.NON_CONVERGENT,
    justification=(
        "repulsive_flow pushes AWAY from the attractor. Convergence is "
        "impossible by design. Used for adversarial testing only."
    ),
)


# ── Flow function → default contract mapping ────────────────────────

def default_contract_for(flow_fn_name: str) -> ConvergenceContract:
    """Return the default convergence contract for a named flow function.

    Unknown flow functions default to Tier 2 (EMPIRICAL).
    """
    _DEFAULTS = {
        'normalize_flow': TIER_1_NORMALIZE,
        'damped_flow': TIER_1_DAMPED,
        'sin_flow': TIER_2_SIN,
        'adaptive_flow': TIER_2_ADAPTIVE,
        'repulsive_flow': TIER_3_REPULSIVE,
    }
    return _DEFAULTS.get(flow_fn_name, TIER_2_DEFAULT)
