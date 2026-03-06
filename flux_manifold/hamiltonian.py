"""FluxManifold Advanced Dynamical Systems — higher-order flows for escaping
geometric traps on semantic manifolds.

Implements five flow variants from the Latent Flux ontology:
  1. Riemannian Gradient Flow  (baseline, already in flows.py)
  2. Conformal Hamiltonian Flow (phase-space inertia + dissipation)
  3. Relativistic Hamiltonian Flow (bounded momentum)
  4. Lorentz-Finsler Flow (asymmetric metric, irreversible forward bias)
  5. Langevin Annealing Flow (SDE with thermal noise + cooling)

All flows operate on both single states (d,) and batch states (N,d)
via numpy vectorization — no Python loops over candidates.

Phase-space flows track (x, p) pairs where:
  x = position on the semantic manifold
  p = generalized conjugate momentum (inertia)

The HamiltonianFlowEngine wraps these into the FlowFn signature
expected by FluxManifold core, maintaining internal momentum state.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HamiltonianState:
    """Phase-space state (x, p) for Hamiltonian flows."""
    x: np.ndarray          # position (d,) or (N, d)
    p: np.ndarray          # momentum (d,) or (N, d)
    step: int = 0          # current integration step
    temperature: float = 1.0  # for Langevin annealing


def _batch_norm(v: np.ndarray) -> np.ndarray:
    """L2 norm — scalar for 1D, (N,1) for 2D."""
    if v.ndim == 1:
        return np.linalg.norm(v)
    return np.linalg.norm(v, axis=1, keepdims=True)


# ── Conformal Hamiltonian Flow ─────────────────────────────────────

def conformal_hamiltonian_step(
    x: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gamma: float = 0.1,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Single integration step of the conformal Hamiltonian system.

    ẋ = ∇_p H = p
    ṗ = -∇_x H - γp = -(x - q) - γp

    The Hamiltonian H(x,p) = ½|p|² + ½|x-q|² acts as Lyapunov function.
    Symplectic 2-form contracts as ω_t = e^{-γt}ω_0, guaranteeing
    convergence to the attractor while momentum carries past local minima.

    Args:
        x: position (d,) or (N, d)
        p: momentum (d,) or (N, d)
        q: attractor (d,)
        gamma: dissipation coefficient (>0)
        dt: integration timestep

    Returns:
        (x_new, p_new) updated phase-space state
    """
    # Potential gradient: ∇_x V = (x - q)
    grad_V = x - q

    # Symplectic leapfrog integration (Störmer-Verlet)
    # Half-step momentum with dissipation
    p_half = p - 0.5 * dt * (grad_V + gamma * p)

    # Full-step position
    x_new = x + dt * p_half

    # Recompute gradient at new position
    grad_V_new = x_new - q

    # Half-step momentum again
    p_new = p_half - 0.5 * dt * (grad_V_new + gamma * p_half)

    return x_new, p_new


# ── Relativistic Hamiltonian Flow ──────────────────────────────────

def relativistic_hamiltonian_step(
    x: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gamma: float = 0.1,
    c: float = 1.0,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Single step of dissipative relativistic Hamiltonian dynamics.

    Imposes a maximum speed c on the momentum to prevent trajectory
    explosion in highly curved regions. The relativistic velocity is:

        v = p / √(1 + |p|²/c²)

    This normalizes momentum intrinsically, bounding ||v|| < c.

    Args:
        x: position (d,) or (N, d)
        p: momentum (d,) or (N, d)
        q: attractor (d,)
        gamma: dissipation coefficient
        c: speed limit (stabilization parameter δ > 0)
        dt: integration timestep

    Returns:
        (x_new, p_new) updated phase-space state
    """
    # Relativistic velocity: v = p / sqrt(1 + |p|²/c²)
    p_norm_sq = np.sum(p * p, axis=-1, keepdims=True)
    lorentz_factor = np.sqrt(1.0 + p_norm_sq / (c * c))
    v = p / lorentz_factor

    # Potential gradient
    grad_V = x - q

    # Half-step momentum with dissipation
    p_half = p - 0.5 * dt * (grad_V + gamma * p)

    # Full-step position with relativistic velocity
    p_half_norm_sq = np.sum(p_half * p_half, axis=-1, keepdims=True)
    lorentz_half = np.sqrt(1.0 + p_half_norm_sq / (c * c))
    v_half = p_half / lorentz_half
    x_new = x + dt * v_half

    # Recompute gradient at new position
    grad_V_new = x_new - q

    # Half-step momentum
    p_new = p_half - 0.5 * dt * (grad_V_new + gamma * p_half)

    return x_new, p_new


# ── Lorentz-Finsler Flow ──────────────────────────────────────────

def finsler_asymmetric_step(
    x: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gamma: float = 0.1,
    asymmetry: float = 0.3,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Single step with Lorentz-Finsler asymmetric metric.

    The Finsler metric F_x: T_xX → ℝ⁺ incorporates directional asymmetry:
    forward motion (toward q) has lower cost than backward motion.

    The anisotropic factor modifies the effective gradient:
        grad_eff = grad_V * (1 - α·cos(θ))
    where θ is the angle between the gradient and the forward direction,
    and α ∈ [0,1) is the asymmetry parameter.

    This creates cone-like structures (analogous to light cones) that
    bias flow irreversibly forward, sealing exits from escaped traps.

    Args:
        x: position (d,) or (N, d)
        p: momentum (d,) or (N, d)
        q: attractor (d,)
        gamma: dissipation coefficient
        asymmetry: Finsler asymmetry parameter α ∈ [0, 1)
        dt: integration timestep

    Returns:
        (x_new, p_new) updated phase-space state
    """
    # Direction toward attractor
    forward = q - x
    forward_norm = _batch_norm(forward)
    if x.ndim == 1:
        forward_hat = forward / max(float(forward_norm), 1e-12)
    else:
        forward_hat = forward / np.maximum(forward_norm, 1e-12)

    # Potential gradient
    grad_V = x - q

    # Compute cosine similarity between -grad_V and forward direction
    # cos(θ) measures alignment of gradient with forward direction
    grad_norm = _batch_norm(grad_V)
    if x.ndim == 1:
        if float(grad_norm) < 1e-12:
            cos_theta = 0.0
        else:
            cos_theta = float(np.dot(-grad_V, forward_hat)) / max(float(grad_norm), 1e-12)
    else:
        cos_theta = np.sum(-grad_V * forward_hat, axis=-1, keepdims=True) / np.maximum(grad_norm, 1e-12)

    # Finsler anisotropic scaling: cheaper forward, expensive backward
    finsler_scale = 1.0 - asymmetry * np.clip(cos_theta, -1.0, 1.0)
    grad_V_eff = grad_V * finsler_scale

    # Hamiltonian integration with asymmetric gradient
    p_half = p - 0.5 * dt * (grad_V_eff + gamma * p)
    x_new = x + dt * p_half

    grad_V_new = x_new - q
    forward_new = q - x_new
    forward_norm_new = _batch_norm(forward_new)
    if x.ndim == 1:
        forward_hat_new = forward_new / max(float(forward_norm_new), 1e-12)
        grad_norm_new = _batch_norm(grad_V_new)
        if float(grad_norm_new) < 1e-12:
            cos_theta_new = 0.0
        else:
            cos_theta_new = float(np.dot(-grad_V_new, forward_hat_new)) / max(float(grad_norm_new), 1e-12)
    else:
        forward_hat_new = forward_new / np.maximum(forward_norm_new, 1e-12)
        grad_norm_new = _batch_norm(grad_V_new)
        cos_theta_new = np.sum(-grad_V_new * forward_hat_new, axis=-1, keepdims=True) / np.maximum(grad_norm_new, 1e-12)

    finsler_scale_new = 1.0 - asymmetry * np.clip(cos_theta_new, -1.0, 1.0)
    grad_V_new_eff = grad_V_new * finsler_scale_new

    p_new = p_half - 0.5 * dt * (grad_V_new_eff + gamma * p_half)

    return x_new, p_new


# ── Langevin Annealing Flow ───────────────────────────────────────

def langevin_annealing_step(
    x: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gamma: float = 0.1,
    dt: float = 0.05,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Single step of underdamped Langevin dynamics with annealing.

    Superimposes stochastic thermal noise over Hamiltonian flow:
        dp = -∇V(x)dt - γp·dt + √(2γT)·dW

    where T is the temperature and dW is a Wiener process increment.
    As T → 0 (annealing), the system transitions from global
    exploration to pure deterministic descent.

    The cooling schedule follows Eyring-Kramers law:
        T(t) = T_0 / (1 + t/τ)

    guaranteeing polynomial decay of escape probability from local minima.

    Regularized Langevin Dynamics (RLD) enforces constant expected
    step norm by normalizing the noise contribution.

    Args:
        x: position (d,) or (N, d)
        p: momentum (d,) or (N, d)
        q: attractor (d,)
        gamma: friction coefficient
        dt: integration timestep
        temperature: current annealing temperature T(t)
        rng: random number generator

    Returns:
        (x_new, p_new) updated phase-space state
    """
    if rng is None:
        rng = np.random.default_rng()

    # Potential gradient
    grad_V = x - q

    # Thermal noise: √(2γT) · dW
    noise_amplitude = np.sqrt(2.0 * gamma * max(temperature, 0.0) * dt)
    noise = noise_amplitude * rng.standard_normal(p.shape).astype(p.dtype)

    # RLD normalization: enforce constant expected step magnitude
    noise_norm = _batch_norm(noise)
    if x.ndim == 1:
        if float(noise_norm) > 1e-12:
            target_norm = noise_amplitude * np.sqrt(float(x.shape[-1]))
            noise = noise * (target_norm / float(noise_norm))
    else:
        d = x.shape[-1]
        target_norm = noise_amplitude * np.sqrt(d)
        noise = np.where(noise_norm > 1e-12,
                         noise * (target_norm / np.maximum(noise_norm, 1e-12)),
                         noise)

    # Underdamped Langevin: half-step momentum
    p_half = p - 0.5 * dt * (grad_V + gamma * p) + 0.5 * noise

    # Full-step position
    x_new = x + dt * p_half

    # Second half-step momentum
    grad_V_new = x_new - q
    p_new = p_half - 0.5 * dt * (grad_V_new + gamma * p_half) + 0.5 * noise

    return x_new, p_new


# ── Cooling Schedules ─────────────────────────────────────────────

def eyring_kramers_schedule(step: int, T0: float = 1.0, tau: float = 50.0) -> float:
    """Eyring-Kramers cooling: T(t) = T_0 / (1 + t/τ).

    Polynomial decay ensures the system explores globally before settling.
    """
    return T0 / (1.0 + step / tau)


def exponential_schedule(step: int, T0: float = 1.0, decay: float = 0.98) -> float:
    """Exponential cooling: T(t) = T_0 · decay^t."""
    return T0 * (decay ** step)


# ── Unified Hamiltonian Flow Engine ───────────────────────────────

class HamiltonianFlowEngine:
    """Unified engine wrapping all advanced flow variants.

    Maintains internal phase-space state (x, p) and provides a step()
    method that integrates the chosen dynamics. Can be used as a drop-in
    replacement for the standard FlowFn-based flow, or driven manually.

    Supported variants:
        "conformal"    — Conformal Hamiltonian (§2.1)
        "relativistic" — Relativistic Hamiltonian (§2.2)
        "finsler"      — Lorentz-Finsler asymmetric (§2.3)
        "langevin"     — Langevin annealing (§2.4)
        "adaptive"     — Auto-selects: Langevin → Finsler → Conformal
    """

    VARIANTS = ("conformal", "relativistic", "finsler", "langevin", "adaptive")

    def __init__(
        self,
        variant: str = "conformal",
        gamma: float = 0.1,
        dt: float = 0.05,
        c: float = 1.0,
        asymmetry: float = 0.3,
        T0: float = 1.0,
        cooling_tau: float = 50.0,
        seed: int = 42,
    ):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}")
        self.variant = variant
        self.gamma = gamma
        self.dt = dt
        self.c = c
        self.asymmetry = asymmetry
        self.T0 = T0
        self.cooling_tau = cooling_tau
        self.rng = np.random.default_rng(seed)
        self._step_count = 0

    def init_momentum(self, x: np.ndarray) -> np.ndarray:
        """Initialize zero momentum matching position shape."""
        return np.zeros_like(x)

    def step(
        self,
        x: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute one integration step of the chosen flow variant.

        Returns:
            (x_new, p_new) updated phase-space state
        """
        self._step_count += 1
        T = eyring_kramers_schedule(self._step_count, self.T0, self.cooling_tau)

        if self.variant == "conformal":
            return conformal_hamiltonian_step(x, p, q, self.gamma, self.dt)

        elif self.variant == "relativistic":
            return relativistic_hamiltonian_step(x, p, q, self.gamma, self.c, self.dt)

        elif self.variant == "finsler":
            return finsler_asymmetric_step(x, p, q, self.gamma, self.asymmetry, self.dt)

        elif self.variant == "langevin":
            return langevin_annealing_step(x, p, q, self.gamma, self.dt, T, self.rng)

        elif self.variant == "adaptive":
            return self._adaptive_step(x, p, q, T)

        raise ValueError(f"Unknown variant: {self.variant!r}")

    def _adaptive_step(
        self, x: np.ndarray, p: np.ndarray, q: np.ndarray, T: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive flow: auto-select dynamics based on temperature phase.

        Phase 1 (T > 0.5): Langevin exploration with thermal noise
        Phase 2 (0.1 < T ≤ 0.5): Finsler asymmetric descent
        Phase 3 (T ≤ 0.1): Conformal Hamiltonian final approach
        """
        if T > 0.5:
            return langevin_annealing_step(x, p, q, self.gamma, self.dt, T, self.rng)
        elif T > 0.1:
            return finsler_asymmetric_step(x, p, q, self.gamma, self.asymmetry, self.dt)
        else:
            return conformal_hamiltonian_step(x, p, q, self.gamma, self.dt)

    @property
    def temperature(self) -> float:
        """Current annealing temperature."""
        return eyring_kramers_schedule(self._step_count, self.T0, self.cooling_tau)

    def reset(self) -> None:
        """Reset step counter and temperature."""
        self._step_count = 0


def hamiltonian_flow(
    s0: np.ndarray,
    q: np.ndarray,
    variant: str = "conformal",
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_steps: int = 500,
    gamma: float = 0.1,
    dt: float = 0.05,
    c: float = 1.0,
    asymmetry: float = 0.3,
    T0: float = 1.0,
    cooling_tau: float = 50.0,
    seed: int = 42,
) -> dict:
    """Run a Hamiltonian flow from s0 toward attractor q.

    Returns dict with:
        converged_state, steps, converged, drift_trace, momentum_trace,
        temperature_trace, variant
    """
    engine = HamiltonianFlowEngine(
        variant=variant, gamma=gamma, dt=dt, c=c,
        asymmetry=asymmetry, T0=T0, cooling_tau=cooling_tau, seed=seed,
    )

    x = s0.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)
    p = engine.init_momentum(x)

    drift_trace: list[float] = []
    momentum_trace: list[float] = []
    temperature_trace: list[float] = []
    converged = False

    for step in range(max_steps):
        x, p = engine.step(x, p, q)

        # NaN/inf safety
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            p = np.zeros_like(p)

        drift = float(np.linalg.norm(x - q))
        p_mag = float(np.linalg.norm(p))
        drift_trace.append(drift)
        momentum_trace.append(p_mag)
        temperature_trace.append(engine.temperature)

        if drift < tol:
            converged = True
            break

    return {
        "converged_state": x,
        "steps": len(drift_trace),
        "converged": converged,
        "drift_trace": drift_trace,
        "momentum_trace": momentum_trace,
        "temperature_trace": temperature_trace,
        "variant": variant,
    }


def hamiltonian_flow_batch(
    S: np.ndarray,
    q: np.ndarray,
    variant: str = "conformal",
    tol: float = 1e-3,
    max_steps: int = 500,
    gamma: float = 0.1,
    dt: float = 0.05,
    c: float = 1.0,
    asymmetry: float = 0.3,
    T0: float = 1.0,
    cooling_tau: float = 50.0,
    seed: int = 42,
) -> dict:
    """Batch Hamiltonian flow for N states toward attractor q.

    Args:
        S: (N, d) initial states
        q: (d,) attractor

    Returns dict with:
        converged_states, steps, converged, drift_traces, variant
    """
    engine = HamiltonianFlowEngine(
        variant=variant, gamma=gamma, dt=dt, c=c,
        asymmetry=asymmetry, T0=T0, cooling_tau=cooling_tau, seed=seed,
    )

    S = S.astype(np.float32, copy=True)
    q = q.astype(np.float32, copy=False)
    N = S.shape[0]
    P = engine.init_momentum(S)

    active = np.ones(N, dtype=bool)
    step_counts = np.zeros(N, dtype=np.int32)
    converged_flags = np.zeros(N, dtype=bool)
    drift_history: list[np.ndarray] = []

    for iteration in range(max_steps):
        n_active = active.sum()
        if n_active == 0:
            break

        Sa = S[active]
        Pa = P[active]

        Sa_new, Pa_new = engine.step(Sa, Pa, q)

        # NaN/inf safety
        bad = np.any(np.isnan(Sa_new) | np.isinf(Sa_new), axis=1)
        Sa_new[bad] = Sa[bad]
        Pa_new[bad] = 0.0

        S[active] = Sa_new
        P[active] = Pa_new

        step_counts[active] += 1

        # Check convergence
        drifts = np.linalg.norm(S[active] - q, axis=1)
        converged_mask = drifts < tol
        active_idx = np.where(active)[0]
        active[active_idx[converged_mask]] = False
        converged_flags[active_idx[converged_mask]] = True

        # Record drift for all states
        all_drifts = np.full(N, np.nan)
        all_drifts[~converged_flags] = np.linalg.norm(
            S[~converged_flags] - q, axis=1
        ) if (~converged_flags).any() else np.array([])
        drift_history.append(all_drifts)

    return {
        "converged_states": S,
        "steps": step_counts,
        "converged": converged_flags,
        "total_steps": len(drift_history),
        "variant": variant,
    }
