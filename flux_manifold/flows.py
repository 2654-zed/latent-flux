"""Flow functions for FluxManifold – define vector fields f(s, q) → delta.

All flow functions accept both single states s:(d,) and batch states s:(N,d).
When given batch input, they return (N,d) gradients computed in parallel via
numpy broadcasting — no Python loops.
"""

from __future__ import annotations

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    """L2 norm — returns scalar for 1D, (N,1) for 2D."""
    if v.ndim == 1:
        return np.linalg.norm(v)
    return np.linalg.norm(v, axis=1, keepdims=True)


def normalize_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Simple normalized difference: pull straight toward q."""
    diff = q - s
    norm = _norm(diff)
    if s.ndim == 1:
        if norm < 1e-12:
            return np.zeros_like(diff)
        return diff / norm
    return np.where(norm > 1e-12, diff / np.maximum(norm, 1e-12), 0.0)


def sin_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sin-modulated flow: adds curvature to the trajectory."""
    diff = q - s
    norm = _norm(diff)
    if s.ndim == 1:
        if norm < 1e-12:
            return np.zeros_like(diff)
        direction = diff / norm
        magnitude = np.sin(norm) / max(norm, 1e-6)
        return direction * magnitude
    direction = np.where(norm > 1e-12, diff / np.maximum(norm, 1e-12), 0.0)
    magnitude = np.sin(norm) / np.maximum(norm, 1e-6)
    return direction * magnitude


def damped_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Damped flow: strength proportional to distance, avoids overshoot."""
    diff = q - s
    norm = _norm(diff)
    if s.ndim == 1:
        if norm < 1e-12:
            return np.zeros_like(diff)
        damping = min(norm, 1.0)
        return (diff / norm) * damping
    damping = np.minimum(norm, 1.0)
    return np.where(norm > 1e-12, (diff / np.maximum(norm, 1e-12)) * damping, 0.0)


def adaptive_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Adaptive flow: reduces step as we near the attractor."""
    diff = q - s
    norm = _norm(diff)
    if s.ndim == 1:
        if norm < 1e-12:
            return np.zeros_like(diff)
        scale = min(1.0, norm)
        return (diff / norm) * scale
    scale = np.minimum(1.0, norm)
    return np.where(norm > 1e-12, (diff / np.maximum(norm, 1e-12)) * scale, 0.0)


def repulsive_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Adversarial repulsive flow – pushes away from q. For kill tests only."""
    diff = q - s
    norm = _norm(diff)
    if s.ndim == 1:
        if norm < 1e-12:
            return np.zeros_like(diff)
        return -(diff / norm)
    return np.where(norm > 1e-12, -(diff / np.maximum(norm, 1e-12)), 0.0)
