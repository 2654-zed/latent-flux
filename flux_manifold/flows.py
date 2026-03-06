"""Flow functions for FluxManifold – define vector fields f(s, q) → delta."""

from __future__ import annotations

import numpy as np


def normalize_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Simple normalized difference: pull straight toward q."""
    diff = q - s
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    return diff / norm


def sin_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sin-modulated flow: adds curvature to the trajectory."""
    diff = q - s
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    direction = diff / norm
    # Modulate magnitude with sin of distance (creates wave-like approach)
    magnitude = np.sin(norm) / max(norm, 1e-6)
    return direction * magnitude


def damped_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Damped flow: strength proportional to distance, avoids overshoot."""
    diff = q - s
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    # Damping factor: pull harder when far, softer when close
    damping = min(norm, 1.0)
    return (diff / norm) * damping


def adaptive_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Adaptive flow: reduces step as we near the attractor."""
    diff = q - s
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    # Scale: full unit when far, proportional when close
    scale = min(1.0, norm)
    return (diff / norm) * scale


def repulsive_flow(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Adversarial repulsive flow – pushes away from q. For kill tests only."""
    diff = q - s
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    return -(diff / norm)
