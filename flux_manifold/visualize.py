"""Visualization module for Latent Flux – path plots, convergence curves, TSP tours.

Uses matplotlib with Agg backend by default (no GUI needed), saves to files.
All plot functions return the Figure so callers can .savefig() or .show().
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Sequence


# ── Flow path visualization (2D) ───────────────────────────────────

def plot_flow_2d(
    path: list[np.ndarray],
    q: np.ndarray,
    title: str = "FluxManifold Flow (2D)",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a 2D flow path from s0 toward attractor q.

    Args:
        path: List of (2,) state vectors from flux_flow_traced.
        q: (2,) attractor.
        save_path: If given, save PNG to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = np.array(path)

    # Flow path
    ax.plot(pts[:, 0], pts[:, 1], "b-", alpha=0.6, linewidth=1.5, label="Flow path")
    ax.plot(pts[0, 0], pts[0, 1], "go", markersize=10, label="s₀ (start)", zorder=5)
    ax.plot(pts[-1, 0], pts[-1, 1], "rs", markersize=10, label="Converged", zorder=5)
    ax.plot(q[0], q[1], "k*", markersize=15, label="q (attractor)", zorder=5)

    # Step arrows (subsample if many)
    n = len(pts)
    step = max(1, n // 20)
    for i in range(0, n - 1, step):
        dx, dy = pts[i + 1] - pts[i]
        ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                     arrowprops=dict(arrowstyle="->", color="blue", alpha=0.3))

    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── Convergence curve ──────────────────────────────────────────────

def plot_convergence(
    drift_trace: list[float],
    tol: float = 1e-3,
    title: str = "Convergence (Drift over Steps)",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot drift-to-attractor over flow steps.

    Args:
        drift_trace: List of drift values from flux_flow_traced.
        tol: Convergence threshold line.
        save_path: If given, save PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = list(range(1, len(drift_trace) + 1))

    ax.semilogy(steps, drift_trace, "b-", linewidth=1.5, label="Drift ‖s − q‖")
    ax.axhline(y=tol, color="r", linestyle="--", alpha=0.7, label=f"tol = {tol}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Drift (log scale)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── Multi-flow comparison ─────────────────────────────────────────

def plot_convergence_comparison(
    traces: dict[str, list[float]],
    tol: float = 1e-3,
    title: str = "Flow Comparison – Convergence",
    save_path: str | Path | None = None,
) -> Figure:
    """Compare convergence of multiple flow functions or methods.

    Args:
        traces: Dict mapping label → drift_trace.
        tol: Convergence threshold line.
        save_path: If given, save PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (label, drift) in enumerate(traces.items()):
        steps = list(range(1, len(drift) + 1))
        ax.semilogy(steps, drift, color=colors[i % len(colors)],
                     linewidth=1.5, label=label)

    ax.axhline(y=tol, color="r", linestyle="--", alpha=0.7, label=f"tol = {tol}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drift (log scale)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── Superposition visualization ───────────────────────────────────

def plot_superposition_2d(
    states: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    title: str = "Superposition Tensor (∑_ψ)",
    save_path: str | Path | None = None,
) -> Figure:
    """Visualize superposition states in 2D, sized by weight.

    Args:
        states: (N, 2) candidate states.
        weights: (N,) weights.
        q: (2,) attractor.
        save_path: If given, save PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Candidates (sized by weight)
    sizes = weights / weights.max() * 200 + 20
    scatter = ax.scatter(states[:, 0], states[:, 1], s=sizes, c=weights,
                         cmap="viridis", alpha=0.7, edgecolors="k", linewidths=0.5,
                         label="Candidates")
    plt.colorbar(scatter, ax=ax, label="Weight")

    # Attractor
    ax.plot(q[0], q[1], "r*", markersize=15, label="q (attractor)", zorder=5)

    # Weighted mean
    mean = np.average(states, axis=0, weights=weights)
    ax.plot(mean[0], mean[1], "r^", markersize=12, label="Weighted mean", zorder=5)

    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── TSP tour visualization ────────────────────────────────────────

def plot_tsp_tour(
    cities: np.ndarray,
    order: np.ndarray | list[int],
    title: str = "TSP Tour",
    tour_length_val: float | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a TSP tour through 2D cities.

    Args:
        cities: (n, 2) city coordinates.
        order: Visit order (permutation).
        title: Plot title.
        tour_length_val: If given, display in title.
        save_path: If given, save PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    order = list(order)
    n = len(order)

    # Draw edges
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        ax.plot([cities[a, 0], cities[b, 0]], [cities[a, 1], cities[b, 1]],
                "b-", linewidth=1.5, alpha=0.6)

    # Draw cities
    ax.scatter(cities[:, 0], cities[:, 1], c="red", s=80, zorder=5, edgecolors="k")
    for i in range(len(cities)):
        ax.annotate(str(i), (cities[i, 0], cities[i, 1]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=9, fontweight="bold")

    # Start city
    start = order[0]
    ax.scatter([cities[start, 0]], [cities[start, 1]], c="green", s=150,
               zorder=6, edgecolors="k", marker="s", label=f"Start (city {start})")

    if tour_length_val is not None:
        title = f"{title} (length: {tour_length_val:.2f})"

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── TSP comparison (before / after) ───────────────────────────────

def plot_tsp_comparison(
    cities: np.ndarray,
    orders: dict[str, tuple | list | np.ndarray],
    lengths: dict[str, float] | None = None,
    title: str = "TSP Tour Comparison",
    save_path: str | Path | None = None,
) -> Figure:
    """Side-by-side comparison of multiple TSP tours.

    Args:
        cities: (n, 2) city coordinates.
        orders: Dict mapping label → visit order, or label → (order, length).
        lengths: Optional dict mapping label → tour length.
        save_path: If given, save PNG.
    """
    # Normalize: accept {label: (order, length)} or {label: order}
    _orders: dict[str, list[int]] = {}
    _lengths: dict[str, float] = dict(lengths) if lengths else {}
    for label, val in orders.items():
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], (int, float)):
            _orders[label] = list(val[0])
            _lengths.setdefault(label, float(val[1]))
        else:
            _orders[label] = list(val)

    n_plots = len(_orders)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    colors = ["blue", "green", "purple", "orange"]

    for idx, (label, order) in enumerate(_orders.items()):
        ax = axes[idx]
        order = list(order)
        n = len(order)
        c = colors[idx % len(colors)]

        for i in range(n):
            a, b = order[i], order[(i + 1) % n]
            ax.plot([cities[a, 0], cities[b, 0]], [cities[a, 1], cities[b, 1]],
                    f"{c[0]}-", linewidth=1.5, alpha=0.6)

        ax.scatter(cities[:, 0], cities[:, 1], c="red", s=60, zorder=5, edgecolors="k")
        for i in range(len(cities)):
            ax.annotate(str(i), (cities[i, 0], cities[i, 1]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)

        subtitle = label
        if _lengths and label in _lengths:
            subtitle += f" ({_lengths[label]:.2f})"
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── Abstraction cascade visualization ─────────────────────────────

def plot_cascade(
    levels: list,
    title: str = "Abstraction Cascade (⇑)",
    save_path: str | Path | None = None,
) -> Figure:
    """Visualize abstraction cascade levels as bar charts of state values.

    Args:
        levels: List of arrays (from cascade/cascade_single) or list of dicts.
        save_path: If given, save PNG.
    """
    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 4))
    if n_levels == 1:
        axes = [axes]

    for i, level in enumerate(levels):
        ax = axes[i]
        # Accept plain arrays or dicts
        if isinstance(level, dict):
            state = level.get("state", level.get("states"))
            dim_label = f"Level {level.get('level', i)} (d={level.get('dim', '?')})"
        elif isinstance(level, np.ndarray):
            state = level
            dim_label = f"Level {i} (d={level.shape[-1] if level.ndim > 0 else 0})"
        else:
            state = None
            dim_label = f"Level {i}"
        if state is not None:
            if state.ndim > 1:
                state = state[0]  # First state if batch
            ax.bar(range(len(state)), state, color="steelblue", alpha=0.7)
        ax.set_title(dim_label)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig


# ── Commitment sink timeline ──────────────────────────────────────

def plot_commitment_timeline(
    drift_trace: list[float],
    commit_step: int | None = None,
    entropy_trace: list[float] | None = None,
    title: str = "Commitment Sink (↓!) Timeline",
    save_path: str | Path | None = None,
) -> Figure:
    """Show drift + entropy over time, marking the commitment point.

    Args:
        drift_trace: Drift values per step.
        commit_step: Step where ↓! triggered. If None, not shown.
        entropy_trace: Optional entropy values per step.
        save_path: If given, save PNG.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    steps = list(range(1, len(drift_trace) + 1))

    ax1.semilogy(steps, drift_trace, "b-", linewidth=1.5, label="Drift")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Drift (log)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    if entropy_trace:
        ax2 = ax1.twinx()
        e_steps = list(range(1, len(entropy_trace) + 1))
        ax2.plot(e_steps, entropy_trace, "g--", linewidth=1.2, label="Entropy")
        ax2.set_ylabel("Entropy (bits)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

    if commit_step is not None:
        ax1.axvline(x=commit_step, color="red", linestyle="-.", linewidth=2,
                     label=f"↓! Commit @ step {commit_step}")

    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return fig
