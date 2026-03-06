"""Tests for the visualization module – verify plots generate without errors."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from flux_manifold.visualize import (
    plot_flow_2d,
    plot_convergence,
    plot_convergence_comparison,
    plot_superposition_2d,
    plot_tsp_tour,
    plot_tsp_comparison,
    plot_cascade,
    plot_commitment_timeline,
)
from matplotlib.figure import Figure


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestPlotFlow2D:
    def test_returns_figure(self):
        path = [np.array([1.0, 1.0]), np.array([0.5, 0.5]), np.array([0.0, 0.0])]
        q = np.array([0.0, 0.0])
        fig = plot_flow_2d(path, q)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_dir):
        path = [np.array([1.0, 1.0]), np.array([0.0, 0.0])]
        q = np.array([0.0, 0.0])
        out = tmp_dir / "flow.png"
        fig = plot_flow_2d(path, q, save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotConvergence:
    def test_returns_figure(self):
        drift = [1.0, 0.5, 0.25, 0.1, 0.05]
        fig = plot_convergence(drift, tol=0.1)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_dir):
        drift = [1.0, 0.5, 0.1]
        out = tmp_dir / "conv.png"
        fig = plot_convergence(drift, save_path=str(out))
        assert out.exists()


class TestPlotConvergenceComparison:
    def test_returns_figure(self):
        traces = {
            "norm": [1.0, 0.5, 0.1],
            "sin": [1.0, 0.8, 0.3, 0.1],
        }
        fig = plot_convergence_comparison(traces)
        assert isinstance(fig, Figure)


class TestPlotSuperposition2D:
    def test_returns_figure(self):
        states = np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
        weights = np.array([0.5, 0.3, 0.2])
        q = np.array([0, 0])
        fig = plot_superposition_2d(states, weights, q)
        assert isinstance(fig, Figure)


class TestPlotTSPTour:
    def test_returns_figure(self):
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        order = [0, 1, 2, 3]
        fig = plot_tsp_tour(cities, order)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_dir):
        cities = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        order = [0, 1, 2]
        out = tmp_dir / "tsp.png"
        fig = plot_tsp_tour(cities, order, save_path=str(out))
        assert out.exists()


class TestPlotTSPComparison:
    def test_returns_figure(self):
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        orders = {
            "Flux": ([0, 1, 2, 3], 4.0),
            "Random": ([3, 1, 0, 2], 5.2),
        }
        fig = plot_tsp_comparison(cities, orders)
        assert isinstance(fig, Figure)


class TestPlotCascade:
    def test_returns_figure(self):
        levels = [
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            np.array([[1, 2], [5, 6]]),
            np.array([[1.5], [5.5]]),
        ]
        fig = plot_cascade(levels)
        assert isinstance(fig, Figure)


class TestPlotCommitmentTimeline:
    def test_returns_figure(self):
        drift = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
        fig = plot_commitment_timeline(drift, commit_step=4)
        assert isinstance(fig, Figure)

    def test_with_entropy(self):
        drift = [1.0, 0.5, 0.1]
        entropy = [2.0, 1.5, 0.8]
        fig = plot_commitment_timeline(drift, commit_step=2, entropy_trace=entropy)
        assert isinstance(fig, Figure)
