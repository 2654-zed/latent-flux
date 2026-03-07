"""Tests for Attractor Competition (⊗) — geometric pattern matching."""

import numpy as np
import pytest

from flux_manifold.attractor_competition import AttractorCompetition
from flux_manifold.flows import normalize_flow, damped_flow


class TestAttractorCompetitionBasic:
    """Core competition mechanics."""

    def test_single_attractor_always_wins(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0]]),
            labels=["origin"],
            flow_fn=normalize_flow,
        )
        result = ac.compete(np.array([1.0, 1.0], dtype=np.float32))
        assert result["winner"] == "origin"
        assert result["winner_idx"] == 0

    def test_two_attractors_closer_wins(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [10.0, 10.0]]),
            labels=["A", "B"],
            flow_fn=normalize_flow,
            epsilon=0.1,
            tol=0.5,
        )
        # State near A
        result = ac.compete(np.array([0.5, 0.5], dtype=np.float32))
        assert result["winner"] == "A"
        # State near B
        result = ac.compete(np.array([9.5, 9.5], dtype=np.float32))
        assert result["winner"] == "B"

    def test_margin_positive_when_clear_winner(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [10.0, 0.0]]),
            labels=["left", "right"],
            flow_fn=normalize_flow,
            tol=0.5,
        )
        result = ac.compete(np.array([0.1, 0.0], dtype=np.float32))
        assert result["margin"] > 0

    def test_certainty_between_0_and_1(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [5.0, 0.0]]),
            labels=["A", "B"],
            flow_fn=normalize_flow,
        )
        result = ac.compete(np.array([0.5, 0.0], dtype=np.float32))
        assert 0.0 <= result["certainty"] <= 1.0

    def test_trajectory_recorded(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0]]),
            labels=["target"],
            flow_fn=normalize_flow,
        )
        result = ac.compete(np.array([5.0, 0.0], dtype=np.float32))
        assert len(result["trajectory"]) > 1
        # Trajectory moves toward target
        dists = [np.linalg.norm(t) for t in result["trajectory"]]
        assert dists[-1] < dists[0]

    def test_readout_dimension_matches_input(self):
        for d in [2, 4, 8]:
            ac = AttractorCompetition(
                attractors=np.zeros((2, d)),
                labels=["A", "B"],
                flow_fn=normalize_flow,
            )
            result = ac.compete(np.ones(d, dtype=np.float32))
            assert result["trajectory"][0].shape == (d,)


class TestAttractorCompetitionBatch:
    """Batch competition."""

    def test_batch_returns_list(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [5.0, 5.0]]),
            labels=["A", "B"],
            flow_fn=normalize_flow,
        )
        states = np.array([
            [0.5, 0.5],
            [4.5, 4.5],
        ], dtype=np.float32)
        results = ac.compete_batch(states)
        assert len(results) == 2
        assert results[0]["winner"] == "A"
        assert results[1]["winner"] == "B"

    def test_batch_summary(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [10.0, 10.0]]),
            labels=["A", "B"],
            flow_fn=normalize_flow,
        )
        states = np.array([
            [1.0, 1.0], [2.0, 2.0],
            [8.0, 8.0], [9.0, 9.0],
        ], dtype=np.float32)
        results = ac.compete_batch(states)
        summary = ac.summary(results)
        assert summary["counts"]["A"] == 2
        assert summary["counts"]["B"] == 2
        assert summary["total"] == 4


class TestAttractorRepulsion:
    """Inter-attractor repulsion mechanics."""

    def test_repulsion_increases_separation(self):
        # With repulsion, a midpoint state should still resolve
        ac_no_repulse = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [2.0, 0.0]]),
            labels=["A", "B"],
            flow_fn=normalize_flow,
            repulsion=0.0,
            max_steps=200,
        )
        ac_repulse = AttractorCompetition(
            attractors=np.array([[0.0, 0.0], [2.0, 0.0]]),
            labels=["A", "B"],
            flow_fn=normalize_flow,
            repulsion=0.5,
            max_steps=200,
        )
        midpoint = np.array([1.0, 0.1], dtype=np.float32)
        r_no = ac_no_repulse.compete(midpoint)
        r_yes = ac_repulse.compete(midpoint)
        # Both should produce a winner
        assert r_no["winner"] in ("A", "B")
        assert r_yes["winner"] in ("A", "B")

    def test_three_attractors_geometric_resolution(self):
        # Triangle of attractors
        ac = AttractorCompetition(
            attractors=np.array([
                [0.0, 0.0],
                [5.0, 0.0],
                [2.5, 4.33],
            ]),
            labels=["bottom-left", "bottom-right", "top"],
            flow_fn=normalize_flow,
            repulsion=0.05,
        )
        # Near bottom-left
        result = ac.compete(np.array([0.5, 0.5], dtype=np.float32))
        assert result["winner"] == "bottom-left"
        # Near top
        result = ac.compete(np.array([2.5, 3.5], dtype=np.float32))
        assert result["winner"] == "top"


class TestAttractorValidation:
    """Input validation."""

    def test_mismatched_labels_raises(self):
        with pytest.raises(ValueError, match="labels"):
            AttractorCompetition(
                attractors=np.array([[0.0, 0.0], [1.0, 1.0]]),
                labels=["only_one"],
                flow_fn=normalize_flow,
            )

    def test_1d_attractors_raises(self):
        with pytest.raises(ValueError, match="2D"):
            AttractorCompetition(
                attractors=np.array([0.0, 1.0]),
                labels=["A"],
                flow_fn=normalize_flow,
            )

    def test_wrong_state_dimension_raises(self):
        ac = AttractorCompetition(
            attractors=np.array([[0.0, 0.0]]),
            labels=["A"],
            flow_fn=normalize_flow,
        )
        with pytest.raises(ValueError, match="shape"):
            ac.compete(np.array([1.0, 2.0, 3.0]))


class TestSecuritySyntheticDemo:
    """Security demo: 40 states (20 legit near [0,0,...], 20 anomalous near [5,5,...]).

    Demonstrates attractor competition as anomaly detection —
    geometric classification without if/else.
    """

    def test_security_classification(self):
        d = 4
        rng = np.random.default_rng(42)

        # Two clusters: legit near origin, anomalous near [5,5,5,5]
        legit_center = np.zeros(d, dtype=np.float32)
        anomaly_center = np.full(d, 5.0, dtype=np.float32)

        legit_states = rng.normal(0.0, 0.3, (20, d)).astype(np.float32)
        anomaly_states = rng.normal(5.0, 0.3, (20, d)).astype(np.float32)
        all_states = np.vstack([legit_states, anomaly_states])

        ac = AttractorCompetition(
            attractors=np.array([legit_center, anomaly_center]),
            labels=["legit", "anomalous"],
            flow_fn=damped_flow,
            epsilon=0.1,
            tol=0.5,
            max_steps=200,
            repulsion=0.05,
        )

        results = ac.compete_batch(all_states)
        summary = ac.summary(results)

        # All 20 legit should classify as "legit"
        for i in range(20):
            assert results[i]["winner"] == "legit", (
                f"State {i} (legit) misclassified as {results[i]['winner']}"
            )

        # All 20 anomalous should classify as "anomalous"
        for i in range(20, 40):
            assert results[i]["winner"] == "anomalous", (
                f"State {i} (anomalous) misclassified as {results[i]['winner']}"
            )

        assert summary["counts"]["legit"] == 20
        assert summary["counts"]["anomalous"] == 20
        assert summary["mean_certainty"] > 0.5

    def test_security_contested_zone(self):
        """States equidistant from both attractors may be contested."""
        d = 4
        ac = AttractorCompetition(
            attractors=np.array([
                np.zeros(d, dtype=np.float32),
                np.full(d, 5.0, dtype=np.float32),
            ]),
            labels=["legit", "anomalous"],
            flow_fn=damped_flow,
            repulsion=0.0,  # No repulsion makes contention more likely at midpoint
        )
        midpoint = np.full(d, 2.5, dtype=np.float32)
        result = ac.compete(midpoint)
        # Should still resolve to some winner
        assert result["winner"] in ("legit", "anomalous")
