"""Tests for SAI substrate: question_store + approval_spike_detector + runner."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest

# Make project importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from surveillance.sai.question_store import (  # noqa: E402
    Question, load_questions, rank, summary, save_questions, WEIGHTS,
)
from surveillance.analytics.approval_spike_detector import (  # noqa: E402
    compute_baseline, z_score, SpikeAlert,
)


# ============================================================
# question_store tests
# ============================================================

def test_question_loads_18_questions():
    """The 18 canonical questions from the 2026-05-16 SAI cycle load cleanly."""
    qs = load_questions()
    assert len(qs) == 18, f"expected 18 questions, got {len(qs)}"
    ids = {q.id for q in qs}
    expected_ids = {f"Q-{i:03d}" for i in range(1, 19)}
    assert ids == expected_ids, f"missing/extra ids: {expected_ids ^ ids}"


def test_priority_score_formula():
    """priority_score = PP*0.3 + A*0.3 + FR*0.3 + U*0.1.

    Verify with a known case (Q-002: PP=5, A=5, U=4, FR=5 -> 4.9).
    """
    q = Question(
        id="Q-TEST", category="x", question="?", status="active", origin="test",
        predictive_power=5, actionability=5, uniqueness=4, failure_reduction=5,
    )
    expected = 5 * 0.3 + 5 * 0.3 + 5 * 0.3 + 4 * 0.1
    assert math.isclose(q.priority_score(), expected, rel_tol=1e-9)
    assert math.isclose(q.priority_score(), 4.9, rel_tol=1e-9)


def test_priority_weights_sum_to_one():
    """Weights must sum to 1.0 so priority_score is bounded by max dimension score."""
    assert math.isclose(sum(WEIGHTS.values()), 1.0)


def test_rank_returns_highest_first():
    qs = load_questions()
    ranked = rank(qs)
    scores = [q.priority_score() for q in ranked]
    assert scores == sorted(scores, reverse=True), "rank() must return desc order"


def test_top_ranked_is_approval_z_score_question():
    """Q-002 (or its parent Q-014) must be tied for top — both are the most
    actionable + highest predictive_power in the substrate."""
    qs = load_questions()
    top = rank(qs)[0]
    top_score = top.priority_score()
    # Top-tier is Q-002 + Q-014, both at 4.90
    assert top.id in ("Q-002", "Q-014"), f"top question should be Q-002 or Q-014, got {top.id}"
    assert math.isclose(top_score, 4.9, rel_tol=1e-9), f"top score {top_score}"


def test_save_and_reload_roundtrip(tmp_path):
    qs = load_questions()
    test_path = tmp_path / "test_questions.yaml"
    save_questions(qs, test_path)
    qs2 = load_questions(test_path)
    assert len(qs) == len(qs2)
    id_map = {q.id: q for q in qs}
    for q2 in qs2:
        original = id_map[q2.id]
        assert q2.predictive_power == original.predictive_power
        assert q2.actionability == original.actionability
        assert math.isclose(q2.priority_score(), original.priority_score(), rel_tol=1e-9)


def test_summary_renders_top_n():
    qs = load_questions()
    s = summary(qs, top_n=3)
    # Should contain "rank" header + 3 question rows
    assert "rank" in s
    assert "Q-002" in s or "Q-014" in s, "top 3 must include the approval-Z-score question"


# ============================================================
# approval_spike_detector tests
# ============================================================

def test_z_score_handles_zero_baseline():
    """When baseline mean=0 and value>0, return a positive Z (smoothed)."""
    z = z_score(value=100, mean=0.0, stddev=0.0, n_obs=14)
    assert z > 0, "first-ever activity should produce positive Z, not 0"
    assert math.isfinite(z), "must not blow up to infinity"


def test_z_score_high_value_vs_baseline_50():
    """4498 approvals vs baseline mean 54.7 / stddev 34 should be Z>100.

    This is the canonical 0x752c5a95 May-9 case. The detector MUST identify it.
    """
    z = z_score(value=4498, mean=54.7, stddev=34.19, n_obs=13)
    assert z > 100, f"Z should be > 100 for the 0x752c5a95 case, got {z}"


def test_z_score_returns_zero_for_zero_zero():
    """value=0 and mean=0 should return Z=0 (no signal, no smoothing artifact)."""
    z = z_score(value=0, mean=0.0, stddev=0.0, n_obs=14)
    assert z == 0.0


def test_z_score_small_n_obs_dampens():
    """When n_obs < 3, raw Z is scaled down (we don't trust thin baselines)."""
    z_thin = z_score(value=100, mean=10.0, stddev=5.0, n_obs=1)
    z_thick = z_score(value=100, mean=10.0, stddev=5.0, n_obs=14)
    assert z_thin < z_thick, "thin baseline should dampen Z"


def test_compute_baseline_uses_zeros_for_missing_days():
    """Days with no recorded approvals contribute as zeros, not omissions.

    Without this, a contract that was idle then suddenly spikes wouldn't
    produce a high Z (because the baseline mean would only be computed
    from non-zero days).
    """
    daily = {
        "2026-05-08": 50,
        # 13 other days are missing => treated as zero
    }
    from datetime import datetime, timezone
    as_of = datetime(2026, 5, 9, tzinfo=timezone.utc)
    mean, stddev, n_obs = compute_baseline(daily, as_of, lookback_days=14)
    # Mean: 50 / 14 = 3.57
    assert math.isclose(mean, 50 / 14, rel_tol=1e-9)
    # Only 1 day with data
    assert n_obs == 1


def test_spike_alert_severity_tiers():
    a = SpikeAlert(
        contract_address="0xabc", chain="base", deployer_address="0xdef",
        deployer_watchlist_label=None, confidence_tier="confirmed",
        as_of_date="2026-05-09",
        same_day_approvals=4498, baseline_mean=54.7, baseline_stddev=34.19,
        z_score=130.0, baseline_days_observed=13,
    )
    assert a.severity() == "T1_IMMINENT"
    a.z_score = 7.5
    assert a.severity() == "T2_ELEVATED"
    a.z_score = 3.5
    assert a.severity() == "T3_NOTEWORTHY"
