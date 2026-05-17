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


# ============================================================
# Q-009 funding_chain_pathfinder tests
# ============================================================

from surveillance.ontology.funding_chain_pathfinder import (  # noqa: E402
    parse_funding_trail, FundingChain, FundingHop,
)


def test_parse_funding_trail_dict():
    """funding_trail is a JSON dict — most common case."""
    raw = '{"funder":"0xabc","value_eth":0.05,"timestamp":"2026-04-03T12:57:31Z"}'
    parsed = parse_funding_trail(raw)
    assert parsed is not None
    assert parsed["funder"] == "0xabc"
    assert parsed["value_eth"] == 0.05


def test_parse_funding_trail_handles_empty_and_malformed():
    assert parse_funding_trail(None) is None
    assert parse_funding_trail("") is None
    assert parse_funding_trail("not-json") is None
    assert parse_funding_trail("[]") is None  # empty list returns None


def test_parse_funding_trail_list_returns_first():
    raw = '[{"funder":"0xabc"},{"funder":"0xdef"}]'
    parsed = parse_funding_trail(raw)
    assert parsed is not None
    assert parsed["funder"] == "0xabc"


def test_funding_chain_resolution_summary():
    chain = FundingChain(drain_caller="0x1d81", drains_in_window=3228)
    chain.hops.append(FundingHop(
        address="0xf70da978", depth=1, value_eth=0.004,
        timestamp="2025-02-14", tx_hash="0xabc",
        oli_severity="HIGH",
    ))
    chain.terminal_reason = "flagged ancestor at hop 1"
    assert chain.is_resolved_to_known()
    assert "OLI_HIT@hop1" in chain.resolution_summary()


def test_q009_against_live_corpus():
    """Q-009 must resolve >=50% of May-9..15 drain volume to known operators."""
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parent.parent.parent / "surveillance" / "data" / "surveillance.db"
    if not db.exists():
        pytest.skip("surveillance.db not present in this environment")
    from surveillance.ontology.funding_chain_pathfinder import trace_drainers_in_window
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        chains = trace_drainers_in_window(conn, "2026-05-09", "2026-05-16", max_hops=3)
    finally:
        conn.close()
    total_volume = sum(c.drains_in_window for c in chains)
    resolved_volume = sum(c.drains_in_window for c in chains if c.is_resolved_to_known())
    # Expectation per the 2026-05-16 SAI cycle: 6,074 / 8,187 = 74.2%
    assert total_volume >= 7000, f"unexpected total drain volume: {total_volume}"
    pct = 100 * resolved_volume / total_volume
    assert pct >= 50.0, f"Q-009 resolved only {pct:.1f}% of drain volume; expected >=50%"


# ============================================================
# Q-005 cross_chain_choreography tests
# ============================================================

def test_q005_catches_pattern_d_for_0x80b12bd0():
    """0x80b12bd0 (Animoca-tagged) must produce a pattern_d_gap signal."""
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parent.parent.parent / "surveillance" / "data" / "surveillance.db"
    if not db.exists():
        pytest.skip("surveillance.db not present in this environment")
    from surveillance.analytics.cross_chain_choreography import detect_for_address
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        c = detect_for_address(conn, "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8")
    finally:
        conn.close()
    assert c.has_choreography(), "0x80b12bd0 must produce at least one signal"
    kinds = {s.kind for s in c.signals}
    assert "pattern_d_gap" in kinds, f"expected pattern_d_gap; got {kinds}"
    # Mainnet first tx 2019-05-23 + L2 first-seen 2026-03-26 = ~2499 days
    assert c.mainnet_first_tx is not None


# ============================================================
# Q-003 oli_temporal_validity tests
# ============================================================

def test_q003_marks_0x80b12bd0_stale():
    """Animoca-tagged 0x80b12bd0 must be marked STALE."""
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parent.parent.parent / "surveillance" / "data" / "surveillance.db"
    if not db.exists():
        pytest.skip("surveillance.db not present in this environment")
    from surveillance.sai.oli_temporal_validity import assess_address
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        v = assess_address(conn, "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8", "LOW")
    finally:
        conn.close()
    assert v.verdict() == "STALE", f"expected STALE for Animoca-flagged-adversarial; got {v.verdict()} (score={v.aggregate_score()})"
    # Specific signal that must fire: adversarial watchlist HIGH
    signal_names = {s.name for s in v.signals}
    assert "adversarial_watchlist_high" in signal_names
    assert "deployed_confirmed_trap" in signal_names


def test_q003_marks_orbiter_bridge_fresh():
    """0x80c67432 (Orbiter Finance Bridge per Blockscout audit) must be FRESH."""
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parent.parent.parent / "surveillance" / "data" / "surveillance.db"
    if not db.exists():
        pytest.skip("surveillance.db not present in this environment")
    from surveillance.sai.oli_temporal_validity import assess_address
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        v = assess_address(conn, "0x80c67432656d59144ceff962e8faf8926599bcf8", "HIGH")
    finally:
        conn.close()
    assert v.verdict() == "FRESH", f"expected FRESH for Orbiter Bridge; got {v.verdict()}"


def test_q003_verdict_thresholds():
    """Verify the score → verdict mapping."""
    from surveillance.sai.oli_temporal_validity import OLIValidity, StaleSignal
    v = OLIValidity(address="0xtest", oli_severity="HIGH")
    assert v.verdict() == "FRESH"   # 0.0
    v.signals.append(StaleSignal("test", 2.0, ""))
    assert v.verdict() == "NEEDS_VERIFICATION"  # 2.0
    v.signals.append(StaleSignal("test2", 3.5, ""))
    assert v.verdict() == "STALE"   # 5.5
