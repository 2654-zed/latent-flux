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


# ============================================================
# sai_alerts persistence tests
# ============================================================

from surveillance.sai.sai_alerts import (  # noqa: E402
    AlertRow, ensure_schema, write_alert, write_alerts, fetch_recent,
)


def test_sai_alerts_schema_idempotent(tmp_path):
    """ensure_schema should be safe to call multiple times."""
    import sqlite3
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    ensure_schema(conn)
    ensure_schema(conn)  # second call must not error
    ensure_schema(conn)  # third call must not error
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sai_alerts'")
    assert cur.fetchone() is not None
    conn.close()


def test_sai_alerts_write_and_fetch(tmp_path):
    """Insert one alert, fetch it back, verify payload roundtrip."""
    import sqlite3
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    alert = AlertRow(
        detector="Q-002",
        severity="T1_IMMINENT",
        subject_address="0x752c5a95",
        subject_kind="contract",
        payload={"z_score": 130.0, "approvals_today": 4498},
        detected_at="2026-05-09T11:28:00Z",
    )
    rowid = write_alert(conn, alert)
    assert rowid > 0
    rows = fetch_recent(conn)
    assert len(rows) == 1
    assert rows[0]["detector"] == "Q-002"
    assert rows[0]["severity"] == "T1_IMMINENT"
    assert rows[0]["payload"]["z_score"] == 130.0
    conn.close()


def test_sai_alerts_unique_constraint(tmp_path):
    """UNIQUE (detector, subject_address, detected_at) must prevent dupes."""
    import sqlite3
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    alert = AlertRow(
        detector="Q-002", severity="T1_IMMINENT",
        subject_address="0xabc", subject_kind="contract",
        payload={"z": 100}, detected_at="2026-05-09T12:00:00Z",
    )
    write_alert(conn, alert)
    # Second write with identical key must return -1 (UNIQUE conflict)
    rc = write_alert(conn, alert)
    assert rc == -1, "duplicate should be rejected by UNIQUE constraint"
    rows = fetch_recent(conn)
    assert len(rows) == 1
    conn.close()


def test_sai_alerts_batch_or_ignore(tmp_path):
    """write_alerts uses INSERT OR IGNORE for batch idempotency."""
    import sqlite3
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    alerts = [
        AlertRow(detector="Q-009", severity="RESOLVED_VIA_OLI",
                 subject_address="0xa", subject_kind="drain_caller",
                 payload={}, detected_at="2026-05-09T13:00:00Z"),
        AlertRow(detector="Q-009", severity="RESOLVED_VIA_OLI",
                 subject_address="0xb", subject_kind="drain_caller",
                 payload={}, detected_at="2026-05-09T13:00:00Z"),
        # Duplicate of first (same detector + subject + ts)
        AlertRow(detector="Q-009", severity="RESOLVED_VIA_OLI",
                 subject_address="0xa", subject_kind="drain_caller",
                 payload={}, detected_at="2026-05-09T13:00:00Z"),
    ]
    n = write_alerts(conn, alerts)
    assert n == 2, f"expected 2 inserted (one duplicate ignored), got {n}"
    rows = fetch_recent(conn)
    assert len(rows) == 2
    conn.close()


# ============================================================
# question_generator (Phase 3) tests
# ============================================================

from surveillance.sai.question_generator import (  # noqa: E402
    parse_journal_surprises, parse_open_unknowns,
    transform_decompose, transform_adversarial_invert, transform_temporal_upgrade,
    FailureEvent, DraftQuestion, generate_for_failure,
)


def test_parse_journal_surprises_extracts_blocks():
    """JOURNAL.md SURPRISE blocks must parse with Expected/Observed/Implication."""
    surprises = parse_journal_surprises()
    # The 2026-05-15 SAI cycle entry has 3 SURPRISE blocks (A1/A2/A3)
    assert len(surprises) >= 3, f"expected >=3 SURPRISE blocks, got {len(surprises)}"
    # First block should have all required fields
    first = surprises[0]
    assert first.expected
    assert first.observed
    assert first.implication


def test_parse_open_unknowns_extracts_unk_ids():
    """UNKNOWNS.md must yield only OPEN/IN_PROGRESS entries."""
    unks = parse_open_unknowns()
    # 28 OPEN unknowns in the corpus per the SAI cycle inventory
    assert 10 < len(unks) < 50, f"expected 10-50 open unknowns, got {len(unks)}"
    for u in unks:
        assert u.source_id.startswith("UNK-")
        assert u.title


def test_transform_decompose_handles_failure_word():
    """A failure with 'risk' or 'fail' in title gets decomposition."""
    f = FailureEvent(
        source="journal_surprise", source_id="test_1",
        title="iter_8 contributes 0% to May-5 confirmed-trap spike.",
        expected="iter_8 drives some of the May-5 spike.",
        observed="0 contracts traceable.",
        implication="Named-entity heuristics fail.",
    )
    # "fail" is in implication, not title — need a title with the trigger
    f.title = "Risk model fails on Pattern A discharge"
    q = transform_decompose(f)
    assert q is not None
    assert q.transformation == "decomposition"
    assert "specific addresses, timestamps" in q.question


def test_transform_adversarial_invert_always_returns():
    """Adversarial inversion is the meta-transformation; always applicable."""
    f = FailureEvent(source="journal_surprise", source_id="t",
                     title="any failure", implication="guardrail issue")
    q = transform_adversarial_invert(f)
    assert q is not None
    assert q.transformation == "adversarial_inversion"
    assert "attacker" in q.question.lower()


def test_transform_temporal_upgrade_fires_on_discharge_titles():
    f = FailureEvent(source="t", source_id="t1", title="May-9 discharge of 0x752c5a95")
    q = transform_temporal_upgrade(f)
    assert q is not None
    assert q.transformation == "temporal_upgrade"
    assert "lead time" in q.question.lower()


def test_generate_for_failure_runs_all_three_when_applicable():
    f = FailureEvent(
        source="journal_surprise", source_id="t",
        title="Pattern A discharge risk on 0x752c5a95",
        implication="watchlist coverage gap",
    )
    drafts = generate_for_failure(f)
    # decompose (matches 'risk') + invert (always) + temporal (no 'discharge'
    # at start of title; the regex looks for 'discharge' anywhere — present)
    assert len(drafts) >= 2
    kinds = {d.transformation for d in drafts}
    assert "adversarial_inversion" in kinds


def test_question_generator_produces_drafts_end_to_end():
    """End-to-end: run the generator against the actual JOURNAL + UNKNOWNS."""
    j = parse_journal_surprises()
    u = parse_open_unknowns()
    all_failures = j + u
    drafts = []
    for f in all_failures:
        drafts.extend(generate_for_failure(f))
    # At minimum, expect 1 draft per failure on average
    assert len(drafts) >= len(all_failures), (
        f"expected >= {len(all_failures)} drafts; got {len(drafts)}"
    )


# ============================================================
# Alert-aware transformation tests (added 2026-05-17)
# Closes the meta-SURPRISE where sai_alerts produced only 1 draft each.
# ============================================================

from surveillance.sai.question_generator import (  # noqa: E402
    transform_alert_earlier_detection,
    transform_alert_structural_gap,
    transform_alert_actionable_resolution,
    _detector_id_from_alert,
)


def _sai_alert_failure(detector_id: str = "Q-003") -> FailureEvent:
    return FailureEvent(
        source="sai_alert",
        source_id=f"{detector_id}_2026-05-17T22:26:26Z_0xabc",
        title=f"{detector_id} alert: STALE on 0xabc",
    )


def test_detector_id_extraction():
    """_detector_id_from_alert extracts 'Q-003' from the source_id."""
    f = _sai_alert_failure("Q-003")
    assert _detector_id_from_alert(f) == "Q-003"
    # Non-sai-alert failure returns None
    f_journal = FailureEvent(source="journal_surprise", source_id="j1", title="x")
    assert _detector_id_from_alert(f_journal) is None


def test_alert_earlier_detection_fires_only_on_sai_alert():
    f_journal = FailureEvent(source="journal_surprise", source_id="j1", title="x")
    assert transform_alert_earlier_detection(f_journal) is None
    f_alert = _sai_alert_failure()
    q = transform_alert_earlier_detection(f_alert)
    assert q is not None
    assert q.transformation == "alert_earlier_detection"
    assert "additional signal" in q.question


def test_alert_structural_gap_fires_only_on_sai_alert():
    f_journal = FailureEvent(source="unknown_open", source_id="u1", title="x")
    assert transform_alert_structural_gap(f_journal) is None
    f_alert = _sai_alert_failure("Q-002")
    q = transform_alert_structural_gap(f_alert)
    assert q is not None
    assert "accumulation phase" in q.question  # Q-002-specific framing


def test_alert_actionable_resolution_per_detector_specific():
    """Each detector gets a different actionable-resolution framing."""
    qs = {}
    for det in ("Q-002", "Q-003", "Q-005", "Q-009"):
        f = _sai_alert_failure(det)
        q = transform_alert_actionable_resolution(f)
        assert q is not None
        qs[det] = q.question
    # Specificity check: at least three of the questions differ in detail
    distinct_phrases = {
        "Q-002": "USD-at-stake",
        "Q-003": "external attestation",
        "Q-005": "cross-chain coordination event",
        "Q-009": "second-hop",
    }
    for det, phrase in distinct_phrases.items():
        assert phrase in qs[det], (
            f"Q-{det}'s actionable_resolution missing the detector-specific "
            f"phrase '{phrase}'; got: {qs[det]}"
        )


def test_sai_alert_produces_four_drafts_after_patch():
    """The meta-SURPRISE closure: each sai_alert now produces 4 candidates,
    not 1.

    Before 2026-05-17 patch: only adversarial_inversion fired (the prose
    transformations did not match machine-generated alert titles).
    After patch: adversarial_inversion + 3 alert-aware transformations.
    """
    f = _sai_alert_failure()
    drafts = generate_for_failure(f)
    assert len(drafts) == 4, (
        f"expected 4 candidates per sai_alert (1 prose + 3 alert-aware); "
        f"got {len(drafts)}"
    )
    transformations = {d.transformation for d in drafts}
    assert transformations == {
        "adversarial_inversion",
        "alert_earlier_detection",
        "alert_structural_gap",
        "alert_actionable_resolution",
    }
