"""Tests for the /api/sai/* endpoints in web/app.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "web"))

try:
    from fastapi.testclient import TestClient
    from app import app as fastapi_app
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


@pytest.fixture(scope="module")
def client():
    if not HAS_FASTAPI:
        pytest.skip("fastapi not available in test env")
    return TestClient(fastapi_app)


def test_sai_alerts_summary_returns_200(client):
    r = client.get("/api/sai/alerts/summary")
    assert r.status_code == 200
    body = r.json()
    # Either populated (production has run --persist) or empty with note
    if "note" in body:
        assert "not yet populated" in body["note"]
    else:
        assert "total" in body
        assert "by_detector_severity" in body
        assert "latest_per_detector" in body
        assert isinstance(body["by_detector_severity"], list)


def test_sai_alerts_list_returns_200(client):
    r = client.get("/api/sai/alerts?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "alerts" in body
    assert "count" in body
    assert body["count"] <= 5


def test_sai_alerts_filter_by_detector(client):
    r = client.get("/api/sai/alerts?detector=Q-002&limit=10")
    assert r.status_code == 200
    body = r.json()
    for a in body["alerts"]:
        assert a["detector"] == "Q-002"


def test_sai_alerts_filter_by_severity(client):
    r = client.get("/api/sai/alerts?severity=STALE&limit=10")
    assert r.status_code == 200
    body = r.json()
    for a in body["alerts"]:
        assert a["severity"] == "STALE"


def test_sai_alerts_filter_by_subject_prefix(client):
    """Short subject (< 42 chars) uses LIKE prefix match."""
    r = client.get("/api/sai/alerts?subject=0x80b12bd0&limit=10")
    assert r.status_code == 200
    body = r.json()
    for a in body["alerts"]:
        assert a["subject_address"].startswith("0x80b12bd0")


def test_sai_alerts_filter_by_subject_full_address(client):
    """Full 42-char subject uses exact match."""
    full = "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8"
    r = client.get(f"/api/sai/alerts?subject={full}&limit=10")
    assert r.status_code == 200
    body = r.json()
    for a in body["alerts"]:
        assert a["subject_address"] == full


def test_sai_questions_returns_18(client):
    r = client.get("/api/sai/questions")
    assert r.status_code == 200
    body = r.json()
    assert "questions" in body
    # 18 canonical questions from the 2026-05-16 SAI cycle
    assert body["count"] == 18
    # Each question should have score, category, priority_score
    for q in body["questions"]:
        assert "id" in q
        assert "priority_score" in q
        assert "score" in q
        assert q["id"].startswith("Q-")


def test_sai_questions_ranked_descending(client):
    r = client.get("/api/sai/questions")
    body = r.json()
    scores = [q["priority_score"] for q in body["questions"]]
    assert scores == sorted(scores, reverse=True)


def test_sai_questions_top_is_approval_z(client):
    """Top-ranked questions are Q-002 and Q-014 (priority 4.90)."""
    r = client.get("/api/sai/questions")
    body = r.json()
    top_two_ids = {q["id"] for q in body["questions"][:2]}
    assert top_two_ids == {"Q-002", "Q-014"}, (
        f"expected top-two {{Q-002, Q-014}}; got {top_two_ids}"
    )
