"""Prediction registry — calibration log for pre-registered predictions.

STATUS: SKELETON. The schema is defined; the write-side and the
calibration-curve compute are stubs. Phase 4 of the SAI plan.

Schema (target):
    CREATE TABLE prediction_registry (
        id              INTEGER PRIMARY KEY,
        session_id      TEXT NOT NULL,
        registered_at   TEXT NOT NULL,
        hypothesis_id   TEXT,          -- e.g., "A1", "A2", "A3"
        claim           TEXT NOT NULL,
        confidence_tag  TEXT,          -- LIKELY, SPLIT, UNLIKELY, SYSTEMIC, etc.
        evidence_required TEXT,
        check_query     TEXT,          -- optional SQL or script ref
        evaluated_at    TEXT,
        outcome         TEXT,          -- CONFIRMED, REFINED, CONTRADICTED, INCONCLUSIVE
        outcome_notes   TEXT,
        calibration_score REAL
    );

Why this exists: Phase A 2026-05-15 falsified 3/3 predictions but the
lesson lives only as prose in JOURNAL.md. There is no aggregate metric
of "our LIKELY-tagged predictions verify at X%." Without this, the
methodological lesson (named-entity heuristics fail) doesn't compound
across sessions — it just gets re-discovered.

CLI (when implemented):
    python -m surveillance.sai.prediction_registry register --hypothesis A1 ...
    python -m surveillance.sai.prediction_registry evaluate --hypothesis A1 ...
    python -m surveillance.sai.prediction_registry calibration
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    hypothesis_id TEXT,
    claim TEXT NOT NULL,
    confidence_tag TEXT,
    evidence_required TEXT,
    check_query TEXT,
    evaluated_at TEXT,
    outcome TEXT,
    outcome_notes TEXT,
    calibration_score REAL
);
CREATE INDEX IF NOT EXISTS idx_pr_session ON prediction_registry(session_id);
CREATE INDEX IF NOT EXISTS idx_pr_outcome ON prediction_registry(outcome);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init", action="store_true",
                    help="create the prediction_registry table")
    args = ap.parse_args()
    if args.init:
        conn = sqlite3.connect(DB_PATH)
        try:
            ensure_schema(conn)
            print(f"prediction_registry schema ensured in {DB_PATH}")
        finally:
            conn.close()
        return 0
    print("[SKELETON] prediction_registry not yet fully implemented.")
    print("Run with --init to create the schema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
