"""Persistent SAI alerts table — used by all SAI detectors when run with
`--persist`.

The table is the durable output surface for SAI detectors when they run as
scheduled jobs in production. Without it, detector findings are ephemeral
(scrollback in the scheduler logs).

Schema:

    CREATE TABLE sai_alerts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at     TEXT NOT NULL,         -- ISO-8601 UTC at detection
        detector        TEXT NOT NULL,         -- question id, e.g. "Q-002"
        severity        TEXT NOT NULL,         -- T1_IMMINENT, STALE, RESOLVED, etc.
        subject_address TEXT,                  -- primary entity, if applicable
        subject_kind    TEXT,                  -- "contract" | "deployer" | "drain_caller" | etc.
        payload         TEXT NOT NULL,         -- full alert as JSON
        UNIQUE (detector, subject_address, detected_at)
    );

Public API:
    ensure_schema(conn)                # idempotent CREATE TABLE
    write_alert(conn, alert)           # insert one
    write_alerts(conn, alerts)         # batch insert (atomic)
    AlertRow(...)                      # canonical dataclass

CLI (for inspection):
    python -m surveillance.sai.sai_alerts                 # show recent alerts
    python -m surveillance.sai.sai_alerts --detector Q-002
    python -m surveillance.sai.sai_alerts --since 2026-05-09
    python -m surveillance.sai.sai_alerts --init          # ensure schema only
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sai_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    detector TEXT NOT NULL,
    severity TEXT NOT NULL,
    subject_address TEXT,
    subject_kind TEXT,
    payload TEXT NOT NULL,
    UNIQUE (detector, subject_address, detected_at)
);
CREATE INDEX IF NOT EXISTS idx_sai_detector ON sai_alerts(detector);
CREATE INDEX IF NOT EXISTS idx_sai_detected_at ON sai_alerts(detected_at);
CREATE INDEX IF NOT EXISTS idx_sai_subject ON sai_alerts(subject_address);
CREATE INDEX IF NOT EXISTS idx_sai_severity ON sai_alerts(severity);
"""


@dataclass
class AlertRow:
    detector: str          # "Q-002", "Q-009", ...
    severity: str          # "T1_IMMINENT", "STALE", "NEEDS_VERIFICATION", "RESOLVED", ...
    subject_address: str | None
    subject_kind: str | None
    payload: dict          # serialized to JSON on insert
    detected_at: str | None = None  # defaults to now()

    def to_tuple(self) -> tuple:
        ts = self.detected_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            ts,
            self.detector,
            self.severity,
            self.subject_address,
            self.subject_kind,
            json.dumps(self.payload, default=str, sort_keys=True),
        )


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent CREATE TABLE. Safe to call on every detector invocation."""
    conn.executescript(SCHEMA)
    conn.commit()


def write_alert(conn: sqlite3.Connection, alert: AlertRow) -> int:
    """Insert one alert. Returns rowid on insert; -1 on UNIQUE violation."""
    ensure_schema(conn)
    try:
        cur = conn.execute(
            """INSERT INTO sai_alerts
               (detected_at, detector, severity, subject_address, subject_kind, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            alert.to_tuple(),
        )
        conn.commit()
        return cur.lastrowid or -1
    except sqlite3.IntegrityError:
        # UNIQUE (detector, subject_address, detected_at) — same detector
        # already fired on same subject in same second. Skip silently.
        return -1


def write_alerts(conn: sqlite3.Connection, alerts: Iterable[AlertRow]) -> int:
    """Batch insert. Returns count of rows actually inserted (ignores UNIQUE
    conflicts via INSERT OR IGNORE)."""
    ensure_schema(conn)
    rows = [a.to_tuple() for a in alerts]
    cur = conn.executemany(
        """INSERT OR IGNORE INTO sai_alerts
           (detected_at, detector, severity, subject_address, subject_kind, payload)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount


def fetch_recent(
    conn: sqlite3.Connection,
    detector: str | None = None,
    since: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return recent alerts as dicts (payload deserialized)."""
    ensure_schema(conn)
    where = []
    params: list = []
    if detector:
        where.append("detector = ?")
        params.append(detector)
    if since:
        where.append("detected_at >= ?")
        params.append(since)
    sql = "SELECT detected_at, detector, severity, subject_address, subject_kind, payload FROM sai_alerts"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY detected_at DESC LIMIT ?"
    params.append(limit)
    out = []
    for r in conn.execute(sql, params):
        out.append({
            "detected_at": r[0],
            "detector": r[1],
            "severity": r[2],
            "subject_address": r[3],
            "subject_kind": r[4],
            "payload": json.loads(r[5]) if r[5] else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init", action="store_true", help="ensure schema and exit")
    ap.add_argument("--detector", default=None, help="filter by detector id")
    ap.add_argument("--since", default=None, help="filter detected_at >= this")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    if args.init:
        conn = sqlite3.connect(args.db)
        try:
            ensure_schema(conn)
            print(f"sai_alerts schema ensured in {args.db}")
        finally:
            conn.close()
        return 0

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = fetch_recent(conn, detector=args.detector, since=args.since, limit=args.limit)
    finally:
        conn.close()
    print(f"\nRecent SAI alerts ({len(rows)}):\n")
    for r in rows:
        addr = r["subject_address"] or "(no subject)"
        kind = r["subject_kind"] or "?"
        print(f"  {r['detected_at']:22s}  {r['detector']:6s}  {r['severity']:18s}  {addr}  [{kind}]")
        payload = r["payload"] or {}
        for k, v in list(payload.items())[:5]:
            s = str(v)
            if len(s) > 80:
                s = s[:77] + "..."
            print(f"    {k}: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
