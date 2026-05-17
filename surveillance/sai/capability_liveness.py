"""Capability liveness monitor — answers Q-008.

STATUS: SKELETON. A capability inventory exists in-code but the
last-run tracking is a session-time scan rather than a continuous
production heartbeat. Phase 4 of the SAI plan calls for this to
become a persistent service.

For every named Layer 3 capability, tracks:
  - last_verified_run_at (timestamp)
  - status (LIVE | STALE | UNVERIFIED)
  - staleness_threshold_days

A capability is STALE if it has not run within its threshold. STALE
capabilities are documented as live but cannot be relied on.

Empirical anchor: regime_monitor was claimed "LIVE first flux_manifold
production consumer" but runs only manually in sessions. The 2026-05-15
session discovered this gap. Episode count: 1.

CLI:
    python -m surveillance.sai.capability_liveness
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


@dataclass
class CapabilityCheck:
    name: str
    description: str
    staleness_threshold_hours: int
    last_verified_at: datetime | None
    check_query: str
    status: str = "UNVERIFIED"
    note: str = ""

    def is_stale(self) -> bool:
        if self.last_verified_at is None:
            return True
        delta = datetime.now(timezone.utc) - self.last_verified_at
        return delta > timedelta(hours=self.staleness_threshold_hours)


# Capability inventory — each entry pairs a documented capability with
# a query that proves it ran recently. TODO: expand to cover everything
# claimed in README.md / CLAUDE.md / STATE.md.
CAPABILITIES: list[tuple[str, str, int, str]] = [
    (
        "regime_monitor",
        "Bayesian changepoint detection on 6 daily signals; writes regime_alerts",
        24 * 7,  # 1 week
        "SELECT MAX(detected_at) FROM regime_alerts",
    ),
    (
        "deployment_monitor",
        "Live deployment ingestion across Base / Arbitrum / Optimism",
        24,  # 1 day
        "SELECT MAX(last_seen) FROM deployers",
    ),
    (
        "approval_monitor",
        "Continuous monitoring of approval_watchlist contracts",
        24,
        "SELECT MAX(logged_at) FROM approval_watchlist",
    ),
    (
        "extraction_event_curator",
        "Curated extraction_events writes (manual)",
        24 * 30,  # 1 month
        "SELECT MAX(documented_at) FROM extraction_events",
    ),
    (
        "regime_alert_to_production",
        "regime_monitor runs in the production worker, not just sessions",
        24,  # 1 day
        # If regime_alerts table is empty, the prod worker isn't running it.
        # When wired, this should always have a recent row.
        "SELECT MAX(detected_at) FROM regime_alerts WHERE detected_at >= datetime('now', '-1 day')",
    ),
]


def check_capability(conn: sqlite3.Connection, name: str, description: str,
                     threshold_hours: int, query: str) -> CapabilityCheck:
    try:
        row = conn.execute(query).fetchone()
        last = row[0] if row else None
    except sqlite3.OperationalError as e:
        return CapabilityCheck(name, description, threshold_hours, None, query,
                               status="ERROR", note=str(e))
    if last is None:
        return CapabilityCheck(name, description, threshold_hours, None, query,
                               status="UNVERIFIED", note="no rows returned")
    try:
        ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return CapabilityCheck(name, description, threshold_hours, None, query,
                               status="ERROR", note=f"unparseable timestamp: {last!r}")
    cap = CapabilityCheck(name, description, threshold_hours, ts, query)
    cap.status = "STALE" if cap.is_stale() else "LIVE"
    if cap.status == "STALE":
        delta = datetime.now(timezone.utc) - ts
        cap.note = f"last run {delta.total_seconds() / 3600:.1f} hours ago (threshold {threshold_hours}h)"
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print("Capability liveness check (answers Q-008):\n")
    print(f"  {'capability':30s}  {'status':12s}  {'last verified':22s}  threshold")
    print("  " + "-" * 88)
    n_stale = 0
    for name, desc, threshold, query in CAPABILITIES:
        cap = check_capability(conn, name, desc, threshold, query)
        last_str = cap.last_verified_at.strftime("%Y-%m-%dT%H:%M:%SZ") if cap.last_verified_at else "(never)"
        print(f"  {name:30s}  {cap.status:12s}  {last_str:22s}  {threshold}h")
        if cap.note:
            print(f"  {'':30s}  note: {cap.note}")
        if cap.status in ("STALE", "UNVERIFIED", "ERROR"):
            n_stale += 1
    conn.close()
    print(f"\n  STALE/UNVERIFIED capabilities: {n_stale}")
    return 0 if n_stale == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
