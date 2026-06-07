"""Per-component health detail — surfaces SILENT failures the raw heartbeat
list misses.

A component that simply stops writing never appears in the heartbeat table, so
the existing /health shows the live chain monitors as fresh while a dead
background process is invisible. That is exactly how the routing monitor died
for 22 days undetected (Correction #23) and how the OLI pipeline ran on empty
data unnoticed (priority #22). This module checks each component's actual
last-activity from its OWN output table and assigns a status:

  OK      — fresh within threshold
  STALE   — past threshold but < 4x (recoverable lag)
  DEAD    — > 4x threshold (process likely down)
  ABSENT  — no activity row at all (never ran / table missing)
  BROKEN  — running but producing structurally-invalid output (e.g. OLI rows
            with empty tags)
  PAUSED  — intentionally disabled (recorded, not a failure)

op priorities #17, #18; reports/adversarial_audit_2026-06-06.md.
0 external calls — pure SQLite reads, safe on a read-only connection.
"""
import sqlite3
from datetime import datetime, timezone


def _age_min(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


def _scalar(conn, sql):
    try:
        r = conn.execute(sql).fetchone()
        return r[0] if r else None
    except sqlite3.Error:
        return None


def _status(age, thresh):
    if age is None:
        return "ABSENT"
    if age <= thresh:
        return "OK"
    if age <= thresh * 4:
        return "STALE"
    return "DEAD"


def component_health(conn) -> dict:
    """Return a per-component freshness report. `conn` may be read-only."""
    comps = []

    def add(name, last_ts, thresh_min, note="", force=None):
        age = _age_min(last_ts)
        comps.append({
            "component": name,
            "last_activity": last_ts,
            "age_min": round(age, 1) if age is not None else None,
            "threshold_min": thresh_min,
            "status": force or _status(age, thresh_min),
            "note": note,
        })

    # 1) chain monitors — per-component heartbeat (ingestion liveness, 60s cadence)
    try:
        rows = conn.execute(
            "SELECT component, MAX(timestamp) FROM heartbeat GROUP BY component"
        ).fetchall()
        if rows:
            for comp, ts in rows:
                if "rout" in (comp or "").lower():
                    continue  # routing monitor handled explicitly below (#23 note)
                add(comp, ts, 5, "chain monitor heartbeat (60s cadence)")
        else:
            add("chain_monitors", None, 5, "no heartbeat rows")
    except sqlite3.Error:
        add("chain_monitors", None, 5, "heartbeat table unreadable")

    # 2) approval tracking (~30m cadence)
    add("approval_tracking",
        _scalar(conn, "SELECT MAX(logged_at) FROM approval_watchlist"),
        60, "approve() watchlist scan")

    # 3) drain detection — intentionally PAUSED (Correction #29)
    ndv = _scalar(conn, "SELECT COUNT(*) FROM drain_initiator_verdicts") or 0
    add("drain_detection",
        _scalar(conn, "SELECT MAX(checked_at) FROM drain_initiator_verdicts"),
        1440, f"PAUSED (Correction #29); tx-initiator verdicts cached={ndv}",
        force="PAUSED")

    # 4) OLI enrichment — BROKEN if every row has empty tags (priority #22)
    oli_total = _scalar(conn, "SELECT COUNT(*) FROM oli_labels")
    oli_empty = _scalar(conn, "SELECT COUNT(*) FROM oli_labels WHERE COALESCE(tag_count,0)=0")
    oli_last = _scalar(conn, "SELECT MAX(fetched_at) FROM oli_labels")
    if oli_total is None:
        add("oli_enrichment", None, 10080, "oli_labels table absent")
    elif oli_total > 0 and oli_empty == oli_total:
        add("oli_enrichment", oli_last, 10080,
            f"BROKEN: all {oli_total} rows tag_count=0 / entity NULL (priority #22)",
            force="BROKEN")
    else:
        add("oli_enrichment", oli_last, 10080, f"rows={oli_total} empty={oli_empty}")

    # 5) routing monitor — Correction #23 (0 signals corpus-wide; likely ABSENT)
    add("routing_monitor",
        _scalar(conn, "SELECT MAX(timestamp) FROM heartbeat WHERE component LIKE '%rout%'"),
        60, "Correction #23: 0 routing signals ever; verify before relying")

    # 6) bot / revert-cluster detector
    add("bot_detector",
        _scalar(conn, "SELECT MAX(last_seen) FROM bot_candidates"),
        1440, "revert-cluster detector")

    bad = [c["component"] for c in comps if c["status"] in ("DEAD", "BROKEN", "ABSENT")]
    stale = [c["component"] for c in comps if c["status"] == "STALE"]
    overall = "degraded" if bad else ("stale" if stale else "nominal")
    return {
        "overall": overall,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "degraded": bad,
        "stale": stale,
        "components": comps,
    }
