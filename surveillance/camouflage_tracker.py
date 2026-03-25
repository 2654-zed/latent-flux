"""Camouflage Tracker — computes daily camouflage metrics for contracts.

Classifies contracts by revert rate into camouflaged (<10%), moderate (10-50%),
and overt (>50%) categories, then stores aggregate metrics per day.
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_available_dates(conn: sqlite3.Connection) -> list[str]:
    """Return all distinct dates in transaction_events."""
    rows = conn.execute(
        "SELECT DISTINCT DATE(timestamp) AS d FROM transaction_events ORDER BY d"
    ).fetchall()
    return [r["d"] for r in rows]


def compute_day(conn: sqlite3.Connection, day: str) -> Optional[dict]:
    """Compute camouflage metrics for a single day.

    Args:
        conn: Database connection.
        day: Date string in YYYY-MM-DD format.

    Returns:
        Dict of metrics or None if no qualifying contracts.
    """
    rows = conn.execute("""
        SELECT contract_address,
            COUNT(*) as tx_count,
            COUNT(DISTINCT interacting_address) as callers,
            ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100, 1) as revert_pct
        FROM transaction_events
        WHERE DATE(timestamp) = ?
        GROUP BY contract_address
        HAVING COUNT(*) >= 10
    """, (day,)).fetchall()

    if not rows:
        return None

    camouflaged = [r for r in rows if r["revert_pct"] < 10]
    moderate = [r for r in rows if 10 <= r["revert_pct"] <= 50]
    overt = [r for r in rows if r["revert_pct"] > 50]
    total = len(rows)

    avg_cam_callers = sum(r["callers"] for r in camouflaged) / max(len(camouflaged), 1)
    avg_overt_callers = sum(r["callers"] for r in overt) / max(len(overt), 1)
    total_cam_victims = sum(r["callers"] for r in camouflaged)
    total_overt_victims = sum(r["callers"] for r in overt)

    metrics = {
        "date": day,
        "chain": "ethereum",
        "total_active_contracts": total,
        "camouflaged_count": len(camouflaged),
        "overt_count": len(overt),
        "moderate_count": len(moderate),
        "camouflage_ratio": round(len(camouflaged) / total, 4) if total else 0,
        "overt_ratio": round(len(overt) / total, 4) if total else 0,
        "avg_camouflaged_callers": round(avg_cam_callers, 1),
        "avg_overt_callers": round(avg_overt_callers, 1),
        "total_camouflaged_victims": total_cam_victims,
        "total_overt_victims": total_overt_victims,
    }

    conn.execute("""
        INSERT OR REPLACE INTO camouflage_metrics
            (date, chain, total_active_contracts, camouflaged_count, overt_count,
             moderate_count, camouflage_ratio, overt_ratio, avg_camouflaged_callers,
             avg_overt_callers, total_camouflaged_victims, total_overt_victims)
        VALUES (:date, :chain, :total_active_contracts, :camouflaged_count, :overt_count,
                :moderate_count, :camouflage_ratio, :overt_ratio, :avg_camouflaged_callers,
                :avg_overt_callers, :total_camouflaged_victims, :total_overt_victims)
    """, metrics)
    conn.commit()
    return metrics


def compute_all(conn: sqlite3.Connection) -> None:
    """Backfill camouflage metrics for all available days."""
    dates = get_available_dates(conn)
    print(f"[camouflage_tracker] Backfilling {len(dates)} days...")
    for day in dates:
        result = compute_day(conn, day)
        if result:
            print(f"  {day}: {result['total_active_contracts']} contracts "
                  f"(cam={result['camouflaged_count']}, mod={result['moderate_count']}, "
                  f"overt={result['overt_count']})")
        else:
            print(f"  {day}: no qualifying contracts")


def compute_today(conn: sqlite3.Connection) -> None:
    """Compute metrics for today only."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"[camouflage_tracker] Computing for {today}...")
    result = compute_day(conn, today)
    if result:
        print(f"  Done: {result['total_active_contracts']} contracts")
    else:
        print("  No qualifying contracts for today.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camouflage Tracker")
    parser.add_argument("--compute-today", action="store_true")
    parser.add_argument("--compute-all", action="store_true")
    args = parser.parse_args()

    conn = get_connection()
    if args.compute_all:
        compute_all(conn)
    elif args.compute_today:
        compute_today(conn)
    else:
        parser.print_help()
    conn.close()
