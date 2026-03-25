"""Trend Forecaster — computes daily metrics and generates simple forecasts.

Aggregates daily_metrics from transaction_events + contracts, then produces
linear-trend and mean-reversion forecasts for key metrics. Scores past predictions.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

REVERT_EQUILIBRIUM = 30.0  # mean-reversion target for revert rate


def get_connection() -> sqlite3.Connection:
    """Return a connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def compute_daily_metrics(conn: sqlite3.Connection, day: str) -> Optional[dict]:
    """Compute daily aggregate metrics for a single day.

    Args:
        conn: Database connection.
        day: Date string YYYY-MM-DD.

    Returns:
        Dict of metrics or None if no data.
    """
    tx = conn.execute("""
        SELECT COUNT(*) as total_tx,
               COUNT(DISTINCT interacting_address) as unique_callers,
               ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/MAX(COUNT(*),1)*100, 1) as revert_rate,
               COUNT(DISTINCT contract_address) as active_contracts
        FROM transaction_events
        WHERE DATE(timestamp) = ?
    """, (day,)).fetchone()

    if not tx or tx["total_tx"] == 0:
        return None

    new_contracts = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE DATE(detection_timestamp) = ?", (day,)
    ).fetchone()[0]

    new_bots = conn.execute(
        "SELECT COUNT(*) FROM bot_candidates WHERE DATE(first_seen) = ?", (day,)
    ).fetchone()[0]

    # Camouflage ratio from camouflage_metrics if available
    cam = conn.execute(
        "SELECT camouflage_ratio FROM camouflage_metrics WHERE date = ?", (day,)
    ).fetchone()
    cam_ratio = cam["camouflage_ratio"] if cam else 0

    avg_callers = round(tx["unique_callers"] / max(tx["active_contracts"], 1), 1)

    metrics = {
        "date": day,
        "chain": "ethereum",
        "new_contracts": new_contracts,
        "total_tx_events": tx["total_tx"],
        "unique_callers": tx["unique_callers"],
        "new_bots": new_bots,
        "revert_rate": tx["revert_rate"],
        "camouflage_ratio": cam_ratio,
        "active_orgs": None,
        "avg_callers_per_contract": avg_callers,
        "contracts_with_zero_interactions": 0,
    }

    conn.execute("""
        INSERT OR REPLACE INTO daily_metrics
            (date, chain, new_contracts, total_tx_events, unique_callers, new_bots,
             revert_rate, camouflage_ratio, active_orgs, avg_callers_per_contract,
             contracts_with_zero_interactions)
        VALUES (:date, :chain, :new_contracts, :total_tx_events, :unique_callers, :new_bots,
                :revert_rate, :camouflage_ratio, :active_orgs, :avg_callers_per_contract,
                :contracts_with_zero_interactions)
    """, metrics)
    conn.commit()
    return metrics


def backfill(conn: sqlite3.Connection) -> None:
    """Compute daily_metrics for all available days."""
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT DATE(timestamp) FROM transaction_events ORDER BY 1"
    ).fetchall()]
    print(f"[trend_forecaster] Backfilling daily_metrics for {len(dates)} days...")
    for day in dates:
        m = compute_daily_metrics(conn, day)
        if m:
            print(f"  {day}: tx={m['total_tx_events']} callers={m['unique_callers']} "
                  f"revert={m['revert_rate']}% new_contracts={m['new_contracts']}")


def compute_today(conn: sqlite3.Connection) -> None:
    """Compute metrics for today."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"[trend_forecaster] Computing metrics for {today}...")
    m = compute_daily_metrics(conn, today)
    if m:
        print(f"  tx={m['total_tx_events']} revert={m['revert_rate']}%")
    else:
        print("  No data for today.")


def forecast(conn: sqlite3.Connection) -> None:
    """Generate forecasts for the next day based on historical daily_metrics."""
    rows = conn.execute(
        "SELECT * FROM daily_metrics ORDER BY date"
    ).fetchall()

    if len(rows) < 3:
        print("[trend_forecaster] Need at least 3 days of data for forecasting.")
        return

    last_date = rows[-1]["date"]
    target = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"[trend_forecaster] Generating forecasts for {target}...")

    # --- Revert rate: mean reversion toward 30% ---
    recent_revert = [r["revert_rate"] for r in rows[-5:] if r["revert_rate"] is not None]
    if recent_revert:
        current = recent_revert[-1]
        mean_rev = current + 0.3 * (REVERT_EQUILIBRIUM - current)
        # Linear trend
        if len(recent_revert) >= 2:
            slope = (recent_revert[-1] - recent_revert[0]) / len(recent_revert)
            linear = current + slope
        else:
            linear = current
        predicted_revert = round((mean_rev + linear) / 2, 1)
        lo = round(predicted_revert - 5, 1)
        hi = round(predicted_revert + 5, 1)

        conn.execute("""
            INSERT OR REPLACE INTO predictions
                (issued_date, target_date, metric_name, predicted_value,
                 predicted_range_low, predicted_range_high, confidence, basis)
            VALUES (?, ?, 'revert_rate', ?, ?, ?, 'medium',
                    'mean_reversion + linear_trend')
        """, (today, target, predicted_revert, lo, hi))
        print(f"  revert_rate: {predicted_revert}% [{lo}-{hi}]")

    # --- New contracts: rolling average ---
    recent_contracts = [r["new_contracts"] for r in rows[-5:] if r["new_contracts"] is not None]
    if recent_contracts:
        avg_c = round(sum(recent_contracts) / len(recent_contracts))
        lo_c = max(0, avg_c - int(avg_c * 0.3))
        hi_c = avg_c + int(avg_c * 0.3)
        conn.execute("""
            INSERT OR REPLACE INTO predictions
                (issued_date, target_date, metric_name, predicted_value,
                 predicted_range_low, predicted_range_high, confidence, basis)
            VALUES (?, ?, 'new_contracts', ?, ?, ?, 'medium', 'rolling_avg_5d')
        """, (today, target, avg_c, lo_c, hi_c))
        print(f"  new_contracts: {avg_c} [{lo_c}-{hi_c}]")

    # --- Total tx: rolling average ---
    recent_tx = [r["total_tx_events"] for r in rows[-5:] if r["total_tx_events"] is not None]
    if recent_tx:
        avg_tx = round(sum(recent_tx) / len(recent_tx))
        lo_tx = max(0, avg_tx - int(avg_tx * 0.25))
        hi_tx = avg_tx + int(avg_tx * 0.25)
        conn.execute("""
            INSERT OR REPLACE INTO predictions
                (issued_date, target_date, metric_name, predicted_value,
                 predicted_range_low, predicted_range_high, confidence, basis)
            VALUES (?, ?, 'total_tx_events', ?, ?, ?, 'medium', 'rolling_avg_5d')
        """, (today, target, avg_tx, lo_tx, hi_tx))
        print(f"  total_tx_events: {avg_tx} [{lo_tx}-{hi_tx}]")

    conn.commit()


def score_predictions(conn: sqlite3.Connection) -> None:
    """Score past predictions against actual values."""
    unscored = conn.execute("""
        SELECT id, target_date, metric_name, predicted_value,
               predicted_range_low, predicted_range_high
        FROM predictions
        WHERE actual_value IS NULL
    """).fetchall()

    if not unscored:
        print("[trend_forecaster] No unscored predictions.")
        return

    scored = 0
    for p in unscored:
        actual_row = conn.execute(
            f"SELECT {p['metric_name']} FROM daily_metrics WHERE date = ?",
            (p["target_date"],)
        ).fetchone()

        if not actual_row:
            continue

        actual = actual_row[0]
        if actual is None:
            continue

        hit = 1 if p["predicted_range_low"] <= actual <= p["predicted_range_high"] else 0
        conn.execute("""
            UPDATE predictions SET actual_value = ?, hit = ?, scored_at = datetime('now')
            WHERE id = ?
        """, (actual, hit, p["id"]))
        scored += 1
        symbol = "HIT" if hit else "MISS"
        print(f"  [{symbol}] {p['metric_name']} on {p['target_date']}: "
              f"predicted={p['predicted_value']}, actual={actual}")

    conn.commit()
    print(f"[trend_forecaster] Scored {scored} predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trend Forecaster")
    parser.add_argument("--compute-today", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    conn = get_connection()
    if args.backfill:
        backfill(conn)
    if args.compute_today:
        compute_today(conn)
    if args.forecast:
        forecast(conn)
    if args.score:
        score_predictions(conn)
    if not any([args.backfill, args.compute_today, args.forecast, args.score]):
        parser.print_help()
    conn.close()
