"""Behavioral Baseline — computes baselines and detects anomalies.

Computes per-entity-type baselines for contracts (callers/day, revert rate,
caller return rate) and deployers (contracts/day, time between deploys).
Flags entities >2 stddev from baseline in behavioral_anomalies.
"""

import argparse
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

ANOMALY_THRESHOLD = 2.0  # standard deviations


def get_connection() -> sqlite3.Connection:
    """Return a connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def _stddev(values: list[float], mean: float) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def compute_baselines(conn: sqlite3.Connection) -> None:
    """Compute behavioral baselines segmented by entity category."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn.execute("DELETE FROM behavioral_baselines")
    conn.commit()

    print("[behavioral_baseline] Computing contract baselines...")

    # Get all entity categories
    categories = conn.execute(
        "SELECT DISTINCT category FROM entity_classification WHERE category IS NOT NULL"
    ).fetchall()
    cat_list = [r["category"] for r in categories] + ["_uncategorized"]

    for cat in cat_list:
        # Get contracts in this category
        if cat == "_uncategorized":
            contract_rows = conn.execute("""
                SELECT te.contract_address,
                       COUNT(DISTINCT te.interacting_address) as callers,
                       JULIANDAY(MAX(te.timestamp)) - JULIANDAY(MIN(te.timestamp)) as span,
                       ROUND(SUM(CASE WHEN te.is_reverted=1 THEN 1.0 ELSE 0 END)/MAX(COUNT(*),1)*100, 1) as revert_rate
                FROM transaction_events te
                WHERE te.contract_address NOT IN (SELECT address FROM entity_classification)
                GROUP BY te.contract_address
                HAVING COUNT(*) >= 5
            """).fetchall()
        else:
            contract_rows = conn.execute("""
                SELECT te.contract_address,
                       COUNT(DISTINCT te.interacting_address) as callers,
                       JULIANDAY(MAX(te.timestamp)) - JULIANDAY(MIN(te.timestamp)) as span,
                       ROUND(SUM(CASE WHEN te.is_reverted=1 THEN 1.0 ELSE 0 END)/MAX(COUNT(*),1)*100, 1) as revert_rate
                FROM transaction_events te
                JOIN entity_classification ec ON te.contract_address = ec.address
                WHERE ec.category = ?
                GROUP BY te.contract_address
                HAVING COUNT(*) >= 5
            """, (cat,)).fetchall()

        if not contract_rows:
            continue

        # callers_per_day
        cpd_vals = []
        revert_vals = []
        for r in contract_rows:
            span = max(r["span"] or 1, 1)
            cpd_vals.append(r["callers"] / span)
            revert_vals.append(r["revert_rate"] or 0)

        for metric_name, vals in [("callers_per_day", cpd_vals), ("revert_rate", revert_vals)]:
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            sd = _stddev(vals, mean)
            median = _percentile(vals, 50)
            p5 = _percentile(vals, 5)
            p95 = _percentile(vals, 95)

            conn.execute("""
                INSERT OR REPLACE INTO behavioral_baselines
                    (entity_type, metric_name, mean_value, median_value, stddev,
                     p5, p95, sample_size, computed_date, chain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ethereum')
            """, (f"contract:{cat}", metric_name, round(mean, 4), round(median, 4),
                  round(sd, 4), round(p5, 4), round(p95, 4), len(vals), today))

        print(f"  contract:{cat} — {len(contract_rows)} contracts baselined")

    # ---- Deployer baselines ----
    print("[behavioral_baseline] Computing deployer baselines...")
    deployer_rows = conn.execute("""
        SELECT d.deployer_address, d.total_contracts_deployed,
               d.typical_deployment_interval_hours,
               JULIANDAY(d.last_seen) - JULIANDAY(d.first_seen) as span_days
        FROM deployers d
        WHERE d.total_contracts_deployed >= 2
    """).fetchall()

    if deployer_rows:
        cpd_vals = []
        interval_vals = []
        for r in deployer_rows:
            span = max(r["span_days"] or 1, 1)
            cpd_vals.append(r["total_contracts_deployed"] / span)
            if r["typical_deployment_interval_hours"] and r["typical_deployment_interval_hours"] > 0:
                interval_vals.append(r["typical_deployment_interval_hours"])

        for metric_name, vals in [("contracts_per_day", cpd_vals),
                                   ("deploy_interval_hours", interval_vals)]:
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            sd = _stddev(vals, mean)
            conn.execute("""
                INSERT OR REPLACE INTO behavioral_baselines
                    (entity_type, metric_name, mean_value, median_value, stddev,
                     p5, p95, sample_size, computed_date, chain)
                VALUES ('deployer', ?, ?, ?, ?, ?, ?, ?, ?, 'ethereum')
            """, (metric_name, round(mean, 4), round(_percentile(vals, 50), 4),
                  round(sd, 4), round(_percentile(vals, 5), 4),
                  round(_percentile(vals, 95), 4), len(vals), today))

        print(f"  deployer — {len(deployer_rows)} deployers baselined")

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM behavioral_baselines").fetchone()[0]
    print(f"[behavioral_baseline] Stored {total} baseline records.")


def detect_anomalies(conn: sqlite3.Connection) -> None:
    """Detect entities with metrics >2 stddev from their baseline."""
    conn.execute("DELETE FROM behavioral_anomalies")
    conn.commit()

    baselines = conn.execute("SELECT * FROM behavioral_baselines").fetchall()
    if not baselines:
        print("[behavioral_baseline] No baselines computed yet.")
        return

    print("[behavioral_baseline] Detecting anomalies (threshold=2 stddev)...")
    anomaly_count = 0

    for bl in baselines:
        if bl["stddev"] is None or bl["stddev"] == 0:
            continue

        entity_type = bl["entity_type"]
        metric = bl["metric_name"]
        mean = bl["mean_value"]
        sd = bl["stddev"]

        # Get individual entity values to check
        if entity_type.startswith("contract:"):
            cat = entity_type.split(":", 1)[1]
            if cat == "_uncategorized":
                rows = conn.execute("""
                    SELECT te.contract_address as address,
                           COUNT(DISTINCT te.interacting_address) as callers,
                           JULIANDAY(MAX(te.timestamp)) - JULIANDAY(MIN(te.timestamp)) as span,
                           ROUND(SUM(CASE WHEN te.is_reverted=1 THEN 1.0 ELSE 0 END)/MAX(COUNT(*),1)*100, 1) as revert_rate
                    FROM transaction_events te
                    WHERE te.contract_address NOT IN (SELECT address FROM entity_classification)
                    GROUP BY te.contract_address
                    HAVING COUNT(*) >= 5
                """).fetchall()
            else:
                rows = conn.execute("""
                    SELECT te.contract_address as address,
                           COUNT(DISTINCT te.interacting_address) as callers,
                           JULIANDAY(MAX(te.timestamp)) - JULIANDAY(MIN(te.timestamp)) as span,
                           ROUND(SUM(CASE WHEN te.is_reverted=1 THEN 1.0 ELSE 0 END)/MAX(COUNT(*),1)*100, 1) as revert_rate
                    FROM transaction_events te
                    JOIN entity_classification ec ON te.contract_address = ec.address
                    WHERE ec.category = ?
                    GROUP BY te.contract_address
                    HAVING COUNT(*) >= 5
                """, (cat,)).fetchall()

            for r in rows:
                span = max(r["span"] or 1, 1)
                if metric == "callers_per_day":
                    val = r["callers"] / span
                elif metric == "revert_rate":
                    val = r["revert_rate"] or 0
                else:
                    continue

                z = (val - mean) / sd
                if abs(z) >= ANOMALY_THRESHOLD:
                    direction = "above" if z > 0 else "below"
                    conn.execute("""
                        INSERT INTO behavioral_anomalies
                            (address, entity_type, metric_name, observed_value,
                             baseline_mean, baseline_stddev, z_score,
                             anomaly_direction, detected_at, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                    """, (r["address"], entity_type, metric, round(val, 4),
                          mean, sd, round(z, 2), direction,
                          f"z={round(z, 2)} ({direction} baseline)"))
                    anomaly_count += 1

        elif entity_type == "deployer":
            rows = conn.execute("""
                SELECT deployer_address as address, total_contracts_deployed,
                       typical_deployment_interval_hours,
                       JULIANDAY(last_seen) - JULIANDAY(first_seen) as span_days
                FROM deployers WHERE total_contracts_deployed >= 2
            """).fetchall()

            for r in rows:
                span = max(r["span_days"] or 1, 1)
                if metric == "contracts_per_day":
                    val = r["total_contracts_deployed"] / span
                elif metric == "deploy_interval_hours":
                    val = r["typical_deployment_interval_hours"] or 0
                    if val == 0:
                        continue
                else:
                    continue

                z = (val - mean) / sd
                if abs(z) >= ANOMALY_THRESHOLD:
                    direction = "above" if z > 0 else "below"
                    conn.execute("""
                        INSERT INTO behavioral_anomalies
                            (address, entity_type, metric_name, observed_value,
                             baseline_mean, baseline_stddev, z_score,
                             anomaly_direction, detected_at, notes)
                        VALUES (?, 'deployer', ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                    """, (r["address"], metric, round(val, 4),
                          mean, sd, round(z, 2), direction,
                          f"z={round(z, 2)} ({direction} baseline)"))
                    anomaly_count += 1

    conn.commit()
    print(f"[behavioral_baseline] Detected {anomaly_count} anomalies.")

    # Summary by type
    summary = conn.execute("""
        SELECT entity_type, metric_name, anomaly_direction, COUNT(*) as cnt
        FROM behavioral_anomalies
        GROUP BY entity_type, metric_name, anomaly_direction
        ORDER BY cnt DESC
    """).fetchall()
    for s in summary:
        print(f"  {s['entity_type']} / {s['metric_name']} / {s['anomaly_direction']}: {s['cnt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Baseline & Anomaly Detection")
    parser.add_argument("--compute", action="store_true", help="Compute baselines")
    parser.add_argument("--detect-anomalies", action="store_true", help="Detect anomalies")
    args = parser.parse_args()

    conn = get_connection()
    if args.compute:
        compute_baselines(conn)
    if args.detect_anomalies:
        detect_anomalies(conn)
    if not args.compute and not args.detect_anomalies:
        parser.print_help()
    conn.close()
