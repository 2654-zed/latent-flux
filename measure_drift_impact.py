"""
Post-Drift impact re-measurement tool.

Usage:
    python measure_drift_impact.py week1    # Run on April 8
    python measure_drift_impact.py week2    # Run on April 15
    python measure_drift_impact.py month1   # Run on May 1

Each run compares pre-Drift baseline (Mar 25-31) against the post-Drift
window up to the current date.
"""
import sqlite3, json, os, sys
from datetime import datetime, timezone
from collections import defaultdict

DB_PATH = os.environ.get(
    "DB_PATH",
    str(os.path.join(os.path.dirname(__file__), "surveillance", "data", "surveillance.db")),
)

def run(period):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc)

    out = []
    def p(s=""): out.append(str(s))

    titles = {
        "week1": ("1-Week Post-Drift Re-Measurement", "April 1-7", "drift_impact_week1.md"),
        "week2": ("2-Week Post-Drift Re-Measurement", "April 1-14", "drift_impact_week2.md"),
        "month1": ("1-Month Post-Drift Re-Measurement", "April 1-30", "drift_impact_month1.md"),
    }
    title, post_range, filename = titles[period]

    p(f"# {title}")
    p(f"Generated: {now.isoformat()[:19]}Z")
    p(f"Baseline: March 25-31 (7 days pre-Drift)")
    p(f"Post-Drift window: {post_range}")
    p("")

    def daily(sql):
        return [dict(r) for r in conn.execute(sql).fetchall()]

    def avg_by_period(rows, day_col='day', val_col='cnt'):
        pre = [r[val_col] for r in rows if r.get(day_col) and '2026-03-25' <= r[day_col] <= '2026-03-31']
        post = [r[val_col] for r in rows if r.get(day_col) and r[day_col] >= '2026-04-01']
        pre_avg = sum(pre) / len(pre) if pre else 0
        post_avg = sum(post) / len(post) if post else 0
        pct = round(100 * (post_avg - pre_avg) / pre_avg, 1) if pre_avg else 0
        return pre_avg, post_avg, pct

    # --- Core metrics ---
    metrics = [
        ("Contract deployments/day",
         "SELECT date(detection_timestamp) as day, COUNT(*) as cnt FROM contracts WHERE detection_timestamp >= '2026-03-25' GROUP BY day"),
        ("TX events/day",
         "SELECT date(timestamp) as day, COUNT(*) as cnt FROM transaction_events WHERE timestamp >= '2026-03-25' GROUP BY day"),
        ("New deployers/day",
         "SELECT date(first_seen) as day, COUNT(*) as cnt FROM deployers WHERE first_seen >= '2026-03-25' GROUP BY day"),
        ("Bot events/day",
         "SELECT date(timestamp) as day, COUNT(*) as cnt FROM bot_candidate_events WHERE timestamp >= '2026-03-25' GROUP BY day"),
        ("Approval events/day",
         "SELECT date(timestamp) as day, COUNT(*) as cnt FROM approval_events WHERE timestamp >= '2026-03-25' GROUP BY day"),
        ("Drains/day",
         "SELECT date(drain_timestamp) as day, COUNT(*) as cnt FROM approval_watchlist WHERE drain_detected=1 AND drain_timestamp >= '2026-03-25' GROUP BY day"),
        ("Liquidity events/day",
         "SELECT date(timestamp) as day, COUNT(*) as cnt FROM liquidity_events WHERE timestamp >= '2026-03-25' GROUP BY day"),
        ("Trap confirmations/day",
         "SELECT date(timestamp) as day, COUNT(*) as cnt FROM trap_events WHERE timestamp >= '2026-03-25' GROUP BY day"),
    ]

    p("## Summary Table")
    p("")
    p("| Metric | Pre-Drift avg | Post-Drift avg | Change | Signal? |")
    p("|--------|--------------|---------------|--------|---------|")
    for name, sql in metrics:
        rows = daily(sql)
        pre_a, post_a, pct = avg_by_period(rows)
        sig = "**YES**" if abs(pct) > 20 else "no"
        p(f"| {name} | {pre_a:,.0f} | {post_a:,.0f} | {pct:+.1f}% | {sig} |")
    p("")

    # --- Daily detail ---
    p("## Daily Contract Deployments")
    p("")
    p("| Day | Contracts | Deployers |")
    p("|-----|-----------|-----------|")
    for r in daily("SELECT date(detection_timestamp) as day, COUNT(*) as cnt, COUNT(DISTINCT deployer_address) as dep FROM contracts WHERE detection_timestamp >= '2026-03-25' GROUP BY day ORDER BY day"):
        marker = " *" if r['day'] == '2026-04-01' else ""
        p(f"| {r['day']}{marker} | {r['cnt']:,} | {r['dep']:,} |")
    p("")

    # --- Org activity ---
    p("## Org Activity")
    p("")
    for org in ['org_001', 'org_002']:
        rows = daily(f"SELECT date(c.detection_timestamp) as day, COUNT(*) as cnt FROM contracts c JOIN deployers d ON c.deployer_address = d.deployer_address WHERE d.funding_trail LIKE '%{org}%' AND c.detection_timestamp >= '2026-03-25' GROUP BY day ORDER BY day")
        pre_a, post_a, pct = avg_by_period(rows)
        p(f"**{org}:** Pre={pre_a:.1f}/day Post={post_a:.1f}/day Change={pct:+.1f}%")
    p("")

    # --- Dormant activation ---
    p("## Dormant Fleet Activation (20+ staged, zero pre-April activity)")
    p("")
    dormant = daily("""
        SELECT c.deployer_address, COUNT(DISTINCT te.contract_address) as activated,
               MIN(te.timestamp) as first_activity
        FROM transaction_events te
        JOIN contracts c ON te.contract_address = c.contract_address
        WHERE te.timestamp >= '2026-04-01'
        AND c.deployer_address IN (
            SELECT deployer_address FROM contracts GROUP BY deployer_address HAVING COUNT(*) >= 20
        )
        AND c.deployer_address NOT IN (
            SELECT DISTINCT c2.deployer_address FROM transaction_events te2
            JOIN contracts c2 ON te2.contract_address = c2.contract_address
            WHERE te2.timestamp < '2026-04-01'
        )
        GROUP BY c.deployer_address ORDER BY activated DESC
    """)
    p(f"Activated: **{len(dormant)}** deployers")
    for d in dormant[:10]:
        p(f"  {d['deployer_address'][:22]}... activated={d['activated']} first={d['first_activity'][:19]}")
    p("")

    # --- Period-specific measurements ---
    if period in ("week2", "month1"):
        p("## Extended Measurements")
        p("")

        # Undrained approval population growth
        undrained = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected = 0").fetchone()[0]
        p(f"Undrained approval population: **{undrained}** (was 5,355 at day 3 post-Drift)")
        p("")

        # Wallet rotation frequency
        rotations = conn.execute("""
            SELECT COUNT(*) FROM deployer_similarity ds
            JOIN deployers da ON ds.deployer_a = da.deployer_address
            JOIN deployers db ON ds.deployer_b = db.deployer_address
            WHERE ds.composite_score >= 0.85
            AND (db.first_seen > da.last_seen OR da.first_seen > db.last_seen)
        """).fetchone()[0]
        p(f"Wallet rotations (>=0.85 + temporal): **{rotations}** (was 274 at day 3)")
        p("")

    if period == "month1":
        # New bytecode families
        new_families = conn.execute("""
            SELECT COUNT(*) FROM bytecode_families WHERE first_seen >= '2026-04-01'
        """).fetchone()[0]
        p(f"New bytecode families post-Drift: **{new_families}**")
        p("")

        # Camouflage ratio current
        cam = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END) as camo
            FROM (SELECT contract_address, CAST(SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) AS REAL)/COUNT(*) as rr
                  FROM transaction_events GROUP BY contract_address HAVING COUNT(*) >= 10)
        """).fetchone()
        if cam and cam[0]:
            p(f"Camouflage ratio: **{round(100*cam[1]/cam[0],1)}%** (was 79.2% at day 3)")
        p("")

    report = "\n".join(out)
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Written to {path} ({len(out)} lines)")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("week1", "week2", "month1"):
        print("Usage: python measure_drift_impact.py [week1|week2|month1]")
        print("  week1  - Run on April 8")
        print("  week2  - Run on April 15")
        print("  month1 - Run on May 1")
        sys.exit(1)
    run(sys.argv[1])
