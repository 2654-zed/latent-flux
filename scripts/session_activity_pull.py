"""Pull notable activity from Railway DB since 2026-04-17 18:00 UTC.
Run via: railway ssh 'python scripts/session_activity_pull.py'"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

DB = "/app/surveillance/data/surveillance.db"
CUTOFF = "2026-04-17T18:00:00"


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    print(f"=== corpus since {CUTOFF} UTC ===\n")

    # 1. Alert volume by type
    print("--- Alerts by type ---")
    rows = c.execute(
        "SELECT alert_type, COUNT(*) FROM alerts WHERE timestamp >= ? "
        "GROUP BY alert_type ORDER BY 2 DESC LIMIT 15",
        (CUTOFF,),
    ).fetchall()
    total = 0
    for r in rows:
        print(f"  {r[0]:38s}  {r[1]:>6}")
        total += r[1]
    print(f"  {'TOTAL':38s}  {total:>6}")

    # 2. Trap events
    print("\n--- Trap events (behavioral confirmations) ---")
    n = c.execute(
        "SELECT COUNT(*) FROM trap_events WHERE timestamp >= ?", (CUTOFF,)
    ).fetchone()[0]
    print(f"  count: {n}")
    for r in c.execute(
        "SELECT trap_contract_address, bot_address, tx_hash, loss_estimate_usd, "
        "failure_signature, block_number FROM trap_events WHERE timestamp >= ? "
        "ORDER BY block_number DESC LIMIT 5",
        (CUTOFF,),
    ):
        loss = f"${r[3]:,.2f}" if r[3] else "n/a"
        sig = (r[4] or "")[:40]
        print(f"  block={r[5]:>10}  trap={r[0][:18]}...  bot={r[1][:18]}...  loss={loss}  sig={sig}")

    # 3. Contracts promoted to confirmed
    print("\n--- Contracts promoted to confirmed in window ---")
    n = c.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'confirmed' "
        "AND last_updated >= ?",
        (CUTOFF,),
    ).fetchone()[0]
    print(f"  count: {n}")
    for r in c.execute(
        "SELECT contract_address, chain, confirmation_tx_hash, "
        "confidence_reason, last_updated FROM contracts "
        "WHERE confidence_tier = 'confirmed' AND last_updated >= ? "
        "ORDER BY last_updated DESC LIMIT 5",
        (CUTOFF,),
    ):
        reason = (r[3] or "")[:60]
        print(f"  {r[1]:10s}  {r[0]}  {r[4][:19]}  {reason}...")

    # 4. High-velocity deployers
    print("\n--- High-velocity deployers ---")
    vel_rows = c.execute(
        "SELECT address, payload, timestamp FROM alerts "
        "WHERE alert_type = 'HIGH_VELOCITY_DEPLOYER' AND timestamp >= ? "
        "ORDER BY timestamp DESC LIMIT 10",
        (CUTOFF,),
    ).fetchall()
    if not vel_rows:
        print("  (none)")
    for r in vel_rows:
        try:
            p = json.loads(r["payload"] or "{}")
            ct = p.get("deployment_count") or p.get("velocity") or p.get("count") or "?"
        except Exception:
            ct = "?"
        print(f"  {r[0][:18]}...  deployments={ct}  ts={r[2][:19]}")

    # 5. Drainer / x402 activity
    print("\n--- Drainer/x402 activity ---")
    rows = c.execute(
        "SELECT alert_type, COUNT(*) FROM alerts WHERE timestamp >= ? "
        "AND (alert_type LIKE '%DRAIN%' OR alert_type LIKE 'X402%') "
        "GROUP BY alert_type ORDER BY 2 DESC",
        (CUTOFF,),
    ).fetchall()
    if not rows:
        print("  (none)")
    for r in rows:
        print(f"  {r[0]:30s}  {r[1]:>4}")

    # Drain addresses seen
    for r in c.execute(
        "SELECT address, COUNT(*) FROM alerts WHERE timestamp >= ? "
        "AND alert_type LIKE 'X402_AGENT_DRAIN' GROUP BY address ORDER BY 2 DESC LIMIT 5",
        (CUTOFF,),
    ):
        print(f"  drain-addr {r[0]}  n={r[1]}")

    # 6. Analysis scheduler freshness probe
    print("\n--- Analysis scheduler output freshness ---")
    def probe(tbl, col):
        try:
            r = c.execute(f"SELECT MAX({col}), COUNT(*) FROM {tbl}").fetchone()
            return r[0], r[1]
        except sqlite3.Error as e:
            return f"err:{e}", 0
    for tbl, col in [
        ("daily_metrics", "date"),
        ("camouflage_metrics", "date"),
        ("predictions", "issued_date"),
        ("trust_amplification", "last_updated"),
        ("deployer_profiles", "profiled_at"),
        ("deployer_similarity", "computed_at"),
        ("bytecode_families", "last_updated"),
    ]:
        latest, rows_n = probe(tbl, col)
        print(f"  {tbl:28s}  rows={rows_n:>8,}  latest={latest}")

    # 7. Deployment velocity
    print("\n--- Deployments in window by chain ---")
    for r in c.execute(
        "SELECT chain, COUNT(*) FROM contracts WHERE detection_timestamp >= ? "
        "GROUP BY chain ORDER BY 2 DESC",
        (CUTOFF,),
    ):
        print(f"  {r[0]:10s}  {r[1]:>6,}")

    # 8. Heartbeat freshness
    print("\n--- Heartbeat freshness ---")
    for r in c.execute("SELECT component, timestamp, blocks, deployments FROM heartbeat"):
        print(f"  {r[0]:38s}  last={r[1][:19]}  blk={r[2]:>8,}  dep={r[3]:>5,}")

    # 9. Analysis scheduler log artifacts — check for [analysis_scheduler] pattern
    # (would be in Railway logs, not DB; skipped here)


if __name__ == "__main__":
    main()
