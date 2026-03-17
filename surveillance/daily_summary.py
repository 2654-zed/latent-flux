"""
Layer 3 -- Daily summary report.

Queries the database for yesterday's activity without stopping
the running surveillance processes. Safe to run from cron or manually.

Usage:
    python3 surveillance/daily_summary.py              # yesterday
    python3 surveillance/daily_summary.py 2026-03-17   # specific date
    python3 surveillance/daily_summary.py --save       # write to file
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db


def generate_daily_summary(conn, target_date: str) -> str:
    """Generate a summary for a specific UTC date (YYYY-MM-DD)."""
    day_start = f"{target_date}T00:00:00+00:00"
    day_end = f"{target_date}T23:59:59+00:00"
    now = datetime.now(timezone.utc).isoformat()

    # Contracts deployed that day
    day_contracts = conn.execute(
        "SELECT * FROM contracts WHERE detection_timestamp >= ? AND detection_timestamp <= ? ORDER BY detection_block",
        (day_start, day_end),
    ).fetchall()
    day_contracts = [dict(r) for r in day_contracts]

    # By tier
    by_tier = {"unknown": 0, "suspected": 0, "confirmed": 0}
    for c in day_contracts:
        tier = c["confidence_tier"]
        by_tier[tier] = by_tier.get(tier, 0) + 1

    # By detection method
    by_method = {}
    for c in day_contracts:
        m = c["detection_method"]
        by_method[m] = by_method.get(m, 0) + 1

    # Deployers active that day
    deployers_today = set(c["deployer_address"] for c in day_contracts)

    # Top deployers
    deployer_counts = {}
    for c in day_contracts:
        d = c["deployer_address"]
        deployer_counts[d] = deployer_counts.get(d, 0) + 1
    top_deployers = sorted(deployer_counts.items(), key=lambda x: -x[1])[:10]

    # Hourly distribution
    hourly = {}
    for c in day_contracts:
        ts = c["detection_timestamp"]
        try:
            hour = int(ts[11:13])
            hourly[hour] = hourly.get(hour, 0) + 1
        except (ValueError, IndexError):
            pass

    # Trap events that day
    day_events = conn.execute(
        "SELECT * FROM trap_events WHERE timestamp >= ? AND timestamp <= ?",
        (day_start, day_end),
    ).fetchall()
    day_events = [dict(r) for r in day_events]

    # Connection gaps that day
    gaps = conn.execute(
        "SELECT * FROM connection_gaps WHERE disconnect >= ? AND disconnect <= ? ORDER BY disconnect",
        (day_start, day_end),
    ).fetchall()
    gaps = [dict(r) for r in gaps]

    # Cache stats
    cache = db.cache_stats(conn)

    # Cumulative totals
    stats = db.stats(conn)

    # Heartbeat status
    heartbeats = db.read_all_heartbeats(conn)

    # Priority deployers active today
    priority_today = []
    for d in deployers_today:
        if db.is_priority_deployer(conn, d):
            priority_today.append(d)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append(f"LAYER 3 -- DAILY SUMMARY: {target_date}")
    lines.append("=" * 70)
    lines.append(f"Generated: {now}")
    lines.append("")

    # Process health
    lines.append("-" * 70)
    lines.append("PROCESS HEALTH")
    lines.append("-" * 70)
    check_time = datetime.now(timezone.utc)
    for hb in heartbeats:
        ts = datetime.fromisoformat(hb["timestamp"])
        age = (check_time - ts).total_seconds()
        status = "OK" if age < 300 else "STALE (>5min)"
        lines.append(f"  {hb['component']}: last heartbeat {int(age)}s ago [{status}]")
    if not heartbeats:
        lines.append("  No heartbeats -- processes may not be running")
    if gaps:
        lines.append(f"  Connection gaps today: {len(gaps)}")
        total_gap_seconds = 0
        for g in gaps:
            if g["reconnect"]:
                d = datetime.fromisoformat(g["disconnect"])
                r = datetime.fromisoformat(g["reconnect"])
                total_gap_seconds += (r - d).total_seconds()
        if total_gap_seconds > 0:
            lines.append(f"  Total gap time: {int(total_gap_seconds)}s ({total_gap_seconds/3600:.1f}h)")
    else:
        lines.append("  Connection gaps today: 0")
    lines.append("")

    # Day's activity
    lines.append("-" * 70)
    lines.append("TODAY'S ACTIVITY")
    lines.append("-" * 70)
    lines.append(f"New contracts:     {len(day_contracts)}")
    lines.append(f"  UNKNOWN:         {by_tier.get('unknown', 0)}")
    lines.append(f"  SUSPECTED:       {by_tier.get('suspected', 0)}")
    lines.append(f"  CONFIRMED:       {by_tier.get('confirmed', 0)}")
    lines.append(f"Active deployers:  {len(deployers_today)}")
    lines.append(f"Trap events:       {len(day_events)}")
    lines.append("")

    # Detection methods
    if by_method:
        lines.append("Detection methods:")
        for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
            lines.append(f"  {method}: {count}")
        lines.append("")

    # Hourly distribution
    if hourly:
        lines.append("-" * 70)
        lines.append("HOURLY DISTRIBUTION (UTC)")
        lines.append("-" * 70)
        max_h = max(hourly.values()) if hourly else 1
        for hour in range(24):
            count = hourly.get(hour, 0)
            bar = "#" * int(count / max_h * 30) if max_h > 0 and count > 0 else ""
            lines.append(f"  {hour:02d}:00  {bar:30s}  {count}")
        if hourly:
            peak = max(hourly, key=hourly.get)
            lines.append(f"  Peak: {peak:02d}:00 UTC ({hourly[peak]} deployments)")
        lines.append("")

    # Priority deployer activity
    if priority_today:
        lines.append("-" * 70)
        lines.append(f"PRIORITY DEPLOYERS ACTIVE TODAY ({len(priority_today)})")
        lines.append("-" * 70)
        for d in priority_today:
            count = deployer_counts.get(d, 0)
            rec = db.get_deployer(conn, d)
            notes = (rec["deployment_pattern_notes"] or "")[:80] if rec else ""
            lines.append(f"  {d}  ({count} new contracts)")
            if notes:
                lines.append(f"    {notes}")
        lines.append("")

    # Top deployers
    if top_deployers and top_deployers[0][1] > 1:
        lines.append("-" * 70)
        lines.append("TOP DEPLOYERS TODAY")
        lines.append("-" * 70)
        for addr, count in top_deployers:
            if count < 2:
                break
            flag = " [PRIORITY]" if db.is_priority_deployer(conn, addr) else ""
            lines.append(f"  {addr}  ({count} contracts){flag}")
        lines.append("")

    # Trap events detail
    if day_events:
        lines.append("-" * 70)
        lines.append(f"TRAP EVENTS ({len(day_events)})")
        lines.append("-" * 70)
        for e in day_events:
            lines.append(f"  Contract: {e['trap_contract_address']}")
            lines.append(f"  Bot:      {e['bot_address']}")
            lines.append(f"  TX:       {e['tx_hash']}")
            lines.append(f"  Loss:     ${e['loss_estimate_usd'] or 'unknown'}")
            lines.append(f"  Sig:      {e['failure_signature']}")
            lines.append("")

    # Connection gaps detail
    if gaps:
        lines.append("-" * 70)
        lines.append(f"CONNECTION GAPS ({len(gaps)})")
        lines.append("-" * 70)
        for g in gaps:
            status = "CLOSED" if g["reconnect"] else "OPEN"
            lines.append(f"  [{status}] {g['component']}")
            lines.append(f"    disconnect: {g['disconnect']}")
            lines.append(f"    reconnect:  {g['reconnect'] or 'STILL DOWN'}")
            lines.append(f"    blocks: {g['last_block']} -> {g['first_block'] or '?'}")
            lines.append(f"    reason: {g['reason']}")
        lines.append("")

    # Cumulative
    lines.append("-" * 70)
    lines.append("CUMULATIVE TOTALS")
    lines.append("-" * 70)
    lines.append(f"Total contracts:   {stats['contracts_total']}")
    lines.append(f"  UNKNOWN:         {stats['contracts_by_tier']['unknown']}")
    lines.append(f"  SUSPECTED:       {stats['contracts_by_tier']['suspected']}")
    lines.append(f"  CONFIRMED:       {stats['contracts_by_tier']['confirmed']}")
    lines.append(f"Total deployers:   {stats['deployers_total']}")
    lines.append(f"Cache entries:     {cache['entries']}")
    lines.append(f"Cache all-time hits: {cache['total_hits']}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF DAILY SUMMARY")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Layer 3 -- Daily summary (queries DB, does not stop processes)"
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        help="Date to summarize (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to surveillance/data/daily_YYYY-MM-DD.txt",
    )
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        target_date = yesterday.strftime("%Y-%m-%d")

    conn = db.init_db()
    report = generate_daily_summary(conn, target_date)
    print(report)

    if args.save:
        out = Path(__file__).parent / "data" / f"daily_{target_date}.txt"
        out.write_text(report, encoding="utf-8")
        print(f"\nSaved to {out}")

    conn.close()


if __name__ == "__main__":
    main()
