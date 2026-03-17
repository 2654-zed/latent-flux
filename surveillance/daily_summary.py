"""
Layer 3 -- Daily summary report.

Queries the database and prints a readable report without stopping
or touching the main surveillance process. No writes. Read-only.

Usage:
    python3 -m surveillance.daily_summary
    python3 -m surveillance.daily_summary --save
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db

DB_PATH = db.DEFAULT_DB_PATH


def _ago(iso_ts: str) -> str:
    """Human-readable time-ago string from an ISO timestamp."""
    try:
        ts = datetime.fromisoformat(iso_ts)
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "unknown"


def generate_summary(conn) -> str:
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    cutoff = (now - timedelta(hours=24)).isoformat()

    stats = db.stats(conn)

    # --- System health ---
    heartbeats = db.read_all_heartbeats(conn)
    hb_lines = []
    if heartbeats:
        for hb in heartbeats:
            ts = datetime.fromisoformat(hb["timestamp"])
            age_s = int((now - ts).total_seconds())
            status = "LIVE" if age_s < 300 else "STALE"
            hb_lines.append(
                f"  {hb['component']:24s} {_ago(hb['timestamp']):12s} "
                f"({hb['timestamp'][:19]})  [{status}]"
            )
    else:
        hb_lines.append("  No heartbeats recorded -- processes may not be running")

    # --- Last 24h contracts ---
    recent_contracts = conn.execute(
        "SELECT * FROM contracts WHERE detection_timestamp >= ? ORDER BY detection_block DESC",
        (cutoff,),
    ).fetchall()
    recent_contracts = [dict(r) for r in recent_contracts]

    recent_suspected = [c for c in recent_contracts if c["confidence_tier"] == "suspected"]
    recent_confirmed = [c for c in recent_contracts if c["confidence_tier"] == "confirmed"]

    recent_deployers = set(c["deployer_address"] for c in recent_contracts)
    all_deployers_before = conn.execute(
        "SELECT DISTINCT deployer_address FROM contracts WHERE detection_timestamp < ?",
        (cutoff,),
    ).fetchall()
    old_deployers = {r[0] for r in all_deployers_before}
    new_deployers = recent_deployers - old_deployers

    # --- Priority deployers ---
    priority_rows = conn.execute(
        "SELECT * FROM deployers WHERE deployment_pattern_notes IS NOT NULL AND deployment_pattern_notes != ''"
    ).fetchall()
    priority_deployers = [dict(r) for r in priority_rows]

    # --- Top new suspected (last 24h) ---
    top_suspected = recent_suspected[:10]

    # --- All confirmed (all time) ---
    all_confirmed = db.get_contracts_by_tier(conn, "confirmed")

    # --- Connection gaps (last 24h) ---
    gaps = db.get_connection_gaps(conn, since=cutoff)

    # --- Cache ---
    cache = db.cache_stats(conn)

    # === Build output ===
    L = []
    w = 55

    L.append("")
    L.append("=" * w)
    L.append("  SURVEILLANCE DAILY SUMMARY")
    L.append(f"  Generated: {now_str}")
    L.append("=" * w)
    L.append("")

    # System health
    L.append("SYSTEM HEALTH")
    for line in hb_lines:
        L.append(line)
    if gaps:
        open_gaps = [g for g in gaps if g["reconnect"] is None]
        closed_gaps = [g for g in gaps if g["reconnect"] is not None]
        L.append(f"  Reconnects (24h):      {len(closed_gaps)}")
        if open_gaps:
            L.append(f"  OPEN GAPS:             {len(open_gaps)}")
            for g in open_gaps:
                L.append(f"    {g['component']} down since {g['disconnect'][:19]}")
                L.append(f"    reason: {g['reason']}")
    L.append("")

    # Corpus totals
    L.append("CORPUS TOTALS")
    L.append(f"  Total contracts:       {stats['contracts_total']}")
    L.append(f"  Total deployers:       {stats['deployers_total']}")
    L.append(f"  UNKNOWN:               {stats['contracts_by_tier']['unknown']}")
    L.append(f"  SUSPECTED:             {stats['contracts_by_tier']['suspected']}")
    L.append(f"  CONFIRMED:             {stats['contracts_by_tier']['confirmed']}")
    L.append(f"  Cache entries:         {cache['entries']}")
    L.append(f"  Cache hits (all time): {cache['total_hits']}")
    L.append("")

    # Last 24 hours
    L.append("LAST 24 HOURS")
    L.append(f"  New contracts:         {len(recent_contracts)}")
    L.append(f"  New deployers:         {len(new_deployers)}")
    L.append(f"  New SUSPECTED:         {len(recent_suspected)}")
    conf_line = f"  New CONFIRMED:         {len(recent_confirmed)}"
    if recent_confirmed:
        conf_line += "  <<<<<"
    L.append(conf_line)
    L.append("")

    # Priority deployers
    if priority_deployers:
        L.append(f"PRIORITY DEPLOYERS ({len(priority_deployers)})")
        for d in priority_deployers:
            addr = d["deployer_address"]
            total = d["total_contracts_deployed"]
            last = _ago(d["last_seen"])
            # Count recent activity
            recent_count = sum(
                1 for c in recent_contracts if c["deployer_address"] == addr
            )
            activity = f"{recent_count} new" if recent_count > 0 else "inactive"
            L.append(
                f"  {addr[:10]}...   last seen: {last:8s}  "
                f"total: {total:3d}   24h: {activity}"
            )
        L.append("")

    # Top new suspected
    if top_suspected:
        L.append(f"TOP NEW SUSPECTED (last 24h, showing {len(top_suspected)})")
        for c in top_suspected:
            method = c["detection_method"]
            deployer = c["deployer_address"]
            # Extract short pattern name from reason
            reason = c["confidence_reason"]
            if "[" in reason and "]" in reason:
                pattern = reason[reason.index("[") + 1:reason.index("]")]
            elif "Auto-SUSPECTED" in reason:
                pattern = "deployer_history"
            elif "velocity" in reason.lower():
                pattern = "velocity"
            else:
                pattern = method
            L.append(
                f"  {c['contract_address'][:10]}...   "
                f"{pattern:24s} deployer: {deployer[:10]}..."
            )
        L.append("")

    # Confirmed events
    L.append("CONFIRMED EVENTS (all time)")
    if all_confirmed:
        for c in all_confirmed:
            L.append(f"  {c['contract_address']}")
            L.append(f"    deployer:   {c['deployer_address']}")
            L.append(f"    confirmed:  {c['confirmation_timestamp']}")
            L.append(f"    tx:         {c['confirmation_tx_hash']}")
            reason = c["confidence_reason"]
            if len(reason) > 80:
                reason = reason[:80] + "..."
            L.append(f"    reason:     {reason}")
            # Trap events
            events = db.get_trap_events_for_contract(conn, c["contract_address"])
            for e in events:
                L.append(
                    f"    event: bot={e['bot_address'][:10]}... "
                    f"loss=${e['loss_estimate_usd'] or '?'} "
                    f"sig={e['failure_signature'][:40]}"
                )
    else:
        L.append("  None yet.")
    L.append("")

    L.append("-" * w)
    L.append(f"Read from: {DB_PATH}")
    L.append("No writes. Safe to run while monitor is live.")
    L.append("")

    return "\n".join(L)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer 3 -- Daily summary (read-only, safe while monitor runs)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also save to surveillance/data/daily_YYYY-MM-DD.txt",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = db.init_db()
    report = generate_summary(conn)
    print(report)

    if args.save:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(__file__).parent / "data" / f"daily_{today}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Saved to {out}")

    conn.close()


if __name__ == "__main__":
    main()
