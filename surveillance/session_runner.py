"""
Layer 3 -- Controlled session runner.

Runs both the deployment monitor and routing monitor for a fixed duration,
then generates a session report. If either component crashes, both are
stopped and the report is generated immediately.

Usage:
    python3.13 surveillance/session_runner.py --duration 3600
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db
from surveillance.deployment_monitor import DeploymentMonitor
from surveillance.bytecode_classifier import classify
from surveillance.routing_monitor import RoutingMonitor

logger = logging.getLogger("surveillance.session")

DATA_DIR = Path(__file__).parent / "data"


def generate_report(conn, duration_seconds: int,
                    monitor: DeploymentMonitor,
                    router: RoutingMonitor,
                    crash_info: str | None = None,
                    session_start_block: int = 0,
                    safe_deployers: set[str] | None = None) -> str:
    """Generate the session report as a string."""
    now = datetime.now(timezone.utc)
    stats = db.stats(conn)

    # Contracts by tier
    unknown_contracts = db.get_contracts_by_tier(conn, "unknown")
    suspected_contracts = db.get_contracts_by_tier(conn, "suspected")
    confirmed_contracts = db.get_contracts_by_tier(conn, "confirmed")

    # Routing presence
    all_contracts = unknown_contracts + suspected_contracts + confirmed_contracts
    routed = [c for c in all_contracts if c["routing_presence"]]

    # Confidence upgrades: confirmed contracts that were originally detected
    # by bytecode_pattern (meaning they started as unknown/suspected and
    # got upgraded via trap event)
    upgrades = [
        c for c in confirmed_contracts
        if c["detection_method"] == "bytecode_pattern"
        and c["confirmation_tx_hash"] is not None
    ]

    # Routing anomaly enrichments
    anomaly_enriched = [
        c for c in all_contracts
        if c["confidence_reason"] and "ROUTING ANOMALY" in c["confidence_reason"]
    ]

    # Top deployers by contract count
    deployer_counts = {}
    for c in all_contracts:
        dep = c["deployer_address"]
        deployer_counts[dep] = deployer_counts.get(dep, 0) + 1
    top_deployers = sorted(deployer_counts.items(), key=lambda x: -x[1])[:10]

    # Returning deployers (active in both this session and a prior one)
    returning = db.get_returning_deployers(conn, session_start_block) if session_start_block > 0 else []

    # Deployer-history auto-suspects (this session)
    deployer_history_suspects = [
        c for c in all_contracts
        if c["detection_method"] == "deployer_history"
        and c["detection_block"] >= session_start_block
    ] if session_start_block > 0 else []

    # Contracts skipped via safe deployers
    safe_deployers = safe_deployers or set()

    # Priority deployer contribution: what % of SUSPECTED came from priority deployers?
    priority_suspected = [
        c for c in suspected_contracts
        if db.is_priority_deployer(conn, c["deployer_address"])
    ]
    priority_pct = (
        (len(priority_suspected) / len(suspected_contracts) * 100)
        if suspected_contracts else 0.0
    )

    # Velocity-flagged deployers (from monitor)
    velocity_flagged = monitor.velocity_flagged if hasattr(monitor, "velocity_flagged") else set()

    # Time-of-day distribution: deployments per UTC hour (this session only)
    hourly_dist: dict[int, int] = {}
    if session_start_block > 0:
        session_contracts = [
            c for c in all_contracts if c["detection_block"] >= session_start_block
        ]
        for c in session_contracts:
            ts = c.get("detection_timestamp", "")
            if len(ts) >= 13:  # ISO 8601 has hour at index 11-12
                try:
                    hour = int(ts[11:13])
                    hourly_dist[hour] = hourly_dist.get(hour, 0) + 1
                except ValueError:
                    pass

    # Trap events
    trap_events_total = stats["trap_events_total"]
    top_bots = db.get_most_trapped_bots(conn, limit=5)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("LAYER 3 -- SESSION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Generated:          {now.isoformat()}")
    lines.append(f"Session duration:   {duration_seconds}s ({duration_seconds // 60}m {duration_seconds % 60}s)")
    if crash_info:
        lines.append(f"Session status:     CRASHED")
        lines.append(f"Crash reason:       {crash_info}")
    else:
        lines.append(f"Session status:     COMPLETED NORMALLY")
    lines.append("")

    lines.append("-" * 70)
    lines.append("MONITOR STATS")
    lines.append("-" * 70)
    lines.append(f"Blocks processed:       {monitor.blocks_processed}")
    lines.append(f"Deployments captured:   {monitor.deployments_found}")
    lines.append(f"Routing poll cycles:    {router.cycles}")
    lines.append(f"Routing registry hits:  {router.routing_hits}")
    lines.append(f"Routing anomalies:      {router.anomalies_found}")
    lines.append(f"Selector events:        {monitor.selector_events}")
    lines.append(f"Bots tagged:            {monitor.bots_tagged}")
    lines.append(f"Bot candidates:         {monitor.bot_candidates_found}")
    lines.append(f"Bot+deployer hits:      {monitor.bot_deployer_hits}")
    lines.append("")

    # Bytecode cache stats
    cache_total = monitor.cache_hits_init + monitor.cache_hits_deployed + monitor.cache_misses
    cache_hit_rate = (
        (monitor.cache_hits_init + monitor.cache_hits_deployed) / cache_total * 100
        if cache_total > 0 else 0.0
    )
    cache_db = db.cache_stats(conn)
    lines.append("-" * 70)
    lines.append("BYTECODE CACHE")
    lines.append("-" * 70)
    lines.append(f"Init code hits (skip RPC):  {monitor.cache_hits_init}")
    lines.append(f"Deployed code hits:         {monitor.cache_hits_deployed}")
    lines.append(f"Misses (full classify):     {monitor.cache_misses}")
    lines.append(f"Hit rate:                   {cache_hit_rate:.1f}%")
    lines.append(f"eth_getCode calls saved:    {monitor.rpc_calls_saved}")
    lines.append(f"Cache entries (all time):   {cache_db['entries']}")
    lines.append(f"Cache hits (all time):      {cache_db['total_hits']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("DATABASE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total contracts:        {stats['contracts_total']}")
    lines.append(f"  UNKNOWN:              {stats['contracts_by_tier']['unknown']}")
    lines.append(f"  SUSPECTED:            {stats['contracts_by_tier']['suspected']}")
    lines.append(f"  CONFIRMED:            {stats['contracts_by_tier']['confirmed']}")
    lines.append(f"Total deployers:        {stats['deployers_total']}")
    lines.append(f"Total trap events:      {trap_events_total}")
    lines.append(f"Routing presence:       {len(routed)} of {len(all_contracts)} contracts")
    lines.append(f"Confidence upgrades:    {len(upgrades)}")
    lines.append(f"Routing anomalies:      {len(anomaly_enriched)}")
    lines.append(f"Deployer-history auto:  {len(deployer_history_suspects)}")
    lines.append(f"Velocity-flagged:       {len(velocity_flagged)} deployers")
    lines.append(f"Safe deployers skipped: {len(safe_deployers)} addresses")
    lines.append(f"Returning deployers:    {len(returning)}")
    lines.append(f"Priority % of SUSPECTED:{priority_pct:5.1f}% ({len(priority_suspected)}/{len(suspected_contracts)})")
    lines.append("")

    if returning:
        lines.append("-" * 70)
        lines.append(f"RETURNING DEPLOYERS ({len(returning)})")
        lines.append("-" * 70)
        for r in returning:
            flag = " [PRIORITY]" if r["deployment_pattern_notes"] else ""
            lines.append(
                f"  {r['deployer_address']}{flag}"
            )
            lines.append(
                f"    prev sessions: {r['prev_count']} contracts, "
                f"this session: {r['new_count']} contracts"
            )
        lines.append("")

    if velocity_flagged:
        lines.append("-" * 70)
        lines.append(f"VELOCITY-FLAGGED DEPLOYERS ({len(velocity_flagged)})")
        lines.append("-" * 70)
        for addr in velocity_flagged:
            dep_rec = db.get_deployer(conn, addr)
            count = dep_rec["total_contracts_deployed"] if dep_rec else "?"
            lines.append(f"  {addr}  ({count} total contracts)")
            if dep_rec and dep_rec["deployment_pattern_notes"]:
                notes = dep_rec["deployment_pattern_notes"]
                if len(notes) > 200:
                    notes = notes[:200] + "..."
                lines.append(f"    notes: {notes}")
        lines.append("")

    if hourly_dist:
        lines.append("-" * 70)
        lines.append("DEPLOYMENT TIME-OF-DAY (UTC, this session)")
        lines.append("-" * 70)
        max_count = max(hourly_dist.values()) if hourly_dist else 1
        for hour in range(24):
            count = hourly_dist.get(hour, 0)
            if count == 0 and hour not in range(
                min(hourly_dist.keys(), default=0),
                max(hourly_dist.keys(), default=23) + 1,
            ):
                continue
            bar = "#" * int(count / max_count * 30) if max_count > 0 else ""
            lines.append(f"  {hour:02d}:00  {bar:30s}  {count}")
        total_session = sum(hourly_dist.values())
        lines.append(f"  Total this session: {total_session}")
        if total_session > 0:
            peak_hour = max(hourly_dist, key=hourly_dist.get)
            lines.append(f"  Peak hour: {peak_hour:02d}:00 UTC ({hourly_dist[peak_hour]} deployments)")
        lines.append("")

    if deployer_history_suspects:
        lines.append("-" * 70)
        lines.append(f"AUTO-SUSPECTED VIA DEPLOYER HISTORY ({len(deployer_history_suspects)})")
        lines.append("-" * 70)
        for c in deployer_history_suspects[:20]:
            lines.append(f"  {c['contract_address']}")
            lines.append(f"    deployer: {c['deployer_address']}")
            lines.append(f"    block:    {c['detection_block']}")
        lines.append("")

    if suspected_contracts:
        lines.append("-" * 70)
        lines.append(f"SUSPECTED CONTRACTS ({len(suspected_contracts)})")
        lines.append("-" * 70)
        for c in suspected_contracts[:20]:
            lines.append(f"  {c['contract_address']}")
            lines.append(f"    deployer:  {c['deployer_address']}")
            lines.append(f"    block:     {c['detection_block']}")
            lines.append(f"    routing:   {'YES' if c['routing_presence'] else 'no'}")
            reason = c["confidence_reason"]
            if len(reason) > 200:
                reason = reason[:200] + "..."
            lines.append(f"    reason:    {reason}")
            lines.append("")

    if confirmed_contracts:
        lines.append("-" * 70)
        lines.append(f"CONFIRMED CONTRACTS ({len(confirmed_contracts)})")
        lines.append("-" * 70)
        for c in confirmed_contracts[:20]:
            lines.append(f"  {c['contract_address']}")
            lines.append(f"    deployer:  {c['deployer_address']}")
            lines.append(f"    conf. tx:  {c['confirmation_tx_hash']}")
            reason = c["confidence_reason"]
            if len(reason) > 200:
                reason = reason[:200] + "..."
            lines.append(f"    reason:    {reason}")
            lines.append("")

    if top_deployers and top_deployers[0][1] > 1:
        lines.append("-" * 70)
        lines.append("TOP DEPLOYERS (by contract count)")
        lines.append("-" * 70)
        for addr, count in top_deployers:
            if count < 2:
                break
            lines.append(f"  {addr}  ({count} contracts)")
        lines.append("")

    if top_bots:
        lines.append("-" * 70)
        lines.append("MOST TRAPPED BOTS")
        lines.append("-" * 70)
        for b in top_bots:
            lines.append(
                f"  {b['bot_address']}  "
                f"({b['trap_count']} traps, ${b['total_loss_usd']:.2f} total loss)"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


async def run_session(duration_seconds: int,
                      safe_deployers: set[str] | None = None):
    """Run both components for the specified duration."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.init_db()

    # Record session start block for returning-deployer tracking
    session_start_block = db.get_max_block(conn) + 1
    logger.info("Session start block boundary: %d", session_start_block)

    rpc_url = os.environ.get("ARB_WSS_URL")
    api_key = os.environ.get("ONEINCH_API_KEY")

    if not rpc_url:
        print("ERROR: ARB_WSS_URL not set", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("ERROR: ONEINCH_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    safe_deployers = safe_deployers or set()
    if safe_deployers:
        logger.info("Safe deployers (skipped): %d addresses", len(safe_deployers))

    monitor = DeploymentMonitor(rpc_url, conn, classifier=classify,
                                safe_deployers=safe_deployers)
    router = RoutingMonitor(api_key, conn, poll_interval=300)

    crash_info = None
    actual_duration = duration_seconds

    start_time = asyncio.get_event_loop().time()

    async def run_monitor():
        await monitor.start()

    async def run_router():
        await router.start()

    async def timer():
        await asyncio.sleep(duration_seconds)
        logger.info("Session timer expired (%ds) -- stopping both components",
                     duration_seconds)
        monitor.stop()
        router.stop()

    monitor_task = asyncio.create_task(run_monitor())
    router_task = asyncio.create_task(run_router())
    timer_task = asyncio.create_task(timer())

    # Wait for any task to complete (crash or timer)
    done, pending = await asyncio.wait(
        [monitor_task, router_task, timer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    elapsed = asyncio.get_event_loop().time() - start_time
    actual_duration = int(elapsed)

    # Check if a component crashed (not the timer)
    for task in done:
        if task is timer_task:
            # Normal completion -- timer fired
            continue
        ex = task.exception()
        if ex is not None:
            name = "deployment_monitor" if task is monitor_task else "routing_monitor"
            crash_info = f"{name}: {type(ex).__name__}: {ex}"
            logger.error("Component crashed: %s", crash_info)
            logger.error("Traceback:\n%s",
                         "".join(traceback.format_exception(ex)))

    # Stop everything
    monitor.stop()
    router.stop()
    timer_task.cancel()

    # Give pending tasks a moment to clean up
    for task in pending:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=10)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass

    # Generate report
    logger.info("Generating session report...")
    report = generate_report(conn, actual_duration, monitor, router, crash_info,
                             session_start_block=session_start_block,
                             safe_deployers=safe_deployers)

    # Save report
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = DATA_DIR / f"session_report_{ts}.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    # Also print to stdout
    print(report)

    conn.close()
    return report_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer 3 -- Controlled session runner"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Session duration in seconds (default: 3600 = 60 minutes)",
    )
    parser.add_argument(
        "--safe-deployers",
        nargs="*",
        default=[],
        help="Deployer addresses to skip (known legitimate factories)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load .env if present
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    safe = {addr.lower() for addr in args.safe_deployers}
    if safe:
        logger.info("Safe deployers: %s", safe)

    logger.info("Starting %d-second session (%dm)",
                args.duration, args.duration // 60)

    try:
        report_path = asyncio.run(run_session(args.duration, safe_deployers=safe))
        logger.info("Session complete. Report: %s", report_path)
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")


if __name__ == "__main__":
    main()
