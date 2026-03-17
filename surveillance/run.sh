#!/usr/bin/env bash
# Layer 3 — Start surveillance system as background processes.
# Both components share the same SQLite DB and .env config.
#
# Usage:
#   ./surveillance/run.sh          # start both
#   ./surveillance/run.sh stop     # stop both
#   ./surveillance/run.sh status   # check heartbeats
#
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
LOGDIR="$DIR/logs"
PIDDIR="$DIR/pids"

mkdir -p "$LOGDIR" "$PIDDIR"

# Load .env
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

case "${1:-start}" in

start)
    echo "Starting Layer 3 surveillance system..."

    # Check for existing processes
    for comp in monitor routing; do
        pidfile="$PIDDIR/$comp.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "  $comp already running (PID $(cat "$pidfile")). Stop first."
            exit 1
        fi
    done

    # Deployment monitor
    nohup python3 -m surveillance.deployment_monitor \
        --log-level INFO \
        > "$LOGDIR/monitor.log" 2>&1 &
    echo $! > "$PIDDIR/monitor.pid"
    echo "  deployment_monitor started (PID $!, log: $LOGDIR/monitor.log)"

    # Routing monitor
    nohup python3 -m surveillance.routing_monitor \
        --log-level INFO \
        > "$LOGDIR/routing.log" 2>&1 &
    echo $! > "$PIDDIR/routing.pid"
    echo "  routing_monitor started (PID $!, log: $LOGDIR/routing.log)"

    echo "Both components running. Check status: $0 status"
    ;;

stop)
    echo "Stopping Layer 3 surveillance system..."
    for comp in monitor routing; do
        pidfile="$PIDDIR/$comp.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "  $comp stopped (PID $pid)"
            else
                echo "  $comp not running (stale PID $pid)"
            fi
            rm -f "$pidfile"
        else
            echo "  $comp: no PID file"
        fi
    done
    ;;

status)
    echo "Layer 3 surveillance status:"
    echo ""

    # Process status
    for comp in monitor routing; do
        pidfile="$PIDDIR/$comp.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "  $comp: RUNNING (PID $(cat "$pidfile"))"
        else
            echo "  $comp: STOPPED"
        fi
    done
    echo ""

    # Heartbeat check
    python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from surveillance import db
from datetime import datetime, timezone

conn = db.init_db()
heartbeats = db.read_all_heartbeats(conn)
now = datetime.now(timezone.utc)

if not heartbeats:
    print('  No heartbeats recorded yet.')
else:
    for hb in heartbeats:
        ts = datetime.fromisoformat(hb['timestamp'])
        age = (now - ts).total_seconds()
        status = 'OK' if age < 300 else 'STALE (>5min)'
        print(f'  {hb[\"component\"]}: last={hb[\"timestamp\"]} age={int(age)}s [{status}]')
        print(f'    blocks={hb[\"blocks\"]} deployments={hb[\"deployments\"]}')

# Recent gaps
gaps = db.get_connection_gaps(conn)
open_gaps = [g for g in gaps if g['reconnect'] is None]
if open_gaps:
    print()
    print(f'  OPEN CONNECTION GAPS: {len(open_gaps)}')
    for g in open_gaps:
        print(f'    {g[\"component\"]} disconnected at {g[\"disconnect\"]}')
        print(f'    reason: {g[\"reason\"]}')

conn.close()
"
    ;;

logs)
    # Tail both logs
    tail -f "$LOGDIR/monitor.log" "$LOGDIR/routing.log"
    ;;

*)
    echo "Usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
