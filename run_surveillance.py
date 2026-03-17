import json
import os
import sqlite3
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

DB_PATH = "/app/surveillance/data/surveillance.db"
PORT = int(os.environ.get("PORT", 8080))


def _query(sql, fetchone=False):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        result = cur.execute(sql).fetchone() if fetchone else cur.execute(sql).fetchall()
    finally:
        con.close()
    return result


class StatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stats":
            stats = {
                "contracts": dict(
                    _query("SELECT confidence_tier, COUNT(*) FROM contracts GROUP BY confidence_tier")
                ),
                "deployers": _query("SELECT COUNT(*) FROM deployers", fetchone=True)[0],
                "trap_events": _query("SELECT COUNT(*) FROM trap_events", fetchone=True)[0],
                "tx_events": _query("SELECT COUNT(*) FROM transaction_events", fetchone=True)[0],
                "bot_candidates": _query("SELECT COUNT(*) FROM bot_candidates", fetchone=True)[0],
                "bot_deployer_hits": _query(
                    "SELECT COUNT(*) FROM bot_candidates WHERE is_deployer = 1", fetchone=True
                )[0],
                "last_heartbeat": _query(
                    "SELECT component, timestamp FROM heartbeat ORDER BY timestamp DESC LIMIT 1",
                    fetchone=True,
                ),
                "cache": dict(
                    zip(
                        ["entries", "total_hits"],
                        _query(
                            "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM bytecode_cache",
                            fetchone=True,
                        ),
                    )
                ),
            }
            self._json(200, stats)

        elif self.path == "/suspected":
            rows = _query(
                """SELECT contract_address, deployer_address, detection_method,
                          confidence_reason, detection_block
                   FROM contracts WHERE confidence_tier = 'suspected'
                   ORDER BY detection_block DESC LIMIT 50"""
            )
            self._json(200, [
                {"contract": r[0], "deployer": r[1], "method": r[2],
                 "reason": r[3][:200], "block": r[4]}
                for r in rows
            ])

        elif self.path == "/priority":
            rows = _query(
                """SELECT deployer_address, total_contracts_deployed,
                          deployment_pattern_notes, last_seen
                   FROM deployers
                   WHERE deployment_pattern_notes IS NOT NULL
                     AND deployment_pattern_notes != ''
                   ORDER BY total_contracts_deployed DESC"""
            )
            self._json(200, [
                {"address": r[0], "contracts": r[1], "notes": r[2], "last_seen": r[3]}
                for r in rows
            ])

        elif self.path == "/bots":
            rows = _query(
                """SELECT address, total_revert_count, is_deployer, first_seen, last_seen
                   FROM bot_candidates ORDER BY total_revert_count DESC LIMIT 50"""
            )
            self._json(200, [
                {"address": r[0], "reverts": r[1], "is_deployer": bool(r[2]),
                 "first_seen": r[3], "last_seen": r[4]}
                for r in rows
            ])

        elif self.path == "/health":
            hb = _query(
                "SELECT component, timestamp, blocks, deployments FROM heartbeat ORDER BY timestamp DESC",
            )
            gaps = _query(
                "SELECT component, disconnect, reconnect, reason FROM connection_gaps ORDER BY disconnect DESC LIMIT 10"
            )
            self._json(200, {
                "heartbeats": [
                    {"component": r[0], "timestamp": r[1], "blocks": r[2], "deployments": r[3]}
                    for r in hb
                ],
                "recent_gaps": [
                    {"component": r[0], "disconnect": r[1], "reconnect": r[2], "reason": r[3]}
                    for r in gaps
                ],
            })

        else:
            self._json(404, {
                "error": "not found",
                "endpoints": ["/stats", "/suspected", "/priority", "/bots", "/health"],
            })

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, *args):
        pass


def run_stats_server():
    server = HTTPServer(("0.0.0.0", PORT), StatsHandler)
    print(f"Stats API listening on :{PORT}", flush=True)
    server.serve_forever()


# Start HTTP stats server in background thread
threading.Thread(target=run_stats_server, daemon=True).start()

# Start surveillance components
monitor = subprocess.Popen(
    [sys.executable, "-m", "surveillance.deployment_monitor"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)

routing = subprocess.Popen(
    [sys.executable, "-m", "surveillance.routing_monitor"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)

# Keep alive — exit only if either process dies
monitor.wait()
routing.wait()
