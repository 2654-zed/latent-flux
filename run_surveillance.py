import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
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
                "top_bot_selectors": [
                    {"selector": r[0], "bots": r[1], "total_calls": r[2]}
                    for r in _query(
                        """SELECT function_selector, COUNT(DISTINCT bot_address), SUM(call_count)
                           FROM bot_candidate_selectors
                           GROUP BY function_selector ORDER BY SUM(call_count) DESC LIMIT 10"""
                    )
                ],
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

        elif self.path == "/tx-events":
            rows = _query(
                """SELECT contract_address, interacting_address, function_selector,
                          bot_tag, gas_price_gwei, max_priority_fee_gwei, gas_pattern,
                          block_number, timestamp, is_reverted, tx_hash
                   FROM transaction_events ORDER BY block_number DESC LIMIT 100"""
            )
            self._json(200, [
                {"contract": r[0], "interactor": r[1], "selector": r[2],
                 "bot_tag": r[3], "gas_gwei": r[4], "priority_fee_gwei": r[5],
                 "gas_pattern": r[6], "block": r[7], "timestamp": r[8],
                 "reverted": bool(r[9]), "tx_hash": r[10]}
                for r in rows
            ])

        elif self.path == "/bot-deployers":
            rows = _query(
                """SELECT bc.address, bc.total_revert_count, bc.first_seen, bc.last_seen,
                          d.total_contracts_deployed, d.deployment_pattern_notes
                   FROM bot_candidates bc
                   JOIN deployers d ON bc.address = d.deployer_address
                   WHERE bc.is_deployer = 1
                   ORDER BY bc.total_revert_count DESC"""
            )
            self._json(200, [
                {"address": r[0], "reverts": r[1], "bot_first_seen": r[2],
                 "bot_last_seen": r[3], "contracts_deployed": r[4], "deployer_notes": r[5]}
                for r in rows
            ])

        elif self.path == "/bot-selectors":
            rows = _query(
                """SELECT bs.bot_address, bs.function_selector, bs.call_count,
                          bs.first_seen, bs.last_seen
                   FROM bot_candidate_selectors bs
                   ORDER BY bs.call_count DESC LIMIT 50"""
            )
            self._json(200, [
                {"bot": r[0], "selector": r[1], "calls": r[2],
                 "first_seen": r[3], "last_seen": r[4]}
                for r in rows
            ])

        elif self.path == "/known-selectors":
            rows = _query(
                "SELECT function_selector, tag, decoded_name, notes FROM known_selectors ORDER BY tag"
            )
            self._json(200, [
                {"selector": r[0], "tag": r[1], "decoded": r[2], "notes": r[3]}
                for r in rows
            ])

        elif self.path == "/clusters":
            rows = _query(
                """SELECT selector_cluster, COUNT(*) as members, SUM(total_revert_count) as reverts
                   FROM bot_candidates WHERE selector_cluster IS NOT NULL
                   GROUP BY selector_cluster ORDER BY members DESC"""
            )
            clusters = []
            for r in rows:
                members = _query(
                    "SELECT address, total_revert_count FROM bot_candidates WHERE selector_cluster = ? ORDER BY total_revert_count DESC",
                    args=(r[0],) if False else None,
                )
                # Can't pass args through _query easily, use inline
                con = sqlite3.connect(DB_PATH)
                member_rows = con.execute(
                    "SELECT address, total_revert_count FROM bot_candidates WHERE selector_cluster = ?",
                    (r[0],),
                ).fetchall()
                con.close()
                clusters.append({
                    "cluster": r[0], "members": r[1], "total_reverts": r[2],
                    "addresses": [{"address": m[0], "reverts": m[1]} for m in member_rows],
                })
            self._json(200, clusters)

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
                "endpoints": ["/stats", "/suspected", "/priority", "/bots",
                              "/health", "/tx-events"],
            })

    def do_POST(self):
        """Write endpoints for remote database updates."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        # Simple auth: require a token from env var
        token = os.environ.get("ADMIN_TOKEN", "")
        auth = self.headers.get("Authorization", "")
        if not token or auth != f"Bearer {token}":
            self._json(403, {"error": "forbidden"})
            return

        if self.path == "/admin/deployer-notes":
            # Update deployment_pattern_notes for a deployer
            addr = data.get("address", "").lower()
            notes = data.get("notes", "")
            if not addr or not notes:
                self._json(400, {"error": "address and notes required"})
                return
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "UPDATE deployers SET deployment_pattern_notes = ? WHERE deployer_address = ?",
                (notes, addr),
            )
            con.commit()
            changed = con.total_changes
            con.close()
            self._json(200, {"updated": changed, "address": addr})

        elif self.path == "/admin/bot-candidate":
            # Upsert a bot candidate manually
            addr = data.get("address", "").lower()
            notes = data.get("notes", "")
            reverts = data.get("reverts", 0)
            if not addr:
                self._json(400, {"error": "address required"})
                return
            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            con.execute(
                """INSERT INTO bot_candidates (address, first_seen, last_seen, total_revert_count, is_deployer)
                   VALUES (?, ?, ?, ?, 0)
                   ON CONFLICT(address) DO UPDATE SET
                       last_seen = ?, total_revert_count = total_revert_count + ?""",
                (addr, now, now, reverts, now, reverts),
            )
            # Check deployer cross-ref
            is_dep = con.execute(
                "SELECT 1 FROM deployers WHERE deployer_address = ?", (addr,)
            ).fetchone()
            if is_dep:
                con.execute("UPDATE bot_candidates SET is_deployer = 1 WHERE address = ?", (addr,))
            con.commit()
            con.close()
            self._json(200, {"upserted": addr, "notes": notes})

        elif self.path == "/admin/flag-address":
            # Add an investigation note to a deployer (create if needed)
            addr = data.get("address", "").lower()
            notes = data.get("notes", "")
            if not addr or not notes:
                self._json(400, {"error": "address and notes required"})
                return
            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            # Ensure deployer row exists
            con.execute(
                """INSERT INTO deployers (deployer_address, chain, first_seen, last_seen,
                       total_contracts_deployed, deployment_pattern_notes)
                   VALUES (?, 'arbitrum', ?, ?, 0, ?)
                   ON CONFLICT(deployer_address) DO UPDATE SET
                       deployment_pattern_notes = ?""",
                (addr, now, now, notes, notes),
            )
            con.commit()
            con.close()
            self._json(200, {"flagged": addr})

        elif self.path == "/admin/upgrade-contracts":
            # Upgrade all contracts from a deployer to a new tier
            deployer = data.get("deployer", "").lower()
            tier = data.get("tier", "suspected")
            reason = data.get("reason", "")
            if not deployer or not reason:
                self._json(400, {"error": "deployer and reason required"})
                return
            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            con.execute(
                """UPDATE contracts SET confidence_tier = ?, confidence_reason = ?, last_updated = ?
                   WHERE deployer_address = ? AND confidence_tier != 'confirmed'""",
                (tier, reason, now, deployer),
            )
            changed = con.total_changes
            con.commit()
            con.close()
            self._json(200, {"upgraded": changed, "deployer": deployer, "tier": tier})

        elif self.path == "/admin/known-selector":
            selector = data.get("selector", "").lower()
            tag = data.get("tag", "")
            decoded = data.get("decoded_name")
            notes = data.get("notes", "")
            if not selector or not tag:
                self._json(400, {"error": "selector and tag required"})
                return
            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            con.execute(
                """INSERT INTO known_selectors (function_selector, tag, decoded_name, notes, created)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(function_selector) DO UPDATE SET tag = ?, decoded_name = ?, notes = ?""",
                (selector, tag, decoded, notes, now, tag, decoded, notes),
            )
            con.commit()
            con.close()
            self._json(200, {"stored": selector, "tag": tag})

        elif self.path == "/admin/bot-cluster":
            cluster_id = data.get("cluster_id", "")
            addresses = [a.lower() for a in data.get("addresses", [])]
            if not cluster_id or not addresses:
                self._json(400, {"error": "cluster_id and addresses required"})
                return
            con = sqlite3.connect(DB_PATH)
            updated = 0
            for addr in addresses:
                con.execute(
                    "UPDATE bot_candidates SET selector_cluster = ? WHERE address = ?",
                    (cluster_id, addr),
                )
                updated += con.total_changes
            con.commit()
            con.close()
            self._json(200, {"cluster": cluster_id, "updated": updated})

        else:
            self._json(404, {"error": "unknown admin endpoint"})

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
