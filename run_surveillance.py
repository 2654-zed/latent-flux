import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

import asyncio

DB_PATH = "/app/surveillance/data/surveillance.db"
PORT = int(os.environ.get("PORT", 8080))
# Alchemy HTTP URL derived from WSS URL
_wss = os.environ.get("ARB_WSS_URL", "")
ALCHEMY_HTTP = _wss.replace("wss://", "https://") if _wss else ""


def _query(sql, fetchone=False):
    con = sqlite3.connect(DB_PATH, timeout=10)
    cur = con.cursor()
    try:
        result = cur.execute(sql).fetchone() if fetchone else cur.execute(sql).fetchall()
    finally:
        con.close()
    return result


class StatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stats":
            # Single connection, all fresh queries, no caching
            con = sqlite3.connect(DB_PATH, timeout=10)
            c = con.cursor()

            stats = {
                "contracts": dict(c.execute(
                    "SELECT confidence_tier, COUNT(*) FROM contracts GROUP BY confidence_tier"
                ).fetchall()),
                "deployers": c.execute("SELECT COUNT(*) FROM deployers").fetchone()[0],
                "trap_events": c.execute("SELECT COUNT(*) FROM trap_events").fetchone()[0],
                "tx_events": c.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0],
                "bot_candidates": c.execute("SELECT COUNT(*) FROM bot_candidates").fetchone()[0],
                "bot_deployer_hits": c.execute(
                    "SELECT COUNT(*) FROM bot_candidates WHERE is_deployer = 1"
                ).fetchone()[0],
                "last_heartbeat": c.execute(
                    "SELECT component, timestamp FROM heartbeat ORDER BY timestamp DESC LIMIT 1"
                ).fetchone(),
                "cache": dict(zip(
                    ["entries", "total_hits"],
                    c.execute(
                        "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM bytecode_cache"
                    ).fetchone(),
                )),
                "top_bot_selectors": [
                    {"selector": r[0], "bots": r[1], "total_calls": r[2]}
                    for r in c.execute(
                        """SELECT function_selector, COUNT(DISTINCT bot_address), SUM(call_count)
                           FROM bot_candidate_selectors
                           GROUP BY function_selector ORDER BY SUM(call_count) DESC LIMIT 10"""
                    ).fetchall()
                ],
            }

            # Build cluster summary in a single query (no N+1 loop)
            cluster_rows = c.execute(
                """SELECT selector_cluster, COUNT(*) as members,
                          SUM(total_revert_count) as reverts,
                          MAX(last_seen) as last_active
                   FROM bot_candidates
                   WHERE selector_cluster IS NOT NULL
                   GROUP BY selector_cluster ORDER BY reverts DESC
                   LIMIT 20"""
            ).fetchall()
            clusters = []
            if cluster_rows:
                # Batch: get all shared selectors for all clusters in one query
                cluster_ids = [cr[0] for cr in cluster_rows]
                placeholders = ",".join("?" * len(cluster_ids))
                shared_sel_rows = c.execute(
                    f"""SELECT bc.selector_cluster, bs.function_selector,
                               COUNT(DISTINCT bs.bot_address) as users
                        FROM bot_candidate_selectors bs
                        JOIN bot_candidates bc ON bs.bot_address = bc.address
                        WHERE bc.selector_cluster IN ({placeholders})
                        GROUP BY bc.selector_cluster, bs.function_selector
                        HAVING users >= 2
                        ORDER BY bc.selector_cluster, users DESC""",
                    cluster_ids,
                ).fetchall()
                # Group shared selectors by cluster
                cluster_sels = {}
                for row in shared_sel_rows:
                    cluster_sels.setdefault(row[0], []).append(row[1])

                # Batch: get all known_selector tags in one query
                all_sels = list({s for sels in cluster_sels.values() for s in sels[:1]})
                tag_map = {}
                if all_sels:
                    sel_ph = ",".join("?" * len(all_sels))
                    tag_rows = c.execute(
                        f"SELECT function_selector, tag FROM known_selectors WHERE function_selector IN ({sel_ph})",
                        all_sels,
                    ).fetchall()
                    tag_map = {r[0]: r[1] for r in tag_rows}

                for cr in cluster_rows:
                    cid = cr[0]
                    shared = cluster_sels.get(cid, [])
                    clusters.append({
                        "id": cid,
                        "tag": tag_map.get(shared[0]) if shared else None,
                        "members": cr[1],
                        "total_reverts": cr[2],
                        "shared_selectors": shared,
                        "last_active": cr[3],
                    })
            stats["clusters"] = clusters

            # Cross-chain intelligence (lightweight)
            chains_monitored = [r[0] for r in c.execute(
                "SELECT DISTINCT chain FROM contracts"
            ).fetchall()]
            shared_deployers = c.execute(
                """SELECT COUNT(*) FROM (
                    SELECT deployer_address
                    FROM contracts
                    GROUP BY deployer_address
                    HAVING COUNT(DISTINCT chain) > 1
                )"""
            ).fetchone()[0]
            stats["cross_chain"] = {
                "chains_monitored": chains_monitored,
                "shared_deployers": shared_deployers,
            }

            # Recent alerts
            alert_rows = c.execute(
                "SELECT alert_type, address, tx_hash, timestamp FROM alerts WHERE COALESCE(false_positive, 0) = 0 ORDER BY id DESC LIMIT 5"
            ).fetchall()
            stats["recent_alerts"] = [
                {"type": r[0], "address": r[1], "tx": r[2], "timestamp": r[3]}
                for r in alert_rows
            ]

            con.close()
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
            con = sqlite3.connect(DB_PATH)
            rows = con.execute(
                """SELECT selector_cluster, COUNT(*) as members, SUM(total_revert_count) as reverts
                   FROM bot_candidates WHERE selector_cluster IS NOT NULL
                   GROUP BY selector_cluster ORDER BY members DESC"""
            ).fetchall()
            clusters = []
            for r in rows:
                member_rows = con.execute(
                    "SELECT address, total_revert_count FROM bot_candidates WHERE selector_cluster = ? ORDER BY total_revert_count DESC",
                    (r[0],),
                ).fetchall()
                clusters.append({
                    "cluster": r[0], "members": r[1], "total_reverts": r[2],
                    "addresses": [{"address": m[0], "reverts": m[1]} for m in member_rows],
                })
            con.close()
            self._json(200, clusters)

        elif self.path == "/funding":
            rows = _query(
                """SELECT deployer_address, total_contracts_deployed,
                          deployment_pattern_notes, funding_trail
                   FROM deployers
                   WHERE funding_trail IS NOT NULL AND funding_trail != ''
                   ORDER BY total_contracts_deployed DESC"""
            )
            results = []
            for r in rows:
                trail = r[3]
                try:
                    trail = json.loads(trail)
                except (json.JSONDecodeError, TypeError):
                    pass
                results.append({
                    "address": r[0], "contracts": r[1],
                    "notes": r[2], "funding_trail": trail,
                })
            self._json(200, results)

        elif self.path == "/cluster-events":
            rows = _query(
                """SELECT cluster_id, bot_address, event_type, timestamp,
                          trigger_selector, revert_count_at_join
                   FROM cluster_events ORDER BY timestamp DESC LIMIT 100"""
            )
            self._json(200, [
                {"cluster": r[0], "address": r[1], "event": r[2],
                 "timestamp": r[3], "selector": r[4], "reverts_at_join": r[5]}
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

        elif self.path == "/verification":
            rows = _query(
                """SELECT cv.contract_address, cv.has_code, cv.code_size, cv.is_proxy,
                          c.confidence_tier, c.deployer_address
                   FROM contract_verification cv
                   JOIN contracts c ON cv.contract_address = c.contract_address
                   ORDER BY cv.code_size DESC"""
            )
            self._json(200, [
                {"address": r[0], "has_code": bool(r[1]), "code_size": r[2],
                 "is_proxy": bool(r[3]), "tier": r[4], "deployer": r[5]}
                for r in rows
            ])

        elif self.path == "/funding-hops":
            rows = _query(
                """SELECT deployer_address, hop_number, source_address,
                          transfer_type, value_eth, tx_hash, timestamp, is_exchange
                   FROM funding_hops ORDER BY deployer_address, hop_number"""
            )
            self._json(200, [
                {"deployer": r[0], "hop": r[1], "source": r[2], "type": r[3],
                 "value_eth": r[4], "tx": r[5], "timestamp": r[6], "exchange": r[7]}
                for r in rows
            ])

        elif self.path == "/traces":
            rows = _query(
                "SELECT tx_hash, from_address, to_address, summary, timestamp FROM traces ORDER BY timestamp DESC LIMIT 20"
            )
            self._json(200, [
                {"tx": r[0], "from": r[1], "to": r[2], "summary": r[3], "timestamp": r[4]}
                for r in rows
            ])

        elif self.path == "/cross-chain":
            rows = _query(
                """SELECT d.deployer_address,
                          GROUP_CONCAT(DISTINCT c.chain) as chains,
                          COUNT(DISTINCT c.chain) as chain_count,
                          COUNT(c.contract_address) as total_contracts,
                          d.deployment_pattern_notes, d.entity_type
                   FROM deployers d
                   JOIN contracts c ON d.deployer_address = c.deployer_address
                   GROUP BY d.deployer_address
                   HAVING chain_count > 1
                   ORDER BY total_contracts DESC"""
            )
            self._json(200, [
                {"deployer": r[0], "chains": r[1], "chain_count": r[2],
                 "contracts": r[3], "notes": r[4], "entity_type": r[5]}
                for r in rows
            ])

        elif self.path == "/exposures":
            rows = _query(
                """SELECT exposed_address, approved_contract, approval_timestamp,
                          approval_amount, status, drain_tx_hash, drain_amount_usd, notes
                   FROM live_exposures ORDER BY
                   CASE status WHEN 'open' THEN 0 WHEN 'drained' THEN 1 ELSE 2 END,
                   approval_timestamp DESC"""
            )
            self._json(200, [
                {"exposed": r[0], "contract": r[1], "approved_at": r[2],
                 "amount": r[3], "status": r[4], "drain_tx": r[5],
                 "drain_usd": r[6], "notes": r[7]}
                for r in rows
            ])

        elif self.path == "/alerts":
            # Exclude false positives by default
            rows = _query(
                """SELECT alert_type, address, tx_hash, block_number, timestamp
                   FROM alerts WHERE COALESCE(false_positive, 0) = 0
                   ORDER BY id DESC LIMIT 50"""
            )
            self._json(200, [
                {"type": r[0], "address": r[1], "tx": r[2], "block": r[3], "timestamp": r[4]}
                for r in rows
            ])

        elif self.path.startswith("/dump"):
            # Full table dump for local DB sync. Requires admin auth via query param.
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            token = os.environ.get("ADMIN_TOKEN", "")
            req_token = params.get("token", [""])[0]
            if not token or req_token != token:
                self._json(403, {"error": "forbidden"})
                return
            # Which table?
            table = params.get("table", [""])[0]
            valid_tables = [
                "contracts", "deployers", "trap_events", "transaction_events",
                "bot_candidates", "bot_candidate_selectors", "bot_candidate_events",
                "known_selectors", "alerts", "live_exposures", "pattern_matches",
                "cluster_events", "funding_hops", "contract_verification", "traces",
                "bytecode_cache", "heartbeat", "connection_gaps",
            ]
            if table not in valid_tables:
                self._json(400, {"error": f"table required, valid: {valid_tables}"})
                return
            offset = int(params.get("offset", ["0"])[0])
            limit = int(params.get("limit", ["5000"])[0])
            limit = min(limit, 10000)
            con = sqlite3.connect(DB_PATH, timeout=10)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"SELECT * FROM [{table}] LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            data = [dict(r) for r in rows]
            total = con.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            con.close()
            self._json(200, {"table": table, "total": total, "offset": offset,
                             "limit": limit, "count": len(data), "rows": data})

        else:
            self._json(404, {
                "error": "not found",
                "endpoints": ["/stats", "/suspected", "/priority", "/bots",
                              "/health", "/tx-events", "/known-selectors",
                              "/clusters", "/cluster-events", "/bot-deployers",
                              "/bot-selectors", "/funding", "/funding-hops",
                              "/verification", "/traces", "/alerts", "/dump"],
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

        # Webhook endpoint — no auth required (Alchemy pushes directly)
        if self.path == "/webhook/alchemy":
            pass  # handled below without auth
        else:
            # Admin endpoints require Bearer token
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

        elif self.path == "/admin/auto-assign-clusters":
            # Auto-assign bots to clusters based on selector rules
            # Input: {"rules": {"cluster_001": ["000000e7"], "cluster_002": ["b2460c48", "6bfd6286"]}}
            rules = data.get("rules", {})
            if not rules:
                self._json(400, {"error": "rules dict required"})
                return
            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            results = {}
            for cluster_id, selectors in rules.items():
                placeholders = ",".join("?" for _ in selectors)
                rows = con.execute(
                    f"""SELECT DISTINCT bs.bot_address, bs.function_selector, bc.total_revert_count
                        FROM bot_candidate_selectors bs
                        JOIN bot_candidates bc ON bs.bot_address = bc.address
                        WHERE bs.function_selector IN ({placeholders})
                          AND (bc.selector_cluster IS NULL OR bc.selector_cluster != ?)""",
                    (*selectors, cluster_id),
                ).fetchall()
                count = 0
                for row in rows:
                    addr, sel, reverts = row[0], row[1], row[2]
                    con.execute(
                        "UPDATE bot_candidates SET selector_cluster = ? WHERE address = ?",
                        (cluster_id, addr),
                    )
                    con.execute(
                        """INSERT INTO cluster_events
                           (cluster_id, bot_address, event_type, timestamp, trigger_selector, revert_count_at_join)
                           VALUES (?, ?, 'joined', ?, ?, ?)""",
                        (cluster_id, addr, now, sel, reverts),
                    )
                    count += 1
                con.commit()
                results[cluster_id] = count
            # Return final membership counts
            for cid in rules:
                members = con.execute(
                    "SELECT COUNT(*) FROM bot_candidates WHERE selector_cluster = ?", (cid,)
                ).fetchone()[0]
                results[f"{cid}_total_members"] = members
            con.close()
            self._json(200, results)

        elif self.path == "/admin/funding-trail":
            addr = data.get("address", "").lower()
            trail = data.get("trail", "")
            if not addr or not trail:
                self._json(400, {"error": "address and trail required"})
                return
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "UPDATE deployers SET funding_trail = ? WHERE deployer_address = ?",
                (trail if isinstance(trail, str) else json.dumps(trail), addr),
            )
            con.commit()
            con.close()
            self._json(200, {"updated": addr})

        elif self.path == "/admin/add-exposure":
            exposed = data.get("exposed_address", "").lower()
            contract = data.get("approved_contract", "").lower()
            tx_hash = data.get("approval_tx_hash", "")
            timestamp = data.get("approval_timestamp", "")
            amount = data.get("approval_amount", "unlimited")
            token = data.get("token_address")
            notes = data.get("notes", "")
            if not exposed or not contract:
                self._json(400, {"error": "exposed_address and approved_contract required"})
                return
            con = sqlite3.connect(DB_PATH)
            con.execute(
                """INSERT OR IGNORE INTO live_exposures
                   (exposed_address, approved_contract, approval_tx_hash,
                    approval_timestamp, approval_amount, token_address, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, 'open', ?)""",
                (exposed, contract, tx_hash, timestamp, amount, token, notes),
            )
            con.commit()
            con.close()
            self._json(200, {"added": exposed, "contract": contract})

        elif self.path == "/admin/mark-false-positive":
            # Mark alerts as false positive by address
            addresses = [a.lower() for a in data.get("addresses", [])]
            if not addresses:
                self._json(400, {"error": "addresses list required"})
                return
            con = sqlite3.connect(DB_PATH)
            total = 0
            for addr in addresses:
                con.execute(
                    "UPDATE alerts SET false_positive = 1 WHERE LOWER(address) = ?",
                    (addr,),
                )
                total += con.total_changes
            con.commit()
            con.close()
            self._json(200, {"marked": total, "addresses": addresses})

        elif self.path == "/admin/entity-type":
            # Batch update entity_type for multiple addresses
            updates = data.get("updates", {})  # {address: entity_type}
            if not updates:
                self._json(400, {"error": "updates dict required: {address: entity_type}"})
                return
            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            count = 0
            for addr, etype in updates.items():
                # Ensure deployer row exists
                con.execute(
                    """INSERT INTO deployers (deployer_address, chain, first_seen, last_seen,
                           total_contracts_deployed, entity_type)
                       VALUES (?, 'arbitrum', ?, ?, 0, ?)
                       ON CONFLICT(deployer_address) DO UPDATE SET entity_type = ?""",
                    (addr.lower(), now, now, etype, etype),
                )
                count += 1
            con.commit()
            con.close()
            self._json(200, {"updated": count})

        elif self.path == "/admin/check-verification":
            # Check contract verification for suspected contracts
            if not ALCHEMY_HTTP:
                self._json(500, {"error": "ALCHEMY_HTTP not configured"})
                return
            addr = data.get("address")
            check_all = data.get("all", False)

            async def _run():
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from surveillance.intelligence import check_contract_verification, check_all_suspected
                con = sqlite3.connect(DB_PATH)
                if check_all:
                    return await check_all_suspected(ALCHEMY_HTTP, con)
                elif addr:
                    return await check_contract_verification(ALCHEMY_HTTP, addr.lower(), con)
                return {"error": "provide address or all=true"}

            result = asyncio.run(_run())
            self._json(200, result)

        elif self.path == "/admin/trace-funding":
            # Multi-hop funding trace
            if not ALCHEMY_HTTP:
                self._json(500, {"error": "ALCHEMY_HTTP not configured"})
                return
            addr = data.get("address")
            all_priority = data.get("all_priority", False)

            async def _run():
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from surveillance.intelligence import trace_funding_hops
                con = sqlite3.connect(DB_PATH)
                if all_priority:
                    deployers = con.execute(
                        """SELECT deployer_address FROM deployers
                           WHERE deployment_pattern_notes IS NOT NULL
                             AND deployment_pattern_notes != ''"""
                    ).fetchall()
                    results = {}
                    for row in deployers:
                        hops = await trace_funding_hops(ALCHEMY_HTTP, row[0], con)
                        results[row[0]] = hops
                    return results
                elif addr:
                    return await trace_funding_hops(ALCHEMY_HTTP, addr.lower(), con)
                return {"error": "provide address or all_priority=true"}

            result = asyncio.run(_run())
            self._json(200, result)

        elif self.path == "/admin/trace-usdc":
            # USDC withdrawal destination trace
            if not ALCHEMY_HTTP:
                self._json(500, {"error": "ALCHEMY_HTTP not configured"})
                return
            addr = data.get("address", "").lower()
            if not addr:
                self._json(400, {"error": "address required"})
                return

            async def _run():
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from surveillance.intelligence import trace_usdc_withdrawals
                con = sqlite3.connect(DB_PATH)
                return await trace_usdc_withdrawals(ALCHEMY_HTTP, addr, con)

            result = asyncio.run(_run())
            self._json(200, result)

        elif self.path == "/admin/trace-tx":
            # Execution trace for a specific transaction
            if not ALCHEMY_HTTP:
                self._json(500, {"error": "ALCHEMY_HTTP not configured"})
                return
            tx_hash = data.get("tx_hash", "")
            if not tx_hash:
                self._json(400, {"error": "tx_hash required"})
                return

            async def _run():
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from surveillance.intelligence import trace_transaction
                con = sqlite3.connect(DB_PATH)
                return await trace_transaction(ALCHEMY_HTTP, tx_hash, con)

            result = asyncio.run(_run())
            self._json(200, result)

        elif self.path == "/admin/scan-bot":
            # Retroactive bot history scan
            if not ALCHEMY_HTTP:
                self._json(500, {"error": "ALCHEMY_HTTP not configured"})
                return
            addr = data.get("address", "").lower()
            if not addr:
                self._json(400, {"error": "address required"})
                return

            async def _run():
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from surveillance.intelligence import scan_bot_history
                con = sqlite3.connect(DB_PATH)
                return await scan_bot_history(ALCHEMY_HTTP, addr, con)

            result = asyncio.run(_run())
            self._json(200, result)

        elif self.path == "/admin/register-webhook":
            # Register Alchemy Notify webhooks from inside Railway
            alchemy_token = os.environ.get("ALCHEMY_AUTH_TOKEN", "")
            if not alchemy_token:
                self._json(500, {"error": "ALCHEMY_AUTH_TOKEN not set"})
                return

            addresses = [a.lower() for a in data.get("addresses", [])]
            webhook_url = data.get("webhook_url", "https://spypy.up.railway.app/webhook/alchemy")
            network = data.get("network", "ARB_MAINNET")

            if not addresses:
                self._json(400, {"error": "addresses list required"})
                return

            import urllib.request
            req_data = json.dumps({
                "network": network,
                "webhook_type": "ADDRESS_ACTIVITY",
                "webhook_url": webhook_url,
                "addresses": addresses,
            }).encode()
            req = urllib.request.Request(
                "https://dashboard.alchemy.com/api/create-webhook",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "X-Alchemy-Token": alchemy_token,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read())
                    # Store webhook config
                    con = sqlite3.connect(DB_PATH)
                    wh_id = result.get("data", {}).get("id", "")
                    con.execute(
                        """INSERT INTO alerts (alert_type, address, tx_hash, block_number, timestamp, payload)
                           VALUES ('webhook_registered', ?, NULL, NULL, ?, ?)""",
                        (json.dumps(addresses), datetime.now(timezone.utc).isoformat(),
                         json.dumps(result)),
                    )
                    con.commit()
                    con.close()
                    self._json(200, {"webhook_id": wh_id, "addresses": addresses, "result": result})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return

        elif self.path == "/webhook/alchemy":
            # Alchemy Notify webhook receiver — no auth (Alchemy signs these)
            # FILTERED: only log when a watched address is the direct from/to
            ALERT_RULES = {
                "0x326eebfc4bd6c7a799afc9359822eb79056ee681": "TRAP_FIRED",
                "0x693523aed717ab3203b6285cdf5d261b6463774e": "TRAP_FIRED",
                "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "OPERATOR_ACTIVE",
                "0xc6962004f452be9203591991d15f6b388e09e8d0": "CASHOUT_MOVEMENT",
                "0x79a2f71187dc9fd9b173781e6dd4ff9960f6f61b": "TRAP_FIRED",
                "0x74b9a8351bd725ca3edd654c9728873b8c6f051e": "TRAP_FIRED",
                "0x3f2cdae910cd13638e38b45881e7a4fc3a9fe320": "VICTIM_ACTIVE",
                "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "LAUNDRY_PIPELINE",
                "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70": "LAUNDRY_PIPELINE",
            }

            con = sqlite3.connect(DB_PATH)
            now = datetime.now(timezone.utc).isoformat()
            activity = data.get("event", {}).get("activity", [])
            logged = 0
            skipped = 0

            for act in activity:
                from_addr = act.get("fromAddress", "").lower()
                to_addr = act.get("toAddress", "").lower()
                tx_hash = act.get("hash", "")
                block = int(act.get("blockNum", "0x0"), 16) if act.get("blockNum") else None

                # Only fire if a watched address is the DIRECT from or to
                triggered = None
                alert_type = None
                for watched, atype in ALERT_RULES.items():
                    if from_addr == watched or to_addr == watched:
                        triggered = watched
                        alert_type = atype
                        break

                if not triggered:
                    skipped += 1
                    continue

                con.execute(
                    """INSERT INTO alerts (alert_type, address, tx_hash, block_number, timestamp, payload)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (alert_type, triggered, tx_hash, block, now, json.dumps(act)),
                )
                logged += 1

            con.commit()
            con.close()
            self._json(200, {"logged": logged, "skipped": skipped})
            return  # webhook doesn't need admin auth

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
    server = ThreadingHTTPServer(("0.0.0.0", PORT), StatsHandler)
    print(f"Stats API listening on :{PORT}", flush=True)
    server.serve_forever()


# Start HTTP stats server in background thread
threading.Thread(target=run_stats_server, daemon=True).start()

# Start surveillance components
processes = []

# Arbitrum deployment monitor
monitor_arb = subprocess.Popen(
    [sys.executable, "-m", "surveillance.deployment_monitor"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)
processes.append(("arbitrum_monitor", monitor_arb))

# Base deployment monitor (if BASE_WSS_URL is set)
if os.environ.get("BASE_WSS_URL"):
    monitor_base = subprocess.Popen(
        [sys.executable, "-m", "surveillance.deployment_monitor",
         "--rpc", os.environ["BASE_WSS_URL"], "--chain", "base"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(("base_monitor", monitor_base))
    print("Base chain monitor started", flush=True)

# Routing monitor (Arbitrum only for now)
routing = subprocess.Popen(
    [sys.executable, "-m", "surveillance.routing_monitor"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)
processes.append(("routing", routing))

# Keep alive — exit if any process dies
for name, proc in processes:
    proc.wait()
    print(f"Process {name} exited with code {proc.returncode}", flush=True)
