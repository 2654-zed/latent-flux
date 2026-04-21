import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing
# Use 'fork' on Linux (Railway). 'spawn' causes infinite re-import of this
# module-level code in child processes. 'fork' inherits the parent's state
# without re-executing module-level code.
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set
from multiprocessing import Process, Queue as MPQueue

import asyncio

DB_PATH = "/app/surveillance/data/surveillance.db"
PORT = int(os.environ.get("PORT", 8080))
# Alchemy HTTP URL derived from WSS URL
_wss = os.environ.get("ARB_WSS_URL", "")
ALCHEMY_HTTP = _wss.replace("wss://", "https://") if _wss else ""
_eth_wss = os.environ.get("ETH_WSS_URL", "")
ETH_HTTP = _eth_wss.replace("wss://", "https://") if _eth_wss else ""


def _ro_connect(timeout=10):
    """Open a read-only SQLite connection for stats/dump queries."""
    try:
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=timeout)
    except Exception:
        # Fallback: DB may not exist yet or mode=ro not supported
        return sqlite3.connect(DB_PATH, timeout=timeout)


def _query(sql, fetchone=False):
    con = _ro_connect()
    cur = con.cursor()
    try:
        result = cur.execute(sql).fetchone() if fetchone else cur.execute(sql).fetchall()
    finally:
        con.close()
    return result


class StatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            # Liveness probe — always returns 200 so Railway's healthcheck
            # succeeds even when the DB is unusable. The real health state
            # lives in /stats; this endpoint exists to keep the container
            # alive during /admin/upload-db recovery windows.
            self._json(200, {"status": "alive"})
            return
        if self.path == "/stats":
            # Single connection, all fresh queries, no caching.
            # Wrapped so DB unreachability reports 200 with degraded shape
            # instead of 500 — the container must stay up for /admin/upload-db.
            try:
                con = _ro_connect()
                c = con.cursor()
            except sqlite3.OperationalError as _db_err:
                self._json(200, {
                    "status": "degraded",
                    "error": f"db_unreachable: {_db_err}",
                    "message": "DB unavailable; container is up for admin recovery only",
                })
                return

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

            # Longitudinal behavioral scoring summary
            try:
                scored = c.execute("SELECT COUNT(*) FROM deployers WHERE behavioral_score IS NOT NULL AND behavioral_score > 0").fetchone()[0]
                high_conf = c.execute("SELECT COUNT(*) FROM deployers WHERE behavioral_score > 0.5").fetchone()[0]
                critical = c.execute("SELECT COUNT(*) FROM deployers WHERE behavioral_score > 0.7").fetchone()[0]
                top = c.execute("SELECT deployer_address, behavioral_score FROM deployers ORDER BY behavioral_score DESC LIMIT 1").fetchone()
                stats["longitudinal"] = {
                    "scored_deployers": scored,
                    "high_confidence": high_conf,
                    "critical": critical,
                    "top_scorer": top[0] if top else None,
                    "top_score": round(top[1], 3) if top and top[1] else 0,
                }
            except Exception:
                stats["longitudinal"] = {"error": "scoring not yet run"}

            # Event monitors summary
            try:
                liq = c.execute("SELECT COUNT(*) FROM liquidity_events").fetchone()[0]
                liq_crit = c.execute("SELECT COUNT(*) FROM liquidity_events WHERE alert_level = 'critical'").fetchone()[0]
                approvals = c.execute("SELECT COUNT(*) FROM approval_events").fetchone()[0]
                bridges = c.execute("SELECT COUNT(*) FROM bridge_events").fetchone()[0]
                pairs = c.execute("SELECT COUNT(*) FROM pair_creation_events").fetchone()[0]
                pairs_crit = c.execute("SELECT COUNT(*) FROM pair_creation_events WHERE alert_level = 'critical'").fetchone()[0]
                cex = c.execute("SELECT COUNT(*) FROM cex_deposit_candidates WHERE flagged = 1").fetchone()[0]
                stats["event_monitors"] = {
                    "liquidity_events": liq,
                    "liquidity_critical": liq_crit,
                    "approval_events": approvals,
                    "bridge_events": bridges,
                    "pair_creations": pairs,
                    "pair_creations_critical": pairs_crit,
                    "cex_deposit_candidates": cex,
                }
            except Exception:
                stats["event_monitors"] = {"status": "tables not yet created"}

            # Funder tracing coverage
            try:
                total_dep = c.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
                has_trail = c.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail IS NOT NULL AND funding_trail != ''").fetchone()[0]
                has_org = c.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%org_link%'").fetchone()[0]
                gas_stations = c.execute("SELECT COUNT(*) FROM deployers WHERE entity_type = 'gas_station'").fetchone()[0]
                hop_traced = c.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%hop_traced%'").fetchone()[0]
                stats["funder_tracing"] = {
                    "deployers_traced": has_trail,
                    "coverage_pct": round(has_trail / total_dep * 100, 1) if total_dep > 0 else 0,
                    "org_links_found": has_org,
                    "gas_stations": gas_stations,
                    "hop_traced": hop_traced,
                }
            except Exception:
                stats["funder_tracing"] = {"status": "not available"}

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
            con = _ro_connect()
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

        elif self.path == "/liquidity-events":
            rows = _query(
                """SELECT tx_hash, block_number, timestamp, chain, router_address,
                          caller_address, event_type, token_address, linked_deployer, alert_level
                   FROM liquidity_events ORDER BY block_number DESC LIMIT 50"""
            )
            self._json(200, [
                {"tx": r[0], "block": r[1], "timestamp": r[2], "chain": r[3],
                 "router": r[4], "caller": r[5], "type": r[6], "token": r[7],
                 "linked_deployer": r[8], "alert": r[9]}
                for r in rows
            ])

        elif self.path == "/pair-creations":
            rows = _query(
                """SELECT tx_hash, block_number, timestamp, chain, factory_address,
                          token0, token1, pair_address, linked_deployer, alert_level
                   FROM pair_creation_events ORDER BY block_number DESC LIMIT 50"""
            )
            self._json(200, [
                {"tx": r[0], "block": r[1], "timestamp": r[2], "chain": r[3],
                 "factory": r[4], "token0": r[5], "token1": r[6], "pair": r[7],
                 "linked_deployer": r[8], "alert": r[9]}
                for r in rows
            ])

        elif self.path == "/cex-candidates":
            rows = _query(
                """SELECT address, chain, unique_senders, total_inflows, total_outflows,
                          first_seen, last_seen
                   FROM cex_deposit_candidates WHERE flagged = 1
                   ORDER BY unique_senders DESC LIMIT 50"""
            )
            self._json(200, [
                {"address": r[0], "chain": r[1], "senders": r[2], "inflows": r[3],
                 "outflows": r[4], "first_seen": r[5], "last_seen": r[6]}
                for r in rows
            ])

        elif self.path == "/bridge-events":
            rows = _query(
                """SELECT tx_hash, block_number, timestamp, chain, bridge_contract,
                          sender, value_wei, org_link, alert_level
                   FROM bridge_events ORDER BY block_number DESC LIMIT 50"""
            )
            self._json(200, [
                {"tx": r[0], "block": r[1], "timestamp": r[2], "chain": r[3],
                 "bridge": r[4], "sender": r[5], "value": r[6], "org": r[7], "alert": r[8]}
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

        elif self.path == "/rib/scores":
            try:
                from surveillance.rib_scorer import run_self_evaluation
                results = run_self_evaluation(DB_PATH)
                self._json(200, results)
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif self.path.startswith("/rib/export"):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            seed = int(params.get("seed", ["42"])[0])
            try:
                from surveillance.rib_export import export_anonymized_dataset
                paths = export_anonymized_dataset(DB_PATH, seed=seed)
                self._json(200, {"status": "exported", "files": paths})
            except Exception as e:
                self._json(500, {"error": str(e)})

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
                "liquidity_events", "approval_events", "bridge_events",
                "pair_creation_events", "cex_deposit_candidates",
                "org_transfer_events", "watchlist", "watchlist_hits",
                "self_test_traps", "approval_watchlist",
                "dormant_activations", "timelock_countdowns", "sload_patterns",
                "permit_events", "proxy_implementations", "api_keys",
                "api_watches", "approval_values", "contract_balances",
                "external_benchmarks",
                # Consumable-surface tables exposed for the Phase 1 trading research
                # experiment (see Desktop/Trading/LAYER3_TRADING_EXPERIMENT.md, Addendum A #6).
                # All seven are already documented as part of the consumable surface in
                # docs/layer3_consumable_intelligence.md §1.6-§1.10.
                "org_wallets",
                "org_candidates",
                "infrastructure_registry",
                "bytecode_families",
                "bytecode_family_members",
                "trust_amplification",
                "extraction_events",
            ]
            if table not in valid_tables:
                self._json(400, {"error": f"table required, valid: {valid_tables}"})
                return
            offset = int(params.get("offset", ["0"])[0])
            limit = int(params.get("limit", ["5000"])[0])
            limit = min(limit, 10000)
            con = _ro_connect()
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"SELECT * FROM [{table}] LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            data = [dict(r) for r in rows]
            total = con.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            con.close()
            self._json(200, {"table": table, "total": total, "offset": offset,
                             "limit": limit, "count": len(data), "rows": data})

        elif self.path.startswith("/old-dump"):
            # Dump from an old mounted volume for data recovery.
            # Mount old volume at /app/old_data, then query like /dump.
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            token = os.environ.get("ADMIN_TOKEN", "")
            req_token = params.get("token", [""])[0]
            if not token or req_token != token:
                self._json(403, {"error": "forbidden"})
                return
            old_db_path = params.get("path", ["/app/old_data/surveillance.db"])[0]
            # Safety: only allow paths under /app/old_data
            if not old_db_path.startswith("/app/old_data"):
                self._json(400, {"error": "path must be under /app/old_data"})
                return
            import pathlib
            if not pathlib.Path(old_db_path).exists():
                # Try to find any .db file under /app/old_data
                found = list(pathlib.Path("/app/old_data").rglob("*.db")) if pathlib.Path("/app/old_data").exists() else []
                self._json(404, {
                    "error": f"DB not found at {old_db_path}",
                    "found_files": [str(f) for f in found[:20]],
                    "old_data_exists": pathlib.Path("/app/old_data").exists(),
                })
                return
            # Open old DB in immutable mode to bypass WAL locks
            db_uri = f"file:{old_db_path}?immutable=1"
            table = params.get("table", [""])[0]
            if table == "":
                # List all tables in the old DB
                try:
                    old_con = sqlite3.connect(db_uri, uri=True, timeout=10)
                    old_con.row_factory = sqlite3.Row
                    tables = [r[0] for r in old_con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall()]
                    counts = {}
                    for t in tables:
                        try:
                            counts[t] = old_con.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                        except Exception:
                            counts[t] = -1
                    old_con.close()
                    self._json(200, {"db_path": old_db_path, "tables": counts})
                except Exception as e:
                    self._json(500, {"error": str(e)})
                return
            offset = int(params.get("offset", ["0"])[0])
            limit = int(params.get("limit", ["5000"])[0])
            limit = min(limit, 10000)
            try:
                old_con = sqlite3.connect(db_uri, uri=True, timeout=10)
                old_con.row_factory = sqlite3.Row
                rows = old_con.execute(
                    f"SELECT * FROM [{table}] LIMIT ? OFFSET ?", (limit, offset)
                ).fetchall()
                data = [dict(r) for r in rows]
                total = old_con.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
                old_con.close()
                self._json(200, {"table": table, "total": total, "offset": offset,
                                 "limit": limit, "count": len(data), "rows": data})
            except Exception as e:
                self._json(500, {"error": str(e)})

        else:
            self._json(404, {
                "error": "not found",
                "endpoints": ["/stats", "/suspected", "/priority", "/bots",
                              "/health", "/tx-events", "/known-selectors",
                              "/clusters", "/cluster-events", "/bot-deployers",
                              "/bot-selectors", "/funding", "/funding-hops",
                              "/verification", "/traces", "/alerts", "/dump",
                              "/old-dump"],
            })

    def do_POST(self):
        """Write endpoints for remote database updates."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"

        # Binary upload endpoints — handle before JSON parsing
        if self.path in ("/admin/upload-db", "/admin/upload-chunk",
                         "/admin/finalize-upload", "/admin/truncate-staging"):
            token = os.environ.get("ADMIN_TOKEN", "")
            auth = self.headers.get("Authorization", "")
            if not token or auth != f"Bearer {token}":
                self._json(403, {"error": "forbidden"})
                return

            if self.path == "/admin/upload-db":
                self._handle_upload_db(body)
            elif self.path == "/admin/upload-chunk":
                # Chunked upload — append body to staging file
                staging = DB_PATH + ".staging"
                offset = int(self.headers.get("X-Chunk-Offset", "-1"))
                try:
                    mode = "wb" if offset == 0 else "ab"
                    with open(staging, mode) as f:
                        f.write(body)
                    size = os.path.getsize(staging)
                    self._json(200, {"status": "ok", "offset": offset, "staging_size": size})
                except Exception as e:
                    self._json(500, {"error": str(e)})
            elif self.path == "/admin/finalize-upload":
                # Finalize: decompress staging file and replace DB
                staging = DB_PATH + ".staging"
                if not os.path.exists(staging):
                    self._json(400, {"error": "no staging file"})
                    return
                # Treat staging as the upload body
                with open(staging, "rb") as f:
                    staging_body = f.read()
                os.unlink(staging)
                self._handle_upload_db(staging_body)
            elif self.path == "/admin/truncate-staging":
                # Truncate the staging file to a specified size. Used to
                # recover from partial-write scenarios when the upload-chunk
                # connection drops mid-write and the server-side size drifts
                # past what the client has confirmed.
                staging = DB_PATH + ".staging"
                try:
                    to_size = int(self.headers.get("X-Target-Size", "-1"))
                    if to_size < 0:
                        self._json(400, {"error": "X-Target-Size header required"})
                        return
                    if not os.path.exists(staging):
                        if to_size == 0:
                            self._json(200, {"status": "ok", "staging_size": 0,
                                             "note": "no staging file (treated as size 0)"})
                            return
                        self._json(400, {"error": "no staging file to truncate"})
                        return
                    current = os.path.getsize(staging)
                    if to_size > current:
                        self._json(400, {
                            "error": f"cannot truncate to {to_size} (current {current})"
                        })
                        return
                    with open(staging, "r+b") as f:
                        f.truncate(to_size)
                    self._json(200, {"status": "ok", "staging_size": os.path.getsize(staging),
                                     "was": current})
                except Exception as e:
                    self._json(500, {"error": str(e)})
            return

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

        elif self.path == "/admin/compact-db":
            # Deduplicate alerts + WAL checkpoint
            con = sqlite3.connect(DB_PATH, timeout=30)
            # Count before
            before = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            # Keep one alert per unique (alert_type, address, tx_hash), delete duplicates
            con.execute("""
                DELETE FROM alerts WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM alerts GROUP BY alert_type, address, tx_hash
                )
            """)
            after = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            deleted = before - after
            con.commit()
            # Prune old bot_candidate_events (keep last 7 days)
            bce_before = con.execute("SELECT COUNT(*) FROM bot_candidate_events").fetchone()[0]
            con.execute("DELETE FROM bot_candidate_events WHERE timestamp < datetime('now', '-7 days')")
            bce_after = con.execute("SELECT COUNT(*) FROM bot_candidate_events").fetchone()[0]
            con.commit()
            # WAL checkpoint + VACUUM
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            try:
                con.execute("VACUUM")
                vacuumed = True
            except Exception:
                vacuumed = False
            # Check DB size
            db_size = con.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()").fetchone()[0]
            con.close()
            self._json(200, {
                "alerts_before": before,
                "alerts_after": after,
                "duplicates_removed": deleted,
                "bot_events_pruned": bce_before - bce_after,
                "wal_checkpoint": "completed",
                "vacuum": vacuumed,
                "db_size_mb": round(db_size / 1024 / 1024, 1),
            })

        elif self.path == "/admin/mark-false-positive":
            # Mark alerts as false positive by address with a reason.
            # Correction #14: reason is now REQUIRED and is written to the
            # `false_positives` audit table atomically with the alert flag.
            # Silent bulk-mark (prior behavior) left a 3,577-row audit gap on
            # local DB; this endpoint is hardened so the gap cannot recur.
            addresses = [a.lower() for a in data.get("addresses", [])]
            reason = (data.get("reason") or "").strip()
            detector_blamed = (data.get("detector_blamed") or "manual_review").strip()
            if not addresses:
                self._json(400, {"error": "addresses list required"})
                return
            if not reason or len(reason) < 10:
                self._json(400, {"error": "reason required (>= 10 chars); see Correction #14"})
                return
            # Two connections: a read-only connection for the pre-flight
            # SELECT (non-blocking in WAL mode), and short per-address write
            # transactions that each commit individually so contention with
            # db_writer resolves in the SQLite busy-timeout window instead
            # of blocking on an explicit BEGIN IMMEDIATE held across all
            # addresses.
            ro = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=15)
            wr = sqlite3.connect(DB_PATH, timeout=30)
            wr.execute("PRAGMA busy_timeout=30000")
            try:
                now_iso = datetime.now(timezone.utc).isoformat()
                marked = 0
                audit_rows = 0
                for addr in addresses:
                    ctx = ro.execute(
                        "SELECT COUNT(*) AS n, "
                        "       GROUP_CONCAT(DISTINCT alert_type) AS types "
                        "FROM alerts WHERE LOWER(address) = ? "
                        "  AND COALESCE(false_positive,0) = 0",
                        (addr,),
                    ).fetchone()
                    n_alerts = ctx[0] if ctx else 0
                    alert_types = ctx[1] if ctx else ""
                    if n_alerts == 0:
                        continue
                    wr.execute(
                        "UPDATE alerts SET false_positive = 1 WHERE LOWER(address) = ?",
                        (addr,),
                    )
                    marked += wr.total_changes if wr.total_changes > 0 else n_alerts
                    wr.execute(
                        """INSERT OR REPLACE INTO false_positives
                           (contract_address, chain, original_tier, original_patterns,
                            fp_reason, fp_method, fp_confidence, detector_blamed, assessed_at)
                           VALUES (?, 'unknown', 'alert_address', ?, ?, 'admin_bulk_mark',
                                   1.0, ?, ?)""",
                        (addr, alert_types or "", reason, detector_blamed, now_iso),
                    )
                    wr.commit()
                    audit_rows += 1
            except Exception as e:
                wr.rollback()
                self._json(500, {"error": f"mark-fp failed: {type(e).__name__}: {e}"})
                return
            finally:
                ro.close()
                wr.close()
            self._json(200, {
                "marked": marked, "audit_rows_written": audit_rows,
                "addresses": addresses, "reason": reason,
                "detector_blamed": detector_blamed,
            })

        elif self.path == "/admin/sync-mainnet-first-tx":
            # Batch UPDATE deployers.mainnet_first_tx from a local backfill.
            # Expects {"updates": {"0xaddr": "0xtxhash_or_empty", ...}}.
            # UPDATE-only: does not create new deployer rows.
            updates = data.get("updates", {})
            if not updates:
                self._json(400, {"error": "updates dict required: {address: mainnet_first_tx}"})
                return
            con = sqlite3.connect(DB_PATH, timeout=60)
            try:
                con.execute("PRAGMA busy_timeout=60000")
                updated = 0
                missing = 0
                for addr, tx in updates.items():
                    cur = con.execute(
                        "UPDATE deployers SET mainnet_first_tx = ? WHERE LOWER(deployer_address) = ?",
                        (tx, addr.lower()),
                    )
                    if cur.rowcount > 0:
                        updated += cur.rowcount
                    else:
                        missing += 1
                con.commit()
            finally:
                con.close()
            self._json(200, {"updated": updated, "missing": missing, "submitted": len(updates)})

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

        elif self.path == "/admin/eth-trace":
            # Ethereum mainnet depth trace — on-demand only
            if not ETH_HTTP:
                self._json(500, {"error": "ETH_WSS_URL not configured"})
                return
            addr = data.get("address", "").lower()
            check_only = data.get("check_only", False)
            hops = data.get("hops", 2)
            # batch mode: check multiple addresses for mainnet presence
            addresses = data.get("addresses", [])

            if not addr and not addresses:
                self._json(400, {"error": "address or addresses[] required"})
                return

            async def _run():
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from surveillance.eth_depth import (
                    trace_eth_mainnet, trace_l2_address_on_eth,
                    batch_check_mainnet_presence, ensure_tables,
                )
                con = sqlite3.connect(DB_PATH)
                ensure_tables(con)
                if addresses:
                    return await batch_check_mainnet_presence(ETH_HTTP, [a.lower() for a in addresses], con)
                elif check_only:
                    return await trace_l2_address_on_eth(ETH_HTTP, addr, con)
                else:
                    return await trace_eth_mainnet(ETH_HTTP, addr, con, max_hops=hops)

            result = asyncio.run(_run())
            self._json(200, result)

        elif self.path == "/admin/replay-gaps":
            # Replay historical blocks to recover missed transaction events
            chain = data.get("chain", "base")
            from_block = data.get("from_block")
            to_block = data.get("to_block")
            fill_gaps = data.get("fill_gaps", False)
            batch_size = data.get("batch", 50)

            if not fill_gaps and (not from_block or not to_block):
                self._json(400, {"error": "provide fill_gaps=true or from_block+to_block"})
                return

            # Run in background thread so it doesn't block the HTTP server
            import threading

            def _replay():
                try:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from surveillance.block_replayer import replay_blocks, fill_gaps as do_fill_gaps
                    if fill_gaps:
                        asyncio.run(do_fill_gaps(chain))
                    else:
                        asyncio.run(replay_blocks(chain, from_block, to_block, batch_size))
                except Exception as e:
                    print(f"[replay] Error: {e}", flush=True)

            t = threading.Thread(target=_replay, daemon=True)
            t.start()
            self._json(200, {
                "status": "started",
                "chain": chain,
                "fill_gaps": fill_gaps,
                "from_block": from_block,
                "to_block": to_block,
                "note": "Running in background. Check logs for progress."
            })

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

    def _handle_upload_db(self, body):
        """Emergency DB replacement — accepts gzipped or raw SQLite DB.

        Streams body to disk first, then decompresses if needed.
        """
        import gzip
        try:
            tmp_gz = DB_PATH + ".upload.gz"
            tmp_path = DB_PATH + ".new"

            # Check if it's gzipped (magic bytes 1f 8b)
            is_gzip = len(body) > 2 and body[0] == 0x1f and body[1] == 0x8b

            if is_gzip:
                # Write gzip to disk, then decompress on disk
                with open(tmp_gz, "wb") as f:
                    f.write(body)
                # Decompress streaming to avoid huge memory spike
                with gzip.open(tmp_gz, "rb") as gz_in, open(tmp_path, "wb") as out:
                    while True:
                        chunk = gz_in.read(4 * 1024 * 1024)  # 4MB chunks
                        if not chunk:
                            break
                        out.write(chunk)
                os.unlink(tmp_gz)
            else:
                # Raw SQLite — write directly
                with open(tmp_path, "wb") as f:
                    f.write(body)

            # Quick check — just verify it opens (skip full integrity for speed)
            c = sqlite3.connect(tmp_path)
            tables = c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()
            contracts = c.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
            c.close()

            # Replace
            backup = DB_PATH + ".corrupt"
            if os.path.exists(DB_PATH):
                os.rename(DB_PATH, backup)
            for ext in ["-wal", "-shm"]:
                p = DB_PATH + ext
                if os.path.exists(p):
                    os.unlink(p)
            os.rename(tmp_path, DB_PATH)
            self._json(200, {
                "status": "ok",
                "size_bytes": os.path.getsize(DB_PATH),
                "tables": tables[0],
                "contracts": contracts,
                "note": "DB replaced. Redeploy to restart monitors."
            })
        except Exception as e:
            # Cleanup temp files
            for p in [DB_PATH + ".upload.gz", DB_PATH + ".new"]:
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
            self._json(500, {"error": str(e)})

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

# ---- Startup DB cleanup (run before monitors start) ----
print("Running startup DB cleanup...", flush=True)
_db_ok = False
try:
    # Quick integrity check — skip cleanup entirely if DB is corrupted
    _test_conn = sqlite3.connect(DB_PATH, timeout=10)
    _integrity = _test_conn.execute("PRAGMA integrity_check").fetchone()[0]
    _test_conn.close()
    if _integrity != "ok":
        raise sqlite3.DatabaseError(f"DB corrupted: {_integrity}")
    _db_ok = True
except Exception as _db_err:
    print(f"  DB CORRUPTED: {_db_err}", flush=True)
    print("  Skipping cleanup. Upload a clean DB via POST /admin/upload-db", flush=True)

if _db_ok:
    try:
        _cleanup_conn = sqlite3.connect(DB_PATH, timeout=30)

        # 0. Ensure watchlist tables exist (may be missing on older Railway deploys)
        try:
            from surveillance.watchlist import ensure_tables as _ensure_wl
            _ensure_wl(_cleanup_conn)
            _cleanup_conn.commit()
            print("  Watchlist tables: ensured", flush=True)
        except Exception as _e:
            print(f"  Watchlist tables: {_e}", flush=True)

        # 1. Prune bytecode cache — remove ALL unknown entries (not just hit_count=0)
        _before = _cleanup_conn.execute("SELECT COUNT(*) FROM bytecode_cache").fetchone()[0]
        _cleanup_conn.execute("""
            DELETE FROM bytecode_cache WHERE confidence_tier = 'unknown'
        """)
        _after = _cleanup_conn.execute("SELECT COUNT(*) FROM bytecode_cache").fetchone()[0]
        print(f"  Bytecode cache: {_before} -> {_after} ({_before - _after} pruned)", flush=True)

        # 2. Deduplicate alerts table
        _alert_before = _cleanup_conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        _cleanup_conn.execute("""
            DELETE FROM alerts WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM alerts GROUP BY alert_type, address, tx_hash
            )
        """)
        _alert_after = _cleanup_conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        print(f"  Alerts: {_alert_before} -> {_alert_after} ({_alert_before - _alert_after} deduped)", flush=True)

        _cleanup_conn.commit()

        # 3. WAL checkpoint + VACUUM to reclaim disk space
        _cleanup_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print("  WAL checkpoint: TRUNCATE complete", flush=True)
        _cleanup_conn.execute("VACUUM")
        print("  VACUUM complete — disk space reclaimed", flush=True)

        # 4. Set WAL auto-checkpoint to smaller threshold (500 pages = ~2MB)
        _cleanup_conn.execute("PRAGMA wal_autocheckpoint = 500")

        _cleanup_conn.close()
        print("Startup cleanup complete.", flush=True)
    except Exception as e:
        print(f"Startup cleanup failed (non-fatal): {e}", flush=True)

# ---- Single-writer architecture ----
# One writer process owns the only read-write connection.
# All monitors push writes via multiprocessing.Queue.
# Stats server and analysis use read-only connections.
# WAL checkpoint is handled by the writer (trivial — single writer, no contention).

from surveillance.db_writer import run as _db_writer_run
from surveillance.process_entries import monitor_entry as _monitor_entry
from surveillance.process_entries import routing_entry as _routing_entry

# ---- Prereq: init_db + write smoke test ----
# Runs migrations in the parent before any child process opens the DB. This
# serializes migration work (no races between db_writer + 3 monitors), and the
# smoke test fails loudly if any column is missing before steady-state writes
# would start silently erroring. See Correction #10.
print("Running parent-process init_db + write smoke test...", flush=True)
from surveillance import db as _surv_db
_PREREQ_OK = False
try:
    import pathlib as _pl
    _prereq_conn = _surv_db.init_db(_pl.Path(DB_PATH))
    _surv_db.verify_write_path(_prereq_conn)
    _prereq_conn.close()
    _PREREQ_OK = True
    print("Prereq OK — spawning producer processes.", flush=True)
except sqlite3.OperationalError as _db_err:
    # DB is unreadable / corrupt at the file level. The correct recovery
    # path is POST /admin/upload-db, which requires the HTTP server to
    # stay reachable. Log loudly, skip producer spawn, keep the process
    # alive in degraded mode so /admin/upload-db can accept a clean DB.
    print(f"DEGRADED: DB unusable ({_db_err!r}). "
          f"Producers NOT spawned. /admin/upload-db ready to accept replacement.",
          flush=True)
except Exception as _prereq_err:
    print(f"FATAL: init_db/write_smoke failed: {_prereq_err}", flush=True)
    raise

_write_queue = MPQueue()

if _PREREQ_OK:
    # Start writer process (must be first — creates/owns the DB connection)
    _writer_proc = Process(
        target=_db_writer_run,
        args=(DB_PATH, _write_queue),
        daemon=True,
        name="db_writer",
    )
    _writer_proc.start()
    print(f"DB writer process started (pid={_writer_proc.pid})", flush=True)
else:
    _writer_proc = None
    print("DB writer NOT started (DB unusable).", flush=True)


# Start surveillance components
processes = []
if not _PREREQ_OK:
    print("Skipping producer spawn — DB unusable. HTTP server remains up for "
          "admin recovery (/admin/upload-db). Run a healthy deploy once the "
          "DB is restored.", flush=True)

# Arbitrum deployment monitor
arb_wss = os.environ.get("ARB_WSS_URL", "")
if arb_wss and _PREREQ_OK:
    p = Process(target=_monitor_entry, args=("arbitrum", arb_wss, _write_queue),
                daemon=True, name="monitor_arbitrum")
    p.start()
    processes.append(("arbitrum_monitor", p))
    print(f"Arbitrum monitor started (pid={p.pid})", flush=True)

# Base deployment monitor
base_wss = os.environ.get("BASE_WSS_URL", "")
if base_wss and _PREREQ_OK:
    p = Process(target=_monitor_entry, args=("base", base_wss, _write_queue),
                daemon=True, name="monitor_base")
    p.start()
    processes.append(("base_monitor", p))
    print(f"Base monitor started (pid={p.pid})", flush=True)

# Optimism deployment monitor
op_wss = os.environ.get("OP_WSS_URL", "")
if op_wss and _PREREQ_OK:
    p = Process(target=_monitor_entry, args=("optimism", op_wss, _write_queue),
                daemon=True, name="monitor_optimism")
    p.start()
    processes.append(("optimism_monitor", p))
    print(f"Optimism monitor started (pid={p.pid})", flush=True)

# Routing monitor
oneinch_key = os.environ.get("ONEINCH_API_KEY", "")
if oneinch_key and _PREREQ_OK:
    p = Process(target=_routing_entry, args=(oneinch_key, _write_queue),
                daemon=True, name="routing_monitor")
    p.start()
    processes.append(("routing", p))
    print(f"Routing monitor started (pid={p.pid})", flush=True)
elif _PREREQ_OK:
    # Routing monitor without API key — start in standalone mode
    routing = subprocess.Popen(
        [sys.executable, "-m", "surveillance.routing_monitor"],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    processes.append(("routing", routing))

# Daily report scheduler — runs at 06:00 UTC every day
def _daily_report_scheduler():
    """Background thread that generates the daily report at 06:00 UTC."""
    import time as _t
    while True:
        now = datetime.now(timezone.utc)
        # Calculate seconds until next 06:00 UTC
        target = now.replace(hour=6, minute=3, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        print(f"Daily report scheduled in {wait/3600:.1f}h ({target.isoformat()[:16]})", flush=True)
        _t.sleep(wait)
        try:
            print("Generating daily report...", flush=True)
            from surveillance.daily_report import generate_report
            path = generate_report()
            print(f"Daily report generated: {path}", flush=True)
        except Exception as e:
            print(f"Daily report failed: {e}", flush=True)

threading.Thread(target=_daily_report_scheduler, daemon=True).start()

# Analysis scheduler — nightly producers for derived-data tables.
# See reports/producer_scheduler_audit.md (Wave 2) and Correction #7 for
# the scope decision. Prior to this scheduler, these tables were written
# manually or by an external cron that died 2026-03-25 → 2026-03-26 and
# never ran again. Result: 22+ days of stale analysis served via API.
#
# Design:
#   - One poller thread checks every 60 s whether any job's UTC time matches.
#   - Each job runs as a subprocess via the existing CLI flags, so a crash
#     in one producer doesn't poison the scheduler.
#   - Dedupe by minute-fingerprint so a slow tick doesn't double-fire.
#   - Failures log at visible level (print with flush=True), not DEBUG.
#
# SQLite concurrency: producers open their own RW sqlite connections,
# competing with db_writer. Acceptable because (a) producers run at
# off-peak hours (00:15 → 04:10 UTC), (b) WAL mode + busy_timeout=5000
# serializes correctly.
ANALYSIS_JOBS = [
    # (hour_utc, minute_utc, days, label, argv)
    #   days: "daily" or "sunday"
    (0, 15, "daily",  "daily_metrics",
     [sys.executable, "-m", "surveillance.trend_forecaster", "--compute-today"]),
    (0, 20, "daily",  "camouflage_metrics",
     [sys.executable, "-m", "surveillance.camouflage_tracker", "--compute-today"]),
    (0, 30, "daily",  "predictions",
     [sys.executable, "-m", "surveillance.trend_forecaster", "--forecast", "--score"]),
    (2, 0,  "daily",  "deployer_profiles",
     [sys.executable, "-m", "surveillance.deployer_profiler", "--profile-all"]),
    (3, 0,  "daily",  "bytecode_families",
     [sys.executable, "-m", "surveillance.bytecode_families", "--cluster"]),
    (4, 0,  "sunday", "deployer_similarity",
     [sys.executable, "-m", "surveillance.deployer_profiler", "--cluster"]),
    (4, 30, "daily",  "confidence_decay",
     [sys.executable, "-m", "surveillance.confidence_decay", "--apply"]),
    (4, 45, "daily",  "org_candidates",
     [sys.executable, "-m", "surveillance.org_candidates", "--apply"]),
]
_JOB_TIMEOUT_SEC = 3600  # 1 h per producer; deployer_profiler is the slowest
_analysis_last_fired: dict[str, str] = {}

def _run_analysis_job(label: str, argv: list) -> None:
    """Invoke a producer subprocess and log the outcome. Never raises."""
    cmd_str = " ".join(argv[2:])  # drop python -m prefix for readability
    print(f"[analysis_scheduler] firing {label}: {cmd_str}", flush=True)
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=_JOB_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        print(f"[analysis_scheduler] {label} TIMEOUT after {_JOB_TIMEOUT_SEC}s",
              flush=True)
        return
    except Exception as e:
        print(f"[analysis_scheduler] {label} EXCEPTION "
              f"{type(e).__name__}: {e}", flush=True)
        return
    if result.returncode == 0:
        print(f"[analysis_scheduler] {label} OK", flush=True)
    else:
        print(f"[analysis_scheduler] {label} FAILED rc={result.returncode}",
              flush=True)
        # Include tail of stderr for diagnosis
        tail = (result.stderr or "")[-1000:]
        if tail.strip():
            print(f"[analysis_scheduler] {label} stderr tail:\n{tail}",
                  flush=True)

def _analysis_scheduler():
    """Poll every minute; fire any job whose (hour, minute, dow) matches."""
    import time as _t
    # First tick log so operators see the scheduler is alive
    print(f"Analysis scheduler started — {len(ANALYSIS_JOBS)} jobs configured",
          flush=True)
    for hour, minute, days, label, _ in ANALYSIS_JOBS:
        print(f"  {hour:02d}:{minute:02d} UTC {days:6s}  {label}", flush=True)
    while True:
        try:
            _t.sleep(60)
            now = datetime.now(timezone.utc)
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            for hour, minute, days, label, argv in ANALYSIS_JOBS:
                if now.hour != hour or now.minute != minute:
                    continue
                if days == "sunday" and now.weekday() != 6:
                    continue
                if _analysis_last_fired.get(label) == minute_key:
                    continue  # already fired this minute
                _analysis_last_fired[label] = minute_key
                _run_analysis_job(label, argv)
        except Exception as e:
            # Belt-and-braces: never let the scheduler thread die
            print(f"[analysis_scheduler] tick error: "
                  f"{type(e).__name__}: {e}", flush=True)

threading.Thread(target=_analysis_scheduler, daemon=True).start()

# Keep alive — restart any process that dies
import time as _time

def _make_proc(name):
    """Recreate a monitor process by name."""
    if name == "arbitrum_monitor" and os.environ.get("ARB_WSS_URL"):
        p = Process(target=_monitor_entry, args=("arbitrum", os.environ["ARB_WSS_URL"], _write_queue),
                    daemon=True, name="monitor_arbitrum")
        p.start()
        return p
    elif name == "base_monitor" and os.environ.get("BASE_WSS_URL"):
        p = Process(target=_monitor_entry, args=("base", os.environ["BASE_WSS_URL"], _write_queue),
                    daemon=True, name="monitor_base")
        p.start()
        return p
    elif name == "optimism_monitor" and os.environ.get("OP_WSS_URL"):
        p = Process(target=_monitor_entry, args=("optimism", os.environ["OP_WSS_URL"], _write_queue),
                    daemon=True, name="monitor_optimism")
        p.start()
        return p
    elif name == "routing":
        _key = os.environ.get("ONEINCH_API_KEY", "")
        if _key:
            p = Process(target=_routing_entry, args=(_key, _write_queue),
                        daemon=True, name="routing_monitor")
            p.start()
            return p
        else:
            return subprocess.Popen(
                [sys.executable, "-m", "surveillance.routing_monitor"],
                stdout=sys.stdout, stderr=sys.stderr,
            )
    return None


def _proc_alive(proc):
    """Check if a process (multiprocessing.Process or subprocess.Popen) is still running."""
    if isinstance(proc, Process):
        return proc.is_alive()
    else:
        return proc.poll() is None


while True:
    for i, (name, proc) in enumerate(processes):
        if not _proc_alive(proc):
            exit_info = proc.exitcode if isinstance(proc, Process) else proc.poll()
            print(f"Process {name} exited ({exit_info}) — restarting...", flush=True)
            new_proc = _make_proc(name)
            if new_proc:
                processes[i] = (name, new_proc)
                pid = new_proc.pid
                print(f"Process {name} restarted (pid={pid})", flush=True)
            else:
                print(f"Cannot restart {name}", flush=True)
    # Also check writer (if prereq allowed it to start)
    if _writer_proc is not None and not _writer_proc.is_alive():
        print("DB writer died — restarting...", flush=True)
        _writer_proc = Process(target=_db_writer_run, args=(DB_PATH, _write_queue),
                               daemon=True, name="db_writer")
        _writer_proc.start()
        print(f"DB writer restarted (pid={_writer_proc.pid})", flush=True)
    _time.sleep(5)
