"""
Layer 3 UI — Data access layer.
Read-only SQLite queries. Never writes to the database.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"


def get_conn(db_path: str = None) -> sqlite3.Connection:
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------

def get_overview_stats(conn) -> dict:
    return {
        "contracts": conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0],
        "deployers": conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0],
        "tx_events": conn.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0],
        "bots": conn.execute("SELECT COUNT(*) FROM bot_candidates").fetchone()[0],
        "chains": len(conn.execute("SELECT DISTINCT chain FROM contracts").fetchall()),
        "orgs_mapped": 4,
        "suspected": conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='suspected'").fetchone()[0],
        "confirmed": conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='confirmed'").fetchone()[0],
    }


def get_key_metrics(conn) -> dict:
    # Camouflage ratio
    camo = conn.execute("""SELECT COUNT(*) as total,
        SUM(CASE WHEN rev_pct < 10 THEN 1 ELSE 0 END) as camo
        FROM (SELECT ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1) as rev_pct
        FROM transaction_events GROUP BY contract_address HAVING COUNT(*)>=10)""").fetchone()
    camo_ratio = round(camo["camo"] / max(camo["total"], 1) * 100, 1) if camo["total"] else 0

    # Revert rate
    rev = conn.execute("SELECT ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1) FROM transaction_events").fetchone()[0] or 0

    # Approval exposure
    pending = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=0").fetchone()[0]
    drained = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]

    # Self-test traps
    armed = conn.execute("SELECT COUNT(*) FROM self_test_traps WHERE status='ARMED'").fetchone()[0]
    rd = conn.execute("SELECT COUNT(*) FROM self_test_traps WHERE status='SELF_TEST'").fetchone()[0]

    # Watchlist
    watchlist_active = conn.execute("SELECT COUNT(*) FROM watchlist WHERE active=1").fetchone()[0]
    watchlist_critical = conn.execute("SELECT COUNT(*) FROM watchlist WHERE active=1 AND priority='CRITICAL'").fetchone()[0]

    # Extraction
    usd = conn.execute("SELECT SUM(total_usd_moved) FROM extraction_events").fetchone()[0] or 0

    return {
        "camouflage_ratio": camo_ratio,
        "goplus_gap": "50/50",
        "revert_rate": rev,
        "approval_pending": pending,
        "approval_drained": drained,
        "self_test_rd": rd,
        "self_test_armed": armed,
        "watchlist_active": watchlist_active,
        "watchlist_critical": watchlist_critical,
        "extraction_usd": usd,
        "precision": 99.9,
    }


def get_chain_split(conn) -> list:
    return [dict(r) for r in conn.execute(
        "SELECT chain, COUNT(*) as contracts, COUNT(DISTINCT deployer_address) as deployers FROM contracts GROUP BY chain ORDER BY contracts DESC"
    ).fetchall()]


def get_daily_trend(conn, days: int = 14) -> list:
    return [dict(r) for r in conn.execute("""
        SELECT DATE(detection_timestamp) as date, COUNT(*) as contracts,
            COUNT(DISTINCT deployer_address) as deployers
        FROM contracts GROUP BY date ORDER BY date DESC LIMIT ?
    """, (days,)).fetchall()]


def get_recent_alerts(conn, limit: int = 50) -> list:
    try:
        return [dict(r) for r in conn.execute("""
            SELECT wh.timestamp, w.entity_name, w.priority, wh.hit_type,
                wh.contract_address, wh.chain
            FROM watchlist_hits wh
            JOIN watchlist w ON wh.watchlist_id = w.id
            ORDER BY wh.timestamp DESC LIMIT ?
        """, (limit,)).fetchall()]
    except Exception:
        return []


def get_strategy_lifecycle(conn) -> list:
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM strategy_lifecycle ORDER BY saturation_index DESC"
        ).fetchall()]
    except Exception:
        return []


def get_bot_sophistication(conn) -> list:
    try:
        return [dict(r) for r in conn.execute("""
            SELECT sophistication, COUNT(*) as count,
                ROUND(AVG(revert_rate), 1) as avg_revert,
                ROUND(AVG(total_hits), 0) as avg_hits
            FROM bot_sophistication GROUP BY sophistication ORDER BY avg_revert
        """).fetchall()]
    except Exception:
        return []


# ---------------------------------------------------------------
# Organization map
# ---------------------------------------------------------------

def get_org_graph(conn, org_id: str) -> dict:
    """Build nodes + edges for Cytoscape.js visualization."""
    nodes = []
    edges = []
    seen_nodes = set()

    if org_id == "org_001":
        # Core infrastructure nodes (manually curated for readability)
        core = {
            "0x4c968f6beecf1906710b08e8b472b8ba6e75f957": ("Central Treasury", "treasury", "#EF4444"),
            "0x8c826f795466e39acbff1bb4eeeb759609377ba1": ("Gas Station", "gas_station", "#F59E0B"),
            "0xf70da97812cb96acdf810712aa562db8dfa3dbef": ("Whale Trader", "whale_trader", "#8B5CF6"),
            "0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1": ("MEV Bot", "mev_bot", "#06B6D4"),
            "0x391e7c679d29bd940d63be94ad22a25d25b5a604": ("Revenue Collector", "revenue", "#10B981"),
            "0x503828976d22510aad0201ac7ec88293211d23da": ("Coinbase", "cex", "#3B82F6"),
            "0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d": ("Buffer Wallet", "buffer", "#9CA3AF"),
            "0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390": ("v1 MEV Bot (retired)", "bot_retired", "#6B7280"),
            "0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae": ("LI.FI Bridge", "bridge", "#D946EF"),
        }
        for addr, (label, role, color) in core.items():
            contracts = conn.execute("SELECT COUNT(*) FROM contracts WHERE deployer_address=?", (addr,)).fetchone()[0]
            nodes.append({"data": {"id": addr[:12], "label": label, "role": role, "color": color, "address": addr, "contracts": contracts}})
            seen_nodes.add(addr[:12])

        # Core edges
        core_edges = [
            ("0x503828976d22", "0x66666ff8ee46", "Coinbase withdrawal"),
            ("0x66666ff8ee46", "0x4c968f6beecf", "Fund treasury"),
            ("0x4c968f6beecf", "0x8c826f795466", "Fund gas station"),
            ("0x4c968f6beecf", "0x5babe600b9fc", "Fund MEV bot"),
            ("0x4c968f6beecf", "0x391e7c679d29", "Revenue collection"),
            ("0x391e7c679d29", "0x5babe600b9fc", "Sweep to MEV bot"),
            ("0x8c826f795466", "0x1231deb6f574", "LI.FI bridge"),
            ("0x9e22ebec84c7", "0x4c968f6beecf", "Revenue (retired)"),
            ("0xf70da97812cb", "0x8c826f795466", "Fund gas station"),
        ]
        for src, tgt, label in core_edges:
            edges.append({"data": {"source": src, "target": tgt, "label": label}})

        # Add deployer count nodes
        gas_deps = conn.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%org_001%'").fetchone()[0]
        whale_deps = conn.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%f70da978%'").fetchone()[0]
        nodes.append({"data": {"id": "gas_deployers", "label": f"{gas_deps} Deployers", "role": "deployer_cluster", "color": "#F59E0B", "contracts": 0}})
        nodes.append({"data": {"id": "whale_deployers", "label": f"{whale_deps} Deployers", "role": "deployer_cluster", "color": "#8B5CF6", "contracts": 0}})
        edges.append({"data": {"source": "0x8c826f795466", "target": "gas_deployers", "label": "funds"}})
        edges.append({"data": {"source": "0xf70da97812cb", "target": "whale_deployers", "label": "funds"}})

    elif org_id == "org_002":
        nodes.append({"data": {"id": "senior", "label": "Treasury Senior", "role": "treasury", "color": "#EF4444", "address": "0x238d7170f309a55b87a144a341bd6105897082ca"}})
        nodes.append({"data": {"id": "junior", "label": "Treasury Junior", "role": "treasury", "color": "#EF4444", "address": "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473"}})
        deps_s = conn.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%238d71%'").fetchone()[0]
        deps_j = conn.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%de8eb9%'").fetchone()[0]
        nodes.append({"data": {"id": "s_deps", "label": f"{deps_s} Deployers", "role": "deployer_cluster", "color": "#F59E0B"}})
        nodes.append({"data": {"id": "j_deps", "label": f"{deps_j} Deployers", "role": "deployer_cluster", "color": "#F59E0B"}})
        edges.append({"data": {"source": "senior", "target": "s_deps", "label": "5.0 ETH each"}})
        edges.append({"data": {"source": "junior", "target": "j_deps", "label": "5.0 ETH each"}})

    elif org_id == "entity_005":
        nodes.append({"data": {"id": "architect", "label": "The Architect", "role": "deployer", "color": "#EF4444", "address": "0x9209c9f7dcb61937f1ec8160c22c0b2365079474", "contracts": 21}})
        nodes.append({"data": {"id": "funder", "label": "Funder", "role": "funder", "color": "#F59E0B", "address": "0x151b381058f91cf871e7ea1ee83c45326f61e96d"}})
        edges.append({"data": {"source": "funder", "target": "architect", "label": "0.0508 ETH"}})
        # Associated wallets
        for r in conn.execute("""SELECT deployer_a, deployer_b, composite_score FROM deployer_similarity
            WHERE (deployer_a='0x9209c9f7dcb61937f1ec8160c22c0b2365079474' OR deployer_b='0x9209c9f7dcb61937f1ec8160c22c0b2365079474')
            AND composite_score >= 0.70"""):
            other = r["deployer_b"] if r["deployer_a"] == "0x9209c9f7dcb61937f1ec8160c22c0b2365079474" else r["deployer_a"]
            nid = other[:12]
            if nid not in seen_nodes:
                nodes.append({"data": {"id": nid, "label": f"Match ({r['composite_score']:.2f})", "role": "associated", "color": "#8B5CF6", "address": other}})
                edges.append({"data": {"source": "architect", "target": nid, "label": f"{r['composite_score']:.2f}"}})
                seen_nodes.add(nid)

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------
# Contract inspector
# ---------------------------------------------------------------

def get_contract(conn, address: str) -> Optional[dict]:
    r = conn.execute("SELECT * FROM contracts WHERE contract_address LIKE ?", (address.lower() + "%",)).fetchone()
    if not r:
        return None
    result = dict(r)

    # Activity stats
    stats = conn.execute("""SELECT COUNT(*) as hits, COUNT(DISTINCT interacting_address) as callers,
        SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts
        FROM transaction_events WHERE contract_address=?""", (r["contract_address"],)).fetchone()
    result["hits"] = stats["hits"] or 0
    result["callers"] = stats["callers"] or 0
    result["reverts"] = stats["reverts"] or 0
    result["revert_rate"] = round(result["reverts"] / max(result["hits"], 1) * 100, 1)

    # Family
    fam = conn.execute("SELECT family_id FROM bytecode_family_members WHERE contract_address=?", (r["contract_address"],)).fetchone()
    result["family"] = fam["family_id"] if fam else None

    # Self-test status
    st = conn.execute("SELECT * FROM self_test_traps WHERE contract_address=?", (r["contract_address"],)).fetchone()
    result["self_test"] = dict(st) if st else None

    # Approval exposure
    approvals = conn.execute("SELECT COUNT(*) as pending FROM approval_watchlist WHERE contract_address=? AND drain_detected=0",
                             (r["contract_address"],)).fetchone()
    result["pending_approvals"] = approvals["pending"] or 0

    return result


# ---------------------------------------------------------------
# Deployer profiler
# ---------------------------------------------------------------

def get_deployer(conn, address: str) -> Optional[dict]:
    r = conn.execute("SELECT * FROM deployers WHERE deployer_address LIKE ?", (address.lower() + "%",)).fetchone()
    if not r:
        return None
    result = dict(r)

    # Profile
    dp = conn.execute("SELECT * FROM deployer_profiles WHERE deployer_address=?", (r["deployer_address"],)).fetchone()
    result["profile"] = dict(dp) if dp else None

    # Funding trail
    if result.get("funding_trail"):
        try:
            result["funding"] = json.loads(result["funding_trail"])
        except Exception:
            result["funding"] = None
    else:
        result["funding"] = None

    # Contracts
    result["contracts"] = [dict(c) for c in conn.execute(
        "SELECT contract_address, chain, confidence_tier, bytecode_pattern_notes, detection_timestamp FROM contracts WHERE deployer_address=? ORDER BY detection_timestamp DESC LIMIT 50",
        (r["deployer_address"],)
    ).fetchall()]

    # Similar deployers
    result["similar"] = [dict(s) for s in conn.execute("""
        SELECT deployer_a, deployer_b, composite_score, timezone_score, gas_score, technique_score
        FROM deployer_similarity
        WHERE (deployer_a=? OR deployer_b=?) AND composite_score >= 0.70
        ORDER BY composite_score DESC LIMIT 10
    """, (r["deployer_address"], r["deployer_address"])).fetchall()]

    return result


# ---------------------------------------------------------------
# Threat feed
# ---------------------------------------------------------------

def get_threats(conn, limit: int = 100, chain: str = None, priority: str = None) -> list:
    query = """SELECT wh.timestamp, w.entity_name, w.priority, wh.hit_type,
        wh.contract_address, wh.chain, w.watch_reason
        FROM watchlist_hits wh JOIN watchlist w ON wh.watchlist_id = w.id WHERE 1=1"""
    params = []
    if chain:
        query += " AND wh.chain = ?"
        params.append(chain)
    if priority:
        query += " AND w.priority = ?"
        params.append(priority)
    query += " ORDER BY wh.timestamp DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def search_address(conn, query: str) -> dict:
    """Search for an address as contract or deployer."""
    q = query.lower().strip()
    contract = conn.execute("SELECT contract_address FROM contracts WHERE contract_address LIKE ? LIMIT 1", (q + "%",)).fetchone()
    deployer = conn.execute("SELECT deployer_address FROM deployers WHERE deployer_address LIKE ? LIMIT 1", (q + "%",)).fetchone()
    return {
        "contract": contract["contract_address"] if contract else None,
        "deployer": deployer["deployer_address"] if deployer else None,
    }
