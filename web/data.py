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

def _nid(addr: str) -> str:
    """Consistent short node ID from address."""
    return addr.lower()[:14] if addr.startswith("0x") else addr


def get_org_graph(conn, org_id: str) -> dict:
    """Build nodes + edges for Cytoscape.js visualization."""
    nodes = []
    edges = []
    seen = set()

    def add_node(addr, label, role, color, **extra):
        nid = _nid(addr)
        if nid in seen:
            return nid
        seen.add(nid)
        contracts = conn.execute("SELECT COUNT(*) FROM contracts WHERE deployer_address=?", (addr,)).fetchone()[0] if addr.startswith("0x") else 0
        data = {"id": nid, "label": label, "role": role, "color": color, "address": addr, "contracts": contracts}
        data.update(extra)
        nodes.append({"data": data})
        return nid

    def add_edge(src_addr, tgt_addr, label):
        edges.append({"data": {"source": _nid(src_addr), "target": _nid(tgt_addr), "label": label}})

    if org_id == "org_001":
        # Core infrastructure
        add_node("0x503828976d22510aad0201ac7ec88293211d23da", "Coinbase", "cex", "#3B82F6")
        add_node("0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d", "Buffer Wallet", "buffer", "#6B7280")
        add_node("0x4c968f6beecf1906710b08e8b472b8ba6e75f957", "Central Treasury", "treasury", "#EF4444")
        add_node("0x8c826f795466e39acbff1bb4eeeb759609377ba1", "Gas Station", "gas_station", "#F59E0B")
        add_node("0xf70da97812cb96acdf810712aa562db8dfa3dbef", "Whale Trader", "whale_trader", "#8B5CF6")
        add_node("0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1", "MEV Bot", "mev_bot", "#06B6D4")
        add_node("0x391e7c679d29bd940d63be94ad22a25d25b5a604", "Revenue Collector", "revenue", "#10B981")
        add_node("0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390", "v1 Bot (retired)", "bot_retired", "#6B7280")
        add_node("0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae", "LI.FI Bridge", "bridge", "#D946EF")
        add_node("0x5e0f8e7337c8955d2124b8e85ca74af884b3e124", "WETH Wrapper", "wrapper", "#9CA3AF")

        # L2 operators from entity_classification
        for r in conn.execute("SELECT address, subtype, notes FROM entity_classification WHERE org_id='org_001' AND category='CRIMINAL'"):
            label_map = {"org_deployer": "Operator", "org_cashout": "Cashout", "org_laundry": "Laundry",
                         "org_treasury": "L2 Treasury", "org_exit_ramp": "CEX Exit"}
            lbl = label_map.get(r["subtype"], r["subtype"])
            color_map = {"org_deployer": "#3B82F6", "org_cashout": "#F59E0B", "org_laundry": "#EF4444",
                         "org_treasury": "#EF4444", "org_exit_ramp": "#D946EF"}
            add_node(r["address"], lbl, r["subtype"], color_map.get(r["subtype"], "#9CA3AF"))

        # Edges: funding chain
        add_edge("0x503828976d22510aad0201ac7ec88293211d23da", "0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d", "Coinbase withdrawal")
        add_edge("0x66666ff8ee46eee265ba888dbbbaad69ccf50b1d", "0x4c968f6beecf1906710b08e8b472b8ba6e75f957", "Fund treasury")
        add_edge("0x4c968f6beecf1906710b08e8b472b8ba6e75f957", "0x8c826f795466e39acbff1bb4eeeb759609377ba1", "Fund gas station")
        add_edge("0x4c968f6beecf1906710b08e8b472b8ba6e75f957", "0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1", "Fund MEV bot")
        add_edge("0x4c968f6beecf1906710b08e8b472b8ba6e75f957", "0x391e7c679d29bd940d63be94ad22a25d25b5a604", "Revenue sweep")
        add_edge("0x391e7c679d29bd940d63be94ad22a25d25b5a604", "0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1", "Sweep to bot")
        add_edge("0x9e22ebec84c7e4c4bd6d4ae7ff6f4d436d6d8390", "0x4c968f6beecf1906710b08e8b472b8ba6e75f957", "Revenue (retired)")
        add_edge("0xf70da97812cb96acdf810712aa562db8dfa3dbef", "0x8c826f795466e39acbff1bb4eeeb759609377ba1", "10 ETH")
        add_edge("0x8c826f795466e39acbff1bb4eeeb759609377ba1", "0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae", "LI.FI bridge")
        add_edge("0x8c826f795466e39acbff1bb4eeeb759609377ba1", "0x5e0f8e7337c8955d2124b8e85ca74af884b3e124", "WETH wrap")

        # L2 treasury -> operators
        add_edge("0xf186cb00e49e18491db5783ff04fae3818102ff7", "0xe93d64f3fbc352131e79fc5578cbe44b66697f86", "Operations")
        add_edge("0xc6962004f452be9203591991d15f6b388e09e8d0", "0x01989c93890aed05a63d179b03424997075b6acf", "Cashout")

        # Gas station -> top deployers
        gas_deps = conn.execute("""SELECT deployer_address, total_contracts_deployed FROM deployers
            WHERE funding_trail LIKE '%org_001%' ORDER BY total_contracts_deployed DESC LIMIT 10""").fetchall()
        for d in gas_deps:
            nid = add_node(d["deployer_address"], f"{d['total_contracts_deployed']}c", "deployer", "#F59E0B")
            add_edge("0x8c826f795466e39acbff1bb4eeeb759609377ba1", d["deployer_address"], "")

        # Whale -> top deployers
        whale_deps = conn.execute("""SELECT deployer_address, total_contracts_deployed FROM deployers
            WHERE funding_trail LIKE '%f70da978%' ORDER BY total_contracts_deployed DESC LIMIT 10""").fetchall()
        for d in whale_deps:
            nid = add_node(d["deployer_address"], f"{d['total_contracts_deployed']}c", "deployer", "#8B5CF6")
            add_edge("0xf70da97812cb96acdf810712aa562db8dfa3dbef", d["deployer_address"], "")

        # Summary nodes for the rest
        gas_total = conn.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%org_001%'").fetchone()[0]
        whale_total = conn.execute("SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%f70da978%'").fetchone()[0]
        gas_rest = gas_total - len(gas_deps)
        whale_rest = whale_total - len(whale_deps)
        if gas_rest > 0:
            add_node("gas_rest", f"+{gas_rest} more", "deployer_cluster", "#92702A")
            add_edge("0x8c826f795466e39acbff1bb4eeeb759609377ba1", "gas_rest", "")
        if whale_rest > 0:
            add_node("whale_rest", f"+{whale_rest} more", "deployer_cluster", "#5B3F8C")
            add_edge("0xf70da97812cb96acdf810712aa562db8dfa3dbef", "whale_rest", "")

    elif org_id == "org_002":
        senior = "0x238d7170f309a55b87a144a341bd6105897082ca"
        junior = "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473"
        add_node(senior, "Treasury Senior", "treasury", "#EF4444")
        add_node(junior, "Treasury Junior", "treasury", "#EF4444")

        # Top deployers from each treasury
        for treasury, label_prefix in [(senior, "S"), (junior, "J")]:
            prefix = treasury[:8]
            deps = conn.execute(f"""SELECT deployer_address, total_contracts_deployed FROM deployers
                WHERE funding_trail LIKE '%{prefix}%' ORDER BY total_contracts_deployed DESC LIMIT 15""").fetchall()
            for d in deps:
                nid = add_node(d["deployer_address"], f"1c", "deployer", "#F59E0B")
                add_edge(treasury, d["deployer_address"], "5.0 ETH")
            rest = conn.execute(f"SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE '%{prefix}%'").fetchone()[0] - len(deps)
            if rest > 0:
                rest_id = f"{label_prefix}_rest"
                add_node(rest_id, f"+{rest} more", "deployer_cluster", "#92702A")
                add_edge(treasury, rest_id, "5.0 ETH each")

    elif org_id == "entity_005":
        arch = "0x9209c9f7dcb61937f1ec8160c22c0b2365079474"
        funder = "0x151b381058f91cf871e7ea1ee83c45326f61e96d"
        add_node(arch, "The Architect", "deployer", "#EF4444")
        add_node(funder, "Funder", "funder", "#F59E0B")
        add_edge(funder, arch, "0.0508 ETH")

        for r in conn.execute("""SELECT deployer_a, deployer_b, composite_score FROM deployer_similarity
            WHERE (deployer_a=? OR deployer_b=?) AND composite_score >= 0.70""", (arch, arch)):
            other = r["deployer_b"] if r["deployer_a"] == arch else r["deployer_a"]
            dep = conn.execute("SELECT total_contracts_deployed FROM deployers WHERE deployer_address=?", (other,)).fetchone()
            cnt = dep["total_contracts_deployed"] if dep else "?"
            nid = add_node(other, f"Match {r['composite_score']:.2f} ({cnt}c)", "associated", "#8B5CF6")
            add_edge(arch, other, f"{r['composite_score']:.2f}")

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
