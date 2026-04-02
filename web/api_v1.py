"""
Layer 3 — Public API v1

Revenue-generating endpoints for contract screening, intelligence feed,
and organizational intelligence. All reads from SQLite, zero API calls.

Mount on the main app:
    from web.api_v1 import router as v1_router
    app.include_router(v1_router, prefix="/api/v1")
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

DB_PATH = os.environ.get(
    "DB_PATH",
    str(Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"),
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ro_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _rw_conn() -> sqlite3.Connection:
    """For api_keys/api_watches tables only."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

def _meta(conn: sqlite3.Connection) -> dict:
    corpus = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    last = conn.execute(
        "SELECT MAX(detection_timestamp) FROM contracts"
    ).fetchone()[0]
    return {
        "corpus_size": corpus,
        "chains": ["base", "arbitrum", "optimism"],
        "last_updated": last,
        "api_version": "1.0",
    }


def _ok(data: dict, conn: sqlite3.Connection) -> dict:
    return {"status": "ok", "data": data, "meta": _meta(conn)}


def _error(code: str, message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        {"status": "error", "error": {"code": code, "message": message}},
        status_code=status,
    )


# ---------------------------------------------------------------------------
# Authentication middleware
# ---------------------------------------------------------------------------

TIER_ROUTES = {
    1: {"/risk", "/check", "/screen"},
    2: {"/risk", "/check", "/screen", "/feed", "/feed/stats", "/watch", "/ecosystem/stats"},
    3: None,  # all routes
}


async def require_auth(request: Request, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key")
    key = authorization[7:]

    conn = _rw_conn()
    try:
        from surveillance.api_keys import ensure_tables, validate_key, track_usage
        ensure_tables(conn)
        record = validate_key(conn, key)
        if not record:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Rate limit check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        requests = record["requests_today"] if record.get("requests_reset_date") == today else 0
        if requests >= record["rate_limit"]:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Tier check
        path = request.url.path.replace("/api/v1", "")
        # Extract the base route (e.g., /risk/base/0x... -> /risk)
        parts = path.strip("/").split("/")
        base_route = "/" + parts[0] if parts else "/"
        allowed = TIER_ROUTES.get(record["tier"])
        if allowed is not None and base_route not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Endpoint requires Tier {_min_tier(base_route)} access",
            )

        track_usage(conn, key)
        conn.commit()
    finally:
        conn.close()

    return record


def _min_tier(route: str) -> int:
    for tier in [1, 2, 3]:
        allowed = TIER_ROUTES.get(tier)
        if allowed is None or route in allowed:
            return tier
    return 3


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _confidence_score(row: dict, victim_count: int = 0) -> Optional[float]:
    tier = row.get("confidence_tier", "unknown")
    if tier == "confirmed":
        base = 0.90
        bonus = min(victim_count / 1000, 0.09)  # up to 0.99
        return round(base + bonus, 2)
    if tier == "suspected":
        sigs = sum(1 for k in ("has_asymmetric_transfer", "has_conditional_revert",
                                "has_unusual_fee_structure") if row.get(k))
        if sigs >= 3:
            return round(0.70 + sigs * 0.05, 2)
        if sigs >= 1:
            return round(0.50 + sigs * 0.08, 2)
        return 0.35
    return None


def _risk_level(confidence: Optional[float], tier: str) -> str:
    if tier == "confirmed":
        return "CRITICAL"
    if confidence is None:
        return "UNKNOWN"
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.50:
        return "MEDIUM"
    if confidence >= 0.30:
        return "LOW"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Phase 2: Tier 1 — Contract Screening
# ---------------------------------------------------------------------------

@router.get("/risk/{chain}/{address}", dependencies=[Depends(require_auth)])
async def risk_by_chain(chain: str, address: str):
    addr = address.lower()
    conn = _ro_conn()
    try:
        row = conn.execute(
            "SELECT * FROM contracts WHERE contract_address = ? AND chain = ?",
            (addr, chain),
        ).fetchone()
        if not row:
            return _error("NOT_FOUND",
                          "Address not found in monitored corpus. Layer 3 covers Base, Arbitrum, and Optimism.",
                          404)
        return JSONResponse(_ok(_build_risk(conn, dict(row)), conn))
    finally:
        conn.close()


@router.get("/check/{address}", dependencies=[Depends(require_auth)])
async def check_address(address: str):
    addr = address.lower()
    conn = _ro_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM contracts WHERE contract_address = ?", (addr,)
        ).fetchall()
        if not rows:
            return _error("NOT_FOUND",
                          "Address not found in monitored corpus. Layer 3 covers Base, Arbitrum, and Optimism.",
                          404)
        results = [_build_risk(conn, dict(r)) for r in rows]
        if len(results) == 1:
            return JSONResponse(_ok(results[0], conn))
        return JSONResponse(_ok({"results": results}, conn))
    finally:
        conn.close()


class ScreenRequest(BaseModel):
    addresses: list[str]
    chain: Optional[str] = None


@router.post("/screen", dependencies=[Depends(require_auth)])
async def screen_batch(body: ScreenRequest):
    if len(body.addresses) > 100:
        return _error("TOO_MANY", "Maximum 100 addresses per batch", 400)
    conn = _ro_conn()
    try:
        results = []
        for addr in body.addresses:
            a = addr.lower()
            if body.chain:
                row = conn.execute(
                    "SELECT * FROM contracts WHERE contract_address = ? AND chain = ?",
                    (a, body.chain),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM contracts WHERE contract_address = ?", (a,)
                ).fetchone()
            if not row:
                results.append({"address": addr, "risk_level": None, "monitored": False})
            else:
                r = dict(row)
                victims = _get_victim_count(conn, a)
                conf = _confidence_score(r, victims)
                results.append({
                    "address": addr,
                    "chain": r["chain"],
                    "risk_level": _risk_level(conf, r["confidence_tier"]),
                    "confidence": conf,
                    "tier": r["confidence_tier"],
                    "monitored": True,
                })
        return JSONResponse(_ok({"results": results}, conn))
    finally:
        conn.close()


def _get_victim_count(conn: sqlite3.Connection, address: str) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT interacting_address) FROM transaction_events WHERE contract_address = ?",
        (address,),
    ).fetchone()
    return row[0] if row else 0


def _get_revert_stats(conn: sqlite3.Connection, address: str) -> dict:
    row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN is_reverted = 1 THEN 1 ELSE 0 END) as reverts,
               COUNT(DISTINCT interacting_address) as callers
        FROM transaction_events WHERE contract_address = ?
    """, (address,)).fetchone()
    if not row or row[0] == 0:
        return {"total": 0, "reverts": 0, "callers": 0, "revert_rate": 0.0}
    return {
        "total": row[0],
        "reverts": row[1],
        "callers": row[2],
        "revert_rate": round(row[1] / row[0], 4) if row[0] else 0.0,
    }


def _build_risk(conn: sqlite3.Connection, row: dict) -> dict:
    addr = row["contract_address"]
    stats = _get_revert_stats(conn, addr)
    victims = stats["callers"]
    conf = _confidence_score(row, victims)
    risk = _risk_level(conf, row["confidence_tier"])

    # Detection info
    sigs = []
    if row.get("has_asymmetric_transfer"):
        sigs.append("asymmetric_transfer")
    if row.get("has_conditional_revert"):
        sigs.append("conditional_revert")
    if row.get("has_unusual_fee_structure"):
        sigs.append("unusual_fee_structure")
    if row.get("bytecode_pattern_notes"):
        notes = row["bytecode_pattern_notes"]
        if "SELFDESTRUCT" in notes:
            sigs.append("selfdestruct_in_token")
        if "delegatecall" in notes.lower():
            sigs.append("delegatecall")
        if "blacklist" in notes.lower() or "SLOAD" in notes:
            sigs.append("blacklist_check")

    # Attribution
    deployer_addr = row.get("deployer_address", "")
    attribution = {"deployer": deployer_addr}
    if deployer_addr:
        ec = conn.execute(
            "SELECT org_id, category, subtype FROM entity_classification WHERE address = ?",
            (deployer_addr,),
        ).fetchone()
        if ec:
            attribution["org_id"] = ec[0]
            attribution["deployer_entity_type"] = ec[1]

        dep = conn.execute(
            "SELECT total_contracts_deployed, funding_trail FROM deployers WHERE deployer_address = ?",
            (deployer_addr,),
        ).fetchone()
        if dep:
            attribution["deployer_contracts_total"] = dep[0]
            if dep[1]:
                try:
                    trail = json.loads(dep[1])
                    attribution["funder"] = trail.get("funder")
                    if trail.get("org_link"):
                        attribution["org_id"] = trail["org_link"]
                except (json.JSONDecodeError, TypeError):
                    pass

        profile = conn.execute(
            "SELECT org_link, timezone_guess, primary_technique FROM deployer_profiles WHERE deployer_address = ?",
            (deployer_addr,),
        ).fetchone()
        if profile:
            if profile[0] and not attribution.get("org_id"):
                attribution["org_id"] = profile[0]
            attribution["timezone"] = profile[1]
            attribution["technique"] = profile[2]

    # Approval exposure
    approval = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN drain_detected = 1 THEN 1 ELSE 0 END) as drained
        FROM approval_watchlist WHERE contract_address = ?
    """, (addr,)).fetchone()
    approval_data = None
    if approval and approval[0] > 0:
        approval_data = {
            "pending_approvals": approval[0] - (approval[1] or 0),
            "approvals_drained": approval[1] or 0,
        }

    # Bytecode family
    family = conn.execute(
        "SELECT bf.family_id, bf.family_name FROM bytecode_family_members bfm "
        "JOIN bytecode_families bf ON bfm.family_id = bf.family_id "
        "WHERE bfm.contract_address = ?",
        (addr,),
    ).fetchone()
    family_data = None
    if family:
        fam_size = conn.execute(
            "SELECT member_count, unique_deployers FROM bytecode_families WHERE family_id = ?",
            (family[0],),
        ).fetchone()
        family_data = {
            "family_id": family[0],
            "family_name": family[1],
            "family_size": fam_size[0] if fam_size else None,
            "family_deployers": fam_size[1] if fam_size else None,
        }

    return {
        "address": addr,
        "chain": row["chain"],
        "risk_level": risk,
        "confidence": conf,
        "tier": row["confidence_tier"],
        "detection": {
            "method": row.get("detection_method"),
            "trap_signatures": sigs,
            "signature_count": len(sigs),
            "behavioral_confirmed": row["confidence_tier"] == "confirmed",
            "confirmed_at": row.get("confirmation_timestamp"),
        },
        "impact": {
            "unique_victims": victims,
            "total_interactions": stats["total"],
            "revert_rate": stats["revert_rate"],
            "camouflaged": stats["revert_rate"] < 0.10 if stats["total"] >= 10 else None,
        },
        "attribution": attribution,
        "approval_exposure": approval_data,
        "bytecode_family": family_data,
        "context": {
            "first_seen": row.get("detection_timestamp"),
            "last_updated": row.get("last_updated"),
            "detection_block": row.get("detection_block"),
        },
    }


# ---------------------------------------------------------------------------
# Phase 3: Tier 2 — Intelligence Feed
# ---------------------------------------------------------------------------

@router.get("/feed", dependencies=[Depends(require_auth)])
async def feed(
    since: str = None,
    severity: str = None,
    type: str = None,
    org: str = None,
    limit: int = 50,
    offset: int = 0,
):
    if not since:
        return _error("MISSING_PARAM", "'since' parameter is required (ISO timestamp)", 400)
    if limit > 200:
        limit = 200
    conn = _ro_conn()
    try:
        conditions = ["timestamp >= ?"]
        params: list = [since]

        if severity:
            sevs = [s.strip().upper() for s in severity.split(",")]
            # Map severity to alert types
            type_map = {
                "CRITICAL": ("TRAP_CONFIRMED", "APPROVAL_DRAIN"),
                "HIGH": ("HIGH_VELOCITY_DEPLOYER", "WATCHLIST_HIT"),
            }
            mapped = []
            for s in sevs:
                mapped.extend(type_map.get(s, ()))
            if mapped:
                placeholders = ",".join("?" * len(mapped))
                conditions.append(f"alert_type IN ({placeholders})")
                params.extend(mapped)

        if type:
            types = [t.strip() for t in type.split(",")]
            placeholders = ",".join("?" * len(types))
            conditions.append(f"alert_type IN ({placeholders})")
            params.extend(types)

        where = " AND ".join(conditions)
        sql = f"SELECT * FROM alerts WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        total = conn.execute(
            f"SELECT COUNT(*) FROM alerts WHERE {' AND '.join(conditions)}",
            params[:-2],
        ).fetchone()[0]

        alerts = []
        for r in rows:
            payload = {}
            if r.get("payload"):
                try:
                    payload = json.loads(r["payload"])
                except (json.JSONDecodeError, TypeError):
                    pass
            alerts.append({
                "id": f"alert_{r.get('id', '')}",
                "timestamp": r.get("timestamp"),
                "type": r.get("alert_type"),
                "severity": _alert_severity(r.get("alert_type")),
                "address": r.get("address"),
                "tx_hash": r.get("tx_hash"),
                "summary": payload.get("message", ""),
                "details": payload,
            })

        return JSONResponse(_ok({
            "alerts": alerts,
            "total": total,
            "has_more": offset + limit < total,
        }, conn))
    finally:
        conn.close()


def _alert_severity(alert_type: str) -> str:
    return {
        "TRAP_CONFIRMED": "CRITICAL",
        "APPROVAL_DRAIN": "CRITICAL",
        "HIGH_VELOCITY_DEPLOYER": "HIGH",
        "WATCHLIST_HIT": "HIGH",
    }.get(alert_type or "", "INFO")


@router.get("/feed/stats", dependencies=[Depends(require_auth)])
async def feed_stats():
    conn = _ro_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()

        def _count_since(hours):
            cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            # Approximate: just use the timestamp string comparison
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            rows = conn.execute("""
                SELECT alert_type, COUNT(*) FROM alerts
                WHERE timestamp >= ? GROUP BY alert_type
            """, (cutoff,)).fetchall()
            result = {"critical": 0, "high": 0, "medium": 0, "info": 0}
            for r in rows:
                sev = _alert_severity(r[0]).lower()
                result[sev] = result.get(sev, 0) + r[1]
            return result

        # Active drains (approvals not yet drained on confirmed contracts)
        pending = conn.execute("""
            SELECT COUNT(*) FROM approval_watchlist
            WHERE drain_detected = 0 AND contract_tier IN ('suspected', 'confirmed')
        """).fetchone()[0]

        return JSONResponse(_ok({
            "last_24h": _count_since(24),
            "last_7d": _count_since(168),
            "pending_approvals_at_risk": pending,
        }, conn))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 3: Watch endpoints
# ---------------------------------------------------------------------------

class WatchRequest(BaseModel):
    address: str
    chain: Optional[str] = None
    alert_types: Optional[list[str]] = None


@router.post("/watch", dependencies=[Depends(require_auth)])
async def create_watch(body: WatchRequest, authorization: str = Header(None)):
    key = authorization[7:] if authorization else ""
    conn = _rw_conn()
    try:
        from surveillance.api_keys import ensure_tables
        ensure_tables(conn)
        conn.execute(
            "INSERT INTO api_watches (api_key, address, chain, alert_types, created_at) VALUES (?, ?, ?, ?, ?)",
            (key, body.address.lower(), body.chain,
             json.dumps(body.alert_types) if body.alert_types else None,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return JSONResponse({"status": "ok", "data": {"address": body.address.lower(), "watching": True}})
    finally:
        conn.close()


@router.get("/watch", dependencies=[Depends(require_auth)])
async def list_watches(authorization: str = Header(None)):
    key = authorization[7:] if authorization else ""
    conn = _ro_conn()
    try:
        rows = conn.execute(
            "SELECT address, chain, alert_types, created_at FROM api_watches WHERE api_key = ?",
            (key,),
        ).fetchall()
        watches = []
        for r in rows:
            at = None
            if r[2]:
                try:
                    at = json.loads(r[2])
                except (json.JSONDecodeError, TypeError):
                    pass
            watches.append({
                "address": r[0], "chain": r[1],
                "alert_types": at, "created_at": r[3],
            })
        return JSONResponse({"status": "ok", "data": {"watches": watches}})
    finally:
        conn.close()


@router.delete("/watch/{address}", dependencies=[Depends(require_auth)])
async def delete_watch(address: str, authorization: str = Header(None)):
    key = authorization[7:] if authorization else ""
    conn = _rw_conn()
    try:
        conn.execute(
            "DELETE FROM api_watches WHERE api_key = ? AND address = ?",
            (key, address.lower()),
        )
        conn.commit()
        return JSONResponse({"status": "ok", "data": {"address": address.lower(), "watching": False}})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 4: Tier 3 — Full Intelligence
# ---------------------------------------------------------------------------

@router.get("/org", dependencies=[Depends(require_auth)])
async def list_orgs():
    conn = _ro_conn()
    try:
        orgs = conn.execute("""
            SELECT org_id, COUNT(DISTINCT address) as deployers
            FROM entity_classification
            WHERE org_id IS NOT NULL AND org_id != ''
            GROUP BY org_id ORDER BY deployers DESC
        """).fetchall()

        results = []
        for o in orgs:
            org_id = o[0]
            contracts = conn.execute("""
                SELECT COUNT(*) FROM contracts c
                JOIN entity_classification ec ON c.deployer_address = ec.address
                WHERE ec.org_id = ?
            """, (org_id,)).fetchone()[0]
            results.append({
                "org_id": org_id,
                "deployers": o[1],
                "contracts": contracts,
            })

        # Also check deployer_profiles.org_link
        profile_orgs = conn.execute("""
            SELECT org_link, COUNT(*) as deployers
            FROM deployer_profiles
            WHERE org_link IS NOT NULL AND org_link != ''
            GROUP BY org_link ORDER BY deployers DESC
        """).fetchall()
        existing = {r["org_id"] for r in results}
        for po in profile_orgs:
            if po[0] not in existing:
                results.append({"org_id": po[0], "deployers": po[1], "contracts": 0})

        return JSONResponse(_ok({"organizations": results}, conn))
    finally:
        conn.close()


@router.get("/org/{org_id}", dependencies=[Depends(require_auth)])
async def org_detail(org_id: str):
    conn = _ro_conn()
    try:
        # Deployers in this org
        deployers = conn.execute("""
            SELECT DISTINCT address FROM entity_classification WHERE org_id = ?
            UNION
            SELECT DISTINCT deployer_address FROM deployer_profiles WHERE org_link = ?
        """, (org_id, org_id)).fetchall()
        dep_addrs = [d[0] for d in deployers]

        if not dep_addrs:
            # Check funding_trail
            dep_addrs = [d[0] for d in conn.execute("""
                SELECT deployer_address FROM deployers
                WHERE funding_trail LIKE ?
            """, (f'%"org_link": "{org_id}"%',)).fetchall()]

        if not dep_addrs:
            return _error("NOT_FOUND", f"Organization '{org_id}' not found", 404)

        # Contracts
        placeholders = ",".join("?" * len(dep_addrs))
        contracts = conn.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT chain) FROM contracts
            WHERE deployer_address IN ({placeholders})
        """, dep_addrs).fetchone()

        chains = [r[0] for r in conn.execute(f"""
            SELECT DISTINCT chain FROM contracts
            WHERE deployer_address IN ({placeholders})
        """, dep_addrs).fetchall()]

        # Tempo
        profiles = conn.execute(f"""
            SELECT timezone_guess, peak_hour, primary_technique
            FROM deployer_profiles WHERE deployer_address IN ({placeholders})
        """, dep_addrs).fetchall()
        tz = Counter_vals([p[0] for p in profiles if p[0]])
        techniques = Counter_vals([p[2] for p in profiles if p[2]])

        return JSONResponse(_ok({
            "org_id": org_id,
            "scale": {
                "deployers": len(dep_addrs),
                "contracts": contracts[0],
                "chains": chains,
            },
            "operational_tempo": {
                "timezone_inference": tz[0] if tz else None,
                "techniques": techniques[:3],
            },
        }, conn))
    finally:
        conn.close()


def Counter_vals(items):
    """Return items sorted by frequency."""
    from collections import Counter
    return [item for item, _ in Counter(items).most_common()]


@router.get("/deployer/{address}", dependencies=[Depends(require_auth)])
async def deployer_detail(address: str):
    addr = address.lower()
    conn = _ro_conn()
    try:
        dep = conn.execute(
            "SELECT * FROM deployers WHERE deployer_address = ?", (addr,)
        ).fetchone()
        if not dep:
            return _error("NOT_FOUND", "Deployer not found", 404)
        dep = dict(dep)

        # Profile
        profile = conn.execute(
            "SELECT * FROM deployer_profiles WHERE deployer_address = ?", (addr,)
        ).fetchone()
        profile = dict(profile) if profile else {}

        # Funding
        funding = {}
        if dep.get("funding_trail"):
            try:
                funding = json.loads(dep["funding_trail"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Similar deployers
        similar = conn.execute("""
            SELECT deployer_a, deployer_b, composite_score, timezone_score,
                   gas_score, technique_score
            FROM deployer_similarity
            WHERE (deployer_a = ? OR deployer_b = ?) AND composite_score >= 0.70
            ORDER BY composite_score DESC LIMIT 10
        """, (addr, addr)).fetchall()
        sim_list = []
        for s in similar:
            other = s[1] if s[0] == addr else s[0]
            matching = []
            if s[3] and s[3] >= 0.7:
                matching.append("timezone")
            if s[4] and s[4] >= 0.7:
                matching.append("gas")
            if s[5] and s[5] >= 0.7:
                matching.append("technique")
            sim_list.append({
                "address": other,
                "similarity": round(s[2], 3),
                "matching_features": matching,
            })

        # Org attribution
        ec = conn.execute(
            "SELECT org_id, category FROM entity_classification WHERE address = ?",
            (addr,),
        ).fetchone()
        org = ec[0] if ec else funding.get("org_link") or profile.get("org_link")

        return JSONResponse(_ok({
            "address": addr,
            "entity_type": dep.get("entity_type"),
            "org_attribution": org,
            "profile": {
                "contracts_deployed": dep.get("total_contracts_deployed"),
                "chain": dep.get("chain"),
                "first_seen": dep.get("first_seen"),
                "last_seen": dep.get("last_seen"),
                "technique": profile.get("primary_technique"),
                "timezone": profile.get("timezone_guess"),
                "avg_gas_gwei": profile.get("gas_avg"),
                "deployment_style": profile.get("deployment_style"),
            },
            "funding": {
                "source": funding.get("funder"),
                "value_eth": funding.get("value_eth"),
                "org_link": funding.get("org_link"),
            },
            "similar_deployers": sim_list,
        }, conn))
    finally:
        conn.close()


@router.get("/contract/{address}", dependencies=[Depends(require_auth)])
async def contract_detail(address: str):
    addr = address.lower()
    conn = _ro_conn()
    try:
        row = conn.execute(
            "SELECT * FROM contracts WHERE contract_address = ?", (addr,)
        ).fetchone()
        if not row:
            return _error("NOT_FOUND", "Contract not found", 404)
        return JSONResponse(_ok(_build_risk(conn, dict(row)), conn))
    finally:
        conn.close()


@router.get("/ecosystem/stats", dependencies=[Depends(require_auth)])
async def ecosystem_stats():
    conn = _ro_conn()
    try:
        contracts = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        deployers = conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
        bots = conn.execute("SELECT COUNT(*) FROM bot_candidates").fetchone()[0]
        confirmed = conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'confirmed'"
        ).fetchone()[0]
        suspected = conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'suspected'"
        ).fetchone()[0]
        traps_24h = conn.execute("""
            SELECT COUNT(*) FROM trap_events
            WHERE timestamp >= datetime('now', '-1 day')
        """).fetchone()[0]
        orgs = conn.execute("""
            SELECT COUNT(DISTINCT org_id) FROM entity_classification
            WHERE org_id IS NOT NULL AND org_id != ''
        """).fetchone()[0]
        rotations = conn.execute(
            "SELECT COUNT(*) FROM deployer_similarity WHERE composite_score >= 0.80"
        ).fetchone()[0]

        # Camouflage
        cam = conn.execute(
            "SELECT camouflage_ratio FROM camouflage_metrics ORDER BY date DESC LIMIT 1"
        ).fetchone()
        cam_ratio = round(cam[0], 3) if cam else None

        return JSONResponse(_ok({
            "corpus": {
                "contracts": contracts,
                "deployers": deployers,
                "bots": bots,
                "chains": 3,
            },
            "detection": {
                "confirmed_threats": confirmed,
                "suspected_threats": suspected,
                "camouflage_ratio": cam_ratio,
                "behavioral_confirmations_24h": traps_24h,
            },
            "organizations": {
                "mapped": orgs,
                "wallet_rotations_detected": rotations,
            },
        }, conn))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 5: Docs
# ---------------------------------------------------------------------------

@router.get("/docs")
async def api_docs():
    docs_path = Path(__file__).resolve().parent / "static" / "API_DOCS.md"
    if docs_path.exists():
        return JSONResponse({"status": "ok", "data": {"docs": docs_path.read_text(encoding="utf-8")}})
    return JSONResponse({"status": "ok", "data": {"docs": "API documentation coming soon."}})
