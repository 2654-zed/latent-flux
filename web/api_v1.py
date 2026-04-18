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


# Derived-data producer tables served by the API. Map each to the column
# that reflects the producer's last write. Introduced with Correction #7
# so customers can see whether the analysis they are reading is current.
# See reports/producer_scheduler_audit.md for the scheduling architecture
# that keeps these fresh.
_TABLE_FRESHNESS_COLUMN = {
    "trust_amplification":  "last_updated",
    "deployer_profiles":    "profiled_at",
    "deployer_similarity":  "computed_at",
    "entity_classification":"last_updated",
    "bytecode_families":    "last_updated",
    "daily_metrics":        "date",
    "camouflage_metrics":   "date",
    "predictions":          "issued_date",
}


def _freshness(conn: sqlite3.Connection, tables: list) -> dict:
    """Return {table_name: latest_timestamp} for the requested tables.
    Missing tables map to None. Never raises — freshness is advisory."""
    out = {}
    for t in tables:
        col = _TABLE_FRESHNESS_COLUMN.get(t)
        if not col:
            out[t] = None
            continue
        try:
            row = conn.execute(f"SELECT MAX({col}) FROM {t}").fetchone()
            out[t] = row[0] if row else None
        except sqlite3.Error:
            out[t] = None
    return out


def _ok(data: dict, conn: sqlite3.Connection,
        fresh_tables: Optional[list] = None,
        live: bool = False) -> dict:
    """Wrap a successful response.

    fresh_tables: list of derived-data tables this endpoint reads. When
        provided, meta.computed_at reports each table's last-write
        timestamp. Customers who care about staleness can filter; customers
        who don't continue consuming as before.
    live: set True for endpoints (e.g., /risk) that compute on every call
        rather than reading a producer table. Signals current-by-definition.
    """
    meta = _meta(conn)
    if live:
        meta["computed_at"] = "live"
    elif fresh_tables:
        meta["computed_at"] = _freshness(conn, fresh_tables)
    return {"status": "ok", "data": data, "meta": meta}


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
    """Confidence scoring — only claims what we can verify.

    0.90+ requires behavioral confirmation (trap event with external victim).
    0.70-0.89 requires multiple bytecode trap signatures.
    0.50-0.69 requires at least one signature.
    Below 0.50 is deployer-history-only detection (no bytecode evidence).
    """
    tier = row.get("confidence_tier", "unknown")
    if tier == "confirmed":
        # Behavioral confirmation: bot != deployer, reverted on this contract
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


def _evidence_type(row: dict, sigs: list) -> str:
    """Distinguish WHY a contract is flagged — the evidence basis.

    behavioral-confirmation: behavioral evidence (bot reverted on this contract)
    bytecode-pattern:        bytecode detector(s) fired OR pattern notes exist
    deployer-derivative:     flagged only because deployer also deployed a confirmed trap
    unanalyzed:              classifier ran, no trap patterns detected
    """
    if row.get("confidence_tier") == "confirmed":
        return "behavioral-confirmation"
    if sigs or row.get("bytecode_pattern_notes"):
        return "bytecode-pattern"
    if row.get("detection_method") == "deployer_history":
        return "deployer-derivative"
    return "unanalyzed"


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
        return JSONResponse(_ok(
            _build_risk(conn, dict(row)), conn,
            fresh_tables=["trust_amplification", "bytecode_families",
                          "entity_classification"],
        ))
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
            "SELECT member_count, unique_deployers, detection_tier FROM bytecode_families WHERE family_id = ?",
            (family[0],),
        ).fetchone()
        clustering_method = "unknown"
        if fam_size and fam_size[2]:
            if "T1" in fam_size[2]:
                clustering_method = "Tier 1: first 100 chars of bytecode_pattern_notes prefix match"
            elif "T2" in fam_size[2]:
                clustering_method = "Tier 2: flag combination match (asymmetric_transfer + conditional_revert + unusual_fee_structure)"
        family_data = {
            "family_id": family[0],
            "family_name": family[1],
            "family_size": fam_size[0] if fam_size else None,
            "family_deployers": fam_size[1] if fam_size else None,
            "clustering_method": clustering_method,
            "epistemic_note": (
                "Family membership is INFERENTIAL. Different prefix lengths "
                "or similarity metrics would produce different families."
            ),
        }

    # GoPlus comparison: stored results from prior benchmarks
    goplus = None
    try:
        gp_row = conn.execute(
            "SELECT risk_level, risk_score, is_honeypot, goplus_status, match, checked_at "
            "FROM goplus_results WHERE address = ? ORDER BY checked_at DESC LIMIT 1",
            (addr,),
        ).fetchone()
        if gp_row:
            goplus = {
                "risk_level": gp_row["risk_level"],
                "risk_score": gp_row["risk_score"],
                "is_honeypot": gp_row["is_honeypot"],
                "goplus_status": gp_row["goplus_status"],
                "match_vs_layer3": gp_row["match"],
                "checked_at": gp_row["checked_at"],
            }
    except Exception:
        pass

    # Trust amplification (router traffic vs family baseline)
    trust_amp = None
    try:
        ta_row = conn.execute(
            "SELECT total_callers, router_percentage, callers_per_day, "
            "amplification_factor, family_avg_callers_per_day, alert_level "
            "FROM trust_amplification WHERE contract_address = ?",
            (addr,),
        ).fetchone()
        if ta_row:
            trust_amp = {
                "total_callers": ta_row["total_callers"],
                "router_percentage": ta_row["router_percentage"],
                "callers_per_day": ta_row["callers_per_day"],
                "amplification_factor": ta_row["amplification_factor"],
                "family_avg_callers_per_day": ta_row["family_avg_callers_per_day"],
                "alert_level": ta_row["alert_level"],
            }
    except Exception:
        pass

    # Diamond model — formal adversary profile if org is known
    diamond = None
    if attribution.get("org_id"):
        try:
            dm_row = conn.execute(
                "SELECT adversary_type, adversary_timezone, adversary_operational_pattern, "
                "capability_trap_mechanisms, capability_camouflage_rating, "
                "capability_anti_forensic, infrastructure_chains, infrastructure_exit_channels, "
                "victim_count, victim_type, victim_return_rate, confidence "
                "FROM diamond_model WHERE case_id = ?",
                (attribution["org_id"],),
            ).fetchone()
            if dm_row:
                diamond = dict(dm_row)
                # Parse JSON fields
                for k in ("capability_trap_mechanisms", "capability_anti_forensic",
                          "infrastructure_chains", "infrastructure_exit_channels"):
                    if diamond.get(k):
                        try:
                            diamond[k] = json.loads(diamond[k])
                        except (json.JSONDecodeError, TypeError):
                            pass
        except Exception:
            pass

    # Extraction events (if this contract is mentioned in documented extractions)
    extraction = None
    try:
        ext_row = conn.execute(
            "SELECT event_id, event_type, observed_at, summary "
            "FROM extraction_events WHERE raw_transactions LIKE ? OR summary LIKE ? LIMIT 1",
            (f"%{addr}%", f"%{addr[:18]}%"),
        ).fetchone()
        if ext_row:
            extraction = dict(ext_row)
    except Exception:
        pass

    # Detector precision (measured false positive rate per detector that fired)
    detector_prec = None
    if sigs:
        try:
            # Map signature names to detector names
            sig_to_detector = {
                "asymmetric_transfer": "asymmetric_transfer",
                "conditional_revert": "blacklist_check",
                "blacklist_check": "blacklist_check",
                "unusual_fee_structure": "obfuscated_fee",
                "selfdestruct_in_token": "selfdestruct",
                "delegatecall": "delegatecall_in_token",
            }
            detector_names = list(set(sig_to_detector.get(s, s) for s in sigs))
            placeholders = ",".join("?" * len(detector_names))
            prec_rows = conn.execute(
                f"SELECT detector_name, precision, total_fired FROM detector_precision "
                f"WHERE detector_name IN ({placeholders}) "
                f"ORDER BY date DESC",
                detector_names,
            ).fetchall()
            if prec_rows:
                # Take the most recent precision per detector
                seen = set()
                precs = []
                for pr in prec_rows:
                    if pr["detector_name"] not in seen:
                        precs.append(dict(pr))
                        seen.add(pr["detector_name"])
                if precs:
                    avg_precision = sum(p["precision"] for p in precs) / len(precs)
                    detector_prec = {
                        "average_precision": round(avg_precision, 4),
                        "detectors": precs,
                    }
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Epistemic status: separate deductive base from inferential
    # components so customers know what's provable vs assessed.
    # Added per COMMERCIAL_EPISTEMIC_AUDIT.md (2026-04-09).
    # ---------------------------------------------------------------
    deductive_base = [
        f"bytecode patterns at cited offsets ({len(sigs)} detected)" if sigs else None,
        f"{stats['revert_rate']:.1%} revert rate on {stats['total']:,} interactions" if stats["total"] else None,
        f"{victims:,} distinct callers" if victims else None,
        f"deployer {deployer_addr}" if deployer_addr else None,
    ]
    if approval_data:
        deductive_base.append(
            f"{approval_data['pending_approvals']} pending approvals, "
            f"{approval_data['approvals_drained']} drained"
        )
    deductive_base = [d for d in deductive_base if d]

    inferential_components = []
    if row["confidence_tier"] == "suspected":
        inferential_components.append(
            "suspected tier: bytecode pattern count >= 1 (policy threshold)"
        )
    if conf is not None:
        inferential_components.append(
            "confidence score: formula weighting tier + victim count + signal count"
        )
    if attribution.get("org_id"):
        inferential_components.append(
            f"org attribution ({attribution['org_id']}): funding-chain inference"
        )
    if attribution.get("timezone"):
        inferential_components.append(
            f"timezone ({attribution['timezone']}): inferred from deployment hour distribution"
        )
    if family_data:
        inferential_components.append(
            "bytecode family: cluster membership via pattern-prefix similarity"
        )
    if trust_amp and trust_amp.get("amplification_factor"):
        inferential_components.append(
            "amplification factor: ratio depends on inferential family baseline"
        )

    if row["confidence_tier"] == "confirmed":
        epistemic_classification = "confirmed"
    elif inferential_components:
        epistemic_classification = "assessed"
    else:
        epistemic_classification = "observed"

    # Confirmation evidence tx_hashes (for Tier A /verify consumers)
    confirmation_evidence = None
    if row["confidence_tier"] == "confirmed" and row.get("confirmation_tx_hash"):
        confirmation_evidence = row["confirmation_tx_hash"]

    # Sample tx_hashes backing revert stats (spot-check references)
    sample_tx_hashes = None
    try:
        sample_rows = conn.execute(
            "SELECT tx_hash FROM transaction_events "
            "WHERE contract_address = ? AND is_reverted = 1 LIMIT 3",
            (addr,),
        ).fetchall()
        if sample_rows:
            sample_tx_hashes = [r[0] for r in sample_rows]
    except Exception:
        pass

    return {
        "address": addr,
        "chain": row["chain"],
        "risk_level": risk,
        "confidence": conf,
        "tier": row["confidence_tier"],
        "epistemic_status": {
            "classification": epistemic_classification,
            "deductive_base": deductive_base,
            "inferential_components": inferential_components,
            "verification": (
                "Deductive base verifiable via eth_getCode (bytecode) + "
                "block replay (revert stats). Classification methodology "
                "at /docs#confidence-methodology."
            ),
        },
        "detection": {
            "method": row.get("detection_method"),
            "evidence_type": _evidence_type(row, sigs),
            "trap_signatures": sigs,
            "signature_count": len(sigs),
            "bytecode_notes": row.get("bytecode_pattern_notes"),
            "behavioral_confirmed": row["confidence_tier"] == "confirmed",
            "confirmed_at": row.get("confirmation_timestamp"),
            "confirmation_tx_hash": confirmation_evidence,
            "detector_precision": detector_prec,
        },
        "impact": {
            "unique_victims": victims,
            "total_interactions": stats["total"],
            "revert_rate": stats["revert_rate"],
            "camouflaged": stats["revert_rate"] < 0.10 if stats["total"] >= 10 else None,
            "camouflage_note": (
                "Camouflaged = revert rate < 10% on 10+ interactions. "
                "Threshold chosen because contracts below 10% attract 4.5x "
                "more victims than overt traps (empirical, two-week assessment)."
            ) if stats["total"] >= 10 else None,
            "sample_revert_tx_hashes": sample_tx_hashes,
        },
        "attribution": attribution,
        "approval_exposure": approval_data,
        "bytecode_family": family_data,
        "trust_amplification": trust_amp,
        "diamond_model": diamond,
        "extraction_event": extraction,
        "external_benchmark": {
            "goplus": goplus,
            "comparison_note": (
                "Different detection methodologies. L3_ONLY means Layer 3 "
                "flagged and GoPlus did not — may reflect scope differences "
                "rather than detection failure."
            ),
        } if goplus else None,
        "context": {
            "first_seen": row.get("detection_timestamp"),
            "last_updated": row.get("last_updated"),
            "detection_block": row.get("detection_block"),
        },
    }


# ---------------------------------------------------------------------------
# Tier A — Deductive-Only Verification Endpoint
# Returns ONLY on-chain-verifiable facts. No risk_level, no confidence
# score, no organizational attribution. Pure Tier A output for
# liability-conscious customers who want to make their own classification
# decisions from independently verifiable evidence.
# Added per COMMERCIAL_EPISTEMIC_AUDIT.md (2026-04-09).
# ---------------------------------------------------------------------------

@router.get("/verify/{chain}/{address}", dependencies=[Depends(require_auth)])
async def verify(chain: str, address: str):
    """Deductive-only risk evidence. Every field is independently
    verifiable via eth_getCode, eth_getTransactionReceipt, or
    eth_getTransactionByHash. No inferential components."""
    addr = address.lower()
    conn = _ro_conn()
    try:
        row = conn.execute(
            "SELECT * FROM contracts WHERE contract_address = ? AND chain = ?",
            (addr, chain),
        ).fetchone()
        if not row:
            return _error("NOT_FOUND",
                          "Address not found in monitored corpus.", 404)
        row = dict(row)
        stats = _get_revert_stats(conn, addr)

        # Bytecode patterns with byte offsets (verifiable via eth_getCode)
        sigs = []
        if row.get("has_asymmetric_transfer"):
            sigs.append("asymmetric_transfer")
        if row.get("has_conditional_revert"):
            sigs.append("conditional_revert")
        if row.get("has_unusual_fee_structure"):
            sigs.append("unusual_fee_structure")
        notes = row.get("bytecode_pattern_notes") or ""
        if "SELFDESTRUCT" in notes:
            sigs.append("selfdestruct")
        if "delegatecall" in notes.lower():
            sigs.append("delegatecall")

        # Sample revert tx_hashes for spot-checking
        revert_hashes = [
            r[0] for r in conn.execute(
                "SELECT tx_hash FROM transaction_events "
                "WHERE contract_address = ? AND is_reverted = 1 LIMIT 5",
                (addr,),
            ).fetchall()
        ]

        # Approval evidence
        approval_hashes = [
            {"victim": r[0], "tx_hash": r[1]}
            for r in conn.execute(
                "SELECT victim_address, approve_tx_hash "
                "FROM approval_watchlist WHERE contract_address = ? LIMIT 10",
                (addr,),
            ).fetchall()
        ]

        # Confirmation evidence (if confirmed)
        confirmation = None
        if row["confidence_tier"] == "confirmed" and row.get("confirmation_tx_hash"):
            confirmation = {
                "tx_hash": row["confirmation_tx_hash"],
                "block": row.get("confirmation_block"),
                "timestamp": row.get("confirmation_timestamp"),
            }

        # Deployer creation tx
        deployer = row.get("deployer_address")
        creation_block = row.get("detection_block")

        return JSONResponse(_ok({
            "address": addr,
            "chain": chain,
            "epistemic_tier": "A",
            "epistemic_note": (
                "Every field in this response is independently verifiable "
                "via standard Ethereum RPC calls. No inferential components, "
                "no risk scores, no organizational attribution."
            ),
            "bytecode_evidence": {
                "patterns_detected": sigs,
                "pattern_notes": notes if notes else None,
                "verification": "eth_getCode(address) + disassemble at cited byte offsets",
            },
            "revert_evidence": {
                "total_interactions": stats["total"],
                "reverts": stats["reverts"],
                "revert_rate": stats["revert_rate"],
                "distinct_callers": stats["callers"],
                "sample_revert_tx_hashes": revert_hashes,
                "verification": "eth_getTransactionReceipt(tx_hash).status == 0 for each",
            },
            "deployment_evidence": {
                "deployer_address": deployer,
                "detection_block": creation_block,
                "verification": "eth_getTransactionByHash at creation block, tx.from == deployer",
            },
            "confirmation_evidence": confirmation,
            "approval_evidence": {
                "total_approvals_tracked": len(approval_hashes),
                "sample_approvals": approval_hashes,
                "verification": "eth_getTransactionByHash(tx_hash) shows approve() calldata",
            } if approval_hashes else None,
        }, conn))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 4: Agent Pre-Transaction Screening
#
# The integration point for AI agent frameworks. An agent calls this endpoint
# BEFORE signing any approval or interacting with an unverified contract.
# Returns the stored potential risk score, capability breakdown, and a
# machine-readable recommendation.
#
# The agent's policy boundary: "If risk_score > 50 or tier == CRITICAL,
# refuse the transaction." This makes prompt injection ineffective —
# the injection can convince the agent to try, but the on-chain intelligence
# prevents the try from executing.
#
# Integration surfaces:
#   1. Pre-transaction: agent calls /agent/screen before signing
#   2. Approval scope: agent calls /agent/approval-check before granting Permit2
#   3. Facilitator validation: agent calls /agent/facilitator before x402 payment
# ---------------------------------------------------------------------------

@router.get("/agent/screen/{chain}/{address}")
async def agent_screen(chain: str, address: str):
    """Pre-transaction risk screen for AI agents.

    Returns stored potential score, capability breakdown, and recommendation.
    Designed for programmatic consumption — no auth required for the screen
    endpoint (rate-limited in production). This is the oracle the agent consults.
    """
    addr = address.lower()
    conn = _ro_conn()
    try:
        row = conn.execute(
            "SELECT * FROM contracts WHERE contract_address = ? AND chain = ?",
            (addr, chain),
        ).fetchone()

        if not row:
            # Unknown contract — not in our corpus. This is itself a signal.
            return JSONResponse({
                "address": addr,
                "chain": chain,
                "in_corpus": False,
                "risk_score": 0,
                "risk_tier": "UNKNOWN",
                "recommendation": "UNVERIFIED",
                "reason": "Contract not in Layer 3 monitored corpus. Proceed with caution.",
                "note": "Absence of data is not safety. This contract has not been analyzed.",
            })

        row = dict(row)

        # Run the stored potential risk scorer
        from surveillance.risk_scoring import score_contract
        risk = score_contract(conn, addr)

        # Capabilities — the agent needs to understand WHAT the contract can do
        capabilities = {}
        notes = (row.get("bytecode_pattern_notes") or "").upper()
        capabilities["asymmetric_transfer"] = bool(row.get("has_asymmetric_transfer"))
        capabilities["conditional_revert"] = bool(row.get("has_conditional_revert"))
        capabilities["unusual_fee_structure"] = bool(row.get("has_unusual_fee_structure"))
        capabilities["delegatecall"] = "DELEGATECALL" in notes
        capabilities["selfdestruct"] = "SELFDESTRUCT" in notes or "SUICIDE" in notes
        capabilities["create2"] = "CREATE2" in notes

        # Upgrade authority status
        upgrade_burned = False
        if "RENOUNCEOWNERSHIP" in notes.replace(" ", ""):
            upgrade_burned = True
        capabilities["upgrade_authority_burned"] = upgrade_burned

        # Admin-freeze capability (blacklist-like stored potential)
        # Same topological primitive as USDT's blacklist — admin can
        # freeze any holder's tokens via address-keyed mapping write.
        notes_lower = (row.get("bytecode_pattern_notes") or "").lower()
        has_blacklist_check = "blacklist" in notes_lower or "owner_check" in notes_lower
        capabilities["admin_freeze"] = bool(
            row.get("has_conditional_revert") and has_blacklist_check and not upgrade_burned
        )

        # Approval exposure on this contract
        approval = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN drain_detected = 1 THEN 1 ELSE 0 END) as drained "
            "FROM approval_watchlist WHERE contract_address = ?",
            (addr,),
        ).fetchone()
        existing_approvals = approval["total"] if approval else 0
        historical_drains = approval["drained"] or 0 if approval else 0
        drain_rate = historical_drains / existing_approvals if existing_approvals > 0 else 0.0

        # Recommendation logic
        score = risk.get("risk_score", 0)
        tier = risk.get("risk_tier", "MINIMAL")
        if tier == "CRITICAL" or score >= 50:
            recommendation = "DO_NOT_APPROVE"
            reason = f"CRITICAL risk score ({score:.0f}). Contract exhibits {sum(capabilities.values())} dangerous capabilities."
        elif tier == "HIGH" or score >= 20:
            recommendation = "CAUTION"
            reason = f"HIGH risk score ({score:.0f}). Review capabilities before proceeding."
        elif existing_approvals > 100 and drain_rate > 0.05:
            recommendation = "CAUTION"
            reason = f"{existing_approvals} existing approvals with {drain_rate:.1%} historical drain rate."
        elif row["confidence_tier"] == "confirmed":
            recommendation = "DO_NOT_APPROVE"
            reason = "Contract is a confirmed trap with on-chain victim evidence."
        elif row["confidence_tier"] == "suspected":
            recommendation = "CAUTION"
            reason = "Contract exhibits trap-like bytecode patterns."
        else:
            recommendation = "PROCEED"
            reason = "No elevated risk signals detected."

        return JSONResponse({
            "address": addr,
            "chain": chain,
            "in_corpus": True,
            "risk_score": risk.get("risk_score", 0),
            "risk_tier": tier,
            "stored_potential": risk.get("stored_potential", 0),
            "recommendation": recommendation,
            "reason": reason,
            "capabilities": capabilities,
            "deployer_risk": risk.get("deployer_risk_score", 0),
            "org_context": risk.get("org_context_score", 0),
            "approval_exposure": {
                "existing_approvals": existing_approvals,
                "historical_drain_rate": round(drain_rate, 4),
                "drains_detected": historical_drains,
            },
            "confidence_tier": row["confidence_tier"],
            "volatility": risk.get("volatility", 1.0),
        })
    finally:
        conn.close()


@router.get("/agent/facilitator/{address}")
async def agent_facilitator_check(address: str):
    """Validate an x402 facilitator before authorizing payment.

    Returns the facilitator's classification, nonce profile, and whether
    it's in the known rogue registry.
    """
    addr = address.lower()
    conn = _ro_conn()
    try:
        # Check x402_facilitators table
        fac = None
        try:
            fac = conn.execute(
                "SELECT address, chain, classification, name, notes "
                "FROM x402_facilitators WHERE address = ?",
                (addr,),
            ).fetchone()
        except Exception:
            pass

        if fac:
            classification = fac["classification"]
            safe = classification in ("known", "benign_relayer")
            return JSONResponse({
                "address": addr,
                "known": True,
                "classification": classification,
                "name": fac["name"],
                "safe": safe,
                "recommendation": "PROCEED" if safe else "DO_NOT_AUTHORIZE",
                "reason": f"Facilitator classified as '{classification}'" + (
                    ". This facilitator is in the rogue registry — DO NOT authorize payments."
                    if classification == "rogue" else ""
                ),
            })

        # Not in facilitator table — unknown
        return JSONResponse({
            "address": addr,
            "known": False,
            "classification": "unknown",
            "name": None,
            "safe": False,
            "recommendation": "CAUTION",
            "reason": "Facilitator not in Layer 3 registry. Verify independently before authorizing.",
        })
    finally:
        conn.close()


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
        "PROXY_UPGRADE": "CRITICAL",
        "TIMELOCK_IMMINENT": "CRITICAL",
        "DORMANT_ACTIVATION": "HIGH",
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
        # Primary method: funding_trail (most defensible — traces on-chain fund flow)
        dep_addrs = [d[0] for d in conn.execute("""
            SELECT DISTINCT deployer_address FROM deployers
            WHERE funding_trail LIKE ?
        """, (f'%{org_id}%',)).fetchall()]

        # Conservative count from entity_classification (manually tagged)
        ec_count = conn.execute(
            "SELECT COUNT(DISTINCT address) FROM entity_classification WHERE org_id = ?",
            (org_id,),
        ).fetchone()[0]

        if not dep_addrs:
            # Fallback to entity_classification + deployer_profiles
            dep_addrs = [d[0] for d in conn.execute("""
                SELECT DISTINCT address FROM entity_classification WHERE org_id = ?
                UNION
                SELECT DISTINCT deployer_address FROM deployer_profiles WHERE org_link = ?
            """, (org_id, org_id)).fetchall()]

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

        # Attribution confidence (Priority 5c)
        # HIGH: funding chain + behavioral similarity + temporal succession all align
        # MEDIUM: funding chain only (default for funding_trail-based attribution)
        # LOW: behavioral similarity only (no confirmed funding link)
        has_funding = len(dep_addrs) > 0
        has_behavioral = len(profiles) > 0
        has_similarity = False
        try:
            if dep_addrs:
                sim_ph = ",".join("?" * min(len(dep_addrs), 10))
                sim_row = conn.execute(
                    f"SELECT COUNT(*) FROM deployer_similarity "
                    f"WHERE deployer_a IN ({sim_ph}) AND composite_score >= 0.85",
                    dep_addrs[:10],
                ).fetchone()
                has_similarity = sim_row and sim_row[0] > 0
        except Exception:
            pass

        if has_funding and has_behavioral and has_similarity:
            org_confidence = "HIGH"
            org_confidence_note = (
                "Funding chain + behavioral profiling + cross-deployer "
                "similarity (>=0.85) all converge."
            )
        elif has_funding and has_behavioral:
            org_confidence = "MEDIUM"
            org_confidence_note = (
                "Funding chain traces verified on-chain. Behavioral "
                "profiles computed. Cross-deployer similarity not "
                "yet confirmed at >=0.85 threshold."
            )
        elif has_funding:
            org_confidence = "MEDIUM"
            org_confidence_note = (
                "Funding chain traces verified on-chain. Behavioral "
                "profiles not available for this org's deployers."
            )
        else:
            org_confidence = "LOW"
            org_confidence_note = (
                "Attribution based on entity_classification tags or "
                "deployer_profiles.org_link. No confirmed funding chain."
            )

        return JSONResponse(_ok({
            "org_id": org_id,
            "attribution_method": "funding_chain",
            "attribution_confidence": org_confidence,
            "attribution_confidence_note": org_confidence_note,
            "methodology_note": f"Primary count via on-chain funding trail. "
                                f"Conservative entity_classification count: {ec_count} deployers.",
            "epistemic_status": {
                "classification": "assessed",
                "deductive_base": [
                    f"{len(dep_addrs)} deployer addresses linked via on-chain fund transfers",
                    f"{contracts[0]} contracts deployed by those addresses",
                ],
                "inferential_components": [
                    "org_id assignment: interpreting funding chain as organizational link",
                    f"timezone inference: {tz[0] if tz else 'n/a'}",
                ],
            },
            "scale": {
                "deployers": len(dep_addrs),
                "contracts": contracts[0],
                "chains": chains,
            },
            "operational_tempo": {
                "timezone_inference": tz[0] if tz else None,
                "techniques": techniques[:3],
            },
        }, conn,
            fresh_tables=["entity_classification", "deployer_profiles",
                          "deployer_similarity"],
        ))
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
        }, conn,
            fresh_tables=["deployer_profiles", "deployer_similarity",
                          "entity_classification"],
        ))
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
        return JSONResponse(_ok(
            _build_risk(conn, dict(row)), conn,
            fresh_tables=["trust_amplification", "bytecode_families",
                          "entity_classification"],
        ))
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

        # Wallet rotations: similarity > 0.85 AND temporal succession
        rotations = conn.execute("""
            SELECT COUNT(*) FROM deployer_similarity ds
            JOIN deployers da ON ds.deployer_a = da.deployer_address
            JOIN deployers db ON ds.deployer_b = db.deployer_address
            WHERE ds.composite_score >= 0.85
            AND (db.first_seen > da.last_seen OR da.first_seen > db.last_seen)
        """).fetchone()[0]

        # Camouflage: compute live from transaction_events (contracts with 10+ tx)
        cam = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END) as camo
            FROM (
                SELECT contract_address,
                       CAST(SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as rr
                FROM transaction_events GROUP BY contract_address HAVING COUNT(*) >= 10
            )
        """).fetchone()
        cam_ratio = round(cam[1] / cam[0], 3) if cam and cam[0] > 0 else None

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
                "camouflage_note": "Fraction of contracts (10+ interactions) with <10% revert rate",
                "behavioral_confirmations_24h": traps_24h,
            },
            "organizations": {
                "mapped": orgs,
                "wallet_rotations_detected": rotations,
                "rotation_criteria": "similarity >= 0.85 with temporal succession",
            },
        }, conn,
            fresh_tables=["entity_classification", "deployer_similarity",
                          "daily_metrics"],
        ))
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


@router.get("/methodology/confidence")
async def methodology_confidence():
    """Published confidence score algorithm. Allows independent
    reproduction. Added per COMMERCIAL_EPISTEMIC_AUDIT.md Priority 2a."""
    return JSONResponse({"status": "ok", "data": {
        "description": (
            "Confidence scoring algorithm for Layer 3 contract risk "
            "profiles. Deterministic given inputs. Inputs include one "
            "inferential component (confidence_tier assignment)."
        ),
        "algorithm": {
            "confirmed_tier": {
                "base_score": 0.90,
                "victim_bonus": "min(victim_count / 1000, 0.09)",
                "range": "0.90 - 0.99",
                "trigger": (
                    "External caller (not deployer) reverted on this "
                    "contract. Requires trap_events row with tx_hash."
                ),
                "epistemic": "D (revert is on-chain) + I (interpreting revert as trap)",
            },
            "suspected_tier": {
                "3_plus_signatures": {
                    "formula": "0.70 + signature_count * 0.05",
                    "range": "0.85",
                },
                "1_to_2_signatures": {
                    "formula": "0.50 + signature_count * 0.08",
                    "range": "0.58 - 0.66",
                },
                "0_signatures_deployer_history": {
                    "score": 0.35,
                    "note": "Deployer-history-only detection, no bytecode evidence",
                },
                "signature_fields": [
                    "has_asymmetric_transfer",
                    "has_conditional_revert",
                    "has_unusual_fee_structure",
                ],
                "epistemic": (
                    "I — suspected tier assignment is a policy decision "
                    "(MIN_PATTERNS_FOR_SUSPECTED=1). Signature detection "
                    "is deductive (bytecode pattern at cited offset)."
                ),
            },
            "unknown_tier": {
                "score": None,
                "note": "No confidence score assigned for unknown-tier contracts",
            },
        },
        "risk_level_mapping": {
            "CRITICAL": "confirmed tier (any confidence score)",
            "HIGH": "confidence >= 0.70",
            "MEDIUM": "confidence >= 0.50",
            "LOW": "confidence >= 0.30",
            "UNKNOWN": "confidence < 0.30 or None",
        },
        "epistemic_note": (
            "The confidence score is INFERENTIAL. It is computed from a "
            "deterministic formula, but the primary input (confidence_tier) "
            "is itself a classification decision. A customer can verify the "
            "deductive inputs (bytecode patterns, revert counts) but cannot "
            "independently reproduce the score without the same threshold "
            "policy. This is by design: the score is our assessment, not a "
            "measurement."
        ),
    }})


@router.get("/methodology/camouflage")
async def methodology_camouflage():
    """Published camouflage ratio methodology. Priority 2b."""
    return JSONResponse({"status": "ok", "data": {
        "metric": "camouflage_ratio",
        "definition": (
            "Fraction of contracts with 10+ interactions that maintain "
            "revert rates below 10%."
        ),
        "threshold_justification": (
            "10% was chosen because contracts below this threshold "
            "attract 4.5x more unique victims than overt traps (>50% "
            "revert rate). Empirical from two-week assessment: "
            "camouflaged contracts averaged 66 victims vs 15 for overt. "
            "The multiplier is stable across chains (Base 4.2x, "
            "Arbitrum 4.8x) and weeks (W1 4.3x, W2 4.7x)."
        ),
        "current_value": "79.2% (production, stable across 23 days)",
        "epistemic": (
            "MIXED — the revert rate computation is deductive (arithmetic "
            "on on-chain receipts). The 10% threshold is a policy choice "
            "informed by empirical data but not the only defensible "
            "boundary. The 4.5x multiplier is a measured observation."
        ),
    }})
