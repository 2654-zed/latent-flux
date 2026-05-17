"""
Layer 3 — Web UI
FastAPI + HTMX + Tailwind + Cytoscape.js

Run: DB_PATH=surveillance/data/surveillance.db uvicorn web.app:app --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.data import (
    get_conn, get_overview_stats, get_key_metrics, get_chain_split,
    get_daily_trend, get_recent_alerts, get_strategy_lifecycle,
    get_bot_sophistication, get_org_graph, get_org_001_stats,
    get_contract, get_deployer, get_address_detail, get_recent_watchlist_hits,
    get_threats, get_threat_counts, get_watched_entities, search_address,
)
from web.api_v1 import router as v1_router

app = FastAPI(title="Layer 3 Intelligence", docs_url=None, redoc_url=None)

# CORS for API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Mount public API v1
app.include_router(v1_router, prefix="/api/v1")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Serve static files (building PNGs etc)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

DB = os.environ.get("DB_PATH", str(BASE_DIR.parent / "surveillance" / "data" / "surveillance.db"))

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def conn():
    return get_conn(DB)


# ---------------------------------------------------------------
# Emergency DB upload — TEMPORARY, remove after recovery
# ---------------------------------------------------------------

@app.put("/admin/upload-db")
async def upload_db(request: Request):
    """Accept a gzipped SQLite DB upload to replace the corrupted production DB."""
    import gzip
    auth = request.headers.get("Authorization", "")
    if not ADMIN_TOKEN or auth != f"Bearer {ADMIN_TOKEN}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)

    db_path = Path(DB)
    tmp_path = db_path.with_suffix(".db.new")
    try:
        # Decompress gzip
        data = gzip.decompress(body)
        tmp_path.write_bytes(data)

        # Quick integrity check
        import sqlite3
        c = sqlite3.connect(str(tmp_path))
        result = c.execute("PRAGMA integrity_check").fetchone()
        tables = c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()
        c.close()

        if result[0] != "ok":
            tmp_path.unlink()
            return JSONResponse({"error": f"integrity check failed: {result[0]}"}, status_code=400)

        # Replace the corrupted DB
        backup_path = db_path.with_suffix(".db.corrupt")
        if db_path.exists():
            db_path.rename(backup_path)
        tmp_path.rename(db_path)

        return JSONResponse({
            "status": "ok",
            "size_bytes": len(data),
            "tables": tables[0],
            "integrity": result[0],
            "note": "DB replaced. Redeploy to restart monitors with new DB.",
        })
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------
# Overview
# ---------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    c = conn()
    stats = get_overview_stats(c)
    metrics = get_key_metrics(c)
    lifecycle = get_strategy_lifecycle(c)
    bot_soph = get_bot_sophistication(c)
    alerts = get_recent_alerts(c, 30)
    c.close()
    return templates.TemplateResponse("overview.html", {
        "request": request, "active_page": "overview",
        "stats": stats, "metrics": metrics, "lifecycle": lifecycle,
        "bot_soph": bot_soph, "alerts": alerts,
    })


# ---------------------------------------------------------------
# Organization Map
# ---------------------------------------------------------------

@app.get("/org-map", response_class=HTMLResponse)
async def org_map(request: Request, org: str = "org_001"):
    c = conn()
    stats = get_overview_stats(c)

    # org_001 gets the curated SVG template
    if org == "org_001":
        org_stats = get_org_001_stats(c)
        c.close()
        return templates.TemplateResponse("org_001_map.html", {
            "request": request, "active_page": "org_map",
            "stats": stats, "org": org_stats,
        })

    c.close()
    return templates.TemplateResponse("org_map.html", {
        "request": request, "active_page": "org_map",
        "stats": stats, "org_id": org,
    })


@app.get("/api/org/{org_id}")
async def api_org(org_id: str):
    c = conn()
    graph = get_org_graph(c, org_id)
    c.close()
    return JSONResponse(graph)


# ---------------------------------------------------------------
# Contract Inspector
# ---------------------------------------------------------------

@app.get("/contract", response_class=HTMLResponse)
@app.get("/contract/{address}", response_class=HTMLResponse)
async def contract_view(request: Request, address: str = None, q: str = None):
    c = conn()
    query = address or q
    contract = get_contract(c, query) if query else None
    stats = get_overview_stats(c)
    c.close()
    return templates.TemplateResponse("contract.html", {
        "request": request, "active_page": "contract",
        "stats": stats, "contract": contract, "query": query,
    })


# ---------------------------------------------------------------
# Deployer Profile
# ---------------------------------------------------------------

@app.get("/deployer", response_class=HTMLResponse)
@app.get("/deployer/{address}", response_class=HTMLResponse)
async def deployer_view(request: Request, address: str = None, q: str = None):
    c = conn()
    query = address or q
    deployer = get_deployer(c, query) if query else None
    stats = get_overview_stats(c)
    c.close()
    return templates.TemplateResponse("deployer.html", {
        "request": request, "active_page": "deployer",
        "stats": stats, "deployer": deployer, "query": query,
    })


# ---------------------------------------------------------------
# Guard — Trust & Safety operator console (React, CDN-loaded, no build step).
# Phase A: static skeleton; subsequent phases wire real data via /api/guard/*.
# ---------------------------------------------------------------

@app.get("/guard", response_class=HTMLResponse)
def guard_view(request: Request):
    return templates.TemplateResponse("guard.html", {
        "request": request, "active_page": "guard",
    })


@app.get("/api/guard/kpi")
def guard_kpi():
    """KPI strip — six rolled-up metrics from real DB tables."""
    c = conn()
    cur = c.cursor()

    contracts = cur.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    contracts_confirmed = cur.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier='confirmed'"
    ).fetchone()[0]
    contracts_suspected = cur.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier='suspected'"
    ).fetchone()[0]

    deployers = cur.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
    deployers_by_chain = {
        r[0]: r[1] for r in cur.execute(
            "SELECT chain, COUNT(*) FROM deployers GROUP BY chain"
        )
    }

    wl = {
        r[0]: r[1] for r in cur.execute(
            "SELECT priority, COUNT(*) FROM watchlist WHERE active=1 GROUP BY priority"
        )
    }
    wl_total = sum(wl.values())

    drains_24h = cur.execute(
        "SELECT COUNT(*) FROM approval_watchlist "
        "WHERE drain_detected=1 AND drain_timestamp >= datetime('now','-1 day')"
    ).fetchone()[0]
    drains_latest = cur.execute(
        "SELECT MAX(drain_timestamp) FROM approval_watchlist WHERE drain_detected=1"
    ).fetchone()[0]

    alerts_24h = cur.execute(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= datetime('now','-1 day')"
    ).fetchone()[0]
    alert_top = cur.execute(
        "SELECT alert_type, COUNT(*) FROM alerts "
        "WHERE timestamp >= datetime('now','-1 day') "
        "GROUP BY alert_type ORDER BY 2 DESC LIMIT 1"
    ).fetchone()

    prec_row = cur.execute(
        "SELECT AVG(precision), COUNT(DISTINCT detector_name) FROM detector_precision"
    ).fetchone()
    c.close()

    return JSONResponse({
        "contracts": contracts,
        "contracts_confirmed": contracts_confirmed,
        "contracts_suspected": contracts_suspected,
        "deployers": deployers,
        "deployers_by_chain": deployers_by_chain,
        "watchlist_total": wl_total,
        "watchlist_by_priority": wl,
        "drains_24h": drains_24h,
        "drains_latest_ts": drains_latest,
        "alerts_24h": alerts_24h,
        "alerts_top_type": alert_top[0] if alert_top else None,
        "alerts_top_count": alert_top[1] if alert_top else 0,
        "detector_precision_pct": round((prec_row[0] or 0) * 100, 1),
        "detector_count": prec_row[1] or 0,
    })


# ---------------------------------------------------------------
# Explore — read-only SQL playground. Power-user escape hatch when
# the curated Guard modules don't have what you need.
# Constraints: PRAGMA query_only, 5s timeout, 5000-row cap, SELECT only.
# ---------------------------------------------------------------

EXPLORE_CATALOG = [
    {
        "id":   "drainer_outflows_24h",
        "name": "Active drainer outflows · 24h",
        "desc": "Drain events grouped by drainer, latest 24h.",
        "sql":  (
            "SELECT drain_caller AS drainer, "
            "       COUNT(*) AS victim_count, "
            "       COUNT(DISTINCT contract_address) AS contracts, "
            "       MIN(drain_timestamp) AS first_drain, "
            "       MAX(drain_timestamp) AS last_drain "
            "FROM approval_watchlist "
            "WHERE drain_detected=1 AND drain_timestamp >= datetime('now','-1 day') "
            "GROUP BY drain_caller ORDER BY victim_count DESC LIMIT 50"
        ),
    },
    {
        "id":   "top_contracts_by_victims",
        "name": "Top contracts by victim count",
        "desc": "Most-approved contracts, joined to tier + drain count.",
        "sql":  (
            "SELECT aw.contract_address, c.confidence_tier, c.chain, "
            "       COUNT(DISTINCT aw.victim_address) AS victims, "
            "       SUM(aw.drain_detected) AS drains "
            "FROM approval_watchlist aw "
            "JOIN contracts c ON c.contract_address = aw.contract_address "
            "GROUP BY aw.contract_address ORDER BY victims DESC LIMIT 30"
        ),
    },
    {
        "id":   "watchlist_by_priority",
        "name": "Watchlist · counts by priority",
        "desc": "Active watchlist entries grouped by priority.",
        "sql":  (
            "SELECT priority, address_type, COUNT(*) AS n "
            "FROM watchlist WHERE active=1 "
            "GROUP BY priority, address_type ORDER BY priority, n DESC"
        ),
    },
    {
        "id":   "vanity_62ac_ee8a",
        "name": "62ac…EE8A vanity-mimic cluster",
        "desc": "Sibling wallets with the same 8-char vanity shape as the F7cFFC27 drain victim.",
        "sql":  (
            "SELECT address, address_type, entity_name, priority, active "
            "FROM watchlist "
            "WHERE LOWER(address) LIKE '0x62ac%' AND LOWER(address) LIKE '%ee8a'"
        ),
    },
    {
        "id":   "pattern_a_clones",
        "name": "Self-deploying single-contract mass-drain (Pattern A clones)",
        "desc": "Drainers with fleet=1 + ≥10 victims in a single contract.",
        "sql":  (
            "SELECT aw.drain_caller, "
            "       COUNT(DISTINCT aw.contract_address) AS contracts, "
            "       COUNT(*) AS victims, "
            "       MIN(aw.drain_timestamp) AS first, "
            "       MAX(aw.drain_timestamp) AS last "
            "FROM approval_watchlist aw "
            "WHERE aw.drain_detected=1 "
            "GROUP BY aw.drain_caller "
            "HAVING contracts = 1 AND victims >= 10 "
            "ORDER BY first DESC LIMIT 30"
        ),
    },
    {
        "id":   "drain_wave_hourly",
        "name": "Drain wave · hourly · 7d",
        "desc": "Hourly drain count over the last 7 days.",
        "sql":  (
            "SELECT strftime('%Y-%m-%dT%H', drain_timestamp) AS hour, "
            "       COUNT(*) AS drains, "
            "       COUNT(DISTINCT drain_caller) AS drainers "
            "FROM approval_watchlist "
            "WHERE drain_detected=1 AND drain_timestamp >= datetime('now','-7 day') "
            "GROUP BY hour ORDER BY hour"
        ),
    },
    {
        "id":   "oli_suppression_impact",
        "name": "OLI suppression impact",
        "desc": "Contracts whose deployer is OLI-tagged HIGH/LOW — drain-detector would have promoted these.",
        "sql":  (
            "SELECT o.primary_entity, o.severity, "
            "       COUNT(DISTINCT aw.contract_address) AS contracts, "
            "       COUNT(*) AS approval_rows "
            "FROM oli_labels o "
            "JOIN approval_watchlist aw ON aw.deployer_address = o.address "
            "WHERE o.severity IN ('HIGH','LOW') "
            "GROUP BY o.primary_entity, o.severity ORDER BY approval_rows DESC"
        ),
    },
    {
        "id":   "detector_precision_today",
        "name": "Detector precision · latest",
        "desc": "Per-detector precision from the latest detector_precision audit.",
        "sql":  (
            "SELECT detector_name, date, precision, total_fired, confirmed_traps, false_positives "
            "FROM detector_precision ORDER BY precision DESC, total_fired DESC"
        ),
    },
    {
        "id":   "forecast_actual",
        "name": "Forecast accuracy · predicted vs actual",
        "desc": "Per-metric predicted vs realized values.",
        "sql":  (
            "SELECT metric_name, target_date, predicted_value, actual_value, hit "
            "FROM predictions WHERE actual_value IS NOT NULL "
            "ORDER BY target_date DESC, metric_name LIMIT 60"
        ),
    },
    {
        "id":   "deployer_velocity",
        "name": "Deployers by total contract count",
        "desc": "Top deployers by total contracts deployed.",
        "sql":  (
            "SELECT deployer_address, chain, total_contracts_deployed AS contracts, "
            "       first_seen, last_seen "
            "FROM deployers ORDER BY total_contracts_deployed DESC LIMIT 30"
        ),
    },
    {
        "id":   "vuln_index_reentrancy",
        "name": "Vuln Index · all reentrancy categories",
        "desc": "Reentrancy entries across all protocol types in the external vulnerability index.",
        "sql":  (
            "SELECT protocol_type, slug, category, severity, findings "
            "FROM vuln_index WHERE slug LIKE '%reentrancy%' "
            "ORDER BY protocol_type, slug"
        ),
    },
    {
        "id":   "vuln_index_high_severity",
        "name": "Vuln Index · HIGH severity by protocol type",
        "desc": "All HIGH-severity vulnerability categories in the external index.",
        "sql":  (
            "SELECT protocol_type, slug, category, findings "
            "FROM vuln_index WHERE severity = 'HIGH' "
            "ORDER BY findings DESC LIMIT 50"
        ),
    },
    {
        "id":   "vuln_index_fts_search",
        "name": "Vuln Index · full-text search example",
        "desc": "FTS5 search across vuln_index — edit the MATCH expression to your query.",
        "sql":  (
            "SELECT protocol_type, slug, category, severity, findings "
            "FROM vuln_index WHERE vuln_index MATCH 'oracle AND manipulation' "
            "ORDER BY rank LIMIT 25"
        ),
    },
]


@app.get("/explore", response_class=HTMLResponse)
async def explore_view(request: Request):
    return templates.TemplateResponse("explore.html", {
        "request": request, "active_page": "explore",
    })


@app.get("/api/guard/explore/catalog")
def guard_explore_catalog():
    return JSONResponse({"queries": EXPLORE_CATALOG})


@app.get("/api/guard/vuln_coverage")
def guard_vuln_coverage():
    """Phase H · A — Layer 3 detector coverage against
    kadenzipfel/protocol-vulnerabilities-index taxonomy.

    Reads from the precomputed `vuln_coverage_report` table populated by
    `scripts/vuln_index_coverage.py`. Returns the report verbatim plus the
    ingest metadata + any demonstrated-incident anchors from `vuln_incidents`
    (see `scripts/seed_vuln_incidents.py`).
    """
    import json
    c = conn()
    cur = c.cursor()
    try:
        row = cur.execute(
            "SELECT computed_at, total_categories, covered, gap, coverage_pct, "
            "       l3_detector_count, report_json "
            "FROM vuln_coverage_report ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        meta = cur.execute(
            "SELECT ingested_at, source_commit, row_count, protocol_types "
            "FROM vuln_index_meta ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    except Exception as e:
        c.close()
        return JSONResponse({"error": f"vuln_coverage_report missing — run scripts/vuln_index_coverage.py: {e}"}, status_code=503)
    if not row:
        c.close()
        return JSONResponse({"error": "no report rows; run scripts/vuln_index_coverage.py"}, status_code=503)

    # Demonstrated-incident anchors (best-effort; table may not exist yet).
    incidents = []
    try:
        for ir in cur.execute(
            "SELECT case_file, name, incident_date, protocol_type, slug, "
            "       magnitude_usd, chain, sub_mechanism, attacker_addr "
            "FROM vuln_incidents ORDER BY incident_date DESC"
        ):
            incidents.append({
                "case_file":     ir[0],
                "name":          ir[1],
                "incident_date": ir[2],
                "protocol_type": ir[3],
                "slug":          ir[4],
                "magnitude_usd": ir[5],
                "chain":         ir[6],
                "sub_mechanism": ir[7],
                "attacker_addr": ir[8],
            })
    except Exception:
        pass
    c.close()

    report = json.loads(row[6])
    return JSONResponse({
        "computed_at":         row[0],
        "total_categories":    row[1],
        "covered":             row[2],
        "gap":                 row[3],
        "coverage_pct":        row[4],
        "l3_detector_count":   row[5],
        "by_protocol":         report.get("by_protocol", []),
        "by_detector":         report.get("by_detector", []),
        "uncovered_sample":    report.get("uncovered_sample", []),
        "uncovered_total":     report.get("uncovered_total", 0),
        "demonstrated_incidents":     incidents,
        "demonstrated_incident_count": len({i["case_file"] for i in incidents}),
        "source": {
            "ingested_at":     meta[0] if meta else None,
            "commit":          meta[1] if meta else None,
            "row_count":       meta[2] if meta else None,
            "protocol_types":  meta[3] if meta else None,
            "repo":            "kadenzipfel/protocol-vulnerabilities-index",
        },
    })


def _is_select_only(sql: str) -> bool:
    """Reject anything that isn't a single SELECT/WITH/PRAGMA-safe read."""
    # Strip leading whitespace + SQL comments.
    s = sql.strip()
    while True:
        if s.startswith("--"):
            nl = s.find("\n")
            s = "" if nl == -1 else s[nl + 1:].lstrip()
        elif s.startswith("/*"):
            end = s.find("*/")
            s = "" if end == -1 else s[end + 2:].lstrip()
        else:
            break
    if not s:
        return False
    first = s.split(None, 1)[0].upper()
    if first not in ("SELECT", "WITH"):
        return False
    # Reject explicit write statements anywhere (defense in depth).
    lo = s.lower()
    for verb in ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "attach ", "vacuum"):
        if verb in lo:
            return False
    return True


@app.post("/api/guard/explore/query")
def guard_explore_query(payload: dict):
    """Run a read-only SQL query against surveillance.db.

    Constraints (all enforced server-side):
    - SQL must start with SELECT or WITH after stripping comments.
    - Connection opens in `query_only` mode; no writes possible.
    - 5-second wall-clock timeout via progress handler.
    - Result capped at 5000 rows; `truncated` flag set if hit.
    """
    import sqlite3
    import time

    sql = (payload or {}).get("sql", "")
    if not isinstance(sql, str) or not sql.strip():
        return JSONResponse({"error": "sql is required"}, status_code=400)
    if len(sql) > 5000:
        return JSONResponse({"error": "sql exceeds 5KB"}, status_code=400)
    if not _is_select_only(sql):
        return JSONResponse({"error": "only SELECT / WITH queries are allowed"}, status_code=400)

    ROW_CAP = 5000
    TIMEOUT_S = 5.0
    started = time.monotonic()

    conn_ro = sqlite3.connect(DB, timeout=2)
    conn_ro.row_factory = sqlite3.Row
    try:
        conn_ro.execute("PRAGMA query_only = ON")

        # Progress handler — abort if wall-clock budget exceeded.
        def _budget():
            return 1 if (time.monotonic() - started) > TIMEOUT_S else 0
        conn_ro.set_progress_handler(_budget, 10000)

        try:
            cur = conn_ro.execute(sql)
        except sqlite3.OperationalError as e:
            return JSONResponse({"error": f"sqlite: {e}"}, status_code=400)
        except sqlite3.DatabaseError as e:
            return JSONResponse({"error": f"db: {e}"}, status_code=400)

        cols = [d[0] for d in (cur.description or [])]
        rows = []
        truncated = False
        for r in cur:
            if len(rows) >= ROW_CAP:
                truncated = True
                break
            rows.append({c: r[c] for c in cols})
        duration_ms = int((time.monotonic() - started) * 1000)
        return JSONResponse({
            "columns":     cols,
            "rows":        rows,
            "row_count":   len(rows),
            "truncated":   truncated,
            "duration_ms": duration_ms,
        })
    finally:
        try:
            conn_ro.set_progress_handler(None, 0)
        except Exception:
            pass
        conn_ro.close()


@app.get("/api/guard/backtest")
def guard_backtest():
    """Backtest panel — L3 vs GoPlus + L3 forecast accuracy.

    Replaces the mockup's Sybil-Net v2 vs v1 with real ai lang data:
    - L3-only catch headline from goplus_results.match (no '100% gap' phrasing
      per CORRECTIONS.md Quick Retirement Index).
    - Per-detector precision distribution from detector_precision.
    - Forecast hit-rate + per-metric predicted-vs-actual series from predictions.
    - Bessel-corrected (n-1) σ across detector precision values.
    - Epistemic tests computed from real boolean conditions on the data.
    """
    import math
    c = conn()
    cur = c.cursor()

    # --- L3 vs GoPlus headline (50 sample contracts, single-day audit 2026-03-27) ---
    gp = cur.execute(
        "SELECT match, COUNT(*) FROM goplus_results GROUP BY match"
    ).fetchall()
    match_counts = {m: n for m, n in gp}
    gp_total = sum(match_counts.values())
    gp_l3_only = match_counts.get("L3_ONLY", 0)
    _audit = cur.execute(
        "SELECT MIN(substr(checked_at, 1, 10)), MAX(substr(checked_at, 1, 10)) "
        "FROM goplus_results"
    ).fetchone()
    gp_audit_date = [_audit[0], _audit[1]] if _audit else [None, None]

    # --- Per-detector precision ---
    det_rows = cur.execute(
        "SELECT detector_name, date, precision, total_fired, confirmed_traps, false_positives "
        "FROM detector_precision ORDER BY precision DESC"
    ).fetchall()
    detectors = [
        {
            "name":      r[0],
            "date":      r[1],
            "precision": r[2],
            "fired":     r[3],
            "confirmed": r[4],
            "fp":        r[5],
        }
        for r in det_rows
    ]
    precisions = [d["precision"] for d in detectors if d["precision"] is not None]
    n = len(precisions)
    mean_prec = sum(precisions) / n if n else 0
    # Bessel-corrected n-1 sample std-dev
    var = (sum((p - mean_prec) ** 2 for p in precisions) / (n - 1)) if n >= 2 else 0
    std_prec = math.sqrt(var)

    # --- Forecast performance (predictions table) ---
    metric_rows = cur.execute(
        "SELECT metric_name, COUNT(*) AS scored, SUM(hit) AS hits "
        "FROM predictions WHERE actual_value IS NOT NULL "
        "GROUP BY metric_name"
    ).fetchall()
    metrics = [
        {"metric": r[0], "scored": r[1], "hits": r[2] or 0,
         "hit_rate": round((r[2] or 0) / r[1], 3) if r[1] else 0}
        for r in metric_rows
    ]
    total_scored = sum(m["scored"] for m in metrics)
    total_hits   = sum(m["hits"]   for m in metrics)
    overall_hit_rate = round(total_hits / total_scored, 3) if total_scored else 0

    # Time-series for the chart — predicted vs actual on `total_tx_events`
    # (most rows / most variance, makes the divergence visible).
    series_rows = cur.execute(
        "SELECT target_date, predicted_value, actual_value "
        "FROM predictions "
        "WHERE metric_name = 'total_tx_events' AND actual_value IS NOT NULL "
        "ORDER BY target_date"
    ).fetchall()
    series = [
        {"t": r[0], "predicted": r[1], "actual": r[2]}
        for r in series_rows
    ]

    c.close()

    # --- Epistemic tests (real checks on the surfaced data) ---
    detector_set = {d["name"] for d in detectors}
    tests = [
        {"name": "Detector count ≥ 5",
         "pass": len(detector_set) >= 5,
         "note": f"{len(detector_set)} detectors"},
        {"name": "Mean detector precision ≥ 0.95",
         "pass": mean_prec >= 0.95,
         "note": f"{mean_prec:.4f}"},
        {"name": "Detector precision σ < 0.05 (Bessel n−1)",
         "pass": std_prec < 0.05,
         "note": f"σ = {std_prec:.4f}"},
        {"name": "GoPlus comparison N ≥ 30",
         "pass": gp_total >= 30,
         "note": f"{gp_total} contracts"},
        {"name": "L3-only catches > 0",
         "pass": gp_l3_only > 0,
         "note": f"{gp_l3_only}/{gp_total}"},
        {"name": "Forecast metrics ≥ 3",
         "pass": len(metrics) >= 3,
         "note": f"{len(metrics)} metrics"},
        {"name": "Forecast sample N ≥ 30",
         "pass": total_scored >= 30,
         "note": f"{total_scored} scored"},
        {"name": "Forecast overall hit-rate ≥ 0.40",
         "pass": overall_hit_rate >= 0.40,
         "note": f"{overall_hit_rate:.2%}"},
        {"name": "Per-detector false positives ≤ 10",
         "pass": all((d["fp"] or 0) <= 10 for d in detectors),
         "note": f"max {max((d['fp'] or 0) for d in detectors) if detectors else 0}"},
        {"name": "Detector precision recomputed in last 7d",
         "pass": False,
         "note": "stale; single-date 2026-03-30"},
    ]
    passed = sum(1 for t in tests if t["pass"])
    held_in_shadow = passed < len(tests)

    return JSONResponse({
        "goplus": {
            "total":       gp_total,
            "l3_only":     gp_l3_only,
            "match_counts": match_counts,
            "audit_window": gp_audit_date,
        },
        "detectors": {
            "items":      detectors,
            "count":      len(detectors),
            "mean":       round(mean_prec, 4),
            "std_bessel": round(std_prec, 5),
        },
        "forecast": {
            "metrics":          metrics,
            "total_scored":     total_scored,
            "total_hits":       total_hits,
            "overall_hit_rate": overall_hit_rate,
            "series":           series,
            "series_metric":    "total_tx_events",
        },
        "epistemic": {
            "tests":          tests,
            "passed":         passed,
            "total":          len(tests),
            "held_in_shadow": held_in_shadow,
        },
    })


@app.get("/api/guard/drift")
def guard_drift(rows: int = 7):
    """Drift Heatmap — top-N flagged contracts × 24 hourly buckets.

    For each contract, computes drift = (recent_hour_count / baseline_per_hour) * 30
    where baseline = 7d-rolling-avg tx/hour (capped at 1.0 floor). Clamped to 0-100.
    Result drives the cool→hot color scale on the frontend.
    """
    from datetime import datetime, timezone, timedelta
    import math
    c = conn()
    cur = c.cursor()

    # Top-N flagged contracts by recent 24h activity.
    top_rows = cur.execute(
        "SELECT te.contract_address, COUNT(*) AS n "
        "FROM transaction_events te "
        "JOIN contracts c ON c.contract_address = te.contract_address "
        "WHERE te.timestamp >= datetime('now','-24 hour') "
        "  AND c.confidence_tier IN ('confirmed','suspected') "
        "GROUP BY te.contract_address "
        "ORDER BY n DESC LIMIT ?",
        (rows,),
    ).fetchall()

    if not top_rows:
        c.close()
        return JSONResponse({"rows": [], "hours": [], "total": 0})

    contracts = [r[0] for r in top_rows]
    ph = ",".join("?" * len(contracts))

    # Per-contract hourly buckets over the last 24h.
    hour_counts = cur.execute(
        f"SELECT contract_address, strftime('%Y-%m-%dT%H', timestamp) AS hb, COUNT(*) "
        f"FROM transaction_events "
        f"WHERE contract_address IN ({ph}) "
        f"  AND timestamp >= datetime('now','-24 hour') "
        f"GROUP BY contract_address, hb",
        contracts,
    ).fetchall()

    # 7-day baseline (per-hour average), excluding the last 24h.
    base_rows = cur.execute(
        f"SELECT contract_address, COUNT(*) "
        f"FROM transaction_events "
        f"WHERE contract_address IN ({ph}) "
        f"  AND timestamp >= datetime('now','-8 day') "
        f"  AND timestamp <  datetime('now','-1 day') "
        f"GROUP BY contract_address",
        contracts,
    ).fetchall()

    # Look up contract tier + watchlist label if any.
    meta_rows = cur.execute(
        f"SELECT contract_address, confidence_tier, chain "
        f"FROM contracts WHERE contract_address IN ({ph})",
        contracts,
    ).fetchall()
    wl_rows = cur.execute(
        f"SELECT LOWER(address), entity_name, priority FROM watchlist "
        f"WHERE active=1 AND LOWER(address) IN ({ph})",
        [a.lower() for a in contracts],
    ).fetchall()
    c.close()

    baseline = {row[0]: max(1.0, row[1] / (7 * 24)) for row in base_rows}
    counts   = {}
    for addr, hb, n in hour_counts:
        counts.setdefault(addr, {})[hb] = n
    meta = {r[0]: {"tier": r[1], "chain": r[2]} for r in meta_rows}
    wl   = {r[0]: {"label": r[1], "priority": r[2]} for r in wl_rows}

    now_h = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [(now_h - timedelta(hours=23 - i)) for i in range(24)]
    hour_labels = [b.strftime("%Y-%m-%dT%H") for b in buckets]
    hour_display = [b.strftime("%H") for b in buckets]

    out_rows = []
    for addr in contracts:
        per_hour = counts.get(addr, {})
        base = baseline.get(addr, 1.0)
        cells = []
        for lab in hour_labels:
            cnt = per_hour.get(lab, 0)
            drift = min(100, int(round((cnt / base) * 30)))
            cells.append({"v": drift, "n": cnt})
        wl_info = wl.get(addr.lower(), {})
        meta_info = meta.get(addr, {})
        label = wl_info.get("label") or f"{addr[:8]}…{addr[-4:]}"
        out_rows.append({
            "address":  addr,
            "label":    label,
            "tier":     meta_info.get("tier"),
            "chain":    meta_info.get("chain"),
            "priority": wl_info.get("priority"),
            "baseline_per_hour": round(base, 2),
            "cells": cells,
        })

    return JSONResponse({
        "rows":        out_rows,
        "hour_labels": hour_display,
        "total":       len(out_rows),
    })


@app.get("/api/guard/matrix")
def guard_matrix(limit: int = 100):
    """Risk Matrix scatter — contracts × victims-at-stake (log Y).
    Risk score derived from confidence_tier + drain count. Severity bucket
    used for chart-side coloring; not a binary safe/unsafe flag.
    """
    import math
    c = conn()
    cur = c.cursor()
    rows = cur.execute(
        "SELECT aw.contract_address AS contract_address, "
        "       c.confidence_tier   AS confidence_tier, "
        "       c.chain             AS chain, "
        "       c.deployer_address  AS deployer_address, "
        "       COUNT(DISTINCT aw.victim_address) AS victims, "
        "       COALESCE(SUM(aw.drain_detected), 0) AS drains "
        "FROM approval_watchlist aw "
        "JOIN contracts c ON c.contract_address = aw.contract_address "
        "GROUP BY aw.contract_address "
        "HAVING victims >= 5 "
        "ORDER BY victims DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    c.close()

    points = []
    for r in rows:
        d = dict(r)
        tier    = d.get("confidence_tier")
        victims = d.get("victims") or 0
        drains  = d.get("drains") or 0
        # Risk score 0-10: tier base + log10(drains+1) adjustment.
        if tier == "confirmed":
            risk = 8.0 + min(2.0, math.log10(drains + 1) / 1.5)
        elif tier == "suspected":
            risk = 5.0 + (min(2.0, math.log10(drains + 1)) if drains > 0 else 0)
        elif tier == "unanalyzed":
            risk = 3.0 + (min(1.5, math.log10(drains + 1)) if drains > 0 else 0)
        else:
            risk = 1.5
        risk = round(min(risk, 10.0), 2)

        # Severity bucket for chart-side coloring (not a binary flag).
        if drains >= 50 or (tier == "confirmed" and victims >= 200):
            sev = "critical"
        elif drains >= 10 or (tier == "confirmed" and victims >= 50):
            sev = "high"
        elif drains >= 1 or (tier == "suspected" and victims >= 20):
            sev = "medium"
        elif tier in ("confirmed", "suspected"):
            sev = "low"
        else:
            sev = "baseline"

        addr = d.get("contract_address") or ""
        short = f"{addr[:8]}…{addr[-4:]}" if len(addr) > 14 else addr
        points.append({
            "name":     short,
            "address":  addr,
            "chain":    d.get("chain"),
            "tier":     tier,
            "deployer": d.get("deployer_address"),
            "risk":     risk,
            "victims":  victims,
            "drains":   drains,
            "sev":      sev,
        })

    by_sev = {}
    for p in points:
        by_sev[p["sev"]] = by_sev.get(p["sev"], 0) + 1

    return JSONResponse({
        "points": points,
        "total": len(points),
        "by_sev": by_sev,
    })


@app.get("/api/guard/triage")
def guard_triage(limit: int = 60):
    """Triage feed — active watchlist entries ranked by priority × recency.
    Each row carries a parsed title + reasoning trace from watch_reason.
    """
    c = conn()
    cur = c.cursor()
    rows = cur.execute(
        "SELECT id, address, address_type, entity_name, watch_reason, priority, "
        "       added_date, last_seen_date, hit_count "
        "FROM watchlist "
        "WHERE active = 1 "
        "ORDER BY "
        "  CASE priority "
        "    WHEN 'CRITICAL' THEN 0 "
        "    WHEN 'HIGH'     THEN 1 "
        "    WHEN 'MEDIUM'   THEN 2 "
        "    WHEN 'LOW'      THEN 3 "
        "    ELSE 4 END, "
        "  COALESCE(last_seen_date, added_date) DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    c.close()

    SEV_MAP  = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
    RISK_MAP = {"CRITICAL": 9.4, "HIGH": 7.2, "MEDIUM": 4.8, "LOW": 2.5}

    def parse_why(reason):
        if not reason:
            return []
        # Split on sentence-end. Filter very short fragments.
        parts = []
        for chunk in reason.replace("\n", " ").split(". "):
            s = chunk.strip().rstrip(".")
            if len(s) >= 6:
                parts.append(s)
        return parts[:8]

    alerts = []
    for r in rows:
        d = dict(r)
        why_all = parse_why(d.get("watch_reason"))
        title = (why_all[0] if why_all else (d.get("entity_name") or "Watched entity"))
        why = why_all[1:7] if len(why_all) > 1 else []
        sev = SEV_MAP.get(d.get("priority"), "baseline")
        if len(title) > 110:
            title = title[:107] + "..."
        alerts.append({
            "id":           f"w{d['id']}",
            "sev":          sev,
            "priority":     d.get("priority"),
            "risk":         RISK_MAP.get(d.get("priority"), 1.0),
            "hit_count":    d.get("hit_count") or 0,
            "target":       d.get("entity_name") or d.get("address"),
            "address":      d.get("address"),
            "address_type": d.get("address_type"),
            "title":        title,
            "why":          why,
            "last_seen":    d.get("last_seen_date"),
            "added_date":   d.get("added_date"),
        })

    by_priority = {}
    for a in alerts:
        by_priority[a["sev"]] = by_priority.get(a["sev"], 0) + 1

    return JSONResponse({
        "alerts": alerts,
        "total": len(alerts),
        "by_priority": by_priority,
    })


@app.get("/api/guard/status")
def guard_status():
    """Status rail — ingest health, model, throughput, active alert count."""
    from datetime import datetime, timezone, timedelta
    c = conn()
    cur = c.cursor()

    hb_rows = cur.execute(
        "SELECT component, MAX(timestamp) AS last_ts, MAX(blocks) AS blocks "
        "FROM heartbeat "
        "WHERE component LIKE 'deployment_monitor_%' AND component != 'deployment_monitor' "
        "GROUP BY component"
    ).fetchall()

    ev_last_hour = cur.execute(
        "SELECT COUNT(*) FROM transaction_events "
        "WHERE timestamp >= datetime('now','-1 hour')"
    ).fetchone()[0]

    crit = cur.execute(
        "SELECT COUNT(*) FROM watchlist WHERE active=1 AND priority='CRITICAL'"
    ).fetchone()[0]
    high = cur.execute(
        "SELECT COUNT(*) FROM watchlist WHERE active=1 AND priority='HIGH'"
    ).fetchone()[0]
    c.close()

    now_utc = datetime.now(timezone.utc)
    heartbeats = []
    most_recent_ts = None
    fresh_count = 0
    for component, ts_str, blocks in hb_rows:
        ts = None
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        stale = ts is None or (now_utc - ts) > timedelta(minutes=10)
        if ts and (most_recent_ts is None or ts > most_recent_ts):
            most_recent_ts = ts
        if not stale:
            fresh_count += 1
        heartbeats.append({
            "component": component,
            "last_ts": ts_str,
            "blocks": blocks,
            "stale": stale,
        })

    # Ingest is nominal if at least 2 of 3 chain monitors are fresh.
    ingest_status = "nominal" if fresh_count >= max(1, len(heartbeats) - 1) else "stale"

    return JSONResponse({
        "ingest": ingest_status,
        "heartbeats": heartbeats,
        "fresh_count": fresh_count,
        "heartbeat_count": len(heartbeats),
        "last_heartbeat": most_recent_ts.isoformat() if most_recent_ts else None,
        "events_per_second": round(ev_last_hour / 3600, 2),
        "events_last_hour": ev_last_hour,
        "active_critical": crit,
        "active_high": high,
        "active_total": crit + high,
        "model": "L3-RISK v0.4",
    })


# ---------------------------------------------------------------
# Threat Feed
# ---------------------------------------------------------------

@app.get("/threats", response_class=HTMLResponse)
async def threats_view(request: Request, chain: str = None, priority: str = None, entity: str = None):
    c = conn()
    threats = get_threats(c, 100, chain or None, priority or None, entity or None)
    counts = get_threat_counts(c)
    entities = get_watched_entities(c)
    stats = get_overview_stats(c)
    c.close()

    return templates.TemplateResponse("threats.html", {
        "request": request, "active_page": "threats",
        "stats": stats, "threats": threats, "counts": counts,
        "entities": entities, "chain": chain, "priority": priority, "entity": entity,
    })


# ---------------------------------------------------------------
# API: Address search
# ---------------------------------------------------------------

@app.get("/api/search")
async def api_search(q: str = ""):
    c = conn()
    result = search_address(c, q)
    c.close()
    return JSONResponse(result)


@app.get("/api/address/{address}")
async def api_address(address: str):
    c = conn()
    result = get_address_detail(c, address)
    c.close()
    if not result:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(result)


@app.get("/api/watchlist/recent")
async def api_watchlist_recent(limit: int = 5):
    c = conn()
    result = get_recent_watchlist_hits(c, limit)
    c.close()
    return JSONResponse(result)


# ---------------------------------------------------------------
# API: SAI alerts (the SAI substrate's persistence layer)
# Backs the question store at memory/questions.yaml and the detectors at
# surveillance/{sai,analytics,ontology}/. Each row is an alert from one
# of the wired SAI detectors (Q-002, Q-003, Q-005, Q-009) plus metadata.
# ---------------------------------------------------------------

@app.get("/api/sai/alerts")
def api_sai_alerts(
    detector: str = "",
    severity: str = "",
    since: str = "",
    subject: str = "",
    limit: int = 100,
):
    """List SAI alerts. Filters: detector, severity, since (ISO date),
    subject (full address or prefix).

    Examples:
      /api/sai/alerts                            — most recent 100
      /api/sai/alerts?detector=Q-002             — only Q-002 alerts
      /api/sai/alerts?severity=STALE             — only STALE
      /api/sai/alerts?since=2026-05-15           — since the given date
      /api/sai/alerts?subject=0x80b12bd0         — for a specific address
    """
    import json as _json
    c = conn()
    where = []
    params = []
    if detector:
        where.append("detector = ?")
        params.append(detector)
    if severity:
        where.append("severity = ?")
        params.append(severity)
    if since:
        where.append("detected_at >= ?")
        params.append(since)
    if subject:
        if len(subject) >= 42:
            where.append("subject_address = ?")
            params.append(subject.lower())
        else:
            where.append("subject_address LIKE ?")
            params.append(subject.lower() + "%")
    sql = ("SELECT detected_at, detector, severity, subject_address, "
           "subject_kind, payload FROM sai_alerts")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY detected_at DESC LIMIT ?"
    params.append(int(limit))
    try:
        rows = c.execute(sql, params).fetchall()
    except Exception as e:
        c.close()
        # Likely: table doesn't exist yet (no scheduled tick has run with
        # --persist on this deployment). Return empty with reason.
        return JSONResponse({
            "alerts": [],
            "count": 0,
            "note": f"sai_alerts table not yet populated: {e}",
        })
    c.close()
    alerts = []
    for r in rows:
        try:
            payload = _json.loads(r[5]) if r[5] else None
        except (_json.JSONDecodeError, TypeError):
            payload = None
        alerts.append({
            "detected_at": r[0],
            "detector": r[1],
            "severity": r[2],
            "subject_address": r[3],
            "subject_kind": r[4],
            "payload": payload,
        })
    return JSONResponse({
        "alerts": alerts,
        "count": len(alerts),
        "filters": {
            "detector": detector or None,
            "severity": severity or None,
            "since": since or None,
            "subject": subject or None,
        },
    })


@app.get("/api/sai/alerts/summary")
def api_sai_alerts_summary():
    """Aggregate summary: counts by detector × severity, latest detection
    timestamp per detector, and total alerts.

    This is the dashboard-card view of the SAI alert surface.
    """
    c = conn()
    try:
        by_det_sev = c.execute(
            "SELECT detector, severity, COUNT(*) FROM sai_alerts "
            "GROUP BY detector, severity ORDER BY detector, severity"
        ).fetchall()
        latest = c.execute(
            "SELECT detector, MAX(detected_at) FROM sai_alerts GROUP BY detector"
        ).fetchall()
        total = c.execute("SELECT COUNT(*) FROM sai_alerts").fetchone()[0]
    except Exception as e:
        c.close()
        return JSONResponse({
            "total": 0,
            "by_detector_severity": [],
            "latest_per_detector": [],
            "note": f"sai_alerts table not yet populated: {e}",
        })
    c.close()
    return JSONResponse({
        "total": total,
        "by_detector_severity": [
            {"detector": d, "severity": s, "count": n}
            for d, s, n in by_det_sev
        ],
        "latest_per_detector": [
            {"detector": d, "latest_detected_at": ts}
            for d, ts in latest
        ],
    })


@app.get("/api/sai/questions")
def api_sai_questions():
    """List the SAI question store with priority ranking.

    Loads memory/questions.yaml and returns the ranked list. This is the
    read-only mirror of the question substrate that feeds the surveillance
    detectors.
    """
    try:
        from surveillance.sai.question_store import load_questions, rank
        qs = rank(load_questions(), status_filter="active")
    except Exception as e:
        return JSONResponse({
            "questions": [],
            "note": f"question store not loadable: {e}",
        })
    return JSONResponse({
        "questions": [
            {
                "id": q.id,
                "category": q.category,
                "question": " ".join(q.question.split()),
                "priority_score": round(q.priority_score(), 4),
                "score": {
                    "predictive_power": q.predictive_power,
                    "actionability": q.actionability,
                    "uniqueness": q.uniqueness,
                    "failure_reduction": q.failure_reduction,
                },
                "implementation_target": q.implementation_target,
                "status": q.status,
                "evolves_from": q.evolves_from,
            }
            for q in qs
        ],
        "count": len(qs),
    })
