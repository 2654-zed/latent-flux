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
