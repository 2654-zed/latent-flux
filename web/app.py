"""
Layer 3 — Web UI
FastAPI + HTMX + Tailwind + Cytoscape.js

Run: DB_PATH=surveillance/data/surveillance.db uvicorn web.app:app --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.data import (
    get_conn, get_overview_stats, get_key_metrics, get_chain_split,
    get_daily_trend, get_recent_alerts, get_strategy_lifecycle,
    get_bot_sophistication, get_org_graph, get_org_001_stats,
    get_contract, get_deployer,
    get_threats, get_threat_counts, get_watched_entities, search_address,
)

app = FastAPI(title="Layer 3 Intelligence", docs_url=None, redoc_url=None)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DB = os.environ.get("DB_PATH", str(BASE_DIR.parent / "surveillance" / "data" / "surveillance.db"))


def conn():
    return get_conn(DB)


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
