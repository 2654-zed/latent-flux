# Layer 3 Web UI

FastAPI + HTMX + Tailwind CSS + Cytoscape.js

## Run

```bash
cd "ai lang"
DB_PATH=surveillance/data/surveillance.db uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Views

| Route | View | Description |
|---|---|---|
| `/` | Overview | System stats, key metrics, strategy lifecycle, bot sophistication, recent alerts |
| `/org-map` | Organization Map | Cytoscape.js graph of criminal organizations. Click nodes for details. |
| `/contract/{addr}` | Contract Inspector | Bytecode classification, interaction stats, self-test status, approval exposure |
| `/deployer/{addr}` | Deployer Profile | Behavioral fingerprint, similar deployers, contract list, funding chain |
| `/threats` | Threat Feed | Filterable watchlist hit stream (chain, priority) via HTMX |

## API Endpoints

| Route | Returns |
|---|---|
| `/api/org/{org_id}` | JSON graph data for Cytoscape.js |
| `/api/search?q=0x...` | Contract/deployer search result |

## Environment

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `surveillance/data/surveillance.db` | Path to SQLite database |

## Stack

- **Backend:** FastAPI, Jinja2 templates
- **Interactivity:** HTMX (no JS framework)
- **Styling:** Tailwind CSS via CDN (dark theme)
- **Graph:** Cytoscape.js via CDN
- **Database:** SQLite (read-only)
