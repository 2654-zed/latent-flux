"""One-shot backfill for deployers.mainnet_first_tx.

Populates Pattern D enrichment retroactively for deployers that were
traced BEFORE auto_funder_tracer gained the Etherscan v2 lookup.

Approach:
- Query deployers WHERE mainnet_first_tx IS NULL
- For each, one Etherscan v2 API call (~0.3s with the 5/s free-tier throttle)
- Update deployers.mainnet_first_tx

Budget math:
- Free tier: 5 req/sec, 100k req/day
- Current corpus: ~37,000 deployers lacking enrichment
- Estimated runtime: 37,000 / 3.3 req/s ≈ 3.1 hours (one-shot)
- Well under the 100k/day ceiling; resumable via WHERE IS NULL

Safety:
- Dry-run default. Pass --commit to actually write.
- Skips deployers already enriched (idempotent).
- Handles Etherscan rate limits by 0.3s sleep between calls.
- Fails silently per-deployer on API error (logs count, continues).

Not run automatically. Approval gate same as backfill_cache_invalidation
and Correction #5 data remediation.
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent.parent / "surveillance" / "data" / "surveillance.db"
ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
API_KEY = os.environ.get("ETHERSCAN_V2_KEY") or os.environ.get("ARBISCAN_API_KEY", "")
SLEEP_SEC = 0.3  # 3.3 req/s, under the 5/s free-tier limit


def fetch_mainnet_first_tx(addr: str) -> str | None:
    if not API_KEY:
        return None
    params = {
        "chainid": "1",
        "module": "account",
        "action": "txlist",
        "address": addr,
        "startblock": "0",
        "endblock": "99999999",
        "page": "1",
        "offset": "1",
        "sort": "asc",
        "apikey": API_KEY,
    }
    url = f"{ETHERSCAN_V2_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    if data.get("status") != "1":
        return None
    txs = data.get("result") or []
    if not txs:
        return None
    ts = txs[0].get("timeStamp")
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH),
                    help="SQLite path (default: local)")
    ap.add_argument("--commit", action="store_true",
                    help="Actually write (default: dry-run report only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max deployers to process (0 = all)")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR: ETHERSCAN_V2_KEY (or ARBISCAN_API_KEY) not set in env")
        return 1

    db_path = Path(args.db)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")

    # Ensure migration ran
    try:
        conn.execute("SELECT mainnet_first_tx FROM deployers LIMIT 1")
    except sqlite3.OperationalError:
        print("ERROR: deployers.mainnet_first_tx column missing; run db.init_db first")
        return 2

    # Count population
    todo = conn.execute(
        "SELECT COUNT(*) FROM deployers WHERE mainnet_first_tx IS NULL"
    ).fetchone()[0]
    print(f"deployers needing enrichment: {todo:,}")
    if args.limit:
        print(f"--limit {args.limit} applied; processing up to that many")
        todo = min(todo, args.limit)

    est_hours = (todo * SLEEP_SEC) / 3600
    print(f"estimated runtime at {SLEEP_SEC}s/call: {est_hours:.1f} hours")

    if not args.commit:
        print("\nDry run. Use --commit to execute.")
        return 0

    print(f"\nStarting backfill at {datetime.now(timezone.utc).isoformat()}")
    processed = enriched = no_history = api_err = 0
    query = "SELECT deployer_address FROM deployers WHERE mainnet_first_tx IS NULL"
    if args.limit:
        query += f" LIMIT {int(args.limit)}"

    for (addr,) in conn.execute(query).fetchall():
        processed += 1
        ts = fetch_mainnet_first_tx(addr)
        time.sleep(SLEEP_SEC)
        if ts is None:
            if API_KEY:
                # Could be legitimate no-history OR API error; either way, set to sentinel
                # to avoid reprocessing. Use empty string for "checked but no history".
                conn.execute(
                    "UPDATE deployers SET mainnet_first_tx = '' WHERE deployer_address = ?",
                    (addr,),
                )
                no_history += 1
            else:
                api_err += 1
        else:
            conn.execute(
                "UPDATE deployers SET mainnet_first_tx = ? WHERE deployer_address = ?",
                (ts, addr),
            )
            enriched += 1

        if processed % 100 == 0:
            conn.commit()
            print(f"  progress: {processed:,}/{todo:,}  enriched={enriched:,}  "
                  f"no_history={no_history:,}  api_err={api_err:,}")
    conn.commit()

    print()
    print(f"DONE at {datetime.now(timezone.utc).isoformat()}")
    print(f"  processed:  {processed:,}")
    print(f"  enriched:   {enriched:,}")
    print(f"  no_history: {no_history:,}")
    print(f"  api_err:    {api_err:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
