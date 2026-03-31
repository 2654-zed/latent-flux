"""
Sync Railway surveillance DB to local SQLite.

Pulls each table via the /dump endpoint and upserts into local DB.
Usage: python sync_railway_db.py [--base-url URL] [--token TOKEN]
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = os.environ.get("RAILWAY_URL", "https://spypy.up.railway.app")
TOKEN = os.environ.get("ADMIN_TOKEN", "jlfsafjiefnajsf")
LOCAL_DB = Path(__file__).resolve().parent / "surveillance" / "data" / "surveillance.db"

# Tables to sync in dependency order (parents before children)
TABLES = [
    "deployers",
    "contracts",
    "trap_events",
    "transaction_events",
    "bot_candidates",
    "bot_candidate_selectors",
    "bot_candidate_events",
    "known_selectors",
    "alerts",
    "live_exposures",
    "pattern_matches",
    "cluster_events",
    "funding_hops",
    "contract_verification",
    "traces",
    "bytecode_cache",
    "heartbeat",
    "connection_gaps",
    "liquidity_events",
    "approval_events",
    "bridge_events",
    "pair_creation_events",
    "cex_deposit_candidates",
    "org_transfer_events",
    "watchlist",
    "watchlist_hits",
    "self_test_traps",
    "approval_watchlist",
]

BATCH_SIZE = 5000


def fetch_table(table, offset=0, limit=BATCH_SIZE):
    """Fetch a batch of rows from Railway /dump endpoint."""
    params = urllib.parse.urlencode({
        "token": TOKEN, "table": table, "offset": offset, "limit": limit
    })
    url = f"{BASE_URL}/dump?{params}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR fetching {table} offset={offset}: {e}")
        return None


def get_table_columns(conn, table):
    """Get column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info([{table}])")
    return [row[1] for row in cursor.fetchall()]


def upsert_rows(conn, table, rows, columns):
    """Insert or replace rows into local DB."""
    if not rows:
        return 0
    # Filter row keys to only columns that exist in local table
    placeholders = ", ".join("?" * len(columns))
    col_names = ", ".join(f"[{c}]" for c in columns)
    sql = f"INSERT OR REPLACE INTO [{table}] ({col_names}) VALUES ({placeholders})"
    count = 0
    for row in rows:
        values = [row.get(c) for c in columns]
        try:
            conn.execute(sql, values)
            count += 1
        except sqlite3.IntegrityError as e:
            # FK constraint or other issue — try INSERT OR IGNORE
            try:
                sql_ignore = f"INSERT OR IGNORE INTO [{table}] ({col_names}) VALUES ({placeholders})"
                conn.execute(sql_ignore, values)
                count += 1
            except Exception:
                pass
    return count


def sync_table(conn, table):
    """Sync a single table from Railway to local DB."""
    local_columns = get_table_columns(conn, table)
    if not local_columns:
        print(f"  SKIP {table}: table not in local DB schema")
        return

    offset = 0
    total_synced = 0
    remote_total = None

    while True:
        data = fetch_table(table, offset=offset)
        if data is None:
            print(f"  FAILED {table} at offset {offset}")
            break

        if remote_total is None:
            remote_total = data.get("total", 0)
            print(f"  {table}: {remote_total} rows on Railway", end="", flush=True)

        rows = data.get("rows", [])
        if not rows:
            break

        # Only use columns that exist in both remote data and local schema
        if rows:
            remote_columns = set(rows[0].keys())
            sync_columns = [c for c in local_columns if c in remote_columns]
        else:
            sync_columns = local_columns

        inserted = upsert_rows(conn, table, rows, sync_columns)
        total_synced += inserted
        offset += len(rows)
        print(".", end="", flush=True)

        if len(rows) < BATCH_SIZE:
            break

        time.sleep(0.2)  # Be nice to Railway

    conn.commit()
    local_count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    print(f" -> synced {total_synced}, local now has {local_count}")


def main():
    # Parse args
    base_url = BASE_URL
    token = TOKEN
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--base-url" and i < len(sys.argv) - 1:
            base_url = sys.argv[i + 1]
        elif arg == "--token" and i < len(sys.argv) - 1:
            token = sys.argv[i + 1]

    print(f"Syncing from {base_url}")
    print(f"Local DB: {LOCAL_DB}")
    print()

    # Ensure local DB has schema
    conn = sqlite3.connect(str(LOCAL_DB), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")  # Disable FK checks during sync

    # First, verify we can reach Railway
    test = fetch_table("contracts", offset=0, limit=1)
    if test is None:
        print("Cannot reach Railway /dump endpoint.")
        print("Make sure the /dump endpoint is deployed (push latest run_surveillance.py)")
        conn.close()
        sys.exit(1)

    print(f"Railway connected. Starting sync...")
    print()

    for table in TABLES:
        sync_table(conn, table)

    conn.execute("PRAGMA foreign_keys=ON")
    conn.close()
    print()
    print("Sync complete.")


if __name__ == "__main__":
    main()
