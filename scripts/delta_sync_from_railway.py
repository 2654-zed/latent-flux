"""Delta-sync Railway -> local.

For each consumable table, computes the local cursor (max id or max
timestamp), then pages `/dump?since_id=...` or `/dump?since_ts=...` on
Railway and INSERTs new rows into the local DB. Stops when the server
returns no more rows for that cursor.

Per-table strategy (tables without a stable insertable id use timestamp
cursors):

  id-cursored (INSERT OR IGNORE on PK id):
    alerts, trap_events, transaction_events (biggest), bot_candidate_events,
    x402_events, liquidity_events, approval_events, bridge_events,
    pair_creation_events, org_transfer_events, behavioral_anomalies,
    connection_gaps, org_candidates, trust_amplification, bait_profiles,
    strategy_lifecycle, predictions, bytecode_families, bytecode_family_members,
    camouflage_metrics, daily_metrics, cluster_events, detector_precision,
    approval_watchlist, x402_permit2_exposure, x402_facilitators,
    dormant_activations, timelock_countdowns, sload_patterns, watchlist_hits,
    bot_sophistication, drain_values

  primary-key-cursored (no auto-id; use PK as cursor, INSERT OR REPLACE):
    contracts (PK=contract_address, cursor_col=detection_timestamp)
    deployers (PK=deployer_address, cursor_col=last_seen)
    bot_candidates (PK=address, cursor_col=last_seen)
    deployer_profiles (PK=deployer_address, cursor_col=profiled_at)
    bot_strategies (PK=bot_address, cursor_col=last_seen)
    entity_classification (PK=address, cursor_col=last_updated)
    infrastructure_registry (PK=(address,chain), cursor_col=verified_at)
    org_wallets (PK=(address,chain), cursor_col=added_at)
    extraction_events (has id, but small table; use since_ts=observed_at)

Skipped (irrelevant for analysis or too large + low-value):
    bytecode_cache (can grow large; recomputable; skip)
    deployer_similarity (556k+ rows; recomputable; skip)
    heartbeat (service metadata)
    api_keys, api_watches (auth infra)
    false_positives (small; use INSERT OR REPLACE)

The script is idempotent — a second run with no new Railway data does
nothing.
"""
import json
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://spypy.up.railway.app"
TOKEN = "jfwufhnsuisfj"
LIMIT = 5000  # per chunk

LOCAL_DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")


# (table, cursor_mode, cursor_col, pk_cols, on_conflict)
# cursor_mode: "id" | "ts"
# on_conflict: "IGNORE" | "REPLACE"
TABLES = [
    # High-volume event streams with auto-increment id (IGNORE duplicates)
    ("alerts",                    "id", "id",                 ["id"], "IGNORE"),
    ("trap_events",               "id", "id",                 ["id"], "IGNORE"),
    ("transaction_events",        "id", "id",                 ["id"], "IGNORE"),
    ("bot_candidate_events",      "id", "id",                 ["id"], "IGNORE"),
    ("x402_events",               "id", "id",                 ["id"], "IGNORE"),
    ("org_transfer_events",       "id", "id",                 ["id"], "IGNORE"),
    ("liquidity_events",          "id", "id",                 ["id"], "IGNORE"),
    ("approval_events",           "id", "id",                 ["id"], "IGNORE"),
    ("bridge_events",             "id", "id",                 ["id"], "IGNORE"),
    ("pair_creation_events",      "id", "id",                 ["id"], "IGNORE"),
    ("cluster_events",            "id", "id",                 ["id"], "IGNORE"),
    ("dormant_activations",       "id", "id",                 ["id"], "IGNORE"),
    ("watchlist_hits",            "id", "id",                 ["id"], "IGNORE"),
    ("x402_permit2_exposure",     "id", "id",                 ["id"], "IGNORE"),
    ("connection_gaps",           "id", "id",                 ["id"], "IGNORE"),
    ("behavioral_anomalies",      "id", "id",                 ["id"], "IGNORE"),

    # Analysis outputs with id (REPLACE to keep latest scores)
    ("trust_amplification",       "id", "id",                 ["id"], "REPLACE"),
    ("bait_profiles",             "id", "id",                 ["id"], "REPLACE"),
    ("strategy_lifecycle",        "id", "id",                 ["id"], "REPLACE"),
    ("bytecode_families",         "id", "id",                 ["id"], "REPLACE"),
    ("bytecode_family_members",   "id", "id",                 ["id"], "REPLACE"),
    ("camouflage_metrics",        "id", "id",                 ["id"], "REPLACE"),
    ("daily_metrics",             "id", "id",                 ["id"], "REPLACE"),
    ("org_candidates",            "id", "id",                 ["id"], "REPLACE"),
    ("bot_strategies",            "id", "id",                 ["id"], "REPLACE"),
    ("predictions",               "id", "id",                 ["id"], "REPLACE"),

    # PK-cursored (no auto-id; use timestamp col)
    ("contracts",                 "ts", "detection_timestamp", ["contract_address"], "REPLACE"),
    ("deployers",                 "ts", "last_seen",          ["deployer_address"], "REPLACE"),
    ("bot_candidates",            "ts", "last_seen",          ["address"], "REPLACE"),
    ("deployer_profiles",         "ts", "profiled_at",        ["deployer_address"], "REPLACE"),
    ("entity_classification",     "ts", "last_updated",       ["address"], "REPLACE"),
    ("infrastructure_registry",   "ts", "verified_at",        ["address", "chain"], "REPLACE"),
    ("org_wallets",               "ts", "added_at",           ["address", "chain"], "REPLACE"),
    ("extraction_events",         "ts", "observed_at",        ["event_id"], "REPLACE"),
    ("approval_watchlist",        "ts", "logged_at",          ["approve_tx_hash"], "REPLACE"),
    ("false_positives",           "ts", "assessed_at",        ["contract_address"], "REPLACE"),
    ("x402_facilitators",         "ts", "last_seen",          ["facilitator_address"], "REPLACE"),
    ("drain_values",              "ts", "scanned_at",         ["contract_address"], "REPLACE"),
    ("bot_sophistication",        "ts", "classified_at",      ["address"], "REPLACE"),
    ("detector_precision",        "ts", "date",               ["detector_name", "date"], "REPLACE"),
    ("timelock_countdowns",       "ts", "detected_at",        ["contract_address"], "REPLACE"),
    ("sload_patterns",            "ts", "categorized_at",     ["contract_address"], "REPLACE"),
]


def fetch_chunk(table: str, cursor_mode: str, cursor_col: str,
                cursor_val, offset: int) -> dict:
    if cursor_mode == "id":
        key = "since_id"
    else:
        key = "since_ts"
    val = str(cursor_val) if cursor_val is not None else ""
    url = (f"{BASE_URL}/dump?token={TOKEN}&table={table}&limit={LIMIT}"
           f"&offset={offset}&{key}={urllib.request.quote(val)}"
           f"&cursor_col={cursor_col}")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"    attempt {attempt+1} err on {table}: {e}; backoff 5s",
                  flush=True)
            time.sleep(5)
    raise RuntimeError(f"fetch failed on {table}")


def local_cursor(conn: sqlite3.Connection, table: str, cursor_col: str):
    """Return local's max(cursor_col) for this table, or None if table empty/missing."""
    try:
        row = conn.execute(f"SELECT MAX([{cursor_col}]) FROM [{table}]").fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def sync_table(conn: sqlite3.Connection, table: str, cursor_mode: str,
               cursor_col: str, pk_cols: list, on_conflict: str) -> tuple[int, int]:
    """Return (fetched, inserted). Handles pagination + client-side INSERT."""
    cursor_val = local_cursor(conn, table, cursor_col)
    fetched = 0
    inserted = 0
    first = fetch_chunk(table, cursor_mode, cursor_col, cursor_val, 0)
    if first.get("error"):
        print(f"    {table}: server error {first['error']}", flush=True)
        return 0, 0
    total_filtered = first.get("total_filtered", 0)
    print(f"  {table:<32}  local_cursor={str(cursor_val)[:20]:<22}  "
          f"railway_new={total_filtered:,}", flush=True)
    offset = 0
    chunk = first
    while True:
        rows = chunk.get("rows", [])
        if not rows:
            break
        inserted += _insert_rows(conn, table, rows, on_conflict)
        fetched += len(rows)
        offset += len(rows)
        if offset >= total_filtered or len(rows) < LIMIT:
            break
        chunk = fetch_chunk(table, cursor_mode, cursor_col, cursor_val, offset)
    conn.commit()
    return fetched, inserted


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list, on_conflict: str) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    col_list = ",".join(f"[{c}]" for c in cols)
    placeholders = ",".join("?" * len(cols))
    verb = "INSERT OR IGNORE" if on_conflict == "IGNORE" else "INSERT OR REPLACE"
    sql = f"{verb} INTO [{table}] ({col_list}) VALUES ({placeholders})"
    total_changes_before = conn.total_changes
    for r in rows:
        try:
            conn.execute(sql, [r.get(c) for c in cols])
        except sqlite3.IntegrityError as e:
            print(f"    IntegrityError on {table}: {e}", flush=True)
    return conn.total_changes - total_changes_before


def main():
    conn = sqlite3.connect(str(LOCAL_DB), timeout=60)
    conn.execute("PRAGMA foreign_keys=OFF")  # many event tables have FKs to deployer/contract

    print(f"Local DB: {LOCAL_DB}", flush=True)
    print(f"Railway:  {BASE_URL}", flush=True)
    print()

    t0 = time.time()
    totals = []
    for table, mode, col, pks, conflict in TABLES:
        try:
            fetched, inserted = sync_table(conn, table, mode, col, pks, conflict)
            totals.append((table, fetched, inserted))
        except Exception as e:
            print(f"    {table}: SYNC ERR {type(e).__name__}: {e}", flush=True)
            totals.append((table, 0, 0))

    conn.close()

    elapsed = time.time() - t0
    total_fetched = sum(t[1] for t in totals)
    total_inserted = sum(t[2] for t in totals)
    print()
    print(f"=== SYNC SUMMARY (elapsed {elapsed:.1f}s) ===")
    print(f"{'table':<32} {'fetched':>10} {'inserted':>10}")
    for table, fetched, inserted in totals:
        if fetched or inserted:
            print(f"{table:<32} {fetched:>10,} {inserted:>10,}")
    print(f"{'-' * 54}")
    print(f"{'TOTAL':<32} {total_fetched:>10,} {total_inserted:>10,}")


if __name__ == "__main__":
    main()
