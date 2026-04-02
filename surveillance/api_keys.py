"""
Layer 3 — API Key Management

Usage:
    python -m surveillance.api_keys --create --name "Customer" --tier 2
    python -m surveillance.api_keys --list
    python -m surveillance.api_keys --revoke l3_xxxxxxxx
"""

import argparse
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            tier INTEGER NOT NULL DEFAULT 1,
            created_at TEXT,
            last_used TEXT,
            requests_today INTEGER DEFAULT 0,
            requests_reset_date TEXT,
            rate_limit INTEGER DEFAULT 100,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS api_watches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            address TEXT NOT NULL,
            chain TEXT,
            alert_types TEXT,
            created_at TEXT,
            FOREIGN KEY (api_key) REFERENCES api_keys(key)
        );

        CREATE INDEX IF NOT EXISTS idx_api_watches_key ON api_watches(api_key);
        CREATE INDEX IF NOT EXISTS idx_api_watches_addr ON api_watches(address);
    """)


def generate_key() -> str:
    return "l3_" + secrets.token_hex(16)


def create_key(conn: sqlite3.Connection, name: str, tier: int = 1,
               rate_limit: int = 100) -> str:
    ensure_tables(conn)
    key = generate_key()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO api_keys (key, name, tier, created_at, rate_limit) VALUES (?, ?, ?, ?, ?)",
        (key, name, tier, now, rate_limit),
    )
    conn.commit()
    return key


def list_keys(conn: sqlite3.Connection) -> list:
    ensure_tables(conn)
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(
        "SELECT key, name, tier, created_at, last_used, requests_today, rate_limit, active FROM api_keys ORDER BY created_at DESC"
    ).fetchall()]


def revoke_key(conn: sqlite3.Connection, key: str) -> bool:
    ensure_tables(conn)
    cursor = conn.execute("UPDATE api_keys SET active = 0 WHERE key = ?", (key,))
    conn.commit()
    return cursor.rowcount > 0


def validate_key(conn: sqlite3.Connection, key: str) -> dict | None:
    """Validate an API key and return its record, or None if invalid."""
    row = conn.execute(
        "SELECT key, name, tier, rate_limit, requests_today, requests_reset_date, active FROM api_keys WHERE key = ?",
        (key,),
    ).fetchone()
    if not row:
        return None
    record = dict(row) if hasattr(row, 'keys') else {
        'key': row[0], 'name': row[1], 'tier': row[2], 'rate_limit': row[3],
        'requests_today': row[4], 'requests_reset_date': row[5], 'active': row[6],
    }
    if not record['active']:
        return None
    return record


def track_usage(conn: sqlite3.Connection, key: str):
    """Increment request count and update last_used."""
    now = datetime.now(timezone.utc)
    today = now.strftime('%Y-%m-%d')
    now_iso = now.isoformat()
    # Reset counter if it's a new day
    conn.execute("""
        UPDATE api_keys SET
            requests_today = CASE WHEN requests_reset_date = ? THEN requests_today + 1 ELSE 1 END,
            requests_reset_date = ?,
            last_used = ?
        WHERE key = ?
    """, (today, today, now_iso, key))


def check_rate_limit(conn: sqlite3.Connection, key: str) -> bool:
    """Return True if under rate limit, False if exceeded."""
    row = conn.execute(
        "SELECT requests_today, rate_limit, requests_reset_date FROM api_keys WHERE key = ?",
        (key,),
    ).fetchone()
    if not row:
        return False
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    requests = row[0] if row[2] == today else 0
    return requests < row[1]


# -- CLI --

def main():
    parser = argparse.ArgumentParser(description="Layer 3 — API Key Manager")
    parser.add_argument("--create", action="store_true", help="Create a new API key")
    parser.add_argument("--list", action="store_true", help="List all API keys")
    parser.add_argument("--revoke", type=str, help="Revoke an API key")
    parser.add_argument("--name", type=str, default="unnamed", help="Customer name")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Access tier")
    parser.add_argument("--rate-limit", type=int, default=100, help="Requests per day")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    db = args.db or os.environ.get("DB_PATH", str(DB_PATH))
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    if args.create:
        key = create_key(conn, args.name, args.tier, args.rate_limit)
        print(f"API key created:")
        print(f"  Key:  {key}")
        print(f"  Name: {args.name}")
        print(f"  Tier: {args.tier}")
        print(f"  Rate: {args.rate_limit} requests/day")
    elif args.list:
        keys = list_keys(conn)
        if not keys:
            print("No API keys found.")
        for k in keys:
            status = "ACTIVE" if k['active'] else "REVOKED"
            print(f"  [{status}] {k['key']}  name={k['name']}  tier={k['tier']}  used={k['last_used'] or 'never'}  requests_today={k['requests_today']}")
    elif args.revoke:
        if revoke_key(conn, args.revoke):
            print(f"Revoked: {args.revoke}")
        else:
            print(f"Key not found: {args.revoke}")
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
