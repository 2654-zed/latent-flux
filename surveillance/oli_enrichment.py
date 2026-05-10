"""Layer 3 — Open Labels Initiative (OLI) enrichment.

Pulls institutional address labels from Blockscout's metadata service
(which aggregates Open Labels Initiative tags). The check exists because
behavioral/topology classifiers cannot distinguish "high-fanout CEX hot
wallet" from "high-fanout single-purpose-funder for trap fleet" by shape
alone — the distinguishing signal is identity, surfaced via public labels.

This module is the architectural fix for the gap that produced
Correction #20 (bb50 / Circle deployer mislabeled as Pristine Solo
Operator) and the 11+ HIGH-severity OLI mismatches surfaced by the
2026-05-09 mass audit (`reports/blockscout_tag_audit_2026-05-09.csv`).

Source: https://metadata.services.blockscout.com/api/v1/metadata
        which aggregates labels from https://www.openlabelsinitiative.org/

Usage:
    # One-off lookup
    python -m surveillance.oli_enrichment --address 0xbb50ce87...

    # Backfill watchlist (all active rows)
    python -m surveillance.oli_enrichment --backfill-watchlist

    # Backfill all malicious-flagged addresses
    # (watchlist + entity_classification CRIMINAL + infra_operator_candidates)
    python -m surveillance.oli_enrichment --backfill-flagged

    # Print HIGH-severity hits from cache (no fetch)
    python -m surveillance.oli_enrichment --hits

Programmatic:
    from surveillance.oli_enrichment import is_known_legitimate
    rec = is_known_legitimate(conn, "0xbb50ce87...", chain_id=1)
    if rec and rec["severity"] == "HIGH":
        # Skip adversarial typology promotion
        ...
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
META_URL = "https://metadata.services.blockscout.com/api/v1/metadata"
DEFAULT_CHAIN_ID = 1  # Mainnet — institutional labels live here
BATCH_SIZE = 50
RATE_SLEEP_S = 0.4

# Severity heuristic — same one used by scripts/blockscout_tag_audit.py.
# HIGH = the address is publicly attributed to a known-legitimate institution
#        and should NOT be flagged as an adversarial operator.
# LOW  = the address has some public tag (deployer-of-X, project-Y) — likely
#        a real entity but lower-stakes than the institutional-class HIGH.
# self-confirming = OLI tag agrees with our adversarial flag (e.g., scam,
#        phishing, drain). Keep as confirmation, not retraction.
# none = no public tag — current classification stands as-is.
HIGH_SIGNAL_KEYWORDS = (
    # Major CEXes
    "circle", "coinbase", "binance", "kraken", "okx", "bybit", "gate", "kucoin",
    "mexc", "bitfinex", "huobi", "crypto.com", "robinhood", "moonpay",
    # DeFi blue chips
    "uniswap", "aave", "compound", "lido", "maker", "curve", "balancer",
    "1inch", "paraswap", "0x-protocol", "cow-protocol",
    # Bridges (high-fanout topology that consistently looks adversarial)
    "layerzero", "wormhole", "axelar", "synapse", "stargate", "across",
    "relay", "orbiter", "owlto", "hop-protocol",
    # Other infrastructure
    "ens", "eas", "safe", "argent", "rabby",
    "centre", "tether", "paypal", "pyusd",
    "thorchain", "railgun",
    # Generic markers
    "infrastructure", "official", "cex", "exchange", "deposit address",
)
SELF_CONFIRMING_KEYWORDS = ("scam", "phishing", "exploit", "drain", "hack", "stolen", "blocked")


# ─────────────────────────────────────────────────────────────────────
# Migration helper — call this once to ensure oli_labels table exists.
# Should also be added to db.py init_db migration chain so production
# acquires the table on next restart. Calling here makes the module
# self-bootstrapping for ad-hoc runs.
# ─────────────────────────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS oli_labels (
            address              TEXT    NOT NULL,
            chain_id             INTEGER NOT NULL DEFAULT 1,
            tags_json            TEXT,
            tag_count            INTEGER NOT NULL DEFAULT 0,
            primary_entity       TEXT,
            primary_tag_name     TEXT,
            severity             TEXT    NOT NULL DEFAULT 'none',
            fetched_at           TEXT    NOT NULL,
            PRIMARY KEY (address, chain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_oli_labels_severity
            ON oli_labels(severity);
        CREATE INDEX IF NOT EXISTS idx_oli_labels_entity
            ON oli_labels(primary_entity);
    """)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────
# Severity classifier — tag list -> severity tier + primary entity
# ─────────────────────────────────────────────────────────────────────
def classify_tags(tags: list[dict]) -> tuple[str, Optional[str], Optional[str]]:
    """Returns (severity, primary_entity, primary_tag_name)."""
    if not tags:
        return ("none", None, None)
    blob = " ".join((t.get("slug") or "") + "|" + (t.get("name") or "") for t in tags).lower()

    # primary_entity: prefer explicit main_entity meta on a name-tagType
    primary_entity = None
    primary_tag_name = None
    for t in tags:
        meta = t.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict) and meta.get("main_entity"):
            primary_entity = meta["main_entity"]
            primary_tag_name = t.get("name")
            break
    if primary_entity is None:
        # Fallback: take the highest-ordinal name-type tag's name
        name_tags = [t for t in tags if t.get("tagType") == "name"]
        if name_tags:
            t = max(name_tags, key=lambda x: x.get("ordinal", 0))
            primary_tag_name = t.get("name")

    if any(k in blob for k in SELF_CONFIRMING_KEYWORDS):
        return ("self-confirming", primary_entity, primary_tag_name)
    if any(k in blob for k in HIGH_SIGNAL_KEYWORDS):
        return ("HIGH", primary_entity, primary_tag_name)
    return ("LOW", primary_entity, primary_tag_name)


# ─────────────────────────────────────────────────────────────────────
# Network fetch
# ─────────────────────────────────────────────────────────────────────
def _fetch_metadata_batch(addresses: list[str], chain_id: int = DEFAULT_CHAIN_ID) -> dict[str, dict]:
    """Batch-call Blockscout metadata service. Returns {addr_lower: tag_dict}."""
    if not addresses:
        return {}
    params = {
        "addresses": ",".join(addresses),
        "chainId": str(chain_id),
        "tagsLimit": "20",
    }
    url = META_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "L3-oli-enrichment/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"  retry {attempt+1} after error: {e}", file=sys.stderr)
            time.sleep(2)
    raw = payload.get("addresses") or {}
    return {k.lower(): v for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────
# DB operations
# ─────────────────────────────────────────────────────────────────────
def _upsert_label(conn: sqlite3.Connection, address: str, chain_id: int, meta: dict) -> dict:
    """Compute severity + persist. Returns the row that was written."""
    tags = (meta or {}).get("tags") or []
    severity, primary_entity, primary_tag_name = classify_tags(tags)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "address": address.lower(),
        "chain_id": chain_id,
        "tags_json": json.dumps(tags) if tags else None,
        "tag_count": len(tags),
        "primary_entity": primary_entity,
        "primary_tag_name": primary_tag_name,
        "severity": severity,
        "fetched_at": now,
    }
    conn.execute("""
        INSERT INTO oli_labels (address, chain_id, tags_json, tag_count,
                                primary_entity, primary_tag_name, severity, fetched_at)
        VALUES (:address, :chain_id, :tags_json, :tag_count,
                :primary_entity, :primary_tag_name, :severity, :fetched_at)
        ON CONFLICT(address, chain_id) DO UPDATE SET
            tags_json = excluded.tags_json,
            tag_count = excluded.tag_count,
            primary_entity = excluded.primary_entity,
            primary_tag_name = excluded.primary_tag_name,
            severity = excluded.severity,
            fetched_at = excluded.fetched_at
    """, row)
    return row


def is_known_legitimate(conn: sqlite3.Connection, address: str,
                        chain_id: int = DEFAULT_CHAIN_ID) -> Optional[dict]:
    """Fast lookup. Returns the oli_labels row if address is HIGH-severity tagged.
    Does NOT trigger a network fetch — caller must call enrich_address first to
    populate the cache. Returns None for missing-from-cache or non-HIGH-severity.
    """
    row = conn.execute(
        "SELECT * FROM oli_labels WHERE address = ? AND chain_id = ?",
        (address.lower(), chain_id),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    return d if d["severity"] == "HIGH" else None


def enrich_address(conn: sqlite3.Connection, address: str,
                   chain_id: int = DEFAULT_CHAIN_ID, force: bool = False) -> dict:
    """Fetch + cache one address. Returns the cached row.
    If force=False and a cached row exists, returns it without network call.
    """
    address = address.lower()
    if not force:
        existing = conn.execute(
            "SELECT * FROM oli_labels WHERE address = ? AND chain_id = ?",
            (address, chain_id),
        ).fetchone()
        if existing is not None:
            return dict(existing)
    metas = _fetch_metadata_batch([address], chain_id=chain_id)
    meta = metas.get(address) or {}
    row = _upsert_label(conn, address, chain_id, meta)
    conn.commit()
    return row


def enrich_batch(conn: sqlite3.Connection, addresses: list[str],
                 chain_id: int = DEFAULT_CHAIN_ID, force: bool = False) -> dict[str, dict]:
    """Fetch + cache many addresses. Returns {addr: row}.
    Skips addresses already cached unless force=True.
    """
    addresses = [a.lower() for a in addresses]
    addresses = list(dict.fromkeys(addresses))  # de-dup, preserve order

    if not force:
        cached = {
            r["address"]: dict(r)
            for r in conn.execute(
                f"SELECT * FROM oli_labels WHERE chain_id = ? AND address IN "
                f"({','.join('?' * len(addresses))})",
                (chain_id, *addresses),
            ).fetchall()
        }
        to_fetch = [a for a in addresses if a not in cached]
    else:
        cached = {}
        to_fetch = addresses

    print(f"  cache hits: {len(cached)}, to fetch: {len(to_fetch)}", file=sys.stderr)

    fetched: dict[str, dict] = {}
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i : i + BATCH_SIZE]
        print(f"  batch {i // BATCH_SIZE + 1}: {len(batch)} addrs", file=sys.stderr)
        metas = _fetch_metadata_batch(batch, chain_id=chain_id)
        for addr in batch:
            row = _upsert_label(conn, addr, chain_id, metas.get(addr) or {})
            fetched[addr] = row
        conn.commit()
        time.sleep(RATE_SLEEP_S)

    return {**cached, **fetched}


# ─────────────────────────────────────────────────────────────────────
# Source queries — addresses we should enrich
# ─────────────────────────────────────────────────────────────────────
def watchlist_addresses(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT LOWER(address) FROM watchlist WHERE active = 1"
    ).fetchall()
    return [r[0] for r in rows]


def flagged_addresses(conn: sqlite3.Connection) -> list[str]:
    """All addresses we've flagged as malicious — same query as the audit script."""
    rows = conn.execute("""
        SELECT DISTINCT LOWER(address) FROM (
            SELECT address FROM watchlist WHERE active = 1
            UNION
            SELECT address FROM entity_classification
              WHERE category = 'CRIMINAL'
                 OR subtype IN (
                     'known_attacker','trap_contract','trap_deployer','trap_inventory',
                     'infrastructure_parasite','mev_factory','bot_operator','rd_bot',
                     'org_laundry','mixer','org_cashout',
                     'org_001_shadow_cex_exit','org_001_shadow_lp_staging'
                 )
            UNION
            SELECT funder_address AS address FROM infrastructure_operator_candidates
        )
    """).fetchall()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def _open_conn(write: bool = True) -> sqlite3.Connection:
    if write:
        conn = sqlite3.connect(str(DB_PATH))
    else:
        conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ensure_table(conn) if write else None
    return conn


def _print_hits(conn: sqlite3.Connection, severity: str = "HIGH") -> int:
    rows = conn.execute("""
        SELECT l.address, l.severity, l.primary_entity, l.primary_tag_name,
               l.tag_count, l.fetched_at,
               (SELECT priority FROM watchlist w WHERE LOWER(w.address) = l.address LIMIT 1) AS wl_priority,
               (SELECT entity_name FROM watchlist w WHERE LOWER(w.address) = l.address LIMIT 1) AS wl_label
        FROM oli_labels l
        WHERE l.severity = ?
        ORDER BY l.primary_entity, l.address
    """, (severity,)).fetchall()
    print(f"\n=== oli_labels severity={severity}: {len(rows)} rows ===")
    for r in rows:
        wl = f"WL:{r['wl_priority']} {r['wl_label']}" if r["wl_priority"] else "(not in watchlist)"
        print(f"  {r['address']} | {r['primary_entity'] or '-':<25} | {r['primary_tag_name'] or '-':<35} | {wl}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--address", help="Lookup/enrich a single address")
    ap.add_argument("--chain-id", type=int, default=DEFAULT_CHAIN_ID,
                    help="Chain ID for metadata service (default 1=mainnet)")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if cached")
    ap.add_argument("--backfill-watchlist", action="store_true",
                    help="Enrich all active watchlist addresses")
    ap.add_argument("--backfill-flagged", action="store_true",
                    help="Enrich all malicious-flagged addresses (watchlist + classification + ISO candidates)")
    ap.add_argument("--hits", action="store_true",
                    help="Print HIGH-severity entries from cache (no fetch)")
    ap.add_argument("--low-hits", action="store_true",
                    help="Also print LOW-severity entries from cache")
    args = ap.parse_args()

    conn = _open_conn(write=True)

    if args.address:
        row = enrich_address(conn, args.address, chain_id=args.chain_id, force=args.force)
        print(json.dumps(row, indent=2, default=str))
        return 0

    if args.backfill_watchlist:
        addrs = watchlist_addresses(conn)
        print(f"Backfilling {len(addrs)} watchlist addresses...", file=sys.stderr)
        enrich_batch(conn, addrs, chain_id=args.chain_id, force=args.force)
        _print_hits(conn, "HIGH")
        if args.low_hits:
            _print_hits(conn, "LOW")
        return 0

    if args.backfill_flagged:
        addrs = flagged_addresses(conn)
        print(f"Backfilling {len(addrs)} flagged addresses...", file=sys.stderr)
        enrich_batch(conn, addrs, chain_id=args.chain_id, force=args.force)
        _print_hits(conn, "HIGH")
        if args.low_hits:
            _print_hits(conn, "LOW")
        return 0

    if args.hits:
        _print_hits(conn, "HIGH")
        if args.low_hits:
            _print_hits(conn, "LOW")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
