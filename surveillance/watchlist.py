"""
Layer 3 — Watchlist Tracking: Zero-API Entity Monitoring

Tracks known entities (deployers, funders, gas fingerprints) through
the existing deployment monitor pipeline. Zero additional API calls —
all detection via SQLite queries on data that already flows in.

Usage:
    python -m surveillance.watchlist --list
    python -m surveillance.watchlist --add 0xADDR --type deployer --entity name --reason why
    python -m surveillance.watchlist --hits --since 24h
    python -m surveillance.watchlist --entity architect
    python -m surveillance.watchlist --deactivate 0xADDR
    python -m surveillance.watchlist --seed
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            address_type TEXT NOT NULL,
            entity_name TEXT,
            watch_reason TEXT,
            priority TEXT DEFAULT 'HIGH',
            added_date TEXT,
            last_seen_date TEXT,
            hit_count INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            UNIQUE(address, address_type)
        );

        CREATE TABLE IF NOT EXISTS watchlist_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watchlist_id INTEGER NOT NULL,
            hit_type TEXT,
            contract_address TEXT,
            chain TEXT,
            timestamp TEXT,
            details TEXT,
            alerted INTEGER DEFAULT 0,
            FOREIGN KEY (watchlist_id) REFERENCES watchlist(id)
        );

        CREATE INDEX IF NOT EXISTS idx_watchlist_active ON watchlist(active);
        CREATE INDEX IF NOT EXISTS idx_watchlist_entity ON watchlist(entity_name);
        CREATE INDEX IF NOT EXISTS idx_whits_alerted ON watchlist_hits(alerted);
    """)


# Known gas fingerprints — distinctive enough to identify operators
GAS_FINGERPRINTS = {
    0.020012: "architect",
}


def add_watch(conn: sqlite3.Connection, address: str, address_type: str,
              entity_name: str, reason: str, priority: str = "HIGH"):
    ensure_tables(conn)
    conn.execute("""
        INSERT OR REPLACE INTO watchlist
        (address, address_type, entity_name, watch_reason, priority, added_date, active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    """, (address.lower(), address_type, entity_name, reason, priority, _now()))
    conn.commit()


def check_watchlist(conn: sqlite3.Connection, deployer: str, chain: str,
                    contract_address: str, timestamp: str,
                    funder: str = None, gas_price: float = None) -> list:
    """
    Check a newly ingested contract against the watchlist.
    Called AFTER normal processing. Zero API calls.
    """
    hits = []

    # Check 1: Deployer match
    for r in conn.execute("""
        SELECT id, address, entity_name, priority, watch_reason
        FROM watchlist WHERE active = 1 AND address_type = 'deployer'
        AND ? LIKE address || '%'
    """, (deployer.lower(),)):
        hits.append({
            "watchlist_id": r["id"], "hit_type": "deployment",
            "entity": r["entity_name"], "priority": r["priority"],
            "reason": r["watch_reason"], "chain": chain,
        })

    # Check 2: Funder match
    if funder:
        for r in conn.execute("""
            SELECT id, address, entity_name, priority, watch_reason
            FROM watchlist WHERE active = 1 AND address_type = 'funder'
            AND ? LIKE address || '%'
        """, (funder.lower(),)):
            hits.append({
                "watchlist_id": r["id"], "hit_type": "funding",
                "entity": r["entity_name"], "priority": r["priority"],
                "reason": r["watch_reason"], "chain": chain,
            })

    # Check 3: Gas fingerprint
    if gas_price is not None:
        for gas_val, entity in GAS_FINGERPRINTS.items():
            if abs(float(gas_price) - gas_val) < 0.000001:
                r = conn.execute(
                    "SELECT id FROM watchlist WHERE entity_name = ? AND active = 1 LIMIT 1",
                    (entity,)
                ).fetchone()
                if r:
                    hits.append({
                        "watchlist_id": r["id"], "hit_type": "gas_fingerprint",
                        "entity": entity, "priority": "CRITICAL",
                        "reason": f"Gas fingerprint match: {gas_price} gwei", "chain": chain,
                    })

    # Record hits
    for hit in hits:
        conn.execute("""
            INSERT INTO watchlist_hits
            (watchlist_id, hit_type, contract_address, chain, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (hit["watchlist_id"], hit["hit_type"], contract_address,
              chain, timestamp, json.dumps(hit)))
        conn.execute("""
            UPDATE watchlist SET last_seen_date = ?, hit_count = hit_count + 1 WHERE id = ?
        """, (timestamp, hit["watchlist_id"]))

    if hits:
        conn.commit()

    return hits


def seed_watchlist(conn: sqlite3.Connection):
    """Populate watchlist with known entities."""
    ensure_tables(conn)

    entries = [
        # The Architect
        ("0x9209c9f7dcb61937f1ec8160c22c0b2365079474", "deployer", "architect",
         "Most sophisticated operator in corpus. 21 R&D contracts on Arbitrum. Full weapon: SD+DC+TS+CL.", "CRITICAL"),
        ("0x151b381058f91cf871e7ea1ee83c45326f61e96d", "funder", "architect",
         "Architect's sole funder. 0.0508 ETH. Single deployer.", "CRITICAL"),

        # Org_001 whale path
        ("0xf70da97812cb96acdf810712aa562db8dfa3dbef", "funder", "org_001_whale",
         "Org_001 whale trader. 68% of deployments now through this path. Binance origin.", "HIGH"),
        ("0x8c826f795466e39acbff1bb4eeeb759609377ba1", "funder", "org_001_gas_station",
         "Org_001 gas station. 173 deployers funded. Coinbase origin.", "HIGH"),

        # Independent gas station
        ("0xbaed383ede0e5d9d72430661f3285daa77e9439f", "funder", "potential_org_004",
         "Independent gas station. 12+ deployers. Mixed trap types.", "HIGH"),

        # Approval trap operator (390 armed)
        ("0x31eadfd9b3c43f522e3192a6c5fa28ed126abb61", "deployer", "approval_trap_0x4a3e",
         "390 armed approvals on 0x4a3e2069. SELFDESTRUCT. Waiting for drain.", "CRITICAL"),
        ("0x987b2308c8ca6f906b797386954e486001f4b3d1", "funder", "approval_trap_funder",
         "Funded 3 deployers iterating approval traps. Escalating capability.", "HIGH"),

        # R&D bot-to-deployer (0x01a16e04)
        ("0x01a16e04e457b82bdb9ee94d102b3e25e01d26d7", "deployer", "rd_bot_deployer",
         "Bot-to-deployer conversion. 8 contracts, 7 iterations. Self-testing at 5,494 hits.", "HIGH"),

        # 0xc0ffeefeed dual operator
        ("0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e", "deployer", "coffee_dual_operator",
         "Dual operator: 53+ contracts deployed + 83 vanity bot wallets.", "MEDIUM"),

        # Org_002 treasuries
        ("0x238d7170f309a55b87a144a341bd6105897082ca", "funder", "org_002_senior",
         "Org_002 senior treasury. 85 deployers. Untraceable origin.", "HIGH"),
        ("0xde8eb937cb5475eee5ac96dce6ba2d18e439c473", "funder", "org_002_junior",
         "Org_002 junior treasury. 88 deployers. Untraceable origin.", "HIGH"),
    ]

    # Add Architect's behavioral matches
    try:
        for r in conn.execute("""
            SELECT deployer_a, deployer_b, composite_score FROM deployer_similarity
            WHERE (deployer_a = '0x9209c9f7dcb61937f1ec8160c22c0b2365079474'
                OR deployer_b = '0x9209c9f7dcb61937f1ec8160c22c0b2365079474')
            AND composite_score >= 0.70
        """):
            other = r["deployer_b"] if r["deployer_a"] == "0x9209c9f7dcb61937f1ec8160c22c0b2365079474" else r["deployer_a"]
            entries.append((
                other, "deployer", "architect_associated",
                f"Behavioral match to Architect at {r['composite_score']:.3f}. Possible alternate wallet.", "HIGH"
            ))
    except Exception:
        pass

    added = 0
    for address, atype, entity, reason, priority in entries:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO watchlist
                (address, address_type, entity_name, watch_reason, priority, added_date, active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (address.lower(), atype, entity, reason, priority, _now()))
            added += 1
        except Exception:
            pass

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM watchlist WHERE active=1").fetchone()[0]
    print(f"[watchlist] Seeded {added} entries. Total active: {total}")


def backfill_hits(conn: sqlite3.Connection):
    """Check existing contracts against watchlist for historical hits."""
    ensure_tables(conn)
    hits = 0

    # Check deployer matches
    for w in conn.execute("SELECT id, address, entity_name FROM watchlist WHERE active=1 AND address_type='deployer'"):
        for c in conn.execute("""
            SELECT contract_address, chain, detection_timestamp FROM contracts
            WHERE deployer_address LIKE ? || '%'
        """, (w["address"],)):
            existing = conn.execute(
                "SELECT 1 FROM watchlist_hits WHERE watchlist_id=? AND contract_address=?",
                (w["id"], c["contract_address"])
            ).fetchone()
            if not existing:
                conn.execute("""
                    INSERT INTO watchlist_hits (watchlist_id, hit_type, contract_address, chain, timestamp, details)
                    VALUES (?, 'deployment', ?, ?, ?, ?)
                """, (w["id"], c["contract_address"], c["chain"], c["detection_timestamp"],
                      json.dumps({"entity": w["entity_name"], "backfill": True})))
                hits += 1

    # Check funder matches
    for w in conn.execute("SELECT id, address, entity_name FROM watchlist WHERE active=1 AND address_type='funder'"):
        for d in conn.execute("""
            SELECT deployer_address FROM deployers WHERE funding_trail LIKE ?
        """, (f'%{w["address"][:16]}%',)):
            for c in conn.execute("SELECT contract_address, chain, detection_timestamp FROM contracts WHERE deployer_address=?",
                                  (d["deployer_address"],)):
                existing = conn.execute(
                    "SELECT 1 FROM watchlist_hits WHERE watchlist_id=? AND contract_address=?",
                    (w["id"], c["contract_address"])
                ).fetchone()
                if not existing:
                    conn.execute("""
                        INSERT INTO watchlist_hits (watchlist_id, hit_type, contract_address, chain, timestamp, details)
                        VALUES (?, 'funding', ?, ?, ?, ?)
                    """, (w["id"], c["contract_address"], c["chain"], c["detection_timestamp"],
                          json.dumps({"entity": w["entity_name"], "deployer": d["deployer_address"], "backfill": True})))
                    hits += 1

    conn.commit()
    print(f"[watchlist] Backfilled {hits} historical hits")


def print_list(conn: sqlite3.Connection):
    ensure_tables(conn)
    print(f"{'Address':44} | {'Type':8} | {'Entity':25} | {'Pri':8} | {'Hits':>4} | {'Last Seen':16}")
    print("-" * 120)
    for r in conn.execute("""
        SELECT address, address_type, entity_name, priority, hit_count, last_seen_date
        FROM watchlist WHERE active=1
        ORDER BY CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END, entity_name
    """):
        last = r["last_seen_date"][:16] if r["last_seen_date"] else "never"
        print(f"  {r['address'][:42]} | {r['address_type']:8} | {r['entity_name']:25} | {r['priority']:8} | {r['hit_count']:>4} | {last}")


def print_hits(conn: sqlite3.Connection, since_hours: int = 24):
    ensure_tables(conn)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    hits = conn.execute("""
        SELECT wh.hit_type, wh.contract_address, wh.chain, wh.timestamp,
               w.entity_name, w.priority, w.watch_reason
        FROM watchlist_hits wh
        JOIN watchlist w ON wh.watchlist_id = w.id
        WHERE wh.timestamp >= ?
        ORDER BY wh.timestamp DESC
    """, (cutoff,)).fetchall()

    if not hits:
        print(f"No watchlist hits in the last {since_hours}h")
        return

    print(f"Watchlist hits (last {since_hours}h): {len(hits)}")
    for h in hits[:20]:
        print(f"  [{h['priority']}] {h['entity_name']} | {h['hit_type']} | {h['chain']} | {h['contract_address'][:18]}... | {h['timestamp'][:16]}")


def print_entity(conn: sqlite3.Connection, entity: str):
    ensure_tables(conn)
    watches = conn.execute("SELECT * FROM watchlist WHERE entity_name LIKE ? AND active=1", (f"%{entity}%",)).fetchall()
    if not watches:
        print(f"No watchlist entries matching '{entity}'")
        return

    for w in watches:
        print(f"\n{w['entity_name']} ({w['priority']})")
        print(f"  Address: {w['address']} ({w['address_type']})")
        print(f"  Reason: {w['watch_reason']}")
        print(f"  Hits: {w['hit_count']} | Last: {w['last_seen_date'] or 'never'}")

        hits = conn.execute("""
            SELECT hit_type, contract_address, chain, timestamp
            FROM watchlist_hits WHERE watchlist_id=?
            ORDER BY timestamp DESC LIMIT 10
        """, (w["id"],)).fetchall()
        if hits:
            print(f"  Recent hits:")
            for h in hits:
                print(f"    {h['timestamp'][:16]} | {h['hit_type']} | {h['chain']} | {h['contract_address'][:18]}...")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watchlist management")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--add", type=str, help="Address to watch")
    parser.add_argument("--type", type=str, choices=["deployer", "funder", "contract", "bot"])
    parser.add_argument("--entity", type=str)
    parser.add_argument("--reason", type=str)
    parser.add_argument("--priority", type=str, default="HIGH")
    parser.add_argument("--hits", action="store_true")
    parser.add_argument("--since", type=str, default="24h")
    parser.add_argument("--deactivate", type=str)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    ensure_tables(conn)

    if args.seed:
        seed_watchlist(conn)
    elif args.backfill:
        backfill_hits(conn)
    elif args.list:
        print_list(conn)
    elif args.hits:
        hours = int(args.since.replace("h", ""))
        print_hits(conn, hours)
    elif args.entity:
        print_entity(conn, args.entity)
    elif args.add:
        if not args.type or not args.entity or not args.reason:
            print("Required: --type, --entity, --reason")
        else:
            add_watch(conn, args.add, args.type, args.entity, args.reason, args.priority)
            print(f"Added {args.add} to watchlist as {args.entity}")
    elif args.deactivate:
        conn.execute("UPDATE watchlist SET active=0 WHERE address LIKE ?", (args.deactivate.lower() + "%",))
        conn.commit()
        print(f"Deactivated {args.deactivate}")
    else:
        parser.print_help()

    conn.close()
