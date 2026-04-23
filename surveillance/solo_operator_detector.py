"""Solo-operator + instant-activation detectors.

Complements surveillance.org_candidates (which requires 3+ deployers
sharing a funder). Catches two classes the cluster detector misses:

  (A) SOLO high-productivity operator:
      - one deployer, fleet >= MIN_FLEET (10)
      - confirmed_count >= MIN_CONFIRMED (3)  OR  trap_events >= MIN_TRAPS (3)
      - emits with classification = 'solo_high_productivity'

  (B) INSTANT-activation operator:
      - deployer first_seen within INSTANT_WINDOW_DAYS (7)
      - confirmed_count >= 1 AND trap_events >= 1
      - emits with classification = 'instant_activation'
      - Same-day deploy+fire is the tell; prep phase absent

Writes candidates into a new table `solo_operator_candidates` with the
same review-workflow shape as org_candidates (pending/promoted/dismissed).
Idempotent — re-runs update existing rows.

CLI:
    python -m surveillance.solo_operator_detector --dry-run
    python -m surveillance.solo_operator_detector --apply
"""
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

MIN_FLEET = 10
MIN_CONFIRMED = 3
MIN_TRAPS = 3
INSTANT_WINDOW_DAYS = 7
MAX_SOLO_FLEET = 500  # Skip CEX-scale funders / infrastructure


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS solo_operator_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deployer_address TEXT NOT NULL UNIQUE,
            classification TEXT NOT NULL,
            chain TEXT,
            fleet_size INTEGER NOT NULL,
            confirmed_count INTEGER NOT NULL,
            trap_count INTEGER NOT NULL,
            first_seen TEXT,
            mainnet_first_tx TEXT,
            funder TEXT,
            detected_at TEXT NOT NULL,
            last_checked TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_solo_classification ON solo_operator_candidates(classification);
        CREATE INDEX IF NOT EXISTS idx_solo_status ON solo_operator_candidates(status);
    """)
    conn.commit()


def find_solo_high_productivity(conn: sqlite3.Connection) -> list[dict]:
    """Class A: fleet >= MIN_FLEET AND (confirmed >= MIN_CONFIRMED OR traps >= MIN_TRAPS).
    Skip deployers already in org_wallets (they're classified) or in
    org_candidates (they're already cluster-tracked)."""
    # Known addresses we skip
    known_org = {r[0].lower() for r in conn.execute(
        "SELECT address FROM org_wallets").fetchall()}
    # Deployers in any cluster
    cluster_deployers = set()
    for r in conn.execute("SELECT deployer_addresses FROM org_candidates").fetchall():
        try:
            import json
            for dep in json.loads(r[0]):
                cluster_deployers.add(dep.lower())
        except Exception:
            pass
    skip = known_org | cluster_deployers

    rows = conn.execute(f"""
        SELECT c.deployer_address,
               (SELECT chain FROM deployers d WHERE d.deployer_address = c.deployer_address LIMIT 1) AS chain,
               COUNT(*) AS fleet,
               SUM(CASE WHEN confidence_tier = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
               (SELECT COUNT(*) FROM trap_events te JOIN contracts c2
                  ON LOWER(c2.contract_address) = LOWER(te.trap_contract_address)
                  WHERE c2.deployer_address = c.deployer_address) AS traps,
               (SELECT first_seen FROM deployers d WHERE d.deployer_address = c.deployer_address) AS first_seen,
               (SELECT mainnet_first_tx FROM deployers d WHERE d.deployer_address = c.deployer_address) AS mainnet_tx,
               (SELECT json_extract(funding_trail, '$.funder') FROM deployers d
                  WHERE d.deployer_address = c.deployer_address) AS funder
        FROM contracts c
        GROUP BY c.deployer_address
        HAVING fleet >= {MIN_FLEET} AND fleet <= {MAX_SOLO_FLEET}
           AND (confirmed >= {MIN_CONFIRMED} OR traps >= {MIN_TRAPS})
    """).fetchall()

    results = []
    for r in rows:
        addr = r[0].lower()
        if addr in skip:
            continue
        results.append({
            "deployer_address": addr,
            "classification": "solo_high_productivity",
            "chain": r[1],
            "fleet_size": r[2],
            "confirmed_count": r[3],
            "trap_count": r[4],
            "first_seen": r[5],
            "mainnet_first_tx": r[6],
            "funder": r[7],
        })
    return results


def find_instant_activation(conn: sqlite3.Connection) -> list[dict]:
    """Class B: deployer within 7 days of first_seen AND has confirmed + trap."""
    known_org = {r[0].lower() for r in conn.execute(
        "SELECT address FROM org_wallets").fetchall()}
    rows = conn.execute(f"""
        SELECT c.deployer_address,
               (SELECT chain FROM deployers d WHERE d.deployer_address = c.deployer_address LIMIT 1) AS chain,
               COUNT(*) AS fleet,
               SUM(CASE WHEN confidence_tier = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
               (SELECT COUNT(*) FROM trap_events te JOIN contracts c2
                  ON LOWER(c2.contract_address) = LOWER(te.trap_contract_address)
                  WHERE c2.deployer_address = c.deployer_address) AS traps,
               (SELECT first_seen FROM deployers d WHERE d.deployer_address = c.deployer_address) AS first_seen,
               (SELECT mainnet_first_tx FROM deployers d WHERE d.deployer_address = c.deployer_address) AS mainnet_tx,
               (SELECT json_extract(funding_trail, '$.funder') FROM deployers d
                  WHERE d.deployer_address = c.deployer_address) AS funder
        FROM contracts c
        GROUP BY c.deployer_address
        HAVING confirmed >= 1 AND traps >= 1
           AND first_seen > datetime('now', '-{INSTANT_WINDOW_DAYS} days')
    """).fetchall()

    results = []
    for r in rows:
        addr = r[0].lower()
        if addr in known_org:
            continue
        results.append({
            "deployer_address": addr,
            "classification": "instant_activation",
            "chain": r[1],
            "fleet_size": r[2],
            "confirmed_count": r[3],
            "trap_count": r[4],
            "first_seen": r[5],
            "mainnet_first_tx": r[6],
            "funder": r[7],
        })
    return results


def apply_candidates(conn: sqlite3.Connection, rows: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    refreshed = 0
    for r in rows:
        existing = conn.execute(
            "SELECT id FROM solo_operator_candidates WHERE deployer_address = ?",
            (r["deployer_address"],),
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE solo_operator_candidates SET
                    classification = ?, chain = ?, fleet_size = ?,
                    confirmed_count = ?, trap_count = ?, first_seen = ?,
                    mainnet_first_tx = ?, funder = ?, last_checked = ?
                WHERE deployer_address = ?
            """, (r["classification"], r["chain"], r["fleet_size"],
                  r["confirmed_count"], r["trap_count"], r["first_seen"],
                  r["mainnet_first_tx"], r["funder"], now, r["deployer_address"]))
            refreshed += 1
        else:
            conn.execute("""
                INSERT INTO solo_operator_candidates
                    (deployer_address, classification, chain, fleet_size,
                     confirmed_count, trap_count, first_seen, mainnet_first_tx,
                     funder, detected_at, last_checked, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (r["deployer_address"], r["classification"], r["chain"],
                  r["fleet_size"], r["confirmed_count"], r["trap_count"],
                  r["first_seen"], r["mainnet_first_tx"], r["funder"], now, now))
            inserted += 1
    conn.commit()
    return {"inserted": inserted, "refreshed": refreshed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", type=str, default=str(DB_PATH))
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("Pass --dry-run or --apply.")

    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row
    if args.apply:
        ensure_table(conn)

    solo = find_solo_high_productivity(conn)
    instant = find_instant_activation(conn)

    print(f"[solo_operator_detector] solo_high_productivity: {len(solo)}")
    print(f"[solo_operator_detector] instant_activation:     {len(instant)}")

    if args.dry_run:
        print("\ntop 10 solo_high_productivity (by confirmed desc):")
        for r in sorted(solo, key=lambda x: -x["confirmed_count"])[:10]:
            print(f"  {r['deployer_address']}  fleet={r['fleet_size']:<4} "
                  f"confirmed={r['confirmed_count']:<3} traps={r['trap_count']:<3} "
                  f"chain={r['chain']}")
        print("\ntop 10 instant_activation (by confirmed desc):")
        for r in sorted(instant, key=lambda x: -x["confirmed_count"])[:10]:
            print(f"  {r['deployer_address']}  fleet={r['fleet_size']:<4} "
                  f"confirmed={r['confirmed_count']:<3} traps={r['trap_count']:<3} "
                  f"chain={r['chain']}  first={r['first_seen'][:10]}")
        return

    all_candidates = solo + instant
    res = apply_candidates(conn, all_candidates)
    print(f"[solo_operator_detector] applied: inserted={res['inserted']} refreshed={res['refreshed']}")
    conn.close()


if __name__ == "__main__":
    main()
