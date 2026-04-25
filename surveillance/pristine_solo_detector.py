"""Pristine-solo detector — third-tier coverage for the 0x752c5a95 class.

Closes the gap left by org_candidates (needs 3+ deployers/funder) and
solo_operator_detector (needs fleet >= 10).

Pattern: small Base/Arb/OP footprint riding on long mainnet vintage.
Behavioral-laundering signature — operator deploys 1-5 contracts using a
years-old mainnet EOA so fresh-wallet heuristics don't fire and per-contract
stored potential is concentrated.

Rule:
  - total_contracts_deployed BETWEEN 1 AND MAX_FLEET (5)
  - mainnet_first_tx populated AND (first_seen - mainnet_first_tx) > MIN_MAINNET_GAP_DAYS (365)
  - confirmed_count >= 1  (at least one observed-harm contract)
  - Skip deployers already in org_wallets / org_candidates / solo_operator_candidates

Writes to `pristine_solo_candidates` with pending/promoted/dismissed workflow,
matching org_candidates and solo_operator_candidates shapes.

CLI:
    python -m surveillance.pristine_solo_detector --dry-run
    python -m surveillance.pristine_solo_detector --apply
"""
import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

MAX_FLEET = 5
MIN_MAINNET_GAP_DAYS = 365


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pristine_solo_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deployer_address TEXT NOT NULL UNIQUE,
            chain TEXT,
            fleet_size INTEGER NOT NULL,
            confirmed_count INTEGER NOT NULL,
            suspected_count INTEGER NOT NULL,
            trap_count INTEGER NOT NULL,
            first_seen TEXT,
            mainnet_first_tx TEXT,
            mainnet_gap_days INTEGER,
            funder TEXT,
            detected_at TEXT NOT NULL,
            last_checked TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pristine_status ON pristine_solo_candidates(status);
        CREATE INDEX IF NOT EXISTS idx_pristine_gap ON pristine_solo_candidates(mainnet_gap_days);
    """)
    conn.commit()


def _build_skip_set(conn: sqlite3.Connection) -> set[str]:
    """Deployers already classified or candidate-tracked elsewhere."""
    skip = set()
    for r in conn.execute("SELECT address FROM org_wallets").fetchall():
        skip.add(r[0].lower())
    for r in conn.execute("SELECT deployer_addresses FROM org_candidates").fetchall():
        try:
            for dep in json.loads(r[0]):
                skip.add(dep.lower())
        except Exception:
            pass
    try:
        for r in conn.execute("SELECT deployer_address FROM solo_operator_candidates").fetchall():
            skip.add(r[0].lower())
    except sqlite3.OperationalError:
        # Table not present (early run) — fine, just skip the filter
        pass
    return skip


def find_pristine_solo(conn: sqlite3.Connection) -> list[dict]:
    """Find pristine-solo deployers per the rule above."""
    skip = _build_skip_set(conn)

    # Pull candidates: small fleet + old mainnet history + confirmed >= 1
    rows = conn.execute(f"""
        SELECT d.deployer_address,
               d.chain,
               d.first_seen,
               d.mainnet_first_tx,
               d.total_contracts_deployed AS fleet,
               json_extract(d.funding_trail, '$.funder') AS funder,
               (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
                  AND confidence_tier = 'confirmed') AS confirmed,
               (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
                  AND confidence_tier = 'suspected') AS suspected,
               (SELECT COUNT(*) FROM trap_events te JOIN contracts c2
                  ON LOWER(c2.contract_address) = LOWER(te.trap_contract_address)
                  WHERE c2.deployer_address = d.deployer_address) AS traps,
               CAST(julianday(d.first_seen) - julianday(d.mainnet_first_tx) AS INTEGER) AS gap_days
        FROM deployers d
        WHERE d.total_contracts_deployed BETWEEN 1 AND {MAX_FLEET}
          AND d.mainnet_first_tx IS NOT NULL AND d.mainnet_first_tx != ''
          AND CAST(julianday(d.first_seen) - julianday(d.mainnet_first_tx) AS INTEGER) > {MIN_MAINNET_GAP_DAYS}
          AND (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
                 AND confidence_tier = 'confirmed') >= 1
    """).fetchall()

    results = []
    for r in rows:
        addr = r[0].lower()
        if addr in skip:
            continue
        results.append({
            "deployer_address": addr,
            "chain": r[1],
            "first_seen": r[2],
            "mainnet_first_tx": r[3],
            "fleet_size": r[4],
            "funder": r[5],
            "confirmed_count": r[6],
            "suspected_count": r[7],
            "trap_count": r[8],
            "mainnet_gap_days": r[9],
        })
    return results


def apply_candidates(conn: sqlite3.Connection, rows: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    refreshed = 0
    for r in rows:
        existing = conn.execute(
            "SELECT id FROM pristine_solo_candidates WHERE deployer_address = ?",
            (r["deployer_address"],),
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE pristine_solo_candidates SET
                    chain = ?, fleet_size = ?,
                    confirmed_count = ?, suspected_count = ?, trap_count = ?,
                    first_seen = ?, mainnet_first_tx = ?, mainnet_gap_days = ?,
                    funder = ?, last_checked = ?
                WHERE deployer_address = ?
            """, (r["chain"], r["fleet_size"],
                  r["confirmed_count"], r["suspected_count"], r["trap_count"],
                  r["first_seen"], r["mainnet_first_tx"], r["mainnet_gap_days"],
                  r["funder"], now, r["deployer_address"]))
            refreshed += 1
        else:
            conn.execute("""
                INSERT INTO pristine_solo_candidates
                    (deployer_address, chain, fleet_size,
                     confirmed_count, suspected_count, trap_count,
                     first_seen, mainnet_first_tx, mainnet_gap_days,
                     funder, detected_at, last_checked, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (r["deployer_address"], r["chain"], r["fleet_size"],
                  r["confirmed_count"], r["suspected_count"], r["trap_count"],
                  r["first_seen"], r["mainnet_first_tx"], r["mainnet_gap_days"],
                  r["funder"], now, now))
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

    rows = find_pristine_solo(conn)
    print(f"[pristine_solo_detector] candidates found: {len(rows)}")

    if args.dry_run:
        print("\nTop 20 by mainnet_gap_days desc:")
        for r in sorted(rows, key=lambda x: -x["mainnet_gap_days"])[:20]:
            print(f"  {r['deployer_address']}  fleet={r['fleet_size']:<2} "
                  f"conf={r['confirmed_count']:<2} sus={r['suspected_count']:<2} "
                  f"traps={r['trap_count']:<2} chain={r['chain']:<9} "
                  f"L2={r['first_seen'][:10]}  mn={r['mainnet_first_tx'][:10]}  "
                  f"gap={r['mainnet_gap_days']}d")
        return

    res = apply_candidates(conn, rows)
    print(f"[pristine_solo_detector] applied: inserted={res['inserted']} refreshed={res['refreshed']}")
    conn.close()


if __name__ == "__main__":
    main()
