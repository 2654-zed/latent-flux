"""Infrastructure-scale operator detector.

Closes the gap surfaced 2026-04-25: org_candidates excludes funders > 50
deployers as 'CEX/faucet noise.' That hypothesis was falsified — the
0xf70da978 cluster (2,684 deployers, 109 confirmed traps) and 11 peers
all looked like CEX/faucet noise to that detector.

This detector targets a distinct threat class: operators who deliberately
fan out to thousands of deployer wallets to defeat clustering heuristics.
Their downstream pattern differs from legitimate faucets/CEXes:

  Faucet/CEX                 Infrastructure operator
  -------------------------- -------------------------------
  Deployers do many things   Deployers are disposable (fleet ≤ 2)
  Contracts mostly unknown   Contracts heavy in suspected/confirmed tier
  Real-traffic on contracts  Mostly dormant / bot-probe only
  No sweep behavior          Drains, traps, MEV-bait

Detection signals (must satisfy all):
  S1. Fanout count >= MIN_FANOUT (default 200)
  S2. Suspected-or-confirmed contract ratio >= MIN_ADVERSARIAL_RATIO (0.10)
  S3. Disposable deployer rate >= MIN_DISPOSABLE_RATE (0.5)

Distinct review workflow from org_candidates: promotion target is a new
`infrastructure_operators` entity class (TBD), NOT org_wallets. We hold
the entity class spec separately and only emit candidates here.

CLI:
    python -m surveillance.infrastructure_operator_detector --dry-run
    python -m surveillance.infrastructure_operator_detector --apply
"""
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

MIN_FANOUT = 200                  # below this = small-cell, handled by org_candidates
MIN_ADVERSARIAL_RATIO = 0.10      # >= 10% of contracts in suspected+confirmed
MIN_DISPOSABLE_RATE = 0.50        # >= 50% of funded deployers have fleet <= 2


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS infrastructure_operator_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            funder_address TEXT NOT NULL UNIQUE,
            chain_signature TEXT,            -- chains this funder seeded into
            deployer_count INTEGER NOT NULL,
            contract_count INTEGER NOT NULL,
            confirmed_count INTEGER NOT NULL,
            suspected_count INTEGER NOT NULL,
            adversarial_ratio REAL NOT NULL,
            disposable_rate REAL NOT NULL,
            avg_fleet_per_deployer REAL,
            funder_first_seen_in_corpus TEXT,
            funder_last_seen_in_corpus TEXT,
            funder_mainnet_first_tx TEXT,    -- null if L2-only
            funder_known_in_deployers INTEGER,  -- 1 if funder itself is a deployer
            detected_at TEXT NOT NULL,
            last_checked TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_infra_status ON infrastructure_operator_candidates(status);
        CREATE INDEX IF NOT EXISTS idx_infra_fanout ON infrastructure_operator_candidates(deployer_count);
        CREATE INDEX IF NOT EXISTS idx_infra_adv ON infrastructure_operator_candidates(adversarial_ratio);
    """)
    conn.commit()


def find_infrastructure_operators(conn: sqlite3.Connection) -> list[dict]:
    """Find funders with fanout >= MIN_FANOUT and adversarial signature."""
    # Step 1: candidate funders (those with large fanouts)
    candidates = conn.execute(f"""
        SELECT LOWER(json_extract(funding_trail, '$.funder')) AS funder,
               COUNT(*) AS n_deployers
        FROM deployers
        WHERE json_extract(funding_trail, '$.funder') IS NOT NULL
        GROUP BY funder
        HAVING n_deployers >= {MIN_FANOUT}
    """).fetchall()

    results = []
    for funder, n_deployers in candidates:
        # Compute the funder's downstream stats
        deployers = [r[0] for r in conn.execute(
            "SELECT LOWER(deployer_address) FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?",
            (funder,)
        )]
        if not deployers:
            continue

        # Disposable rate: fraction with fleet <= 2
        disposable = conn.execute(f"""
            SELECT COUNT(*) FROM deployers
            WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
              AND total_contracts_deployed <= 2
        """, (funder,)).fetchone()[0]
        disposable_rate = disposable / n_deployers

        # Contract counts via single batch (chunked to dodge SQL var limits)
        chunk = 500
        contract_count = confirmed = suspected = 0
        chains = set()
        for i in range(0, len(deployers), chunk):
            sub = deployers[i:i+chunk]
            ph = ",".join("?" * len(sub))
            row = conn.execute(f"""
                SELECT COUNT(*),
                       SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END)
                FROM contracts WHERE LOWER(deployer_address) IN ({ph})
            """, sub).fetchone()
            contract_count += row[0] or 0
            confirmed += row[1] or 0
            suspected += row[2] or 0
            for r in conn.execute(f"SELECT DISTINCT chain FROM deployers WHERE LOWER(deployer_address) IN ({ph})", sub):
                if r[0]:
                    chains.add(r[0])

        if contract_count == 0:
            continue
        adversarial_ratio = (confirmed + suspected) / contract_count
        avg_fleet = contract_count / n_deployers if n_deployers else 0

        # Apply gating signals
        if adversarial_ratio < MIN_ADVERSARIAL_RATIO:
            continue
        if disposable_rate < MIN_DISPOSABLE_RATE:
            continue

        # Funder's own corpus presence (date range)
        first_dep = conn.execute("""
            SELECT MIN(first_seen), MAX(first_seen) FROM deployers
            WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
        """, (funder,)).fetchone()

        # Is funder itself a deployer?
        funder_dep = conn.execute("""
            SELECT mainnet_first_tx FROM deployers WHERE LOWER(deployer_address) = ?
        """, (funder,)).fetchone()
        mainnet_first_tx = (funder_dep[0] if funder_dep else None) or None
        if mainnet_first_tx == '':
            mainnet_first_tx = None
        funder_known = 1 if funder_dep else 0

        results.append({
            "funder_address": funder,
            "chain_signature": ",".join(sorted(chains)),
            "deployer_count": n_deployers,
            "contract_count": contract_count,
            "confirmed_count": confirmed,
            "suspected_count": suspected,
            "adversarial_ratio": adversarial_ratio,
            "disposable_rate": disposable_rate,
            "avg_fleet_per_deployer": avg_fleet,
            "funder_first_seen_in_corpus": first_dep[0] if first_dep else None,
            "funder_last_seen_in_corpus": first_dep[1] if first_dep else None,
            "funder_mainnet_first_tx": mainnet_first_tx,
            "funder_known_in_deployers": funder_known,
        })
    return results


def apply_candidates(conn: sqlite3.Connection, rows: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    refreshed = 0
    for r in rows:
        existing = conn.execute(
            "SELECT id FROM infrastructure_operator_candidates WHERE funder_address = ?",
            (r["funder_address"],),
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE infrastructure_operator_candidates SET
                    chain_signature = ?, deployer_count = ?, contract_count = ?,
                    confirmed_count = ?, suspected_count = ?, adversarial_ratio = ?,
                    disposable_rate = ?, avg_fleet_per_deployer = ?,
                    funder_first_seen_in_corpus = ?, funder_last_seen_in_corpus = ?,
                    funder_mainnet_first_tx = ?, funder_known_in_deployers = ?,
                    last_checked = ?
                WHERE funder_address = ?
            """, (r["chain_signature"], r["deployer_count"], r["contract_count"],
                  r["confirmed_count"], r["suspected_count"], r["adversarial_ratio"],
                  r["disposable_rate"], r["avg_fleet_per_deployer"],
                  r["funder_first_seen_in_corpus"], r["funder_last_seen_in_corpus"],
                  r["funder_mainnet_first_tx"], r["funder_known_in_deployers"],
                  now, r["funder_address"]))
            refreshed += 1
        else:
            conn.execute("""
                INSERT INTO infrastructure_operator_candidates
                    (funder_address, chain_signature, deployer_count, contract_count,
                     confirmed_count, suspected_count, adversarial_ratio, disposable_rate,
                     avg_fleet_per_deployer, funder_first_seen_in_corpus,
                     funder_last_seen_in_corpus, funder_mainnet_first_tx,
                     funder_known_in_deployers, detected_at, last_checked, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (r["funder_address"], r["chain_signature"], r["deployer_count"],
                  r["contract_count"], r["confirmed_count"], r["suspected_count"],
                  r["adversarial_ratio"], r["disposable_rate"], r["avg_fleet_per_deployer"],
                  r["funder_first_seen_in_corpus"], r["funder_last_seen_in_corpus"],
                  r["funder_mainnet_first_tx"], r["funder_known_in_deployers"],
                  now, now))
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

    conn = sqlite3.connect(args.db, timeout=120)
    conn.row_factory = sqlite3.Row
    if args.apply:
        ensure_table(conn)

    rows = find_infrastructure_operators(conn)
    print(f"[infrastructure_operator_detector] candidates: {len(rows)}")

    if args.dry_run:
        print("\nAll candidates by deployer_count desc:")
        for r in sorted(rows, key=lambda x: -x["deployer_count"]):
            mn = (r["funder_mainnet_first_tx"] or "L2-only")[:10]
            print(f"  {r['funder_address']}  "
                  f"deps={r['deployer_count']:<5} contracts={r['contract_count']:<5} "
                  f"conf={r['confirmed_count']:<4} sus={r['suspected_count']:<5} "
                  f"adv={r['adversarial_ratio']*100:>5.1f}% disp={r['disposable_rate']*100:>5.1f}% "
                  f"avg_fleet={r['avg_fleet_per_deployer']:.1f}  "
                  f"chains={r['chain_signature']:<20} mainnet={mn}")
        return

    res = apply_candidates(conn, rows)
    print(f"[infrastructure_operator_detector] applied: inserted={res['inserted']} refreshed={res['refreshed']}")
    conn.close()


if __name__ == "__main__":
    main()
