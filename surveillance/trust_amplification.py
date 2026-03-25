"""Trust Amplification — identifies contracts whose traffic is router-dominated.

For contracts with 50+ callers, measures what percentage of calls route through
selector 3593564c. Compares callers/day against bytecode family average to
compute an amplification factor.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

ROUTER_SELECTOR = "3593564c"


def get_connection() -> sqlite3.Connection:
    """Return a connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _get_family_avg(conn: sqlite3.Connection, contract_address: str) -> Optional[float]:
    """Get the avg callers/day for the bytecode family this contract belongs to."""
    fam = conn.execute(
        "SELECT family_id FROM bytecode_family_members WHERE contract_address = ?",
        (contract_address,)
    ).fetchone()
    if not fam:
        return None

    siblings = conn.execute("""
        SELECT bfm.contract_address
        FROM bytecode_family_members bfm
        WHERE bfm.family_id = ?
    """, (fam["family_id"],)).fetchall()

    if not siblings:
        return None

    total_cpd = 0.0
    count = 0
    for s in siblings:
        row = conn.execute("""
            SELECT COUNT(DISTINCT interacting_address) as callers,
                   JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp)) as span_days
            FROM transaction_events
            WHERE contract_address = ?
        """, (s["contract_address"],)).fetchone()
        if row and row["callers"] and row["span_days"] and row["span_days"] > 0:
            total_cpd += row["callers"] / row["span_days"]
            count += 1

    return round(total_cpd / count, 2) if count else None


def analyze(conn: sqlite3.Connection) -> None:
    """Analyze trust amplification for all qualifying contracts."""
    print("[trust_amplification] Finding contracts with 50+ callers...")

    contracts = conn.execute("""
        SELECT contract_address,
               COUNT(DISTINCT interacting_address) as total_callers,
               COUNT(*) as total_tx,
               JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp)) as span_days,
               ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100, 1) as revert_rate,
               MIN(timestamp) as first_seen
        FROM transaction_events
        GROUP BY contract_address
        HAVING COUNT(DISTINCT interacting_address) >= 50
    """).fetchall()

    print(f"  Found {len(contracts)} contracts with 50+ callers")

    conn.execute("DELETE FROM trust_amplification")
    inserted = 0

    for c in contracts:
        addr = c["contract_address"]
        # Count calls through router selector
        router_row = conn.execute("""
            SELECT COUNT(*) as router_calls,
                   COUNT(DISTINCT interacting_address) as router_callers
            FROM transaction_events
            WHERE contract_address = ? AND function_selector = ?
        """, (addr, ROUTER_SELECTOR)).fetchone()

        router_callers = router_row["router_callers"] if router_row else 0
        router_pct = round(router_callers / c["total_callers"] * 100, 1) if c["total_callers"] else 0

        span = max(c["span_days"] or 1, 1)
        callers_per_day = round(c["total_callers"] / span, 2)

        # Get family avg
        family_avg = _get_family_avg(conn, addr)

        # Look up family
        fam_row = conn.execute(
            "SELECT family_id FROM bytecode_family_members WHERE contract_address = ?",
            (addr,)
        ).fetchone()
        family_id = fam_row["family_id"] if fam_row else None

        if family_avg and family_avg > 0:
            amplification = round(callers_per_day / family_avg, 2)
        else:
            amplification = 1.0
            family_avg = callers_per_day  # self-baseline

        # Alert level
        if amplification > 10 and c["total_callers"] >= 100:
            alert = "CRITICAL"
        elif amplification > 5 and c["total_callers"] >= 50:
            alert = "WARNING"
        elif amplification > 2:
            alert = "INFO"
        else:
            alert = None

        conn.execute("""
            INSERT OR REPLACE INTO trust_amplification
                (contract_address, total_callers, router_callers, router_percentage,
                 callers_per_day, bytecode_family, family_avg_callers_per_day,
                 amplification_factor, revert_rate, first_seen, last_updated, alert_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
        """, (addr, c["total_callers"], router_callers, router_pct,
              callers_per_day, family_id, family_avg, amplification,
              c["revert_rate"], c["first_seen"], alert))
        inserted += 1

        if alert:
            print(f"  [{alert}] {addr[:16]}... amp={amplification}x "
                  f"router={router_pct}% callers={c['total_callers']}")

    conn.commit()
    print(f"[trust_amplification] Done. {inserted} contracts analyzed.")

    # Summary
    for level in ("CRITICAL", "WARNING", "INFO"):
        cnt = conn.execute(
            "SELECT COUNT(*) FROM trust_amplification WHERE alert_level = ?", (level,)
        ).fetchone()[0]
        if cnt:
            print(f"  {level}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trust Amplification Analyzer")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    args = parser.parse_args()

    if args.analyze:
        conn = get_connection()
        analyze(conn)
        conn.close()
    else:
        parser.print_help()
