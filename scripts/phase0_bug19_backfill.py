"""Phase 0 / Bug #19 backfill — re-flag phantom drain_detected=1 rows in
approval_watchlist that map to reverted transactions.

Per Correction #24 (2026-05-21), surveillance/approval_drain_monitor.py's
check_drains() function had two bugs that combined to credit phantom drain
rows against entire approval pools when a single transferFrom call hit a
contract:

  Bug A — Method 1 (transferFrom scan): no tx-status filter. A reverted
  transferFrom on the contract was credited as draining every prior
  approver. The LIMIT 1 on the subquery meant every approver was credited
  against the same (first-seen) tx_hash.

  Bug B — Method 2 (deployer drain scan): same missing tx-status filter
  for deployer-mediated drains.

The Correction-#24 anchor case: 0x752c5a95 (OneFootball Club ERC-20).
Three failed transferFrom transactions on 2026-05-09 produced 4,587
phantom drain_detected=1 rows credited against the entire approval pool.

This script does the backfill, idempotently:

  1. For every approval_watchlist row with drain_detected=1 and
     drain_tx_hash NOT NULL:
       a) Look up the tx_hash in transaction_events.
       b) If the tx is_reverted = 1, the row is a phantom — reset.
       c) If the tx is not in transaction_events at all (e.g., it's a
          reverted tx we never ingested), treat as phantom — reset.
       d) Otherwise keep the drain attribution.

  2. Phantom rows are reset to drain_detected=0, drain_tx_hash=NULL,
     drain_timestamp=NULL, drain_caller=NULL. The original approval
     evidence (victim_address, contract_address, approve_tx_hash,
     approve_timestamp) is preserved — only the drain attribution
     becomes pending again.

The corresponding code fix (adding `AND te.is_reverted = 0` to the
transferFrom scan + deployer drain scan in approval_drain_monitor.py)
is shipped in the same commit so the bug cannot regenerate.

CLI:
    python scripts/phase0_bug19_backfill.py
        # dry-run, local DB

    python scripts/phase0_bug19_backfill.py --apply
        # apply on local DB

    python scripts/phase0_bug19_backfill.py --db /app/surveillance/data/surveillance.db
        # dry-run against a specific DB path (e.g. prod via railway ssh)
"""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"


def detect_hash_format(conn: sqlite3.Connection, table: str, column: str) -> str:
    """Return '0x' if the column stores '0x' prefixed hashes, '' otherwise."""
    rows = conn.execute(
        f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 5"
    ).fetchall()
    if not rows:
        return ""
    sample = rows[0][0] or ""
    return "0x" if sample.lower().startswith("0x") else ""


def run_backfill(conn: sqlite3.Connection, apply: bool) -> dict:
    aw_fmt = detect_hash_format(conn, "approval_watchlist", "drain_tx_hash")
    te_fmt = detect_hash_format(conn, "transaction_events", "tx_hash")
    print(f"  storage formats: aw.drain_tx_hash uses {aw_fmt!r}-prefix; tx_events.tx_hash uses {te_fmt!r}-prefix")

    if aw_fmt == te_fmt:
        cmp_expr = "te.tx_hash = {aw}.drain_tx_hash"
    elif aw_fmt == "0x" and te_fmt == "":
        cmp_expr = "te.tx_hash = SUBSTR({aw}.drain_tx_hash, 3)"
    elif aw_fmt == "" and te_fmt == "0x":
        cmp_expr = "te.tx_hash = ('0x' || {aw}.drain_tx_hash)"
    else:
        raise RuntimeError(f"Unexpected formats: aw={aw_fmt} te={te_fmt}")

    # cmp_expr for correlated subquery; need to refer to the alias the outer query uses
    cmp_aw = cmp_expr.format(aw="aw")
    cmp_full = cmp_expr.format(aw="approval_watchlist")

    print("  computing counts (this can take 30-90s on a large corpus)...")
    drain_rows_total = conn.execute(
        "SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1"
    ).fetchone()[0]
    drain_rows_with_tx = conn.execute(
        "SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL"
    ).fetchone()[0]

    tx_in_events_reverted = conn.execute(f"""
        SELECT COUNT(*) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_aw} AND te.is_reverted=1)
    """).fetchone()[0]
    tx_in_events_success = conn.execute(f"""
        SELECT COUNT(*) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_aw} AND te.is_reverted=0)
    """).fetchone()[0]
    tx_not_in_events = drain_rows_with_tx - tx_in_events_reverted - tx_in_events_success
    phantom = tx_in_events_reverted + tx_not_in_events

    phantom_contracts = conn.execute(f"""
        SELECT COUNT(DISTINCT aw.contract_address) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND (
          EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_aw} AND te.is_reverted=1)
          OR NOT EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_aw})
        )
    """).fetchone()[0]

    phantom_txs = conn.execute(f"""
        SELECT COUNT(DISTINCT aw.drain_tx_hash) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND (
          EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_aw} AND te.is_reverted=1)
          OR NOT EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_aw})
        )
    """).fetchone()[0]

    result = {
        "drain_rows_total": drain_rows_total,
        "drain_rows_with_tx": drain_rows_with_tx,
        "tx_in_events_reverted": tx_in_events_reverted,
        "tx_in_events_success": tx_in_events_success,
        "tx_not_in_events": tx_not_in_events,
        "phantom_rows_total": phantom,
        "phantom_distinct_contracts": phantom_contracts,
        "phantom_distinct_tx_hashes": phantom_txs,
    }

    if not apply:
        return result

    print("  applying backfill...")
    cur = conn.execute(f"""
        UPDATE approval_watchlist
        SET drain_detected = 0,
            drain_tx_hash = NULL,
            drain_timestamp = NULL,
            drain_caller = NULL
        WHERE drain_detected = 1 AND drain_tx_hash IS NOT NULL
        AND (
          EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_full} AND te.is_reverted=1)
          OR NOT EXISTS (SELECT 1 FROM transaction_events te WHERE {cmp_full})
        )
    """)
    conn.commit()
    result["rows_reset"] = cur.rowcount
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help="path to surveillance.db (default: local)")
    ap.add_argument("--apply", action="store_true",
                    help="actually reset phantom rows (default: dry-run)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    print(f"=== {db_path} ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    conn = sqlite3.connect(db_path)
    try:
        result = run_backfill(conn, apply=args.apply)
    finally:
        conn.close()

    print()
    print("  Summary:")
    for k, v in result.items():
        if isinstance(v, int):
            print(f"    {k:32s}: {v:>12,}")
    if args.apply:
        print()
        print(f"  Reset {result.get('rows_reset', 0):,} phantom drain rows.")
    else:
        print()
        print("  (Dry run — no rows modified. Add --apply to commit.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
