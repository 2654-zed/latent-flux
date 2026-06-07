"""Revert ALL approval_watchlist.drain_detected=1 flags to 0 (Correction #29).

Rationale: check_drains_blockscout's n_out>0 signal conflates DEX sales with
drains (~98% false positives per on-chain tx-initiator sampling). Every current
drain=1 flag is therefore unreliable and must be re-derived by the corrected
tx-initiator-gated detector. The user opted to audit ALL flags (today's 40,144
backfill + the ~6,449 older t1_apply / old-method rows), so we clear them all to
pending=0 and let the fixed detector re-flag the genuine drains.

Non-destructive: every cleared row is snapshotted to drain_flags_backup_20260606
first (with its checked_at provenance), so the revert is fully reversible.
audit_drain_legs (the n_out cache) is PRESERVED — it is still valid raw input;
the fix layers the tx-initiator test on top of it.

CLI:  python scripts/revert_drain_fps_20260606.py            # dry-run
      python scripts/revert_drain_fps_20260606.py --apply
"""
from __future__ import annotations
import argparse, sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
BK = "drain_flags_backup_20260606"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    before = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
    print(f"drain_detected=1 before: {before}")
    # provenance split
    for r in conn.execute("""
        SELECT COALESCE(substr(a.checked_at,1,10),'(no-cache)') d, COUNT(*) n
        FROM approval_watchlist aw
        LEFT JOIN audit_drain_legs a ON a.victim=aw.victim_address AND a.contract=aw.contract_address
        WHERE aw.drain_detected=1 GROUP BY d ORDER BY n DESC"""):
        print(f"  source checked_at {r[0]}: {r[1]}")

    if not args.apply:
        print("\n(DRY-RUN — no changes. Re-run with --apply.)")
        return

    # 1) backup (idempotent: drop+recreate so re-runs are clean)
    conn.execute(f"DROP TABLE IF EXISTS {BK}")
    conn.execute(f"""
        CREATE TABLE {BK} AS
        SELECT aw.victim_address, aw.contract_address, aw.drain_detected,
               aw.drain_tx_hash, aw.drain_timestamp, aw.drain_caller,
               a.checked_at AS cache_checked_at, a.n_out AS cache_n_out
        FROM approval_watchlist aw
        LEFT JOIN audit_drain_legs a
          ON a.victim=aw.victim_address AND a.contract=aw.contract_address
        WHERE aw.drain_detected=1
    """)
    nbk = conn.execute(f"SELECT COUNT(*) FROM {BK}").fetchone()[0]
    conn.commit()
    print(f"backed up {nbk} rows -> {BK}")

    # 2) revert
    cur = conn.execute("""
        UPDATE approval_watchlist
        SET drain_detected=0, drain_tx_hash=NULL, drain_timestamp=NULL, drain_caller=NULL
        WHERE drain_detected=1
    """)
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
    pend = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=0").fetchone()[0]
    cache = conn.execute("SELECT COUNT(*) FROM audit_drain_legs").fetchone()[0]
    print(f"reverted {cur.rowcount} rows; drain_detected=1 after: {after}")
    print(f"pending (drain_detected=0) now: {pend}")
    print(f"audit_drain_legs cache PRESERVED: {cache} rows")


if __name__ == "__main__":
    main()
