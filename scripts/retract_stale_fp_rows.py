"""Retract stale false_positives rows where the contract has observed harm.

Class of bug: post-write mutation of related state. FP scanner ran when the
contract was 'suspected'; the contract was later promoted to 'confirmed' via
trap_event, but the FP row persisted as a contradicting assertion.

Per Correction #17 (2026-04-25). Retracts only rows where:
  - contract is currently confidence_tier='confirmed'
  - AND has at least one trap_events row

Idempotent. Dry-run default; pass --commit to execute.

Usage:
    python scripts/retract_stale_fp_rows.py --dry-run
    python scripts/retract_stale_fp_rows.py --commit
"""
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_DEFAULT = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")


def find_stale(conn: sqlite3.Connection) -> list[dict]:
    return list(conn.execute("""
        SELECT fp.contract_address, fp.fp_method, fp.fp_reason, fp.fp_confidence,
               fp.assessed_at,
               (SELECT COUNT(*) FROM trap_events
                WHERE LOWER(trap_contract_address) = LOWER(fp.contract_address)) AS traps,
               ct.confidence_tier
        FROM false_positives fp
        JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(fp.contract_address)
        WHERE ct.confidence_tier = 'confirmed'
          AND (SELECT COUNT(*) FROM trap_events
               WHERE LOWER(trap_contract_address) = LOWER(fp.contract_address)) > 0
        ORDER BY fp.contract_address
    """))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=str(DB_DEFAULT))
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    do_commit = args.commit and not args.dry_run if args.dry_run else args.commit
    # Simpler: --commit overrides
    do_commit = args.commit

    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row

    stale = find_stale(conn)
    print(f"Stale FP rows (confirmed tier + observed trap_events): {len(stale)}")
    print()
    print(f"  {'address':<44} {'fp_method':<22} {'traps':<6} {'fp_confidence':<14} {'assessed_at'}")
    for r in stale:
        print(f"  {r['contract_address']}  {r['fp_method']:<22} {r['traps']:<6} "
              f"{r['fp_confidence']:<14} {r['assessed_at']}")

    if not stale:
        return

    if not do_commit:
        print()
        print("(dry-run; pass --commit to retract)")
        return

    # Also flip the alerts.false_positive=1 flag back to 0 for any matching alerts
    # so the downstream feed reflects the corrected classification.
    deleted = 0
    alert_unflipped = 0
    for r in stale:
        addr = r["contract_address"].lower()
        conn.execute("DELETE FROM false_positives WHERE LOWER(contract_address) = ?", (addr,))
        deleted += 1
        # un-silence any alerts on this contract
        cur = conn.execute(
            "UPDATE alerts SET false_positive = 0 WHERE LOWER(address) = ? AND false_positive = 1",
            (addr,))
        alert_unflipped += cur.rowcount
    conn.commit()
    print()
    print(f"Retracted {deleted} false_positives rows.")
    print(f"Un-silenced {alert_unflipped} alerts.")
    conn.close()


if __name__ == "__main__":
    main()
