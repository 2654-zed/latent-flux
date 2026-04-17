"""
Correction: NULL-bucket reclassification.

Fixes two classes of mislabels produced by the T2-eaef6a5d7678 "NULL family":

1. 2,119 contracts with tier=suspected + method=bytecode_pattern + all flags=0
   + notes=NULL are moved to tier=unknown. The bytecode classifier found no
   trap patterns; these contracts should never have been marked suspected.

2. The bytecode_families table / members entries for T2-eaef6a5d7678 are
   removed. The "NULL family" is not a family -- it is the absence of
   classification. Each contract stays in the contracts table; only its
   family membership is deleted.

Idempotent: safe to run multiple times. Reports counts at each step.
Dry-run by default -- pass --apply to write changes.

Usage:
    python correction_null_bucket.py                 # dry run (report only)
    python correction_null_bucket.py --apply         # apply to local DB
    python correction_null_bucket.py --apply --db /path/to/surveillance.db
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone


def run(db_path: str, apply_changes: bool):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    now_iso = datetime.now(timezone.utc).isoformat()

    FID = "T2-eaef6a5d7678"

    # --- Step 1: Identify misclassified contracts ---
    # tier=suspected + method=bytecode_pattern + all flags=0 + notes=NULL
    rows = conn.execute("""
        SELECT contract_address, confidence_reason
        FROM contracts
        WHERE confidence_tier = 'suspected'
          AND detection_method = 'bytecode_pattern'
          AND COALESCE(has_asymmetric_transfer, 0) = 0
          AND COALESCE(has_conditional_revert, 0) = 0
          AND COALESCE(has_unusual_fee_structure, 0) = 0
          AND (bytecode_pattern_notes IS NULL OR bytecode_pattern_notes = '')
    """).fetchall()
    misclassified = len(rows)
    print(f"[step 1] Contracts to downgrade suspected->unknown: {misclassified}")

    # --- Step 2: Confirm deployer-history suspected count (no change to these) ---
    history_suspected = conn.execute("""
        SELECT COUNT(*) FROM contracts
        WHERE confidence_tier = 'suspected' AND detection_method = 'deployer_history'
    """).fetchone()[0]
    print(f"[step 2] Deployer-history suspected (KEPT in suspected): {history_suspected}")

    # --- Step 3: NULL family membership to delete ---
    family_members = conn.execute(
        "SELECT COUNT(*) FROM bytecode_family_members WHERE family_id = ?", (FID,)
    ).fetchone()[0]
    family_exists = conn.execute(
        "SELECT COUNT(*) FROM bytecode_families WHERE family_id = ?", (FID,)
    ).fetchone()[0]
    print(f"[step 3] NULL family members to delete: {family_members}")
    print(f"[step 3] NULL family record to delete: {family_exists}")

    # --- Current totals before change ---
    total_suspected_before = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'suspected'"
    ).fetchone()[0]
    total_unknown_before = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'unknown'"
    ).fetchone()[0]
    print(f"\n[before] suspected={total_suspected_before:,} unknown={total_unknown_before:,}")

    projected_suspected = total_suspected_before - misclassified
    projected_unknown = total_unknown_before + misclassified
    print(f"[after]  suspected={projected_suspected:,} unknown={projected_unknown:,}")
    print(f"[delta]  suspected {-misclassified:+,}, unknown {+misclassified:+,}")

    if not apply_changes:
        print("\nDRY RUN -- no changes written. Pass --apply to execute.")
        conn.close()
        return

    # --- Apply ---
    print(f"\nApplying changes to {db_path}...")

    # 1. Downgrade the 2,119 misclassified
    if misclassified:
        new_reason = (
            "Bytecode classifier ran, no trap patterns detected (all flags=0, "
            "no pattern notes). Reclassified from suspected to unknown on "
            f"{now_iso[:10]} -- previous suspected label had no supporting evidence."
        )
        conn.execute("""
            UPDATE contracts
            SET confidence_tier = 'unknown',
                confidence_reason = ? || CHAR(10) || '[prior: ' || COALESCE(confidence_reason, '') || ']',
                last_updated = ?
            WHERE confidence_tier = 'suspected'
              AND detection_method = 'bytecode_pattern'
              AND COALESCE(has_asymmetric_transfer, 0) = 0
              AND COALESCE(has_conditional_revert, 0) = 0
              AND COALESCE(has_unusual_fee_structure, 0) = 0
              AND (bytecode_pattern_notes IS NULL OR bytecode_pattern_notes = '')
        """, (new_reason, now_iso))
        print(f"  downgraded {misclassified} contracts: suspected -> unknown")

    # 2. Remove NULL family membership and family record
    if family_members:
        conn.execute("DELETE FROM bytecode_family_members WHERE family_id = ?", (FID,))
        print(f"  deleted {family_members} family members from {FID}")
    if family_exists:
        conn.execute("DELETE FROM bytecode_families WHERE family_id = ?", (FID,))
        print(f"  deleted family record {FID}")

    conn.commit()

    # Verify
    after_suspected = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'suspected'"
    ).fetchone()[0]
    after_unknown = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'unknown'"
    ).fetchone()[0]
    print(f"\n[verified] suspected={after_suspected:,} unknown={after_unknown:,}")
    conn.close()
    print("Correction applied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--db", default=os.environ.get(
        "DB_PATH", "surveillance/data/surveillance.db"))
    args = parser.parse_args()
    run(args.db, args.apply)
