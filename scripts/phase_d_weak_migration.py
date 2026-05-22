"""Phase D follow-up — migrate Phase B LIKELY_FP_WEAK contracts.

Per reports/confirmed_tier_audit_plan.md Phase B + Phase D follow-up:

  After Phase A migrated the 116 STRONG LIKELY_FP cases, Phase B applied
  internal heuristics to the residual. The LIKELY_FP_WEAK class (40
  contracts) is:

    - self-loop or BACKFILL reason
    - solo deployer (no recidivism — no other confirmed contracts)
    - no drain activity in approval_watchlist
    - no bytecode_cache row
    - no institutional OLI tag

  Per the audit plan's Class D guidance: "downgraded en masse pending
  stronger evidence." This script does that migration.

Input: reports/confirmed_tier_audit_phase_b_2026-05-22.csv
       Filter: phase_b_verdict='LIKELY_FP_WEAK'

Action: UPDATE contracts SET confidence_tier='unanalyzed',
        confidence_reason=<audit annotation> + <original>,
        last_updated=NOW WHERE contract_address=...

CLI:
    python scripts/phase_d_weak_migration.py             # dry-run local
    python scripts/phase_d_weak_migration.py --apply     # apply local
    python scripts/phase_d_weak_migration.py --db /app/surveillance/data/surveillance.db
    python scripts/phase_d_weak_migration.py --db ... --apply
"""
from __future__ import annotations
import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_b_2026-05-22.csv"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

AUDIT_ANNOTATION = (
    "[AUDIT 2026-05-22 / Correction #25 Phase B: LIKELY_FP_WEAK - "
    "{detail}. Solo deployer (no recidivism), no drain activity, no "
    "bytecode evidence. Original reason preserved below.] | "
)


def load_weak_candidates(csv_path: Path) -> list[dict]:
    """Load rows where phase_b_verdict='LIKELY_FP_WEAK'."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["phase_b_verdict"] == "LIKELY_FP_WEAK":
                rows.append(r)
    return rows


def annotation_detail(row: dict) -> str:
    rc = row.get("reason_class", "")
    if rc == "self_loop":
        return "self-loop reason"
    if rc == "backfill":
        return "BACKFILL self-loop reason"
    return rc or "weak-evidence class"


def run_migration(conn: sqlite3.Connection, rows: list[dict], apply: bool) -> dict:
    counts = {
        "candidates": len(rows),
        "found_confirmed": 0,
        "already_migrated": 0,
        "not_in_db": 0,
        "tier_mismatch": 0,
        "migrated": 0,
    }
    for r in rows:
        addr = r["contract_address"]
        existing = conn.execute(
            "SELECT confidence_tier, confidence_reason FROM contracts WHERE contract_address=?",
            (addr,)
        ).fetchone()
        if not existing:
            counts["not_in_db"] += 1
            continue
        cur_tier, cur_reason = existing[0], existing[1] or ""
        if cur_tier == "unanalyzed" and "[AUDIT 2026-05-22 / Correction #25 Phase B" in cur_reason:
            counts["already_migrated"] += 1
            continue
        if cur_tier != "confirmed":
            counts["tier_mismatch"] += 1
            continue

        counts["found_confirmed"] += 1
        if not apply:
            continue

        ann = AUDIT_ANNOTATION.format(detail=annotation_detail(r))
        new_reason = ann + cur_reason
        conn.execute(
            "UPDATE contracts SET confidence_tier='unanalyzed', confidence_reason=?, "
            "last_updated=? WHERE contract_address=?",
            (new_reason, NOW, addr)
        )
        counts["migrated"] += 1

    if apply:
        conn.commit()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    print(f"=== Phase B + D migration ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    print(f"  CSV: {csv_path}")
    print(f"  DB:  {db_path}")
    rows = load_weak_candidates(csv_path)
    print(f"  {len(rows)} LIKELY_FP_WEAK candidates from CSV")

    conn = sqlite3.connect(db_path)
    try:
        counts = run_migration(conn, rows, apply=args.apply)
    finally:
        conn.close()

    print()
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if args.apply:
        print()
        print(f"  Migrated {counts['migrated']} contracts (confirmed -> unanalyzed).")
    else:
        print()
        print("  (Dry run - no rows modified. Add --apply to commit.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
