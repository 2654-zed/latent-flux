"""Phase D follow-up — migrate the 29 sample-validated FP_FROM_SAMPLE contracts.

Per the user's Option B choice on Phase C residual: migrate only the
contracts that the per-contract sample review individually validated
as FP. The other 180 STILL_AMBIGUOUS remain in confirmed pending
broader review.

Per-contract rationale is preserved in the audit annotation. These are
the hand-validated contracts from the 50-row stratified sample.

Input: reports/confirmed_tier_audit_phase_c_sample_review_2026-05-22.json
       Filter: verdict='FP_FROM_SAMPLE'

CLI:
    python scripts/phase_d_sample_migration.py             # dry-run local
    python scripts/phase_d_sample_migration.py --apply     # apply local
    python scripts/phase_d_sample_migration.py --db /app/...   # prod path
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_JSON = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_c_sample_review_2026-05-22.json"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

AUDIT_ANNOTATION = (
    "[AUDIT 2026-05-22 / Correction #25 Phase C sample review: "
    "FP_FROM_SAMPLE - {rationale}. Original reason preserved below.] | "
)


def load_fp_candidates(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [r for r in data if r["verdict"] == "FP_FROM_SAMPLE"]


def run_migration(conn: sqlite3.Connection, rows: list[dict], apply: bool) -> dict:
    counts = {"candidates": len(rows), "found_confirmed": 0, "already_migrated": 0,
              "not_in_db": 0, "tier_mismatch": 0, "migrated": 0}
    for r in rows:
        addr = r["contract"]
        existing = conn.execute(
            "SELECT confidence_tier, confidence_reason FROM contracts WHERE contract_address=?",
            (addr,)
        ).fetchone()
        if not existing:
            counts["not_in_db"] += 1
            continue
        cur_tier, cur_reason = existing[0], existing[1] or ""
        if cur_tier == "unanalyzed" and "[AUDIT 2026-05-22 / Correction #25 Phase C sample" in cur_reason:
            counts["already_migrated"] += 1
            continue
        if cur_tier != "confirmed":
            counts["tier_mismatch"] += 1
            continue

        counts["found_confirmed"] += 1
        if not apply:
            continue

        # Truncate rationale to keep annotation length sane
        rat = (r.get("rationale") or "")[:160].replace("|", "/")
        ann = AUDIT_ANNOTATION.format(rationale=rat)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    json_path = Path(args.json)
    db_path = Path(args.db)
    if not json_path.exists():
        print(f"JSON not found: {json_path}"); return 1
    if not db_path.exists():
        print(f"DB not found: {db_path}"); return 1

    rows = load_fp_candidates(json_path)
    print(f"=== Phase D sample migration ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    print(f"  DB: {db_path}")
    print(f"  {len(rows)} FP_FROM_SAMPLE candidates")

    conn = sqlite3.connect(db_path)
    try:
        counts = run_migration(conn, rows, apply=args.apply)
    finally:
        conn.close()

    print()
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if args.apply:
        print(f"\n  Migrated {counts['migrated']} contracts (confirmed -> unanalyzed).")
    else:
        print("\n  (Dry run - no rows modified. Add --apply to commit.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
