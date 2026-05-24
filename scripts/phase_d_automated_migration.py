"""Phase D — migrate the 162 Phase C automated LIKELY_FP contracts.

Per the audit plan + user authorization. Migrates contracts where
Phase C's automated classifier produced one of:

  LIKELY_FP_FROM_SOURCE    (130) — verified-source matches OZ/Animoca/
                                   Solady/Solmate/etc. framework
  LIKELY_FP_FROM_ACTIVITY  (25)  — massive interactor diversity (real
                                   DeFi infrastructure)
  LIKELY_FP_FROM_CLUSTER   (7)   — sibling of a Phase A FP

Annotation captures the per-contract rationale.

CLI:
    python scripts/phase_d_automated_migration.py
    python scripts/phase_d_automated_migration.py --apply
"""
from __future__ import annotations
import argparse, csv, sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_c_2026-05-22.csv"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

FP_VERDICTS = {"LIKELY_FP_FROM_SOURCE", "LIKELY_FP_FROM_ACTIVITY", "LIKELY_FP_FROM_CLUSTER"}

AUDIT_ANNOTATION = (
    "[AUDIT 2026-05-22 / Correction #25 Phase C automated: "
    "{verdict} - {rationale}. Original reason preserved below.] | "
)


def load_fp_candidates(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["phase_c_verdict"] in FP_VERDICTS:
                rows.append(r)
    return rows


def run_migration(conn, rows, apply: bool) -> dict:
    counts = {"candidates": len(rows), "found_confirmed": 0, "already_migrated": 0,
              "not_in_db": 0, "tier_mismatch": 0, "migrated": 0}
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
        if cur_tier == "unanalyzed" and "[AUDIT 2026-05-22 / Correction #25 Phase C automated" in cur_reason:
            counts["already_migrated"] += 1
            continue
        if cur_tier != "confirmed":
            counts["tier_mismatch"] += 1
            continue

        counts["found_confirmed"] += 1
        if not apply:
            continue

        rat = (r.get("phase_c_rationale") or "")[:160].replace("|", "/")
        ann = AUDIT_ANNOTATION.format(verdict=r["phase_c_verdict"], rationale=rat)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = load_fp_candidates(Path(args.csv))
    print(f"=== Phase D automated migration ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    print(f"  DB: {args.db}")
    print(f"  {len(rows)} LIKELY_FP candidates (source+activity+cluster)")

    conn = sqlite3.connect(args.db)
    try:
        counts = run_migration(conn, rows, apply=args.apply)
    finally:
        conn.close()
    print()
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
