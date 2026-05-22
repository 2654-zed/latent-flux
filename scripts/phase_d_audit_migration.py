"""Phase D migration — move audited LIKELY_FP contracts from confirmed to unanalyzed.

Per reports/confirmed_tier_audit_plan.md Phase D + correction_log #25:

  For every contract Phase A identified as LIKELY_FP (high-confidence
  false-positive per Blockscout enrichment):
    - Move from `confirmed` tier to `unanalyzed` (the Correction #3
      precedent — don't move to suspected, which still carries
      adversarial connotation).
    - Annotate `confidence_reason` with the audit's verdict + date.
    - Preserve the original reason for traceability (prepended audit
      annotation, not destructive).

Input: reports/confirmed_tier_audit_2026-05-22.csv (the Phase A CSV).
       Each row with preliminary_verdict='LIKELY_FP' becomes a migration
       candidate.

This script does NOT touch:
  - The bytecode_cache table (Correction #5 cache-invalidation already
    fires on confidence updates via db.update_contract_confidence; but
    we bypass that helper here to preserve the audit annotation format).
  - The approval_watchlist (drain_detected is already corrected via
    Phase 0 / Bug #19 backfill).
  - The deployers table.

CLI:
    python scripts/phase_d_audit_migration.py
        # dry-run on local DB

    python scripts/phase_d_audit_migration.py --apply
        # apply on local DB

    python scripts/phase_d_audit_migration.py --db /app/surveillance/data/surveillance.db
        # dry-run against a specific DB (e.g. prod via railway ssh)

    python scripts/phase_d_audit_migration.py --db ... --apply
        # apply on prod
"""
from __future__ import annotations
import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_2026-05-22.csv"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

AUDIT_ANNOTATION = (
    "[AUDIT 2026-05-22 / Correction #25: STRONG LIKELY_FP per Blockscout enrichment - "
    "{evidence_short}. Original reason preserved below.] | "
)


def load_fp_candidates(csv_path: Path) -> list[dict]:
    """Load LIKELY_FP rows from the audit CSV."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["preliminary_verdict"] == "LIKELY_FP":
                rows.append(r)
    return rows


def evidence_short(row: dict) -> str:
    """Build a short audit-evidence string for the confidence_reason annotation."""
    parts = []
    if row.get("token_name"):
        parts.append(f"token={row['token_name'][:30]}")
    if row.get("is_verified") == "True":
        parts.append("verified-source")
    if row.get("holders_count"):
        try:
            hc = int(str(row["holders_count"]).replace(",", ""))
            if hc > 0:
                parts.append(f"{hc:,} holders")
        except Exception:
            pass
    if row.get("market_cap_usd"):
        try:
            mcap = float(row["market_cap_usd"])
            if mcap > 0:
                parts.append(f"mcap=${mcap:,.0f}")
        except Exception:
            pass
    return ", ".join(parts) if parts else "Blockscout enrichment indicates legitimate"


def run_migration(conn: sqlite3.Connection, fp_rows: list[dict], apply: bool) -> dict:
    """Migrate each LIKELY_FP contract from confirmed → unanalyzed,
    preserving original confidence_reason with audit annotation prepended."""
    counts = {
        "candidates": len(fp_rows),
        "found_in_db_as_confirmed": 0,
        "already_migrated": 0,
        "not_in_db": 0,
        "tier_mismatch": 0,
        "would_migrate": 0,
        "migrated": 0,
    }

    for r in fp_rows:
        addr = r["contract_address"]
        ann = AUDIT_ANNOTATION.format(evidence_short=evidence_short(r))

        existing = conn.execute(
            "SELECT confidence_tier, confidence_reason FROM contracts WHERE contract_address=?",
            (addr,)
        ).fetchone()
        if not existing:
            counts["not_in_db"] += 1
            continue

        cur_tier, cur_reason = existing[0], existing[1] or ""
        if cur_tier == "unanalyzed" and "[AUDIT 2026-05-22" in cur_reason:
            counts["already_migrated"] += 1
            continue
        if cur_tier != "confirmed":
            counts["tier_mismatch"] += 1
            continue

        counts["found_in_db_as_confirmed"] += 1
        counts["would_migrate"] += 1

        if not apply:
            continue

        new_reason = ann + cur_reason
        conn.execute(
            "UPDATE contracts SET confidence_tier='unanalyzed', "
            "confidence_reason=?, last_updated=? "
            "WHERE contract_address=?",
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

    print(f"=== Phase D migration ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    print(f"  CSV: {csv_path}")
    print(f"  DB:  {db_path}")
    fp_rows = load_fp_candidates(csv_path)
    print(f"  {len(fp_rows)} LIKELY_FP candidates from CSV")

    conn = sqlite3.connect(db_path)
    try:
        counts = run_migration(conn, fp_rows, apply=args.apply)
    finally:
        conn.close()

    print()
    for k, v in counts.items():
        print(f"  {k:32s}: {v:>6,}")

    if args.apply:
        print()
        print(f"  Migrated {counts['migrated']} contracts from confirmed → unanalyzed.")
    else:
        print()
        print("  (Dry run — no rows modified. Add --apply to commit.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
