"""
One-shot backfill: delete stale bytecode_cache rows whose stored tier no
longer matches their source_contract's current tier in the contracts table.

Introduced 2026-04-17 alongside the cache-invalidation fix in
db.update_contract_confidence and db.insert_trap_event. Prior to that fix,
any post-insert mutation of confidence_tier left the cache entry stamped
with the old label, causing future cache-hit deployments to inherit stale
classifications.

Usage:
    python -m surveillance.backfill_cache_invalidation              # dry-run
    python -m surveillance.backfill_cache_invalidation --commit     # execute

The dry-run prints the breakdown (cached_tier × current_tier) and totals.
--commit deletes only rows where the tiers differ; entries whose source
is missing from contracts are left untouched (separate orphan-cleanup).
"""
import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "data" / "surveillance.db"

STALE_QUERY = """
SELECT bc.confidence_tier AS cached_tier,
       c.confidence_tier  AS current_tier,
       COUNT(*)           AS n,
       SUM(bc.hit_count)  AS hits
FROM bytecode_cache bc
JOIN contracts c ON c.contract_address = bc.source_contract
WHERE bc.confidence_tier != c.confidence_tier
GROUP BY bc.confidence_tier, c.confidence_tier
ORDER BY n DESC
"""

DELETE_QUERY = """
DELETE FROM bytecode_cache
WHERE source_contract IN (
  SELECT bc.source_contract
  FROM bytecode_cache bc
  JOIN contracts c ON c.contract_address = bc.source_contract
  WHERE bc.confidence_tier != c.confidence_tier
)
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH),
                    help="SQLite path (default: surveillance/data/surveillance.db)")
    ap.add_argument("--commit", action="store_true",
                    help="Actually perform the DELETE (default: dry-run)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print(f"DB: {db_path}  ({db_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Mode: {'COMMIT' if args.commit else 'dry-run'}")
    print("=" * 72)

    total_rows = conn.execute("SELECT COUNT(*) FROM bytecode_cache").fetchone()[0]
    print(f"bytecode_cache rows total: {total_rows:,}")
    print()
    print("Stale entries (cache tier != source's current tier):")
    rows = conn.execute(STALE_QUERY).fetchall()
    total_stale = total_hits = 0
    for r in rows:
        print(f"  cached={r['cached_tier']:10s}  current={r['current_tier']:10s}  "
              f"entries={r['n']:>6,}  downstream_lookups={r['hits'] or 0:>5,}")
        total_stale += r["n"]
        total_hits += r["hits"] or 0
    print(f"  {'TOTAL':>10s}  {'':10s}  entries={total_stale:>6,}  "
          f"downstream_lookups={total_hits:>5,}")

    if not args.commit:
        print()
        print(f"Dry-run complete. Re-run with --commit to delete {total_stale:,} rows.")
        return 0

    if total_stale == 0:
        print("Nothing to do.")
        return 0

    print()
    cur = conn.execute(DELETE_QUERY)
    deleted = cur.rowcount
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM bytecode_cache").fetchone()[0]
    print(f"Deleted {deleted:,} rows. bytecode_cache now has {remaining:,} rows "
          f"(was {total_rows:,}).")

    # Verify
    still_stale = conn.execute(
        "SELECT COUNT(*) FROM bytecode_cache bc "
        "JOIN contracts c ON c.contract_address = bc.source_contract "
        "WHERE bc.confidence_tier != c.confidence_tier"
    ).fetchone()[0]
    print(f"Residual stale entries after delete: {still_stale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
