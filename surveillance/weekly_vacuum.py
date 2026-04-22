"""Weekly VACUUM maintenance job.

Replaces the unconditional startup VACUUM with a smarter periodic one:
  - Checks freelist_count first; skips VACUUM if <5000 pages (~20 MB) reclaimable
  - Checks free disk; skips if free < 2x DB size (leaves safety margin for temp copy)
  - Runs PRAGMA wal_checkpoint(TRUNCATE) unconditionally (always cheap + useful)

Scheduled via run_surveillance._analysis_scheduler Sunday 05:00 UTC.

CLI:
    python -m surveillance.weekly_vacuum          # run if conditions met
    python -m surveillance.weekly_vacuum --force  # run regardless
    python -m surveillance.weekly_vacuum --dry-run
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

# Thresholds
MIN_FREELIST_PAGES = 5000      # skip if fewer than ~20 MB reclaimable
MIN_FREE_DISK_MULT = 2.0        # need free >= 2x db_size for safe VACUUM


def db_size_bytes() -> int:
    return DB_PATH.stat().st_size if DB_PATH.exists() else 0


def disk_free_bytes() -> int:
    return shutil.disk_usage(str(DB_PATH.parent)).free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Run VACUUM regardless of guards.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check guards; don't run VACUUM.")
    ap.add_argument("--db", type=str, default=str(DB_PATH))
    args = ap.parse_args()

    db_size = db_size_bytes()
    free = disk_free_bytes()
    print(f"[weekly_vacuum] db_size={db_size:,} free={free:,} "
          f"ratio={(free/max(db_size,1)):.1f}x", flush=True)

    # Always checkpoint WAL — cheap, keeps WAL small
    conn = sqlite3.connect(args.db, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    t0 = time.time()
    r = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    print(f"[weekly_vacuum] wal_checkpoint(TRUNCATE): busy={r[0]} "
          f"log_pages={r[1]} checkpointed={r[2]} in {time.time()-t0:.1f}s",
          flush=True)

    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    reclaimable_mb = freelist * page_size / (1024 * 1024)
    print(f"[weekly_vacuum] freelist_count={freelist:,} "
          f"(~{reclaimable_mb:.1f} MB reclaimable)", flush=True)

    should_vacuum = args.force
    reason = "forced" if args.force else ""
    if not args.force:
        if freelist < MIN_FREELIST_PAGES:
            reason = f"freelist {freelist} < threshold {MIN_FREELIST_PAGES}"
        elif free < db_size * MIN_FREE_DISK_MULT:
            reason = (f"free disk {free:,} < {MIN_FREE_DISK_MULT}x db_size "
                      f"({int(db_size * MIN_FREE_DISK_MULT):,})")
        else:
            should_vacuum = True
            reason = f"freelist {freelist} >= {MIN_FREELIST_PAGES} and disk headroom OK"

    print(f"[weekly_vacuum] decision: vacuum={should_vacuum}  reason: {reason}",
          flush=True)

    if args.dry_run:
        print("[weekly_vacuum] --dry-run; exiting", flush=True)
        conn.close()
        return

    if should_vacuum:
        t1 = time.time()
        try:
            conn.execute("VACUUM")
            # Post-VACUUM checkpoint to fold WAL changes into main file
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            new_size = db_size_bytes()
            freed = db_size - new_size
            print(f"[weekly_vacuum] VACUUM complete in {time.time()-t1:.1f}s  "
                  f"db {db_size:,} -> {new_size:,} (freed {freed:,} bytes "
                  f"= {freed/(1024*1024):.1f} MB)", flush=True)
        except sqlite3.OperationalError as e:
            print(f"[weekly_vacuum] VACUUM failed: {e}", flush=True)
            conn.close()
            sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
