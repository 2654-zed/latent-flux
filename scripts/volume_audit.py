"""Railway volume audit — what's eating the space.

Reports:
  - Disk free/used via df
  - Per-table row counts + estimated byte sizes (via SQLite's dbstat)
  - Per-table composition as % of DB
  - Growth vs local (if local is accessible)
  - Identified reclaimable candidates (WAL, shm, corrupt leftovers, old rows)
"""
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

for p in (Path("/app/surveillance/data/surveillance.db"),
          Path("surveillance/data/surveillance.db")):
    if p.exists():
        DB = p
        break

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

print("=" * 72)
print("  Volume audit")
print("=" * 72)
print()
print(f"DB:   {DB}")
print(f"Size: {DB.stat().st_size:,} bytes  ({DB.stat().st_size / (1024**3):.2f} GB)")
print()

# List files next to the DB
data_dir = DB.parent
print(f"## Files in {data_dir}")
for f in sorted(data_dir.iterdir()):
    try:
        s = f.stat()
        print(f"  {s.st_size:>14,}  {f.name}  (mod {datetime.fromtimestamp(s.st_mtime).strftime('%Y-%m-%d %H:%M')})")
    except Exception:
        print(f"  (stat failed) {f.name}")
print()

# Page size + total pages — gives us the DB footprint
ps = c.execute("PRAGMA page_size").fetchone()[0]
pc = c.execute("PRAGMA page_count").fetchone()[0]
fc = c.execute("PRAGMA freelist_count").fetchone()[0]
print(f"## DB structure")
print(f"  page_size        = {ps:,} bytes")
print(f"  page_count       = {pc:,}  ({pc * ps / (1024**3):.2f} GB)")
print(f"  freelist_count   = {fc:,}  ({fc * ps / (1024**2):.1f} MB reclaimable via VACUUM)")
print()

# Per-table estimated sizes via dbstat (SQLite extension, usually available)
print("## Per-table row counts + estimated sizes")
tables = [r[0] for r in c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
).fetchall()]

rows = []
for t in tables:
    try:
        n = c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    except Exception:
        continue
    # Try dbstat — if unavailable, fall back to a heuristic
    bytes_estimate = None
    try:
        est = c.execute(
            "SELECT SUM(pgsize) FROM dbstat WHERE name = ?", (t,)
        ).fetchone()
        if est and est[0] is not None:
            bytes_estimate = est[0]
    except Exception:
        pass
    rows.append((t, n, bytes_estimate))

# Sort by size if available, else by row count
rows.sort(key=lambda r: (r[2] or 0, r[1]), reverse=True)
total_bytes_est = sum(r[2] or 0 for r in rows)
print(f"  (dbstat total: {total_bytes_est:,} bytes = {total_bytes_est/(1024**3):.2f} GB)")
print()
print(f"  {'table':<32} {'rows':>12} {'bytes':>14} {'% of DB':>8}  {'B/row':>8}")
print(f"  {'-'*32} {'-'*12} {'-'*14} {'-'*8}  {'-'*8}")
for t, n, b in rows:
    if b is None and n == 0:
        continue
    pct = (100 * b / total_bytes_est) if b and total_bytes_est else 0
    bpr = (b / n) if (b and n) else 0
    b_str = f"{b:,}" if b else "?"
    print(f"  {t:<32} {n:>12,} {b_str:>14} {pct:>7.1f}% {bpr:>8.0f}")
print()

# Timestamp of oldest row in big tables (are we accumulating unbounded history?)
print("## Temporal boundaries of top tables")
for t in ["transaction_events", "alerts", "org_transfer_events",
          "x402_events", "bot_candidate_events", "liquidity_events",
          "approval_events", "dormant_activations"]:
    try:
        min_ts = c.execute(f"SELECT MIN(timestamp) FROM [{t}]").fetchone()[0]
        max_ts = c.execute(f"SELECT MAX(timestamp) FROM [{t}]").fetchone()[0]
        n = c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        print(f"  {t:<32} n={n:>10,}  range={min_ts} -> {max_ts}")
    except Exception as e:
        print(f"  {t:<32} err: {e}")
print()

# Old-row retention — are there rows >30 days old in the big event tables?
print("## >30-day-old row counts in event tables (retention candidates)")
for t in ["transaction_events", "alerts", "org_transfer_events",
          "x402_events", "bot_candidate_events", "liquidity_events",
          "approval_events", "dormant_activations"]:
    try:
        n = c.execute(
            f"SELECT COUNT(*) FROM [{t}] WHERE timestamp < datetime('now', '-30 days')"
        ).fetchone()[0]
        total = c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        pct = (100 * n / total) if total else 0
        print(f"  {t:<32} {n:>10,}/{total:,} ({pct:.1f}%) older than 30 days")
    except Exception as e:
        print(f"  {t:<32} err: {e}")
print()

# Disk usage (Railway only)
try:
    df = subprocess.check_output(["df", "-h", str(data_dir)], text=True)
    print("## df -h")
    for line in df.splitlines():
        print(f"  {line}")
except Exception as e:
    print(f"  df failed: {e}")
print()

c.close()
