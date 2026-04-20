"""One-shot probe: dump contracts schema and row counts from Railway DB."""
import sqlite3
import os
from pathlib import Path

candidates = [
    Path("data/surveillance.db"),
    Path("/data/surveillance.db"),
    Path("surveillance/data/surveillance.db"),
    Path("/app/data/surveillance.db"),
    Path("/app/surveillance/data/surveillance.db"),
]
found = None
for p in candidates:
    if p.exists():
        found = p
        break
if not found:
    print("NO DB at any expected path")
    print("cwd:", Path.cwd())
    print("listing:", [str(x) for x in Path.cwd().iterdir()][:30])
    raise SystemExit(1)

print(f"db: {found}  size={found.stat().st_size:,} bytes")
c = sqlite3.connect(str(found))
cols = [(r[1], r[2]) for r in c.execute("PRAGMA table_info(contracts)").fetchall()]
print(f"contracts columns ({len(cols)}):")
for n, t in cols:
    print(f"  {n:<28} {t}")
ct = c.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
print(f"contracts rows: {ct:,}")

sql = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='contracts'").fetchone()[0]
print()
print("CHECK constraint has 'unanalyzed':", "'unanalyzed'" in sql)
print()
print("has decayed_at column:", any(n == "decayed_at" for n, _ in cols))
print("has deployed_code_hash column:", any(n == "deployed_code_hash" for n, _ in cols))
print("has prior_confidence_tier column:", any(n == "prior_confidence_tier" for n, _ in cols))

# Also check if there's an orphaned contracts_new
orphan = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contracts_new'").fetchone()
print("contracts_new exists:", orphan is not None)
if orphan:
    nr = c.execute("SELECT COUNT(*) FROM contracts_new").fetchone()[0]
    print(f"  contracts_new rows: {nr:,}")
    ncols = [r[1] for r in c.execute("PRAGMA table_info(contracts_new)").fetchall()]
    print(f"  contracts_new columns: {ncols}")
