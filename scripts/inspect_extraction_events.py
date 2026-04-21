"""Dump extraction_events rows + columns for P5 planning."""
import sqlite3
import sys
from pathlib import Path

for p in (Path("/app/surveillance/data/surveillance.db"),
          Path("surveillance/data/surveillance.db")):
    if p.exists():
        DB = p
        break
c = sqlite3.connect(str(DB))
c.row_factory = sqlite3.Row

print("=== extraction_events schema ===")
for col in c.execute("PRAGMA table_info(extraction_events)").fetchall():
    print(f"  {col[1]:<25} {col[2]}")
print()
print("=== distinct event_type values ===")
for r in c.execute("SELECT event_type, COUNT(*) FROM extraction_events GROUP BY event_type"):
    print(f"  {r[0]:<50} {r[1]}")
print()
print("=== all rows (compact) ===")
for r in c.execute(
    "SELECT event_id, event_type, summary, total_usd_moved, chain, monitored_chain "
    "FROM extraction_events"
):
    print(f"  {r['event_id']:<18}  {r['event_type']:<44}  chain={r['chain']:<10} mon={r['monitored_chain']}  usd={r['total_usd_moved']}")
    print(f"     summary: {(r['summary'] or '')[:150]}")
c.close()
