"""Apply Tier-1 promotions, watchlist additions, and solo-operator detector
on Railway. Run via `railway ssh --service latent-flux`.
"""
import os
import sqlite3
import sys
from datetime import datetime, timezone

DB = "/app/surveillance/data/surveillance.db"
SQL_PATH = "/app/scripts/tier1_promotion.sql"

now = datetime.now(timezone.utc).isoformat()

c = sqlite3.connect(DB, timeout=60)
c.execute("PRAGMA busy_timeout=60000")

# 1. Tier-1 promotion SQL (forgiving encoding)
sql = open(SQL_PATH, "rb").read().decode("utf-8", errors="replace")
c.executescript(sql)
print("[1] tier1_promotion.sql: applied")

# 2. Watchlist entries
for addr, atype, name, reason in [
    ("0x666521000c595a632fb3e99f392b12e937b77586", "deployer",
     "high-productivity solo operator",
     "2026-04-23 investigator review: fleet 50, 6 confirmed, 4 hits today. 22-month Pattern D gap."),
    ("0xefef185e2c89bbede21a1c41427bdf1332eca392", "deployer",
     "high-confirmation-ratio operator",
     "2026-04-23 investigator review: fleet 15 with 9 confirmed (60% rate). 2 hits today."),
]:
    c.execute("""
        INSERT OR REPLACE INTO watchlist
          (address, address_type, entity_name, watch_reason, priority,
           added_date, hit_count, active)
        VALUES (?, ?, ?, ?, 'HIGH', ?, 0, 1)
    """, (addr.lower(), atype, name, reason, now))
c.commit()
print("[2] watchlist: 2 HIGH-priority entries added")

# 3. Verify
print()
print("=== org_wallets by org_id ===")
for r in c.execute("SELECT org_id, COUNT(*) FROM org_wallets GROUP BY org_id ORDER BY org_id"):
    print(f"  {r[0]:<10} {r[1]}")
print()
print("=== org_candidates by status ===")
for r in c.execute("SELECT status, COUNT(*) FROM org_candidates GROUP BY status"):
    print(f"  {r[0]:<12} {r[1]}")
print()
print("=== watchlist HIGH priority (top 3) ===")
for r in c.execute("SELECT address, entity_name FROM watchlist WHERE priority='HIGH' ORDER BY added_date DESC LIMIT 3"):
    print(f"  {r[0]}  {r[1]}")

c.close()
