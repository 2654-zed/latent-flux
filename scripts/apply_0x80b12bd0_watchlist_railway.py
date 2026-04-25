"""Add 0x80b12bd0 to Railway watchlist (HIGH). Run via railway ssh."""
import sqlite3
from datetime import datetime, timezone

DB = "/app/surveillance/data/surveillance.db"
DEPLOYER = "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8"

reason = (
    "2026-04-24 investigator review: deployer of 0x752c5a95 "
    "(confirmed pre-drain contract, 1,898 standing approvals, 0 drains). "
    "2019-05-23 mainnet vintage provides behavioral-laundering cover. "
    "Two-contract Base footprint defeats fleet detectors (both solo and cluster). "
    "Watch for sweep-tx firing against 0x752c5a95 and any further deploys."
)

now = datetime.now(timezone.utc).isoformat()
c = sqlite3.connect(DB, timeout=60)
c.execute("PRAGMA busy_timeout=60000")

c.execute("""
    INSERT OR REPLACE INTO watchlist
      (address, address_type, entity_name, watch_reason, priority,
       added_date, hit_count, active)
    VALUES (?, 'deployer', 'pristine-reputation solo operator (0x752c5a95 deployer)',
            ?, 'HIGH', ?, 0, 1)
""", (DEPLOYER.lower(), reason, now))
c.commit()
print(f"[railway] watchlist HIGH entry: {DEPLOYER}")

print()
print("=== Current HIGH watchlist (top 5 most recent) ===")
for r in c.execute("""
    SELECT address, entity_name FROM watchlist
    WHERE priority='HIGH' AND active=1
    ORDER BY added_date DESC LIMIT 5
"""):
    print(f"  {r[0]}  {r[1]}")

c.close()
