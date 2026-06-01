"""Load the decode-leg cache into prod's audit_drain_legs from a gzipped
JSON passed via /tmp/legs.json.gz. Idempotent (INSERT OR REPLACE)."""
import gzip, json, sqlite3, sys
from datetime import datetime, timezone
DB = "/app/surveillance/data/surveillance.db"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
data = json.loads(gzip.open("/tmp/legs.json.gz", "rt").read())
c = sqlite3.connect(DB)
c.execute("""CREATE TABLE IF NOT EXISTS audit_drain_legs(
    victim TEXT, contract TEXT, n_out INTEGER, n_in INTEGER,
    truncated INTEGER, err TEXT, checked_at TEXT, PRIMARY KEY (victim, contract))""")
n = 0
for r in data:
    c.execute("INSERT OR REPLACE INTO audit_drain_legs VALUES (?,?,?,?,?,?,?)",
              (r["v"], r["c"], r["o"], r["i"], 0, r["e"], NOW))
    n += 1
c.commit()
print(f"loaded {n} leg rows into prod audit_drain_legs")
print("total rows now:", c.execute("SELECT COUNT(*) FROM audit_drain_legs").fetchone()[0])
