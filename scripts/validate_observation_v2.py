"""Validate observation_capability primitive against in-corpus high-EOA contracts."""
import sqlite3
import sys
from pathlib import Path

for p in (Path("/app/surveillance/data/surveillance.db"),
          Path("surveillance/data/surveillance.db")):
    if p.exists():
        DB = p
        break

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
from surveillance.risk_scoring import score_contract

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

# Pull top 10 contracts by distinct interacting EOAs — these are the
# contracts most "observed by" users in Layer 3's corpus
rows = c.execute("""
    SELECT contract_address, COUNT(DISTINCT interacting_address) AS n
    FROM transaction_events
    GROUP BY contract_address
    ORDER BY n DESC
    LIMIT 10
""").fetchall()

print(f"{'addr':<44} {'eoas':>6} {'tier':<10} {'obs':>4} {'stored':>7} {'risk':>8}")
print("-" * 95)
for r in rows:
    addr = r["contract_address"]
    eoas = r["n"]
    res = score_contract(c, addr)
    obs = res.get("observation_capability_score", 0)
    sp = res.get("stored_potential", 0)
    rs = res.get("risk_score", 0)
    tier = res.get("risk_tier", "?")
    print(f"{addr:<44} {eoas:>6} {tier:<10} {obs:>4} {sp:>7} {rs:>8.2f}")

# Also include an org_candidate for contrast
print()
print("=== contrast: one contract from a pending org_candidate cluster ===")
oc = c.execute(
    "SELECT deployer_addresses FROM org_candidates ORDER BY cluster_size DESC LIMIT 1"
).fetchone()
if oc:
    import json as _j
    try:
        deployers = _j.loads(oc["deployer_addresses"])
        if deployers:
            cr = c.execute(
                "SELECT contract_address FROM contracts WHERE deployer_address = ? LIMIT 1",
                (deployers[0],),
            ).fetchone()
            if cr:
                addr = cr["contract_address"]
                res = score_contract(c, addr)
                obs = res.get("observation_capability_score", 0)
                sp = res.get("stored_potential", 0)
                rs = res.get("risk_score", 0)
                tier = res.get("risk_tier", "?")
                print(f"{addr}  obs={obs}  stored={sp}  risk={rs:.2f}  tier={tier}")
    except Exception as e:
        print(f"  (contrast lookup failed: {e})")
c.close()
