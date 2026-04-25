"""Task 4 only: investigate the 24 deployer-victims of 0x752c5a95."""
import sqlite3
from collections import Counter
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
TARGET_CONTRACT = "0x752c5a95d202972e124390f30a50154409d3c858"
FUNDER = "0x153e9afc5e6a1166e6cc89d8493f162c45e5cc62"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

rows = c.execute("""
    SELECT DISTINCT LOWER(aw.victim_address) AS addr,
           MIN(aw.approve_timestamp) AS approve_ts
    FROM approval_watchlist aw
    WHERE LOWER(aw.contract_address) = ?
    GROUP BY LOWER(aw.victim_address)
""", (TARGET_CONTRACT.lower(),)).fetchall()
victims = {r["addr"]: r["approve_ts"] for r in rows}

deployer_victims = []
for v, ts in victims.items():
    d = c.execute("""
        SELECT deployer_address, chain, first_seen, total_contracts_deployed,
               mainnet_first_tx,
               json_extract(funding_trail, '$.funder') AS funder
        FROM deployers WHERE LOWER(deployer_address) = ?
    """, (v,)).fetchone()
    if d:
        fleet = c.execute("""
            SELECT COALESCE(SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END), 0) AS conf,
                   COALESCE(SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END), 0) AS sus,
                   COUNT(*) AS total
            FROM contracts WHERE LOWER(deployer_address) = ?
        """, (v,)).fetchone()
        deployer_victims.append({
            "address": v,
            "approve_ts": ts,
            "first_seen": d["first_seen"],
            "fleet": d["total_contracts_deployed"] or 0,
            "confirmed": fleet["conf"],
            "suspected": fleet["sus"],
            "mainnet": d["mainnet_first_tx"],
            "funder": (d["funder"] or "").lower() if d["funder"] else None,
        })

print(f"{len(deployer_victims)} deployer-victims found\n")
print(f"{'address':<44} {'approve':<12} {'fleet':<6} {'conf':<5} {'sus':<4} {'mainnet':<11} {'funder'}")
print("-" * 140)
for dv in sorted(deployer_victims, key=lambda x: x["approve_ts"]):
    mn = (dv["mainnet"] or "")[:10] or "-"
    fdr = (dv["funder"] or "-")[:44]
    print(f"{dv['address']}  {dv['approve_ts'][:10]}  {dv['fleet']:<6} "
          f"{dv['confirmed']:<5} {dv['suspected']:<4} {mn:<11} {fdr}")

print()
print("=== Funder concentration among deployer-victims ===\n")
funders = Counter(dv["funder"] for dv in deployer_victims if dv["funder"])
for fdr, n in funders.most_common():
    if n >= 2:
        print(f"  funder={fdr}  count={n}")
        for dv in deployer_victims:
            if dv["funder"] == fdr:
                print(f"    -> {dv['address']}  mainnet={(dv['mainnet'] or '-')[:10]}  "
                      f"fleet={dv['fleet']} confirmed={dv['confirmed']} suspected={dv['suspected']}")

print()
# cross-check against 0x752 operator's funder
if FUNDER.lower() in funders:
    print(f"!! MATCH: victim-deployer(s) share funder with 0x752c5a95 operator!")
    print(f"  funder 0x153e9afc funds both 0x80b12bd0 AND {funders[FUNDER.lower()]} victim-deployer(s)")
else:
    print(f"(no victim-deployer shares funder 0x153e9afc with the 0x752 operator)")

print()
print("=== Deployer-victims in any org_candidate cluster ===\n")
matches = 0
for dv in deployer_victims:
    cands = c.execute("""
        SELECT candidate_id, cluster_size, shared_funding_source FROM org_candidates
        WHERE deployer_addresses LIKE '%' || ? || '%'
    """, (dv["address"],)).fetchall()
    for cand in cands:
        print(f"  {dv['address']}  ->  {cand['candidate_id']} (size={cand['cluster_size']})")
        matches += 1
if matches == 0:
    print("  (none — these victim-deployers are not in any existing cluster)")

print()
print("=== Self-funded (funder == deployer) victims ===\n")
self_funded = [dv for dv in deployer_victims if dv["funder"] == dv["address"]]
for dv in self_funded:
    print(f"  {dv['address']}  mainnet={(dv['mainnet'] or '-')[:10]}  fleet={dv['fleet']}")
print(f"  total self-funded: {len(self_funded)}")

print()
print("=== Confirmed or suspected trap deployers among victim-deployers ===\n")
flagged = [dv for dv in deployer_victims if dv["confirmed"] > 0 or dv["suspected"] > 0]
for dv in flagged:
    print(f"  {dv['address']}  confirmed={dv['confirmed']}  suspected={dv['suspected']}  fleet={dv['fleet']}  "
          f"mainnet={(dv['mainnet'] or '-')[:10]}")
print(f"  total: {len(flagged)} of {len(deployer_victims)} deployer-victims have flagged contracts themselves")

c.close()
