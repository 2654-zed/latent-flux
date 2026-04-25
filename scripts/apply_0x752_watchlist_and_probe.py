"""Task 1: Add 0x80b12bd0 deployer to watchlist HIGH (local + Railway).
Task 3: Score sibling 0xda42fe397 via risk_scoring.score_contract.
Task 4: Investigate the 24 deployer-victims of 0x752c5a95.

Task 2 (funder probe) is a Railway-only /admin/eth-trace call — separate.
"""
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
sys.path.insert(0, str(Path(r"C:\Users\jason\Desktop\ai lang")))

TARGET_CONTRACT = "0x752c5a95d202972e124390f30a50154409d3c858"
DEPLOYER = "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8"
FUNDER = "0x153e9afc5e6a1166e6cc89d8493f162c45e5cc62"
SIBLING = "0xda42fe397c3fc9d08ac6675eecd2709880fdfd73"

now = datetime.now(timezone.utc).isoformat()

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

# =========================================================================
# TASK 1: Add deployer to watchlist HIGH
# =========================================================================
print("=" * 70)
print("TASK 1: Add deployer 0x80b12bd0 to watchlist HIGH")
print("=" * 70)

reason = (
    "2026-04-24 investigator review: deployer of 0x752c5a95 "
    "(confirmed pre-drain contract, 1,898 standing approvals, 0 drains). "
    "2019-05-23 mainnet vintage provides behavioral-laundering cover. "
    "Two-contract Base footprint defeats fleet detectors. Watch for "
    "sweep-tx firing against 0x752c5a95 and any further deploys."
)

c.execute("""
    INSERT OR REPLACE INTO watchlist
      (address, address_type, entity_name, watch_reason, priority,
       added_date, hit_count, active)
    VALUES (?, 'deployer', 'pristine-reputation solo operator (0x752c5a95 deployer)',
            ?, 'HIGH', ?, 0, 1)
""", (DEPLOYER.lower(), reason, now))
c.commit()
print(f"  local: inserted {DEPLOYER}")

# Verify
for r in c.execute(
    "SELECT address, priority, entity_name FROM watchlist WHERE priority='HIGH' "
    "ORDER BY added_date DESC LIMIT 5"):
    print(f"  HIGH: {r['address']}  {r['entity_name']}")

# =========================================================================
# TASK 3: Score sibling 0xda42fe397
# =========================================================================
print()
print("=" * 70)
print("TASK 3: Risk-score sibling 0xda42fe397")
print("=" * 70)

try:
    from surveillance.risk_scoring import score_contract
    res = score_contract(c, SIBLING)
    print(f"  chain: base")
    print(f"  address: {SIBLING}")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
    else:
        for k, v in res.items():
            if k != "components":
                print(f"  {k}: {v}")
        if "components" in res:
            print(f"  --- components (methodology-internal) ---")
            import json
            for comp, detail in res["components"].items():
                print(f"    {comp}: {json.dumps(detail)[:120]}")
except Exception as e:
    print(f"  SCORING FAILED: {type(e).__name__}: {e}")

# Supplementary: what does the sibling look like in-corpus?
print()
print("  sibling contract record:")
r = c.execute("""
    SELECT contract_address, confidence_tier, detection_method,
           has_asymmetric_transfer, has_conditional_revert, has_unusual_fee_structure,
           detection_timestamp
    FROM contracts WHERE LOWER(contract_address) = ?
""", (SIBLING.lower(),)).fetchone()
if r:
    for k in r.keys():
        print(f"    {k}: {r[k]}")

# sibling traffic
print()
print("  sibling traffic since deploy:")
t = c.execute("""
    SELECT COUNT(*) AS n, COUNT(DISTINCT interacting_address) AS ds,
           SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS rev
    FROM transaction_events WHERE LOWER(contract_address) = ?
""", (SIBLING.lower(),)).fetchone()
print(f"    tx_events={t['n']} distinct_senders={t['ds']} reverts={t['rev']}")

t_app = c.execute(
    "SELECT COUNT(*) FROM approval_watchlist WHERE LOWER(contract_address) = ?",
    (SIBLING.lower(),)).fetchone()[0]
print(f"    approvals_on_record={t_app}")

# =========================================================================
# TASK 4: 24 deployer-victims — who are they?
# =========================================================================
print()
print("=" * 70)
print("TASK 4: Investigate 24 deployer-victims")
print("=" * 70)

rows = c.execute("""
    SELECT DISTINCT LOWER(aw.victim_address) AS addr, aw.approve_timestamp
    FROM approval_watchlist aw
    WHERE LOWER(aw.contract_address) = ?
""", (TARGET_CONTRACT.lower(),)).fetchall()
victims = {r["addr"]: r["approve_timestamp"] for r in rows}

deployer_victims = []
for v, ts in victims.items():
    d = c.execute("""
        SELECT deployer_address, chain, first_seen, total_contracts_deployed,
               mainnet_first_tx,
               json_extract(funding_trail, '$.funder') AS funder
        FROM deployers WHERE LOWER(deployer_address) = ?
    """, (v,)).fetchone()
    if d:
        # Count their fleet tiers
        fleet = c.execute("""
            SELECT SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END) AS conf,
                   SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END) AS sus,
                   COUNT(*) AS total
            FROM contracts WHERE LOWER(deployer_address) = ?
        """, (v,)).fetchone()
        deployer_victims.append({
            "address": v,
            "approve_ts": ts,
            "chain": d["chain"],
            "first_seen": d["first_seen"],
            "fleet": d["total_contracts_deployed"] or 0,
            "confirmed": fleet["conf"] or 0,
            "suspected": fleet["sus"] or 0,
            "mainnet": d["mainnet_first_tx"],
            "funder": (d["funder"] or "").lower() if d["funder"] else None,
        })

print(f"\n  confirmed count: {len(deployer_victims)} deployer-victims found\n")
print(f"  {'address':<44} {'approve_ts':<20} {'fleet':<6} {'conf':<5} {'sus':<4} {'mainnet':<11} {'funder':<44}")
for dv in sorted(deployer_victims, key=lambda x: x["approve_ts"]):
    mn = (dv["mainnet"] or "")[:10] or "-"
    fdr = (dv["funder"] or "-")[:44]
    print(f"  {dv['address']}  {dv['approve_ts'][:19]}  {dv['fleet']:<6} "
          f"{dv['confirmed']:<5} {dv['suspected']:<4} {mn:<11} {fdr}")

# Do any share a funder?
print()
print("  funder concentration (2+ deployer-victims sharing a funder):")
from collections import Counter
funders = Counter(dv["funder"] for dv in deployer_victims if dv["funder"])
for fdr, n in funders.most_common():
    if n >= 2:
        print(f"    funder={fdr}  count={n}")
        for dv in deployer_victims:
            if dv["funder"] == fdr:
                print(f"      → {dv['address']}  mainnet={dv['mainnet'][:10] if dv['mainnet'] else '-'}  fleet={dv['fleet']}")

# Does the 0x752c5a95 funder (0x153e9afc) match any of these victim-deployer funders?
if FUNDER.lower() in funders:
    print(f"\n  ★ MATCH: victim-deployer(s) share funder with 0x752c5a95 operator!")
    print(f"    funder 0x153e9afc funds both the deployer AND {funders[FUNDER.lower()]} victim-deployer(s)")

# Do any deployer-victims belong to org_candidates?
print()
print("  deployer-victims also in org_candidates (cluster membership):")
for dv in deployer_victims:
    cands = c.execute("""
        SELECT candidate_id, cluster_size FROM org_candidates
        WHERE deployer_addresses LIKE '%' || ? || '%'
    """, (dv["address"],)).fetchall()
    for cand in cands:
        print(f"    {dv['address']}  →  {cand['candidate_id']} (size={cand['cluster_size']})")

c.close()
print()
print("=" * 70)
print("DONE (local). Railway sync + funder eth-trace next.")
print("=" * 70)
