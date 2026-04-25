"""Deep investigation of 0x752c5a95d202972e124390f30a50154409d3c858.
47 new approvals from 47 distinct victims in ~12h to a tier=confirmed contract.
Goal: understand what this contract is, who is granting approvals, and whether a drain is imminent or already occurred.
"""
import json
import sqlite3
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
ADDR = "0x752c5a95d202972e124390f30a50154409d3c858"

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

def h(s): print(f"\n=== {s} ===\n")

h("1. Contract identity")
r = c.execute("""
    SELECT contract_address, chain, detection_method, confidence_tier,
           detection_timestamp, decayed_at, prior_confidence_tier,
           deployed_code_hash, has_asymmetric_transfer, has_conditional_revert,
           has_unusual_fee_structure, deployer_address
    FROM contracts WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
if r:
    for k in r.keys():
        print(f"  {k}: {r[k]}")
else:
    print("  NOT FOUND IN CONTRACTS TABLE")

h("2. Deployer profile")
if r and r["deployer_address"]:
    d = c.execute("""
        SELECT deployer_address, chain, first_seen, last_seen,
               total_contracts_deployed, mainnet_first_tx, entity_type,
               json_extract(funding_trail, '$.funder') AS funder
        FROM deployers WHERE LOWER(deployer_address) = ?
    """, (r["deployer_address"].lower(),)).fetchone()
    if d:
        for k in d.keys():
            print(f"  {k}: {d[k]}")
        # fleet stats
        fleet = c.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END) AS confirmed,
                   SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END) AS suspected,
                   SUM(CASE WHEN confidence_tier='unanalyzed' THEN 1 ELSE 0 END) AS unanalyzed
            FROM contracts WHERE LOWER(deployer_address) = ?
        """, (r["deployer_address"].lower(),)).fetchone()
        print(f"  fleet_breakdown: total={fleet['total']} confirmed={fleet['confirmed']} "
              f"suspected={fleet['suspected']} unanalyzed={fleet['unanalyzed']}")
    else:
        print("  deployer not in deployers table")

h("3. Is this deployer in any org or cluster?")
dep = r["deployer_address"].lower() if r else None
if dep:
    ow = c.execute(
        "SELECT org_id, role, chain, reason FROM org_wallets WHERE LOWER(address) = ?",
        (dep,)).fetchall()
    for o in ow:
        print(f"  ORG: {o['org_id']}:{o['role']} reason={o['reason']}")
    if not ow:
        print("  not in org_wallets")
    # check org_candidates
    cand = c.execute("""
        SELECT candidate_id, cluster_size, shared_funding_source, shared_chain, status
        FROM org_candidates
        WHERE deployer_addresses LIKE '%' || ? || '%'
    """, (dep,)).fetchall()
    for o in cand:
        print(f"  CANDIDATE: {o['candidate_id']} size={o['cluster_size']} "
              f"funder={o['shared_funding_source']} status={o['status']}")
    if not cand:
        print("  not in any org_candidates")
    # solo_operator_candidates
    try:
        solo = c.execute(
            "SELECT * FROM solo_operator_candidates WHERE LOWER(deployer_address) = ?",
            (dep,)).fetchone()
        if solo:
            print(f"  SOLO: classification={solo['classification']} fleet={solo['fleet_size']} "
                  f"confirmed={solo['confirmed_count']} status={solo['status']}")
        else:
            print("  not in solo_operator_candidates")
    except sqlite3.OperationalError:
        pass

h("4. Bytecode family membership")
for r2 in c.execute("""
    SELECT bfm.family_id, bf.family_name, bf.member_count, bf.unique_deployers,
           bf.is_cross_deployer, bf.avg_revert_rate, bf.total_victims
    FROM bytecode_family_members bfm
    JOIN bytecode_families bf ON bf.family_id = bfm.family_id
    WHERE LOWER(bfm.contract_address) = ?
""", (ADDR.lower(),)):
    print(f"  family={r2['family_id']} ({r2['family_name']})")
    print(f"  members={r2['member_count']} deployers={r2['unique_deployers']} "
          f"cross_deployer={r2['is_cross_deployer']} avg_revert={r2['avg_revert_rate']} "
          f"total_victims={r2['total_victims']}")

h("5. Approvals — who are the 47 victims?")
apps = c.execute("""
    SELECT victim_address, approve_timestamp, approve_tx_hash,
           contract_tier, drain_detected, drain_tx_hash, drain_timestamp
    FROM approval_watchlist
    WHERE LOWER(contract_address) = ?
    ORDER BY approve_timestamp DESC
""", (ADDR.lower(),)).fetchall()
print(f"  total approvals on record: {len(apps)}")
drains = [a for a in apps if a["drain_detected"]]
print(f"  drains detected: {len(drains)}")
# time distribution
from collections import defaultdict
hour_counts = defaultdict(int)
for a in apps:
    hour_counts[a["approve_timestamp"][:13]] += 1
print("  approval hour distribution (recent):")
for h2, n in sorted(hour_counts.items())[-10:]:
    print(f"    {h2}  {n:<3} {'#' * n}")

print(f"\n  first 15 victims (most recent first):")
for a in apps[:15]:
    drain_flag = f" DRAINED at {a['drain_timestamp'][:19]}" if a["drain_detected"] else ""
    print(f"    {a['approve_timestamp'][:19]}  {a['victim_address']}{drain_flag}")

h("6. Are victims related? (overlap with deployers table, approvers repeat elsewhere)")
victims = {a["victim_address"].lower() for a in apps}
print(f"  unique victims: {len(victims)}")
# are any victims also deployers?
in_dep = 0
for v in victims:
    if c.execute("SELECT 1 FROM deployers WHERE LOWER(deployer_address) = ? LIMIT 1", (v,)).fetchone():
        in_dep += 1
print(f"  victims that are also deployers in our corpus: {in_dep}")
# are any victims also bots?
in_bot = 0
for v in victims:
    if c.execute("SELECT 1 FROM bot_candidates WHERE LOWER(address) = ? LIMIT 1", (v,)).fetchone():
        in_bot += 1
print(f"  victims that are also bot_candidates: {in_bot}")
# are any victims also known-org wallets?
in_org = 0
for v in victims:
    if c.execute("SELECT 1 FROM org_wallets WHERE LOWER(address) = ? LIMIT 1", (v,)).fetchone():
        in_org += 1
print(f"  victims that are also org_wallets: {in_org}")

h("7. Interactions with the contract (transaction_events)")
try:
    te = c.execute("""
        SELECT COUNT(*) AS n, MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen,
               COUNT(DISTINCT interacting_address) AS distinct_senders,
               SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS reverts
        FROM transaction_events
        WHERE LOWER(contract_address) = ?
    """, (ADDR.lower(),)).fetchone()
    print(f"  total tx events: {te['n']}")
    print(f"  first: {te['first_seen']}  last: {te['last_seen']}")
    print(f"  distinct senders: {te['distinct_senders']}")
    print(f"  reverts: {te['reverts']}")
    if te['n']:
        print(f"  revert_rate: {te['reverts']/te['n']:.3f}")
    # top selectors
    print("  top function selectors:")
    for ts in c.execute("""
        SELECT function_selector, COUNT(*) AS n FROM transaction_events
        WHERE LOWER(contract_address) = ? AND function_selector IS NOT NULL
        GROUP BY function_selector ORDER BY n DESC LIMIT 10
    """, (ADDR.lower(),)):
        print(f"    {ts['function_selector']}  n={ts['n']}")
except sqlite3.OperationalError as e:
    print(f"  transaction_events check failed: {e}")

h("8. Trap events against this contract")
for t in c.execute("""
    SELECT timestamp, bot_address, failure_signature, loss_estimate_usd, tx_hash
    FROM trap_events WHERE LOWER(trap_contract_address) = ?
    ORDER BY timestamp
""", (ADDR.lower(),)):
    print(f"  {t['timestamp'][:19]}  bot={t['bot_address']}  sig={t['failure_signature']}")

h("9. Alerts on this address")
for a in c.execute("""
    SELECT timestamp, alert_type, false_positive, substr(payload,1,120) AS pay
    FROM alerts WHERE LOWER(address) = ?
    ORDER BY timestamp DESC LIMIT 20
""", (ADDR.lower(),)):
    fp = " [FP]" if a["false_positive"] else ""
    print(f"  {a['timestamp'][:19]}  {a['alert_type']:<28}{fp}  {a['pay']}")

h("10. Trust amplification stats (routing)")
ta = c.execute("""
    SELECT total_callers, router_callers, router_percentage,
           amplification_factor, revert_rate, alert_level
    FROM trust_amplification WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
if ta:
    for k in ta.keys():
        print(f"  {k}: {ta[k]}")
else:
    print("  no trust_amplification row (not yet analyzed)")

h("11. False-positive audit check")
fp = c.execute("""
    SELECT fp_reason, fp_method, detector_blamed, assessed_at
    FROM false_positives WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
print(f"  false_positives row: {'YES' if fp else 'NO'}")
if fp:
    for k in fp.keys():
        print(f"    {k}: {fp[k]}")

h("12. Other contracts by same deployer — do they drain too?")
if dep:
    peers = c.execute("""
        SELECT contract_address, confidence_tier, detection_timestamp
        FROM contracts WHERE LOWER(deployer_address) = ?
        ORDER BY detection_timestamp DESC LIMIT 15
    """, (dep,)).fetchall()
    print(f"  recent contracts by same deployer (showing 15):")
    for p in peers:
        # count approvals against this peer
        appn = c.execute(
            "SELECT COUNT(*) FROM approval_watchlist WHERE LOWER(contract_address) = ?",
            (p["contract_address"].lower(),)).fetchone()[0]
        drn = c.execute(
            "SELECT COUNT(*) FROM approval_watchlist WHERE LOWER(contract_address) = ? AND drain_detected = 1",
            (p["contract_address"].lower(),)).fetchone()[0]
        print(f"    {p['contract_address']}  tier={p['confidence_tier']:<10} "
              f"detected={p['detection_timestamp'][:10]}  approvals={appn:<3} drains={drn}")

c.close()
