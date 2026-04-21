"""Follow-ups a + b from the activity analysis.

(a) Did the 16 sibling suspected contracts from deployer 0xcadf9ebe...
    fire, decay, or stay dormant?
(b) Does funder 0x08ee48398b6390b44987ee5e9f1f1c73fbeeacbc fund other
    deployers besides 0xcadf9ebe...? (novel-org cluster candidate)
"""
import sqlite3
import sys
from pathlib import Path

for p in (Path("/app/surveillance/data/surveillance.db"),
          Path("surveillance/data/surveillance.db")):
    if p.exists():
        DB = p
        break
c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

DEPLOYER = "0xcadf9ebe57ce822cb4f2f36c514599f7b4f98154"
FUNDER = "0x08ee48398b6390b44987ee5e9f1f1c73fbeeacbc"

print("=" * 72)
print("  (a) Siblings of the confirmed-contract deployer")
print("=" * 72)
print(f"  deployer: {DEPLOYER}")
print()

siblings = c.execute("""
    SELECT c.contract_address, c.confidence_tier, c.detection_timestamp,
           c.decayed_at,
           (SELECT COUNT(*) FROM trap_events te
            WHERE LOWER(te.trap_contract_address) = LOWER(c.contract_address)) AS trap_hits,
           (SELECT COUNT(*) FROM transaction_events tx
            WHERE tx.contract_address = c.contract_address) AS tx_count,
           (SELECT COUNT(DISTINCT tx.interacting_address) FROM transaction_events tx
            WHERE tx.contract_address = c.contract_address) AS eoa_count
    FROM contracts c
    WHERE c.deployer_address = ?
    ORDER BY c.detection_timestamp
""", (DEPLOYER,)).fetchall()

print(f"  total contracts from deployer: {len(siblings)}")
print()
print(f"  {'#':<3} {'address':<44} {'tier':<12} {'trap':<5} {'tx':<4} {'eoa':<4} {'decayed':<20} detected")
print("  " + "-" * 120)
for i, s in enumerate(siblings, 1):
    dec = (s["decayed_at"] or "")[:19]
    print(f"  {i:<3} {s['contract_address']:<44} {s['confidence_tier']:<12} "
          f"{s['trap_hits']:<5} {s['tx_count']:<4} {s['eoa_count']:<4} "
          f"{dec:<20} {s['detection_timestamp'][:19]}")

# Summary
n_conf = sum(1 for s in siblings if s["confidence_tier"] == "confirmed")
n_susp = sum(1 for s in siblings if s["confidence_tier"] == "suspected")
n_dec = sum(1 for s in siblings if s["decayed_at"])
n_fired = sum(1 for s in siblings if s["trap_hits"] > 0)
n_any_tx = sum(1 for s in siblings if s["tx_count"] > 0)
print()
print(f"  SUMMARY: confirmed={n_conf}  suspected={n_susp}  decayed={n_dec}  "
      f"ever-fired={n_fired}  ever-called={n_any_tx}  dormant={len(siblings)-n_any_tx}")
print()

print("=" * 72)
print("  (b) Deployers funded by this funder")
print("=" * 72)
print(f"  funder: {FUNDER}")
print()

# Find other deployers whose funding_trail shows this funder
funded = c.execute("""
    SELECT deployer_address, chain, first_seen, last_seen,
           total_contracts_deployed, entity_type, funding_trail
    FROM deployers
    WHERE funding_trail LIKE ?
    ORDER BY first_seen
""", (f'%"funder": "{FUNDER}"%',)).fetchall()

print(f"  deployers funded by {FUNDER[:14]}...: {len(funded)}")
print()
if funded:
    print(f"  {'#':<3} {'deployer':<44} {'chain':<10} {'deployments':<4} entity  first_seen")
    print("  " + "-" * 100)
    for i, d in enumerate(funded, 1):
        print(f"  {i:<3} {d['deployer_address']:<44} {d['chain']:<10} "
              f"{d['total_contracts_deployed']:<11} "
              f"{(d['entity_type'] or ''):<8} {d['first_seen'][:19]}")
    print()

    # If 3+, this is a potential novel-org cluster
    if len(funded) >= 3:
        print(f"  >= 3 deployers share this funder -- NOVEL-ORG CLUSTER CANDIDATE")
        # Count contracts + confirmed hits across all deployers
        addrs = [d["deployer_address"] for d in funded]
        placeholders = ",".join("?" for _ in addrs)
        totals = c.execute(f"""
            SELECT COUNT(*) AS contracts,
                   SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END) AS confirmed,
                   SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END) AS suspected
            FROM contracts WHERE deployer_address IN ({placeholders})
        """, addrs).fetchone()
        print(f"    across all {len(funded)} deployers: "
              f"{totals['contracts']} contracts, "
              f"{totals['confirmed']} confirmed, {totals['suspected']} suspected")

        trap_total = c.execute(f"""
            SELECT COUNT(*) FROM trap_events te
            WHERE LOWER(te.trap_contract_address) IN (
                SELECT LOWER(contract_address) FROM contracts
                WHERE deployer_address IN ({placeholders})
            )
        """, addrs).fetchone()[0]
        print(f"    observable trap_events across cluster: {trap_total}")

# Is this funder already in org_candidates?
oc = c.execute(
    "SELECT candidate_id, cluster_size, status FROM org_candidates "
    "WHERE LOWER(shared_funding_source) = LOWER(?)",
    (FUNDER,),
).fetchone()
print()
if oc:
    print(f"  org_candidates entry: {oc['candidate_id']}  size={oc['cluster_size']}  status={oc['status']}")
else:
    print("  not yet in org_candidates (detector window / min-size may not match)")
c.close()
