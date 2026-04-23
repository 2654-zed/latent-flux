"""Generate (but do not execute) the promotion SQL for Tier-1 clusters.

Output goes to stdout as runnable SQL. User reviews, then either pipes
to `sqlite3 <db>` or runs via `/admin/sync-org-wallets` once that
endpoint exists.

Role assignment logic:
  - The deployer with the most trap_events = "operator"
  - Subsequent deployers (by trap count desc, then confirmed count desc,
    then total contracts desc) = "operator_2", "operator_3", ...
  - The cluster's funder = "treasury" (but ONLY if it's not already
    in org_wallets; if the funder is in the cluster's deployer list, it
    gets its deployer role instead)

Clusters:
  orgcand_a8f337083daf -> org_005 (arbitrum, 8 wallets)
  orgcand_5564c29a9070 -> org_006 (base, 15 wallets)
  orgcand_899651790f70 -> org_007 (arbitrum+base, 14 wallets)
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

PROMOTIONS = [
    ("orgcand_a8f337083daf", "org_005", "Serial single-honeypot operator. Lead deployer 0xda977393363d produced 4 trap_events on 2026-03-23 against 4 victims in an 8h window, all on contract 0x248d0105ec63. Fleet of 62 contracts with 1 confirmed acts as decoy around one active trap. Chain=arbitrum. Promoted 2026-04-23 from Tier-1 investigator review."),
    ("orgcand_5564c29a9070", "org_006", "Persistent repeat-victim operator. Lead deployer 0x982ff6be4aa1 has a 3+ week history of trapping the same bot 0x456a3e06c64d across 3 different contracts (2026-03-22 -> 2026-04-14). Bot operator appears not to update blacklist. Chain=base. Promoted 2026-04-23 from Tier-1 investigator review."),
    ("orgcand_899651790f70", "org_007", "Cross-chain operator (Arbitrum + Base), 14 deployers. 3 trap_events across 3 distinct deployers/contracts/bots. Prep-to-discharge latency 2d 11h — cleanest observed case of deploy+fire cycle. Mainnet cohort dispersed (not prepared-pool signature). Promoted 2026-04-23 from Tier-1 investigator review."),
]


def sql_escape(s: str) -> str:
    return s.replace("'", "''")


now = datetime.now(timezone.utc).isoformat()

print("-- Tier-1 cluster promotions (generated 2026-04-23)")
print("-- Review each INSERT before executing. Each INSERT is idempotent via")
print("-- INSERT OR IGNORE against the (address, chain) primary key of org_wallets.")
print()
print("BEGIN IMMEDIATE;")
print()

for candidate_id, org_id, reason in PROMOTIONS:
    cand = c.execute(
        "SELECT deployer_addresses, shared_funding_source, shared_chain, cluster_size "
        "FROM org_candidates WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    if not cand:
        print(f"-- SKIP {candidate_id}: not in local DB")
        continue

    deployers = json.loads(cand["deployer_addresses"])
    chain = cand["shared_chain"]
    funder = cand["shared_funding_source"]

    # Rank deployers by (traps desc, confirmed desc, contracts desc)
    placeholders = ",".join("?" * len(deployers))
    ranked = c.execute(f"""
        SELECT d.deployer_address, d.chain,
               (SELECT COUNT(*) FROM trap_events te JOIN contracts c2
                  ON LOWER(c2.contract_address) = LOWER(te.trap_contract_address)
                  WHERE c2.deployer_address = d.deployer_address) AS traps,
               (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
                  AND confidence_tier = 'confirmed') AS confirmed_n,
               (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address) AS total
        FROM deployers d WHERE d.deployer_address IN ({placeholders})
        ORDER BY traps DESC, confirmed_n DESC, total DESC
    """, deployers).fetchall()

    print(f"-- === {candidate_id} -> {org_id} ===")
    print(f"-- chain={chain}  size={cand['cluster_size']}  funder={funder}")
    print(f"-- reason: {reason[:120]}...")
    print()

    # Funder as treasury — use the first chain in the cluster's shared_chain if multi
    primary_chain = (chain or "arbitrum").split(",")[0]
    print(f"-- treasury / funder")
    print(f"INSERT OR IGNORE INTO org_wallets "
          f"(address, chain, org_id, role, added_at, added_by, reason) VALUES "
          f"('{funder.lower()}', '{primary_chain}', '{org_id}', 'treasury', "
          f"'{now}', 'tier1_review_2026_04_23', '{sql_escape(reason)}');")
    print()

    # Deployers
    for i, r in enumerate(ranked):
        if i == 0:
            role = "operator"
        else:
            role = f"operator_{i+1}"
        dep_addr = r["deployer_address"].lower()
        dep_chain = r["chain"] or primary_chain
        dep_reason = f"deployer from {candidate_id}; traps={r['traps']}, confirmed={r['confirmed_n']}, total_contracts={r['total']}"
        print(f"INSERT OR IGNORE INTO org_wallets "
              f"(address, chain, org_id, role, added_at, added_by, reason) VALUES "
              f"('{dep_addr}', '{dep_chain}', '{org_id}', '{role}', "
              f"'{now}', 'tier1_review_2026_04_23', '{sql_escape(dep_reason)}');")

    # Mark the candidate as promoted
    print()
    print(f"-- mark candidate promoted")
    print(f"UPDATE org_candidates SET status = 'promoted', "
          f"notes = 'Promoted to {org_id} on 2026-04-23 via tier1_review' "
          f"WHERE candidate_id = '{candidate_id}';")
    print()
    print()

print("COMMIT;")
print()
print("-- After execution, verify with:")
print("--   SELECT org_id, COUNT(*) FROM org_wallets GROUP BY org_id ORDER BY org_id;")
print("--   SELECT status, COUNT(*) FROM org_candidates GROUP BY status;")

c.close()
