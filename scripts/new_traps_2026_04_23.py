"""Investigate the 20 new trap_events since 2026-04-23 01:43 UTC."""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
CUTOFF = "2026-04-23T01:43:39"

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

# Full per-trap listing
rows = c.execute("""
    SELECT te.id, te.timestamp, te.trap_contract_address, te.bot_address,
           te.tx_hash, te.loss_estimate_usd, te.failure_signature,
           c.chain, c.confidence_tier, c.deployer_address, c.detection_timestamp
    FROM trap_events te
    LEFT JOIN contracts c ON LOWER(c.contract_address) = LOWER(te.trap_contract_address)
    WHERE te.timestamp > ?
    ORDER BY te.timestamp
""", (CUTOFF,)).fetchall()

print(f"=== {len(rows)} new trap_events since {CUTOFF} ===\n")
print(f"{'#':<3} {'ts':<20} {'chain':<10} {'trap':<44} {'bot':<44} {'sig'}")
print("-" * 160)
for i, r in enumerate(rows, 1):
    sig = (r["failure_signature"] or "")[:30]
    print(f"{i:<3} {r['timestamp'][:19]:<20} {str(r['chain']):<10} "
          f"{r['trap_contract_address']:<44} {r['bot_address']:<44} {sig}")

# Chain split
print()
print("## By chain")
chain_counts = defaultdict(int)
for r in rows:
    chain_counts[r["chain"] or "unknown"] += 1
for ch, n in sorted(chain_counts.items(), key=lambda x: -x[1]):
    print(f"  {ch:<12} {n}")

# Time clustering — are these spread out or bursty?
print()
print("## Time distribution (hourly)")
hour_counts = defaultdict(int)
for r in rows:
    hour = r["timestamp"][:13]  # YYYY-MM-DDTHH
    hour_counts[hour] += 1
for h, n in sorted(hour_counts.items()):
    bar = "#" * n
    print(f"  {h}  {n:<3} {bar}")

# Deployer concentration
print()
print("## Deployer concentration (top 10)")
dep_rows = c.execute("""
    SELECT c.deployer_address, COUNT(*) AS n_traps,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = c.deployer_address) AS total_contracts,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = c.deployer_address
              AND confidence_tier = 'confirmed') AS confirmed,
           (SELECT first_seen FROM deployers WHERE deployer_address = c.deployer_address) AS first_seen,
           (SELECT mainnet_first_tx FROM deployers WHERE deployer_address = c.deployer_address) AS mainnet_tx
    FROM contracts c
    JOIN trap_events te ON LOWER(te.trap_contract_address) = LOWER(c.contract_address)
    WHERE te.timestamp > ?
    GROUP BY c.deployer_address
    ORDER BY n_traps DESC, confirmed DESC LIMIT 10
""", (CUTOFF,)).fetchall()
for r in dep_rows:
    mn = (r["mainnet_tx"] or "")[:10] or "-"
    print(f"  {r['deployer_address']}  "
          f"n={r['n_traps']}  fleet={r['total_contracts']:<4} confirmed={r['confirmed']:<3}  "
          f"first={r['first_seen'][:10]}  mainnet={mn}")

# Same-contract repeats (one honeypot catching many bots)
print()
print("## Contracts with multiple trap_events in this window")
for r in c.execute("""
    SELECT trap_contract_address, COUNT(*) AS n,
           GROUP_CONCAT(DISTINCT bot_address) AS bots
    FROM trap_events WHERE timestamp > ?
    GROUP BY trap_contract_address HAVING n >= 2
    ORDER BY n DESC
""", (CUTOFF,)):
    print(f"  {r['trap_contract_address']}  hits={r['n']}  distinct_bots={len(r['bots'].split(','))}")

# Victim bot concentration
print()
print("## Victim bots with multiple hits")
for r in c.execute("""
    SELECT bot_address, COUNT(*) AS n
    FROM trap_events WHERE timestamp > ?
    GROUP BY bot_address HAVING n >= 2
    ORDER BY n DESC
""", (CUTOFF,)):
    print(f"  {r['bot_address']}  hits={r['n']}")

# Any traps from known-org deployers?
print()
print("## Traps from known-org deployers")
for r in c.execute("""
    SELECT ow.org_id, ow.role, ow.address, COUNT(*) AS n_traps
    FROM trap_events te
    JOIN contracts c2 ON LOWER(c2.contract_address) = LOWER(te.trap_contract_address)
    JOIN org_wallets ow ON LOWER(ow.address) = LOWER(c2.deployer_address)
    WHERE te.timestamp > ?
    GROUP BY ow.org_id, ow.role, ow.address
    ORDER BY n_traps DESC
""", (CUTOFF,)):
    print(f"  {r['org_id']}:{r['role']}  deployer={r['address']}  n={r['n_traps']}")

# Any traps from candidates we haven't promoted yet?
print()
print("## Traps from deployers in pending org_candidates")
for r in c.execute("""
    SELECT oc.candidate_id, oc.cluster_size, oc.shared_funding_source,
           c2.deployer_address, COUNT(*) AS n_traps
    FROM trap_events te
    JOIN contracts c2 ON LOWER(c2.contract_address) = LOWER(te.trap_contract_address)
    JOIN org_candidates oc ON oc.deployer_addresses LIKE '%' || LOWER(c2.deployer_address) || '%'
    WHERE te.timestamp > ? AND oc.status = 'pending'
    GROUP BY oc.candidate_id, c2.deployer_address
    ORDER BY n_traps DESC
""", (CUTOFF,)):
    print(f"  {r['candidate_id']}  deployer={r['deployer_address']}  n={r['n_traps']}")

c.close()
