"""Analyze the 324 novel-org candidates surfaced by Railway's detector.

Uses the local DB (which is now synced with Railway per delta_sync).
"""
import json
import sqlite3
from collections import Counter
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

print("=" * 72)
print("  org_candidates analysis")
print("=" * 72)
print()

total = c.execute("SELECT COUNT(*) FROM org_candidates").fetchone()[0]
print(f"Total candidates: {total}")
print()

# Size distribution
print("## Size distribution")
for r in c.execute("""
    SELECT CASE
        WHEN cluster_size < 5 THEN '3-4'
        WHEN cluster_size < 10 THEN '5-9'
        WHEN cluster_size < 20 THEN '10-19'
        WHEN cluster_size < 30 THEN '20-29'
        ELSE '30+'
    END AS bucket, COUNT(*) AS n
    FROM org_candidates GROUP BY bucket ORDER BY MIN(cluster_size)
"""):
    print(f"  size {r['bucket']:<6} {r['n']:,}")
print()

# Chain distribution
print("## Chain distribution")
for r in c.execute("""
    SELECT shared_chain, COUNT(*) AS n
    FROM org_candidates GROUP BY shared_chain ORDER BY n DESC
"""):
    print(f"  {str(r['shared_chain']):<30} {r['n']:,}")
print()

# Timing span
print("## Detection timing")
min_d = c.execute("SELECT MIN(detected_at) FROM org_candidates").fetchone()[0]
max_d = c.execute("SELECT MAX(detected_at) FROM org_candidates").fetchone()[0]
print(f"  detected range: {min_d} -> {max_d}")
min_fs = c.execute("SELECT MIN(first_seen) FROM org_candidates").fetchone()[0]
max_fs = c.execute("SELECT MAX(first_seen) FROM org_candidates").fetchone()[0]
print(f"  first_seen range: {min_fs} -> {max_fs}")
print()

# Top 20 by size
print("## Top 20 candidates by cluster_size")
print(f"  {'candidate_id':<22} {'size':>4}  {'chain':<22} {'funder':<44} {'span'}")
print(f"  {'-'*22} {'-'*4}  {'-'*22} {'-'*44} {'-'*21}")
top = c.execute("""
    SELECT candidate_id, cluster_size, shared_chain, shared_funding_source,
           first_seen, last_seen, deployer_addresses, status
    FROM org_candidates ORDER BY cluster_size DESC LIMIT 20
""").fetchall()
for r in top:
    span = f"{r['first_seen'][:10]}..{r['last_seen'][:10]}"
    print(f"  {r['candidate_id']:<22} {r['cluster_size']:>4}  {str(r['shared_chain']):<22} {str(r['shared_funding_source']):<44} {span}")
print()

# Funders that appear in multiple clusters (strong signal of structural reuse)
print("## Funders appearing in multiple candidates")
for r in c.execute("""
    SELECT shared_funding_source, COUNT(*) AS n_clusters,
           SUM(cluster_size) AS total_deployers
    FROM org_candidates
    WHERE shared_funding_source IS NOT NULL AND shared_funding_source != ''
    GROUP BY shared_funding_source
    HAVING n_clusters >= 2
    ORDER BY n_clusters DESC
    LIMIT 15
"""):
    print(f"  {str(r['shared_funding_source']):<44} clusters={r['n_clusters']:<3} total_deployers={r['total_deployers']}")
print()

# For the top 5, enumerate member deployers and see if any have confirmed traps
print("## Top 5 clusters — what do their deployers look like?")
for r in top[:5]:
    print(f"\n  === {r['candidate_id']} (size={r['cluster_size']}, chain={r['shared_chain']}) ===")
    try:
        deployers = json.loads(r["deployer_addresses"])
    except Exception:
        deployers = []
    if not deployers:
        print("    (no deployers in payload)")
        continue
    placeholders = ",".join("?" * len(deployers))

    # Confidence tier breakdown across all contracts from these deployers
    tiers = c.execute(f"""
        SELECT confidence_tier, COUNT(*) AS n
        FROM contracts
        WHERE deployer_address IN ({placeholders})
        GROUP BY confidence_tier
    """, deployers).fetchall()
    tier_str = " ".join(f"{t['confidence_tier']}={t['n']}" for t in tiers)
    total_contracts = sum(t['n'] for t in tiers)
    print(f"    {len(deployers)} deployers, {total_contracts} contracts total: {tier_str}")

    # Trap_events
    trap_n = c.execute(f"""
        SELECT COUNT(*) FROM trap_events te
        WHERE LOWER(te.trap_contract_address) IN (
            SELECT LOWER(contract_address) FROM contracts
            WHERE deployer_address IN ({placeholders})
        )
    """, deployers).fetchone()[0]
    print(f"    observable trap_events: {trap_n}")

    # Sample deployer with mainnet_first_tx
    pat_d = c.execute(f"""
        SELECT deployer_address, mainnet_first_tx, first_seen
        FROM deployers
        WHERE deployer_address IN ({placeholders})
          AND mainnet_first_tx IS NOT NULL AND mainnet_first_tx != ''
        ORDER BY mainnet_first_tx
        LIMIT 3
    """, deployers).fetchall()
    if pat_d:
        print(f"    Pattern D signals (earliest mainnet):")
        for p in pat_d:
            print(f"      {p['deployer_address']}  mainnet={p['mainnet_first_tx'][:10]}  l2_first={p['first_seen'][:10]}")

c.close()
