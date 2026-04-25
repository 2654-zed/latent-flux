"""Post-sync look at what's interesting since 2026-04-23T20:00 UTC."""
import sqlite3
from pathlib import Path
from collections import defaultdict

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
CUTOFF = "2026-04-23T20:00:00"

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

print(f"=== 1. New trap_events since {CUTOFF} ===\n")
rows = c.execute("""
    SELECT te.timestamp, te.trap_contract_address, te.bot_address,
           te.failure_signature, te.loss_estimate_usd,
           ct.chain, ct.confidence_tier, ct.deployer_address
    FROM trap_events te
    LEFT JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
    WHERE te.timestamp > ?
    ORDER BY te.timestamp
""", (CUTOFF,)).fetchall()
print(f"{len(rows)} trap events")
for r in rows:
    print(f"  {r['timestamp'][:19]} {str(r['chain']):<9} trap={r['trap_contract_address']} "
          f"bot={r['bot_address']} tier={r['confidence_tier']}")

print("\n=== 2. Deployer concentration in those traps ===\n")
for r in c.execute("""
    SELECT ct.deployer_address, COUNT(*) AS n,
      (SELECT COUNT(*) FROM contracts WHERE deployer_address = ct.deployer_address) AS fleet,
      (SELECT COUNT(*) FROM contracts WHERE deployer_address = ct.deployer_address
         AND confidence_tier='confirmed') AS confirmed,
      (SELECT first_seen FROM deployers WHERE deployer_address = ct.deployer_address) AS first_seen,
      (SELECT mainnet_first_tx FROM deployers WHERE deployer_address = ct.deployer_address) AS mn
    FROM trap_events te
    JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
    WHERE te.timestamp > ?
    GROUP BY ct.deployer_address
    ORDER BY n DESC
""", (CUTOFF,)):
    mn = (r['mn'] or '')[:10] or '-'
    print(f"  dep={r['deployer_address']} n={r['n']} fleet={r['fleet']} confirmed={r['confirmed']} "
          f"first={(r['first_seen'] or '')[:10]} mainnet={mn}")

print("\n=== 3. Did the two watchlist HIGH entries fire? ===\n")
for watched in [
    "0x666521000c595a632fb3e99f392b12e937b77586",
    "0xefef185e2c89bbede21a1c41427bdf1332eca392",
]:
    # Check their fleet for new traps
    n = c.execute("""
        SELECT COUNT(*) FROM trap_events te
        JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
        WHERE LOWER(ct.deployer_address) = ? AND te.timestamp > ?
    """, (watched.lower(), CUTOFF)).fetchone()[0]
    # Check watchlist_hits via watchlist.address join
    hits = c.execute("""
        SELECT COUNT(*) FROM watchlist_hits wh
        JOIN watchlist wl ON wl.id = wh.watchlist_id
        WHERE LOWER(wl.address) = ? AND wh.timestamp > ?
    """, (watched.lower(), CUTOFF)).fetchone()[0]
    print(f"  {watched}  new_traps={n}  watchlist_hits={hits}")

print("\n=== 4. New org_candidates (20 new) ===\n")
for r in c.execute("""
    SELECT candidate_id, cluster_size, shared_funding_source, shared_chain,
           first_seen, status
    FROM org_candidates WHERE detected_at > ?
    ORDER BY cluster_size DESC LIMIT 20
""", (CUTOFF,)):
    print(f"  {r['candidate_id']}  size={r['cluster_size']:<3} chain={r['shared_chain']:<6} "
          f"funder={r['shared_funding_source'][:20] if r['shared_funding_source'] else '-'}  "
          f"first={(r['first_seen'] or '')[:10]}")

print("\n=== 5. Alert-type breakdown since cutoff ===\n")
alerts = c.execute("""
    SELECT alert_type, COUNT(*) AS n FROM alerts
    WHERE timestamp > ? AND COALESCE(false_positive, 0) = 0
    GROUP BY alert_type ORDER BY n DESC
""", (CUTOFF,)).fetchall()
for r in alerts:
    print(f"  {r['alert_type']:<32} {r['n']}")

print("\n=== 6. New timelock_countdowns (64 new — proxy upgrades scheduled?) ===\n")
for r in c.execute("""
    SELECT contract_address, chain, confidence_tier, activation_iso, status,
           deployer_address, bytecode_evidence
    FROM timelock_countdowns WHERE detected_at > ?
    ORDER BY activation_iso LIMIT 15
""", (CUTOFF,)):
    ev = (r['bytecode_evidence'] or '')[:40]
    print(f"  {r['contract_address'][:10]}... chain={r['chain']:<6} tier={r['confidence_tier'] or '-':<10} "
          f"activ={(r['activation_iso'] or '')[:16]} status={r['status']} dep={r['deployer_address'][:10]}  {ev}")

print("\n=== 7. New approval_watchlist entries — anyone granting to confirmed contracts? ===\n")
for r in c.execute("""
    SELECT aw.contract_address, aw.contract_tier, COUNT(*) AS approvals,
           COUNT(DISTINCT aw.victim_address) AS victims
    FROM approval_watchlist aw
    WHERE aw.approve_timestamp > ?
    GROUP BY aw.contract_address, aw.contract_tier
    HAVING aw.contract_tier = 'confirmed'
    ORDER BY approvals DESC LIMIT 10
""", (CUTOFF,)):
    print(f"  {r['contract_address']} tier={r['contract_tier']} "
          f"approvals={r['approvals']} victims={r['victims']}")

print("\n=== 8. Dormant activations in window ===\n")
for r in c.execute("""
    SELECT deployer_address, chain, fleet_size, fleet_active_before, fleet_active_after,
           first_interaction_timestamp, first_interacting_address
    FROM dormant_activations WHERE detected_at > ?
    ORDER BY fleet_size DESC LIMIT 10
""", (CUTOFF,)):
    print(f"  dep={r['deployer_address']} chain={r['chain']:<6} fleet={r['fleet_size']:<4} "
          f"active_before={r['fleet_active_before']:<3} active_after={r['fleet_active_after']:<3} "
          f"first_call_by={r['first_interacting_address'][:10] if r['first_interacting_address'] else '-'}")

print("\n=== 9. Bytecode families — new cross-deployer ones? ===\n")
for r in c.execute("""
    SELECT family_id, family_name, member_count, unique_deployers,
           is_cross_deployer, avg_revert_rate, total_victims
    FROM bytecode_families
    WHERE last_updated > ? AND is_cross_deployer = 1
      AND unique_deployers >= 3
    ORDER BY total_victims DESC, member_count DESC LIMIT 10
""", (CUTOFF,)):
    print(f"  {r['family_id']} size={r['member_count']:<4} deployers={r['unique_deployers']:<3} "
          f"victims={r['total_victims']:<3} revert={r['avg_revert_rate']:.2f}  {r['family_name']}")

print("\n=== 10. solo_operator_candidates — did yesterday's new module write anything? ===\n")
try:
    rows = c.execute("""
        SELECT classification, COUNT(*) AS n FROM solo_operator_candidates
        GROUP BY classification
    """).fetchall()
    for r in rows:
        print(f"  {r['classification']:<30} {r['n']}")
except sqlite3.OperationalError:
    print("  (table doesn't exist locally — only on Railway)")

c.close()
