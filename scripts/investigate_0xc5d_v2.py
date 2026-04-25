"""Faster 0xc5d133296e investigation — pre-fetch contract list, then aggregate."""
import sqlite3
import json
from collections import Counter
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
ADDR = "0xc5d133296e17ba25df0409a6c31607bf3b78e3e3"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

# Pre-fetch contract list ONCE
contracts = [r[0].lower() for r in c.execute(
    "SELECT contract_address FROM contracts WHERE LOWER(deployer_address) = ?",
    (ADDR.lower(),)).fetchall()]
print(f"fleet contracts: {len(contracts)}")

if not contracts:
    raise SystemExit("no contracts")

# Build temp table for IN
c.execute("CREATE TEMP TABLE IF NOT EXISTS fleet_addrs(addr TEXT PRIMARY KEY)")
c.executemany("INSERT OR IGNORE INTO fleet_addrs VALUES (?)", [(a,) for a in contracts])

h("7. Fleet-aggregate traffic (efficient)")
agg = c.execute("""
    SELECT COUNT(*) AS n,
           COUNT(DISTINCT contract_address) AS contracts_with_traffic,
           COUNT(DISTINCT interacting_address) AS distinct_senders,
           SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS reverts,
           MIN(timestamp) AS first_tx, MAX(timestamp) AS last_tx
    FROM transaction_events
    WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
""").fetchone()
print(f"  total tx events on fleet: {agg['n']:,}")
print(f"  contracts with traffic: {agg['contracts_with_traffic']}")
print(f"  distinct senders: {agg['distinct_senders']:,}")
print(f"  reverts: {agg['reverts']}")
if agg['n']:
    print(f"  fleet revert_rate: {agg['reverts']/agg['n']:.4f}")
print(f"  first_tx: {agg['first_tx']}")
print(f"  last_tx: {agg['last_tx']}")

# Top selectors across fleet
print()
print("  Top function selectors across fleet:")
for r in c.execute("""
    SELECT function_selector, COUNT(*) AS n FROM transaction_events
    WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
      AND function_selector IS NOT NULL
    GROUP BY function_selector ORDER BY n DESC LIMIT 8
"""):
    print(f"    {r['function_selector']}  n={r['n']:,}")

h("8. Approval activity on fleet contracts")
ap = c.execute("""
    SELECT COUNT(*) AS approvals, COUNT(DISTINCT victim_address) AS victims,
           SUM(drain_detected) AS drains
    FROM approval_watchlist
    WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
""").fetchone()
print(f"  approvals: {ap['approvals']}  victims: {ap['victims']}  drains: {ap['drains']}")

h("9. Bytecode family membership across fleet")
fams = list(c.execute("""
    SELECT bf.family_id, bf.family_name, bf.member_count, bf.unique_deployers,
           bf.is_cross_deployer,
           COUNT(*) AS in_fleet
    FROM bytecode_family_members bfm
    JOIN bytecode_families bf ON bf.family_id = bfm.family_id
    WHERE LOWER(bfm.contract_address) IN (SELECT addr FROM fleet_addrs)
    GROUP BY bf.family_id ORDER BY in_fleet DESC LIMIT 8
"""))
for r in fams:
    print(f"  {r['family_id']}  members={r['member_count']:<5} deployers={r['unique_deployers']:<3} "
          f"cross={r['is_cross_deployer']}  in_fleet={r['in_fleet']}")

h("10. Trust amplification on fleet")
for r in c.execute("""
    SELECT contract_address, total_callers, router_callers, router_percentage,
           amplification_factor, revert_rate, alert_level
    FROM trust_amplification
    WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
    ORDER BY amplification_factor DESC LIMIT 8
"""):
    print(f"  {r['contract_address'][:14]}  callers={r['total_callers']:<5} "
          f"router%={r['router_percentage']:<5} amp={r['amplification_factor']:<5} "
          f"revert={r['revert_rate']:<6} alert={r['alert_level']}")

h("11. Watchlist hits — per-day distribution last 14 days")
for r in c.execute("""
    SELECT substr(wh.timestamp, 1, 10) AS d, COUNT(*) AS n
    FROM watchlist_hits wh
    JOIN watchlist wl ON wl.id = wh.watchlist_id
    WHERE LOWER(wl.address) = ? AND wh.timestamp > datetime('now', '-14 days')
    GROUP BY d ORDER BY d DESC
""", (ADDR.lower(),)):
    print(f"  {r['d']}  {r['n']:<3} {'#'*r['n']}")

c.close()
