"""Corpus state analysis — 2026-04-24.

Beyond the daily delta: look at distributions, coverage, and anything off-pattern.
"""
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

# =====================================================================
h("1. Corpus size & tier distribution")
# =====================================================================
total = c.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
print(f"  contracts:           {total:>10,}")
print(f"  deployers:           {c.execute('SELECT COUNT(*) FROM deployers').fetchone()[0]:>10,}")
print(f"  trap_events:         {c.execute('SELECT COUNT(*) FROM trap_events').fetchone()[0]:>10,}")
print(f"  alerts (non-FP):     {c.execute('SELECT COUNT(*) FROM alerts WHERE COALESCE(false_positive,0)=0').fetchone()[0]:>10,}")
print(f"  approval_watchlist:  {c.execute('SELECT COUNT(*) FROM approval_watchlist').fetchone()[0]:>10,}")
print(f"  transaction_events:  {c.execute('SELECT COUNT(*) FROM transaction_events').fetchone()[0]:>10,}")

print()
print("  contracts by confidence_tier:")
for r in c.execute("""
    SELECT confidence_tier, COUNT(*) AS n FROM contracts
    GROUP BY confidence_tier ORDER BY n DESC
"""):
    pct = 100.0 * r['n'] / total
    print(f"    {r['confidence_tier'] or '(null)':<12} {r['n']:>8,}  ({pct:5.2f}%)")

# =====================================================================
h("2. Per-chain split")
# =====================================================================
print(f"  {'chain':<10} {'contracts':>10} {'confirmed':>10} {'suspected':>10} {'recent_24h':>12}")
for r in c.execute("""
    SELECT chain,
      COUNT(*) AS total,
      SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END) AS conf,
      SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END) AS sus,
      SUM(CASE WHEN detection_timestamp > datetime('now', '-1 day') THEN 1 ELSE 0 END) AS recent
    FROM contracts GROUP BY chain ORDER BY total DESC
"""):
    print(f"  {r['chain'] or '(null)':<10} {r['total']:>10,} {r['conf']:>10,} {r['sus']:>10,} {r['recent']:>12,}")

# =====================================================================
h("3. Trap-event velocity (7-day rollup)")
# =====================================================================
print("  date         events  uniq_traps  uniq_bots  uniq_deployers")
for r in c.execute("""
    SELECT substr(timestamp, 1, 10) AS d,
           COUNT(*) AS n,
           COUNT(DISTINCT trap_contract_address) AS traps,
           COUNT(DISTINCT bot_address) AS bots,
           COUNT(DISTINCT (SELECT deployer_address FROM contracts ct
                          WHERE LOWER(ct.contract_address) = LOWER(te.trap_contract_address))) AS deps
    FROM trap_events te
    WHERE timestamp > datetime('now', '-7 days')
    GROUP BY d ORDER BY d
"""):
    print(f"  {r['d']}   {r['n']:>5}  {r['traps']:>10}  {r['bots']:>9}  {r['deps']:>14}")

# =====================================================================
h("4. Top trap-deployers — last 7 days")
# =====================================================================
print("  deployer                                      n   fleet  conf  rate  first_seen   mainnet")
for r in c.execute("""
    SELECT ct.deployer_address, COUNT(*) AS n,
      (SELECT COUNT(*) FROM contracts WHERE deployer_address = ct.deployer_address) AS fleet,
      (SELECT COUNT(*) FROM contracts WHERE deployer_address = ct.deployer_address
         AND confidence_tier='confirmed') AS conf,
      (SELECT first_seen FROM deployers WHERE deployer_address = ct.deployer_address) AS first_seen,
      (SELECT mainnet_first_tx FROM deployers WHERE deployer_address = ct.deployer_address) AS mn
    FROM trap_events te
    JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
    WHERE te.timestamp > datetime('now', '-7 days')
    GROUP BY ct.deployer_address ORDER BY n DESC LIMIT 12
"""):
    rate = (r['conf'] / r['fleet'] * 100) if r['fleet'] else 0
    mn = (r['mn'] or '-')[:10] or '-'
    print(f"  {r['deployer_address']}  {r['n']:>3}  {r['fleet']:>5}  {r['conf']:>4}  {rate:>4.0f}%  "
          f"{r['first_seen'][:10]}   {mn}")

# =====================================================================
h("5. Org coverage")
# =====================================================================
print("  org_wallets:")
for r in c.execute("SELECT org_id, COUNT(*) AS n FROM org_wallets GROUP BY org_id ORDER BY org_id"):
    print(f"    {r['org_id']:<12} {r['n']:>4} wallets")

print()
print("  org_candidates by status / cluster size:")
for r in c.execute("""
    SELECT status, COUNT(*) AS n,
           SUM(CASE WHEN cluster_size <= 5 THEN 1 ELSE 0 END) AS small,
           SUM(CASE WHEN cluster_size > 5 AND cluster_size <= 15 THEN 1 ELSE 0 END) AS mid,
           SUM(CASE WHEN cluster_size > 15 THEN 1 ELSE 0 END) AS big
    FROM org_candidates GROUP BY status
"""):
    print(f"    {r['status']:<12} total={r['n']:>4}  small(<=5)={r['small']:>4}  mid(6-15)={r['mid']:>4}  big(>15)={r['big']:>3}")

print()
try:
    print("  solo_operator_candidates by classification:")
    for r in c.execute("""
        SELECT classification, COUNT(*) AS n,
               AVG(fleet_size) AS avg_fleet, MAX(fleet_size) AS max_fleet,
               SUM(confirmed_count) AS total_conf
        FROM solo_operator_candidates GROUP BY classification
    """):
        print(f"    {r['classification']:<25} n={r['n']:>3}  avg_fleet={r['avg_fleet']:.1f}  "
              f"max_fleet={r['max_fleet']}  total_confirmed={r['total_conf']}")
except sqlite3.OperationalError:
    print("    (table not present locally)")

# =====================================================================
h("6. Detection coverage — Pattern D (cross-chain reputation import)")
# =====================================================================
# Deployers with confirmed/suspected fleet AND old mainnet history
print("  Confirmed-tier deployers with mainnet_first_tx >= 1 year before first_seen on L2:")
for r in c.execute("""
    SELECT d.deployer_address, d.chain, d.first_seen, d.mainnet_first_tx,
           d.total_contracts_deployed AS fleet,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
              AND confidence_tier='confirmed') AS conf
    FROM deployers d
    WHERE d.mainnet_first_tx IS NOT NULL AND d.mainnet_first_tx != ''
      AND CAST(julianday(d.first_seen) - julianday(d.mainnet_first_tx) AS INT) > 365
      AND (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
             AND confidence_tier='confirmed') >= 1
    ORDER BY conf DESC LIMIT 15
"""):
    gap_days = "?"
    try:
        from datetime import datetime
        a = datetime.fromisoformat(r['first_seen'].replace('Z', '+00:00').replace('+00:00', ''))
        b = datetime.fromisoformat(r['mainnet_first_tx'].replace('Z', '+00:00').replace('+00:00', ''))
        gap_days = f"{(a-b).days}d"
    except Exception:
        pass
    print(f"  {r['deployer_address']}  fleet={r['fleet']:<4} confirmed={r['conf']:<3} "
          f"L2_first={r['first_seen'][:10]}  mainnet={r['mainnet_first_tx'][:10]}  gap={gap_days}")

# =====================================================================
h("7. Pristine-solo gap (the 0x752c5a95 class)")
# =====================================================================
# Small fleet (<=5) + has confirmed + old mainnet
print("  Confirmed-tier operators with fleet <=5 AND old mainnet (>1y), invisible to both detectors:")
for r in c.execute("""
    SELECT d.deployer_address, d.chain, d.first_seen, d.mainnet_first_tx,
           d.total_contracts_deployed AS fleet,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
              AND confidence_tier='confirmed') AS conf,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
              AND confidence_tier='suspected') AS sus
    FROM deployers d
    WHERE d.total_contracts_deployed BETWEEN 1 AND 5
      AND d.mainnet_first_tx IS NOT NULL AND d.mainnet_first_tx != ''
      AND CAST(julianday(d.first_seen) - julianday(d.mainnet_first_tx) AS INT) > 365
      AND (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
             AND confidence_tier='confirmed') >= 1
    ORDER BY (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
              AND confidence_tier='confirmed') DESC,
             julianday(d.first_seen) - julianday(d.mainnet_first_tx) DESC
    LIMIT 20
"""):
    from datetime import datetime
    try:
        a = datetime.fromisoformat(r['first_seen'].replace('Z','').replace('+00:00',''))
        b = datetime.fromisoformat(r['mainnet_first_tx'].replace('Z','').replace('+00:00',''))
        gap = f"{(a-b).days}d"
    except Exception:
        gap = "?"
    print(f"  {r['deployer_address']}  fleet={r['fleet']} conf={r['conf']} sus={r['sus']}  "
          f"L2={r['first_seen'][:10]}  mn={r['mainnet_first_tx'][:10]}  gap={gap}")

# =====================================================================
h("8. Active approval-harvest scenarios (live exposures)")
# =====================================================================
print("  Confirmed contracts with the most standing approvals (zero drains so far):")
for r in c.execute("""
    SELECT aw.contract_address, aw.contract_tier,
           COUNT(*) AS approvals,
           COUNT(DISTINCT aw.victim_address) AS victims,
           SUM(aw.drain_detected) AS drains,
           MIN(aw.approve_timestamp) AS first_approve,
           MAX(aw.approve_timestamp) AS last_approve
    FROM approval_watchlist aw
    WHERE aw.contract_tier = 'confirmed'
    GROUP BY aw.contract_address
    HAVING drains = 0 AND approvals >= 50
    ORDER BY approvals DESC LIMIT 10
"""):
    print(f"  {r['contract_address']}  approvals={r['approvals']:>4}  victims={r['victims']:>4}  "
          f"first={r['first_approve'][:10]}  last={r['last_approve'][:10]}")

# =====================================================================
h("9. Camouflage state (last 7 days)")
# =====================================================================
for r in c.execute("""
    SELECT date, chain, total_active_contracts, camouflaged_count, camouflage_ratio,
           adversary_total_contracts, adversary_low_revert_count, adversary_low_revert_ratio
    FROM camouflage_metrics
    WHERE date > datetime('now', '-7 days')
    ORDER BY date DESC LIMIT 10
"""):
    pop_r = f"{r['camouflage_ratio']:.3f}" if r['camouflage_ratio'] is not None else '-'
    adv_r = f"{r['adversary_low_revert_ratio']:.3f}" if r['adversary_low_revert_ratio'] is not None else '-'
    print(f"  {r['date'][:10]} chain={r['chain']:<10} pop_active={r['total_active_contracts']:<5} "
          f"camo={r['camouflaged_count']:<4} pop_ratio={pop_r}  "
          f"adv_active={r['adversary_total_contracts']:<5} adv_low={r['adversary_low_revert_count']:<4} adv_ratio={adv_r}")

# =====================================================================
h("10. Watchlist HIGH — recent hit activity")
# =====================================================================
for r in c.execute("""
    SELECT wl.address, wl.entity_name,
           wl.priority, wl.added_date,
           (SELECT COUNT(*) FROM watchlist_hits WHERE watchlist_id = wl.id) AS total_hits,
           (SELECT COUNT(*) FROM watchlist_hits WHERE watchlist_id = wl.id
              AND timestamp > datetime('now', '-7 days')) AS recent_hits,
           (SELECT MAX(timestamp) FROM watchlist_hits WHERE watchlist_id = wl.id) AS last_hit
    FROM watchlist wl
    WHERE wl.priority='HIGH' AND wl.active=1
    ORDER BY wl.added_date DESC LIMIT 10
"""):
    last = (r['last_hit'] or '-')[:19]
    print(f"  {r['address']}  total={r['total_hits']:>4}  recent_7d={r['recent_hits']:>3}  "
          f"last={last}  added={r['added_date'][:10]}")

# =====================================================================
h("11. 0x752c5a95 watch — has the drain fired yet?")
# =====================================================================
TARGET = "0x752c5a95d202972e124390f30a50154409d3c858"
total = c.execute(
    "SELECT COUNT(*) FROM approval_watchlist WHERE LOWER(contract_address)=?",
    (TARGET.lower(),)).fetchone()[0]
drained = c.execute(
    "SELECT COUNT(*) FROM approval_watchlist WHERE LOWER(contract_address)=? AND drain_detected=1",
    (TARGET.lower(),)).fetchone()[0]
recent = c.execute("""
    SELECT COUNT(*) FROM approval_watchlist
    WHERE LOWER(contract_address)=? AND approve_timestamp > datetime('now', '-1 day')
""", (TARGET.lower(),)).fetchone()[0]
print(f"  total approvals: {total}")
print(f"  drains detected: {drained}")
print(f"  approvals last 24h: {recent}")
# Any new tx events on the contract?
recent_tx = c.execute("""
    SELECT COUNT(*) AS n,
           SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS rev,
           COUNT(DISTINCT interacting_address) AS senders
    FROM transaction_events
    WHERE LOWER(contract_address) = ? AND timestamp > datetime('now', '-1 day')
""", (TARGET.lower(),)).fetchone()
print(f"  tx_events last 24h: {recent_tx['n']}  reverts={recent_tx['rev']}  unique_senders={recent_tx['senders']}")

c.close()
