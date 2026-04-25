"""Check-in on 0xd4624228cce5baa0814c9e7f666a8a2c83b6f159 — the Uniswap routing parasite.

Last documented state: 14.2x trust amplification, 2,910 victims, 98.7%
router-delivered traffic. Cantina rejected as bounty submission 2026-03-25
('the code did what it was supposed to do').
"""
import sqlite3
import json
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
ADDR = "0xd4624228cce5baa0814c9e7f666a8a2c83b6f159"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

h("1. Contract state")
r = c.execute("""
    SELECT contract_address, chain, detection_method, confidence_tier,
           detection_timestamp, decayed_at, prior_confidence_tier,
           has_asymmetric_transfer, has_conditional_revert, has_unusual_fee_structure,
           deployer_address, last_updated
    FROM contracts WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
if r:
    for k in r.keys():
        print(f"  {k}: {r[k]}")
else:
    print("  NOT FOUND")
    raise SystemExit()

h("2. Deployer profile")
d = c.execute("""
    SELECT deployer_address, chain, first_seen, last_seen,
           total_contracts_deployed, mainnet_first_tx, entity_type,
           json_extract(funding_trail, '$.funder') AS funder
    FROM deployers WHERE LOWER(deployer_address) = ?
""", (r['deployer_address'].lower(),)).fetchone()
if d:
    for k in d.keys():
        print(f"  {k}: {d[k]}")

# Are they on watchlist?
w = c.execute(
    "SELECT priority, entity_name, watch_reason FROM watchlist WHERE LOWER(address) = ?",
    (r['deployer_address'].lower(),)).fetchone()
if w:
    print(f"  WATCHLIST: priority={w[0]} name={w[1]}")
    print(f"    reason: {w[2]}")

h("3. Trust amplification — current vs documented")
ta = c.execute("""
    SELECT total_callers, router_callers, router_percentage,
           amplification_factor, revert_rate, alert_level, last_updated
    FROM trust_amplification WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
if ta:
    print("  Current trust_amplification row:")
    for k in ta.keys():
        print(f"    {k}: {ta[k]}")
else:
    print("  No trust_amplification row")
print()
print("  Documented baseline (from lexicon): 14.2x amp, 2,910 victims, 98.7% router")

h("4. Traffic over time — has it grown / changed?")
agg = c.execute("""
    SELECT COUNT(*) AS n, COUNT(DISTINCT interacting_address) AS distinct_senders,
           SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS reverts,
           MIN(timestamp) AS first_tx, MAX(timestamp) AS last_tx
    FROM transaction_events WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
print(f"  total tx_events: {agg['n']:,}")
print(f"  distinct senders: {agg['distinct_senders']:,}")
print(f"  reverts: {agg['reverts']:,}")
if agg['n']:
    print(f"  revert_rate: {agg['reverts']/agg['n']:.4f}")
print(f"  first_tx: {agg['first_tx']}")
print(f"  last_tx: {agg['last_tx']}")

# Last 30d / 7d / 1d
print()
print("  Activity windows:")
for window, label in [("-1 day", "last 24h"), ("-7 days", "last 7d"), ("-30 days", "last 30d")]:
    r2 = c.execute(f"""
        SELECT COUNT(*) AS n, COUNT(DISTINCT interacting_address) AS senders,
               COALESCE(SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END), 0) AS rev
        FROM transaction_events WHERE LOWER(contract_address) = ?
          AND timestamp > datetime('now', ?)
    """, (ADDR.lower(), window)).fetchone()
    n = r2['n'] or 0
    rev = r2['rev'] or 0
    rate = rev/n if n else 0
    print(f"    {label:<10} n={n:<6} senders={r2['senders'] or 0:<5} reverts={rev:<5} rate={rate:.3f}")

# Per-day for last 30d
print()
print("  Per-day (last 14 days):")
for r2 in c.execute("""
    SELECT substr(timestamp,1,10) AS d, COUNT(*) AS n,
           COUNT(DISTINCT interacting_address) AS senders
    FROM transaction_events WHERE LOWER(contract_address) = ?
      AND timestamp > datetime('now', '-14 days')
    GROUP BY d ORDER BY d DESC LIMIT 14
""", (ADDR.lower(),)):
    bar = "#" * min(r2['n'] // 10, 50)
    print(f"    {r2['d']}  n={r2['n']:<5} senders={r2['senders']:<4} {bar}")

h("5. Top function selectors")
for r2 in c.execute("""
    SELECT function_selector, COUNT(*) AS n,
           SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS rev
    FROM transaction_events WHERE LOWER(contract_address) = ?
      AND function_selector IS NOT NULL AND function_selector != ''
    GROUP BY function_selector ORDER BY n DESC LIMIT 8
""", (ADDR.lower(),)):
    rate = r2['rev']/r2['n'] if r2['n'] else 0
    print(f"    {r2['function_selector']}  total={r2['n']:<6} reverts={r2['rev']:<5} rate={rate:.3f}")

h("6. Approval activity — Permit2 / approve()")
ap = c.execute("""
    SELECT COUNT(*) AS approvals, COUNT(DISTINCT victim_address) AS victims,
           SUM(drain_detected) AS drains,
           MIN(approve_timestamp) AS first_app, MAX(approve_timestamp) AS last_app
    FROM approval_watchlist WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
print(f"  total approvals: {ap['approvals']}")
print(f"  unique approvers: {ap['victims']}")
print(f"  drains detected: {ap['drains']}")
print(f"  approve range: {ap['first_app']} to {ap['last_app']}")

h("7. Trap events / direct harm")
te = c.execute("""
    SELECT COUNT(*) FROM trap_events WHERE LOWER(trap_contract_address) = ?
""", (ADDR.lower(),)).fetchone()[0]
print(f"  trap_events on this contract: {te}")
if te > 0:
    for r2 in c.execute("""
        SELECT timestamp, bot_address, failure_signature, loss_estimate_usd
        FROM trap_events WHERE LOWER(trap_contract_address) = ?
        ORDER BY timestamp DESC LIMIT 5
    """, (ADDR.lower(),)):
        print(f"    {r2['timestamp'][:19]} bot={r2['bot_address']} loss={r2['loss_estimate_usd']}")

h("8. Alert history")
for r2 in c.execute("""
    SELECT alert_type, timestamp, false_positive, substr(payload, 1, 100) AS p
    FROM alerts WHERE LOWER(address) = ?
    ORDER BY timestamp DESC LIMIT 15
""", (ADDR.lower(),)):
    fp = " [FP]" if r2['false_positive'] else ""
    print(f"  {r2['timestamp'][:19]}  {r2['alert_type']:<28}{fp}  {r2['p']}")

h("9. False-positive audit row?")
fp = c.execute("""
    SELECT fp_reason, fp_method, detector_blamed, assessed_at
    FROM false_positives WHERE LOWER(contract_address) = ?
""", (ADDR.lower(),)).fetchone()
if fp:
    for k in fp.keys():
        print(f"    {k}: {fp[k]}")
else:
    print("  No false_positive audit row.")

h("10. Bytecode family + risk score")
fams = list(c.execute("""
    SELECT bf.family_id, bf.family_name, bf.member_count, bf.is_cross_deployer
    FROM bytecode_family_members bfm
    JOIN bytecode_families bf ON bf.family_id = bfm.family_id
    WHERE LOWER(bfm.contract_address) = ?
""", (ADDR.lower(),)))
for f in fams[:5]:
    print(f"  family={f['family_id']} members={f['member_count']} cross={f['is_cross_deployer']}")

print()
print("  Live risk score:")
import sys
sys.path.insert(0, '.')
try:
    from surveillance.risk_scoring import score_contract
    res = score_contract(c, ADDR)
    if "error" in res:
        print(f"    error: {res['error']}")
    else:
        for k, v in res.items():
            if k != "components":
                print(f"    {k}: {v}")
except Exception as e:
    print(f"    SCORING ERROR: {type(e).__name__}: {e}")

c.close()
