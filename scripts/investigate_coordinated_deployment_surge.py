"""Investigate the COORDINATED_DEPLOYMENT alert surge from ~240/day to 8,797/day.

Phase 4 framing: Refines the lexicon §2.7 entry on COORDINATED_DEPLOYMENT
("velocity-escalation pipeline; over-fires") with new volume data and
attribution.

Questions to answer:
1. Are these alerts concentrated by deployer? Or distributed across many?
2. What's the bytecode family distribution?
3. Time-clustered (single-burst) or distributed across the 24h window?
4. Does the alert payload reveal the velocity threshold being tripped?
5. Are these new operators or existing ones in INDEX.md?
"""
import sqlite3
import json
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
CUTOFF = "2026-04-25T03:04:06"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

h("1. COORDINATED_DEPLOYMENT alert volume by hour")
print(f"  {'hour':<14} {'alerts':<8}")
for r in c.execute("""
    SELECT substr(timestamp, 1, 13) AS hr, COUNT(*) AS n
    FROM alerts
    WHERE alert_type='COORDINATED_DEPLOYMENT'
      AND timestamp > ? AND COALESCE(false_positive,0)=0
    GROUP BY hr ORDER BY hr
""", (CUTOFF,)):
    bar = "#" * min(r['n'] // 30, 60)
    print(f"  {r['hr']}  {r['n']:<6} {bar}")

h("2. Concentration by address (alerts.address field)")
addr_counts = list(c.execute("""
    SELECT address, COUNT(*) AS n
    FROM alerts
    WHERE alert_type='COORDINATED_DEPLOYMENT'
      AND timestamp > ? AND COALESCE(false_positive,0)=0
    GROUP BY address ORDER BY n DESC
""", (CUTOFF,)))
print(f"  total addresses with COORDINATED_DEPLOYMENT alerts in window: {len(addr_counts)}")
print(f"  top 15 by alert count:")
for r in addr_counts[:15]:
    # Look up if it's a deployer + classification
    dep = c.execute("SELECT total_contracts_deployed, json_extract(funding_trail, '$.funder') AS funder FROM deployers WHERE LOWER(deployer_address) = ?", (r['address'].lower() if r['address'] else '',)).fetchone()
    fleet = dep['total_contracts_deployed'] if dep else '-'
    funder = dep['funder'] if dep else None
    wl = c.execute("SELECT priority, entity_name FROM watchlist WHERE LOWER(address) = ?", (r['address'].lower() if r['address'] else '',)).fetchone()
    wl_str = f" [{wl['priority']}/{wl['entity_name'][:40]}]" if wl else ""
    print(f"    {r['address']}  alerts={r['n']:<5} fleet={fleet:<5} funder={funder[:14] if funder else '-':<14}{wl_str}")

h("3. Sample alert payload — what triggers the alert?")
sample = c.execute("""
    SELECT timestamp, address, payload
    FROM alerts
    WHERE alert_type='COORDINATED_DEPLOYMENT'
      AND timestamp > ? AND COALESCE(false_positive,0)=0
    LIMIT 5
""", (CUTOFF,)).fetchall()
for r in sample:
    print(f"  {r['timestamp'][:19]} addr={r['address']}")
    try:
        p = json.loads(r['payload'])
        for k, v in p.items():
            sv = str(v)
            if len(sv) > 80: sv = sv[:80] + "..."
            print(f"    {k}: {sv}")
    except Exception:
        print(f"    payload: {r['payload'][:200]}")
    print()

h("4. Funders in window — top by alert count from their downstream deployers")
funder_counts = Counter()
for r in addr_counts:
    if not r['address']:
        continue
    dep = c.execute("SELECT json_extract(funding_trail, '$.funder') AS f FROM deployers WHERE LOWER(deployer_address) = ?", (r['address'].lower(),)).fetchone()
    if dep and dep['f']:
        funder_counts[dep['f'].lower()] += r['n']
print(f"  Top 10 funders by aggregate COORDINATED_DEPLOYMENT volume:")
for f, n in funder_counts.most_common(10):
    # Funder corpus stats
    n_deps = c.execute("SELECT COUNT(*) FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?", (f,)).fetchone()[0]
    # Watchlist match
    wl = c.execute("SELECT priority, entity_name FROM watchlist WHERE LOWER(address) = ?", (f,)).fetchone()
    wl_str = f" [{wl['priority']}/{wl['entity_name'][:30]}]" if wl else ""
    print(f"    {f}  alerts={n:<5} downstream_deployers={n_deps:<5}{wl_str}")

h("5. Are the alerted contracts entering known bytecode families?")
# Get the contract addresses from the alerts (address field is sometimes the contract, sometimes the deployer; check payload)
# Sample 200 alerts and look at their contracts
print("  Sampling 200 most-recent alerts to extract contract addresses from payload...")
fam_counts = Counter()
contracts_sampled = 0
for r in c.execute("""
    SELECT payload FROM alerts
    WHERE alert_type='COORDINATED_DEPLOYMENT'
      AND timestamp > ? AND COALESCE(false_positive,0)=0
    ORDER BY timestamp DESC LIMIT 200
""", (CUTOFF,)):
    try:
        p = json.loads(r['payload'])
        addr = p.get('contract_address') or p.get('contract') or p.get('address')
        if not addr:
            continue
        fam = c.execute("SELECT family_id FROM bytecode_family_members WHERE LOWER(contract_address) = ? LIMIT 1", (addr.lower(),)).fetchone()
        if fam:
            fam_counts[fam['family_id']] += 1
        contracts_sampled += 1
    except Exception:
        pass
print(f"  contracts with extractable address in sample: {contracts_sampled}")
print(f"  family hits:")
for fid, n in fam_counts.most_common(8):
    fr = c.execute("SELECT family_name, member_count, is_cross_deployer FROM bytecode_families WHERE family_id=? LIMIT 1", (fid,)).fetchone()
    name = (fr['family_name'] or '-')[:50] if fr else '-'
    print(f"    {fid}  in_sample={n}  members={fr['member_count'] if fr else '-':<5} cross={fr['is_cross_deployer'] if fr else '-'}  {name}")

h("6. Comparison to prior 24h window")
# How many COORDINATED_DEPLOYMENT in the window 2026-04-24 → 2026-04-25 (the prior session's window)?
prior = c.execute("""
    SELECT COUNT(*) FROM alerts
    WHERE alert_type='COORDINATED_DEPLOYMENT'
      AND timestamp > '2026-04-24T03:00:00' AND timestamp < '2026-04-25T03:04:06'
      AND COALESCE(false_positive,0)=0
""").fetchone()[0]
this_window = c.execute("""
    SELECT COUNT(*) FROM alerts
    WHERE alert_type='COORDINATED_DEPLOYMENT'
      AND timestamp > ? AND COALESCE(false_positive,0)=0
""", (CUTOFF,)).fetchone()[0]
print(f"  prior 24h (04-24 → 04-25):  {prior}")
print(f"  this window (since cutoff): {this_window}")
print(f"  multiple: {this_window/prior if prior else float('inf'):.1f}x")

c.close()
