"""Profile the 0xc0ffee... vanity-prefix bot fleet.

These bots hit both 0x604be06b and 0xc0ffeefeed trap fleets.
Question: how many of them, who funds them, what are they extracting?
"""
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
PREFIX = "0xc0ffee"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

h("1. Population size — bots with the c0ffee prefix")
# Distinct bot addresses ever seen in trap_events
bots = [r[0].lower() for r in c.execute(
    "SELECT DISTINCT bot_address FROM trap_events WHERE LOWER(bot_address) LIKE ?",
    (f"{PREFIX}%",))]
print(f"  c0ffee bots that have hit at least one trap: {len(bots)}")
# In bot_candidates table?
in_cands = c.execute(
    "SELECT COUNT(*) FROM bot_candidates WHERE LOWER(address) LIKE ?",
    (f"{PREFIX}%",)).fetchone()[0]
print(f"  c0ffee addresses in bot_candidates: {in_cands}")
# Anywhere in deployers? (Bots are usually EOAs, but some run from contracts)
in_deps = c.execute(
    "SELECT COUNT(*) FROM deployers WHERE LOWER(deployer_address) LIKE ?",
    (f"{PREFIX}%",)).fetchone()[0]
print(f"  c0ffee addresses in deployers (have deployed contracts): {in_deps}")
# Anywhere in contracts (a c0ffee contract address)?
in_contracts = c.execute(
    "SELECT COUNT(*) FROM contracts WHERE LOWER(contract_address) LIKE ?",
    (f"{PREFIX}%",)).fetchone()[0]
print(f"  c0ffee contract addresses: {in_contracts}")

h("2. Top c0ffee bots by hits taken")
print(f"  {'bot_address':<44} {'total_hits':<11} {'distinct_traps':<15} {'distinct_deployers':<18} {'first_hit':<11} {'last_hit'}")
for r in c.execute("""
    SELECT bot_address, COUNT(*) AS n,
      COUNT(DISTINCT trap_contract_address) AS distinct_traps,
      COUNT(DISTINCT (SELECT deployer_address FROM contracts ct
                      WHERE LOWER(ct.contract_address) = LOWER(te.trap_contract_address))) AS distinct_deps,
      MIN(timestamp) AS first_hit, MAX(timestamp) AS last_hit
    FROM trap_events te
    WHERE LOWER(bot_address) LIKE ?
    GROUP BY bot_address ORDER BY n DESC LIMIT 15
""", (f"{PREFIX}%",)):
    print(f"  {r['bot_address']}  {r['n']:<11} {r['distinct_traps']:<15} {r['distinct_deps']:<18} "
          f"{r['first_hit'][:10]} {r['last_hit'][:10]}")

h("3. Aggregate: how much have c0ffee bots lost to traps?")
agg = c.execute("""
    SELECT COUNT(*) AS n, COUNT(DISTINCT bot_address) AS distinct_bots,
           COUNT(DISTINCT trap_contract_address) AS distinct_traps,
           SUM(loss_estimate_usd) AS total_loss_usd
    FROM trap_events WHERE LOWER(bot_address) LIKE ?
""", (f"{PREFIX}%",)).fetchone()
print(f"  total c0ffee trap-event records: {agg['n']:,}")
print(f"  distinct c0ffee bots involved: {agg['distinct_bots']}")
print(f"  distinct traps they've hit: {agg['distinct_traps']:,}")
print(f"  total loss_estimate_usd: {agg['total_loss_usd']}")

h("4. Are c0ffee bots concentrated against specific operators?")
# What deployers have c0ffee bots hit the most?
print(f"  {'deployer':<44} {'c0ffee_hits':<12} {'fleet':<6} {'confirmed':<10}")
for r in c.execute("""
    SELECT ct.deployer_address, COUNT(*) AS n,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = ct.deployer_address) AS fleet,
           (SELECT COUNT(*) FROM contracts WHERE deployer_address = ct.deployer_address
              AND confidence_tier='confirmed') AS conf
    FROM trap_events te
    JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
    WHERE LOWER(te.bot_address) LIKE ?
    GROUP BY ct.deployer_address ORDER BY n DESC LIMIT 10
""", (f"{PREFIX}%",)):
    print(f"  {r['deployer_address']}  {r['n']:<12} {r['fleet']:<6} {r['conf']}")

h("5. Sister-prefix bots — are these same addresses appearing as deployers anywhere?")
# E.g., is c0ffeefeed... (a known operator) in this same prefix family?
print(f"  Distinct c0ffee addresses across roles:")
for table_label, q in [
    ("contracts (deployer)", "SELECT DISTINCT LOWER(deployer_address) FROM contracts WHERE LOWER(deployer_address) LIKE ?"),
    ("contracts (contract address)", "SELECT DISTINCT LOWER(contract_address) FROM contracts WHERE LOWER(contract_address) LIKE ?"),
    ("trap_events (bot)", "SELECT DISTINCT LOWER(bot_address) FROM trap_events WHERE LOWER(bot_address) LIKE ?"),
    ("trap_events (trap)", "SELECT DISTINCT LOWER(trap_contract_address) FROM trap_events WHERE LOWER(trap_contract_address) LIKE ?"),
    ("alerts (address)", "SELECT DISTINCT LOWER(address) FROM alerts WHERE LOWER(address) LIKE ?"),
]:
    rows = c.execute(q, (f"{PREFIX}%",)).fetchall()
    print(f"    {table_label:<32} {len(rows)}")

h("6. Funding sources of the c0ffee bot fleet")
# These are EOAs presumably — do they have funding records?
funder_counts = Counter()
for b in bots:
    f = c.execute("""
        SELECT json_extract(funding_trail, '$.funder') FROM deployers
        WHERE LOWER(deployer_address) = ?
    """, (b,)).fetchone()
    if f and f[0]:
        funder_counts[f[0].lower()] += 1
# Most c0ffee bots aren't deployers, so funder_counts may be small
print(f"  c0ffee bots with funder records (in deployers table): {sum(funder_counts.values())}")
print(f"  unique funders: {len(funder_counts)}")
print("  top funders:")
for fdr, n in funder_counts.most_common(10):
    print(f"    {fdr}  funded={n}")

h("7. Time pattern of c0ffee bot activity (last 7 days)")
print(f"  {'date':<11}  {'events':<6}  {'distinct_bots':<13}")
for r in c.execute("""
    SELECT substr(timestamp,1,10) AS d, COUNT(*) AS n,
           COUNT(DISTINCT bot_address) AS bots
    FROM trap_events WHERE LOWER(bot_address) LIKE ?
    AND timestamp > datetime('now', '-7 days')
    GROUP BY d ORDER BY d DESC
""", (f"{PREFIX}%",)):
    print(f"  {r['d']}    {r['n']:<6}  {r['bots']}")

h("8. Which selectors do they probe? (the bait pattern)")
print("  Selectors c0ffee bots most often revert on:")
for r in c.execute("""
    SELECT failure_signature, COUNT(*) AS n
    FROM trap_events WHERE LOWER(bot_address) LIKE ?
    AND failure_signature IS NOT NULL AND failure_signature != ''
    GROUP BY failure_signature ORDER BY n DESC LIMIT 15
""", (f"{PREFIX}%",)):
    print(f"    {r['failure_signature'][:60]:<62} n={r['n']}")

h("9. Sample c0ffee addresses — sanity-check the prefix")
print("  10 distinct c0ffee bots seen:")
for b in bots[:10]:
    print(f"    {b}")

c.close()
