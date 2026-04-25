"""Faster 0x604be06b vs 0xc0ffeefeed comparison — pre-fetch contract lists."""
import sqlite3
from collections import Counter
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
A = "0x604be06b9f6b6663f78e755db0c5965eb2337e3d"
B = "0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

contracts_A = {r[0].lower() for r in c.execute(
    "SELECT contract_address FROM contracts WHERE LOWER(deployer_address) = ?",
    (A.lower(),))}
contracts_B = {r[0].lower() for r in c.execute(
    "SELECT contract_address FROM contracts WHERE LOWER(deployer_address) = ?",
    (B.lower(),))}
print(f"A contracts: {len(contracts_A)}  B contracts: {len(contracts_B)}")
print(f"contracts shared (impossible if different deployers): {len(contracts_A & contracts_B)}")

# temp tables for fast IN
c.execute("CREATE TEMP TABLE A_addrs(addr TEXT PRIMARY KEY)")
c.execute("CREATE TEMP TABLE B_addrs(addr TEXT PRIMARY KEY)")
c.executemany("INSERT INTO A_addrs VALUES (?)", [(a,) for a in contracts_A])
c.executemany("INSERT INTO B_addrs VALUES (?)", [(a,) for a in contracts_B])

h("3. Bytecode family overlap")
fams_A = {r[0] for r in c.execute(
    "SELECT DISTINCT family_id FROM bytecode_family_members WHERE LOWER(contract_address) IN (SELECT addr FROM A_addrs)")}
fams_B = {r[0] for r in c.execute(
    "SELECT DISTINCT family_id FROM bytecode_family_members WHERE LOWER(contract_address) IN (SELECT addr FROM B_addrs)")}
shared = fams_A & fams_B
print(f"  A families: {len(fams_A)}")
print(f"  B families: {len(fams_B)}")
print(f"  shared: {len(shared)}")
if shared:
    print("  shared family details:")
    for fid in list(shared):
        r = c.execute("""
            SELECT family_name, member_count, unique_deployers, is_cross_deployer
            FROM bytecode_families WHERE family_id = ?
        """, (fid,)).fetchone()
        if r:
            # how many of A's contracts and B's contracts are in this family?
            a_n = c.execute(
                "SELECT COUNT(*) FROM bytecode_family_members WHERE family_id = ? AND LOWER(contract_address) IN (SELECT addr FROM A_addrs)",
                (fid,)).fetchone()[0]
            b_n = c.execute(
                "SELECT COUNT(*) FROM bytecode_family_members WHERE family_id = ? AND LOWER(contract_address) IN (SELECT addr FROM B_addrs)",
                (fid,)).fetchone()[0]
            print(f"    {fid}  members={r['member_count']} deployers={r['unique_deployers']} "
                  f"cross={r['is_cross_deployer']}  A_in={a_n} B_in={b_n}")

h("4. Victim-bot overlap (bots that hit BOTH operators' traps)")
bots_A = {r[0].lower() for r in c.execute("""
    SELECT DISTINCT bot_address FROM trap_events
    WHERE LOWER(trap_contract_address) IN (SELECT addr FROM A_addrs)
""")}
bots_B = {r[0].lower() for r in c.execute("""
    SELECT DISTINCT bot_address FROM trap_events
    WHERE LOWER(trap_contract_address) IN (SELECT addr FROM B_addrs)
""")}
shared_bots = bots_A & bots_B
print(f"  bots that hit A's traps: {len(bots_A)}")
print(f"  bots that hit B's traps: {len(bots_B)}")
print(f"  bots that hit BOTH: {len(shared_bots)}")
if shared_bots:
    print("  shared victim-bots (top 10 by total hits):")
    for b in list(shared_bots)[:10]:
        a_n = c.execute(
            "SELECT COUNT(*) FROM trap_events WHERE LOWER(trap_contract_address) IN (SELECT addr FROM A_addrs) AND LOWER(bot_address)=?",
            (b,)).fetchone()[0]
        b_n = c.execute(
            "SELECT COUNT(*) FROM trap_events WHERE LOWER(trap_contract_address) IN (SELECT addr FROM B_addrs) AND LOWER(bot_address)=?",
            (b,)).fetchone()[0]
        print(f"    {b}  hit_A={a_n}  hit_B={b_n}")

h("5. Deployment per-day timing (last 14 days)")
print("  date         A_deploys   B_deploys")
days_A = dict(c.execute("""
    SELECT substr(detection_timestamp,1,10) AS d, COUNT(*) FROM contracts
    WHERE LOWER(deployer_address) = ? AND detection_timestamp > datetime('now','-14 days')
    GROUP BY d
""", (A.lower(),)).fetchall())
days_B = dict(c.execute("""
    SELECT substr(detection_timestamp,1,10) AS d, COUNT(*) FROM contracts
    WHERE LOWER(deployer_address) = ? AND detection_timestamp > datetime('now','-14 days')
    GROUP BY d
""", (B.lower(),)).fetchall())
all_days = sorted(set(days_A) | set(days_B), reverse=True)
for d in all_days:
    a = days_A.get(d, 0); b = days_B.get(d, 0)
    bar_a = "A" * min(a, 30); bar_b = "B" * min(b, 30)
    print(f"  {d}      {a:>5}     {b:>5}     {bar_a}|{bar_b}")

h("6. Hour-of-day signature (last 14d, UTC)")
print("  hr | A | B")
hr_A = dict(c.execute("""
    SELECT CAST(substr(detection_timestamp,12,2) AS INTEGER) AS h, COUNT(*) FROM contracts
    WHERE LOWER(deployer_address) = ? AND detection_timestamp > datetime('now','-14 days')
    GROUP BY h
""", (A.lower(),)).fetchall())
hr_B = dict(c.execute("""
    SELECT CAST(substr(detection_timestamp,12,2) AS INTEGER) AS h, COUNT(*) FROM contracts
    WHERE LOWER(deployer_address) = ? AND detection_timestamp > datetime('now','-14 days')
    GROUP BY h
""", (B.lower(),)).fetchall())
for hr in range(24):
    a = hr_A.get(hr, 0); b = hr_B.get(hr, 0)
    if a == 0 and b == 0:
        continue
    bar = "#" * (max(a, b) // 4)
    print(f"  {hr:02d} | {a:>4} | {b:>4}  {bar}")

h("7. Funder relationship — already shown but recap")
fA = c.execute("SELECT json_extract(funding_trail, '$.funder') FROM deployers WHERE LOWER(deployer_address) = ?",
               (A.lower(),)).fetchone()[0]
fB = c.execute("SELECT json_extract(funding_trail, '$.funder') FROM deployers WHERE LOWER(deployer_address) = ?",
               (B.lower(),)).fetchone()[0]
print(f"  A funder: {fA}")
print(f"  B funder: {fB}")
print(f"  match: {fA and fB and fA.lower() == fB.lower()}")

# Other deployers funded by B's funder (A's was 10+ above; check B's separately)
print("\n  Other deployers funded by B's funder:")
if fB:
    for r in c.execute("""
        SELECT deployer_address, total_contracts_deployed, first_seen
        FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
        AND LOWER(deployer_address) != ?
        LIMIT 10
    """, (fB.lower(), B.lower())):
        print(f"    {r['deployer_address']}  fleet={r['total_contracts_deployed']} first={r['first_seen'][:10]}")

h("8. Trap-hour correlation (last 7 days)")
ta = c.execute("""
    SELECT substr(timestamp,1,13) AS hr, COUNT(*) AS n FROM trap_events
    WHERE LOWER(trap_contract_address) IN (SELECT addr FROM A_addrs)
    AND timestamp > datetime('now', '-7 days')
    GROUP BY hr ORDER BY hr DESC
""").fetchall()
tb = c.execute("""
    SELECT substr(timestamp,1,13) AS hr, COUNT(*) AS n FROM trap_events
    WHERE LOWER(trap_contract_address) IN (SELECT addr FROM B_addrs)
    AND timestamp > datetime('now', '-7 days')
    GROUP BY hr ORDER BY hr DESC
""").fetchall()
hours_A = {r['hr']: r['n'] for r in ta}
hours_B = {r['hr']: r['n'] for r in tb}
shared_hours = set(hours_A) & set(hours_B)
print(f"  A trap-hours: {len(hours_A)}  B trap-hours: {len(hours_B)}  shared: {len(shared_hours)}")
print("  hours where BOTH had traps fire:")
for hr in sorted(shared_hours, reverse=True)[:15]:
    print(f"    {hr}  A={hours_A[hr]:<3} B={hours_B[hr]:<3}")

h("9. Per-contract revert rates — same operating signature?")
# What's the typical revert rate of a fleet contract for each?
a_rates = c.execute("""
    SELECT AVG(rev_rate) AS avg_rate, MIN(rev_rate) AS min_rate, MAX(rev_rate) AS max_rate, COUNT(*) AS n
    FROM (SELECT (CAST(SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) AS rev_rate
          FROM transaction_events
          WHERE LOWER(contract_address) IN (SELECT addr FROM A_addrs)
          GROUP BY contract_address HAVING COUNT(*) >= 5)
""").fetchone()
b_rates = c.execute("""
    SELECT AVG(rev_rate) AS avg_rate, MIN(rev_rate) AS min_rate, MAX(rev_rate) AS max_rate, COUNT(*) AS n
    FROM (SELECT (CAST(SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) AS rev_rate
          FROM transaction_events
          WHERE LOWER(contract_address) IN (SELECT addr FROM B_addrs)
          GROUP BY contract_address HAVING COUNT(*) >= 5)
""").fetchone()
print(f"  A: contracts_with_5plus_tx={a_rates['n']}  avg_revert={a_rates['avg_rate']}  min={a_rates['min_rate']}  max={a_rates['max_rate']}")
print(f"  B: contracts_with_5plus_tx={b_rates['n']}  avg_revert={b_rates['avg_rate']}  min={b_rates['min_rate']}  max={b_rates['max_rate']}")

c.close()
