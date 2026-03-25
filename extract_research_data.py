"""Extract all quantitative data for the deep research prompt."""
import sqlite3
import os
import json

conn = sqlite3.connect("surveillance/data/surveillance.db")
conn.row_factory = sqlite3.Row

V = {}  # All variables

# SECTION 1: CORPUS
r = conn.execute("""SELECT COUNT(*) as tc, COUNT(DISTINCT deployer_address) as ud,
    MIN(detection_timestamp) as first, MAX(detection_timestamp) as last,
    ROUND(julianday(MAX(detection_timestamp)) - julianday(MIN(detection_timestamp)),1) as days
    FROM contracts""").fetchone()
V["TOTAL_CONTRACTS"] = f"{r[0]:,}"
V["UNIQUE_DEPLOYERS"] = f"{r[1]:,}"
V["DAYS_RUNNING"] = r[4]

tx = conn.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0]
callers = conn.execute("SELECT COUNT(DISTINCT interacting_address) FROM transaction_events").fetchone()[0]
V["TOTAL_TX_EVENTS"] = f"{tx:,}"
V["UNIQUE_CALLERS"] = f"{callers:,}"

for r in conn.execute("SELECT chain, COUNT(*) FROM contracts GROUP BY chain"):
    V[f"CHAIN_{r[0].upper()}"] = f"{r[1]:,}"

r = conn.execute("""SELECT COUNT(*), SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END),
    ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1)
    FROM transaction_events""").fetchone()
V["REVERT_PCT"] = r[2]

for r in conn.execute("SELECT confidence_tier, COUNT(*) FROM contracts GROUP BY confidence_tier"):
    V[f"TIER_{r[0].upper()}"] = f"{r[1]:,}"

# SECTION 2: TRUST AMPLIFICATION
r = conn.execute("""SELECT COUNT(DISTINCT interacting_address), COUNT(*),
    ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,2),
    ROUND(julianday(MAX(timestamp))-julianday(MIN(timestamp)),2)
    FROM transaction_events WHERE contract_address = '0xd4624228cce5baa0814c9e7f666a8a2c83b6f159'""").fetchone()
V["TRUST_VICTIMS"] = f"{r[0]:,}"
V["TRUST_HITS"] = f"{r[1]:,}"
V["TRUST_REVERT_PCT"] = r[2]
V["TRUST_DAYS"] = r[3]
vpd = round(r[0] / max(r[3], 0.01))
V["TRUST_VPD"] = f"{vpd:,}"

repeats = conn.execute("""SELECT COUNT(*) FROM (SELECT interacting_address FROM transaction_events
    WHERE contract_address = '0xd4624228cce5baa0814c9e7f666a8a2c83b6f159'
    GROUP BY interacting_address HAVING COUNT(*)>1)""").fetchone()[0]
V["REPEAT_VICTIMS"] = f"{repeats:,}"
V["REPEAT_PCT"] = round(repeats / int(V["TRUST_VICTIMS"].replace(",", "")) * 100)

router = conn.execute("""SELECT COUNT(*) FROM transaction_events
    WHERE contract_address = '0xd4624228cce5baa0814c9e7f666a8a2c83b6f159'
    AND function_selector='3593564c'""").fetchone()[0]
V["ROUTER_PCT"] = round(router / int(V["TRUST_HITS"].replace(",", "")) * 100, 1)

t = conn.execute("""SELECT COUNT(DISTINCT interacting_address), COUNT(*),
    ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,2),
    ROUND(julianday(MAX(timestamp))-julianday(MIN(timestamp)),2)
    FROM transaction_events WHERE contract_address = '0x9da33ece6fdf36ecf99e10dbd6ecd0cb529e257e'""").fetchone()
V["TRAD_VICTIMS"] = f"{t[0]:,}"
V["TRAD_REVERT_PCT"] = t[2]
V["TRAD_DAYS"] = t[3]
trad_vpd = round(t[0] / max(t[3], 0.01))
V["TRAD_VPD"] = f"{trad_vpd:,}"

overlap = conn.execute("""SELECT COUNT(*) FROM (
    SELECT DISTINCT interacting_address FROM transaction_events
    WHERE contract_address = '0xd4624228cce5baa0814c9e7f666a8a2c83b6f159'
    INTERSECT SELECT DISTINCT interacting_address FROM transaction_events
    WHERE contract_address = '0x9da33ece6fdf36ecf99e10dbd6ecd0cb529e257e')""").fetchone()[0]
V["VICTIM_OVERLAP"] = overlap
V["TRUST_AMP_FACTOR"] = round(vpd / max(trad_vpd, 1), 1)

fee_total = conn.execute("SELECT COUNT(*) FROM contracts WHERE has_unusual_fee_structure=1").fetchone()[0]
fee_active = conn.execute("""SELECT COUNT(DISTINCT c.contract_address) FROM contracts c
    JOIN transaction_events te ON c.contract_address = te.contract_address
    WHERE c.has_unusual_fee_structure=1""").fetchone()[0]
V["FEE_TOTAL"] = fee_total
V["FEE_DORMANT"] = fee_total - fee_active

# SECTION 3: CAMOUFLAGE
r = conn.execute("""SELECT COUNT(*),
    SUM(CASE WHEN rp<10 THEN 1 ELSE 0 END),
    SUM(CASE WHEN rp>=10 AND rp<50 THEN 1 ELSE 0 END),
    SUM(CASE WHEN rp>=50 THEN 1 ELSE 0 END),
    ROUND(SUM(CASE WHEN rp<10 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1)
    FROM (SELECT ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1) as rp
          FROM transaction_events GROUP BY contract_address HAVING COUNT(*)>=10)""").fetchone()
V["ACTIVE_CONTRACTS"] = r[0]
V["CAMOUFLAGED"] = r[1]
V["MODERATE"] = r[2]
V["OVERT"] = r[3]
V["CAMOUFLAGE_RATIO"] = r[4]

# Revert trend
revert_trend = []
for r in conn.execute("""SELECT DATE(timestamp), COUNT(*),
    ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1)
    FROM transaction_events GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)"""):
    revert_trend.append(f"{r[0]}: {r[2]}%")
V["REVERT_TREND_FIRST"] = revert_trend[0] if revert_trend else "N/A"
V["REVERT_TREND_LAST"] = revert_trend[-1] if revert_trend else "N/A"

# SECTION 4: ORGS
bots = conn.execute("SELECT COUNT(*) FROM bot_candidates").fetchone()[0]
V["BOTS"] = bots

entity_lines = []
for r in conn.execute("SELECT category, COUNT(*) FROM entity_classification GROUP BY category ORDER BY COUNT(*) DESC"):
    entity_lines.append(f"{r[0]}: {r[1]}")
V["ENTITY_BREAKDOWN"] = ", ".join(entity_lines)

classified = conn.execute("SELECT COUNT(*) FROM entity_classification").fetchone()[0]
total_unique = conn.execute("""SELECT COUNT(DISTINCT addr) FROM (
    SELECT deployer_address as addr FROM deployers UNION SELECT interacting_address FROM transaction_events
    UNION SELECT contract_address FROM contracts)""").fetchone()[0]
V["CLASSIFIED"] = f"{classified:,}"
V["TOTAL_UNIQUE"] = f"{total_unique:,}"
V["CLASSIFICATION_PCT"] = round(classified / total_unique * 100, 1)

# SECTION 5: TEMPLATES
templates = conn.execute("""SELECT bytecode_pattern_notes, COUNT(*) as c, COUNT(DISTINCT deployer_address) as d
    FROM contracts WHERE bytecode_pattern_notes IS NOT NULL AND bytecode_pattern_notes != ''
    GROUP BY bytecode_pattern_notes HAVING d > 5 ORDER BY c DESC LIMIT 1""").fetchone()
if templates:
    V["TEMPLATE_CONTRACTS"] = templates[1]
    V["TEMPLATE_DEPLOYERS"] = templates[2]
else:
    V["TEMPLATE_CONTRACTS"] = "N/A"
    V["TEMPLATE_DEPLOYERS"] = "N/A"

# SECTION 6: NOTABLE
cases = len([f for f in os.listdir("surveillance/data/cases") if f.endswith(".md")])
V["CASE_FILES"] = cases

ext = conn.execute("SELECT COUNT(*), COALESCE(SUM(total_usd_moved),0) FROM extraction_events").fetchone()
V["EXTRACTION_EVENTS"] = ext[0]
V["TOTAL_USD"] = f"${ext[1]:,.0f}"

conn.close()

# Print reference table
print("=" * 60)
print("REFERENCE TABLE")
print("=" * 60)
for k, v in sorted(V.items()):
    print(f"  {k:<30} {v}")
