"""Today-scoped activity pull (2026-04-18 00:00 UTC onwards)."""
import json
import sqlite3
from collections import Counter, defaultdict

DB = "/app/surveillance/data/surveillance.db"
CUTOFF = "2026-04-18T00:00:00"

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print(f"=== TODAY (since {CUTOFF} UTC) ===\n")

# 1. Alert totals
print("--- Alerts ---")
rows = c.execute(
    "SELECT alert_type, COUNT(*) FROM alerts WHERE timestamp >= ? "
    "GROUP BY alert_type ORDER BY 2 DESC",
    (CUTOFF,),
).fetchall()
total = sum(r[1] for r in rows)
for r in rows:
    print(f"  {r[0]:38s}  {r[1]:>6}")
print(f"  {'TOTAL':38s}  {total:>6}")

# 2. Drains: attacker, victim, amount
print("\n--- X402_AGENT_DRAIN events ---")
drain_rows = c.execute(
    "SELECT address, payload, timestamp, tx_hash FROM alerts "
    "WHERE alert_type = 'X402_AGENT_DRAIN' AND timestamp >= ? "
    "ORDER BY timestamp",
    (CUTOFF,),
).fetchall()
facilitator_hits = Counter()
total_usd = 0.0
for r in drain_rows:
    try:
        p = json.loads(r["payload"] or "{}")
        facilitator = p.get("facilitator") or ""
        amt = p.get("amount_normalized_6dec") or 0.0
        facilitator_hits[facilitator] += 1
        total_usd += amt
        print(f"  {r['timestamp'][:19]}  victim={p.get('payer','?')[:18]}...  "
              f"${amt:>10,.2f}  facilitator={facilitator[:18]}...  chain={p.get('chain')}")
    except Exception as e:
        print(f"  parse err: {e}")
print(f"  TOTAL today: {len(drain_rows)} drains, ${total_usd:,.2f}")
print(f"  By facilitator:")
for f, n in facilitator_hits.most_common():
    print(f"    {f}  n={n}")

# 3. Trap confirmations (behavioral)
print("\n--- New trap_events (behavioral confirmations) ---")
tr = c.execute(
    "SELECT trap_contract_address, bot_address, tx_hash, block_number, timestamp, "
    "failure_signature FROM trap_events WHERE timestamp >= ? "
    "ORDER BY block_number DESC",
    (CUTOFF,),
).fetchall()
print(f"  count: {len(tr)}")
bot_counts = Counter()
for r in tr:
    bot_counts[r["bot_address"][:18]] += 1
    print(f"  blk={r['block_number']:>10}  trap={r['trap_contract_address'][:18]}...  "
          f"bot={r['bot_address'][:18]}...")
print("  bot diversity:")
for bot, n in bot_counts.most_common(10):
    print(f"    {bot}...  n={n}")

# 4. Contracts promoted to confirmed today
print("\n--- Contracts promoted to confirmed today ---")
n = c.execute(
    "SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'confirmed' "
    "AND last_updated >= ?",
    (CUTOFF,),
).fetchone()[0]
print(f"  count: {n}")
for r in c.execute(
    "SELECT contract_address, chain, confidence_reason, last_updated FROM contracts "
    "WHERE confidence_tier = 'confirmed' AND last_updated >= ? "
    "ORDER BY last_updated DESC",
    (CUTOFF,),
):
    reason = (r[2] or "")[:70]
    print(f"  {r[1]:10s}  {r[0]}  {r[3][:19]}  {reason}")

# 5. Behavioral-laundering-shape candidates (SUSPECTED_HIGH_TRAFFIC + TRUST_AMPLIFICATION co-occurrence)
print("\n--- Behavioral-laundering-shape candidates (high traffic + trust amp WARNING/INFO) ---")
for r in c.execute(
    "SELECT address, payload, timestamp, alert_type FROM alerts "
    "WHERE alert_type IN ('SUSPECTED_HIGH_TRAFFIC', 'TRUST_AMPLIFICATION') "
    "AND timestamp >= ? ORDER BY address, alert_type",
    (CUTOFF,),
):
    try:
        p = json.loads(r["payload"] or "{}")
        print(f"  {r['timestamp'][:19]}  {r['alert_type']:24s}  {r['address']}")
        msg = p.get("message") or ""
        if msg:
            print(f"       {msg[:110]}")
    except Exception:
        pass

# 6. Dormant activations
print("\n--- DORMANT_ACTIVATION today ---")
for r in c.execute(
    "SELECT address, payload, timestamp FROM alerts "
    "WHERE alert_type = 'DORMANT_ACTIVATION' AND timestamp >= ? "
    "ORDER BY timestamp DESC",
    (CUTOFF,),
):
    try:
        p = json.loads(r["payload"] or "{}")
        print(f"  {r['timestamp'][:19]}  deployer={r['address']}  "
              f"fleet={p.get('fleet_size','?')}  activated={p.get('newly_activated','?')}  "
              f"org={p.get('org') or 'unattributed'}")
    except Exception:
        print(f"  {r['timestamp'][:19]}  deployer={r['address']}  (payload unparseable)")

# 7. Velocity alerts
print("\n--- HIGH_VELOCITY_DEPLOYER today ---")
for r in c.execute(
    "SELECT address, payload, timestamp FROM alerts "
    "WHERE alert_type = 'HIGH_VELOCITY_DEPLOYER' AND timestamp >= ? "
    "ORDER BY timestamp DESC",
    (CUTOFF,),
):
    try:
        p = json.loads(r["payload"] or "{}")
        cnt = p.get("deployment_count") or p.get("velocity") or p.get("count") or "?"
    except Exception:
        cnt = "?"
    print(f"  {r['timestamp'][:19]}  {r['address']}  deployments={cnt}")

# 8. Deployment count by chain today
print("\n--- Deployments today by chain ---")
for r in c.execute(
    "SELECT chain, COUNT(*) FROM contracts WHERE detection_timestamp >= ? "
    "GROUP BY chain ORDER BY 2 DESC",
    (CUTOFF,),
):
    print(f"  {r[0]:10s}  {r[1]:>6,}")

# 9. Scheduler firing check — look for rows written today via the nightly cycle
# daily_metrics for 2026-04-18 would have been written by either my manual catchup at ~05:00
# or the 00:15 scheduler if it fired. Check the timestamp pattern.
print("\n--- Scheduler fire-time analysis ---")
for tbl, col in [
    ("daily_metrics", "date"),
    ("camouflage_metrics", "date"),
    ("predictions", "issued_date"),
]:
    r = c.execute(f"SELECT {col}, COUNT(*) FROM {tbl} WHERE {col} LIKE '2026-04-18%' GROUP BY {col}").fetchone()
    if r:
        print(f"  {tbl:28s}  date={r[0]}  rows={r[1]}")
    else:
        print(f"  {tbl:28s}  NO 2026-04-18 rows")

# Heartbeat
print("\n--- Heartbeat (live monitors) ---")
for r in c.execute(
    "SELECT component, timestamp, blocks, deployments FROM heartbeat "
    "WHERE component LIKE 'deployment_monitor_%' OR component = 'routing_monitor' "
    "ORDER BY component"
):
    print(f"  {r[0]:38s}  last={r[1][:19]}  blk={r[2]:>8,}  dep={r[3]:>5,}")

# 10. Any new confirmed traps deployed by a deployer with NO prior history
# (Pattern A reputation-sacrifice fingerprint — single high-stakes deployment from a clean deployer)
print("\n--- Pattern-A candidate check (confirmed today, deployer has ≤1 prior contract) ---")
cands = c.execute(
    "SELECT c.contract_address, c.chain, c.deployer_address, c.last_updated, "
    "(SELECT COUNT(*) FROM contracts c2 WHERE c2.deployer_address = c.deployer_address "
    " AND c2.contract_address != c.contract_address) as prior_ct "
    "FROM contracts c WHERE c.confidence_tier = 'confirmed' "
    "AND c.last_updated >= ?",
    (CUTOFF,),
).fetchall()
pattern_a = [r for r in cands if r[4] <= 1]
print(f"  candidates: {len(pattern_a)} of {len(cands)} confirmed-today")
for r in pattern_a:
    print(f"  {r[1]:10s}  {r[0]}  deployer={r[2][:18]}...  prior_contracts={r[4]}")
