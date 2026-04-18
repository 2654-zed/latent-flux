"""Deeper drill on the notable alerts from the session activity pull."""
import sqlite3, json, datetime

DB = "/app/surveillance/data/surveillance.db"
CUTOFF = "2026-04-17T18:00:00"

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("--- DORMANT_ACTIVATION detail ---")
for r in c.execute(
    "SELECT address, payload, timestamp FROM alerts "
    "WHERE alert_type = 'DORMANT_ACTIVATION' AND timestamp >= ?",
    (CUTOFF,),
):
    print(f"  fleet={r[0][:18]}...  ts={r[2][:19]}")
    try:
        p = json.loads(r[1])
        print(f"  payload: {json.dumps(p, indent=2)[:600]}")
    except Exception:
        print(f"  payload: {(r[1] or '')[:500]}")

print("\n--- X402_AGENT_DRAIN detail ---")
for r in c.execute(
    "SELECT address, payload, timestamp, tx_hash FROM alerts "
    "WHERE alert_type = 'X402_AGENT_DRAIN' AND timestamp >= ?",
    (CUTOFF,),
):
    print(f"  drainer={r[0]}  ts={r[2][:19]}  tx={r[3]}")
    try:
        p = json.loads(r[1])
        print(f"     amount_usd={p.get('amount_usd') or p.get('amount')}  "
              f"victim={p.get('victim') or p.get('payer')}  chain={p.get('chain')}")
    except Exception:
        pass

print("\n--- drainer novelty check (prior alert count) ---")
for addr in (
    "0x95be5368fdec2817b87c98713a2b1a2180422ba7",
    "0x96ec2c1574ef42d2a345cbbcdf439832084f2b08",
    "0xe81256140cbdeec04304463cca5397aadd99f194",
):
    prior = c.execute(
        "SELECT COUNT(*), MIN(timestamp) FROM alerts "
        "WHERE LOWER(address) = ? AND timestamp < ?",
        (addr, CUTOFF),
    ).fetchone()
    print(f"  {addr}  prior_alerts={prior[0]}  first_seen={prior[1]}")

print("\n--- SUSPECTED_HIGH_TRAFFIC detail ---")
for r in c.execute(
    "SELECT address, payload, timestamp FROM alerts "
    "WHERE alert_type = 'SUSPECTED_HIGH_TRAFFIC' AND timestamp >= ?",
    (CUTOFF,),
):
    print(f"  addr={r[0]}  ts={r[2][:19]}")
    try:
        p = json.loads(r[1])
        print(f"     callers={p.get('callers')}  deployer={p.get('deployer')}  message={(p.get('message') or '')[:100]}")
    except Exception:
        pass

print("\n--- TRUST_AMPLIFICATION detail ---")
for r in c.execute(
    "SELECT address, payload, timestamp FROM alerts "
    "WHERE alert_type = 'TRUST_AMPLIFICATION' AND timestamp >= ?",
    (CUTOFF,),
):
    print(f"  addr={r[0]}  ts={r[2][:19]}")
    try:
        p = json.loads(r[1])
        print(f"     amp={p.get('amplification_factor')}  "
              f"router_pct={p.get('router_percentage')}  callers={p.get('total_callers')}  "
              f"revert={p.get('revert_rate')}  alert={p.get('alert_level')}")
    except Exception:
        pass

# Was today Saturday or Sunday? deployer_similarity scheduler is Sunday only.
d = datetime.date(2026, 4, 18)
print(f"\n2026-04-18 weekday: {d.strftime('%A')}")

# Are any of the 3 new drainers classified in x402_facilitators?
print("\n--- x402_facilitators classification of new drainers ---")
for addr in (
    "0x95be5368fdec2817b87c98713a2b1a2180422ba7",
    "0x96ec2c1574ef42d2a345cbbcdf439832084f2b08",
    "0xe81256140cbdeec04304463cca5397aadd99f194",
):
    row = c.execute(
        "SELECT classification, source, tx_count FROM x402_facilitators "
        "WHERE LOWER(address) = ?",
        (addr,),
    ).fetchone()
    if row:
        print(f"  {addr}  class={row[0]}  src={row[1]}  tx_count={row[2]}")
    else:
        print(f"  {addr}  NOT in x402_facilitators")

# What's the payload shape on the first DRAIN to make sense of amounts?
print("\n--- FIRST X402_AGENT_DRAIN raw payload (for schema) ---")
r = c.execute(
    "SELECT payload FROM alerts WHERE alert_type = 'X402_AGENT_DRAIN' "
    "AND timestamp >= ? LIMIT 1",
    (CUTOFF,),
).fetchone()
if r:
    print(r[0][:800])
