"""Pull CE5E lifetime drain stats for the case file."""
import sqlite3, json

DB = "/app/surveillance/data/surveillance.db"
CE5E = "0xce5ec7336f863931fda2ee3e4b9dad99fcc53c91"

c = sqlite3.connect(DB)

print("=== CE5E lifetime drain footprint ===")
total_drains = 0
total_usd = 0.0
chains = {}
earliest = None
latest = None
victims = set()
for r in c.execute(
    "SELECT payload, timestamp FROM alerts WHERE alert_type = 'X402_AGENT_DRAIN'"
):
    try:
        p = json.loads(r[0])
    except Exception:
        continue
    if (p.get("facilitator") or "").lower() != CE5E:
        continue
    total_drains += 1
    total_usd += p.get("amount_normalized_6dec", 0) or 0
    chain = p.get("chain", "?")
    chains[chain] = chains.get(chain, 0) + 1
    victims.add((p.get("payer") or "").lower())
    if earliest is None or r[1] < earliest:
        earliest = r[1]
    if latest is None or r[1] > latest:
        latest = r[1]
print(f"  total drain events: {total_drains}")
print(f"  total USD (6-dec normalized; accurate for stablecoins only): ${total_usd:,.2f}")
print(f"  unique victims: {len(victims)}")
print(f"  chains: {chains}")
print(f"  first seen: {earliest}")
print(f"  last seen:  {latest}")

# Today scope
print()
print("=== today (2026-04-18) CE5E drains ===")
today_count = 0
today_usd = 0.0
today_rows = []
for r in c.execute(
    "SELECT payload, timestamp, tx_hash FROM alerts "
    "WHERE alert_type = 'X402_AGENT_DRAIN' AND timestamp >= '2026-04-18T00:00:00' "
    "ORDER BY timestamp"
):
    try:
        p = json.loads(r[0])
    except Exception:
        continue
    if (p.get("facilitator") or "").lower() != CE5E:
        continue
    today_count += 1
    usd = p.get("amount_normalized_6dec", 0) or 0
    today_usd += usd
    today_rows.append((r[1][:19], p.get("chain", "?"), usd, p.get("payer"), r[2]))
    print(f"  {r[1][:19]}  {p.get('chain','?'):8s}  ${usd:>10,.2f}  victim={p.get('payer')}  tx={r[2]}")
print(f"  today count: {today_count}  today total: ${today_usd:,.2f}")

# Victim overlap — are today's victims known in approval_events, x402_permit2_exposure, or prior alerts?
print()
print("=== today's victim history check ===")
for _, _, _, victim_addr, _ in today_rows:
    if not victim_addr:
        continue
    v = victim_addr.lower()
    prior_alerts = c.execute(
        "SELECT COUNT(*) FROM alerts WHERE LOWER(address) = ? AND timestamp < '2026-04-18T00:00:00'",
        (v,),
    ).fetchone()[0]
    approvals = c.execute(
        "SELECT COUNT(*) FROM approval_events WHERE LOWER(approver) = ?",
        (v,),
    ).fetchone()[0]
    exposure = c.execute(
        "SELECT COUNT(*) FROM x402_permit2_exposure WHERE LOWER(owner_address) = ?",
        (v,),
    ).fetchone()[0]
    print(f"  {victim_addr}  prior_alerts={prior_alerts}  approval_events={approvals}  permit2_exposure={exposure}")

# Lifetime victims — are they concentrated or wide distribution?
print()
print("=== lifetime victim concentration ===")
from collections import Counter
victim_counts = Counter()
for r in c.execute("SELECT payload FROM alerts WHERE alert_type = 'X402_AGENT_DRAIN'"):
    try:
        p = json.loads(r[0])
    except Exception:
        continue
    if (p.get("facilitator") or "").lower() != CE5E:
        continue
    victim_counts[(p.get("payer") or "").lower()] += 1
repeat_victims = [v for v, n in victim_counts.items() if n > 1]
print(f"  repeat victims (hit 2+ times): {len(repeat_victims)}")
for v, n in victim_counts.most_common(5):
    print(f"    {v}  n={n}")
