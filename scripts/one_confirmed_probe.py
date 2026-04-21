"""Pull the single confirmed contract from the last 4 days with full context."""
import json
import sqlite3
import sys
from pathlib import Path

DB = None
for p in (Path("/app/surveillance/data/surveillance.db"),
          Path("surveillance/data/surveillance.db")):
    if p.exists():
        DB = p
        break
if DB is None:
    print("NO DB", file=sys.stderr); sys.exit(1)

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
from surveillance.risk_scoring import score_contract

print("=" * 80)
print("  The one confirmed contract from 2026-04-17 onward")
print("=" * 80)

row = c.execute("""
    SELECT * FROM contracts
    WHERE confidence_tier = 'confirmed'
      AND detection_timestamp > datetime('now','-4 days')
    ORDER BY detection_timestamp
""").fetchone()

if not row:
    print("no confirmed contract in window")
    sys.exit(0)

addr = row["contract_address"]
print()
print("## Contract row")
for k in row.keys():
    v = row[k]
    if v is None:
        continue
    print(f"  {k:<28} {str(v)[:120]}")
print()

# Deployer profile
dep = c.execute(
    "SELECT * FROM deployers WHERE deployer_address = ?", (row["deployer_address"],)
).fetchone()
if dep:
    print("## Deployer")
    for k in dep.keys():
        v = dep[k]
        if v is None or v == "":
            continue
        print(f"  {k:<32} {str(v)[:120]}")
print()

# All trap_events linked to this contract
print("## trap_events (max 5)")
te_rows = c.execute("""
    SELECT timestamp, bot_address, tx_hash, loss_estimate_usd, failure_signature
    FROM trap_events
    WHERE LOWER(trap_contract_address) = LOWER(?)
    ORDER BY timestamp LIMIT 5
""", (addr,)).fetchall()
total_loss = c.execute(
    "SELECT SUM(loss_estimate_usd), COUNT(*) FROM trap_events WHERE LOWER(trap_contract_address) = LOWER(?)",
    (addr,),
).fetchone()
print(f"  total trap_events: {total_loss[1]}   total loss_estimate_usd: {total_loss[0]}")
for r in te_rows:
    print(f"  {r['timestamp']}  bot={str(r['bot_address'])[:14]}  tx={str(r['tx_hash'])[:14]}  loss={r['loss_estimate_usd']}  sig={r['failure_signature']}")
print()

# Tx activity snapshot
print("## Recent transaction activity")
summary = c.execute("""
    SELECT COUNT(*) AS n_tx, COUNT(DISTINCT interacting_address) AS n_callers,
           SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) AS reverts,
           MIN(timestamp) AS first_tx, MAX(timestamp) AS last_tx
    FROM transaction_events WHERE contract_address = ?
""", (addr,)).fetchone()
print(f"  tx total: {summary['n_tx']}  unique callers: {summary['n_callers']}  reverts: {summary['reverts']}")
print(f"  first tx: {summary['first_tx']}  last tx: {summary['last_tx']}")
if summary['n_tx']:
    print("  top 5 selectors:")
    for r in c.execute("""
        SELECT function_selector, COUNT(*) n, SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) AS rev
        FROM transaction_events WHERE contract_address = ?
        GROUP BY function_selector ORDER BY n DESC LIMIT 5
    """, (addr,)):
        print(f"    sel={r['function_selector']} count={r['n']}  reverts={r['rev']}")
print()

# Alerts on this address
print("## Alerts on this contract")
for r in c.execute("""
    SELECT timestamp, alert_type, payload FROM alerts
    WHERE LOWER(address) = LOWER(?)
    ORDER BY timestamp DESC LIMIT 8
""", (addr,)):
    pl = (r["payload"] or "")[:100].replace("\n", " ")
    print(f"  {r['timestamp']}  {r['alert_type']:<32}  {pl}")
print()

# Full risk score
print("## Risk score (with P2 observation_capability)")
r = score_contract(c, addr)
for k in ("stored_potential", "approval_scope_score", "capability_score",
          "deployer_risk_score", "org_context_score",
          "observation_capability_score", "realized_value", "volatility",
          "risk_score", "risk_tier"):
    print(f"  {k:<32} {r.get(k)}")
print()
print("## Risk components breakdown")
for k, v in r["components"].items():
    print(f"  {k}:")
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"    {kk}: {str(vv)[:120]}")
    else:
        print(f"    {v}")
c.close()
