"""Pull the delta since the local DB upload — what has Railway observed
in the few hours since it resumed writing?"""
import sqlite3
import sys
from pathlib import Path

for p in (Path("/app/surveillance/data/surveillance.db"),
          Path("surveillance/data/surveillance.db")):
    if p.exists():
        DB = p
        break

c = sqlite3.connect(str(DB), timeout=30)
c.row_factory = sqlite3.Row

CUTOFF = "2026-04-22T00:28:00"  # just before my DB upload finalized

print(f"=== Delta since {CUTOFF} (post-restore window) ===\n")

# New confirmed contracts
print("## New CONFIRMED contracts")
rows = c.execute("""
    SELECT contract_address, chain, detection_method, detection_timestamp,
           confidence_reason, deployer_address, bytecode_pattern_notes
    FROM contracts
    WHERE confidence_tier = 'confirmed'
      AND detection_timestamp > ?
    ORDER BY detection_timestamp DESC
""", (CUTOFF,)).fetchall()
for r in rows:
    print(f"  {r['contract_address']}  [{r['chain']}]  {r['detection_timestamp']}")
    print(f"    method={r['detection_method']}  deployer={r['deployer_address']}")
    print(f"    reason: {(r['confidence_reason'] or '')[:120]}")
    print(f"    bytecode: {(r['bytecode_pattern_notes'] or '')[:120]}")
print(f"  -- {len(rows)} new confirmed")
print()

# New trap_events (observable harm)
print("## New trap_events (observable harm)")
rows = c.execute("""
    SELECT timestamp, trap_contract_address, bot_address, tx_hash,
           loss_estimate_usd, failure_signature
    FROM trap_events
    WHERE timestamp > ?
    ORDER BY timestamp DESC
""", (CUTOFF,)).fetchall()
for r in rows:
    print(f"  {r['timestamp']}  trap={r['trap_contract_address']}  bot={r['bot_address']}")
    print(f"    tx={r['tx_hash']}  loss_usd={r['loss_estimate_usd']}  sig={r['failure_signature']}")
print(f"  -- {len(rows)} new trap events")
print()

# New alerts by type
print("## New alerts by type (non-FP)")
for r in c.execute("""
    SELECT alert_type, COUNT(*) AS n
    FROM alerts
    WHERE timestamp > ? AND COALESCE(false_positive,0) = 0
    GROUP BY alert_type ORDER BY n DESC
""", (CUTOFF,)):
    print(f"  {r['alert_type']:<35} {r['n']:>5}")
print()

# New deployers — anyone with mainnet_first_tx already populated?
print("## New deployers with mainnet_first_tx already set (Pattern D)")
rows = c.execute("""
    SELECT deployer_address, chain, first_seen, total_contracts_deployed,
           mainnet_first_tx, funding_trail
    FROM deployers
    WHERE first_seen > ?
      AND mainnet_first_tx IS NOT NULL
      AND mainnet_first_tx != ''
    ORDER BY first_seen DESC
    LIMIT 15
""", (CUTOFF,)).fetchall()
for r in rows:
    import json as _j
    funder = ""
    try:
        t = _j.loads(r['funding_trail']) if r['funding_trail'] else {}
        funder = str(t.get('funder',''))[:14]
    except Exception:
        pass
    print(f"  {r['deployer_address']}  [{r['chain']}]  first={r['first_seen']}")
    print(f"    mainnet={r['mainnet_first_tx'][:19]}  deployed={r['total_contracts_deployed']}  funder={funder}")
print(f"  -- showing top 15 of Pattern D candidates")
print()

# Total new deployers
n_total = c.execute(
    "SELECT COUNT(*) FROM deployers WHERE first_seen > ?", (CUTOFF,)
).fetchone()[0]
n_pat_d = c.execute("""
    SELECT COUNT(*) FROM deployers WHERE first_seen > ?
      AND mainnet_first_tx IS NOT NULL AND mainnet_first_tx != ''
""", (CUTOFF,)).fetchone()[0]
print(f"new deployers total: {n_total}  with mainnet Pattern D: {n_pat_d} ({100*n_pat_d/max(n_total,1):.0f}%)")
print()

# Most-interactive new contract (EOA visibility per P2 observation_capability)
print("## Top new contracts by distinct EOAs (observation_capability signal)")
for r in c.execute("""
    SELECT c.contract_address, c.confidence_tier, c.chain,
           COUNT(DISTINCT te.interacting_address) AS eoas,
           COUNT(*) AS tx_n
    FROM contracts c
    JOIN transaction_events te ON te.contract_address = c.contract_address
    WHERE c.detection_timestamp > ?
    GROUP BY c.contract_address
    HAVING COUNT(DISTINCT te.interacting_address) > 1
    ORDER BY eoas DESC
    LIMIT 10
"""):
    print(f"  {r['contract_address']}  [{r['chain']}]  tier={r['confidence_tier']}  eoas={r['eoas']}  tx={r['tx_n']}")
print()

# Money-motion sample
print("## Money-motion alerts in window")
for r in c.execute("""
    SELECT timestamp, alert_type, address, tx_hash,
           SUBSTR(payload, 1, 120) AS pl
    FROM alerts
    WHERE timestamp > ? AND COALESCE(false_positive,0) = 0
      AND (alert_type LIKE 'X402_%' OR alert_type LIKE 'DRAIN%'
           OR alert_type LIKE 'LAUNDRY%' OR alert_type LIKE 'CASHOUT%'
           OR alert_type LIKE 'BRIDGE%' OR alert_type LIKE 'DORMANT%'
           OR alert_type LIKE 'LIVE_%')
    ORDER BY timestamp DESC LIMIT 15
""", (CUTOFF,)):
    print(f"  {r['timestamp']}  {r['alert_type']:<32} {str(r['address'])[:14]}  {r['pl'][:90]}")
print()

# org_candidates count (will auto-run at 04:45 UTC)
print("## org_candidates snapshot")
oc = c.execute("SELECT COUNT(*), status FROM org_candidates GROUP BY status").fetchall()
for r in oc:
    print(f"  status={r[1]}: {r[0]}")
print()

# x402 activity sample
print("## x402 activity in window (top 5 by amount)")
try:
    for r in c.execute("""
        SELECT timestamp, chain, facilitator_address, payer_address, payee_address,
               token_symbol, amount, x402_type
        FROM x402_events WHERE timestamp > ?
        ORDER BY COALESCE(amount,0) DESC LIMIT 5
    """, (CUTOFF,)):
        print(f"  {r['timestamp']}  [{r['chain']}] {r['x402_type']:<10} {r['amount']} {r['token_symbol']}")
        print(f"    payer={r['payer_address']}  payee={r['payee_address']}  facilitator={r['facilitator_address']}")
except Exception as e:
    print(f"  err: {e}")
print()

c.close()
