"""Pull Railway state needed to document the consumable intelligence surface.

Output shape:
  - per-table row count + freshness timestamp where available
  - per-table column list
  - confidence_tier distribution snapshot
  - alert-type distribution (last 24h + lifetime)
  - infrastructure_registry classification breakdown
  - org_wallets + org_candidates current status
  - camouflage live numbers (pop vs adversary vs confirmed)
"""
import json
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

print("=" * 72)
print("  Railway state snapshot for consumable-intelligence inventory")
print("=" * 72)
print(f"  DB: {DB}  size={DB.stat().st_size:,} bytes")
print()

# --- All tables + row counts ---
all_tables = [r[0] for r in c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
    "ORDER BY name"
).fetchall()]
print(f"## {len(all_tables)} tables")
print()
print(f"{'table':<38} {'rows':>10}")
print("-" * 52)
sizes = {}
for t in all_tables:
    try:
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        sizes[t] = n
        print(f"{t:<38} {n:>10,}")
    except Exception as e:
        print(f"{t:<38} ERR: {e}")
print()

# --- Per-table column shape for the key consumable tables ---
key_tables = [
    "contracts", "deployers", "deployer_profiles", "trap_events",
    "alerts", "extraction_events", "org_wallets", "org_candidates",
    "infrastructure_registry", "bytecode_families", "bytecode_family_members",
    "entity_classification", "camouflage_metrics", "daily_metrics",
    "false_positives", "trust_amplification", "approval_watchlist",
    "drain_events", "bot_candidates", "bot_strategies",
    "behavioral_anomalies", "deployer_similarity", "predictions",
    "strategy_lifecycle", "bait_profiles", "x402_events",
    "x402_permit2_exposure", "live_exposures", "watchlist",
    "diamond_model", "connection_gaps",
]
print(f"## Column shapes for {len(key_tables)} key tables")
print()
for t in key_tables:
    if t not in sizes:
        print(f"### {t}  (table missing)")
        continue
    print(f"### {t}  rows={sizes[t]:,}")
    for col in c.execute(f"PRAGMA table_info({t})").fetchall():
        print(f"  {col[1]:<34} {col[2]}")
    print()

# --- Confidence tier breakdown with decay ---
print("## confidence_tier distribution (production)")
for r in c.execute(
    "SELECT confidence_tier, COUNT(*), "
    "       SUM(CASE WHEN decayed_at IS NOT NULL THEN 1 ELSE 0 END) AS decayed "
    "FROM contracts GROUP BY confidence_tier ORDER BY 2 DESC"
):
    print(f"  {r[0]:<15} {r[1]:>8,}  decayed={r[2]:>6,}")
print()

# --- Alert type distribution (last 24h + lifetime) ---
print("## Alert types — last 24h (non-FP)")
for r in c.execute(
    "SELECT alert_type, COUNT(*) "
    "FROM alerts WHERE COALESCE(false_positive,0)=0 "
    "  AND timestamp > datetime('now','-24 hours') "
    "GROUP BY alert_type ORDER BY 2 DESC LIMIT 20"
):
    print(f"  {r[0]:<40} {r[1]:>5}")
print()

print("## Alert types — lifetime (non-FP, top 15)")
for r in c.execute(
    "SELECT alert_type, COUNT(*) "
    "FROM alerts WHERE COALESCE(false_positive,0)=0 "
    "GROUP BY alert_type ORDER BY 2 DESC LIMIT 15"
):
    print(f"  {r[0]:<40} {r[1]:>5}")
print()

# --- Infrastructure registry ---
print("## infrastructure_registry classifications")
for r in c.execute(
    "SELECT classification, COUNT(*) FROM infrastructure_registry GROUP BY classification ORDER BY 2 DESC"
):
    print(f"  {r[0]:<40} {r[1]}")
print()

# --- org_wallets + org_candidates ---
print("## org_wallets snapshot")
for r in c.execute("SELECT org_id, COUNT(*) FROM org_wallets GROUP BY org_id ORDER BY 2 DESC"):
    print(f"  {r[0]:<15} {r[1]}")
print()
print("## org_candidates snapshot")
for r in c.execute("SELECT status, COUNT(*) FROM org_candidates GROUP BY status ORDER BY 2 DESC"):
    print(f"  {r[0]:<15} {r[1]}")
print()

# --- Camouflage numbers (live) ---
print("## Camouflage — lifetime (population vs adversary vs confirmed)")
try:
    pop = c.execute("""
        SELECT COUNT(*), SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END)
        FROM (SELECT contract_address,
                     CAST(SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) AS REAL)/COUNT(*) AS rr
              FROM transaction_events GROUP BY contract_address HAVING COUNT(*) >= 10)
    """).fetchone()
    adv = c.execute("""
        SELECT COUNT(*), SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END)
        FROM (SELECT te.contract_address,
                     CAST(SUM(CASE WHEN te.is_reverted=1 THEN 1 ELSE 0 END) AS REAL)/COUNT(*) AS rr
              FROM transaction_events te
              JOIN contracts c ON c.contract_address = te.contract_address
              WHERE c.confidence_tier IN ('confirmed','suspected')
              GROUP BY te.contract_address HAVING COUNT(*) >= 10)
    """).fetchone()
    conf = c.execute("""
        SELECT COUNT(*), SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END)
        FROM (SELECT te.contract_address,
                     CAST(SUM(CASE WHEN te.is_reverted=1 THEN 1 ELSE 0 END) AS REAL)/COUNT(*) AS rr
              FROM transaction_events te
              JOIN contracts c ON c.contract_address = te.contract_address
              WHERE c.confidence_tier = 'confirmed'
              GROUP BY te.contract_address HAVING COUNT(*) >= 10)
    """).fetchone()
    print(f"  population: {pop[0]} total, {pop[1]} low-revert, ratio={pop[1]/max(pop[0],1):.3f}")
    print(f"  adversary:  {adv[0]} total, {adv[1]} low-revert, ratio={adv[1]/max(adv[0],1):.3f}")
    print(f"  confirmed:  {conf[0]} total, {conf[1]} low-revert, ratio={conf[1]/max(conf[0],1):.3f}")
except Exception as e:
    print(f"  err: {e}")
print()

# --- observable-harm PPV by tier ---
print("## PPV by confidence_tier (trap_events as outcome proxy)")
for t in ("confirmed", "suspected", "unanalyzed", "unknown"):
    total = c.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier = ?", (t,)).fetchone()[0]
    with_harm = c.execute("""
        SELECT COUNT(DISTINCT c.contract_address)
        FROM contracts c
        JOIN trap_events te ON LOWER(te.trap_contract_address) = LOWER(c.contract_address)
        WHERE c.confidence_tier = ?
    """, (t,)).fetchone()[0]
    ppv = with_harm / total if total else 0
    print(f"  {t:<12}  total={total:>8,}  with_harm={with_harm:>6}  ppv={100*ppv:>6.2f}%")
print()

# --- Most recent extraction_events ---
print("## extraction_events (documented)")
for r in c.execute("""
    SELECT event_id, event_type, event_type_suggestion, chain, monitored_chain,
           total_usd_moved, observed_at
    FROM extraction_events ORDER BY id
"""):
    print(f"  {r['event_id']}  type={r['event_type']:<42}  suggestion={r['event_type_suggestion']}  chain={r['chain']}  mon={r['monitored_chain']}  usd={r['total_usd_moved']}")
print()

# --- Scheduler job cadence (from code reality; no DB source) ---
print("## Most recent daily_metrics / camouflage_metrics / predictions rows")
for t in ("daily_metrics", "camouflage_metrics", "predictions"):
    try:
        r = c.execute(f"SELECT * FROM {t} ORDER BY date DESC LIMIT 1").fetchone()
        if r:
            d = dict(r)
            print(f"  {t}: date={d.get('date')}  chain={d.get('chain','?')}  keys={list(d.keys())[:8]}")
    except Exception as e:
        print(f"  {t}: err {e}")
print()

c.close()
