"""Analyze the past few days of Layer 3 activity.

Runs directly on Railway (via railway ssh). Covers the window since the
"advance all" session began (2026-04-17) through now. Focus areas:
  - corpus deltas
  - trap_events arriving during the window
  - alerts by type
  - org_candidates new vs refreshed
  - confidence decay effect on stats
  - any drain events / large movements
  - outage-window gap (19:50-20:38 UTC on 2026-04-20)
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
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

WINDOW_DAYS = 4
now = datetime.now(timezone.utc)
print(f"=== Layer 3 activity report ===")
print(f"Generated at: {now.isoformat()}")
print(f"Window:       last {WINDOW_DAYS} days")
print(f"DB:           {DB}  size={DB.stat().st_size:,} bytes")
print()

def single(sql, *params):
    r = c.execute(sql, params).fetchone()
    return r[0] if r else None

def rows(sql, *params):
    return c.execute(sql, params).fetchall()


# -------- corpus snapshot --------
print("## Corpus snapshot")
print(f"  contracts total:         {single('SELECT COUNT(*) FROM contracts'):>8,}")
for t in ("confirmed", "suspected", "unanalyzed", "unknown"):
    n = single("SELECT COUNT(*) FROM contracts WHERE confidence_tier = ?", t)
    print(f"    {t:<12}           {n:>8,}")
print(f"  deployers:               {single('SELECT COUNT(*) FROM deployers'):>8,}")
print(f"  transaction_events:      {single('SELECT COUNT(*) FROM transaction_events'):>8,}")
print(f"  trap_events:             {single('SELECT COUNT(*) FROM trap_events'):>8,}")
print(f"  alerts total:            {single('SELECT COUNT(*) FROM alerts'):>8,}")
print()

# -------- activity in the window --------
print(f"## Activity in last {WINDOW_DAYS} days")
print()
print(f"  NEW contracts:           {single(f'SELECT COUNT(*) FROM contracts WHERE detection_timestamp > datetime(\"now\", \"-{WINDOW_DAYS} days\")'):>8,}")
for t in ("confirmed", "suspected", "unknown"):
    n = single(f"SELECT COUNT(*) FROM contracts WHERE confidence_tier = ? AND detection_timestamp > datetime('now', '-{WINDOW_DAYS} days')", t)
    print(f"    {t:<12}           {n:>8,}")

print()
print(f"  NEW trap_events:         {single(f'SELECT COUNT(*) FROM trap_events WHERE timestamp > datetime(\"now\", \"-{WINDOW_DAYS} days\")'):>8,}")
print(f"  NEW alerts:              {single(f'SELECT COUNT(*) FROM alerts WHERE timestamp > datetime(\"now\", \"-{WINDOW_DAYS} days\")'):>8,}")
print(f"  NEW deployers:           {single(f'SELECT COUNT(*) FROM deployers WHERE first_seen > datetime(\"now\", \"-{WINDOW_DAYS} days\")'):>8,}")
print()

# -------- alert-type distribution in window --------
print(f"## Alert types in last {WINDOW_DAYS} days (top 12)")
for r in rows(f"""
    SELECT alert_type, COUNT(*) AS n,
           SUM(CASE WHEN COALESCE(false_positive,0)=0 THEN 1 ELSE 0 END) AS real
    FROM alerts
    WHERE timestamp > datetime('now','-{WINDOW_DAYS} days')
    GROUP BY alert_type ORDER BY n DESC LIMIT 12
"""):
    print(f"  {r['alert_type']:<42} total={r['n']:>6}  non-FP={r['real']:>6}")
print()

# -------- chains --------
print(f"## New contracts by chain")
for r in rows(f"""
    SELECT chain, COUNT(*) AS n
    FROM contracts
    WHERE detection_timestamp > datetime('now','-{WINDOW_DAYS} days')
    GROUP BY chain ORDER BY n DESC
"""):
    print(f"  {r['chain']:<12} {r['n']:>6,}")
print()

# -------- drain / x402 / laundry spike check --------
print(f"## Money-motion alerts in last {WINDOW_DAYS} days (sample)")
try:
    for r in rows(f"""
        SELECT timestamp, alert_type, address, tx_hash, payload
        FROM alerts
        WHERE timestamp > datetime('now','-{WINDOW_DAYS} days')
          AND COALESCE(false_positive,0) = 0
          AND (alert_type LIKE 'X402_%'
               OR alert_type LIKE 'DRAIN%'
               OR alert_type LIKE 'LAUNDRY%'
               OR alert_type LIKE 'CASHOUT%'
               OR alert_type LIKE 'MOVEMENT%')
        ORDER BY timestamp DESC LIMIT 10
    """):
        pl = (r["payload"] or "")[:80].replace("\n", " ")
        print(f"  {r['timestamp']}  {r['alert_type']:<25} addr={str(r['address'])[:14]} tx={str(r['tx_hash'])[:14]} pl={pl}")
except Exception as e:
    print(f"  (query failed: {e})")
print()

# -------- decay / unanalyzed confirmation --------
print("## Confidence decay effect (Correction #9)")
decayed = single("SELECT COUNT(*) FROM contracts WHERE decayed_at IS NOT NULL")
print(f"  total decayed:           {decayed:,}")
by_chain = rows("""
    SELECT chain, COUNT(*) AS n FROM contracts
    WHERE decayed_at IS NOT NULL GROUP BY chain
""")
for r in by_chain:
    print(f"    {r['chain']:<12} {r['n']:,}")
last_decay = single("SELECT MAX(decayed_at) FROM contracts WHERE decayed_at IS NOT NULL")
print(f"  last decay timestamp:    {last_decay}")
print()

# -------- outage window diagnostic (19:50-20:38 UTC on 2026-04-20) --------
print("## Outage window check (2026-04-20 19:50-20:38 UTC)")
try:
    outage_contracts = single("""
        SELECT COUNT(*) FROM contracts
        WHERE detection_timestamp BETWEEN '2026-04-20T19:50:00' AND '2026-04-20T20:38:00'
    """)
    print(f"  contracts persisted in window: {outage_contracts}")
    outage_alerts = single("""
        SELECT COUNT(*) FROM alerts
        WHERE timestamp BETWEEN '2026-04-20T19:50:00' AND '2026-04-20T20:38:00'
    """)
    print(f"  alerts persisted in window:    {outage_alerts}")
except Exception as e:
    print(f"  query err: {e}")
print()

# -------- org_candidates --------
print("## org_candidates status")
if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='org_candidates'").fetchone():
    total = single("SELECT COUNT(*) FROM org_candidates")
    pending = single("SELECT COUNT(*) FROM org_candidates WHERE status = 'pending'")
    print(f"  total: {total}  pending: {pending}")
    print(f"  size distribution (bucketed):")
    for r in rows("""
        SELECT CASE
            WHEN cluster_size < 5 THEN '3-4'
            WHEN cluster_size < 10 THEN '5-9'
            WHEN cluster_size < 20 THEN '10-19'
            WHEN cluster_size < 30 THEN '20-29'
            ELSE '30+'
        END AS bucket, COUNT(*) AS n
        FROM org_candidates GROUP BY bucket ORDER BY MIN(cluster_size)
    """):
        print(f"    size {r['bucket']:<6}  {r['n']:,}")
    print(f"  top 5 by cluster size:")
    for r in rows("""
        SELECT candidate_id, cluster_size, shared_chain, shared_funding_source, first_seen, last_seen
        FROM org_candidates ORDER BY cluster_size DESC LIMIT 5
    """):
        print(f"    {r['candidate_id']}  size={r['cluster_size']}  chain={r['shared_chain']}  funder={r['shared_funding_source']}  span={r['first_seen'][:10]}..{r['last_seen'][:10]}")
else:
    print("  table not created yet")
print()

# -------- bot/selector churn --------
print(f"## Bot & selector activity in last {WINDOW_DAYS} days")
print(f"  new bot_candidates:      {single(f'SELECT COUNT(*) FROM bot_candidates WHERE first_seen > datetime(\"now\", \"-{WINDOW_DAYS} days\")'):>6}")
try:
    n_bce = single(f"SELECT COUNT(*) FROM bot_candidate_events WHERE timestamp > datetime('now','-{WINDOW_DAYS} days')")
    print(f"  bot_candidate_events:    {n_bce:>6}")
except Exception as e:
    print(f"  bot_candidate_events: err {e}")
print()

# -------- heartbeat / connection gaps --------
print(f"## Connection stability")
for r in rows(f"""
    SELECT chain, COUNT(*) AS gaps,
           MAX(gap_seconds) AS max_gap,
           SUM(gap_seconds) AS total_gap_s
    FROM connection_gaps
    WHERE start_time > datetime('now','-{WINDOW_DAYS} days')
    GROUP BY chain
"""):
    print(f"  {r['chain']:<12} gaps={r['gaps']:>3}  max_gap={r['max_gap']}s  total_gap={r['total_gap_s']}s")
print()

c.close()
print("=== end report ===")
