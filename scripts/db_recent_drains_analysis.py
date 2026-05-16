"""Broad analysis of recent corpus activity + drains. Looks for things that
stand out from baseline: spikes, new operators, cross-chain patterns,
mappings onto v3 attack categories.

Run window: 2026-05-01 .. 2026-05-15 (last 15 days). Compare against
2026-04-15 .. 2026-04-30 baseline (prior 15-day window).
"""
from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def hdr(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def short(addr, n=10):
    return addr[:n] + ".." if addr and len(addr) > n else (addr or "")


# ============================================================
# Section 1: Recent drain landscape
# ============================================================

hdr("Section 1: approval_watchlist drains by day, 2026-05-01..05-15")
for r in c.execute(
    """SELECT substr(drain_timestamp, 1, 10) AS d,
              COUNT(*) AS drains,
              COUNT(DISTINCT victim_address) AS victims,
              COUNT(DISTINCT drain_caller) AS drainers
       FROM approval_watchlist
       WHERE drain_detected=1
         AND drain_timestamp >= '2026-05-01'
         AND drain_timestamp < '2026-05-16'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  drains={r[1]:>5}  victims={r[2]:>5}  drainers={r[3]:>4}")


hdr("Section 1b: Apr-15..30 baseline (drains by day)")
for r in c.execute(
    """SELECT substr(drain_timestamp, 1, 10) AS d, COUNT(*)
       FROM approval_watchlist
       WHERE drain_detected=1
         AND drain_timestamp >= '2026-04-15'
         AND drain_timestamp < '2026-05-01'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  drains={r[1]:>5}")


# ============================================================
# Section 2: Top drainers in last 14 days, cross-ref watchlist
# ============================================================

hdr("Section 2: Top drain_callers in 2026-05-01..05-15 (volume + watchlist status)")
watchlist_addrs = {r[0].lower() for r in c.execute(
    "SELECT address FROM watchlist WHERE active=1 AND address IS NOT NULL"
)}
watchlist_info = {r[0].lower(): (r[1], r[2], r[3]) for r in c.execute(
    "SELECT address, entity_name, priority, watch_reason FROM watchlist WHERE active=1 AND address IS NOT NULL"
) if r[0]}
print(f"  (watchlist active: {len(watchlist_addrs)} entries)")
print(f"  {'drainer':44s}  {'count':>6s}  {'first_drain':12s}  {'on_watchlist':>12s}")
for r in c.execute(
    """SELECT drain_caller,
              COUNT(*) AS n,
              substr(MIN(drain_timestamp), 1, 10) AS first
       FROM approval_watchlist
       WHERE drain_detected=1
         AND drain_timestamp >= '2026-05-01'
         AND drain_timestamp < '2026-05-16'
       GROUP BY drain_caller
       ORDER BY n DESC LIMIT 20"""
):
    in_wl = r[0] and r[0].lower() in watchlist_addrs
    flag = watchlist_info.get(r[0].lower(), ("(not on watchlist)", "", ""))[0] if r[0] else "?"
    pri = watchlist_info.get(r[0].lower(), ("", "", ""))[1] if r[0] else ""
    print(f"  {r[0]:44s}  {r[1]:>6}  {r[2]:12s}  {flag[:30]:30s}  {pri}")


# ============================================================
# Section 3: Top drained contracts (last 14 days) — bytecode families
# ============================================================

hdr("Section 3: Top contracts being drained-from, 2026-05-01..05-15 + bytecode")
top_contracts_rows = list(c.execute(
    """SELECT contract_address, COUNT(*) AS n,
              COUNT(DISTINCT victim_address) AS victims,
              COUNT(DISTINCT drain_caller) AS drainers
       FROM approval_watchlist
       WHERE drain_detected=1
         AND drain_timestamp >= '2026-05-01'
         AND drain_timestamp < '2026-05-16'
       GROUP BY contract_address
       ORDER BY n DESC LIMIT 15"""
))
print(f"  {'contract':44s}  {'n':>4s}  {'victims':>7s}  {'drainers':>8s}  {'chain':8s}  {'tier':10s}  {'code_hash':20s}")
for r in top_contracts_rows:
    ctr = c.execute(
        "SELECT chain, confidence_tier, deployed_code_hash, deployer_address FROM contracts WHERE contract_address=?",
        (r[0],)
    ).fetchone()
    chain = ctr[0] if ctr else "?"
    tier = ctr[1] if ctr else "?"
    ch = (ctr[2] or "")[:18] if ctr else ""
    print(f"  {r[0]:44s}  {r[1]:>4}  {r[2]:>7}  {r[3]:>8}  {chain:8s}  {tier:10s}  {ch:20s}")


# ============================================================
# Section 4: Recent trap_events
# ============================================================

hdr("Section 4: trap_events 2026-05-01..05-15 daily counts")
for r in c.execute(
    """SELECT substr(timestamp,1,10), COUNT(*),
              COUNT(DISTINCT trap_contract_address) AS distinct_traps,
              COUNT(DISTINCT bot_address) AS distinct_victims
       FROM trap_events
       WHERE timestamp >= '2026-05-01' AND timestamp < '2026-05-16'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  events={r[1]:>4}  traps={r[2]:>4}  victims={r[3]:>4}")


hdr("Section 4b: Top trap contracts hitting bots, 2026-05-01..05-15")
for r in c.execute(
    """SELECT trap_contract_address, COUNT(*) AS hits,
              COUNT(DISTINCT bot_address) AS distinct_bots
       FROM trap_events
       WHERE timestamp >= '2026-05-01' AND timestamp < '2026-05-16'
       GROUP BY trap_contract_address
       ORDER BY hits DESC LIMIT 10"""
):
    ctr = c.execute(
        "SELECT chain, deployer_address FROM contracts WHERE contract_address=?",
        (r[0],)
    ).fetchone()
    chain = ctr[0] if ctr else "?"
    dep = ctr[1] if ctr else "?"
    print(f"  {r[0]:44s}  hits={r[1]:>4}  bots={r[2]:>4}  chain={chain:9s}  deployer={dep}")


# ============================================================
# Section 5: x402 surge analysis (1.59M events!)
# ============================================================

hdr("Section 5: x402 events daily for full window 2026-04-22..05-15")
for r in c.execute(
    """SELECT substr(timestamp,1,10), COUNT(*)
       FROM x402_events
       WHERE timestamp >= '2026-04-22' AND timestamp < '2026-05-16'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  events={r[1]:>7,}")


hdr("Section 5b: x402 — top facilitators in last 14 days")
for r in c.execute(
    """SELECT facilitator_address, COUNT(*) AS n, COUNT(DISTINCT payer_address) AS payers,
              COUNT(DISTINCT payee_address) AS payees
       FROM x402_events
       WHERE timestamp >= '2026-05-01' AND timestamp < '2026-05-16'
       GROUP BY facilitator_address ORDER BY n DESC LIMIT 10"""
):
    print(f"  {r[0]:44s}  n={r[1]:>7,}  payers={r[2]:>6,}  payees={r[3]:>5,}")


# ============================================================
# Section 6: 2026-05-09 approval spike characterization
# ============================================================

hdr("Section 6: What's special about 2026-05-09? (regime alert: approval=6,446)")
print("  Top contracts receiving approvals on May 9:")
for r in c.execute(
    """SELECT contract_address, COUNT(*) AS n, COUNT(DISTINCT victim_address) AS vics
       FROM approval_watchlist
       WHERE approve_timestamp >= '2026-05-09' AND approve_timestamp < '2026-05-10'
       GROUP BY contract_address ORDER BY n DESC LIMIT 10"""
):
    ctr = c.execute(
        "SELECT chain, deployer_address, confidence_tier FROM contracts WHERE contract_address=?",
        (r[0],)
    ).fetchone()
    chain = ctr[0] if ctr else "?"
    dep = ctr[1] if ctr else "?"
    tier = ctr[2] if ctr else "?"
    print(f"  {r[0]:44s}  n={r[1]:>5}  vics={r[2]:>5}  chain={chain:9s}  tier={tier:10s}")
    print(f"    deployer={dep}")


hdr("Section 6b: May 9 approval-events how many led to drains?")
drained = c.execute(
    """SELECT COUNT(*) FROM approval_watchlist
       WHERE approve_timestamp >= '2026-05-09' AND approve_timestamp < '2026-05-10'
         AND drain_detected=1"""
).fetchone()[0]
total = c.execute(
    """SELECT COUNT(*) FROM approval_watchlist
       WHERE approve_timestamp >= '2026-05-09' AND approve_timestamp < '2026-05-10'"""
).fetchone()[0]
print(f"  May-9 total approvals: {total}")
print(f"  May-9 approvals that became drains: {drained}  ({100*drained/total:.1f}%)")


# ============================================================
# Section 7: Recent dormant_activations
# ============================================================

hdr("Section 7: dormant_activations 2026-05-01..05-15 daily")
for r in c.execute(
    """SELECT substr(first_interaction_timestamp,1,10) AS d, COUNT(*)
       FROM dormant_activations
       WHERE first_interaction_timestamp >= '2026-05-01' AND first_interaction_timestamp < '2026-05-16'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  {r[1]:>5}")


hdr("Section 7b: Top dormant-activation contracts (any time) by fleet_size")
for r in c.execute(
    """SELECT contract_address, deployer_address, chain, fleet_size, fleet_active_before, fleet_active_after,
              substr(first_interaction_timestamp,1,10)
       FROM dormant_activations
       WHERE first_interaction_timestamp >= '2026-05-01'
       ORDER BY fleet_size DESC LIMIT 10"""
):
    print(f"  contract={r[0]}  dep={r[1]}  chain={r[2]}")
    print(f"    fleet_size={r[3]}  before={r[4]}  after={r[5]}  date={r[6]}")


# ============================================================
# Section 8: Pattern D — recent deployers with mainnet first tx
# ============================================================

hdr("Section 8: Pattern D — recent deployers with mainnet_first_tx (cross-chain reputation import)")
# Recent deployers (first_seen since 2026-05-09) with mainnet_first_tx not null
# AND the gap is >60 days
for r in c.execute(
    """SELECT deployer_address, chain, first_seen, mainnet_first_tx, total_contracts_deployed, behavioral_score
       FROM deployers
       WHERE first_seen >= '2026-05-09'
         AND mainnet_first_tx IS NOT NULL
         AND mainnet_first_tx < '2026-03-01'
       ORDER BY mainnet_first_tx ASC LIMIT 15"""
):
    print(f"  dep={r[0]}  chain={r[1]:9s}  L2_first={r[2][:10]}  mainnet_first={r[3][:10]}  contracts={r[4]}  score={r[5]}")


# ============================================================
# Section 9: extraction_events — the curated case events
# ============================================================

hdr("Section 9: extraction_events (curated case rows)")
for r in c.execute(
    """SELECT event_id, event_type, observed_at, total_usd_moved, nodes_active, chain, summary
       FROM extraction_events ORDER BY observed_at DESC"""
):
    print(f"  {r[0]}  {r[1]:25s}  observed={r[2][:10]}  ${r[3] or 0:>12,.0f}  nodes={r[4]}  chain={r[5]}")
    print(f"    summary: {(r[6] or '')[:120]}")


# ============================================================
# Section 10: Permit2 exposure recency
# ============================================================

hdr("Section 10: x402_permit2_exposure — recent exposures (potential drain surface)")
for r in c.execute(
    """SELECT substr(last_seen, 1, 10), COUNT(*),
              COUNT(DISTINCT owner_address) AS owners,
              COUNT(DISTINCT spender_address) AS spenders
       FROM x402_permit2_exposure
       WHERE last_seen >= '2026-05-01' AND last_seen < '2026-05-16'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  exposures={r[1]:>5}  owners={r[2]:>5}  spenders={r[3]:>4}")
print()
print("  Top spenders by exposure count (any time):")
for r in c.execute(
    """SELECT spender_address, COUNT(*) AS n, COUNT(DISTINCT owner_address) AS owners
       FROM x402_permit2_exposure
       GROUP BY spender_address ORDER BY n DESC LIMIT 10"""
):
    print(f"    {r[0]:44s}  exposures={r[1]:>5}  owners={r[2]:>5}")
