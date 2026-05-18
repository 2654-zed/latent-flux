"""Review NEW suspected contracts in the post-wave window (2026-05-16 onwards).

The May 9-15 drain wave is over. Question: what's brewing in the corpus
since then? Anything that looks like the early-accumulation phase of a
new wave, novel bytecode patterns, or fresh operators?
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def hdr(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ============================================================
# 1. New SUSPECTED contracts since May-15 with the highest approval volume
#    (= candidates for the next Pattern A discharge)
# ============================================================
hdr("1. New suspected contracts (deployed >=2026-05-15) by approval volume")
print(f"{'contract':44s}  {'chain':8s}  {'deployed':10s}  {'approvals':>9s}  {'victims':>7s}  {'drains':>6s}")
for r in c.execute(
    """SELECT c.contract_address, c.chain, substr(c.detection_timestamp,1,10) AS d,
              COUNT(aw.id) AS appr, COUNT(DISTINCT aw.victim_address) AS vics,
              SUM(CASE WHEN aw.drain_detected=1 THEN 1 ELSE 0 END) AS drains,
              c.deployer_address
       FROM contracts c
       LEFT JOIN approval_watchlist aw ON aw.contract_address = c.contract_address
       WHERE c.confidence_tier = 'suspected'
         AND c.detection_timestamp >= '2026-05-15'
       GROUP BY c.contract_address
       HAVING appr > 0
       ORDER BY appr DESC LIMIT 15"""
):
    print(f"  {r[0]:44s}  {r[1]:8s}  {r[2]:10s}  {r[3]:>9}  {r[4]:>7}  {r[5]:>6}")


# ============================================================
# 2. Contracts in PRE-DISCHARGE phase: lots of recent approvals, 0 drains yet
# ============================================================
hdr("2. Pre-discharge candidates (>=50 approvals last 7 days, 0 drains)")
for r in c.execute(
    """SELECT aw.contract_address,
              COUNT(*) AS recent_appr,
              COUNT(DISTINCT aw.victim_address) AS vics,
              MAX(aw.approve_timestamp) AS last_appr
       FROM approval_watchlist aw
       WHERE aw.approve_timestamp >= '2026-05-11'
         AND aw.drain_detected = 0
         AND NOT EXISTS (
             SELECT 1 FROM approval_watchlist aw2
             WHERE aw2.contract_address = aw.contract_address
               AND aw2.drain_detected = 1
         )
       GROUP BY aw.contract_address
       HAVING recent_appr >= 50
       ORDER BY recent_appr DESC LIMIT 15"""
):
    ctr = c.execute(
        "SELECT chain, deployer_address, confidence_tier, detection_timestamp, deployed_code_hash FROM contracts WHERE contract_address=?",
        (r[0],)
    ).fetchone()
    chain = ctr[0] if ctr else "?"
    dep = ctr[1] if ctr else "?"
    tier = ctr[2] if ctr else "?"
    age_day = (ctr[3] or "")[:10] if ctr else ""
    ch = (ctr[4] or "")[:18] if ctr else ""
    wl = c.execute(
        "SELECT entity_name, priority FROM watchlist WHERE address=? AND active=1",
        (dep,)
    ).fetchone()
    print(f"  {r[0]}  chain={chain}  tier={tier}  age_start={age_day}")
    print(f"    recent_appr={r[1]:>4}  vics={r[2]:>4}  last={r[3][:19]}")
    print(f"    deployer={dep}  code_hash={ch}")
    print(f"    watchlist: {wl or '(off)'}")
    print()


# ============================================================
# 3. NEW bytecode hashes — never seen before, appearing since 2026-05-15
# ============================================================
hdr("3. NEW bytecode hashes (first seen >=2026-05-15)")
# A "new" hash is one whose earliest contract detection is on or after 2026-05-15
for r in c.execute(
    """SELECT deployed_code_hash,
              COUNT(*) AS contracts,
              MIN(detection_timestamp) AS first_seen,
              COUNT(DISTINCT deployer_address) AS deployers,
              SUM(CASE WHEN confidence_tier='suspected' THEN 1 ELSE 0 END) AS susp,
              SUM(CASE WHEN confidence_tier='confirmed' THEN 1 ELSE 0 END) AS conf
       FROM contracts
       WHERE deployed_code_hash IS NOT NULL
         AND deployed_code_hash != ''
       GROUP BY deployed_code_hash
       HAVING first_seen >= '2026-05-15'
         AND contracts >= 3
       ORDER BY contracts DESC LIMIT 15"""
):
    h = (r[0] or "")[:60]
    print(f"  {h:60s}")
    print(f"    contracts={r[1]:>5}  deployers={r[3]:>4}  susp={r[4]:>4}  conf={r[5]:>3}  first_seen={r[2][:19]}")


# ============================================================
# 4. New OPERATORS: deployers first_seen >=2026-05-15 with high contract count
#    OR funded by a known operator
# ============================================================
hdr("4. New deployers since 2026-05-15 with > 1 contract")
for r in c.execute(
    """SELECT d.deployer_address, d.chain, substr(d.first_seen,1,10) AS d_seen,
              d.total_contracts_deployed,
              d.mainnet_first_tx,
              d.funding_trail
       FROM deployers d
       WHERE d.first_seen >= '2026-05-15'
         AND d.total_contracts_deployed > 1
       ORDER BY d.total_contracts_deployed DESC LIMIT 15"""
):
    has_mainnet = "yes" if r[4] else "no"
    ft_short = (r[5] or "")[:100]
    print(f"  {r[0]}  chain={r[1]:9s}  first_seen={r[2]}  contracts={r[3]:>3}  mainnet_id={has_mainnet}")
    if r[4]:
        from datetime import datetime, timezone
        try:
            mn = datetime.fromisoformat(r[4].replace("Z","+00:00"))
            ls = datetime.fromisoformat(r[2] + "T00:00:00+00:00")
            gap = (ls - mn).days
            print(f"    mainnet_first_tx={r[4][:10]}  GAP={gap} days (Pattern D candidate if >=60)")
        except Exception:
            pass
    if r[5]:
        # Extract funder
        import json
        try:
            ft = json.loads(r[5])
            funder = ft.get("funder") if isinstance(ft, dict) else None
        except Exception:
            funder = None
        if funder:
            wl = c.execute("SELECT entity_name FROM watchlist WHERE address=?", (funder,)).fetchone()
            oli = c.execute("SELECT severity FROM oli_labels WHERE address=?", (funder,)).fetchone()
            tag = f"watchlist={wl[0]}" if wl else (f"OLI={oli[0]}" if oli else "unknown")
            print(f"    funder={funder}  [{tag}]")
    print()


# ============================================================
# 5. Bytecode-family clustering: are new deployers all spawning same template?
# ============================================================
hdr("5. Bytecode hashes most common among May-15+ new suspected contracts")
for r in c.execute(
    """SELECT deployed_code_hash,
              COUNT(*) AS contracts,
              COUNT(DISTINCT deployer_address) AS deployers,
              MIN(detection_timestamp) AS first_seen
       FROM contracts
       WHERE confidence_tier = 'suspected'
         AND detection_timestamp >= '2026-05-15'
         AND deployed_code_hash IS NOT NULL
       GROUP BY deployed_code_hash
       HAVING contracts >= 5
       ORDER BY contracts DESC LIMIT 12"""
):
    h = (r[0] or "")[:60]
    print(f"  {h:60s}  contracts={r[1]:>4}  deployers={r[2]:>4}  first={r[3][:10]}")


# ============================================================
# 6. Drain executors that appeared post-wave (since 2026-05-15)
# ============================================================
hdr("6. New drain executors since 2026-05-15 (post-wave residual activity)")
for r in c.execute(
    """SELECT drain_caller, COUNT(*) AS drains,
              COUNT(DISTINCT contract_address) AS contracts,
              MIN(drain_timestamp) AS first_drain
       FROM approval_watchlist
       WHERE drain_detected = 1
         AND drain_timestamp >= '2026-05-15'
       GROUP BY drain_caller
       ORDER BY drains DESC LIMIT 10"""
):
    wl = c.execute("SELECT entity_name, priority FROM watchlist WHERE address=?", (r[0],)).fetchone()
    print(f"  {r[0]}  drains={r[1]:>4}  contracts={r[2]:>3}  first={r[3][:19]}")
    print(f"    watchlist: {wl or '(off)'}")


# ============================================================
# 7. Watchlisted operators with NEW deployments in last 3 days
# ============================================================
hdr("7. Watchlisted operators deploying since 2026-05-15 (active threats)")
for r in c.execute(
    """SELECT w.entity_name, w.priority, w.address,
              COUNT(c.contract_address) AS recent_deploys,
              MAX(c.detection_timestamp) AS last_deploy
       FROM watchlist w
       LEFT JOIN contracts c ON c.deployer_address = w.address
                              AND c.detection_timestamp >= '2026-05-15'
       WHERE w.active = 1
       GROUP BY w.address
       HAVING recent_deploys > 0
       ORDER BY recent_deploys DESC LIMIT 15"""
):
    print(f"  {r[2]}  recent_deploys={r[3]:>3}  last={r[4][:19] if r[4] else '?'}")
    print(f"    entity_name: {r[0]}  ({r[1]})")
