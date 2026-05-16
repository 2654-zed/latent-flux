"""Investigate the pristine-reputation solo operator 0x80b12bd0 and
characterize the May-9..14 drain wave (4,587 + 548 + 362 + 531 drains).

Cross-reference findings against POTENTIAL_ATTACKS_V3 categories.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def hdr(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


OPERATOR = "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8"

hdr(f"OPERATOR {OPERATOR}")
# Watchlist details
wl = c.execute(
    "SELECT * FROM watchlist WHERE address=?",
    (OPERATOR,)
).fetchone()
if wl:
    cols = [d[0] for d in c.execute("PRAGMA table_info(watchlist)").fetchall()]
    for col, val in zip(cols, wl):
        print(f"  {col}: {val}")

# Deployer metadata
print()
print("  --- deployer row ---")
d = c.execute(
    "SELECT * FROM deployers WHERE deployer_address=?",
    (OPERATOR,)
).fetchone()
if d:
    cols = [x[0] for x in c.execute("PRAGMA table_info(deployers)").fetchall()]
    for col, val in zip(cols, d):
        if val is not None and val != '':
            print(f"  {col}: {val}")

# All contracts deployed by operator
print()
print("  --- all contracts deployed by operator ---")
for r in c.execute(
    """SELECT contract_address, chain, confidence_tier, detection_timestamp, deployed_code_hash
       FROM contracts WHERE deployer_address=? ORDER BY detection_timestamp""",
    (OPERATOR,)
):
    print(f"  {r[0]}  chain={r[1]:9s}  tier={r[2]:9s}  deployed={r[3][:10]}  hash={r[4][:18] if r[4] else ''}")


# Now let's see what bytecode family 0x752c5a95 belongs to and what other contracts share it
hdr("Bytecode family of 0x752c5a95")
bait = "0x752c5a95d202972e124390f30a50154409d3c858"
code_hash = c.execute("SELECT deployed_code_hash FROM contracts WHERE contract_address=?", (bait,)).fetchone()
if code_hash and code_hash[0]:
    ch = code_hash[0]
    print(f"  code_hash: {ch}")
    n_same = c.execute("SELECT COUNT(*) FROM contracts WHERE deployed_code_hash=?", (ch,)).fetchone()[0]
    print(f"  contracts with same code_hash: {n_same}")
    # bytecode_family lookup
    fam = c.execute("SELECT * FROM bytecode_families WHERE code_hash=?", (ch,)).fetchone()
    if fam:
        cols = [d[0] for d in c.execute("PRAGMA table_info(bytecode_families)").fetchall()]
        print("  bytecode_family row:")
        for col, val in zip(cols, fam):
            print(f"    {col}: {val}")
else:
    print(f"  no code_hash recorded for {bait}")


# Now profile the May-9..15 drain wave's contracts
hdr("Top drain-targets across full May-9..15 wave (combined)")
for r in c.execute(
    """SELECT aw.contract_address, COUNT(*) AS drains,
              COUNT(DISTINCT aw.victim_address) AS vics,
              COUNT(DISTINCT aw.drain_caller) AS drainers,
              substr(MIN(aw.drain_timestamp),1,10) AS first,
              substr(MAX(aw.drain_timestamp),1,10) AS last
       FROM approval_watchlist aw
       WHERE aw.drain_detected=1
         AND aw.drain_timestamp >= '2026-05-09'
         AND aw.drain_timestamp < '2026-05-16'
       GROUP BY aw.contract_address
       ORDER BY drains DESC LIMIT 15"""
):
    ctr = c.execute(
        "SELECT chain, confidence_tier, deployer_address, deployed_code_hash, detection_timestamp FROM contracts WHERE contract_address=?",
        (r[0],)
    ).fetchone()
    chain = ctr[0] if ctr else "?"
    tier = ctr[1] if ctr else "?"
    dep = ctr[2] if ctr else "?"
    ch = (ctr[3] or "")[:18] if ctr else ""
    deployed = (ctr[4] or "")[:10] if ctr else ""
    print(f"  contract={r[0]}")
    print(f"    drains={r[1]}  vics={r[2]}  drainers={r[3]}  window={r[4]}..{r[5]}  chain={chain}  tier={tier}")
    print(f"    deployed_on={deployed}  code_hash={ch}  deployer={dep}")
    # check if deployer on watchlist
    wl = c.execute("SELECT entity_name, priority FROM watchlist WHERE address=?", (dep,)).fetchone()
    print(f"    deployer_watchlist: {wl if wl else '(not on watchlist)'}")


# Cohort breakdown: which May-9..15 drainers ARE on watchlist vs not
hdr("May-9..15 drainers: watchlist coverage")
wl_addrs = {r[0].lower(): (r[1], r[2]) for r in c.execute(
    "SELECT address, entity_name, priority FROM watchlist WHERE active=1 AND address IS NOT NULL"
) if r[0]}
on_wl_drains = 0
off_wl_drains = 0
on_wl_drainers = set()
off_wl_drainers = set()
for r in c.execute(
    """SELECT drain_caller, COUNT(*) FROM approval_watchlist
       WHERE drain_detected=1
         AND drain_timestamp >= '2026-05-09' AND drain_timestamp < '2026-05-16'
       GROUP BY drain_caller"""
):
    if r[0] and r[0].lower() in wl_addrs:
        on_wl_drains += r[1]
        on_wl_drainers.add(r[0])
    else:
        off_wl_drains += r[1]
        off_wl_drainers.add(r[0])
total = on_wl_drains + off_wl_drains
print(f"  Total drains in window: {total}")
print(f"  By on-watchlist drainers: {on_wl_drains} ({100*on_wl_drains/total:.1f}%) from {len(on_wl_drainers)} addresses")
print(f"  By off-watchlist drainers: {off_wl_drains} ({100*off_wl_drains/total:.1f}%) from {len(off_wl_drainers)} addresses")


# Trace 0x752c5a95 history: detection through final drain
hdr("Timeline of bait contract 0x752c5a95 — full history")
print("  Deploy + approval + drain timeline:")
deploy_ts = c.execute("SELECT detection_timestamp FROM contracts WHERE contract_address=?", (bait,)).fetchone()
print(f"    Deployed: {deploy_ts[0] if deploy_ts else '?'}")
appr_min, appr_max, appr_count = c.execute(
    """SELECT MIN(approve_timestamp), MAX(approve_timestamp), COUNT(*)
       FROM approval_watchlist WHERE contract_address=?""", (bait,)
).fetchone()
print(f"    Approvals: {appr_count} total; first={appr_min}  last={appr_max}")
drain_min, drain_max, drain_count = c.execute(
    """SELECT MIN(drain_timestamp), MAX(drain_timestamp), COUNT(*)
       FROM approval_watchlist WHERE contract_address=? AND drain_detected=1""", (bait,)
).fetchone()
print(f"    Drains:    {drain_count} total; first={drain_min}  last={drain_max}")
# Approvals per week leading into drain
print()
print("  Approvals per day for 0x752c5a95 (Mar-26 thru May-15):")
for r in c.execute(
    """SELECT substr(approve_timestamp,1,10), COUNT(*)
       FROM approval_watchlist WHERE contract_address=?
       GROUP BY 1 ORDER BY 1""",
    (bait,)
):
    print(f"    {r[0]}  {r[1]:>4}")


# Check the May-11/12/13/14 wave — which contracts?
hdr("May-11..14 secondary drain wave — what contracts?")
for r in c.execute(
    """SELECT substr(drain_timestamp,1,10) AS d,
              contract_address, COUNT(*) AS n
       FROM approval_watchlist
       WHERE drain_detected=1
         AND drain_timestamp >= '2026-05-11' AND drain_timestamp < '2026-05-15'
       GROUP BY d, contract_address
       HAVING n >= 25
       ORDER BY d, n DESC"""
):
    ctr = c.execute(
        "SELECT chain, deployer_address FROM contracts WHERE contract_address=?",
        (r[1],)
    ).fetchone()
    dep = ctr[1] if ctr else "?"
    wl = c.execute("SELECT entity_name FROM watchlist WHERE address=?", (dep,)).fetchone()
    print(f"  {r[0]}  contract={r[1]}  n={r[2]:>4}  deployer={dep}  wl={wl[0] if wl else '(not on watchlist)'}")
