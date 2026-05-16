"""Phase A2 + A3 investigation.

A2 — May-5 confirmed_traps spike (210 confirmed, 4-10x surrounding days).
Hypothesis space:
  (a) iter_8 of drainer-spawn hub 0xf7883e3f (May-5 = iter_8 spawn day per INDEX.md)
  (b) bytecode classifier rule added/modified producing retroactive re-classification
  (c) backfill / re-scan job confirmed earlier-deployed contracts
Pre-registered prediction: SPLIT — partly (a), more likely (b) or (c).

A3 — Coffee Fleet vs. approval-events decay (Apr-23 -> Apr-25).
Hypothesis: Coffee Fleet's victim acquisition slowed Apr-23 -> 25. Three causes
considered: bots learned, operators retired contracts, victim pool saturated.
Pre-registered prediction: SYSTEMIC — corpus-level decay, Coffee Fleet's
share roughly constant.
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


COFFEE = "0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e"
ITER_HUB = "0xf7883e3fef23c8e645deba4b540549d78028a616"
ITER_8_WALLET = "0xa8c7ac1cdc33"  # prefix per INDEX.md


# ============================================================
# A2 — May-5 confirmed-traps spike
# ============================================================

hdr("A2-1: confirmed-trap counts by date around May-5")
for r in c.execute(
    """SELECT substr(detection_timestamp,1,10), COUNT(*) FROM contracts
       WHERE confidence_tier='confirmed'
         AND detection_timestamp>='2026-05-01' AND detection_timestamp<'2026-05-12'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  {r[1]:>5,}")


hdr("A2-2: top deployers on May-5 confirmed-tier contracts")
for r in c.execute(
    """SELECT deployer_address, COUNT(*) FROM contracts
       WHERE confidence_tier='confirmed'
         AND detection_timestamp>='2026-05-05' AND detection_timestamp<'2026-05-06'
       GROUP BY deployer_address ORDER BY 2 DESC LIMIT 15"""
):
    print(f"  {r[0]}  {r[1]:>4}")


hdr("A2-3: how many May-5 confirmed-tier contracts came from iter_8 hub family?")
# Find addresses associated with iter_8 hub
print(f"  ITER_HUB = {ITER_HUB}")
# Look for contracts deployed by addresses funded by ITER_HUB
n_hub_dep = c.execute(
    "SELECT COUNT(*) FROM contracts WHERE deployer_address=?",
    (ITER_HUB,)
).fetchone()[0]
print(f"  Contracts directly deployed by ITER_HUB: {n_hub_dep}")
# Look for addresses with funding_sources containing the hub
n_hub_funded = c.execute(
    "SELECT COUNT(*) FROM deployers WHERE funding_sources LIKE ?",
    (f"%{ITER_HUB}%",)
).fetchone()[0]
print(f"  Deployers with ITER_HUB in funding_sources: {n_hub_funded}")
# Look for iter_8 wallet prefix specifically
n_iter8_funded = c.execute(
    "SELECT COUNT(*) FROM deployers WHERE funding_sources LIKE ?",
    (f"%{ITER_8_WALLET}%",)
).fetchone()[0]
print(f"  Deployers with iter_8 wallet ({ITER_8_WALLET}) in funding_sources: {n_iter8_funded}")


hdr("A2-4: bytecode hash distribution of May-5 confirmed-tier contracts")
for r in c.execute(
    """SELECT deployed_code_hash, COUNT(*) FROM contracts
       WHERE confidence_tier='confirmed'
         AND detection_timestamp>='2026-05-05' AND detection_timestamp<'2026-05-06'
       GROUP BY deployed_code_hash ORDER BY 2 DESC LIMIT 10"""
):
    h = (r[0] or "(null)")[:70]
    print(f"  {h}  {r[1]:>4}")


hdr("A2-5: detection vs deployment lag — were May-5 confirmed traps DEPLOYED on May-5 or earlier?")
# Schema: detection_timestamp vs (no separate deploy timestamp; but the deployer first_seen tells us approximately)
print("  Comparing detection_timestamp to deployer.first_seen for May-5 confirmed contracts:")
n_same_day = 0
n_earlier = 0
total = 0
for r in c.execute(
    """SELECT contracts.detection_timestamp, deployers.first_seen
       FROM contracts
       JOIN deployers ON deployers.deployer_address=contracts.deployer_address
       WHERE contracts.confidence_tier='confirmed'
         AND contracts.detection_timestamp>='2026-05-05' AND contracts.detection_timestamp<'2026-05-06'"""
):
    total += 1
    det_day = r[0][:10]
    dep_day = r[1][:10] if r[1] else None
    if dep_day == det_day:
        n_same_day += 1
    elif dep_day and dep_day < det_day:
        n_earlier += 1
print(f"  total May-5 confirmed: {total}")
print(f"  deployer first_seen ON May-5: {n_same_day}  ({100*n_same_day/total:.1f}%)")
print(f"  deployer first_seen BEFORE May-5: {n_earlier}  ({100*n_earlier/total:.1f}%)")


# ============================================================
# A3 — Coffee Fleet vs approval decay
# ============================================================

hdr("A3-1: Coffee Fleet contracts (deployed by 0xc0ffee...) — daily approval counts Apr-22..27")
print(f"  COFFEE = {COFFEE}")
# Coffee Fleet's contracts
coffee_contracts = [r[0] for r in c.execute(
    "SELECT contract_address FROM contracts WHERE deployer_address=?",
    (COFFEE,)
)]
print(f"  Coffee Fleet contract count: {len(coffee_contracts):,}")

# Approvals on those contracts, day by day
placeholders = ",".join(["?"] * len(coffee_contracts)) if coffee_contracts else "NULL"
if coffee_contracts:
    coffee_daily: dict[str, int] = {}
    BATCH = 500
    for i in range(0, len(coffee_contracts), BATCH):
        batch = coffee_contracts[i:i+BATCH]
        ph = ",".join(["?"] * len(batch))
        for r in c.execute(
            f"""SELECT substr(approve_timestamp,1,10), COUNT(*)
                FROM approval_watchlist
                WHERE contract_address IN ({ph})
                  AND approve_timestamp>='2026-04-22' AND approve_timestamp<'2026-04-28'
                GROUP BY 1""",
            batch
        ):
            coffee_daily[r[0]] = coffee_daily.get(r[0], 0) + r[1]
    print("  Coffee Fleet daily approvals (Apr-22..27):")
    for d in sorted(coffee_daily):
        print(f"    {d}  {coffee_daily[d]:>5,}")


hdr("A3-2: Coffee Fleet share of TOTAL daily approvals")
# Get total approvals by day
total_daily: dict[str, int] = {}
for r in c.execute(
    """SELECT substr(approve_timestamp,1,10), COUNT(*)
       FROM approval_watchlist
       WHERE approve_timestamp>='2026-04-22' AND approve_timestamp<'2026-04-28'
       GROUP BY 1"""
):
    total_daily[r[0]] = r[1]
print(f"  {'date':12s}  {'coffee':>7s}  {'total':>7s}  {'share':>7s}")
for d in sorted(total_daily):
    coffee = coffee_daily.get(d, 0)
    total = total_daily[d]
    share = 100 * coffee / total if total else 0
    print(f"  {d:12s}  {coffee:>7,}  {total:>7,}  {share:>6.1f}%")


hdr("A3-3: Coffee Fleet deployment activity Apr-22..27 (new contracts deployed in window)")
for r in c.execute(
    """SELECT substr(detection_timestamp,1,10), COUNT(*) FROM contracts
       WHERE deployer_address=?
         AND detection_timestamp>='2026-04-22' AND detection_timestamp<'2026-04-28'
       GROUP BY 1 ORDER BY 1""",
    (COFFEE,)
):
    print(f"  {r[0]}  {r[1]:>5,}")
