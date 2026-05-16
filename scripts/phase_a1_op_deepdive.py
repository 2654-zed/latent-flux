"""Phase A1 deep-dive: characterize the Apr-25 Optimism deployer-mass (6,638
deployers) that emerged with no funding_sources data and produced few
post-deployment approvals.
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


# ---- 1: hourly distribution of the 6,638 first_seen on Apr-25 ----
hdr("Hourly distribution of Apr-25 Optimism first_seen events")
print(f"  {'hour':10s}  {'count':>6s}")
for r in c.execute(
    """SELECT substr(first_seen,12,2) AS h, COUNT(*) FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]:10s}  {r[1]:>6,}")


# ---- 2: how many of them deployed actual contracts? ----
hdr("Contracts deployed by Apr-25 Optimism mass")
n_with_contracts = c.execute(
    """SELECT COUNT(DISTINCT deployer_address) FROM contracts
       WHERE deployer_address IN (
         SELECT deployer_address FROM deployers
         WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'
       )"""
).fetchone()[0]
total = c.execute(
    """SELECT COUNT(*) FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'"""
).fetchone()[0]
print(f"  total Apr-25 OP deployers: {total:,}")
print(f"  of those, deployers with contracts in contracts table: {n_with_contracts:,}")

print()
print("Total contracts deployed by this cohort, by chain (might be cross-chain):")
for r in c.execute(
    """SELECT chain, COUNT(*) FROM contracts
       WHERE deployer_address IN (
         SELECT deployer_address FROM deployers
         WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'
       )
       GROUP BY chain"""
):
    print(f"  {r[0]:10s}  {r[1]:,}")


# ---- 3: confidence tier distribution of their contracts ----
hdr("Confidence-tier distribution of contracts from Apr-25 OP cohort")
for r in c.execute(
    """SELECT confidence_tier, COUNT(*) FROM contracts
       WHERE deployer_address IN (
         SELECT deployer_address FROM deployers
         WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'
       )
       GROUP BY confidence_tier ORDER BY 2 DESC"""
):
    print(f"  {r[0] or '(null)':20s}  {r[1]:,}")


# ---- 4: total_contracts_deployed distribution of the cohort ----
hdr("Per-deployer total_contracts_deployed histogram (Apr-25 OP cohort)")
buckets: dict[str, int] = {}
for r in c.execute(
    """SELECT total_contracts_deployed FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'"""
):
    n = r[0] or 0
    if n == 0:
        key = "0"
    elif n == 1:
        key = "1"
    elif n <= 5:
        key = "2-5"
    elif n <= 20:
        key = "6-20"
    else:
        key = "21+"
    buckets[key] = buckets.get(key, 0) + 1
for k in ["0", "1", "2-5", "6-20", "21+"]:
    print(f"  {k:6s}  {buckets.get(k, 0):,}")


# ---- 5: are the contracts they deployed clustered by bytecode? ----
hdr("Bytecode-family clustering (deployed_code_hash) of the cohort's contracts")
print(f"  {'code_hash':70s}  {'count':>6s}")
for r in c.execute(
    """SELECT deployed_code_hash, COUNT(*) FROM contracts
       WHERE deployer_address IN (
         SELECT deployer_address FROM deployers
         WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'
       )
       GROUP BY deployed_code_hash ORDER BY 2 DESC LIMIT 10"""
):
    h = (r[0] or "(null)")[:70]
    print(f"  {h:70s}  {r[1]:>6,}")


# ---- 6: timing — same gas_price / interval — script signature? ----
hdr("Gas-price clustering of the cohort (typical_gas_price_gwei)")
buckets: dict[str, int] = {}
for r in c.execute(
    """SELECT typical_gas_price_gwei FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'"""
):
    g = r[0]
    if g is None:
        key = "null"
    elif g < 0.001:
        key = "<0.001"
    elif g < 0.01:
        key = "0.001-0.01"
    elif g < 0.1:
        key = "0.01-0.1"
    elif g < 1:
        key = "0.1-1"
    else:
        key = "1+"
    buckets[key] = buckets.get(key, 0) + 1
for k in ["null", "<0.001", "0.001-0.01", "0.01-0.1", "0.1-1", "1+"]:
    print(f"  {k:12s}  {buckets.get(k, 0):,}")


# ---- 7: sample 5 random deployers from the cohort ----
hdr("Sample 5 deployers from Apr-25 OP cohort")
for r in c.execute(
    """SELECT deployer_address, first_seen, total_contracts_deployed,
              typical_gas_price_gwei, deployment_pattern_notes,
              entity_type, behavioral_score
       FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'
       ORDER BY total_contracts_deployed DESC
       LIMIT 5"""
):
    print(f"  {r[0]}")
    print(f"    first_seen={r[1]}  contracts={r[2]}  gas_gwei={r[3]}")
    print(f"    entity_type={r[5]!r}  behavioral_score={r[6]}")
    print(f"    notes={(r[4] or '')[:200]}")
