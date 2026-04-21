"""Compare population-level vs adversary-scoped low-revert ratios.

Answers the Correction #13 question: does the 70-79% figure reflect
adversary Nash equilibrium or ecosystem contract-design baseline?
"""
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

print("=" * 70)
print("  Camouflage comparison — population vs adversary-scoped")
print("=" * 70)
print()

# Population-level (all active contracts, lifetime)
pop = c.execute("""
    SELECT COUNT(*) AS total, SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END) AS camo
    FROM (
        SELECT contract_address,
               CAST(SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS rr
        FROM transaction_events
        GROUP BY contract_address
        HAVING COUNT(*) >= 10
    )
""").fetchone()
pop_ratio = pop["camo"] / pop["total"] if pop["total"] else 0
print(f"POPULATION (all contracts with 10+ tx, lifetime):")
print(f"  total active contracts: {pop['total']:>8,}")
print(f"  low-revert (<10%):      {pop['camo']:>8,}")
print(f"  ratio:                  {pop_ratio:.3f}  ({100*pop_ratio:.1f}%)")
print()

# Adversary-scoped (suspected + confirmed only)
adv = c.execute("""
    SELECT COUNT(*) AS total, SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END) AS camo
    FROM (
        SELECT te.contract_address,
               CAST(SUM(CASE WHEN te.is_reverted=1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS rr
        FROM transaction_events te
        JOIN contracts c ON c.contract_address = te.contract_address
        WHERE c.confidence_tier IN ('confirmed', 'suspected')
        GROUP BY te.contract_address
        HAVING COUNT(*) >= 10
    )
""").fetchone()
adv_ratio = adv["camo"] / adv["total"] if adv["total"] else 0
print(f"ADVERSARY (suspected + confirmed, 10+ tx, lifetime):")
print(f"  total adversary contracts: {adv['total']:>8,}")
print(f"  low-revert (<10%):         {adv['camo']:>8,}")
print(f"  ratio:                     {adv_ratio:.3f}  ({100*adv_ratio:.1f}%)")
print()

# Confirmed-only — the cleanest adversary signal
conf = c.execute("""
    SELECT COUNT(*) AS total, SUM(CASE WHEN rr < 0.10 THEN 1 ELSE 0 END) AS camo
    FROM (
        SELECT te.contract_address,
               CAST(SUM(CASE WHEN te.is_reverted=1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS rr
        FROM transaction_events te
        JOIN contracts c ON c.contract_address = te.contract_address
        WHERE c.confidence_tier = 'confirmed'
        GROUP BY te.contract_address
        HAVING COUNT(*) >= 10
    )
""").fetchone()
conf_ratio = conf["camo"] / conf["total"] if conf["total"] else 0
print(f"CONFIRMED-ONLY (strictest adversary signal, 10+ tx, lifetime):")
print(f"  total confirmed contracts: {conf['total']:>8,}")
print(f"  low-revert (<10%):         {conf['camo']:>8,}")
print(f"  ratio:                     {conf_ratio:.3f}  ({100*conf_ratio:.1f}%)")
print()

# Interpretation
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
divergence_adv = abs(pop_ratio - adv_ratio)
divergence_conf = abs(pop_ratio - conf_ratio)
print(f"  pop vs adversary divergence:  {divergence_adv:.3f}  ({100*divergence_adv:.1f}pp)")
print(f"  pop vs confirmed divergence:  {divergence_conf:.3f}  ({100*divergence_conf:.1f}pp)")
print()
if divergence_adv < 0.05:
    print("  VERDICT: ratios track closely. The population-level camouflage_ratio")
    print("           is a general property of low-revert contract design, not a")
    print("           specific adversary equilibrium. Correction #13 stands.")
elif adv_ratio < pop_ratio:
    print(f"  VERDICT: adversaries are LESS low-revert than the population. Not")
    print(f"           a disguise strategy. The 'camouflage' name is backward.")
else:
    print(f"  VERDICT: adversaries concentrate in the low-revert band relative to")
    print(f"           the population. The equilibrium interpretation has some")
    print(f"           empirical support — the divergence ({100*divergence_adv:.1f}pp) is the actual strength of")
    print(f"           the claim, not the 70-79% headline.")
c.close()
