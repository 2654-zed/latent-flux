"""T2-eaef6a5d ∩ top-12-funders intersection.

If T2 covers 75% of corpus by contract and top-12 funders cover 35% by deployer,
the question is: do these blind spots overlap, or are they two independent shadow
populations? Result determines whether corpus-wide statistics still mean what
they appeared to mean.

We need:
  - T2-eaef6a5d family member contract list
  - top-12 funder downstream deployer list
  - intersection: how many of T2's contracts were deployed by top-12-funded
    deployers, and how many of top-12-cluster's contracts are in T2?
"""
import sqlite3
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

TOP_12 = [
    "0xf70da97812cb96acdf810712aa562db8dfa3dbef",
    "0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda",
    "0x3304e22ddaa22bcdc5fca2269b418046ae7b566a",
    "0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078",
    "0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07",
    "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473",
    "0x0e6e91775d24d34b90e0f3d806a90705f0199999",
    "0x238d7170f309a55b87a144a341bd6105897082ca",
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1",
    "0x8ca702323c341a8d46ee94a2abeddb08798ca10d",
    "0x39591e7c099a379fd7b349ebfecaeef439c40454",
    "0xca7ece5e43ef44de8e430629a5b535eca48e251b",
]

print("=== Loading T2-eaef6a5d family ===")
# Find T2 family ID(s)
fams = list(c.execute("""
    SELECT family_id, family_name, member_count, unique_deployers
    FROM bytecode_families WHERE family_id LIKE '%eaef6a5d%' OR family_name LIKE '%eaef6a5d%'
"""))
for f in fams:
    print(f"  family_id={f['family_id']}  name={f['family_name']}  members={f['member_count']} deployers={f['unique_deployers']}")

if not fams:
    # Maybe T2-eaef6a5d is a family_id prefix; try a broader match
    fams = list(c.execute("SELECT family_id, family_name, member_count FROM bytecode_families WHERE family_id LIKE 'T2-%' ORDER BY member_count DESC LIMIT 10"))
    print("  No exact match — top 10 T2 families:")
    for f in fams:
        print(f"    family_id={f['family_id']}  name={f['family_name']}  members={f['member_count']}")
    raise SystemExit("Inspect family list, edit script with correct ID")

family_ids = [f['family_id'] for f in fams]
ph = ",".join("?" * len(family_ids))
t2_contracts = {r[0].lower() for r in c.execute(
    f"SELECT contract_address FROM bytecode_family_members WHERE family_id IN ({ph})",
    family_ids)}
print(f"  T2 contract count: {len(t2_contracts):,}")

print()
print("=== Loading top-12-funded deployers ===")
ph12 = ",".join("?" * len(TOP_12))
funded_deployers = {r[0].lower() for r in c.execute(
    f"SELECT deployer_address FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) IN ({ph12})",
    TOP_12)}
print(f"  top-12-funded deployer count: {len(funded_deployers):,}")

# Load contracts deployed by funded deployers
print()
print("=== Loading contracts of top-12-funded deployers ===")
funded_contracts = set()
# Chunked IN clause to avoid SQL parameter limits
funded_list = list(funded_deployers)
chunk = 500
for i in range(0, len(funded_list), chunk):
    sub = funded_list[i:i+chunk]
    ph_sub = ",".join("?" * len(sub))
    for r in c.execute(
        f"SELECT contract_address FROM contracts WHERE LOWER(deployer_address) IN ({ph_sub})",
        sub):
        funded_contracts.add(r[0].lower())
print(f"  contracts deployed by top-12-funded deployers: {len(funded_contracts):,}")

print()
print("=== INTERSECTION ===")
overlap = t2_contracts & funded_contracts
print(f"  T2 contracts:                   {len(t2_contracts):>8,}")
print(f"  Top-12 funded contracts:        {len(funded_contracts):>8,}")
print(f"  Intersection:                   {len(overlap):>8,}")
if t2_contracts:
    print(f"  T2 ∩ funded / T2:               {len(overlap)/len(t2_contracts)*100:>7.2f}%")
if funded_contracts:
    print(f"  T2 ∩ funded / funded:           {len(overlap)/len(funded_contracts)*100:>7.2f}%")

# Total corpus
total_contracts = c.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
print()
print(f"  total corpus contracts:         {total_contracts:>8,}")
print(f"  T2 / total:                     {len(t2_contracts)/total_contracts*100:>7.2f}%")
print(f"  Top-12 funded / total:          {len(funded_contracts)/total_contracts*100:>7.2f}%")
print(f"  Union (T2 ∪ funded) / total:    {len(t2_contracts | funded_contracts)/total_contracts*100:>7.2f}%")

# Confirmed/suspected within the intersection vs outside
print()
print("=== Tier distribution in regions ===")
def tier_dist(addrs):
    if not addrs:
        return {}
    addrs = list(addrs)
    counts = {"confirmed": 0, "suspected": 0, "unknown": 0, "unanalyzed": 0}
    for i in range(0, len(addrs), chunk):
        sub = addrs[i:i+chunk]
        ph_sub = ",".join("?" * len(sub))
        for r in c.execute(
            f"SELECT confidence_tier, COUNT(*) FROM contracts WHERE LOWER(contract_address) IN ({ph_sub}) GROUP BY confidence_tier",
            sub):
            counts[r[0] or "unknown"] = counts.get(r[0] or "unknown", 0) + r[1]
    return counts

print("  T2 ∩ funded (the overlap):")
for k, v in tier_dist(overlap).items():
    print(f"    {k:<12} {v:,}")
print()
print("  T2 \\ funded (T2 but not from a top-12 funder):")
t2_only = t2_contracts - funded_contracts
for k, v in tier_dist(t2_only).items():
    print(f"    {k:<12} {v:,}")
print()
print("  funded \\ T2 (top-12 contracts NOT in T2):")
funded_only = funded_contracts - t2_contracts
for k, v in tier_dist(funded_only).items():
    print(f"    {k:<12} {v:,}")

c.close()
