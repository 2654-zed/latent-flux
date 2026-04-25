"""How dominant are the top-12 funders in the corpus?

If the camouflage ratio, bytecode family diversity, etc. are computed
against a population dominated by 12 funders' downstream wallets, the
'corpus-wide' adjective stops meaning what it appeared to mean.

This script measures:
  - top-12-funded contracts as fraction of corpus
  - top-12-funded share within each tier (confirmed / suspected / unanalyzed)
  - top-12-funded share within each large bytecode family
  - how the camouflage-relevant population (>=10 tx/day contracts) splits
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

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

# === Load funded set ===
print("Loading top-12 funded deployers + contracts...")
ph = ",".join("?" * len(TOP_12))
funded_deployers = {r[0].lower() for r in c.execute(
    f"SELECT deployer_address FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) IN ({ph})",
    TOP_12)}
print(f"  funded deployers: {len(funded_deployers):,}")

c.execute("CREATE TEMP TABLE funded_addrs(addr TEXT PRIMARY KEY)")
c.executemany("INSERT INTO funded_addrs VALUES (?)", [(a,) for a in funded_deployers])

funded_contracts = {r[0].lower() for r in c.execute(
    "SELECT contract_address FROM contracts WHERE LOWER(deployer_address) IN (SELECT addr FROM funded_addrs)")}
print(f"  funded contracts: {len(funded_contracts):,}")

c.execute("CREATE TEMP TABLE funded_contracts(addr TEXT PRIMARY KEY)")
c.executemany("INSERT INTO funded_contracts VALUES (?)", [(a,) for a in funded_contracts])

total_contracts = c.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
total_deployers = c.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]

h("1. Corpus-share by deployer and contract")
print(f"  total deployers in corpus:                 {total_deployers:,}")
print(f"  top-12-funded deployers:                   {len(funded_deployers):,}")
print(f"  fraction:                                  {len(funded_deployers)/total_deployers*100:.2f}%")
print()
print(f"  total contracts in corpus:                 {total_contracts:,}")
print(f"  contracts deployed by top-12-funded:       {len(funded_contracts):,}")
print(f"  fraction:                                  {len(funded_contracts)/total_contracts*100:.2f}%")

h("2. Tier distribution: corpus-wide vs top-12-funded")
print(f"  {'tier':<12} {'corpus_total':<14} {'top12_funded':<14} {'top12_share':<11}")
for tier in ['confirmed', 'suspected', 'unanalyzed', 'unknown', None]:
    if tier is None:
        total = c.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier IS NULL").fetchone()[0]
        funded_n = c.execute(
            "SELECT COUNT(*) FROM contracts WHERE confidence_tier IS NULL AND LOWER(contract_address) IN (SELECT addr FROM funded_contracts)"
        ).fetchone()[0]
        tier_label = "(null)"
    else:
        total = c.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier = ?", (tier,)).fetchone()[0]
        funded_n = c.execute(
            "SELECT COUNT(*) FROM contracts WHERE confidence_tier = ? AND LOWER(contract_address) IN (SELECT addr FROM funded_contracts)",
            (tier,)).fetchone()[0]
        tier_label = tier
    if total > 0:
        share = funded_n / total * 100
    else:
        share = 0
    print(f"  {tier_label:<12} {total:<14,} {funded_n:<14,} {share:>9.2f}%")

h("3. Top-12 share within large bytecode families (top 15 by members)")
print(f"  {'family_id':<22} {'corpus_members':<15} {'top12_members':<14} {'top12_share':<11} {'name'}")
for r in c.execute("""
    SELECT family_id, family_name, member_count
    FROM bytecode_families ORDER BY member_count DESC LIMIT 15
"""):
    fid = r['family_id']
    fname = (r['family_name'] or '-')[:50]
    fam_total = c.execute(
        "SELECT COUNT(*) FROM bytecode_family_members WHERE family_id = ?", (fid,)
    ).fetchone()[0]
    fam_top12 = c.execute("""
        SELECT COUNT(*) FROM bytecode_family_members
        WHERE family_id = ? AND LOWER(contract_address) IN (SELECT addr FROM funded_contracts)
    """, (fid,)).fetchone()[0]
    share = fam_top12 / fam_total * 100 if fam_total else 0
    print(f"  {fid:<22} {fam_total:<15,} {fam_top12:<14,} {share:>9.2f}%  {fname}")

h("4. Camouflage-population dominance check")
# The camouflage ratio is computed against contracts with >= 10 tx in a day.
# How much of THAT population is top-12-funded?
print("  Total contracts with any tx_events:")
total_with_tx = c.execute("""
    SELECT COUNT(DISTINCT contract_address) FROM transaction_events
""").fetchone()[0]
print(f"    {total_with_tx:,}")

print("  Top-12-funded contracts with any tx_events:")
funded_with_tx = c.execute("""
    SELECT COUNT(DISTINCT contract_address) FROM transaction_events
    WHERE LOWER(contract_address) IN (SELECT addr FROM funded_contracts)
""").fetchone()[0]
print(f"    {funded_with_tx:,}")
if total_with_tx:
    print(f"    fraction: {funded_with_tx/total_with_tx*100:.2f}%")

h("5. Trap-event share")
total_traps = c.execute("SELECT COUNT(*) FROM trap_events").fetchone()[0]
funded_traps = c.execute("""
    SELECT COUNT(*) FROM trap_events
    WHERE LOWER(trap_contract_address) IN (SELECT addr FROM funded_contracts)
""").fetchone()[0]
print(f"  total trap_events:           {total_traps:,}")
print(f"  on top-12-funded contracts:  {funded_traps:,}")
if total_traps:
    print(f"  fraction:                    {funded_traps/total_traps*100:.2f}%")

h("6. Per-funder fanout breakdown")
print(f"  {'funder':<44} {'deployers':<10} {'contracts':<10} {'confirmed':<10} {'suspected':<10} {'first_seen'}")
for fdr in TOP_12:
    n_deps = c.execute("""
        SELECT COUNT(*) FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
    """, (fdr,)).fetchone()[0]
    fan_contracts = c.execute("""
        SELECT COUNT(*) FROM contracts ct
        WHERE LOWER(ct.deployer_address) IN (
            SELECT LOWER(deployer_address) FROM deployers
            WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
        )
    """, (fdr,)).fetchone()[0]
    conf = c.execute("""
        SELECT COUNT(*) FROM contracts ct
        WHERE LOWER(ct.deployer_address) IN (
            SELECT LOWER(deployer_address) FROM deployers
            WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
        ) AND confidence_tier = 'confirmed'
    """, (fdr,)).fetchone()[0]
    sus = c.execute("""
        SELECT COUNT(*) FROM contracts ct
        WHERE LOWER(ct.deployer_address) IN (
            SELECT LOWER(deployer_address) FROM deployers
            WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
        ) AND confidence_tier = 'suspected'
    """, (fdr,)).fetchone()[0]
    first = c.execute("""
        SELECT MIN(first_seen) FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
    """, (fdr,)).fetchone()[0]
    print(f"  {fdr}  {n_deps:<10,} {fan_contracts:<10,} {conf:<10,} {sus:<10,} {(first or '')[:10]}")

c.close()
