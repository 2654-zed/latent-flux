"""Precise blast radius: of the 347 audit-migrated contracts
(confirmed->unanalyzed), how many had drain_detected evidence in
approval_watchlist that the migration heuristics never checked?

This is the real Finding 4. No conclusions in code — dump the numbers.
"""
import sqlite3
from pathlib import Path
DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_audit_migration_drain_gate.txt"
c = sqlite3.connect(DB)
L=[]
def p(s=""): L.append(str(s))

migrated = [r[0] for r in c.execute(
    "SELECT contract_address FROM contracts WHERE confidence_tier='unanalyzed' "
    "AND confidence_reason LIKE '%Correction #25%'")]
p(f"total audit-migrated contracts: {len(migrated)}")

# Which migration batch each came from (parse the annotation)
batches = {"Phase A (holders/verified)":0, "Phase B (self-loop/BACKFILL)":0,
           "Phase C sample":0, "Phase C FROM_SOURCE":0, "Phase C FROM_ACTIVITY":0,
           "Phase C FROM_CLUSTER":0, "other":0}
def batch_of(reason):
    if "Phase B" in reason: return "Phase B (self-loop/BACKFILL)"
    if "Phase C sample" in reason: return "Phase C sample"
    if "LIKELY_FP_FROM_SOURCE" in reason: return "Phase C FROM_SOURCE"
    if "LIKELY_FP_FROM_ACTIVITY" in reason: return "Phase C FROM_ACTIVITY"
    if "LIKELY_FP_FROM_CLUSTER" in reason: return "Phase C FROM_CLUSTER"
    if "Correction #25:" in reason: return "Phase A (holders/verified)"
    return "other"

# For each migrated contract: does it have drain_detected evidence?
withdrain = []
for a in migrated:
    reason = c.execute("SELECT confidence_reason FROM contracts WHERE contract_address=?",(a,)).fetchone()[0] or ""
    b = batch_of(reason)
    batches[b]+=1
    d = c.execute("""SELECT SUM(CASE WHEN drain_detected=1 THEN 1 ELSE 0 END),
        COUNT(DISTINCT CASE WHEN drain_detected=1 THEN drain_tx_hash END)
        FROM approval_watchlist WHERE contract_address=?""",(a,)).fetchone()
    drained, dtx = d[0] or 0, d[1] or 0
    if drained > 0:
        withdrain.append((a, b, drained, dtx, reason[:90]))

p("\nmigrated contracts by batch:")
for k,v in batches.items():
    p(f"    {k:32s}: {v}")

p(f"\n*** migrated contracts WITH drain_detected evidence: {len(withdrain)} ***")
p("(these had on-chain drain rows the migration never checked)")
withdrain.sort(key=lambda x:-x[3])
for a,b,drained,dtx,reason in withdrain:
    p(f"\n  {a}")
    p(f"    batch={b}  drains={drained} distinct_drain_tx={dtx}")
    p(f"    {reason}")

# also: how many had a SELFDESTRUCT / deferred_threat bytecode signal?
p("\n\nmigrated contracts with SELFDESTRUCT/deferred-threat bytecode signal:")
sd = 0
for a in migrated:
    bc = c.execute("""SELECT b.bytecode_signals FROM contracts ct
        JOIN bytecode_cache b ON b.code_hash=ct.deployed_code_hash
        WHERE ct.contract_address=? AND b.bytecode_signals LIKE '%selfdestruct%'""",(a,)).fetchone()
    if bc:
        sd+=1
        p(f"    {a}: {bc[0][:120]}")
p(f"  total: {sd}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
