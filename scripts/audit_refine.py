"""Refinement queries for the data-integrity audit. Single clean run."""
import json, sqlite3
from pathlib import Path
DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_audit_refine.txt"
c = sqlite3.connect(DB)
L=[]
def p(s=""): L.append(str(s))

# 1. Refine the Correction #3/#4 signature: suspected + bytecode_pattern + ALL FLAGS ZERO
p("CORRECTION #3/#4 SIGNATURE — refined to all-flags-zero")
total_susp = c.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='suspected'").fetchone()[0]
susp_bp = c.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='suspected' AND detection_method='bytecode_pattern'").fetchone()[0]
p(f"  suspected total: {total_susp:,}")
p(f"  suspected + detection_method=bytecode_pattern: {susp_bp:,}")
# all-flags-zero among those with a cache row
allzero = c.execute("""
  SELECT COUNT(*) FROM contracts ct JOIN bytecode_cache b ON b.code_hash=ct.deployed_code_hash
  WHERE ct.confidence_tier='suspected' AND ct.detection_method='bytecode_pattern'
  AND b.bytecode_signals IS NOT NULL
  AND json_extract(b.bytecode_signals,'$.has_asymmetric_transfer') IN (0,'false',NULL)
  AND json_extract(b.bytecode_signals,'$.has_conditional_revert') IN (0,'false',NULL)
  AND json_extract(b.bytecode_signals,'$.has_unusual_fee_structure') IN (0,'false',NULL)
""").fetchone()[0]
p(f"  of those, bytecode_cache all-flags-zero (true Correction #3/#4 signature): {allzero:,}")
# suspected by detection_method overall
p("\n  suspected by detection_method:")
for r in c.execute("SELECT detection_method, COUNT(*) FROM contracts WHERE confidence_tier='suspected' GROUP BY detection_method ORDER BY 2 DESC"):
    p(f"    {str(r[0]):28s}: {r[1]:,}")

# 2. Bug #19b precise: drains where victim_address != actual transferFrom from-arg can't be checked w/o decoding,
#    so quantify: contracts where a SINGLE tx_hash is credited to N>1 victims (definitional over-credit)
p("\nBUG #19b — single tx_hash credited to multiple victims (definitional over-credit)")
multi = c.execute("""
  SELECT COUNT(*) FROM (
    SELECT drain_tx_hash, COUNT(DISTINCT victim_address) v
    FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL
    GROUP BY drain_tx_hash HAVING v > 1)
""").fetchone()[0]
total_tx = c.execute("SELECT COUNT(DISTINCT drain_tx_hash) FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL").fetchone()[0]
rows_in_multi = c.execute("""
  SELECT COALESCE(SUM(v),0) FROM (
    SELECT drain_tx_hash, COUNT(DISTINCT victim_address) v
    FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL
    GROUP BY drain_tx_hash HAVING v > 1)
""").fetchone()[0]
p(f"  drain tx_hashes credited to >1 victim: {multi:,} / {total_tx:,} distinct tx")
p(f"  drain rows attributable to those multi-victim tx: {rows_in_multi:,} / 7,227")
p(f"  (a legit batched/multicall drain CAN hit many victims in 1 tx, so this is an UPPER bound on Bug#19b inflation, not proof)")

# worst single tx
p("\n  worst single tx_hash by victim count:")
for r in c.execute("""
  SELECT drain_tx_hash, contract_address, COUNT(DISTINCT victim_address) v
  FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL
  GROUP BY drain_tx_hash ORDER BY v DESC LIMIT 6"""):
    p(f"    {str(r[0])[:24]}  contract={r[1][:18]}  victims={r[2]}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
