"""Consolidated data-integrity audit (dark-window 2026-05-27).

One script, one clean run, output written to a file then read via the
Read tool. Covers: drain Bug#19b residue, tier-vs-evidence consistency,
watchlist-22 characterization, OLI silent failure, SAI alert validity.

Read-only. No mutations.
"""
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_audit_consolidated.txt"
c = sqlite3.connect(DB)
L = []
def p(s=""): L.append(str(s))


def col_exists(table, col):
    return any(r[1] == col for r in c.execute(f"PRAGMA table_info({table})"))


# ---------------------------------------------------------------- DRAIN Bug#19b
p("=" * 72)
p("A. DRAIN INTEGRITY — Bug #19b (credits all approvers vs actual `from`)")
p("=" * 72)
allc = c.execute("""SELECT COUNT(*), COUNT(DISTINCT drain_tx_hash)
  FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL""").fetchone()
p(f"drain_detected=1 rows: {allc[0]:,} across {allc[1]:,} distinct tx_hashes")
p("\nTop drain/tx-ratio contracts (high ratio = one tx credited to many victims):")
for r in c.execute("""
  SELECT contract_address, COUNT(*) d, COUNT(DISTINCT drain_tx_hash) t,
         CAST(COUNT(*) AS REAL)/COUNT(DISTINCT drain_tx_hash) ratio
  FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL
  GROUP BY contract_address HAVING t>0 ORDER BY ratio DESC LIMIT 12"""):
    p(f"    {r[0]}  drains={r[1]:>5}  txs={r[2]:>4}  ratio={r[3]:>8.1f}")
hi = c.execute("""SELECT COALESCE(SUM(d),0) FROM (
    SELECT COUNT(*) d, COUNT(DISTINCT drain_tx_hash) t
    FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL
    GROUP BY contract_address HAVING CAST(COUNT(*) AS REAL)/COUNT(DISTINCT drain_tx_hash) >= 30)""").fetchone()[0]
p(f"\nDrains in ratio>=30 contracts (Bug #19b inflation candidates): {hi:,} ({100*hi/max(allc[0],1):.1f}%)")
# unique victims vs drain rows
uv = c.execute("SELECT COUNT(DISTINCT victim_address) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
p(f"Distinct victim_address among drains: {uv:,} (vs {allc[0]:,} drain rows)")

# ---------------------------------------------------------------- TIER EVIDENCE
p("\n" + "=" * 72)
p("B. TIER-VS-EVIDENCE CONSISTENCY (Correction #3/#4 recurrence check)")
p("=" * 72)
# confirmed contracts by detection_method
p("\nconfirmed-tier by detection_method:")
for r in c.execute("""SELECT detection_method, COUNT(*) FROM contracts
  WHERE confidence_tier='confirmed' GROUP BY detection_method ORDER BY 2 DESC"""):
    p(f"    {str(r[0]):28s}: {r[1]:,}")
# Correction #3/#4 signature: suspected + bytecode_pattern + all-flags-zero
p("\nCorrection #3/#4 mislabel signature (suspected + detection_method=bytecode_pattern\n  + bytecode_cache all-flags-zero):")
sig = c.execute("""
  SELECT COUNT(*) FROM contracts ct
  JOIN bytecode_cache b ON b.code_hash = ct.deployed_code_hash
  WHERE ct.confidence_tier='suspected' AND ct.detection_method='bytecode_pattern'
  AND b.bytecode_signals IS NOT NULL""").fetchone()[0]
p(f"    suspected+bytecode_pattern with a bytecode_cache row: {sig:,}")
# confirmed with NO bytecode evidence and NO approval activity and NO recidivism
p("\nconfirmed-tier 'thin evidence' triple-negative\n  (no bytecode_cache row AND no approval rows AND deployer has only this 1 confirmed):")
thin = c.execute("""
  SELECT COUNT(*) FROM contracts ct WHERE ct.confidence_tier='confirmed'
  AND NOT EXISTS (SELECT 1 FROM bytecode_cache b WHERE b.code_hash=ct.deployed_code_hash)
  AND NOT EXISTS (SELECT 1 FROM approval_watchlist a WHERE a.contract_address=ct.contract_address)
  AND (SELECT COUNT(*) FROM contracts c2 WHERE c2.deployer_address=ct.deployer_address
       AND c2.confidence_tier='confirmed') = 1
""").fetchone()[0]
conf_tot = c.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='confirmed'").fetchone()[0]
p(f"    {thin:,} / {conf_tot:,} confirmed contracts ({100*thin/max(conf_tot,1):.1f}%)")

# ---------------------------------------------------------------- WATCHLIST 22
p("\n" + "=" * 72)
p("C. WATCHLIST: 22 rows resolving to neither contract nor deployer")
p("=" * 72)
rows = c.execute("""
  SELECT w.address, w.address_type, w.entity_name, w.priority
  FROM watchlist w WHERE w.active=1
  AND NOT EXISTS (SELECT 1 FROM contracts c WHERE c.contract_address=w.address)
  AND NOT EXISTS (SELECT 1 FROM deployers d WHERE d.deployer_address=w.address)
  ORDER BY w.priority DESC, w.entity_name""").fetchall()
p(f"\n{len(rows)} unresolved active watchlist rows:")
for r in rows:
    p(f"    {r[0]}  [{r[1] or '?'}] {(r[2] or '')[:42]:42s} {r[3]}")

# ---------------------------------------------------------------- OLI
p("\n" + "=" * 72)
p("D. OLI ENRICHMENT SILENT FAILURE (priority #22)")
p("=" * 72)
oli_cols = [r[1] for r in c.execute("PRAGMA table_info(oli_labels)")]
p(f"oli_labels columns: {oli_cols}")
p(f"oli_labels total rows: {c.execute('SELECT COUNT(*) FROM oli_labels').fetchone()[0]}")
for r in c.execute("SELECT * FROM oli_labels LIMIT 15"):
    p(f"    {r}")

# ---------------------------------------------------------------- SAI ALERTS
p("\n" + "=" * 72)
p("E. SAI ALERT VALIDITY (alerts built on retracted data?)")
p("=" * 72)
p(f"sai_alerts total: {c.execute('SELECT COUNT(*) FROM sai_alerts').fetchone()[0]}")
p("by detector + severity:")
for r in c.execute("SELECT detector, severity, COUNT(*) FROM sai_alerts GROUP BY detector, severity ORDER BY 1,2"):
    p(f"    {r[0]:8s} {str(r[1]):26s}: {r[2]}")
# SAI alerts whose subject_address was migrated to unanalyzed by the audit
mig = c.execute("""
  SELECT COUNT(*) FROM sai_alerts s
  WHERE EXISTS (SELECT 1 FROM contracts c WHERE c.contract_address=s.subject_address
               AND c.confidence_tier='unanalyzed' AND c.confidence_reason LIKE '%Correction #25%')""").fetchone()[0]
p(f"\nSAI alerts whose subject is an audit-migrated (now-legitimate) contract: {mig}")
# SAI alerts referencing a deployer of a migrated contract
p("subject_kind distribution:")
for r in c.execute("SELECT subject_kind, COUNT(*) FROM sai_alerts GROUP BY subject_kind ORDER BY 2 DESC"):
    p(f"    {str(r[0]):28s}: {r[1]}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT} ({len(L)} lines)")
