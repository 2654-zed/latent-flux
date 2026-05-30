"""Verify the local DB's Phase-0 drain-backfill state (Correction #24 / Bug #19).

Phase 0 reset phantom drain rows that mapped to reverted txs. Expected
post-fix local drain_detected=1 ~= 7,227. Confirms the reset persisted
and quantifies Bug #19b drain/tx-ratio outliers. Read-only.
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_audit_phase0_state.txt"
c = sqlite3.connect(DB)
L = []
def p(s=""): L.append(str(s))

drain1 = c.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
drain1_tx = c.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL").fetchone()[0]
total = c.execute("SELECT COUNT(*) FROM approval_watchlist").fetchone()[0]
p(f"approval_watchlist total rows:   {total:,}")
p(f"  drain_detected=1:              {drain1:,}  (post-Phase-0 expected ~7,227; pre-fix ~11,850)")
p(f"    with drain_tx_hash:          {drain1_tx:,}")

rev = c.execute("""SELECT COUNT(*) FROM approval_watchlist aw
  WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
  AND EXISTS (SELECT 1 FROM transaction_events te
              WHERE REPLACE(te.tx_hash,'0x','')=REPLACE(aw.drain_tx_hash,'0x','') AND te.is_reverted=1)""").fetchone()[0]
p(f"\n[normalized] drain rows -> REVERTED tx (should be 0 post-Phase-0): {rev:,}")

p("\nTop drain/tx-ratio contracts (Bug #19b 'credits all approvers' signature):")
for r in c.execute("""SELECT contract_address, COUNT(*) d, COUNT(DISTINCT drain_tx_hash) t,
    CAST(COUNT(*) AS REAL)/COUNT(DISTINCT drain_tx_hash) ratio
    FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL
    GROUP BY contract_address HAVING t>0 ORDER BY ratio DESC LIMIT 12"""):
    p(f"    {r[0]}  drains={r[1]:>5} txs={r[2]:>4} ratio={r[3]:>8.1f}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
