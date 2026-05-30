"""Triage the 7 contracts that are BOTH audit-migrated (confirmed->unanalyzed)
AND documented as confirmed traps. Use only local-DB signals to pre-sort
lean-FP vs lean-false-negative. NO conclusions in code — just dump facts.
"""
import sqlite3
from pathlib import Path
DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_audit_contradiction_triage.txt"
c = sqlite3.connect(DB)
L=[]
def p(s=""): L.append(str(s))

addrs = [
    "0x12577cf0d8a07363224d6909c54c056a183e13b3",
    "0xaeac0e69f6d2f6d88149cdca003c1689c9ed9eb8",
    "0xd6cd943bfc0711125bc01cff7b7dfb87be1d10c8",
    "0x44a2ee1369c3eecf86f8de7c73c3e3602523a198",
    "0x955b2c75efffa1ee9ee54e21e9c5c7cf772fdcb0",
    "0xa7e1e8ab7b7c93f9e3ceb10724843a4b74f5308c",
    "0x752c5a95d202972e124390f30a50154409d3c858",
]
for a in addrs:
    p("="*72)
    p(a)
    row = c.execute("""SELECT confidence_tier, detection_method, deployer_address,
        SUBSTR(confidence_reason,1,300) FROM contracts WHERE contract_address=?""",(a,)).fetchone()
    if not row:
        p("  NOT IN contracts table")
        continue
    p(f"  tier={row[0]}  method={row[1]}  deployer={row[2]}")
    p(f"  reason: {row[3]}")
    # approval + drain evidence
    aw = c.execute("""SELECT COUNT(*), SUM(CASE WHEN drain_detected=1 THEN 1 ELSE 0 END),
        COUNT(DISTINCT drain_tx_hash) FROM approval_watchlist WHERE contract_address=?""",(a,)).fetchone()
    p(f"  approval_watchlist: rows={aw[0]} drained={aw[1] or 0} distinct_drain_tx={aw[2] or 0}")
    # distinct drain tx detail (is the drain real per Bug#19b lens?)
    if aw[1]:
        for r in c.execute("""SELECT drain_tx_hash, COUNT(DISTINCT victim_address) v, drain_caller
            FROM approval_watchlist WHERE contract_address=? AND drain_detected=1
            GROUP BY drain_tx_hash ORDER BY v DESC LIMIT 3""",(a,)):
            # is that tx a transferFrom?
            te = c.execute("SELECT function_selector, is_reverted FROM transaction_events WHERE tx_hash=? OR tx_hash=? LIMIT 1",
                           (r[0], r[0] if str(r[0]).startswith('0x') else '0x'+str(r[0]))).fetchone()
            p(f"    drain_tx {str(r[0])[:20]} victims={r[1]} caller={str(r[2])[:16]} selector={te[0] if te else '?'} reverted={te[1] if te else '?'}")
    # bytecode evidence
    bc = c.execute("""SELECT b.bytecode_signals FROM contracts ct
        JOIN bytecode_cache b ON b.code_hash=ct.deployed_code_hash
        WHERE ct.contract_address=?""",(a,)).fetchone()
    p(f"  bytecode_signals: {bc[0] if bc else 'NO CACHE ROW'}")
    # blockscout enrichment if cached
    bs = c.execute("SELECT raw_json FROM audit_blockscout_cache WHERE address=? LIMIT 1",(a,)).fetchone()
    if bs and bs[0]:
        import json
        try:
            d = json.loads(bs[0])
            tok = d.get("token") or {}
            p(f"  blockscout: verified={d.get('is_verified')} name={d.get('name')} "
              f"token={tok.get('name') if isinstance(tok,dict) else None} "
              f"holders={tok.get('holders') if isinstance(tok,dict) else None}")
        except Exception as e:
            p(f"  blockscout parse err: {e}")
    else:
        p("  blockscout: not cached")
    # deployer recidivism
    if row[2]:
        rec = c.execute("SELECT COUNT(*) FROM contracts WHERE deployer_address=? AND confidence_tier='confirmed'",(row[2],)).fetchone()[0]
        p(f"  deployer other confirmed contracts: {rec}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
