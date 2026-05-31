"""FIX STEP 1 (zero Alchemy CU) — classify the 45 drain-tainted migrated
contracts by drain SHAPE, using only the local DB.

Distinguishes:
  REAL_DRAINER     — many distinct drain tx, few victims per tx
                     (sustained drain op; migration was a false negative)
  BUG19B_ARTIFACT  — 1-2 tx fanned to many victims (over-credit; migration
                     was correct, the "drain evidence" was never real)
  NEEDS_DECODE     — ambiguous; resolve via Blockscout token-transfers (step 2)

Metrics per contract (all from approval_watchlist + transaction_events):
  distinct_drain_tx, total_drain_rows, median victims/tx, max victims/tx,
  selector mix of the drain tx (transferFrom 23b872dd vs other),
  reverted-tx count among drain tx.

No mutations. Writes a JSON + a human report.
"""
import json, sqlite3, statistics
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_fix1_drain_shape.txt"
OUTJSON = Path(__file__).resolve().parent.parent / "reports" / "_fix1_drain_shape.json"
c = sqlite3.connect(DB)
L=[]
def p(s=""): L.append(str(s))

migrated = [r[0] for r in c.execute(
    "SELECT contract_address FROM contracts WHERE confidence_tier='unanalyzed' "
    "AND confidence_reason LIKE '%Correction #25%'")]

def norm(h):
    if not h: return h
    h=h.lower()
    return h[2:] if h.startswith('0x') else h

results=[]
for a in migrated:
    rows = c.execute("""SELECT drain_tx_hash, victim_address FROM approval_watchlist
        WHERE contract_address=? AND drain_detected=1 AND drain_tx_hash IS NOT NULL""",(a,)).fetchall()
    if not rows:
        continue
    per_tx={}
    for txh, victim in rows:
        per_tx.setdefault(txh, set()).add(victim)
    victims_per_tx = sorted((len(v) for v in per_tx.values()), reverse=True)
    distinct_tx = len(per_tx)
    total_rows = len(rows)
    med = statistics.median(victims_per_tx) if victims_per_tx else 0
    mx = max(victims_per_tx) if victims_per_tx else 0
    # selector mix + reverted among the drain tx
    sels={}; reverted=0; in_te=0
    for txh in per_tx:
        te = c.execute("""SELECT function_selector, is_reverted FROM transaction_events
            WHERE REPLACE(tx_hash,'0x','')=? LIMIT 1""",(norm(txh),)).fetchone()
        if te:
            in_te+=1
            sels[te[0]] = sels.get(te[0],0)+1
            if te[1]==1: reverted+=1
    transferFrom_tx = sels.get('23b872dd',0)
    # migration batch
    reason = c.execute("SELECT confidence_reason FROM contracts WHERE contract_address=?",(a,)).fetchone()[0] or ""
    if "Phase B" in reason: batch="B"
    elif "Phase C sample" in reason: batch="C-sample"
    elif "FROM_SOURCE" in reason: batch="C-source"
    elif "FROM_ACTIVITY" in reason: batch="C-activity"
    elif "FROM_CLUSTER" in reason: batch="C-cluster"
    else: batch="A"

    # zero-CU classification
    if distinct_tx >= 5 and med <= 5:
        verdict="REAL_DRAINER"          # sustained: many tx, few victims each
    elif distinct_tx <= 2 and mx >= 20:
        verdict="BUG19B_ARTIFACT"       # fan-out: 1-2 tx, many victims
    elif distinct_tx >= 5 and med <= 15:
        verdict="LIKELY_REAL_DECODE"    # leans real, verify
    else:
        verdict="NEEDS_DECODE"
    results.append({
        "address":a,"batch":batch,"distinct_tx":distinct_tx,"total_drain_rows":total_rows,
        "median_victims_per_tx":med,"max_victims_per_tx":mx,
        "transferFrom_tx":transferFrom_tx,"tx_in_te":in_te,"reverted_tx":reverted,
        "verdict":verdict,
    })

results.sort(key=lambda r:(r["verdict"], -r["distinct_tx"]))
from collections import Counter
vc=Counter(r["verdict"] for r in results)
p("="*72)
p(f"DRAIN-SHAPE CLASSIFICATION of {len(results)} drain-tainted migrated contracts")
p("="*72)
for k,v in vc.items(): p(f"  {k:20s}: {v}")
p("")
p(f"  {'address':44s} {'batch':9s} {'tx':>4s} {'rows':>5s} {'med/tx':>6s} {'max/tx':>6s} {'tF':>4s} {'rev':>4s}  verdict")
for r in results:
    p(f"  {r['address']} {r['batch']:9s} {r['distinct_tx']:>4} {r['total_drain_rows']:>5} "
      f"{r['median_victims_per_tx']:>6.0f} {r['max_victims_per_tx']:>6} {r['transferFrom_tx']:>4} "
      f"{r['reverted_tx']:>4}  {r['verdict']}")

OUT.write_text("\n".join(L), encoding="utf-8")
OUTJSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"wrote {OUT} and {OUTJSON} ({len(results)} contracts)")
