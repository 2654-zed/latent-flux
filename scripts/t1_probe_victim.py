"""Task 1, step 2 — find where the victim->contract drain leg actually lives.

For FIRE, take 3 credited victims and pull their ERC-20 transfer history for
the FIRE token via Blockscout address token-transfers, filtered to the FIRE
token. Question we must answer: does a credited victim show a Transfer of FIRE
with from=victim (the drain pull)? In which tx? Does that tx match the stored
drain_tx_hash, or is the stored hash the contract's later dump tx?

0 Alchemy CU.
"""
import json, sqlite3, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "_t1_probe_victim.txt"
DB = ROOT / "surveillance" / "data" / "surveillance.db"
BASE = "https://base.blockscout.com/api/v2"
CONTRACT = "0xa7e1e8ab7b7c93f9e3ceb10724843a4b74f5308c"

c = sqlite3.connect(DB)
L = []
def p(s=""): L.append(str(s))
def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "Mozilla/5.0 (L3-probe)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())

# 3 credited victims of FIRE + their stored drain tx
rows = c.execute("""SELECT DISTINCT victim_address, drain_tx_hash
    FROM approval_watchlist
    WHERE contract_address=? AND drain_detected=1 AND drain_tx_hash IS NOT NULL
    LIMIT 3""", (CONTRACT,)).fetchall()
p(f"CONTRACT (token): {CONTRACT}")
p(f"sample credited victims + stored drain_tx_hash:")
for v, t in rows:
    p(f"  victim={v}  drain_tx={t}")
p("=" * 70)

for victim, stored_tx in rows:
    p(f"\n### victim {victim}")
    p(f"    stored drain_tx: {stored_tx}")
    # address token-transfers for this victim, filter to FIRE token
    url = f"{BASE}/addresses/{victim}/token-transfers?type=ERC-20&token={CONTRACT}"
    try:
        d = get(url)
        items = d.get("items", [])
        p(f"    FIRE transfers touching this victim: {len(items)}")
        for it in items[:6]:
            frm = ((it.get("from") or {}).get("hash") or "").lower()
            to = ((it.get("to") or {}).get("hash") or "").lower()
            txh = (it.get("transaction_hash") or "").lower()
            direction = "OUT(from=victim)" if frm == victim.lower() else ("IN(to=victim)" if to == victim.lower() else "other")
            match = "  <== MATCHES stored drain_tx" if txh.replace("0x","") == str(stored_tx).lower().replace("0x","") else ""
            p(f"      {direction:18s} from={frm[:12]} to={to[:12]} tx={txh[:14]}{match}")
    except Exception as e:
        p(f"    ERROR: {type(e).__name__}: {e}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
