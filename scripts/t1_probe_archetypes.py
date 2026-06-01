"""Task 1, step 3 — characterize victim token-flow direction across the
contract archetypes, to define what a REAL approval-drain actually looks
like on-chain vs a distribution/airdrop mislabel.

For each sample contract, take up to 5 credited victims and classify their
transfer history of THAT token:
  OUT_to_contract_or_eoa : victim SENT the token (real drain candidate)
  IN_only                : victim only RECEIVED (distribution mislabel)
  none                   : no transfers of this token touch the victim

0 Alchemy CU.
"""
import json, sqlite3, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "_t1_archetypes.txt"
DB = ROOT / "surveillance" / "data" / "surveillance.db"
BASE = {"base": "https://base.blockscout.com/api/v2",
        "arbitrum": "https://arbitrum.blockscout.com/api/v2",
        "optimism": "https://optimism.blockscout.com/api/v2"}

SAMPLES = [
    ("0xa7e1e8ab7b7c93f9e3ceb10724843a4b74f5308c", "FIRE — 194 rows/99 tx (was 'real drainer')"),
    ("0xd6cd943bfc0711125bc01cff7b7dfb87be1d10c8", "Yupp AI — 118 rows/19 tx +SELFDESTRUCT"),
    ("0xb738b1568f08b0d6894a580ef805e9298ebfab46", "0xb738 — 1618 rows/2 tx (fan-out shape)"),
    ("0xb0a4741f19cde0bf2fd2ed598c55a6fe724c3653", "0xb0a4 — 319 rows/1 tx (fan-out shape)"),
    ("0xaa9c087543f791dfda8f060126b2d81b014901aa", "0xaa9c — 399 rows/5 tx"),
]
c = sqlite3.connect(DB)
L = []
def p(s=""): L.append(str(s))
def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "Mozilla/5.0 (L3-probe)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())

for contract, label in SAMPLES:
    chain = (c.execute("SELECT chain FROM contracts WHERE contract_address=?", (contract,)).fetchone() or ["base"])[0]
    base = BASE.get(chain, BASE["base"])
    victims = [r[0] for r in c.execute(
        "SELECT DISTINCT victim_address FROM approval_watchlist "
        "WHERE contract_address=? AND drain_detected=1 LIMIT 5", (contract,)).fetchall()]
    p("=" * 72)
    p(f"{label}")
    p(f"  contract={contract} chain={chain} sample_victims={len(victims)}")
    agg = {"OUT": 0, "IN_only": 0, "none": 0}
    for v in victims:
        try:
            d = get(f"{base}/addresses/{v}/token-transfers?type=ERC-20&token={contract}")
            items = d.get("items", [])
        except Exception as e:
            p(f"    {v[:12]} ERROR {type(e).__name__}")
            continue
        out_legs = sum(1 for it in items if ((it.get('from') or {}).get('hash') or '').lower() == v.lower())
        in_legs = sum(1 for it in items if ((it.get('to') or {}).get('hash') or '').lower() == v.lower())
        if out_legs > 0:
            cls = "OUT"; agg["OUT"] += 1
        elif in_legs > 0:
            cls = "IN_only"; agg["IN_only"] += 1
        else:
            cls = "none"; agg["none"] += 1
        p(f"    {v[:12]}  out_legs={out_legs:>3} in_legs={in_legs:>3}  -> {cls}")
        time.sleep(0.1)
    p(f"  SUMMARY: OUT(real-drain shape)={agg['OUT']}  IN_only(distribution mislabel)={agg['IN_only']}  none={agg['none']}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
