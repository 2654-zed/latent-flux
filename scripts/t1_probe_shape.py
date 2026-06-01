"""Task 1, step 1 — DISCOVER the Blockscout token-transfers payload shape.
Do NOT assume keys. Dump the real structure of one known-real drain tx
(FIRE 0xa7e1e8ab7b, tx c4a74a86...) to a file for inspection.

0 Alchemy CU (Blockscout free REST).
"""
import json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "_t1_probe_shape.txt"

TX = "0xc4a74a86c018ccef328310d9880f05c10e7bd389a146df5aebb1a63814e2a692"  # FIRE top drain tx
CONTRACT = "0xa7e1e8ab7b7c93f9e3ceb10724843a4b74f5308c"
BASE = "https://base.blockscout.com/api/v2"

L = []
def p(s=""): L.append(str(s))

def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "Mozilla/5.0 (L3-probe)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())

p(f"TX: {TX}")
p(f"CONTRACT: {CONTRACT}")
p("=" * 70)

# (a) the token-transfers endpoint, type=ERC-20
url = f"{BASE}/transactions/{TX}/token-transfers?type=ERC-20"
p(f"\n[A] GET {url}")
try:
    d = get(url)
    items = d.get("items", [])
    p(f"  top-level keys: {list(d.keys())}")
    p(f"  item count: {len(items)}")
    if items:
        p(f"\n  --- FULL items[0] (pretty) ---")
        p(json.dumps(items[0], indent=2)[:2500])
        p(f"\n  --- key paths present in items[0] ---")
        def walk(o, prefix=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{prefix}.{k}")
            elif isinstance(o, list):
                walk(o[0] if o else None, f"{prefix}[0]")
            else:
                p(f"    {prefix} = {o!r}")
        walk(items[0])
except Exception as e:
    p(f"  ERROR: {type(e).__name__}: {e}")

# (b) same endpoint, NO type filter (in case ERC-20 filter drops items)
url2 = f"{BASE}/transactions/{TX}/token-transfers"
p(f"\n[B] GET {url2} (no type filter)")
try:
    d2 = get(url2)
    items2 = d2.get("items", [])
    p(f"  item count: {len(items2)}")
    # token-type distribution
    types = {}
    for it in items2:
        t = (it.get("token") or {}).get("type") if isinstance(it.get("token"), dict) else None
        types[t] = types.get(t, 0) + 1
    p(f"  token.type distribution: {types}")
except Exception as e:
    p(f"  ERROR: {type(e).__name__}: {e}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
