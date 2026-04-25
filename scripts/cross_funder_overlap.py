"""Cross-funder downstream-deployer overlap.

For each pair of the 12 top funders, intersect their funded-deployers sets.
Any non-zero overlap is structural: same downstream actor seeded from
multiple 'independent' funders. Distinguishes single-actor-multi-wallet
from many-independent-operators.

Also walks one hop further: are any of the 12 funders themselves funded
by the same upstream? (We have eth-trace data for all 12; checking the
inbound[].from sets for shared mainnet sources.)
"""
import json
import sqlite3
from itertools import combinations
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
TRACE_DIR = Path(r"C:\Users\jason\Desktop\ai lang\scripts\funder_traces")
F70_TRACE = Path(r"C:\Users\jason\Desktop\ai lang\scripts\eth_trace_f70da978.json")

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

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

# Build downstream sets per funder (deployers funded by each)
print("Loading downstream-deployer sets per funder...")
fanout = {}
for fdr in TOP_12:
    deps = {r[0].lower() for r in c.execute(
        "SELECT deployer_address FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?",
        (fdr,))}
    fanout[fdr] = deps
    print(f"  {fdr}: {len(deps):,} deployers")

h("1. Pairwise downstream overlap")
print(f"  {'A':<44} {'B':<44} {'A_n':<6} {'B_n':<6} {'shared':<7} {'%min'}")
results = []
for a, b in combinations(TOP_12, 2):
    inter = fanout[a] & fanout[b]
    if inter:
        share_a = len(inter) / len(fanout[a]) * 100
        share_b = len(inter) / len(fanout[b]) * 100
        results.append({
            "a": a, "b": b, "a_n": len(fanout[a]), "b_n": len(fanout[b]),
            "shared": len(inter), "share_min": min(share_a, share_b)
        })
        print(f"  {a}  {b}  {len(fanout[a]):<6,} {len(fanout[b]):<6,} {len(inter):<7,} "
              f"{min(share_a, share_b):>5.2f}%")
    # else: silent (zero overlap is the common case)

print()
print(f"  pairs with non-zero overlap: {len(results)} of {len(list(combinations(TOP_12, 2)))}")
print(f"  pairs with >= 5% overlap: {sum(1 for r in results if r['share_min'] >= 5)}")
print(f"  pairs with >= 25% overlap: {sum(1 for r in results if r['share_min'] >= 25)}")
print(f"  pairs with >= 50% overlap: {sum(1 for r in results if r['share_min'] >= 50)}")

h("2. Total deployers in any top-12 fanout (union)")
union = set()
for fdr in TOP_12:
    union |= fanout[fdr]
total_funded_distinct = len(union)
sum_individual = sum(len(fanout[f]) for f in TOP_12)
print(f"  union (distinct):         {total_funded_distinct:,}")
print(f"  sum of individual sets:   {sum_individual:,}")
print(f"  duplicate-funding count:  {sum_individual - total_funded_distinct:,}")
if sum_individual > 0:
    print(f"  duplication rate:         {(sum_individual-total_funded_distinct)/sum_individual*100:.2f}%")

# How many deployers are funded by 2+ of the 12?
multi_funded = []
for d in union:
    cnt = sum(1 for f in TOP_12 if d in fanout[f])
    if cnt >= 2:
        multi_funded.append((d, cnt))
multi_funded.sort(key=lambda x: -x[1])
print(f"  deployers funded by >=2 of the 12: {len(multi_funded)}")
print(f"  deployers funded by >=3 of the 12: {sum(1 for _, n in multi_funded if n >= 3)}")
if multi_funded:
    print("  Top 10 multi-funded deployers (by funder count):")
    for d, n in multi_funded[:10]:
        # Per-deployer fleet info
        r = c.execute("""
            SELECT chain, total_contracts_deployed, first_seen FROM deployers WHERE LOWER(deployer_address) = ?
        """, (d,)).fetchone()
        if r:
            print(f"    {d}  funded_by={n}  chain={r[0]}  fleet={r[1]}  first={r[2][:10]}")

h("3. Upstream mainnet-source overlap (do the 12 share inbound counterparties?)")
trace_files = [F70_TRACE] + sorted(TRACE_DIR.glob("*.json"))
upstream_sources = {}  # funder_addr -> set(inbound_from)
for f in trace_files:
    try:
        data = json.load(open(f))
    except Exception:
        continue
    addr = data.get("address", "").lower()
    inbound_set = {tx["from"].lower() for tx in data.get("inbound", [])}
    upstream_sources[addr] = inbound_set

# Pairwise upstream overlap
print("  Pairs where the two funders share at least one upstream mainnet sender:")
for a, b in combinations(sorted(upstream_sources.keys()), 2):
    inter = upstream_sources[a] & upstream_sources[b]
    if inter:
        print(f"    {a}  ∩  {b}: {len(inter)} shared upstream(s)")
        for src in list(inter)[:5]:
            print(f"        {src}")

# Aggregate: which mainnet addresses fund 2+ of the 12?
upstream_count = {}
for fdr, srcs in upstream_sources.items():
    for src in srcs:
        upstream_count.setdefault(src, set()).add(fdr)
shared_upstream = [(src, fdrs) for src, fdrs in upstream_count.items() if len(fdrs) >= 2]
print(f"\n  mainnet addresses that funded 2+ of the 12: {len(shared_upstream)}")
shared_upstream.sort(key=lambda x: -len(x[1]))
for src, fdrs in shared_upstream[:10]:
    print(f"    {src}  funds {len(fdrs)} of the 12: {sorted(fdrs)}")

c.close()
