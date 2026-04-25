"""Pairwise bot-overlap retrospective across all known operator pairs.

For each pair of trap-deployers with >= MIN_HITS trap_events, compute:
  - bots_A: set of distinct bots that hit A's traps
  - bots_B: set of distinct bots that hit B's traps
  - overlap = |A ∩ B| / min(|A|, |B|)  (smaller-set fraction)

If 100% (or near-100%) overlap shows up in 3+ pairs across operators that
don't share funder or family, prey-driven synchronization is a pattern.
If only the A-B (604be06b / c0ffeefeed) pair shows it, anecdote.
"""
import sqlite3
from itertools import combinations
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
MIN_TRAPS_PER_OP = 3   # min trap_events to count an operator
MIN_OVERLAP_RATIO = 0.5  # threshold to print interesting pairs

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

# Pull operators with >= MIN_TRAPS trap events
ops = [r[0].lower() for r in c.execute(f"""
    SELECT ct.deployer_address, COUNT(*) AS n
    FROM trap_events te
    JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
    GROUP BY ct.deployer_address HAVING n >= {MIN_TRAPS_PER_OP}
    ORDER BY n DESC
""").fetchall()]
print(f"operators with >= {MIN_TRAPS_PER_OP} trap_events: {len(ops)}")

# Build bot-set per operator
op_bots = {}
op_meta = {}
for op in ops:
    bots = set()
    for r in c.execute("""
        SELECT DISTINCT LOWER(te.bot_address) FROM trap_events te
        JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
        WHERE LOWER(ct.deployer_address) = ?
    """, (op,)):
        bots.add(r[0])
    op_bots[op] = bots
    # operator metadata
    m = c.execute("""
        SELECT total_contracts_deployed,
               json_extract(funding_trail, '$.funder') AS funder,
               (SELECT COUNT(*) FROM contracts WHERE deployer_address = d.deployer_address
                  AND confidence_tier='confirmed') AS conf
        FROM deployers d WHERE LOWER(deployer_address) = ?
    """, (op,)).fetchone()
    op_meta[op] = {
        "fleet": m[0] if m else 0,
        "funder": (m[1] or "").lower() if m and m[1] else None,
        "confirmed": m[2] if m else 0,
        "n_bots": len(bots),
    }

# bytecode families per operator (for separation check)
op_fams = {}
for op in ops:
    fams = set()
    for r in c.execute("""
        SELECT DISTINCT bfm.family_id FROM bytecode_family_members bfm
        JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(bfm.contract_address)
        WHERE LOWER(ct.deployer_address) = ?
    """, (op,)):
        fams.add(r[0])
    op_fams[op] = fams

print()
print("=== Top operators by bot-fleet size ===")
for op in sorted(ops, key=lambda x: -op_meta[x]["n_bots"])[:15]:
    m = op_meta[op]
    print(f"  {op}  bots={m['n_bots']:<4} fleet={m['fleet']:<5} confirmed={m['confirmed']:<4} funder={m['funder']}")

print()
print(f"=== Pairs with overlap >= {MIN_OVERLAP_RATIO*100:.0f}% (sorted by overlap, then |A|+|B| desc) ===")
results = []
for a, b in combinations(ops, 2):
    sa, sb = op_bots[a], op_bots[b]
    if not sa or not sb:
        continue
    inter = sa & sb
    ratio = len(inter) / min(len(sa), len(sb))
    if ratio >= MIN_OVERLAP_RATIO:
        results.append({
            "a": a, "b": b,
            "a_bots": len(sa), "b_bots": len(sb),
            "intersect": len(inter),
            "ratio": ratio,
            "shared_funder": op_meta[a]["funder"] == op_meta[b]["funder"] if op_meta[a]["funder"] else False,
            "shared_family": bool(op_fams[a] & op_fams[b]),
        })

results.sort(key=lambda x: (-x["ratio"], -(x["a_bots"]+x["b_bots"])))
print(f"  {'A':<44} {'B':<44} {'A_bots':<7} {'B_bots':<7} {'shared':<7} {'ratio':<7} {'fund?':<6} {'fam?':<6}")
print("  " + "-" * 130)
for r in results[:30]:
    print(f"  {r['a']}  {r['b']}  {r['a_bots']:<7} {r['b_bots']:<7} {r['intersect']:<7} "
          f"{r['ratio']*100:>5.1f}%  {'Y' if r['shared_funder'] else 'N':<6} {'Y' if r['shared_family'] else 'N':<6}")

print()
print(f"=== Summary ===")
print(f"  total operator pairs evaluated: {len(list(combinations(ops, 2)))}")
print(f"  pairs >= {MIN_OVERLAP_RATIO*100:.0f}% overlap: {len(results)}")
print(f"  pairs at 100% overlap: {sum(1 for r in results if r['ratio'] == 1.0)}")
print(f"  pairs >= 90% overlap: {sum(1 for r in results if r['ratio'] >= 0.9)}")
print(f"  of those, separate-funder + separate-family: "
      f"{sum(1 for r in results if r['ratio'] >= 0.9 and not r['shared_funder'] and not r['shared_family'])}")

# 100% pairs that are NOT cluster-related (different funder + different family)
print()
print("=== 100% overlap pairs that are infrastructure-separate (different funder + different family) ===")
print("  These are the prey-driven synchronization candidates.")
for r in results:
    if r["ratio"] == 1.0 and not r["shared_funder"] and not r["shared_family"]:
        print(f"  A={r['a']}  B={r['b']}  shared={r['intersect']}/{min(r['a_bots'], r['b_bots'])} bots")

c.close()
