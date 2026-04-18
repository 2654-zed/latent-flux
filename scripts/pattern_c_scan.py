"""Pattern C — CEX-laundered funding.

Hypothesis: operator funds deployer from a CEX hot wallet so auto_funder_tracer
concludes 'legitimate origin.' But the USDC/ETH the operator deposited into
that CEX hot wallet came from a wallet with trap-adjacent behavior.

Query approach (no RPC this pass; flag whether RPC would help):
1. Pull CEX hot wallets from entity_classification (category INFRASTRUCTURE,
   subtype cex_hot_wallet).
2. Find deployers whose funding_trail.funder is in that set.
3. For each, report: deployer address, CEX hot wallet used, deployer
   contract counts (confirmed/suspected/unknown), chain, first_seen.
4. Report the list; RPC budget for one-hop-back trace (up to 200 calls)
   is authorized but not spent this pass — the SQL surface is incomplete
   on its own, so we flag where RPC would go next.
"""
import sqlite3, json

DB = "/app/surveillance/data/surveillance.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# 1. CEX hot wallets from entity_classification
cex_hots = set()
for r in c.execute(
    "SELECT LOWER(address) FROM entity_classification "
    "WHERE category = 'INFRASTRUCTURE' AND subtype = 'cex_hot_wallet'"
):
    cex_hots.add(r[0])
print(f"CEX hot wallets in corpus: {len(cex_hots)}")
for a in sorted(cex_hots):
    print(f"  {a}")

# 2. Deployers whose funding_trail.funder is one of those
print("\nDeployers funded from CEX hot wallets:")
dep_matches = []
for r in c.execute("SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL"):
    try:
        t = json.loads(r["funding_trail"])
    except Exception:
        continue
    funder = (t.get("funder") or "").lower()
    if funder in cex_hots:
        dep_matches.append({
            "deployer": r["deployer_address"],
            "funder": funder,
            "value_eth": t.get("value_eth"),
            "traced_at": t.get("traced_at"),
            "org_link": t.get("org_link"),
        })

print(f"  count: {len(dep_matches)}")
for d in dep_matches[:30]:
    print(f"  deployer={d['deployer']}  funder={d['funder'][:18]}...  "
          f"value_eth={d['value_eth']}  org={d['org_link']}")

# 3. For each match: deployment tier breakdown
print("\nFor each CEX-funded deployer — contract tier distribution:")
for d in dep_matches[:30]:
    dep = d["deployer"]
    tiers = dict(c.execute(
        "SELECT confidence_tier, COUNT(*) FROM contracts "
        "WHERE deployer_address = ? GROUP BY confidence_tier",
        (dep,),
    ).fetchall())
    total = sum(tiers.values())
    conf = tiers.get("confirmed", 0)
    susp = tiers.get("suspected", 0)
    unk = tiers.get("unknown", 0)
    verdict = ""
    if conf > 0:
        verdict = " ** HAS CONFIRMED TRAPS **"
    elif susp >= 3:
        verdict = " * SUSPECTED-HEAVY *"
    print(f"  {dep}  total={total:>3}  confirmed={conf:>3}  suspected={susp:>3}  unknown={unk:>3}{verdict}")

# 4. Subset: deployers that are both CEX-funded AND have confirmed traps — Pattern C candidates
print("\nPattern C candidates (CEX-funded + has confirmed traps):")
candidates = []
for d in dep_matches:
    conf = c.execute(
        "SELECT COUNT(*) FROM contracts WHERE deployer_address = ? AND confidence_tier = 'confirmed'",
        (d["deployer"],),
    ).fetchone()[0]
    if conf > 0:
        candidates.append({
            "deployer": d["deployer"],
            "funder": d["funder"],
            "confirmed_ct": conf,
            "value_eth": d["value_eth"],
        })
candidates.sort(key=lambda x: -x["confirmed_ct"])
for r in candidates:
    print(f"  deployer={r['deployer']}  confirmed={r['confirmed_ct']}  "
          f"funded_from={r['funder'][:18]}...  value_eth={r['value_eth']}")

print(f"\nPattern C candidate count: {len(candidates)}")
print()
print("RPC budget note: for each candidate, one-hop-back-from-CEX trace would")
print("require ~1-5 RPC calls to fetch the CEX deposit tx and its origin. At")
print(f"{len(candidates)} candidates × ~3 calls each = ~{len(candidates)*3} total — within the 200-call")
print("budget. NOT spent in this pass; the candidate list itself is the")
print("primary deliverable and we pause for review before chasing pre-CEX origins.")
