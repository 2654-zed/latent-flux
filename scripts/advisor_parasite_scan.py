"""Advisor-parasite pattern scan against the corpus.

Hypothesis: structurally similar to phishing drainers (hub-and-spoke approval
pattern, many victims to one collector) but with a radically different
TEMPORAL signature — months of small extractions, victims retain balance and
keep interacting, no single-shot drain.

What we can query: approval_events (approver, spender, timestamp),
transaction_events (interacting_address, contract_address, timestamp),
alerts (X402_AGENT_DRAIN victim list).

What we cannot query: advisor-parasite hubs that aren't in our monitored
contract set, victim funding chains, affinity-fraud signals (all off-chain).

Corpus age is ~30 days (ingest started 2026-03-17), so the 'months of
extractions' ideal signature is time-compressed to 'multi-week duration
with victims retaining balance'.

Output: candidates scored on duration + diversity + drain-exclusion,
ranked by the strength of the NON-drain signal.
"""
import sqlite3, json
from collections import Counter, defaultdict

DB = "/app/surveillance/data/surveillance.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# ---- Known legitimate infrastructure we should always exclude ----
KNOWN_INFRA = {
    # Permit2 (Uniswap canonical, all chains)
    "0x000000000022d473030f116ddee9f6b43ac78ba3",
    # Uniswap Universal Router (various addresses; just tag by pattern)
    # CCTP v2 from infrastructure_registry
    "0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
    "0x81d40f21f12a8f0e3252bccb954d722d4c464b64",
    "0xfd78ee919681417d192449715b2594ab58f5d002",
    "0xec546b6b005471ecf012e5af77fbec07e0fd8f78",
}
# Also pull the infrastructure_registry classifications
try:
    for r in c.execute("SELECT DISTINCT LOWER(address) FROM infrastructure_registry"):
        KNOWN_INFRA.add(r[0])
except sqlite3.Error:
    pass

# ---- Known drainer facilitators (the opposite of advisor-parasite) ----
# Pull from x402_facilitators classified rogue + high-hit-count unknowns
DRAINERS = set()
try:
    for r in c.execute("SELECT LOWER(address) FROM x402_facilitators WHERE classification = 'rogue'"):
        DRAINERS.add(r[0])
except sqlite3.Error:
    pass

# Drained victim set — approvers who later appeared as X402_AGENT_DRAIN payers
DRAINED_VICTIMS = set()
for r in c.execute("SELECT payload FROM alerts WHERE alert_type = 'X402_AGENT_DRAIN'"):
    try:
        p = json.loads(r[0])
        v = (p.get("payer") or "").lower()
        if v:
            DRAINED_VICTIMS.add(v)
    except Exception:
        pass

print(f"=== reference sets ===")
print(f"  known infrastructure addresses:  {len(KNOWN_INFRA)}")
print(f"  classified rogue drainers:       {len(DRAINERS)}")
print(f"  drained victim addresses (from alerts): {len(DRAINED_VICTIMS)}")
print()

# ---- Candidate query ----
# Spenders with:
#  - >= 50 unique approvers
#  - duration >= 14 days (gated by corpus age ~30 days)
#  - NOT in known infrastructure
#  - NOT in classified drainer set
# Then score each candidate on:
#  - drain exclusion rate: fraction of approvers who were NOT subsequently drained
#  - duration (longer = more advisor-like)
#  - approver count (higher = more hub-like)

candidates = c.execute(
    """SELECT LOWER(spender) as spender,
              COUNT(DISTINCT LOWER(approver)) as approver_ct,
              MIN(timestamp) as first_ts,
              MAX(timestamp) as last_ts,
              julianday(MAX(timestamp)) - julianday(MIN(timestamp)) as duration_days,
              COUNT(*) as approval_ct
       FROM approval_events
       GROUP BY LOWER(spender)
       HAVING approver_ct >= 50 AND duration_days >= 14
       ORDER BY duration_days DESC, approver_ct DESC"""
).fetchall()

print(f"=== candidates: ≥50 unique approvers, ≥14-day window ===")
print(f"  total before filtering: {len(candidates)}")

results = []
for r in candidates:
    spender = r["spender"]
    if spender in KNOWN_INFRA:
        continue
    if spender in DRAINERS:
        continue

    # How many approvers were drained?
    approvers = set(
        a[0].lower() for a in
        c.execute("SELECT DISTINCT approver FROM approval_events WHERE LOWER(spender) = ?", (spender,))
    )
    drained = approvers & DRAINED_VICTIMS
    retention_rate = 1 - (len(drained) / len(approvers)) if approvers else 0

    results.append({
        "spender": spender,
        "approver_ct": r["approver_ct"],
        "approval_ct": r["approval_ct"],
        "duration_days": round(r["duration_days"], 1),
        "first_ts": r["first_ts"][:10],
        "last_ts": r["last_ts"][:10],
        "drained_approvers": len(drained),
        "retention_rate": round(retention_rate, 3),
    })

# Rank by advisor-parasite signal strength:
#   high retention (approvers NOT drained) + long duration + many approvers
# Specifically: retention * duration * log(approvers)
import math
for r in results:
    r["score"] = round(
        r["retention_rate"] * r["duration_days"] * math.log(max(r["approver_ct"], 2)),
        2,
    )
results.sort(key=lambda x: -x["score"])

print(f"  after excluding known infra + rogue drainers: {len(results)}")
print()
print(f"{'score':>7}  {'spender':42}  {'appr':>5}  {'days':>5}  {'first':>10}  {'last':>10}  {'drained':>7}  {'retention':>9}")
for r in results[:25]:
    print(f"{r['score']:>7}  {r['spender']:42}  {r['approver_ct']:>5}  "
          f"{r['duration_days']:>5}  {r['first_ts']:>10}  {r['last_ts']:>10}  "
          f"{r['drained_approvers']:>7}  {r['retention_rate']:>9.3f}")

# ---- Cross-check via transaction_events: addresses receiving many distinct interactors ----
print()
print(f"=== cross-check: contracts with many distinct interacting_addresses (≥100, ≥14d) ===")
txc = c.execute(
    """SELECT LOWER(contract_address) as addr,
              COUNT(DISTINCT LOWER(interacting_address)) as interactor_ct,
              MIN(timestamp) as first_ts,
              MAX(timestamp) as last_ts,
              julianday(MAX(timestamp)) - julianday(MIN(timestamp)) as duration_days,
              COUNT(*) as tx_ct
       FROM transaction_events
       GROUP BY LOWER(contract_address)
       HAVING interactor_ct >= 100 AND duration_days >= 14
       ORDER BY duration_days DESC, interactor_ct DESC
       LIMIT 30"""
).fetchall()
print(f"{'addr':42}  {'interactors':>11}  {'days':>5}  {'tx':>7}  {'first':>10}  {'last':>10}  {'tier':>10}")
for r in txc:
    addr = r["addr"]
    if addr in KNOWN_INFRA:
        continue
    tier_row = c.execute(
        "SELECT confidence_tier FROM contracts WHERE LOWER(contract_address) = ?",
        (addr,),
    ).fetchone()
    tier = tier_row[0] if tier_row else "?"
    print(f"{addr:42}  {r['interactor_ct']:>11}  {r['duration_days']:>5.1f}  "
          f"{r['tx_ct']:>7}  {r['first_ts'][:10]:>10}  {r['last_ts'][:10]:>10}  {tier:>10}")
