"""
Open-work investigation:
  (1) Root cause of the 20,936 evidence-free suspected mislabel (now cleaned).
      We look at the contracts AFTER the correction: which are in tier=unknown
      with a confidence_reason containing velocity/routing signatures? Those
      reveal the pipeline paths that mislabel.
  (2) Deployer-history inheritance audit: for each priority deployer, how
      many derivative-suspected contracts do they produce per confirmed trap?
      Find the most over-broad inheritance cases.
"""
import sqlite3
from collections import Counter

conn = sqlite3.connect('surveillance/data/surveillance.db', timeout=30)
conn.row_factory = sqlite3.Row

print("=" * 68)
print("(1) ROOT CAUSE: which paths produced evidence-free 'suspected' labels?")
print("=" * 68)

# Correction appended "[prior: <old reason>]" to confidence_reason. We can
# classify the downgraded contracts by their prior reason to see which paths
# contributed.
rows = conn.execute("""
    SELECT contract_address, confidence_reason
    FROM contracts
    WHERE confidence_tier = 'unknown'
      AND confidence_reason LIKE '%Reclassified from suspected%'
""").fetchall()
print(f"Total downgraded contracts: {len(rows)}")

# Classify prior-reason fingerprints
buckets = Counter()
samples = {}
for r in rows:
    reason = (r["confidence_reason"] or "").lower()
    if "velocity escalation" in reason:
        key = "velocity_escalation"
    elif "routing anomaly" in reason or "1inch pathfinder" in reason:
        key = "routing_anomaly"
    elif "[cache " in reason:
        key = "cache_transplant"
    elif "auto-suspected" in reason or "priority deployer" in reason:
        key = "auto_suspected_deployer"
    elif "bytecode exhibits" in reason:
        key = "bytecode_exhibits_but_flags_zero"
    elif "bytecode analyzed" in reason:
        key = "classifier_ran_no_patterns"
    else:
        key = "other"
    buckets[key] += 1
    if key not in samples:
        samples[key] = r["confidence_reason"][:200]

print("\nPath attribution (from prior confidence_reason):")
for path, cnt in buckets.most_common():
    pct = 100.0 * cnt / max(len(rows), 1)
    print(f"  {path:<38s} {cnt:>6,} ({pct:>5.1f}%)")

print("\nOne sample per path (truncated to 200 chars):")
for path, s in samples.items():
    print(f"  [{path}]")
    print(f"    {s}")

print()
print("=" * 68)
print("(2) DEPLOYER-HISTORY INHERITANCE AUDIT")
print("=" * 68)

# For every deployer, count their confirmed traps vs derivative suspected
audit = conn.execute("""
    SELECT
      c.deployer_address,
      SUM(CASE WHEN c.confidence_tier='confirmed' THEN 1 ELSE 0 END) as confirmed,
      SUM(CASE WHEN c.confidence_tier='suspected'
               AND c.detection_method='deployer_history' THEN 1 ELSE 0 END) as derivative,
      COUNT(*) as total
    FROM contracts c
    WHERE c.deployer_address IS NOT NULL AND c.deployer_address != ''
    GROUP BY c.deployer_address
    HAVING derivative > 0
""").fetchall()

print(f"Deployers with >=1 derivative-suspected contract: {len(audit):,}")

# Over-broad cases: 0 confirmed traps, many derivative suspected
no_confirmed = [a for a in audit if a["confirmed"] == 0]
print(f"\nDeployers with ZERO confirmed traps but derivative-suspected children: {len(no_confirmed):,}")
print("These are pure-inheritance flags -- the deployer was marked priority for")
print("a non-trap reason (velocity, funder, etc.) and their contracts inherited.")

# Distribution of inheritance breadth
buckets_inh = Counter()
for a in audit:
    d = a["derivative"]
    if d == 1:
        k = "1"
    elif d <= 5:
        k = "2-5"
    elif d <= 20:
        k = "6-20"
    elif d <= 100:
        k = "21-100"
    else:
        k = "100+"
    buckets_inh[k] += 1

print("\nDerivative-suspected breadth per deployer:")
for k in ["1", "2-5", "6-20", "21-100", "100+"]:
    print(f"  {k:<8s} contracts: {buckets_inh.get(k, 0):,} deployers")

# Ratio: derivative per confirmed trap (ignoring div/0)
have_both = [a for a in audit if a["confirmed"] > 0]
# Build plain dicts to avoid Row comparison errors
ratios = sorted(
    [
        {
            "deployer": a["deployer_address"],
            "conf": a["confirmed"],
            "deriv": a["derivative"],
            "ratio": a["derivative"] / a["confirmed"],
        }
        for a in have_both
    ],
    key=lambda x: x["ratio"],
    reverse=True,
)
print(f"\nDeployers with both confirmed and derivative: {len(have_both):,}")
print("Top 15 over-broad inheritance (highest derivative / confirmed ratio):")
print(f"{'deployer':<46s} {'conf':>5s} {'deriv':>6s} {'ratio':>8s}")
for x in ratios[:15]:
    print(f"  {x['deployer']}  {x['conf']:>4d} {x['deriv']:>6d} {x['ratio']:>8.1f}x")

print("\nBottom 5 by ratio (derivative flags look well-supported):")
for x in ratios[-5:]:
    print(f"  {x['deployer']}  conf={x['conf']:<4d} deriv={x['deriv']:<4d} ratio={x['ratio']:.2f}x")

# What fraction of the 31,976 derivative suspected come from deployers
# with ZERO confirmed traps?
no_conf_count = sum(a["derivative"] for a in no_confirmed)
total_deriv = sum(a["derivative"] for a in audit)
pct = 100.0 * no_conf_count / max(total_deriv, 1)
print(f"\nOf {total_deriv:,} derivative-suspected contracts:")
print(f"  {no_conf_count:,} ({pct:.1f}%) come from deployers with ZERO confirmed traps")
print(f"  {total_deriv - no_conf_count:,} come from deployers with >=1 confirmed trap")

# Distribution of confirmed-trap counts among derivative-producing deployers
print("\nConfirmed-trap count distribution among derivative-producing deployers:")
conf_buckets = Counter()
for a in audit:
    c = a["confirmed"]
    if c == 0:
        k = "0"
    elif c == 1:
        k = "1"
    elif c <= 5:
        k = "2-5"
    else:
        k = "6+"
    conf_buckets[k] += 1
for k in ["0", "1", "2-5", "6+"]:
    print(f"  confirmed={k:<4s}: {conf_buckets.get(k, 0):,} deployers")

conn.close()
