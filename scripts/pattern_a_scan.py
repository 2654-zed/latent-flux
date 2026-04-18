"""Pattern A — reputation-building sacrifices.

Hypothesis: a deployer operates N legitimate-looking contracts (real activity,
no trap signatures, low risk) before deploying a weaponized contract.

SQL proxy (no live risk_scoring calls — too slow at 80ms each):
For each deployer with ≥5 contracts:
  - count early deployments (first K) by confidence_tier
  - look at the latest deployment's confidence_tier
  - trajectory delta: early-unknown-ratio vs final-tier

Signal: deployer with ≥3 consecutive early 'unknown' contracts followed by
a 'confirmed' late deployment, where the tier change was NOT retroactive
(i.e., classification happened at or near deployment, not months later via
bulk reclassification).

Guardrails:
- Exclude deployers where ALL contracts share the same bytecode family
  (they're running one template, not a reputation trajectory)
- Minimum trajectory length: first deployment < final deployment by ≥ 3 days
- Final contract must be 'confirmed' tier with behavioral confirmation
  reason (bot-trapped), not just 'suspected' via deployer_history
"""
import json
import sqlite3

DB = "/app/surveillance/data/surveillance.db"

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# Pull deployers with >= 5 contracts
deps = c.execute(
    """SELECT deployer_address, COUNT(*) as n
       FROM contracts GROUP BY deployer_address
       HAVING n >= 5 ORDER BY n DESC"""
).fetchall()
print(f"deployers with >= 5 contracts: {len(deps)}")

candidates = []
for d in deps:
    dep = d["deployer_address"]
    rows = c.execute(
        """SELECT contract_address, confidence_tier, confidence_reason,
                  detection_timestamp, chain
           FROM contracts WHERE deployer_address = ?
           ORDER BY detection_timestamp""",
        (dep,),
    ).fetchall()
    if len(rows) < 5:
        continue

    early = rows[:3]
    final = rows[-1]

    # Require early all unknown
    if not all(r["confidence_tier"] == "unknown" for r in early):
        continue
    # Require final CONFIRMED via behavioral trigger
    if final["confidence_tier"] != "confirmed":
        continue
    final_reason = (final["confidence_reason"] or "").lower()
    if "behavioral confirmation" not in final_reason:
        continue

    # Trajectory duration
    from datetime import datetime
    try:
        t_early = datetime.fromisoformat(early[0]["detection_timestamp"].replace("Z", "+00:00"))
        t_final = datetime.fromisoformat(final["detection_timestamp"].replace("Z", "+00:00"))
        days = (t_final - t_early).total_seconds() / 86400
    except Exception:
        days = 0
    if days < 3:
        continue

    # Check: do early contracts share the same bytecode family with the final?
    # If yes, it's template replication, not reputation sacrifice.
    fam_counts = c.execute(
        """SELECT bytecode_families FROM deployer_profiles
           WHERE deployer_address = ?""",
        (dep,),
    ).fetchone()
    same_family_flag = "?"
    if fam_counts and fam_counts[0]:
        try:
            fams = json.loads(fam_counts[0])
            if isinstance(fams, list) and len(fams) == 1:
                same_family_flag = "single-family (likely template, NOT reputation trajectory)"
            elif isinstance(fams, dict) and len(fams) == 1:
                same_family_flag = "single-family (likely template, NOT reputation trajectory)"
            else:
                same_family_flag = f"{len(fams)} families"
        except Exception:
            pass

    candidates.append({
        "deployer": dep,
        "n_contracts": len(rows),
        "early_tiers": [r["confidence_tier"] for r in early],
        "final_tier": final["confidence_tier"],
        "final_reason": final_reason[:80],
        "trajectory_days": round(days, 1),
        "chain": final["chain"],
        "family_flag": same_family_flag,
    })

# Rank: longer trajectories are more advisor-like
candidates.sort(key=lambda x: -x["trajectory_days"])

print(f"\nPattern A candidates (early all 'unknown' → final 'confirmed' behavioral): {len(candidates)}")
print()
for r in candidates[:30]:
    print(f"  deployer={r['deployer']}  n={r['n_contracts']}  "
          f"days={r['trajectory_days']:>5}  chain={r['chain']:8s}  "
          f"family={r['family_flag']}")
    print(f"    final_reason: {r['final_reason']}")
