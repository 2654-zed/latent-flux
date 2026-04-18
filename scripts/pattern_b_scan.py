"""Pattern B — temporal pattern normalization.

Hypothesis: operators deliberately deploying on a Western-workday schedule
(9-5 Europe / US morning hours) to look like a regular dev team, while
their contracts score suspicious on OTHER dimensions.

Baseline: among org_001 deployers, 'night_shift' (3AM Americas / 10AM Asia)
and 'asia_morning' dominate (89 + 50 = 139 of 236 labeled, or 59%).
'europe_business' represents 12% of org_001. Among the corpus at large,
'europe_business' is ~14%. So a europe_business deployer is NOT automatically
suspect — many legitimate Western devs also deploy during workday.

Signal we're looking for:
  europe_business AND active_window_pct >= 0.7 (concentrated in that window)
    AND active_days >= 14 (sustained pattern)
    AND has contracts with tier in (suspected, confirmed)
    AND NOT currently org_001-004 linked

Baseline comparison: same filter with tz != europe_business to see if
europe_business over- or under-indexes on trap-adjacency.
"""
import sqlite3
from collections import Counter

DB = "/app/surveillance/data/surveillance.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# Baseline: total trap-adjacent count per timezone_guess
print("=== trap-adjacency rate by timezone_guess (sustained deployers, active_days >= 14, window_pct >= 0.7) ===")
rows = c.execute(
    """SELECT dp.timezone_guess,
              COUNT(*) as n_deployers,
              SUM(CASE WHEN exists_trap.trap_ct > 0 THEN 1 ELSE 0 END) as n_with_trap
       FROM deployer_profiles dp
       LEFT JOIN (
         SELECT deployer_address, COUNT(*) as trap_ct
         FROM contracts WHERE confidence_tier IN ('suspected', 'confirmed')
         GROUP BY deployer_address
       ) exists_trap ON exists_trap.deployer_address = dp.deployer_address
       WHERE dp.active_days >= 14 AND dp.active_window_pct >= 0.7
       GROUP BY dp.timezone_guess
       ORDER BY 2 DESC"""
).fetchall()
for r in rows:
    rate = r["n_with_trap"] / r["n_deployers"] if r["n_deployers"] else 0
    print(f"  {str(r['timezone_guess']):20s}  deployers={r['n_deployers']:>5}  with_trap={r['n_with_trap']:>5}  rate={rate:.1%}")

# Candidate list: europe_business, sustained, with trap-adjacency, not org-linked
print("\n=== Pattern B candidates ===")
cands = c.execute(
    """SELECT dp.deployer_address, dp.peak_hour, dp.active_window_start,
              dp.active_window_pct, dp.active_days, dp.primary_technique,
              dp.total_contracts, dp.deployment_style, dp.org_link,
              trap.trap_ct, trap.conf_ct
       FROM deployer_profiles dp
       JOIN (
         SELECT deployer_address, COUNT(*) as trap_ct,
                SUM(CASE WHEN confidence_tier = 'confirmed' THEN 1 ELSE 0 END) as conf_ct
         FROM contracts WHERE confidence_tier IN ('suspected', 'confirmed')
         GROUP BY deployer_address
       ) trap ON trap.deployer_address = dp.deployer_address
       WHERE dp.timezone_guess = 'europe_business'
         AND dp.active_window_pct >= 0.7
         AND dp.active_days >= 14
         AND (dp.org_link IS NULL OR dp.org_link = '')
         AND trap.trap_ct >= 1
       ORDER BY trap.conf_ct DESC, trap.trap_ct DESC"""
).fetchall()

print(f"total: {len(cands)} europe_business deployers with trap-adjacency, ≥14 active days, ≥0.7 window concentration, not org-linked")
print()
print(f"{'deployer':44}  {'contracts':>9}  {'win_start':>9}  {'peak_hr':>7}  {'active_days':>11}  {'trap_ct':>7}  {'confirmed':>9}  {'technique':15}  {'style':10}")
for r in cands[:40]:
    print(f"  {r['deployer_address']}  "
          f"{r['total_contracts']:>9}  "
          f"{r['active_window_start']:>9}  "
          f"{r['peak_hour']:>7}  "
          f"{r['active_days']:>11}  "
          f"{r['trap_ct']:>7}  "
          f"{r['conf_ct']:>9}  "
          f"{str(r['primary_technique'])[:15]:15}  "
          f"{str(r['deployment_style'])[:10]}")

# Extra diagnostic: show a few normalized deployers' hour_distribution if available
print(f"\n=== sample hour_distribution for top candidates ===")
for r in cands[:3]:
    dep = r["deployer_address"]
    hd = c.execute(
        "SELECT hour_distribution FROM deployer_profiles WHERE deployer_address = ?",
        (dep,),
    ).fetchone()
    if hd and hd[0]:
        print(f"  {dep}: {hd[0][:200]}")
