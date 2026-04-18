"""Probe deployer_profiles temporal dimensions for Pattern B design."""
import sqlite3, json

c = sqlite3.connect("/app/surveillance/data/surveillance.db")

cols = [r[1] for r in c.execute("PRAGMA table_info(deployer_profiles)")]
print("deployer_profiles cols:", cols)

print("\ntimezone_guess distribution:")
for r in c.execute(
    "SELECT timezone_guess, COUNT(*) FROM deployer_profiles "
    "GROUP BY timezone_guess ORDER BY 2 DESC"
):
    label = r[0] if r[0] else "None"
    print(f"  {label}: {r[1]}")

print("\norg-linked deployers timezone breakdown:")
for r in c.execute(
    "SELECT org_link, timezone_guess, COUNT(*) FROM deployer_profiles "
    "WHERE org_link IS NOT NULL AND org_link != '' "
    "GROUP BY org_link, timezone_guess ORDER BY 1, 3 DESC"
):
    print(f"  org={r[0]}  tz={r[1]}  n={r[2]}")

print("\nactive_window_start distribution (hour of day):")
for r in c.execute(
    "SELECT active_window_start, COUNT(*) FROM deployer_profiles "
    "WHERE active_window_start IS NOT NULL "
    "GROUP BY active_window_start ORDER BY 1"
):
    print(f"  start_hour={r[0]}  n={r[1]}")

print("\nactive_window_pct distribution (how concentrated in the window):")
rows = c.execute(
    "SELECT active_window_pct FROM deployer_profiles WHERE active_window_pct IS NOT NULL"
).fetchall()
if rows:
    vals = sorted([r[0] for r in rows])
    n = len(vals)
    print(f"  n={n}  min={vals[0]:.2f}  p50={vals[n//2]:.2f}  p90={vals[int(n*0.9)]:.2f}  max={vals[-1]:.2f}")

# Sample a few profiles
print("\nSample profiles (10 rows):")
for r in c.execute(
    "SELECT deployer_address, timezone_guess, peak_hour, active_window_start, "
    "active_window_pct, hour_concentration, deployment_style, active_days, "
    "total_contracts FROM deployer_profiles "
    "WHERE total_contracts >= 5 LIMIT 10"
):
    print(f"  {r[0][:18]}...  tz={r[1]}  peak={r[2]}  "
          f"win_start={r[3]}  win_pct={r[4]}  conc={r[5]}  "
          f"style={r[6]}  days={r[7]}  contracts={r[8]}")
