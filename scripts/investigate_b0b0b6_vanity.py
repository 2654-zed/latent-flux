"""0xb0b0b6 vanity-prefix investigation.

The L2-only Optimism funder 0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07
has 7 chars of vanity prefix (b0b0b69 → ~268M attempts to generate).
Vanity addresses are not accidental. Two prior corpus precedents:
  - 0xc0ffee... (Coffee Fleet — branding/identity, 84 bots + 1 deployer)
  - org_001 shadow wallets (vanity-spoofed prefix collision for
    intelligence-layer anti-forensics)

Questions:
  1. Are there other 0xb0b0b6 / 0xb0b0b... vanity addresses in the corpus?
  2. Is 0xb0b0b690's downstream (5,775+ deployers) also vanity-prefixed?
  3. Are the microETH-return senders documented earlier vanity-themed?
  4. Cross-reference with org_001 / org_002 / Coffee Fleet vanity patterns.
"""
import sqlite3
import re
from collections import Counter
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
TARGET = "0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

h("1. Other 0xb0b0... vanity addresses in corpus")
print("  Searching deployers, contracts, bot_candidates, watchlist:")
for table, col in [
    ("deployers", "deployer_address"),
    ("contracts", "contract_address"),
    ("contracts", "deployer_address"),
    ("bot_candidates", "address"),
    ("watchlist", "address"),
]:
    try:
        rows = list(c.execute(f"SELECT DISTINCT LOWER({col}) AS a FROM {table} WHERE LOWER({col}) LIKE '0xb0b0b6%'"))
        if rows:
            print(f"  {table}.{col}: {len(rows)} match")
            for r in rows[:8]:
                print(f"    {r['a']}")
    except sqlite3.OperationalError:
        pass

h("2. Is 0xb0b0b690's downstream fleet vanity-prefixed?")
# Sample the 5,775 deployers funded by 0xb0b0b690
deployers = [r[0].lower() for r in c.execute(
    "SELECT deployer_address FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?",
    (TARGET.lower(),))]
print(f"  Total downstream deployers: {len(deployers)}")
# How many start with shared vanity prefixes?
prefix_counts = Counter()
for d in deployers:
    prefix_counts[d[:6]] += 1  # 4-byte prefix (8 hex chars after 0x)
top = prefix_counts.most_common(15)
print("  Top 15 4-byte prefixes among downstream deployers (random distribution would have all near 1):")
for p, n in top:
    print(f"    {p}  {n}")

# Most are random — but check specifically vanity-shape prefixes
print()
print("  Vanity-shape prefix matches (deployers starting with these):")
for vp in ["0xb0b0", "0xc0ffee", "0xdeadbe", "0xc0c0", "0xfeed", "0x1234", "0x0000"]:
    n = sum(1 for d in deployers if d.startswith(vp))
    if n:
        print(f"    {vp}*  {n}")

h("3. Microtransaction senders to 0xb0b0b690 — vanity?")
# Pre-fetch some sender addresses from earlier L2 trace data
# The earlier l2_trace_v2.py output showed inbound senders like
# 0x797e6979ba79f59c8a09360c689d3e38ac7766e1, 0xa4e9bab86a644cedb35998db75d67db210c90a30, etc.
# Those don't have vanity prefixes — but worth checking if any returned ETH from a vanity address
trace_path = Path("/app/scripts/funder_traces_l2") / f"{TARGET}_optimism.json"
import os, json
local_trace = Path(r"C:\Users\jason\Desktop\ai lang\scripts\funder_traces_l2") / f"{TARGET}_optimism.json"
trace_data = None
for p in [local_trace, Path(r"C:\Users\jason\Desktop\ai lang\scripts\eth_trace_b0b0b6.json")]:
    if p.exists():
        try:
            trace_data = json.load(open(p))
            print(f"  Loaded: {p}")
            break
        except Exception as e:
            print(f"  failed loading {p}: {e}")
if not trace_data:
    print("  No local trace data — would need fresh L2 trace via Railway")
else:
    inbound = trace_data.get("inbound", [])
    outbound = trace_data.get("outbound", [])
    print(f"  inbound: {len(inbound)}, outbound: {len(outbound)}")
    senders = [tx.get("from", "").lower() for tx in inbound if tx.get("from")]
    sender_prefixes = Counter(s[:8] for s in senders)
    print("  Top 5 sender prefixes:")
    for p, n in sender_prefixes.most_common(5):
        print(f"    {p}  {n}")
    # Vanity check
    vanity = [s for s in senders if s.startswith("0xb0b0") or "0xc0ffee" in s or s.startswith("0x0000")]
    if vanity:
        print(f"  Vanity-prefixed senders: {len(vanity)}")
        for v in set(vanity)[:5]:
            print(f"    {v}")

h("4. Comparison with documented vanity infrastructure")
# Coffee Fleet bot count
n_coffee_bots = c.execute("SELECT COUNT(*) FROM bot_candidates WHERE LOWER(address) LIKE '0xc0ffee%'").fetchone()[0]
n_coffee_deployers = c.execute("SELECT COUNT(*) FROM deployers WHERE LOWER(deployer_address) LIKE '0xc0ffee%'").fetchone()[0]
print(f"  0xc0ffee* in corpus: {n_coffee_bots} bots + {n_coffee_deployers} deployers (Coffee Fleet, branding/identity)")

# org_001 shadow wallets are vanity-spoofed against legitimate addresses
print(f"  0x01989c93890aed05* vanity-spoofing pair (org_001 Shadow Wallet 1):")
for a in c.execute("SELECT DISTINCT LOWER(deployer_address) FROM deployers WHERE LOWER(deployer_address) LIKE '0x01989c93890aed05%'"):
    print(f"    {a[0]}")

# Vanity prefix concentrations across all deployers
print()
print("  Top vanity-prefix concentrations across the entire deployers table (4+ instances of same 6-char prefix):")
all_deps = [r[0].lower() for r in c.execute("SELECT deployer_address FROM deployers")]
all_prefixes = Counter(d[:8] for d in all_deps)  # 6-char prefix after 0x
for p, n in sorted(all_prefixes.items(), key=lambda x: -x[1]):
    if n >= 4 and not p.startswith("0x000"):  # exclude generic patterns
        # Skip prefixes that are likely just statistical
        if n >= 10:
            print(f"    {p}*  {n} addresses")

h("5. 0xb0b0b690 specific timing pattern")
# When was the funder first seen? When did it start funding?
r = c.execute("""SELECT first_seen, last_seen, total_contracts_deployed,
                 mainnet_first_tx, json_extract(funding_trail, '$.funder') AS upstream
              FROM deployers WHERE LOWER(deployer_address) = ?""", (TARGET.lower(),)).fetchone()
if r:
    for k in r.keys():
        print(f"  {k}: {r[k]}")

# Funding cadence to downstream deployers
print()
print("  Downstream deployer first_seen distribution (when 0xb0b0b690 funded each):")
for r in c.execute("""SELECT substr(first_seen, 1, 10) AS d, COUNT(*) AS n
    FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
    GROUP BY d ORDER BY d""", (TARGET.lower(),)):
    bar = "#" * min(r['n'] // 50, 60)
    print(f"    {r['d']}  {r['n']:>5} {bar}")

c.close()
