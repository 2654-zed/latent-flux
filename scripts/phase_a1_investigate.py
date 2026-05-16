"""Phase A1 investigation (v2 — index-friendly): Apr-23 approval spike vs
Apr-25 deployer spike.

Hypothesis (NEXT_SESSION_PLAN.md): Apr-25 b0b0b690 vanity-funder mass-fund
(8,052 new deployers, P(CP)=1.000) was preceded by approval-side victim
accumulation on Apr-23 (4,329 approvals vs ~1,500 baseline).

Pre-registered prediction: LIKELY — staging pattern.

All addresses are lowercase-canonical in storage (verified). Avoid LOWER()
calls on join keys to keep indexes usable.
"""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def hdr(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def timed(label, fn):
    t0 = time.time()
    r = fn()
    print(f"  [{label}: {time.time()-t0:.1f}s]")
    return r


# ---- Step 1: daily new deployers per chain (uses no index on first_seen
# but partition by chain helps; total rows scanned bounded to ~74k)
hdr("STEP 1: Apr-22..27 daily new-deployer counts per chain")
for chain in ("base", "arbitrum", "optimism"):
    print(f"  --- {chain} ---")
    for r in c.execute(
        """SELECT substr(first_seen,1,10), COUNT(*) FROM deployers
           WHERE chain=? AND first_seen>='2026-04-22' AND first_seen<'2026-04-28'
           GROUP BY 1 ORDER BY 1"""
        , (chain,)):
        print(f"    {r[0]}  {r[1]:>6,}")


# ---- Step 2: daily approval counts in window
hdr("STEP 2: Apr-22..27 daily approval_watchlist counts")
for r in c.execute(
    """SELECT substr(approve_timestamp,1,10), COUNT(*) FROM approval_watchlist
       WHERE approve_timestamp>='2026-04-22' AND approve_timestamp<'2026-04-28'
       GROUP BY 1 ORDER BY 1"""
):
    print(f"  {r[0]}  {r[1]:>6,}")


# ---- Step 3: chain attribution of approvals via contracts.chain
# Strategy: collect distinct contract_address values in date range first
# (uses idx_approval_contract less directly; still a scan but only on the
# slim approve_timestamp filter — still slow without index, but bounded)
hdr("STEP 3: chain attribution of Apr-22..27 approvals")
print("  collecting distinct contract_address in window...")
t0 = time.time()
distinct_contracts = [r[0] for r in c.execute(
    """SELECT DISTINCT contract_address FROM approval_watchlist
       WHERE approve_timestamp>='2026-04-22' AND approve_timestamp<'2026-04-28'"""
)]
print(f"  found {len(distinct_contracts):,} distinct contracts in window  [{time.time()-t0:.1f}s]")

# Look up each contract's chain (uses contracts PK)
chain_counts_by_date: dict[tuple[str, str], int] = {}
chain_of_contract: dict[str, str] = {}
t0 = time.time()
# Batch lookup chains
BATCH = 500
for i in range(0, len(distinct_contracts), BATCH):
    batch = distinct_contracts[i:i+BATCH]
    placeholders = ",".join(["?"] * len(batch))
    for row in c.execute(
        f"SELECT contract_address, chain FROM contracts WHERE contract_address IN ({placeholders})",
        batch
    ):
        chain_of_contract[row[0]] = row[1]
print(f"  chain-lookups: {len(chain_of_contract):,} resolved of {len(distinct_contracts):,}  [{time.time()-t0:.1f}s]")

# Now aggregate approvals by date+chain
t0 = time.time()
for r in c.execute(
    """SELECT contract_address, substr(approve_timestamp,1,10), COUNT(*)
       FROM approval_watchlist
       WHERE approve_timestamp>='2026-04-22' AND approve_timestamp<'2026-04-28'
       GROUP BY contract_address, substr(approve_timestamp,1,10)"""
):
    contract, date, n = r
    chain = chain_of_contract.get(contract, "unknown")
    chain_counts_by_date[(date, chain)] = chain_counts_by_date.get((date, chain), 0) + n
print(f"  aggregated  [{time.time()-t0:.1f}s]")
print(f"  {'date':12s} {'chain':10s} {'count':>8s}")
for (d, ch), n in sorted(chain_counts_by_date.items()):
    print(f"  {d:12s} {ch:10s} {n:>8,}")


# ---- Step 4: search for any funder address starting with 0xb0b0b6
hdr("STEP 4: search for 0xb0b0b6 prefix in deployer-side fields")
for field in ("funding_sources", "known_associated_deployers"):
    n = c.execute(
        f"SELECT COUNT(*) FROM deployers WHERE {field} LIKE '%0xb0b0b6%'"
    ).fetchone()[0]
    print(f"  deployers.{field} matching '%0xb0b0b6%': {n}")
    if 0 < n <= 20:
        for r in c.execute(
            f"SELECT deployer_address, chain, first_seen, {field} FROM deployers WHERE {field} LIKE '%0xb0b0b6%' LIMIT 10"
        ):
            print(f"    dep={r[0]}  chain={r[1]}  first_seen={r[2]}")
            print(f"      {field}={r[3][:200]}")


# ---- Step 5: who funded the Apr-25 Optimism deployer mass?
hdr("STEP 5: Apr-25 Optimism deployer-mass funder analysis")
# Look at deployers.first_seen for chain=optimism on Apr-25 and parse funding_sources
print("  parsing funding_sources for Apr-25 Optimism deployers (this scans ~6,638 rows)...")
t0 = time.time()
import json
funder_counts: dict[str, int] = {}
total = 0
empty = 0
for r in c.execute(
    """SELECT funding_sources FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'"""
):
    total += 1
    src = r[0]
    if not src or src == "[]":
        empty += 1
        continue
    try:
        parsed = json.loads(src)
        if isinstance(parsed, list):
            for entry in parsed:
                if isinstance(entry, dict):
                    f = entry.get("funder") or entry.get("address") or entry.get("from")
                elif isinstance(entry, str):
                    f = entry
                else:
                    f = None
                if f:
                    funder_counts[f.lower()] = funder_counts.get(f.lower(), 0) + 1
        elif isinstance(parsed, dict):
            f = parsed.get("funder") or parsed.get("address")
            if f:
                funder_counts[f.lower()] = funder_counts.get(f.lower(), 0) + 1
    except Exception:
        pass
print(f"  scanned {total:,} rows; {empty:,} empty funding_sources; {len(funder_counts):,} distinct funders  [{time.time()-t0:.1f}s]")
print(f"  Top 15 funders of Apr-25 Optimism deployer mass:")
for funder, n in sorted(funder_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"    {funder}  {n:>6,}")


# ---- Step 6: confirm or refute b0b0b690 was the funder
hdr("STEP 6: b0b0b690 specifically — funding signature in corpus")
b0_match = sorted([(f, n) for f, n in funder_counts.items() if f.startswith("0xb0b0b6")], key=lambda x: -x[1])
print(f"  Apr-25 Optimism deployers funded by 0xb0b0b6...:")
if b0_match:
    for f, n in b0_match:
        print(f"    {f}  funded {n:,} Apr-25 Optimism deployers")
else:
    print("    (none — b0b0b690 NOT the master funder of Apr-25 OP mass)")


# ---- Step 7: chain-overlap of Apr-23 approvals' deployers vs Apr-25 OP deployers
hdr("STEP 7: deployer-overlap Apr-23 approval-side <-> Apr-25 OP deployer mass")
# Collect distinct deployer_address from approval_watchlist on Apr-23
print("  collecting Apr-23 approval-side deployers...")
t0 = time.time()
apr23_app_deployers = {r[0] for r in c.execute(
    """SELECT DISTINCT deployer_address FROM approval_watchlist
       WHERE approve_timestamp>='2026-04-23' AND approve_timestamp<'2026-04-24'
         AND deployer_address IS NOT NULL"""
)}
print(f"  Apr-23 approval-side distinct deployers: {len(apr23_app_deployers):,}  [{time.time()-t0:.1f}s]")

# Collect Apr-25 OP deployer addresses
t0 = time.time()
apr25_op_deployers = {r[0] for r in c.execute(
    """SELECT deployer_address FROM deployers
       WHERE chain='optimism' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'"""
)}
print(f"  Apr-25 Optimism deployers: {len(apr25_op_deployers):,}  [{time.time()-t0:.1f}s]")

overlap = apr23_app_deployers & apr25_op_deployers
print(f"  Overlap: {len(overlap)} deployers appear in both sets")
if overlap and len(overlap) <= 20:
    for d in sorted(overlap):
        print(f"    {d}")


# ---- Step 8: same overlap but with Apr-23 Base-side deployers and Apr-25 Base-side mass
hdr("STEP 8: same-chain Base-side overlap")
t0 = time.time()
apr25_base_deployers = {r[0] for r in c.execute(
    """SELECT deployer_address FROM deployers
       WHERE chain='base' AND first_seen>='2026-04-25' AND first_seen<'2026-04-26'"""
)}
print(f"  Apr-25 Base deployers: {len(apr25_base_deployers):,}  [{time.time()-t0:.1f}s]")
base_overlap = apr23_app_deployers & apr25_base_deployers
print(f"  Overlap with Apr-23 approval-side deployers: {len(base_overlap)}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
