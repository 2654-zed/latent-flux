"""Investigate the three unknown May-9 super-drainers:
- 0x1d81aff2a24c822d715ec09a0f81801face6e6fd  (3,228 drains)
- 0xa9f65861c9bf68497bce6f30c5b20d0ed64d216e  (1,618 drains)
- 0x0e2224685fe775b471b457c643913e4bbd66c8d2  (1,359 drains)

Combined 6,205 victims on one day. None on watchlist. Largest unattributed
single-day drain event in the corpus.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def hdr(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


DRAINERS = [
    "0x1d81aff2a24c822d715ec09a0f81801face6e6fd",
    "0xa9f65861c9bf68497bce6f30c5b20d0ed64d216e",
    "0x0e2224685fe775b471b457c643913e4bbd66c8d2",
]


for drainer in DRAINERS:
    hdr(f"DRAINER {drainer}")

    # All drain activity for this caller
    print("  --- daily drain activity (any time) ---")
    for r in c.execute(
        """SELECT substr(drain_timestamp, 1, 10), COUNT(*)
           FROM approval_watchlist
           WHERE drain_caller=? GROUP BY 1 ORDER BY 1""",
        (drainer,)
    ):
        print(f"    {r[0]}  {r[1]:>5}")

    # Which contracts is it draining from?
    print()
    print("  --- top contracts drained ---")
    for r in c.execute(
        """SELECT contract_address, COUNT(*),
                  COUNT(DISTINCT victim_address)
           FROM approval_watchlist
           WHERE drain_caller=?
           GROUP BY contract_address
           ORDER BY 2 DESC LIMIT 5""",
        (drainer,)
    ):
        ctr = c.execute(
            "SELECT chain, confidence_tier, deployer_address FROM contracts WHERE contract_address=?",
            (r[0],)
        ).fetchone()
        chain = ctr[0] if ctr else "?"
        tier = ctr[1] if ctr else "?"
        dep = ctr[2] if ctr else "?"
        print(f"    {r[0]}  drains={r[1]:>5}  vics={r[2]:>5}  chain={chain:9s}  tier={tier:9s}")
        print(f"      deployer={dep}")

    # Is the drainer itself a deployer?
    print()
    print("  --- this address as deployer ---")
    rows = list(c.execute(
        "SELECT COUNT(*), MIN(detection_timestamp), MAX(detection_timestamp) FROM contracts WHERE deployer_address=?",
        (drainer,)
    ))
    if rows[0][0]:
        print(f"    deployed {rows[0][0]} contracts; first={rows[0][1]}  last={rows[0][2]}")
    else:
        print("    deployed 0 contracts")

    # Does this address exist in deployers table at all?
    dep_row = c.execute(
        """SELECT first_seen, chain, total_contracts_deployed, behavioral_score, entity_type, funding_sources, mainnet_first_tx
           FROM deployers WHERE deployer_address=?""",
        (drainer,)
    ).fetchone()
    if dep_row:
        print(f"    in deployers table: first_seen={dep_row[0]} chain={dep_row[1]} contracts={dep_row[2]}")
        print(f"      entity_type={dep_row[4]}  score={dep_row[3]}  mainnet_first_tx={dep_row[6]}")
        print(f"      funding_sources={(dep_row[5] or '')[:200]}")
    else:
        print("    NOT in deployers table (drainer not previously known as deployer)")

    # Is it in entity_classification or watchlist?
    in_ec = c.execute("SELECT subtype FROM entity_classification WHERE address=?", (drainer,)).fetchone()
    print(f"    entity_classification: {in_ec[0] if in_ec else '(none)'}")
    in_wl = c.execute("SELECT entity_name, priority, watch_reason FROM watchlist WHERE address=?", (drainer,)).fetchone()
    print(f"    watchlist: {in_wl if in_wl else '(none)'}")

    # OLI labels?
    in_oli = c.execute("SELECT severity, primary_tag_name, primary_entity FROM oli_labels WHERE address=?", (drainer,)).fetchone()
    print(f"    oli_labels: {in_oli if in_oli else '(none)'}")

    # bot_candidates?
    in_bots = c.execute("SELECT * FROM bot_candidates WHERE address=?", (drainer,)).fetchone()
    print(f"    bot_candidates: {'YES' if in_bots else 'no'}")


# Cross-drainer overlap analysis
hdr("VICTIM OVERLAP between the three drainers")
victim_sets = {}
for d in DRAINERS:
    s = set(r[0] for r in c.execute(
        "SELECT DISTINCT victim_address FROM approval_watchlist WHERE drain_caller=?",
        (d,)
    ))
    victim_sets[d] = s
    print(f"  {d}: {len(s):,} distinct victims")

print()
print("  pairwise overlap:")
import itertools
for a, b in itertools.combinations(DRAINERS, 2):
    overlap = victim_sets[a] & victim_sets[b]
    print(f"    {a[:14]} & {b[:14]} = {len(overlap)}")

triple = victim_sets[DRAINERS[0]] & victim_sets[DRAINERS[1]] & victim_sets[DRAINERS[2]]
print(f"  triple overlap: {len(triple)}")


# All contracts the three drained — are they from one operator?
hdr("CONTRACTS drained by the three (deployer overlap?)")
all_drained_contracts = set()
for d in DRAINERS:
    for r in c.execute(
        "SELECT DISTINCT contract_address FROM approval_watchlist WHERE drain_caller=?",
        (d,)
    ):
        all_drained_contracts.add(r[0])
print(f"  Distinct contracts drained by any of the 3: {len(all_drained_contracts)}")

# Deployers of those contracts
deployer_counts = {}
for ctr in all_drained_contracts:
    row = c.execute("SELECT deployer_address FROM contracts WHERE contract_address=?", (ctr,)).fetchone()
    if row and row[0]:
        deployer_counts[row[0]] = deployer_counts.get(row[0], 0) + 1
print(f"  Distinct deployers of those contracts: {len(deployer_counts)}")
print("  Top 10 deployers:")
for d, n in sorted(deployer_counts.items(), key=lambda x: -x[1])[:10]:
    in_wl = c.execute("SELECT entity_name, priority FROM watchlist WHERE address=?", (d,)).fetchone()
    wl = f"{in_wl[0]} ({in_wl[1]})" if in_wl else "(not on watchlist)"
    print(f"    {d}  contracts={n:>3}  {wl}")


# Who funds the drainers? Look at transaction_events
hdr("Where did the three drainers come from? (transaction_events to their addresses)")
for d in DRAINERS:
    print(f"  --- {d} ---")
    # First inbound tx
    first_in = c.execute(
        """SELECT from_address, value, timestamp, chain FROM transaction_events
           WHERE to_address=? AND event_type='transfer'
           ORDER BY timestamp ASC LIMIT 3""",
        (d,)
    ).fetchall()
    if first_in:
        for fr in first_in:
            print(f"    earliest inbound: from={fr[0]}  value={fr[1]}  at={fr[2]}  chain={fr[3]}")
    else:
        # try without event_type filter
        any_in = c.execute(
            "SELECT from_address, value, timestamp, chain FROM transaction_events WHERE to_address=? ORDER BY timestamp ASC LIMIT 3",
            (d,)
        ).fetchall()
        for fr in any_in:
            print(f"    earliest inbound: from={fr[0]}  value={fr[1]}  at={fr[2]}  chain={fr[3]}")
        if not any_in:
            print("    (no inbound transaction_events found)")
