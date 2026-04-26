"""Probe of 0x00169219376146760298417404949075285cab72 — novel high-confirmation-rate operator
surfaced 2026-04-26. 5 trap events in 24h, fleet 22, 5 confirmed, mainnet 2024-09-03.

Phase 4 framing: Novel — no prior documentation in INDEX.md, cases/, reports/.
"""
import sqlite3
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
ADDR = "0x00169219376146760298417404949075285cab72"

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

h("1. Deployer profile")
d = c.execute("""
    SELECT deployer_address, chain, first_seen, last_seen,
           total_contracts_deployed, mainnet_first_tx, entity_type,
           json_extract(funding_trail, '$.funder') AS funder,
           funding_trail
    FROM deployers WHERE LOWER(deployer_address) = ?
""", (ADDR.lower(),)).fetchone()
if d:
    for k in d.keys():
        v = d[k]
        if k == "funding_trail" and v and len(v) > 200:
            v = v[:200] + "..."
        print(f"  {k}: {v}")

h("2. Watchlist + org/cluster classification")
w = c.execute("SELECT priority, entity_name, watch_reason FROM watchlist WHERE LOWER(address) = ?", (ADDR.lower(),)).fetchone()
print(f"  watchlist: {dict(w) if w else 'NOT ON WATCHLIST'}")
ow = c.execute("SELECT org_id, role FROM org_wallets WHERE LOWER(address) = ?", (ADDR.lower(),)).fetchone()
print(f"  org_wallets: {dict(ow) if ow else 'no org membership'}")
oc = c.execute("SELECT candidate_id, cluster_size, shared_funding_source FROM org_candidates WHERE deployer_addresses LIKE '%' || ? || '%'", (ADDR.lower(),)).fetchone()
print(f"  org_candidates: {dict(oc) if oc else 'no candidate cluster'}")
try:
    s = c.execute("SELECT classification, fleet_size, confirmed_count, status FROM solo_operator_candidates WHERE LOWER(deployer_address) = ?", (ADDR.lower(),)).fetchone()
    print(f"  solo_operator_candidates: {dict(s) if s else 'not in solo'}")
except sqlite3.OperationalError: pass
try:
    p = c.execute("SELECT * FROM pristine_solo_candidates WHERE LOWER(deployer_address) = ?", (ADDR.lower(),)).fetchone()
    print(f"  pristine_solo_candidates: {dict(p) if p else 'not in pristine_solo'}")
except sqlite3.OperationalError: pass
try:
    iso = c.execute("SELECT * FROM infrastructure_operator_candidates WHERE LOWER(funder_address) = ?", (ADDR.lower(),)).fetchone()
    print(f"  infrastructure_operator_candidates (as funder): {'yes' if iso else 'no'}")
except sqlite3.OperationalError: pass

h("3. Fleet tier breakdown")
for r in c.execute("""
    SELECT confidence_tier, COUNT(*) AS n,
           SUM(has_asymmetric_transfer) AS asym,
           SUM(has_conditional_revert) AS revert,
           SUM(has_unusual_fee_structure) AS fee
    FROM contracts WHERE LOWER(deployer_address) = ?
    GROUP BY confidence_tier ORDER BY n DESC
""", (ADDR.lower(),)):
    print(f"  {r['confidence_tier'] or '(null)':<12} n={r['n']:<3} asym={r['asym']} revert={r['revert']} fee={r['fee']}")

h("4. All contracts deployed by this operator")
for r in c.execute("""
    SELECT contract_address, chain, confidence_tier, detection_method,
           detection_timestamp, has_asymmetric_transfer, has_conditional_revert,
           has_unusual_fee_structure
    FROM contracts WHERE LOWER(deployer_address) = ?
    ORDER BY detection_timestamp DESC
""", (ADDR.lower(),)):
    flags = []
    if r['has_asymmetric_transfer']: flags.append("asym")
    if r['has_conditional_revert']: flags.append("revert")
    if r['has_unusual_fee_structure']: flags.append("fee")
    print(f"  {r['contract_address']}  chain={r['chain']:<9} tier={r['confidence_tier']:<10} "
          f"deploy={r['detection_timestamp'][:19]}  flags={','.join(flags) or '-'}")

h("5. Trap events on this operator's contracts")
for r in c.execute("""
    SELECT te.timestamp, te.trap_contract_address, te.bot_address, te.failure_signature
    FROM trap_events te
    JOIN contracts ct ON LOWER(ct.contract_address) = LOWER(te.trap_contract_address)
    WHERE LOWER(ct.deployer_address) = ?
    ORDER BY te.timestamp DESC
""", (ADDR.lower(),)):
    sig = (r['failure_signature'] or '')[:40]
    print(f"  {r['timestamp'][:19]}  trap={r['trap_contract_address'][:14]}  bot={r['bot_address'][:14]}  sig={sig}")

h("6. Fleet aggregate traffic")
# Pre-fetch contracts for fast aggregate
contracts = [r[0].lower() for r in c.execute("SELECT contract_address FROM contracts WHERE LOWER(deployer_address) = ?", (ADDR.lower(),))]
if contracts:
    c.execute("CREATE TEMP TABLE fleet_addrs(addr TEXT PRIMARY KEY)")
    c.executemany("INSERT INTO fleet_addrs VALUES (?)", [(a,) for a in contracts])
    agg = c.execute("""
        SELECT COUNT(*) AS n, COUNT(DISTINCT contract_address) AS contracts_with_traffic,
               COUNT(DISTINCT interacting_address) AS distinct_senders,
               SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS reverts,
               MIN(timestamp) AS first_tx, MAX(timestamp) AS last_tx
        FROM transaction_events
        WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
    """).fetchone()
    print(f"  total tx events on fleet: {agg['n']:,}")
    print(f"  contracts with traffic: {agg['contracts_with_traffic']}/{len(contracts)}")
    print(f"  distinct senders: {agg['distinct_senders']:,}")
    print(f"  reverts: {agg['reverts']}")
    if agg['n']:
        print(f"  fleet revert_rate: {agg['reverts']/agg['n']:.3f}")
    print(f"  first_tx: {agg['first_tx']}")
    print(f"  last_tx: {agg['last_tx']}")

    # Top selectors
    print("\n  Top selectors hit on fleet:")
    for r in c.execute("""
        SELECT function_selector, COUNT(*) AS n,
               SUM(CASE WHEN is_reverted THEN 1 ELSE 0 END) AS rev
        FROM transaction_events
        WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
          AND function_selector IS NOT NULL AND function_selector != ''
        GROUP BY function_selector ORDER BY n DESC LIMIT 8
    """):
        rate = r['rev']/r['n'] if r['n'] else 0
        print(f"    {r['function_selector']}  total={r['n']:<5} reverts={r['rev']:<5} rate={rate:.3f}")

h("7. Bytecode family memberships")
fams = list(c.execute("""
    SELECT DISTINCT bf.family_id, bf.family_name, bf.member_count, bf.is_cross_deployer,
           COUNT(*) AS in_fleet
    FROM bytecode_family_members bfm
    JOIN bytecode_families bf ON bf.family_id = bfm.family_id
    WHERE LOWER(bfm.contract_address) IN (SELECT addr FROM fleet_addrs)
    GROUP BY bf.family_id ORDER BY in_fleet DESC LIMIT 5
""") if contracts else [])
for r in fams:
    print(f"  {r['family_id']}  members={r['member_count']:<5} cross={r['is_cross_deployer']}  in_fleet={r['in_fleet']}")
if not fams:
    print("  no bytecode family membership")

h("8. Approval-watchlist activity on fleet")
ap = c.execute("""
    SELECT COUNT(*) AS approvals, COUNT(DISTINCT victim_address) AS victims,
           SUM(drain_detected) AS drains
    FROM approval_watchlist
    WHERE LOWER(contract_address) IN (SELECT addr FROM fleet_addrs)
""").fetchone() if contracts else None
if ap:
    print(f"  approvals: {ap['approvals']}  victims: {ap['victims']}  drains: {ap['drains']}")

h("9. Funder downstream — does the same funder fund other corpus deployers?")
funder = d['funder'] if d else None
if funder:
    n = c.execute("SELECT COUNT(*) FROM deployers WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?", (funder.lower(),)).fetchone()[0]
    print(f"  funder {funder} also funds {n} corpus deployers")
    print("  Top 10 by fleet size:")
    for r in c.execute("""
        SELECT deployer_address, total_contracts_deployed, first_seen,
               (SELECT COUNT(*) FROM contracts WHERE deployer_address=d.deployer_address AND confidence_tier='confirmed') AS conf
        FROM deployers d WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
        ORDER BY total_contracts_deployed DESC LIMIT 10
    """, (funder.lower(),)):
        print(f"    {r['deployer_address']}  fleet={r['total_contracts_deployed']:<5} confirmed={r['conf']:<3} first={r['first_seen'][:10]}")

c.close()
