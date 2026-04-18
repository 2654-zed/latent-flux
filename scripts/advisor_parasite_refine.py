"""Refine the advisor-parasite candidate list by removing legitimate
DeFi infrastructure and surfacing the unknowns."""
import sqlite3, json

DB = "/app/surveillance/data/surveillance.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# Candidate spenders from the first scan
CANDIDATES = [
    "0x57df6092665eb6058de53939612413ff4b09114e",
    "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24",
    "0xccc88a9d1b4ed6b0eaba998850414b24f1c315be",
    "0x111111125421ca6dc452d289314280a0f8842a65",
    "0x0000000000001ff3684f28c67538d4d072c22734",
    "0xd8ba9d1a99fc21f0eca24e9b85737c28a194a4e2",
    "0x9dda6ef3d919c9bc8885d5560999a3640431e8e6",
    "0x6131b5fae19ea4f9d964eac0408e4408b66337b5",
    "0xec3576c579cc93d1ab40ee93a3fcf733435706cd",
    "0xac4c6e212a361c968f1725b4d055b47e63f80b75",
    "0x91a65ef694ab6341a2ef0e2ddbd312893c04168c",
    "0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae",
    "0x337685fdab40d39bd02028545a4ffa7d287cc3e2",
    "0xb300000b72deaeb607a12d5f54773d1c19c7028d",
    "0x07964f135f276412b3182a3b2407b8dd45000000",
    "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",
]

# Known legitimate DeFi infra by address — adding the clearly-identified ones
KNOWN_LEGIT = {
    "0x111111125421ca6dc452d289314280a0f8842a65": "1inch Router v6",
    "0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae": "LI.FI Diamond (bridge aggregator)",
    # Vanity-prefix style that suggests public infra:
    "0x0000000000001ff3684f28c67538d4d072c22734": "(vanity prefix — likely Uniswap-family or similar)",
    "0x07964f135f276412b3182a3b2407b8dd45000000": "(trailing zeros vanity — likely protocol contract)",
    "0xb300000b72deaeb607a12d5f54773d1c19c7028d": "(vanity prefix)",
}

print(f"{'spender':44}  {'in_contracts?':14}  {'tier':10}  {'in_infra_reg?':14}  {'selector_patterns':40}  {'hint'}")
print("-" * 180)
for addr in CANDIDATES:
    # Is it in our contracts table? (we'd know its bytecode)
    row = c.execute(
        "SELECT confidence_tier, chain, detection_method FROM contracts "
        "WHERE LOWER(contract_address) = ?",
        (addr,),
    ).fetchone()
    in_contracts = row is not None
    tier = row[0] if row else "-"

    # Is it in infrastructure_registry? (known-legit)
    in_reg = c.execute(
        "SELECT COUNT(*) FROM infrastructure_registry WHERE LOWER(address) = ?",
        (addr,),
    ).fetchone()[0] > 0

    # What selectors are being called on this address as a contract?
    sels = c.execute(
        """SELECT function_selector, COUNT(*) FROM transaction_events
           WHERE LOWER(contract_address) = ? GROUP BY function_selector
           ORDER BY 2 DESC LIMIT 3""",
        (addr,),
    ).fetchall()
    sel_str = ", ".join(f"{s[0]}:{s[1]}" for s in sels) if sels else "(no tx data)"

    hint = KNOWN_LEGIT.get(addr, "UNKNOWN — candidate")
    print(f"{addr}  {str(in_contracts):14}  {tier:10}  "
          f"{str(in_reg):14}  {sel_str[:38]:40}  {hint}")

# Deep-dive on the top unknown: check approvers' behavior
print()
print("=== deep-dive on top unknowns (not in KNOWN_LEGIT) ===")
UNKNOWN = [a for a in CANDIDATES if a not in KNOWN_LEGIT]
for addr in UNKNOWN[:5]:
    print(f"\n--- {addr} ---")
    # Approver list with per-approver approval count
    rows = c.execute(
        """SELECT LOWER(approver) as a, COUNT(*) as n,
                  MIN(timestamp) as first, MAX(timestamp) as last
           FROM approval_events WHERE LOWER(spender) = ?
           GROUP BY a ORDER BY n DESC LIMIT 5""",
        (addr,),
    ).fetchall()
    print(f"  top approvers (by approval count to this spender):")
    for r in rows:
        print(f"    {r['a']}  approvals={r['n']}  first={r['first'][:10]}  last={r['last'][:10]}")

    # Chain distribution
    chains = c.execute(
        "SELECT chain, COUNT(*) FROM approval_events WHERE LOWER(spender) = ? GROUP BY chain",
        (addr,),
    ).fetchall()
    print(f"  chains: {[(r[0], r[1]) for r in chains]}")

    # Is it a contract we deployed? (if yes we might have bytecode info)
    row = c.execute(
        "SELECT confidence_tier, bytecode_pattern_notes FROM contracts WHERE LOWER(contract_address) = ?",
        (addr,),
    ).fetchone()
    if row:
        notes = (row[1] or "")[:150]
        print(f"  contracts row: tier={row[0]}  notes={notes}")
    else:
        print(f"  contracts row: NOT PRESENT (pre-corpus-start or never deployed a contract in our window)")
