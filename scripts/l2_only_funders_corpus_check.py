"""What does the corpus already know about the 4 L2-only funders?

These have no mainnet history. Before we use RPC, let's see what's in
our Base/Arb/OP corpus — they may have appeared as recipients in
transaction_events, bridge_events, or org_transfer_events.
"""
import sqlite3
from pathlib import Path

DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")

L2_ONLY = [
    "0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07",
    "0x39591e7c099a379fd7b349ebfecaeef439c40454",
    "0x8ca702323c341a8d46ee94a2abeddb08798ca10d",
    "0x0e6e91775d24d34b90e0f3d806a90705f0199999",
]

c = sqlite3.connect(str(DB), timeout=60)
c.row_factory = sqlite3.Row

def h(s): print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")

for funder in L2_ONLY:
    h(f"Funder: {funder}")
    f = funder.lower()

    # Are they themselves a deployer?
    d = c.execute("""
        SELECT chain, first_seen, last_seen, total_contracts_deployed,
               json_extract(funding_trail, '$.funder') AS upstream
        FROM deployers WHERE LOWER(deployer_address) = ?
    """, (f,)).fetchone()
    if d:
        print(f"  in deployers: chain={d['chain']} first={d['first_seen']} fleet={d['total_contracts_deployed']} upstream={d['upstream']}")
    else:
        print(f"  not in deployers (haven't deployed contracts)")

    # In bridge_events as sender or recipient?
    n_bridge = c.execute(
        "SELECT COUNT(*) FROM bridge_events WHERE LOWER(sender) = ? OR LOWER(decoded_l1_recipient) = ?",
        (f, f)).fetchone()[0]
    print(f"  bridge_events involving address: {n_bridge}")
    if n_bridge:
        for r in c.execute("""
            SELECT chain, timestamp, sender, decoded_l1_recipient, value_eth, bridge_name, function_name
            FROM bridge_events WHERE LOWER(sender) = ? OR LOWER(decoded_l1_recipient) = ?
            ORDER BY timestamp DESC LIMIT 5
        """, (f, f)):
            print(f"    {r['timestamp'][:19]} chain={r['chain']} bridge={r['bridge_name']} fn={r['function_name']} value={r['value_eth']}")

    # In transaction_events as interacting (sending) or contract (receiving)?
    n_tx_send = c.execute(
        "SELECT COUNT(*) FROM transaction_events WHERE LOWER(interacting_address) = ?",
        (f,)).fetchone()[0]
    print(f"  transaction_events as sender: {n_tx_send}")

    # In org_transfer_events?
    n_org = c.execute(
        "SELECT COUNT(*) FROM org_transfer_events WHERE LOWER(from_address) = ? OR LOWER(to_address) = ?",
        (f, f)).fetchone()[0]
    print(f"  org_transfer_events involving address: {n_org}")
    if n_org:
        for r in c.execute("""
            SELECT chain, timestamp, from_address, to_address, value_eth, tx_hash
            FROM org_transfer_events WHERE LOWER(from_address) = ? OR LOWER(to_address) = ?
            ORDER BY timestamp DESC LIMIT 5
        """, (f, f)):
            print(f"    {r['timestamp'][:19]} chain={r['chain']}  {r['from_address'][:12]} -> {r['to_address'][:12]}  {r['value_eth']} ETH")

    # Also check if they're in alerts as recipients/subjects
    n_alert = c.execute(
        "SELECT COUNT(*) FROM alerts WHERE LOWER(address) = ?",
        (f,)).fetchone()[0]
    print(f"  alerts on this address: {n_alert}")
    if n_alert:
        for r in c.execute("""
            SELECT alert_type, timestamp, substr(payload,1,80) FROM alerts
            WHERE LOWER(address) = ? ORDER BY timestamp DESC LIMIT 5
        """, (f,)):
            print(f"    {r[0]:<30} {r[1][:19]}  {r[2]}")

    # First chain we saw them on (via the deployers they fund)
    first_act = c.execute("""
        SELECT MIN(first_seen), MIN(chain) FROM deployers
        WHERE LOWER(json_extract(funding_trail, '$.funder')) = ?
    """, (f,)).fetchone()
    print(f"  first downstream deployer activity: {first_act[0]} on {first_act[1]}")

    # Per-chain split of funded deployers
    print("  funded-deployer per-chain split:")
    for r in c.execute("""
        SELECT chain, COUNT(*) FROM deployers
        WHERE LOWER(json_extract(funding_trail, '$.funder')) = ? GROUP BY chain
    """, (f,)):
        print(f"    {r[0]:<10} {r[1]:,}")

c.close()
