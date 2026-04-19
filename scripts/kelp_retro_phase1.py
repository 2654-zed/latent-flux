"""Kelp retrospective replay — Phase 1.

Corpus presence check for Kelp-related addresses. Zero RPC calls. Local
SQLite only. Probes every table where the address might plausibly appear.

Expected outcome: most Ethereum addresses return zero hits (we don't
monitor Ethereum). The interesting hits are on Arbitrum IF the attacker
operated there post-Ethereum-drain, OR in approval_events if any
monitored-chain user granted approvals to these addresses.
"""
import sqlite3
from collections import defaultdict

DB = "/app/surveillance/data/surveillance.db"

# Key addresses from the Kelp public post-mortem (DK27ss PoC repo + Blockaid)
KEY_ADDRESSES = {
    "kelp_oft_adapter_ethereum": "0x85d456b2dff1fd8245387c0bfb64dfb700e98ef3",
    "kelp_dvn_ethereum":          "0x589dedbd617e0cbcb916a9223f4d1300c294236b",
    "kelp_dvn_unichain":          "0x282b3386571f7f794450d5789911a9804fa346b4",
    "attack_recipient":           "0x8b1b6c9a6db1304000412dd21ae6a70a82d60d3b",
    "eth_endpoint_receive_lib":   "0xc02ab410f0734efa3f14628780e6e695156024c2",
    "unichain_endpoint_send_lib": "0xc39161c743d0307eb9bcc9fef03eeb9dc4802de7",
}

# Normalize to lowercase
ADDRS = {name: addr.lower() for name, addr in KEY_ADDRESSES.items()}


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    results = defaultdict(list)

    for label, addr in ADDRS.items():
        # Table: column pairs to check
        checks = [
            ("contracts", "contract_address"),
            ("contracts", "deployer_address"),
            ("deployers", "deployer_address"),
            ("transaction_events", "contract_address"),
            ("transaction_events", "interacting_address"),
            ("approval_events", "approver"),
            ("approval_events", "spender"),
            ("approval_events", "linked_deployer"),
            ("approval_events", "token_contract"),
            ("entity_classification", "address"),
            ("bytecode_cache", "source_contract"),
            ("bytecode_family_members", "contract_address"),
            ("infrastructure_registry", "address"),
            ("alerts", "address"),
            ("trap_events", "trap_contract_address"),
            ("trap_events", "bot_address"),
        ]

        for table, col in checks:
            try:
                q = f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {table} WHERE LOWER({col}) = ?" \
                    if col != "address" and table in ("entity_classification", "bytecode_family_members", "infrastructure_registry") \
                    else f"SELECT COUNT(*) FROM {table} WHERE LOWER({col}) = ?"
                # Simpler: just count; only some tables have timestamp
                r = c.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE LOWER({col}) = ?",
                    (addr,),
                ).fetchone()
                n = r[0]
                if n > 0:
                    results[label].append((table, col, n))
            except sqlite3.Error:
                # table or column missing; ignore
                pass

    print("=== Kelp retrospective Phase 1: corpus presence check ===\n")
    any_hits = False
    for label, addr in ADDRS.items():
        hits = results.get(label, [])
        if hits:
            any_hits = True
            print(f"HIT  {label}  ({addr})")
            for table, col, n in hits:
                print(f"       {table}.{col}: {n} row(s)")
        else:
            print(f"miss {label}  ({addr})")

    if not any_hits:
        print("\nAll six key addresses absent from the corpus. Consistent with")
        print("the monitoring footprint — we don't ingest Ethereum or Unichain,")
        print("and none of these addresses have interacted with contracts we")
        print("monitor on Arbitrum / Base / Optimism.")
    else:
        print("\nSome key addresses are present. See hits above for follow-up.")

    # Extra: full outbound scan for attack_recipient across ALL monitored chains
    # (the only address the post-mortem suggests MIGHT touch our chains downstream)
    print("\n=== Deeper probe: attack_recipient address across all string columns ===")
    recipient = ADDRS["attack_recipient"]
    # Scan LIKE queries against tables with string columns that might embed the address
    wide_scan = [
        ("alerts", "payload"),
        ("deployers", "funding_trail"),
        ("contracts", "confidence_reason"),
        ("contracts", "bytecode_pattern_notes"),
    ]
    for table, col in wide_scan:
        try:
            r = c.execute(
                f"SELECT COUNT(*) FROM {table} WHERE LOWER({col}) LIKE ?",
                (f"%{recipient}%",),
            ).fetchone()
            if r[0] > 0:
                print(f"  FOUND recipient address string in {table}.{col}: {r[0]} row(s)")
        except sqlite3.Error:
            pass

    # Extra: Kelp OFT adapter string probe
    print("\n=== Deeper probe: Kelp OFT adapter address across all string columns ===")
    adapter = ADDRS["kelp_oft_adapter_ethereum"]
    for table, col in wide_scan:
        try:
            r = c.execute(
                f"SELECT COUNT(*) FROM {table} WHERE LOWER({col}) LIKE ?",
                (f"%{adapter}%",),
            ).fetchone()
            if r[0] > 0:
                print(f"  FOUND adapter address string in {table}.{col}: {r[0]} row(s)")
        except sqlite3.Error:
            pass


if __name__ == "__main__":
    main()
