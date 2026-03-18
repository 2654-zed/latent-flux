"""Extract Arbitrum exploit addresses from DeFiHackLabs and seed our DB."""
import re
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surveillance import db

EXPLOIT_FILES = [
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2025-01/Paribus_exp.sol",
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2025-06/Stead_exp.sol",
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2025-07/gmx_exp.sol",
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2025-08/Bebop_dex_exp.sol",
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2025-08/EverValueCoin_exp.sol",
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2025-10/SharwaFinance_exp.sol",
    "C:/Users/jason/Desktop/DeFiHackLabs/src/test/2026-01/futureswap_exp.sol",
]

# Known infrastructure — skip these
INFRA = {
    "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
    "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
    "0x625e7708f30ca75bfd92586e17077590c60eb4cd",
    "0x0000000000000000000000000000000000000000",
}


def extract_from_file(filepath):
    """Extract categorized addresses from an exploit PoC file."""
    filename = os.path.basename(filepath)
    folder = filepath.split("/test/")[-1].split("/")[0]

    try:
        content = open(filepath, encoding="utf-8", errors="replace").read()
    except Exception:
        return []

    lines = content.split("\n")
    header = [l for l in lines[:35] if l.startswith("//")]

    # Extract total lost
    lost = ""
    for line in header:
        if "Total Lost" in line:
            lost = line.split(":", 1)[-1].strip()

    entries = []
    for line in lines[:35]:
        line_lower = line.lower()
        addrs = re.findall(r"0x[0-9a-fA-F]{40}", line)
        for addr in addrs:
            addr_l = addr.lower()
            if addr_l in INFRA:
                continue
            role = "unknown"
            if any(k in line_lower for k in ("attack", "mev", "exploiter")):
                role = "attacker"
            elif any(k in line_lower for k in ("vulnerable", "victim", "implementation")):
                role = "vulnerable_contract"
            elif "proxy" in line_lower:
                role = "vulnerable_proxy"

            entries.append({
                "address": addr_l,
                "role": role,
                "source": filename,
                "period": folder,
                "lost": lost,
            })

    return entries


def seed_database(entries):
    """Insert DeFiHackLabs entries into our contracts/deployers tables."""
    conn = db.init_db()
    now = datetime.now(timezone.utc).isoformat()
    added = 0

    for entry in entries:
        addr = entry["address"]
        role = entry["role"]
        source = entry["source"]
        lost = entry["lost"]
        period = entry["period"]

        reason = (
            f"DeFiHackLabs ground truth ({period}/{source}). "
            f"Role: {role}. Lost: {lost}"
        )

        if role in ("vulnerable_contract", "vulnerable_proxy"):
            # Add as confirmed contract
            existing = db.get_contract(conn, addr)
            if existing:
                # Already in DB — just update confidence if not already confirmed
                if existing["confidence_tier"] != "confirmed":
                    db.update_contract_confidence(conn, addr, "confirmed", reason)
                    print(f"  UPGRADED: {addr} -> confirmed ({source})")
                    added += 1
                else:
                    print(f"  ALREADY CONFIRMED: {addr}")
            else:
                # Not in our DB yet — add to deployers as ground truth
                try:
                    conn.execute(
                        """INSERT INTO deployers (deployer_address, chain, first_seen, last_seen,
                               total_contracts_deployed, deployment_pattern_notes, entity_type)
                           VALUES (?, 'arbitrum', ?, ?, 0, ?, 'ground_truth')
                           ON CONFLICT(deployer_address) DO UPDATE SET
                               deployment_pattern_notes = ?,
                               entity_type = 'ground_truth'""",
                        (addr, now, now, reason, reason),
                    )
                    conn.commit()
                    print(f"  ADDED (deployers): {addr} as ground_truth ({source})")
                    added += 1
                except Exception as e:
                    print(f"  ERROR: {addr}: {e}")

        elif role == "attacker":
            # Add attacker to deployers table
            try:
                conn.execute(
                    """INSERT INTO deployers (deployer_address, chain, first_seen, last_seen,
                           total_contracts_deployed, deployment_pattern_notes, entity_type)
                       VALUES (?, 'arbitrum', ?, ?, 0, ?, 'known_attacker')
                       ON CONFLICT(deployer_address) DO UPDATE SET
                           deployment_pattern_notes = COALESCE(deployment_pattern_notes, '') || '; ' || ?,
                           entity_type = 'known_attacker'""",
                    (addr, now, now, reason, reason),
                )
                conn.commit()
                print(f"  ADDED (attacker): {addr} ({source})")
                added += 1
            except Exception as e:
                print(f"  ERROR: {addr}: {e}")

    conn.close()
    return added


def main():
    print("=" * 60)
    print("DeFiHackLabs Arbitrum Seed Import")
    print("=" * 60)
    print()

    all_entries = []
    for filepath in EXPLOIT_FILES:
        if not os.path.exists(filepath):
            print(f"SKIP: {filepath} not found")
            continue
        entries = extract_from_file(filepath)
        filename = os.path.basename(filepath)
        folder = filepath.split("/test/")[-1].split("/")[0]
        print(f"{folder}/{filename}: {len(entries)} addresses")
        for e in entries:
            print(f"  {e['role']:20s} {e['address']}")
        all_entries.extend(entries)
        print()

    # Deduplicate
    seen = set()
    unique = []
    for e in all_entries:
        if e["address"] not in seen:
            seen.add(e["address"])
            unique.append(e)

    print(f"Total unique addresses: {len(unique)}")
    print(f"  Attackers:    {sum(1 for e in unique if e['role'] == 'attacker')}")
    print(f"  Vulnerable:   {sum(1 for e in unique if 'vulnerable' in e['role'])}")
    print(f"  Other:        {sum(1 for e in unique if e['role'] == 'unknown')}")
    print()

    print("Seeding database...")
    added = seed_database(unique)
    print(f"\nTotal added/updated: {added}")


if __name__ == "__main__":
    main()
