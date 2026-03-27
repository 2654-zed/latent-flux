"""
Layer 3 — Vanity Address Detection

Detects hex-readable vanity prefixes in addresses. Vanity addresses are
a behavioral signal: the operator invested compute to generate a
memorable address, indicating sophistication and ego.

Used by:
- db.ensure_deployer() — tags deployers on first sight
- selector_monitor — tags callers
- deployer_profiler — clustering signal

Zero cost: pure string matching, no API calls.
"""

import re
import sqlite3
from datetime import datetime, timezone

# Hex-speakable words (4+ chars, case-insensitive in hex)
# These are patterns where the hex characters spell readable English
VANITY_PREFIXES = {
    "c0ffee": "COFFEE",
    "c0de":   "CODE",
    "deadbe": "DEAD_BEEF",
    "dead":   "DEAD",
    "beef":   "BEEF",
    "feed":   "FEED",
    "face":   "FACE",
    "babe":   "BABE",
    "f00d":   "FOOD",
    "cafe":   "CAFE",
    "decaf":  "DECAF",
    "fade":   "FADE",
    "aced":   "ACED",
    "b00b":   "BOOB",
    "defi":   "DEFI",
    "d00d":   "DOOD",
    "b0b":    "BOB",
    "ace":    "ACE",
    "bad":    "BAD",
    "bed":    "BED",
    "cab":    "CAB",
    "dab":    "DAB",
    "fab":    "FAB",
}

# Numeric/cultural patterns
VANITY_NUMERIC = {
    "1337":   "LEET",
    "6969":   "SIXTY_NINE",
    "4200":   "FOUR_TWENTY",
    "420":    "FOUR_TWENTY",
    "0420":   "FOUR_TWENTY",
}

# Structural patterns (compute flex)
ZERO_PREFIX_MIN = 6  # 0x000000... = 6+ zeros after 0x


def detect_vanity(address: str) -> tuple[str, str] | None:
    """
    Check if an address has a vanity prefix.

    Returns:
        (pattern_type, label) or None

    pattern_type: 'hex_word', 'numeric', 'zero_flex', 'repeating'
    label: human-readable description
    """
    if not address:
        return None

    addr = address.lower().replace("0x", "")

    # 1. Hex-speakable words (longest match first)
    for prefix, label in sorted(VANITY_PREFIXES.items(), key=lambda x: -len(x[0])):
        if addr.startswith(prefix):
            return ("hex_word", label)

    # 2. Numeric/cultural
    for prefix, label in VANITY_NUMERIC.items():
        if addr.startswith(prefix):
            return ("numeric", label)

    # 3. Zero flex (leading zeros = compute prestige)
    leading_zeros = len(addr) - len(addr.lstrip("0"))
    if leading_zeros >= ZERO_PREFIX_MIN:
        return ("zero_flex", f"ZERO_x{leading_zeros}")

    # 4. Repeating characters (0xaaaa, 0xffff, etc.)
    if len(addr) >= 4 and addr[0] == addr[1] == addr[2] == addr[3]:
        return ("repeating", f"REPEAT_{addr[0].upper()}")

    return None


def tag_address(conn: sqlite3.Connection, address: str,
                address_type: str = "deployer") -> dict | None:
    """
    Detect vanity and store in vanity_tags table.

    Args:
        conn: database connection
        address: the address to check
        address_type: 'deployer', 'caller', 'funder'

    Returns:
        tag dict if vanity detected, None otherwise
    """
    result = detect_vanity(address)
    if not result:
        return None

    pattern_type, label = result

    conn.execute("""
        INSERT OR IGNORE INTO vanity_tags
        (address, address_type, pattern_type, label, detected_at)
        VALUES (?, ?, ?, ?, ?)
    """, (address.lower(), address_type, pattern_type, label,
          datetime.now(timezone.utc).isoformat()))

    return {
        "address": address,
        "pattern_type": pattern_type,
        "label": label,
    }


def ensure_table(conn: sqlite3.Connection):
    """Create vanity_tags table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vanity_tags (
            address TEXT PRIMARY KEY,
            address_type TEXT,
            pattern_type TEXT,
            label TEXT,
            detected_at TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_vanity_label ON vanity_tags(label)
    """)
    conn.commit()


def scan_existing(conn: sqlite3.Connection) -> dict:
    """Scan all existing deployers and callers for vanity prefixes."""
    ensure_table(conn)
    tagged = {"deployers": 0, "callers": 0}

    # Deployers
    for r in conn.execute("SELECT deployer_address FROM deployers"):
        if tag_address(conn, r[0], "deployer"):
            tagged["deployers"] += 1

    # Callers (distinct interacting addresses)
    for r in conn.execute("SELECT DISTINCT interacting_address FROM transaction_events"):
        if tag_address(conn, r[0], "caller"):
            tagged["callers"] += 1

    conn.commit()

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM vanity_tags").fetchone()[0]
    print(f"[vanity_detector] Scanned existing corpus. Tagged: {tagged['deployers']} deployers, {tagged['callers']} callers. Total: {total}")

    for r in conn.execute("""
        SELECT label, pattern_type, COUNT(*) as c, GROUP_CONCAT(DISTINCT address_type) as types
        FROM vanity_tags GROUP BY label ORDER BY c DESC
    """):
        print(f"  {r['label']:15} | {r['pattern_type']:12} | {r['c']:>4} addresses | {r['types']}")

    return tagged


# CLI
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Vanity address detection")
    parser.add_argument("--scan", action="store_true", help="Scan existing corpus")
    parser.add_argument("--check", type=str, help="Check a single address")
    args = parser.parse_args()

    if args.check:
        result = detect_vanity(args.check)
        if result:
            print(f"{args.check}: {result[0]} — {result[1]}")
        else:
            print(f"{args.check}: no vanity pattern")
    elif args.scan:
        db_path = Path(__file__).resolve().parent / "data" / "surveillance.db"
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        scan_existing(conn)
        conn.close()
    else:
        parser.print_help()
