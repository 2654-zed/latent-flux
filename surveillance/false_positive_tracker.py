"""
Layer 3 — False Positive Tracking & Classifier Precision

Tracks contracts that were flagged as suspected but are likely legitimate.
Measures per-detector precision to identify which patterns over-fire.

Detection methods for false positives:
1. Balanced interaction — contracts with roughly equal buy/sell ratios
2. High unique caller count with low revert — organic traffic pattern
3. GoPlus cross-reference — GoPlus says clean, we say suspected
4. Long-lived with sustained activity — traps burn out, legit tokens persist
5. Manual override — analyst marks as false positive with reason

Usage:
    python -m surveillance.false_positive_tracker --scan
    python -m surveillance.false_positive_tracker --precision
    python -m surveillance.false_positive_tracker --override 0xADDR --reason "legitimate DEX pool"
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS false_positives (
            contract_address TEXT PRIMARY KEY,
            chain TEXT,
            original_tier TEXT,
            original_patterns TEXT,
            fp_reason TEXT,
            fp_method TEXT,
            fp_confidence REAL,
            detector_blamed TEXT,
            assessed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS detector_precision (
            detector_name TEXT NOT NULL,
            date TEXT NOT NULL,
            total_fired INTEGER,
            confirmed_traps INTEGER,
            false_positives INTEGER,
            unknown INTEGER,
            precision REAL,
            UNIQUE(detector_name, date)
        );
    """)


def scan_false_positives(conn: sqlite3.Connection) -> dict:
    """
    Scan suspected contracts for false positive signals.
    Uses heuristics — no API calls.
    """
    ensure_tables(conn)
    now = _now()
    today = now[:10]
    found = 0

    suspected = conn.execute("""
        SELECT c.contract_address, c.chain, c.bytecode_pattern_notes, c.confidence_tier,
               c.detection_timestamp
        FROM contracts c
        LEFT JOIN false_positives fp ON fp.contract_address = c.contract_address
        WHERE c.confidence_tier = 'suspected'
        AND fp.contract_address IS NULL
        AND c.bytecode_pattern_notes IS NOT NULL AND c.bytecode_pattern_notes != ''
    """).fetchall()

    for c in suspected:
        addr = c["contract_address"]
        patterns = c["bytecode_pattern_notes"] or ""

        # Get interaction stats
        stats = conn.execute("""
            SELECT COUNT(*) as hits,
                COUNT(DISTINCT interacting_address) as callers,
                SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts,
                COUNT(DISTINCT function_selector) as selectors
            FROM transaction_events WHERE contract_address=?
        """, (addr,)).fetchone()

        hits = stats["hits"] or 0
        callers = stats["callers"] or 0
        reverts = stats["reverts"] or 0
        selectors = stats["selectors"] or 0

        if hits < 10:
            continue  # Not enough data to assess

        # Guard: do not write FP rows for contracts with any observed harm.
        # The infrastructure-parasite class produces high-traffic + low-revert
        # patterns by design; the sustained_traffic and balanced_interaction
        # heuristics fire on exactly the surface that defines the parasite.
        # If the contract has ever produced a trap_event, that is observed-harm
        # ground truth and should override any heuristic-based FP signal.
        # See Correction #17 (2026-04-25) for the 0xd4624228 cascade.
        traps = conn.execute(
            "SELECT 1 FROM trap_events WHERE LOWER(trap_contract_address) = LOWER(?) LIMIT 1",
            (addr,),
        ).fetchone()
        if traps:
            continue

        revert_rate = reverts / hits if hits > 0 else 0
        fp_reason = None
        fp_method = None
        fp_confidence = 0.0
        detector_blamed = None

        # Heuristic 1: Balanced interaction pattern
        # Legitimate tokens have diverse selectors (transfer, approve, balanceOf, etc.)
        # Traps tend to have 1-2 selectors dominating
        if selectors >= 5 and callers >= 50 and revert_rate < 0.05:
            fp_reason = f"Diverse interface ({selectors} selectors), high organic traffic ({callers} callers), very low revert ({revert_rate:.1%})"
            fp_method = "balanced_interaction"
            fp_confidence = 0.7

        # Heuristic 2: High caller count, sustained over multiple days, low revert
        if callers >= 100 and revert_rate < 0.02:
            days_active = conn.execute("""
                SELECT COUNT(DISTINCT DATE(timestamp)) as days
                FROM transaction_events WHERE contract_address=?
            """, (addr,)).fetchone()["days"]
            if days_active >= 3:
                fp_reason = f"Sustained organic traffic: {callers} callers over {days_active} days, {revert_rate:.1%} revert"
                fp_method = "sustained_traffic"
                fp_confidence = 0.8

        # Heuristic 3: Only triggered by weak detectors
        # Some patterns are inherently noisy (e.g., CALLER check appears in legitimate access control)
        weak_only = True
        strong_patterns = ["selfdestruct", "callback", "tx.origin", "tx_origin"]
        for sp in strong_patterns:
            if sp.lower() in patterns.lower():
                weak_only = False
                break

        if weak_only and "CALLER" in patterns and callers >= 30 and revert_rate < 0.1:
            fp_reason = f"Only weak detector fired (CALLER check), likely legitimate access control. {callers} callers, {revert_rate:.1%} revert"
            fp_method = "weak_detector_only"
            fp_confidence = 0.5
            detector_blamed = "asymmetric_transfer"

        # Heuristic 4: Contract is a known deployer's self-interaction target
        # (false positive from self-test detection — the contract IS a trap, but
        # the CALLER pattern is the access control, not the trap mechanism)
        # Skip this — self-test traps are true positives

        if fp_reason and fp_confidence >= 0.5:
            # Determine which detector to blame
            if not detector_blamed:
                if "CALLER" in patterns and "SLOAD" in patterns:
                    detector_blamed = "blacklist_check"
                elif "CALLER" in patterns:
                    detector_blamed = "asymmetric_transfer"
                elif "SHA3" in patterns and "DIV" in patterns:
                    detector_blamed = "obfuscated_fee"
                elif "DELEGATECALL" in patterns:
                    detector_blamed = "delegatecall_in_token"
                else:
                    detector_blamed = "unknown"

            conn.execute("""
                INSERT OR IGNORE INTO false_positives
                (contract_address, chain, original_tier, original_patterns,
                 fp_reason, fp_method, fp_confidence, detector_blamed, assessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (addr, c["chain"], c["confidence_tier"], patterns,
                  fp_reason, fp_method, fp_confidence, detector_blamed, now))
            found += 1

    conn.commit()
    return {"false_positives_found": found}


def compute_precision(conn: sqlite3.Connection):
    """Compute per-detector precision metrics."""
    ensure_tables(conn)
    today = _now()[:10]

    # Count how many times each detector fired
    detectors = [
        ("asymmetric_transfer", "CALLER at"),
        ("blacklist_check", "address-keyed storage"),
        ("tx_origin_conditional", "tx.origin conditional"),
        ("callback_trap", "callback"),
        ("hidden_fee", "hidden fee"),
        ("selfdestruct", "SELFDESTRUCT"),
        ("delegatecall_in_token", "DELEGATECALL"),
        ("timestamp_activation", "TIMESTAMP"),
        ("origin_eoa_gate", "tx.origin"),
        ("obfuscated_fee", "KECCAK256-keyed"),
    ]

    print(f"{'Detector':30} | {'Fired':>6} | {'FP':>4} | {'Confirmed':>9} | {'Precision':>9}")
    print("-" * 75)

    for name, pattern in detectors:
        # Total contracts where this detector fired
        fired = conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE bytecode_pattern_notes LIKE ?",
            (f"%{pattern}%",)
        ).fetchone()[0]

        # False positives attributed to this detector
        fps = conn.execute(
            "SELECT COUNT(*) FROM false_positives WHERE detector_blamed=?",
            (name,)
        ).fetchone()[0]

        # Confirmed traps (tier = confirmed)
        confirmed = conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE confidence_tier='confirmed' AND bytecode_pattern_notes LIKE ?",
            (f"%{pattern}%",)
        ).fetchone()[0]

        unknown = fired - fps - confirmed
        precision = (fired - fps) / fired if fired > 0 else 1.0

        conn.execute("""
            INSERT OR REPLACE INTO detector_precision
            (detector_name, date, total_fired, confirmed_traps, false_positives, unknown, precision)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, today, fired, confirmed, fps, unknown, round(precision, 3)))

        print(f"  {name:28} | {fired:>6} | {fps:>4} | {confirmed:>9} | {precision:>8.1%}")

    conn.commit()


def manual_override(conn: sqlite3.Connection, address: str, reason: str):
    """Manually mark a contract as false positive."""
    ensure_tables(conn)

    c = conn.execute("SELECT * FROM contracts WHERE contract_address LIKE ?",
                     (address.lower() + "%",)).fetchone()
    if not c:
        print(f"Contract {address} not found")
        return

    patterns = c["bytecode_pattern_notes"] or ""
    detector = "manual"
    if "CALLER" in patterns:
        detector = "asymmetric_transfer"
    elif "SHA3" in patterns:
        detector = "obfuscated_fee"
    elif "DELEGATECALL" in patterns:
        detector = "delegatecall_in_token"

    conn.execute("""
        INSERT OR REPLACE INTO false_positives
        (contract_address, chain, original_tier, original_patterns,
         fp_reason, fp_method, fp_confidence, detector_blamed, assessed_at)
        VALUES (?, ?, ?, ?, ?, 'manual_override', 1.0, ?, ?)
    """, (c["contract_address"], c["chain"], c["confidence_tier"],
          patterns, reason, detector, _now()))
    conn.commit()
    print(f"Marked {c['contract_address'][:20]}... as false positive: {reason}")


def print_summary(conn: sqlite3.Connection):
    ensure_tables(conn)

    total_suspected = conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='suspected'").fetchone()[0]
    total_fp = conn.execute("SELECT COUNT(*) FROM false_positives").fetchone()[0]
    total_confirmed = conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='confirmed'").fetchone()[0]

    print(f"Suspected contracts: {total_suspected:,}")
    print(f"Confirmed traps: {total_confirmed}")
    print(f"Identified false positives: {total_fp}")
    if total_suspected > 0:
        print(f"Estimated precision: {(total_suspected - total_fp) / total_suspected:.1%}")
    print()

    # By method
    print("False positives by detection method:")
    for r in conn.execute("SELECT fp_method, COUNT(*) as c FROM false_positives GROUP BY fp_method ORDER BY c DESC"):
        print(f"  {r['fp_method']:25} | {r['c']}")

    # By blamed detector
    print()
    print("False positives by detector blamed:")
    for r in conn.execute("SELECT detector_blamed, COUNT(*) as c FROM false_positives GROUP BY detector_blamed ORDER BY c DESC"):
        print(f"  {r['detector_blamed']:25} | {r['c']}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="False positive tracking")
    parser.add_argument("--scan", action="store_true", help="Scan for false positives")
    parser.add_argument("--precision", action="store_true", help="Compute detector precision")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--override", type=str, help="Manually mark as false positive")
    parser.add_argument("--reason", type=str, help="Reason for override")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    if args.scan:
        r = scan_false_positives(conn)
        print(f"[fp_tracker] Found {r['false_positives_found']} probable false positives")
        print()
        print_summary(conn)
    elif args.precision:
        compute_precision(conn)
    elif args.summary:
        print_summary(conn)
    elif args.override:
        if not args.reason:
            print("--reason required")
        else:
            manual_override(conn, args.override, args.reason)
    else:
        parser.print_help()

    conn.close()
