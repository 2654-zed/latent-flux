"""Confidence decay — downgrade aged 'suspected' contracts with zero observable harm.

Finding that motivates this module (P0 of exception-as-rule audit, 2026-04-19):
  - 43,985 'suspected' contracts; 8 with trap_events (0.02% observable harm rate).
  - 3,797 'suspected' aged >=30 days; ZERO fired.
  - 579 'confirmed' -> 482 with trap_events (83.2% PPV — the label means something).
  - Conclusion: 'suspected' at 30d+ with no observable harm is noise, not signal.

This module moves such rows to confidence_tier='unanalyzed' and records:
  - prior_confidence_tier (so re-promotion is possible if harm later appears)
  - decayed_at timestamp (audit trail)
The original confidence_reason and bytecode_pattern_notes are preserved verbatim.

CLI:
    python -m surveillance.confidence_decay --dry-run
    python -m surveillance.confidence_decay --apply
    python -m surveillance.confidence_decay --apply --age-days 45
"""

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
DEFAULT_AGE_DAYS = 30


def compute_candidates(conn: sqlite3.Connection, age_days: int) -> list[dict]:
    """Return suspected contracts aged >= age_days with zero observable harm."""
    rows = conn.execute(
        """
        SELECT c.contract_address,
               c.chain,
               c.detection_timestamp,
               c.confidence_reason
        FROM contracts c
        WHERE c.confidence_tier = 'suspected'
          AND c.detection_timestamp < datetime('now', ? )
          AND c.decayed_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM trap_events te
              WHERE LOWER(te.trap_contract_address) = LOWER(c.contract_address)
          )
        """,
        (f"-{age_days} days",),
    ).fetchall()
    return [
        {
            "contract_address": r[0],
            "chain": r[1],
            "detection_timestamp": r[2],
            "confidence_reason": r[3],
        }
        for r in rows
    ]


def apply_decay(conn: sqlite3.Connection, age_days: int) -> dict:
    """Downgrade aged suspected contracts to unanalyzed. Returns summary."""
    candidates = compute_candidates(conn, age_days)
    now = datetime.now(timezone.utc).isoformat()
    for c in candidates:
        conn.execute(
            """
            UPDATE contracts
               SET prior_confidence_tier = confidence_tier,
                   confidence_tier = 'unanalyzed',
                   decayed_at = ?
             WHERE contract_address = ?
            """,
            (now, c["contract_address"]),
        )
    conn.commit()
    return {
        "decayed_count": len(candidates),
        "age_days_threshold": age_days,
        "ran_at": now,
    }


def corpus_stats(conn: sqlite3.Connection) -> dict:
    """Snapshot of confidence_tier distribution for reporting."""
    dist = dict(
        conn.execute(
            "SELECT COALESCE(confidence_tier,'NULL'), COUNT(*) FROM contracts GROUP BY 1"
        ).fetchall()
    )
    total = sum(dist.values())
    return {
        "total": total,
        "by_tier": dist,
        "suspected_pct": round(100 * dist.get("suspected", 0) / max(total, 1), 2),
        "unanalyzed_pct": round(100 * dist.get("unanalyzed", 0) / max(total, 1), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Decay aged suspected contracts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would decay without modifying the DB.")
    parser.add_argument("--apply", action="store_true",
                        help="Apply decay to the DB.")
    parser.add_argument("--age-days", type=int, default=DEFAULT_AGE_DAYS,
                        help=f"Age threshold in days (default {DEFAULT_AGE_DAYS}).")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to surveillance.db.")
    args = parser.parse_args()

    if not (args.dry_run or args.apply):
        parser.error("Pass --dry-run or --apply.")

    conn = sqlite3.connect(args.db, timeout=60)
    before = corpus_stats(conn)

    if args.dry_run:
        candidates = compute_candidates(conn, args.age_days)
        print(f"[dry-run] age_days={args.age_days}")
        print(f"[dry-run] would decay {len(candidates):,} suspected contracts")
        print(f"[dry-run] before: suspected={before['by_tier'].get('suspected',0):,} "
              f"({before['suspected_pct']}%), "
              f"unanalyzed={before['by_tier'].get('unanalyzed',0):,}")
        if candidates:
            # Preview by chain
            by_chain: dict = {}
            for c in candidates:
                by_chain[c["chain"]] = by_chain.get(c["chain"], 0) + 1
            print("[dry-run] decay by chain:")
            for ch, n in sorted(by_chain.items()):
                print(f"  {ch:<12} {n:,}")
        conn.close()
        return

    result = apply_decay(conn, args.age_days)
    after = corpus_stats(conn)
    conn.close()

    print(f"[apply] age_days={args.age_days}")
    print(f"[apply] decayed: {result['decayed_count']:,}")
    print(f"[apply] before -> after suspected: "
          f"{before['by_tier'].get('suspected',0):,} ({before['suspected_pct']}%) -> "
          f"{after['by_tier'].get('suspected',0):,} ({after['suspected_pct']}%)")
    print(f"[apply] before -> after unanalyzed: "
          f"{before['by_tier'].get('unanalyzed',0):,} ({before['unanalyzed_pct']}%) -> "
          f"{after['by_tier'].get('unanalyzed',0):,} ({after['unanalyzed_pct']}%)")
    print(f"[apply] ran_at: {result['ran_at']}")


if __name__ == "__main__":
    main()
