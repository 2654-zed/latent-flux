"""
Layer 3 — Deployer Count Refresh

Recomputes total_contracts_deployed and last_seen for all deployers from
the contracts table. Pure SQLite, zero RPC calls.

Background: the deployers table sets total_contracts_deployed once at
first detection and never updates. The coffee fleet 0xc0ffeefeed8b9d27
was recorded at 55 but actually has 209 contracts. This affects every
active deployer whose fleet has grown since first detection.

Run as a one-shot or daily maintenance job.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from typing import Iterable

logger = logging.getLogger("surveillance.refresh_deployer_counts")


def refresh_deployer_counts(conn: sqlite3.Connection) -> dict:
    """
    Recompute deployers.total_contracts_deployed and last_seen from the
    contracts table. Returns a summary dict.
    """
    # Step 1: Compute actual counts per deployer
    actual = {
        row[0]: (row[1], row[2])
        for row in conn.execute(
            """
            SELECT deployer_address,
                   COUNT(*)                        AS cnt,
                   MAX(detection_timestamp)        AS last_deploy
            FROM contracts
            WHERE deployer_address IS NOT NULL AND deployer_address != ''
            GROUP BY deployer_address
            """
        ).fetchall()
    }

    # Step 2: Load current recorded values for deployers that have ≥1 contract
    recorded = {
        row[0]: (row[1], row[2])
        for row in conn.execute(
            """
            SELECT deployer_address, total_contracts_deployed, last_seen
            FROM deployers
            """
        ).fetchall()
    }

    stale_counts = 0
    stale_last_seen = 0
    top_deltas: list[tuple[str, int, int, int]] = []

    for addr, (actual_cnt, last_deploy) in actual.items():
        if addr not in recorded:
            continue
        rec_cnt, rec_last = recorded[addr]
        delta = actual_cnt - (rec_cnt or 0)

        needs_count_update = rec_cnt != actual_cnt
        needs_last_update = (last_deploy or "") > (rec_last or "")

        if needs_count_update or needs_last_update:
            conn.execute(
                """
                UPDATE deployers
                SET total_contracts_deployed = ?,
                    last_seen = CASE
                        WHEN ? > COALESCE(last_seen, '') THEN ?
                        ELSE last_seen
                    END
                WHERE deployer_address = ?
                """,
                (actual_cnt, last_deploy, last_deploy, addr),
            )
            if needs_count_update:
                stale_counts += 1
                top_deltas.append((addr, rec_cnt or 0, actual_cnt, delta))
            if needs_last_update:
                stale_last_seen += 1

    conn.commit()

    top_deltas.sort(key=lambda x: x[3], reverse=True)
    summary = {
        "total_deployers_evaluated": len(actual),
        "stale_counts_fixed": stale_counts,
        "stale_last_seen_fixed": stale_last_seen,
        "top_deltas": top_deltas[:20],
    }

    logger.info(
        "refresh_deployer_counts: evaluated=%d stale_counts=%d stale_last_seen=%d",
        summary["total_deployers_evaluated"],
        summary["stale_counts_fixed"],
        summary["stale_last_seen_fixed"],
    )
    return summary


def _print_summary(summary: dict) -> None:
    print(f"Deployers evaluated:    {summary['total_deployers_evaluated']}")
    print(f"Stale counts fixed:     {summary['stale_counts_fixed']}")
    print(f"Stale last_seen fixed:  {summary['stale_last_seen_fixed']}")
    if summary["top_deltas"]:
        print("\nTop 20 count deltas (was -> now, delta):")
        for addr, was, now, delta in summary["top_deltas"]:
            print(f"  {addr}  {was:>6} -> {now:>6}  (+{delta})")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_path = "surveillance/data/surveillance.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    conn = sqlite3.connect(db_path)
    try:
        summary = refresh_deployer_counts(conn)
        _print_summary(summary)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
