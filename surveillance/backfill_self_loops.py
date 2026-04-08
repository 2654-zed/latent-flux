"""
Layer 3 — Self-Loop Promotion Backfill

Finds all cases in transaction_events where a deployer called a contract
they themselves deployed in a reverted transaction, and promotes those
contracts to CONFIRMED with a backfill-tagged reason.

This mirrors the live rule in surveillance/revert_cluster_detector.py
(commit 552baab) against existing historical data. Only fires on NEW
revert events going forward, so this one-shot backfill is needed to
catch dual-role operators already in the corpus.

Safety criteria (all must hold):
  1. Deployer address is in bot_candidates with is_deployer=1 (the
     revert detector has previously flagged them as dual-role)
  2. They called their own contract at least 2 times (filters out
     one-off admin actions or legitimate initialization)
  3. At least 1 of those self-calls was reverted (filters out
     legitimate config/admin flows which typically succeed)
  4. The contract is not already confirmed (idempotent)

Pure SQLite, zero RPC calls. Idempotent — re-running is a no-op.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from datetime import datetime, timezone

logger = logging.getLogger("surveillance.backfill_self_loops")


def backfill_self_loops(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """
    Promote contracts to confirmed where the deployer is a flagged
    bot_candidate AND has called their own contract with at least one
    reverted self-call.

    Returns a summary dict with promotion counts and top operators.
    """
    # Find promotion candidates: join transaction_events (caller == deployer),
    # contracts (current tier), and bot_candidates (dual-role filter).
    candidates = conn.execute(
        """
        SELECT  c.contract_address,
                c.deployer_address,
                c.confidence_tier,
                c.chain,
                COUNT(*)                              AS self_calls,
                SUM(COALESCE(te.is_reverted, 0))      AS self_reverts
        FROM    contracts c
        JOIN    transaction_events te
                ON te.contract_address = c.contract_address
               AND te.interacting_address = c.deployer_address
        JOIN    bot_candidates bc
                ON bc.address = c.deployer_address
               AND bc.is_deployer = 1
        WHERE   c.confidence_tier != 'confirmed'
        GROUP BY c.contract_address, c.deployer_address, c.confidence_tier, c.chain
        HAVING  self_calls   >= 2
            AND self_reverts >= 1
        ORDER BY self_reverts DESC, self_calls DESC
        """
    ).fetchall()

    now_iso = datetime.now(timezone.utc).isoformat()
    promoted = 0
    by_deployer: dict[str, int] = {}
    top_rows: list[tuple] = []

    for contract, deployer, old_tier, chain, calls, reverts in candidates:
        reason = (
            f"BACKFILL: Self-loop detected. Deployer {deployer} "
            f"called own contract {calls} times ({reverts} reverted). "
            f"BOT+DEPLOYER historical backfill."
        )
        if not dry_run:
            conn.execute(
                """
                UPDATE contracts
                SET confidence_tier = 'confirmed',
                    confidence_reason = ?,
                    last_updated      = ?
                WHERE contract_address = ?
                """,
                (reason, now_iso, contract),
            )
        promoted += 1
        by_deployer[deployer] = by_deployer.get(deployer, 0) + 1
        if len(top_rows) < 50:
            top_rows.append((contract, deployer, old_tier, chain, calls, reverts))

    if not dry_run:
        conn.commit()

    top_operators = sorted(by_deployer.items(), key=lambda x: -x[1])[:20]

    summary = {
        "candidates_evaluated": len(candidates),
        "promoted": promoted,
        "dry_run": dry_run,
        "top_operators": top_operators,
        "sample_rows": top_rows[:20],
    }

    logger.info(
        "backfill_self_loops: promoted=%d dry_run=%s unique_operators=%d",
        promoted, dry_run, len(by_deployer),
    )
    return summary


def _print_summary(summary: dict) -> None:
    prefix = "[DRY RUN] " if summary["dry_run"] else ""
    print(f"{prefix}Candidates evaluated: {summary['candidates_evaluated']}")
    print(f"{prefix}Contracts promoted:   {summary['promoted']}")
    print(f"{prefix}Unique operators:     {len(summary['top_operators'])}")
    if summary["top_operators"]:
        print("\nTop operators by promoted-contract count:")
        for addr, n in summary["top_operators"]:
            print(f"  {addr}  {n}")
    if summary["sample_rows"]:
        print("\nSample promoted contracts (top 20 by self-revert count):")
        for contract, deployer, old_tier, chain, calls, reverts in summary["sample_rows"]:
            print(f"  {contract} chain={chain} was={old_tier:>9} calls={calls:>6} rv={reverts:>6}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_path = "surveillance/data/surveillance.db"
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if args:
        db_path = args[0]
    conn = sqlite3.connect(db_path)
    try:
        summary = backfill_self_loops(conn, dry_run=dry_run)
        _print_summary(summary)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
