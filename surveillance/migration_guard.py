"""Drain-evidence veto for confidence-tier migrations (Correction #27, Finding 4).

The Correction #25 audit migrated 347 "false-positive" contracts
confirmed->unanalyzed using Blockscout-legitimacy + activity heuristics.
None of those heuristics checked whether the contract had on-chain drain
evidence. Result (scripts/audit_migration_drain_gate.py): 45 of the 347
had drain_detected rows no heuristic looked at. Local drain-shape triage
(scripts/fix1_drain_shape_classify.py) splits those into sustained-drainer-
shaped vs Bug#19b fan-out artifacts; an on-chain Blockscout token-transfer
decode is required to confirm which are genuine before restoring any tier.

Root rule encoded here: a legitimacy signal (verified source, holder
count, framework match) must NEVER override the presence of real harm
evidence. Any migration that downgrades a contract OUT of an adversarial
tier must first call migration_blocked_by_drains() and skip on True.

"Real drain evidence" is defined by SHAPE, not raw drain_detected count,
because the raw count is itself inflated by Bug #19b (Correction #27):
  - >= MIN_DISTINCT_TX distinct drain tx with small median victims/tx
    => sustained drain operation => BLOCK migration.
  - 1-2 tx fanned to many victims => Bug#19b artifact, not a real drainer
    => do NOT block.
"""
from __future__ import annotations
import sqlite3
import statistics

MIN_DISTINCT_TX = 3              # >= this many distinct drain tx => sustained op
MAX_MEDIAN_VICTIMS_PER_TX = 15   # above this, looks like Bug#19b fan-out


def drain_shape(conn: sqlite3.Connection, address: str) -> dict:
    """Return drain-shape metrics for a contract from approval_watchlist."""
    pertx: dict[str, set] = {}
    for txh, victim in conn.execute(
        "SELECT drain_tx_hash, victim_address FROM approval_watchlist "
        "WHERE contract_address=? AND drain_detected=1 AND drain_tx_hash IS NOT NULL",
        (address,),
    ):
        pertx.setdefault(txh, set()).add((victim or "").lower())
    if not pertx:
        return {"distinct_tx": 0, "median_victims_per_tx": 0, "max_victims_per_tx": 0}
    vpt = sorted((len(v) for v in pertx.values()), reverse=True)
    return {
        "distinct_tx": len(pertx),
        "median_victims_per_tx": statistics.median(vpt),
        "max_victims_per_tx": max(vpt),
    }


def migration_blocked_by_drains(conn: sqlite3.Connection, address: str) -> bool:
    """True if `address` has real (shape-consistent) drain evidence and
    must NOT be auto-migrated out of an adversarial tier. Conservative:
    blocks on sustained-drainer shape; Bug#19b fan-out artifacts do not
    block. Borderline cases should route to manual review, not auto-migrate.
    """
    s = drain_shape(conn, address)
    return (
        s["distinct_tx"] >= MIN_DISTINCT_TX
        and s["median_victims_per_tx"] <= MAX_MEDIAN_VICTIMS_PER_TX
    )
