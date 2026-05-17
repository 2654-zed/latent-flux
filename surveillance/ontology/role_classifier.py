"""Role classifier — answers Q-001 (role lattice).

STATUS: SKELETON. The classification logic is sketched but not trained
against the full corpus. The data schema is defined; the rules are TODO.

Full build target: 1-week effort. Closes the 97.6% drain-coverage gap
documented in the 2026-05-15 SAI cycle.

Role lattice (per Q-001 spec):
    operator       — Pattern A deployer: a long-running cover identity
                     that operates bait contracts. Profile: small total
                     deployment count, high per-contract approval volume,
                     watchlist HIGH-priority "pristine_solo_operator" or
                     similar.
    execution_cell — drain_caller-only address. Profile: appears in
                     approval_watchlist.drain_caller but rarely as
                     deployer; short active window; tight time correlation
                     with operator discharge.
    funder         — appears in deployers.funding_sources upstream of
                     multiple operators. Currently tracked but not
                     cross-linked to execution cells.
    infrastructure — provides shared services (relayers, routers, gas
                     stations). Listed in infrastructure_operator_candidates.
    intermediary   — pass-through node in laundering chains. NOT yet
                     fully captured by Layer 3 schema.

CLI:
    python -m surveillance.ontology.role_classifier
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


@dataclass
class RoleClassification:
    address: str
    roles: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    rationale: dict[str, str] = field(default_factory=dict)


def classify_address(conn: sqlite3.Connection, address: str) -> RoleClassification:
    """Classify a single address across the role lattice.

    TODO: full implementation. Current sketch only checks the obvious
    discriminators (is_drain_caller, is_deployer, is_funder).
    """
    out = RoleClassification(address=address)

    # is_execution_cell: drain_caller in approval_watchlist
    n_drains = conn.execute(
        "SELECT COUNT(*) FROM approval_watchlist WHERE drain_caller=? AND drain_detected=1",
        (address,)
    ).fetchone()[0]
    if n_drains > 0:
        out.roles.append("execution_cell")
        out.confidence["execution_cell"] = min(1.0, n_drains / 100.0)
        out.rationale["execution_cell"] = f"drain_caller in {n_drains} approval_watchlist rows"

    # is_operator: in deployers AND has confirmed-tier contract AND watchlist HIGH
    is_op = conn.execute(
        """SELECT COUNT(*) FROM deployers d
           JOIN watchlist w ON w.address = d.deployer_address AND w.active = 1
           WHERE d.deployer_address = ?""",
        (address,)
    ).fetchone()[0]
    if is_op > 0:
        # Stronger signal: bait contract with high approval volume
        bait_approvals = conn.execute(
            """SELECT MAX(n) FROM (
                 SELECT COUNT(*) AS n FROM approval_watchlist aw
                 JOIN contracts c ON c.contract_address = aw.contract_address
                 WHERE c.deployer_address = ?
                 GROUP BY aw.contract_address
               )""",
            (address,)
        ).fetchone()
        max_baits = bait_approvals[0] if bait_approvals and bait_approvals[0] else 0
        if max_baits > 100:
            out.roles.append("operator")
            out.confidence["operator"] = min(1.0, max_baits / 1000.0)
            out.rationale["operator"] = f"on watchlist + max bait approval volume = {max_baits}"

    # TODO: is_funder, is_infrastructure, is_intermediary
    return out


def classify_recent_drainers(conn: sqlite3.Connection, since: str = "2026-05-01") -> list[RoleClassification]:
    """Classify every drain_caller seen since `since`."""
    cur = conn.execute(
        """SELECT DISTINCT drain_caller FROM approval_watchlist
           WHERE drain_detected=1 AND drain_timestamp >= ?
             AND drain_caller IS NOT NULL""",
        (since,)
    )
    return [classify_address(conn, row[0]) for row in cur]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default="2026-05-01")
    ap.add_argument("--address", default=None, help="classify a single address")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        if args.address:
            rc = classify_address(conn, args.address)
            print(f"Address: {rc.address}")
            print(f"Roles:   {', '.join(rc.roles) or '(none)'}")
            for r in rc.roles:
                print(f"  {r}: confidence={rc.confidence.get(r, 0):.2f}  rationale={rc.rationale.get(r, '')}")
            return 0
        # All recent drainers
        sys.stderr.write(f"[SKELETON] Classifying drainers since {args.since}...\n")
        rcs = classify_recent_drainers(conn, since=args.since)
        print(f"Classified {len(rcs)} drain-callers (SKELETON — partial rules only):")
        for rc in rcs:
            print(f"  {rc.address}  roles={rc.roles}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
