"""Funding chain pathfinder — answers Q-009.

For each drain_caller in a target window, walks the funding-chain upstream
to find:
  - Direct funder (from deployers.funding_trail JSON)
  - Whether funder is on watchlist
  - Whether funder is in oli_labels (HIGH-severity OLI is a strong signal
    of an institutionally-tagged funding source — see UNK-031 for the
    why-this-matters case)
  - Whether funder ALSO drains (multi-cell operator: same upstream EOA
    funds multiple execution cells)
  - Up to N hops upstream (default 3)

The actionable output: surface drain executors whose 1-2 hop ancestors
are watchlisted but who are themselves not watchlisted. These are the
execution cells that the operator-side flag did not catch.

Empirical anchor: 2026-05-15 SAI cycle found 34 of 38 May-9..15 drain
executors off-watchlist (97.6% drain volume uncovered). The funding chain
should resolve a significant fraction of these to known operators.

CLI:
    python -m surveillance.ontology.funding_chain_pathfinder
    python -m surveillance.ontology.funding_chain_pathfinder --since 2026-05-09
    python -m surveillance.ontology.funding_chain_pathfinder --address 0x1d81...
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


@dataclass
class FundingHop:
    address: str
    depth: int
    value_eth: float | None
    timestamp: str | None
    tx_hash: str | None
    watchlist_label: str | None = None
    watchlist_priority: str | None = None
    oli_severity: str | None = None
    is_known_drainer: bool = False
    deployer_count: int = 0
    notes: str = ""


@dataclass
class FundingChain:
    drain_caller: str
    drains_in_window: int
    hops: list[FundingHop] = field(default_factory=list)
    terminal_reason: str = ""

    def is_resolved_to_known(self) -> bool:
        """True if any hop is on the watchlist or has an OLI tag."""
        return any(
            (h.watchlist_label is not None) or (h.oli_severity is not None)
            for h in self.hops
        )

    def resolution_summary(self) -> str:
        for h in self.hops:
            if h.watchlist_label:
                return (
                    f"WATCHLIST_HIT@hop{h.depth}: {h.address} "
                    f"({h.watchlist_label} / {h.watchlist_priority})"
                )
            if h.oli_severity:
                return f"OLI_HIT@hop{h.depth}: {h.address} ({h.oli_severity})"
            if h.is_known_drainer:
                return f"OTHER_DRAINER@hop{h.depth}: {h.address}"
        return self.terminal_reason or "UNRESOLVED"


def parse_funding_trail(raw: str | None) -> dict | None:
    """Parse the funding_trail JSON field. Tolerates malformed entries."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            # If it's a list, the first entry is the most-recent funder
            return data[0] if isinstance(data[0], dict) else None
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def lookup_hop_metadata(conn: sqlite3.Connection, address: str) -> dict:
    """Return all known labels for an address: watchlist, oli, deployer count, etc."""
    out = {
        "watchlist_label": None,
        "watchlist_priority": None,
        "oli_severity": None,
        "is_known_drainer": False,
        "deployer_count": 0,
    }
    row = conn.execute(
        "SELECT entity_name, priority FROM watchlist WHERE address=? AND active=1",
        (address,)
    ).fetchone()
    if row:
        out["watchlist_label"] = row[0]
        out["watchlist_priority"] = row[1]
    row = conn.execute(
        "SELECT severity FROM oli_labels WHERE address=?",
        (address,)
    ).fetchone()
    if row:
        out["oli_severity"] = row[0]
    # Drainer in our corpus?
    n_drains = conn.execute(
        "SELECT COUNT(*) FROM approval_watchlist WHERE drain_caller=? AND drain_detected=1",
        (address,)
    ).fetchone()[0]
    if n_drains > 0:
        out["is_known_drainer"] = True
        out["n_drains"] = n_drains
    # Has it deployed contracts?
    row = conn.execute(
        "SELECT total_contracts_deployed FROM deployers WHERE deployer_address=?",
        (address,)
    ).fetchone()
    if row and row[0] is not None:
        out["deployer_count"] = row[0]
    return out


def walk_chain(conn: sqlite3.Connection, start_address: str, max_hops: int = 3) -> FundingChain:
    """Walk the funding chain upstream from start_address up to max_hops."""
    drains = conn.execute(
        "SELECT COUNT(*) FROM approval_watchlist WHERE drain_caller=? AND drain_detected=1",
        (start_address,)
    ).fetchone()[0]
    chain = FundingChain(drain_caller=start_address, drains_in_window=drains)
    seen: set[str] = {start_address}
    current = start_address
    for depth in range(1, max_hops + 1):
        row = conn.execute(
            "SELECT funding_trail FROM deployers WHERE deployer_address=?",
            (current,)
        ).fetchone()
        if row is None:
            chain.terminal_reason = f"address not in deployers table at depth {depth-1}"
            break
        trail = parse_funding_trail(row[0])
        if not trail:
            chain.terminal_reason = f"no funding_trail at depth {depth-1}"
            break
        funder = trail.get("funder")
        if not funder:
            chain.terminal_reason = f"funding_trail missing funder at depth {depth-1}"
            break
        funder = funder.lower()
        if funder in seen:
            chain.terminal_reason = f"cycle detected at hop {depth}"
            break
        seen.add(funder)
        meta = lookup_hop_metadata(conn, funder)
        chain.hops.append(FundingHop(
            address=funder,
            depth=depth,
            value_eth=trail.get("value_eth"),
            timestamp=trail.get("timestamp"),
            tx_hash=trail.get("tx_hash"),
            watchlist_label=meta["watchlist_label"],
            watchlist_priority=meta["watchlist_priority"],
            oli_severity=meta["oli_severity"],
            is_known_drainer=meta["is_known_drainer"],
            deployer_count=meta["deployer_count"],
        ))
        # Terminate early if we hit a flagged ancestor
        if meta["watchlist_label"] or meta["oli_severity"]:
            chain.terminal_reason = f"flagged ancestor at hop {depth}"
            break
        current = funder
    else:
        chain.terminal_reason = f"max_hops ({max_hops}) reached without flagged ancestor"
    return chain


def trace_drainers_in_window(
    conn: sqlite3.Connection,
    since: str,
    until: str,
    max_hops: int = 3,
) -> list[FundingChain]:
    """Find all distinct drain_callers in [since, until) and trace each."""
    drainers = [r[0] for r in conn.execute(
        """SELECT DISTINCT drain_caller FROM approval_watchlist
           WHERE drain_detected=1
             AND drain_timestamp >= ? AND drain_timestamp < ?
             AND drain_caller IS NOT NULL""",
        (since, until)
    )]
    return [walk_chain(conn, d, max_hops=max_hops) for d in drainers]


def summarize(chains: list[FundingChain]) -> str:
    n = len(chains)
    n_resolved = sum(1 for c in chains if c.is_resolved_to_known())
    n_watchlist = sum(
        1 for c in chains
        if any(h.watchlist_label for h in c.hops)
    )
    n_oli = sum(
        1 for c in chains
        if any(h.oli_severity for h in c.hops)
    )
    n_other_drainer = sum(
        1 for c in chains
        if any(h.is_known_drainer for h in c.hops)
    )
    total_drains = sum(c.drains_in_window for c in chains)
    drains_resolved = sum(c.drains_in_window for c in chains if c.is_resolved_to_known())
    lines = [
        f"  Drain-callers traced: {n}",
        f"  Total drain volume in window: {total_drains:,}",
        f"  Drainers with watchlist/OLI ancestor: {n_resolved} ({100*n_resolved/n:.1f}%)",
        f"    via watchlist: {n_watchlist}",
        f"    via oli_labels: {n_oli}",
        f"    via other-known-drainer: {n_other_drainer}",
        f"  Drain volume resolved to known operators: {drains_resolved:,} "
        f"({100*drains_resolved/max(total_drains,1):.1f}%)",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default="2026-05-09")
    ap.add_argument("--until", default="2026-05-16")
    ap.add_argument("--address", default=None,
                    help="trace a single drain_caller instead of the window")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--show-unresolved", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        if args.address:
            chain = walk_chain(conn, args.address.lower(), args.max_hops)
            print(f"\nDrain-caller: {chain.drain_caller}  drains={chain.drains_in_window}")
            for h in chain.hops:
                flags = []
                if h.watchlist_label: flags.append(f"WL={h.watchlist_label[:30]} ({h.watchlist_priority})")
                if h.oli_severity: flags.append(f"OLI={h.oli_severity}")
                if h.is_known_drainer: flags.append("KNOWN_DRAINER")
                if h.deployer_count: flags.append(f"deployer_count={h.deployer_count}")
                print(f"  hop{h.depth} {h.address}  {' | '.join(flags) or '(no labels)'}")
                print(f"       funded with {h.value_eth} ETH at {h.timestamp}  tx={h.tx_hash}")
            print(f"  Terminal: {chain.terminal_reason}")
            print(f"  Resolution: {chain.resolution_summary()}")
            return 0

        chains = trace_drainers_in_window(conn, args.since, args.until, args.max_hops)
        print(f"\nFunding-chain pathfinder — answers Q-009")
        print(f"  Window: {args.since} .. {args.until}")
        print(f"  Max hops: {args.max_hops}\n")
        print(summarize(chains))
        print()
        # Show resolved (flagged-ancestor) cases — the actionable ones
        print(f"\n  --- RESOLVED to known ancestor (actionable) ---")
        for c in sorted(chains, key=lambda x: -x.drains_in_window):
            if c.is_resolved_to_known():
                print(f"  drainer={c.drain_caller}  drains={c.drains_in_window}")
                print(f"    -> {c.resolution_summary()}")
        if args.show_unresolved:
            print(f"\n  --- UNRESOLVED (no flagged ancestor in {args.max_hops} hops) ---")
            for c in sorted(chains, key=lambda x: -x.drains_in_window):
                if not c.is_resolved_to_known():
                    print(f"  drainer={c.drain_caller}  drains={c.drains_in_window}")
                    print(f"    terminal: {c.terminal_reason}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
