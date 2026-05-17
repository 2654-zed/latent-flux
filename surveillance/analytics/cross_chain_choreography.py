"""Cross-chain choreography detector — answers Q-005.

For every contract deploy by a watchlisted or OLI-tagged address, surfaces
same-address activity on OTHER chains in a tight time window. This is the
leading-indicator surveillance for operators using one EOA as a cover
identity across chains.

Signals counted as cross-chain activity:
  - Another contract deploy by the same address on another chain
    (the strongest signal; bytecode + chain + time correlation)
  - A bridge_events row from the same address on another chain
    (sparse but high-fidelity)
  - The deployer's mainnet_first_tx predating L2 first_seen by >= 60 days
    (Pattern D — Cross-Chain Reputation Import; weakest signal but most
    populated)

Empirical anchor: 0x80b12bd0 deployed bait on Base 2026-03-26 03:12 UTC;
at 04:02 UTC (50 min later) the same address executed a LayerZero send()
on Ethereum bridging 1 OFC token to itself on Base. The Animoca case was
detected by accident in the 2026-05-15 investigation — this module
operationalizes the detection.

Honest gap: 0x80b12bd0's specific LayerZero send() is NOT in our
bridge_events table (only 51 rows total). What we CAN detect for that
address is the Pattern D signature: mainnet_first_tx=2019-05-23,
Base L2 first_seen=2026-03-26 (~7-year gap).

CLI:
    python -m surveillance.analytics.cross_chain_choreography
    python -m surveillance.analytics.cross_chain_choreography --address 0x80b12bd0
    python -m surveillance.analytics.cross_chain_choreography --since 2026-03-01
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"


@dataclass
class CrossChainSignal:
    """One piece of evidence that a deployer is cross-chain choreographing."""
    kind: str         # "multi_chain_deploys" | "bridge_event_same_address" | "pattern_d_gap"
    primary_event: dict
    correlated_event: dict | None
    score: float
    rationale: str


@dataclass
class OperatorChoreography:
    address: str
    watchlist_label: str | None
    oli_severity: str | None
    mainnet_first_tx: str | None
    deploys_by_chain: dict[str, int] = field(default_factory=dict)
    signals: list[CrossChainSignal] = field(default_factory=list)

    def aggregate_score(self) -> float:
        return sum(s.score for s in self.signals)

    def has_choreography(self) -> bool:
        return len(self.signals) > 0


def fetch_watchlisted_or_oli_addresses(conn: sqlite3.Connection) -> dict[str, dict]:
    """Return {address: {watchlist_label, watchlist_priority, oli_severity}}."""
    out: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT address, entity_name, priority FROM watchlist WHERE active=1 AND address IS NOT NULL"
    ):
        out[r[0].lower()] = {
            "watchlist_label": r[1],
            "watchlist_priority": r[2],
            "oli_severity": None,
        }
    for r in conn.execute("SELECT address, severity FROM oli_labels"):
        a = r[0].lower()
        if a not in out:
            out[a] = {
                "watchlist_label": None,
                "watchlist_priority": None,
                "oli_severity": r[1],
            }
        else:
            out[a]["oli_severity"] = r[1]
    return out


def detect_for_address(
    conn: sqlite3.Connection,
    address: str,
    since: datetime | None = None,
    bridge_window_hours: int = 6,
    pattern_d_gap_days: int = 60,
) -> OperatorChoreography:
    """Run all detection signals for a single address."""
    # Deployer record (chain + mainnet history)
    d_row = conn.execute(
        "SELECT chain, first_seen, mainnet_first_tx FROM deployers WHERE deployer_address=?",
        (address,)
    ).fetchone()
    mainnet_first_tx = d_row[2] if d_row else None

    # Lookup labels
    labels = conn.execute(
        "SELECT entity_name, priority FROM watchlist WHERE address=? AND active=1",
        (address,)
    ).fetchone()
    oli = conn.execute(
        "SELECT severity FROM oli_labels WHERE address=?",
        (address,)
    ).fetchone()

    chore = OperatorChoreography(
        address=address,
        watchlist_label=labels[0] if labels else None,
        oli_severity=oli[0] if oli else None,
        mainnet_first_tx=mainnet_first_tx,
    )

    # Signal 1: multi-chain deploys by same address
    deploys = conn.execute(
        """SELECT chain, contract_address, detection_timestamp
           FROM contracts WHERE deployer_address=?
           ORDER BY detection_timestamp""",
        (address,)
    ).fetchall()
    by_chain: dict[str, list[tuple[str, str]]] = {}
    for chain, ca, ts in deploys:
        by_chain.setdefault(chain, []).append((ca, ts))
        chore.deploys_by_chain[chain] = chore.deploys_by_chain.get(chain, 0) + 1

    if len(by_chain) >= 2:
        # Cross-chain deploys: find pairs from different chains within the window
        all_events: list[tuple[str, str, str]] = []
        for chain, evs in by_chain.items():
            for ca, ts in evs:
                all_events.append((chain, ca, ts))
        all_events.sort(key=lambda x: x[2])
        for i, (c1, ca1, t1) in enumerate(all_events):
            for c2, ca2, t2 in all_events[i + 1:]:
                if c1 == c2:
                    continue
                try:
                    dt1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
                    dt2 = datetime.fromisoformat(t2.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if dt1.tzinfo is None:
                    dt1 = dt1.replace(tzinfo=timezone.utc)
                if dt2.tzinfo is None:
                    dt2 = dt2.replace(tzinfo=timezone.utc)
                hours = abs((dt2 - dt1).total_seconds()) / 3600.0
                if hours <= bridge_window_hours:
                    chore.signals.append(CrossChainSignal(
                        kind="multi_chain_deploys",
                        primary_event={"chain": c1, "contract": ca1, "ts": t1},
                        correlated_event={"chain": c2, "contract": ca2, "ts": t2},
                        score=3.0,
                        rationale=f"deploys on {c1} and {c2} {hours:.1f}h apart",
                    ))
                    break  # one is enough for the pair; move on

    # Signal 2: bridge_events from same address (sparse but high-fidelity)
    bridge_rows = conn.execute(
        "SELECT chain, timestamp, tx_hash, value_eth, bridge_name FROM bridge_events WHERE sender=?",
        (address,)
    ).fetchall()
    if bridge_rows and deploys:
        for chain_b, ts_b, tx_b, val_b, bn_b in bridge_rows:
            try:
                dt_b = datetime.fromisoformat(ts_b.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt_b.tzinfo is None:
                dt_b = dt_b.replace(tzinfo=timezone.utc)
            for chain_d, ca, ts_d in deploys:
                try:
                    dt_d = datetime.fromisoformat(ts_d.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if dt_d.tzinfo is None:
                    dt_d = dt_d.replace(tzinfo=timezone.utc)
                if chain_b == chain_d:
                    continue
                hours = abs((dt_b - dt_d).total_seconds()) / 3600.0
                if hours <= bridge_window_hours:
                    chore.signals.append(CrossChainSignal(
                        kind="bridge_event_same_address",
                        primary_event={"chain": chain_d, "contract": ca, "ts": ts_d},
                        correlated_event={"chain": chain_b, "tx": tx_b,
                                          "ts": ts_b, "value_eth": val_b,
                                          "bridge": bn_b},
                        score=4.0,
                        rationale=f"deploy on {chain_d} + bridge on {chain_b} "
                                  f"({bn_b}) {hours:.1f}h apart",
                    ))
                    break

    # Signal 3: Pattern D — mainnet_first_tx predates L2 by >= N days
    if d_row and mainnet_first_tx and d_row[1]:
        try:
            l2_first = datetime.fromisoformat(d_row[1].replace("Z", "+00:00"))
            mn_first = datetime.fromisoformat(mainnet_first_tx.replace("Z", "+00:00"))
            if l2_first.tzinfo is None: l2_first = l2_first.replace(tzinfo=timezone.utc)
            if mn_first.tzinfo is None: mn_first = mn_first.replace(tzinfo=timezone.utc)
            gap_days = (l2_first - mn_first).total_seconds() / 86400.0
            if gap_days >= pattern_d_gap_days:
                # Score scales weakly with gap up to 5+ years
                score = 1.0 + min(2.0, gap_days / 365.0)
                chore.signals.append(CrossChainSignal(
                    kind="pattern_d_gap",
                    primary_event={"l2_first_seen": d_row[1]},
                    correlated_event={"mainnet_first_tx": mainnet_first_tx},
                    score=score,
                    rationale=f"mainnet identity {gap_days:.0f} days before L2 first-seen",
                ))
        except (ValueError, AttributeError):
            pass

    return chore


def scan_recent_deployers(
    conn: sqlite3.Connection,
    since: str = "2026-03-01",
    min_score: float = 2.0,
) -> list[OperatorChoreography]:
    """Find all watchlisted/OLI deployers with recent deploys, run detection on each."""
    addresses = list(fetch_watchlisted_or_oli_addresses(conn).keys())
    recent = [a for a in addresses if conn.execute(
        """SELECT 1 FROM contracts WHERE deployer_address=?
           AND detection_timestamp >= ? LIMIT 1""",
        (a, since)
    ).fetchone()]
    sys.stderr.write(f"  scanning {len(recent)} watchlisted/OLI addresses with deploys since {since}\n")

    chores = []
    for addr in recent:
        c = detect_for_address(conn, addr, since=datetime.fromisoformat(since + "T00:00:00+00:00"))
        if c.aggregate_score() >= min_score:
            chores.append(c)
    chores.sort(key=lambda c: -c.aggregate_score())
    return chores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--address", default=None,
                    help="check a single address rather than scan")
    ap.add_argument("--min-score", type=float, default=2.0)
    ap.add_argument("--window-hours", type=int, default=6)
    ap.add_argument("--persist", action="store_true",
                    help="write findings to sai_alerts table")
    args = ap.parse_args()

    conn_mode = "rw" if args.persist else "ro"
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode={conn_mode}", uri=True)
    try:
        if args.address:
            c = detect_for_address(conn, args.address.lower())
            label = c.watchlist_label or "(off-watchlist)"
            oli = c.oli_severity or "(no OLI)"
            print(f"\n{c.address}")
            print(f"  watchlist={label}  oli={oli}")
            print(f"  mainnet_first_tx={c.mainnet_first_tx}")
            print(f"  deploys_by_chain={dict(c.deploys_by_chain)}")
            print(f"  aggregate_score={c.aggregate_score():.2f}")
            print(f"  signals:")
            for s in c.signals:
                print(f"    [{s.kind}] +{s.score:.1f} — {s.rationale}")
                if s.primary_event:
                    print(f"      primary:    {s.primary_event}")
                if s.correlated_event:
                    print(f"      correlated: {s.correlated_event}")
            return 0

        print(f"\nCross-chain choreography detector — answers Q-005")
        print(f"  Window since: {args.since}")
        print(f"  Bridge correlation window: ±{args.window_hours}h")
        print(f"  Min aggregate score: {args.min_score}\n")
        chores = scan_recent_deployers(conn, since=args.since, min_score=args.min_score)
        print(f"  Detected {len(chores)} operators with cross-chain choreography signals:\n")
        for c in chores[:30]:
            label = c.watchlist_label or "(off-watchlist)"
            if c.oli_severity:
                label += f" / OLI:{c.oli_severity}"
            print(f"  score={c.aggregate_score():>5.1f}  {c.address}  {label[:60]}")
            print(f"    chains: {dict(c.deploys_by_chain)}  mainnet_first_tx={c.mainnet_first_tx}")
            for s in c.signals[:3]:
                print(f"      [{s.kind}] {s.rationale}")

        if args.persist:
            from surveillance.sai.sai_alerts import AlertRow, write_alerts
            rows = []
            for c in chores:
                # Severity tiers for cross-chain: T1 if any bridge_event signal,
                # T2 if multi-chain deploys with >=2 chains, T3 if Pattern D only
                kinds = {s.kind for s in c.signals}
                if "bridge_event_same_address" in kinds:
                    sev = "T1_BRIDGE_CORRELATION"
                elif "multi_chain_deploys" in kinds:
                    sev = "T2_MULTI_CHAIN_DEPLOY"
                else:
                    sev = "T3_PATTERN_D"
                rows.append(AlertRow(
                    detector="Q-005",
                    severity=sev,
                    subject_address=c.address,
                    subject_kind="deployer",
                    payload={
                        "aggregate_score": c.aggregate_score(),
                        "watchlist_label": c.watchlist_label,
                        "oli_severity": c.oli_severity,
                        "mainnet_first_tx": c.mainnet_first_tx,
                        "deploys_by_chain": dict(c.deploys_by_chain),
                        "signal_count": len(c.signals),
                        "signal_kinds": list(kinds),
                    },
                ))
            n = write_alerts(conn, rows)
            print(f"\n  persisted {n} of {len(rows)} choreography alerts to sai_alerts")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
