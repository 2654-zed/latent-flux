"""OLI temporal validity scorer — answers Q-003.

The OLI guardrail (INV-007) treats an OLI tag as a presence-binary: an
address has a tag or it doesn't, and HIGH-severity tags redirect to
COMMERCIAL/institutional_oli_tagged. The 0x80b12bd0 case (2026-05-15)
showed this is safe-by-accident: the address is a genuine Animoca tag
AND is actively adversarial. A HIGH-severity-tagged compromised address
would have silently bypassed our adversarial classification.

This module produces a staleness verdict for each OLI-tagged address:

    FRESH               — tag is current; institution still controls the key.
                          Behavior is consistent with institutional usage,
                          no adversarial signal.
    STALE               — tag may not reflect current control. Address is
                          on adversarial watchlist, OR is funding known
                          drainers, OR has deployed confirmed-tier traps.
    NEEDS_VERIFICATION  — the tag carries signals in both directions; a
                          human review (or external attestation) is
                          warranted before relying on the tag.

The verdict is consultable by entity_classifier so INV-007 redirects only
on FRESH tags.

Empirical anchors:
  - 0x80b12bd0  (Animoca, LOW) → STALE (adversarial watchlist HIGH,
                                        bait deployed, 4,587 victims drained)
  - 0x80c67432  (Orbiter Bridge, HIGH) → FRESH (legitimate bridge protocol,
                                                 no adversarial behavior)

CLI:
    python -m surveillance.sai.oli_temporal_validity
    python -m surveillance.sai.oli_temporal_validity --address 0x80b12bd0
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"

# Severity tiers — STALE signals are weighted to be load-bearing.
STALENESS_WEIGHTS = {
    "adversarial_watchlist_high": 5.0,    # CRITICAL signal
    "adversarial_watchlist_med":  3.0,
    "funded_known_drainer":       3.0,    # The Q-009 cross-link
    "deployed_confirmed_trap":    3.0,    # Behavioral evidence
    "deployed_suspected_trap":    1.5,
    "is_drain_caller_itself":     4.0,    # Tagged address pulled funds
    "in_entity_classification_criminal": 2.5,
}


@dataclass
class StaleSignal:
    name: str
    weight: float
    detail: str


@dataclass
class OLIValidity:
    address: str
    oli_severity: str
    signals: list[StaleSignal] = field(default_factory=list)

    def aggregate_score(self) -> float:
        return sum(s.weight for s in self.signals)

    def verdict(self) -> str:
        score = self.aggregate_score()
        if score >= 5.0:
            return "STALE"
        if score >= 2.0:
            return "NEEDS_VERIFICATION"
        return "FRESH"

    def recommended_action(self) -> str:
        v = self.verdict()
        if v == "STALE":
            return "DO NOT redirect via OLI guardrail. The tag's temporal validity is broken."
        if v == "NEEDS_VERIFICATION":
            return "Hold redirect pending external attestation (e.g., institution-side key-control proof)."
        return "Redirect via OLI guardrail is safe. No staleness signals fired."


def assess_address(conn: sqlite3.Connection, address: str,
                    oli_severity: str | None = None) -> OLIValidity:
    """Run all staleness signals against an address.

    If oli_severity isn't passed, looks it up in oli_labels.
    """
    if oli_severity is None:
        row = conn.execute(
            "SELECT severity FROM oli_labels WHERE address=?",
            (address,)
        ).fetchone()
        oli_severity = row[0] if row else "(no OLI tag)"

    v = OLIValidity(address=address, oli_severity=oli_severity)

    # Signal 1: adversarial watchlist entry
    wl = conn.execute(
        "SELECT entity_name, priority, watch_reason FROM watchlist WHERE address=? AND active=1",
        (address,)
    ).fetchone()
    if wl:
        priority = (wl[1] or "").upper()
        # Determine if the watchlist entry is adversarial (entity_name suggests trap/drainer)
        name = (wl[0] or "").lower()
        watch_reason = (wl[2] or "").lower()
        adversarial_markers = [
            "drainer", "trap", "scam", "phish", "criminal", "drain_caller",
            "self_deploying", "pristine_solo_operator", "rd_bot", "bot_operator",
            "pristine-reputation", "trap_deployer",
        ]
        is_adversarial = any(m in name or m in watch_reason for m in adversarial_markers)
        if is_adversarial:
            if priority in ("CRITICAL", "HIGH"):
                v.signals.append(StaleSignal(
                    "adversarial_watchlist_high",
                    STALENESS_WEIGHTS["adversarial_watchlist_high"],
                    f"on watchlist {priority}: {wl[0]}",
                ))
            elif priority == "MEDIUM":
                v.signals.append(StaleSignal(
                    "adversarial_watchlist_med",
                    STALENESS_WEIGHTS["adversarial_watchlist_med"],
                    f"on watchlist MEDIUM: {wl[0]}",
                ))

    # Signal 2: is it itself a drain_caller?
    n_drains = conn.execute(
        "SELECT COUNT(*) FROM approval_watchlist WHERE drain_caller=? AND drain_detected=1",
        (address,)
    ).fetchone()[0]
    if n_drains > 0:
        v.signals.append(StaleSignal(
            "is_drain_caller_itself",
            STALENESS_WEIGHTS["is_drain_caller_itself"],
            f"executed {n_drains} drains in corpus",
        ))

    # Signal 3: did it fund any known drainer?
    n_funded_drainers = conn.execute(
        """SELECT COUNT(DISTINCT d.deployer_address)
           FROM deployers d
           JOIN approval_watchlist aw ON aw.drain_caller = d.deployer_address
                                       AND aw.drain_detected = 1
           WHERE d.funding_trail LIKE ?""",
        (f"%{address}%",)
    ).fetchone()[0]
    if n_funded_drainers > 0:
        v.signals.append(StaleSignal(
            "funded_known_drainer",
            STALENESS_WEIGHTS["funded_known_drainer"],
            f"funded {n_funded_drainers} addresses that became drain_callers",
        ))

    # Signal 4: deployed confirmed-tier trap?
    n_confirmed = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE deployer_address=? AND confidence_tier='confirmed'",
        (address,)
    ).fetchone()[0]
    if n_confirmed > 0:
        v.signals.append(StaleSignal(
            "deployed_confirmed_trap",
            STALENESS_WEIGHTS["deployed_confirmed_trap"],
            f"deployed {n_confirmed} confirmed-tier trap contracts",
        ))

    n_suspected = conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE deployer_address=? AND confidence_tier='suspected'",
        (address,)
    ).fetchone()[0]
    if n_suspected > 0:
        v.signals.append(StaleSignal(
            "deployed_suspected_trap",
            STALENESS_WEIGHTS["deployed_suspected_trap"],
            f"deployed {n_suspected} suspected-tier trap contracts",
        ))

    # Signal 5: entity_classification CRIMINAL
    ec = conn.execute(
        "SELECT category, subtype FROM entity_classification WHERE address=? AND category='CRIMINAL'",
        (address,)
    ).fetchone()
    if ec:
        v.signals.append(StaleSignal(
            "in_entity_classification_criminal",
            STALENESS_WEIGHTS["in_entity_classification_criminal"],
            f"entity_classification: {ec[0]}/{ec[1]}",
        ))

    return v


def scan_all_oli(conn: sqlite3.Connection) -> list[OLIValidity]:
    """Run staleness assessment on every address in oli_labels."""
    rows = conn.execute("SELECT address, severity FROM oli_labels").fetchall()
    return [assess_address(conn, addr, sev) for addr, sev in rows]


def fmt_verdict_row(v: OLIValidity) -> str:
    lines = [
        f"  {v.address}  OLI={v.oli_severity}  score={v.aggregate_score():>5.1f}  verdict={v.verdict()}"
    ]
    for s in v.signals:
        lines.append(f"      [+{s.weight:>3.1f}] {s.name}: {s.detail}")
    if not v.signals:
        lines.append("      (no staleness signals — tag treated as FRESH)")
    lines.append(f"      action: {v.recommended_action()}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--address", default=None)
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        if args.address:
            v = assess_address(conn, args.address.lower())
            print(fmt_verdict_row(v))
            return 0 if v.verdict() == "FRESH" else 2
        print("\nOLI temporal validity scan (answers Q-003):")
        results = scan_all_oli(conn)
        # Sort STALE first, then NEEDS_VERIFICATION, then FRESH
        order = {"STALE": 0, "NEEDS_VERIFICATION": 1, "FRESH": 2}
        results.sort(key=lambda v: (order[v.verdict()], -v.aggregate_score()))
        n_stale = sum(1 for v in results if v.verdict() == "STALE")
        n_nv = sum(1 for v in results if v.verdict() == "NEEDS_VERIFICATION")
        n_fresh = sum(1 for v in results if v.verdict() == "FRESH")
        print(f"  Total OLI-tagged addresses: {len(results)}")
        print(f"    STALE:              {n_stale}")
        print(f"    NEEDS_VERIFICATION: {n_nv}")
        print(f"    FRESH:              {n_fresh}\n")
        for v in results:
            print(fmt_verdict_row(v))
            print()
        return 0 if n_stale == 0 else 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
