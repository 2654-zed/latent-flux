"""Flag c0ffeefeed-209-fleet deployer for priority watch.

Adds a WATCH tag to deployers.deployment_pattern_notes. This makes
`is_priority_deployer()` return True for the deployer, which means any
NEW deployment from this address will be auto-SUSPECTED via the
velocity-escalation path in deployment_monitor.

For the 209 already-dormant contracts, the note is advisory only —
human reviewers see the tag when inspecting the deployer; existing
dormant contracts are not retroactively upgraded (which matches the
Correction #4 inheritance-breadth discipline: tier upgrades need
matching evidence, and 'deployer is on a watchlist' is not enough
Tier A for a bulk upgrade).

Idempotent: only appends if the WATCH tag isn't already present.
"""
import sqlite3
import time

DB = "/app/surveillance/data/surveillance.db"
DEPLOYER = "0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e"

WATCH_TAG = (
    "WATCH-2026-04-18: 209-contract dormant fleet. 2 activations 2026-04-18 "
    "(08:28, 19:00 UTC). Attack-2 (dormant fleet + proxy upgrade) candidate per "
    "compositional zero-days catalog. Escalate if >=3 activations in 48h window. "
    "See scripts/fleet_activation_watch.py for current state."
)


def main():
    c = sqlite3.connect(DB, timeout=60)
    c.execute("PRAGMA busy_timeout = 60000")

    row = c.execute(
        "SELECT deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
        (DEPLOYER,),
    ).fetchone()
    if row is None:
        print(f"NO DEPLOYER ROW for {DEPLOYER}")
        return 1

    existing = row[0] or ""
    print(f"existing pattern_notes: {existing[:200]!r}")

    if "WATCH-2026-04-18" in existing:
        print("watch tag already present — no change")
        return 0

    new_notes = WATCH_TAG + (("; " + existing) if existing else "")
    new_notes = new_notes[:500]  # cap for schema safety

    for attempt in range(5):
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                "UPDATE deployers SET deployment_pattern_notes = ? "
                "WHERE deployer_address = ?",
                (new_notes, DEPLOYER),
            )
            c.commit()
            print("updated pattern_notes")
            break
        except sqlite3.OperationalError as e:
            try:
                c.rollback()
            except sqlite3.Error:
                pass
            print(f"attempt {attempt+1} locked: {e}")
            time.sleep(5)
    else:
        print("failed to acquire lock after retries")
        return 2

    row = c.execute(
        "SELECT deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
        (DEPLOYER,),
    ).fetchone()
    print(f"post-update notes: {row[0][:300]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
