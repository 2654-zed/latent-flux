"""Correction to the 2026-04-18 c0ffeefeed flag.

Original flag claimed "2 activations 2026-04-18" as the Attack-2 signal.
Direct DB inspection found only 1 lifetime DORMANT_ACTIVATION alert for
this deployer (2026-04-04), not today. My earlier activity-pull output
was misread. Correcting the note to reflect the accurate state:

- Last deploy by c0ffeefeed: 2026-04-07 (11 days quiet)
- 209 total contracts, 100 confirmed + 109 suspected + 0 unknown
- 48% confirmed rate — classifier has caught most of the fleet
- Not a "waking dormant fleet" — a trap operator whose contracts keep
  getting tested by bots after the operator stopped deploying

Priority tag stays (any future deploy from this address auto-SUSPECTED
via the existing velocity-escalation path), but the descriptive text
should not reference a false Attack-2 signal.
"""
import sqlite3
import time

DB = "/app/surveillance/data/surveillance.db"
DEPLOYER = "0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e"

REPLACEMENT_TAG = (
    "WATCH-2026-04-18 (corrected): 209-contract trap fleet. Last deploy "
    "2026-04-07 (11-day deploy gap). Fleet 48% confirmed / 52% suspected; "
    "classifier has caught most of it. Priority flag kept so any future "
    "deploy from this address auto-SUSPECTED via velocity path. NOT a "
    "dormant fleet candidate in the Attack-2 sense — the dormancy is at "
    "the deployer level, not at the fleet-ready-to-activate level."
)
OLD_TAG_PREFIX = "WATCH-2026-04-18: 209-contract dormant fleet"


def main():
    c = sqlite3.connect(DB, timeout=60)
    c.execute("PRAGMA busy_timeout = 60000")

    row = c.execute(
        "SELECT deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
        (DEPLOYER,),
    ).fetchone()
    if row is None:
        print("no deployer row")
        return 1
    existing = row[0] or ""
    if OLD_TAG_PREFIX not in existing:
        print("old tag not present — nothing to correct")
        return 0

    # Strip the old tag and prepend the corrected one
    # Find the "; " that separates old tag from prior content
    idx = existing.find(OLD_TAG_PREFIX)
    # Try to find the end of the old tag (next "; " after the tag start)
    # Use the fact that we wrote the old tag followed by "; " + prior notes
    after_tag_idx = existing.find("; ", idx)
    if after_tag_idx != -1:
        remainder = existing[after_tag_idx + 2:]
    else:
        remainder = ""
    new_notes = (REPLACEMENT_TAG + ("; " + remainder if remainder else ""))[:500]

    for attempt in range(5):
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                "UPDATE deployers SET deployment_pattern_notes = ? "
                "WHERE deployer_address = ?",
                (new_notes, DEPLOYER),
            )
            c.commit()
            print("corrected")
            break
        except sqlite3.OperationalError as e:
            try:
                c.rollback()
            except sqlite3.Error:
                pass
            print(f"attempt {attempt+1} locked: {e}")
            time.sleep(5)
    row = c.execute(
        "SELECT deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
        (DEPLOYER,),
    ).fetchone()
    print(f"post-correction notes: {row[0][:400]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
