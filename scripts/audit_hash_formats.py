"""Data-integrity audit: hash-format consistency in transaction_events.

Finding: transaction_events.tx_hash has MIXED 0x-prefix formats
(99.1% bare 64-char, 0.9% 0x-prefixed 66-char). Any exact-match join
on tx_hash silently fails for the minority format. This audit
characterizes WHICH ingest path writes the 0x variant so the root
cause can be fixed, and quantifies the blast radius (drain detection,
Phase 0 backfill, bot tracking).
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
c = sqlite3.connect(DB)


def main():
    print("=" * 72)
    print("HASH FORMAT AUDIT — transaction_events.tx_hash")
    print("=" * 72)

    total = c.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0]
    with0x = c.execute("SELECT COUNT(*) FROM transaction_events WHERE tx_hash LIKE '0x%'").fetchone()[0]
    print(f"\nTotal tx_events: {total:,}")
    print(f"  0x-prefixed: {with0x:,} ({100*with0x/total:.2f}%)")
    print(f"  bare:        {total-with0x:,} ({100*(total-with0x)/total:.2f}%)")

    # Which chain?
    print("\n0x-prefixed rows by chain:")
    for r in c.execute("SELECT chain, COUNT(*) FROM transaction_events WHERE tx_hash LIKE '0x%' GROUP BY chain ORDER BY 2 DESC"):
        print(f"    {r[0] or 'NULL':12s}: {r[1]:,}")
    print("bare rows by chain:")
    for r in c.execute("SELECT chain, COUNT(*) FROM transaction_events WHERE tx_hash NOT LIKE '0x%' GROUP BY chain ORDER BY 2 DESC"):
        print(f"    {r[0] or 'NULL':12s}: {r[1]:,}")

    # Time window of the 0x rows (by block_number range as proxy + any ts col)
    cols = [r[1] for r in c.execute("PRAGMA table_info(transaction_events)")]
    print(f"\ntransaction_events columns: {cols}")
    if "timestamp" in cols:
        print("\n0x-prefixed rows: timestamp range")
        r = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM transaction_events WHERE tx_hash LIKE '0x%'").fetchone()
        print(f"    {r[0]} -> {r[1]}")
        print("0x-prefixed rows by day:")
        for row in c.execute("""SELECT DATE(timestamp) d, COUNT(*) FROM transaction_events
                                 WHERE tx_hash LIKE '0x%' GROUP BY d ORDER BY 2 DESC LIMIT 12"""):
            print(f"    {row[0]}: {row[1]:,}")

    # bot_tag correlation — is it a specific monitor?
    if "bot_tag" in cols:
        print("\n0x-prefixed rows by bot_tag (top 8):")
        for r in c.execute("""SELECT bot_tag, COUNT(*) FROM transaction_events
                               WHERE tx_hash LIKE '0x%' GROUP BY bot_tag ORDER BY 2 DESC LIMIT 8"""):
            print(f"    {str(r[0])[:30]:30s}: {r[1]:,}")

    # function_selector correlation
    if "function_selector" in cols:
        print("\n0x-prefixed rows by function_selector (top 8):")
        for r in c.execute("""SELECT function_selector, COUNT(*) FROM transaction_events
                               WHERE tx_hash LIKE '0x%' GROUP BY function_selector ORDER BY 2 DESC LIMIT 8"""):
            print(f"    {str(r[0])[:20]:20s}: {r[1]:,}")

    # Are the 0x rows ALSO duplicated as bare? (same tx ingested twice in 2 formats)
    print("\nDuplicate check: is a 0x-row's bare-equivalent also present?")
    dupe = c.execute("""
        SELECT COUNT(*) FROM transaction_events a
        WHERE a.tx_hash LIKE '0x%'
        AND EXISTS (SELECT 1 FROM transaction_events b
                    WHERE b.tx_hash = SUBSTR(a.tx_hash,3))
    """).fetchone()[0]
    print(f"    0x rows whose bare-equivalent ALSO exists (double-ingest): {dupe:,}")

    print("\n" + "=" * 72)
    print("BLAST RADIUS")
    print("=" * 72)

    # Phase 0 completeness: drains mapping to reverted tx under normalized join
    drain_rows = c.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1 AND drain_tx_hash IS NOT NULL").fetchone()[0]
    print(f"\napproval_watchlist drain_detected=1 with tx_hash: {drain_rows:,}")

    # How many drains reference a tx that ONLY exists in 0x form (Phase 0 exact-join would miss)?
    only_0x = c.execute("""
        SELECT COUNT(*) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM transaction_events te WHERE te.tx_hash = aw.drain_tx_hash)
        AND EXISTS (SELECT 1 FROM transaction_events te WHERE te.tx_hash = ('0x' || aw.drain_tx_hash))
    """).fetchone()[0]
    print(f"  drains matchable ONLY via 0x-prefixed tx (Phase 0 exact-join missed): {only_0x:,}")

    # Normalized phantom + reverted recount
    phantom_norm = c.execute("""
        SELECT COUNT(*) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM transaction_events te
                        WHERE REPLACE(te.tx_hash,'0x','') = REPLACE(aw.drain_tx_hash,'0x',''))
    """).fetchone()[0]
    reverted_norm = c.execute("""
        SELECT COUNT(*) FROM approval_watchlist aw
        WHERE aw.drain_detected=1 AND aw.drain_tx_hash IS NOT NULL
        AND EXISTS (SELECT 1 FROM transaction_events te
                    WHERE REPLACE(te.tx_hash,'0x','') = REPLACE(aw.drain_tx_hash,'0x','') AND te.is_reverted=1)
    """).fetchone()[0]
    print(f"  [normalized] drains with NO matching tx at all: {phantom_norm:,}")
    print(f"  [normalized] drains mapping to a REVERTED tx (Phase 0 residue): {reverted_norm:,}")


if __name__ == "__main__":
    main()
