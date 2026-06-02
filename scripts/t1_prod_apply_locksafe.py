"""Lock-safe prod apply of the Bug #19b reconciliation.

Designed to coexist with the LIVE chain monitors writing to surveillance.db:
  - PRAGMA busy_timeout=60000 (wait up to 60s for writer locks)
  - reads the COMPLETE decode cache (audit_drain_legs) — NO Blockscout
    fetches (the cache is loaded separately from the local export)
  - performs ALL mutations in ONE transaction, committed once at the end,
    minimizing the write-lock window

Decision logic identical to t1_apply.py (verified on local):
  phantom (victim,contract) = n_out <= 0  -> drain_detected reset
  contract restored if >=2 real victims (n_out>0) remain

Dry-run default. --apply to mutate. --db points at prod path.
"""
import argparse, sqlite3, sys
from collections import defaultdict
from datetime import datetime, timezone

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
RESTORE_MIN_REAL = 2
RESTORE_ANNOT = (
    "[RESTORED 2026-06-01 / Correction #27 Finding 4: on-chain-verified drainer "
    "({rv} victims with outbound token legs via Blockscout, 0 Alchemy CU). The "
    "Correction #25 migration to unanalyzed was a FALSE NEGATIVE. Prior reason follows.] | "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")

    # require a complete cache
    legs = {}
    have = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='audit_drain_legs'").fetchone()[0]
    if not have:
        print("ERROR: audit_drain_legs not present — load the cache first."); return 1
    for victim, contract, n_out, err in conn.execute("SELECT victim, contract, n_out, err FROM audit_drain_legs"):
        legs[(victim, contract)] = (n_out, err)
    print(f"cache rows: {len(legs)}")

    # migrated contracts that carry drains
    mig = set(r[0] for r in conn.execute(
        "SELECT contract_address FROM contracts WHERE confidence_tier='unanalyzed' "
        "AND confidence_reason LIKE '%Correction #25%'"))
    pairs = conn.execute("""
        SELECT DISTINCT a.victim_address, a.contract_address
        FROM approval_watchlist a JOIN contracts c ON c.contract_address=a.contract_address
        WHERE a.drain_detected=1 AND a.victim_address IS NOT NULL
          AND c.confidence_tier='unanalyzed' AND c.confidence_reason LIKE '%Correction #25%'""").fetchall()

    real = defaultdict(int)
    phantom = []
    missing = 0
    for victim, contract in pairs:
        rec = legs.get((victim, contract))
        if rec is None or rec[1] is not None:   # not cached, or cached-with-error -> skip (no guess)
            missing += 1
            continue
        if rec[0] and rec[0] > 0:
            real[contract] += 1
        else:
            phantom.append((victim, contract))

    restore = [c for c in mig if real.get(c, 0) >= RESTORE_MIN_REAL]
    print(f"pairs={len(pairs)} phantom={len(phantom)} restore_contracts={len(restore)} "
          f"missing_from_cache={missing}")
    if missing:
        print(f"  WARNING: {missing} pairs not in cache — they will NOT be reset (left as-is).")

    if not args.apply:
        print("(DRY-RUN — add --apply)")
        return 0

    # ---- single transaction ----
    conn.execute("BEGIN")
    nreset = 0
    for victim, contract in phantom:
        cur = conn.execute("""UPDATE approval_watchlist SET drain_detected=0, drain_tx_hash=NULL,
            drain_timestamp=NULL, drain_caller=NULL
            WHERE victim_address=? AND contract_address=? AND drain_detected=1""", (victim, contract))
        nreset += cur.rowcount
    nrest = 0
    for contract in restore:
        row = conn.execute("SELECT confidence_tier, confidence_reason FROM contracts WHERE contract_address=?", (contract,)).fetchone()
        if not row or row[0] == "confirmed":
            continue
        conn.execute("UPDATE contracts SET confidence_tier='confirmed', confidence_reason=?, last_updated=? WHERE contract_address=?",
                     (RESTORE_ANNOT.format(rv=real[contract]) + (row[1] or ""), NOW, contract))
        nrest += 1
    conn.commit()
    print(f"APPLIED: reset {nreset} phantom rows; restored {nrest} contracts.")
    print("post confirmed:", conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier='confirmed'").fetchone()[0])
    print("post drain_detected=1:", conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
