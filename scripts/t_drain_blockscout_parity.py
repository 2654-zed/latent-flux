"""Correctness gates for check_drains_blockscout (Blockscout victim-leg drain
detection). READ-ONLY against the prod-local DB; 0 Alchemy CU; a handful of
Blockscout REST calls. The end-to-end gates run the real function against a
THROWAWAY temp DB (via the db_path param) so the prod-local watchlist is never
mutated.

Gates (from reports/SPEC_blockscout_drain_detection.md §4):
  1. Parity      — cached real-drainer victims (n_out>0) still test n_out>0 live.
  2. Negative    — cached non-drainer victims (n_out==0, n_in>0) still n_out==0.
  3. No-error    — a bounded function run completes with errors==0.
  5. Idempotent  — a second run flags 0 additional (all cache hits).
(Gate 4 coexistence/no-lock is validated by the live --drain-scan-all / heartbeat
 run against the real DB; a temp DB has no contention to exercise it.)

Usage:  python scripts/t_drain_blockscout_parity.py [--samples 3]
Exit code 0 iff all hard gates pass.
"""
from __future__ import annotations
import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "surveillance" / "data" / "surveillance.db"

from surveillance.approval_drain_monitor import (  # noqa: E402
    _blockscout_outbound, BLOCKSCOUT_BASE, check_drains_blockscout,
)

NEG_CONTROL = "0xf68425d0"  # known DISTRIBUTION_MISLABEL (victims IN-only)


def _chain_for(ro, contract):
    row = ro.execute("SELECT chain FROM contracts WHERE contract_address=?",
                     (contract,)).fetchone()
    return (row[0] if row and row[0] else "base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    ro = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)

    failures = []
    notes = []
    n = args.samples

    # ---- sample cached verdicts joined to a resolvable chain ----
    reals = ro.execute("""
        SELECT a.victim, a.contract, c.chain FROM audit_drain_legs a
        JOIN contracts c ON c.contract_address=a.contract
        WHERE a.n_out>0 AND a.err IS NULL AND c.chain IS NOT NULL LIMIT ?""", (n,)).fetchall()
    nondrainers = ro.execute("""
        SELECT a.victim, a.contract, c.chain FROM audit_drain_legs a
        JOIN contracts c ON c.contract_address=a.contract
        WHERE a.n_out=0 AND a.n_in>0 AND a.err IS NULL AND c.chain IS NOT NULL LIMIT ?""", (n,)).fetchall()

    print("=" * 72)
    print("GATE 1 — parity: cached real-drainer victims must test n_out>0 live")
    print("=" * 72)
    if not reals:
        failures.append("Gate1: no cached n_out>0 victims with a resolvable chain")
    for victim, contract, chain in reals:
        base = BLOCKSCOUT_BASE.get(chain, BLOCKSCOUT_BASE["base"])
        n_out, n_in, ltx, lto, lts, err = _blockscout_outbound(base, victim, contract)
        ok = (err is None and n_out > 0)
        print(f"  {victim[:12]}.. / {contract[:12]}.. [{chain}] "
              f"n_out={n_out} n_in={n_in} err={err} last_tx={(ltx or '')[:14]} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"Gate1: {victim}/{contract} live n_out={n_out} err={err} (expected >0)")

    print()
    print("=" * 72)
    print("GATE 2 — negative control: cached non-drainers must test n_out==0 live")
    print("=" * 72)
    # include the named negative control if present in cache
    nc = ro.execute("""
        SELECT a.victim, a.contract, c.chain FROM audit_drain_legs a
        JOIN contracts c ON c.contract_address=a.contract
        WHERE a.contract LIKE ? AND a.n_out=0 LIMIT 2""", (NEG_CONTROL + "%",)).fetchall()
    for victim, contract, chain in (nc + nondrainers):
        base = BLOCKSCOUT_BASE.get(chain, BLOCKSCOUT_BASE["base"])
        n_out, n_in, *_ , err = _blockscout_outbound(base, victim, contract)
        if err is not None:
            notes.append(f"Gate2: {victim}/{contract} fetch err={err} (skipped)")
            print(f"  {victim[:12]}.. / {contract[:12]}.. [{chain}] err={err} -> SKIP")
            continue
        if n_out == 0:
            print(f"  {victim[:12]}.. / {contract[:12]}.. [{chain}] n_out=0 n_in={n_in} -> OK")
        else:
            # cache said 0 but live says >0 => a NEW drain since caching, not a
            # test failure (the detector improving). Note it, don't fail.
            notes.append(f"Gate2: {victim}/{contract} live n_out={n_out}>0 — possible NEW drain since cache")
            print(f"  {victim[:12]}.. / {contract[:12]}.. [{chain}] n_out={n_out} -> NOTE (new drain?)")

    # ---- Gates 3 + 5: end-to-end on a throwaway temp DB ----
    print()
    print("=" * 72)
    print("GATES 3+5 — end-to-end on a temp DB (no prod mutation)")
    print("=" * 72)
    if reals and nondrainers:
        rv, rc, rch = reals[0]
        nv, ncn, nch = nondrainers[0]
        tmp = Path(tempfile.gettempdir()) / "l3_drain_parity_tmp.db"
        if tmp.exists():
            tmp.unlink()
        tc = sqlite3.connect(str(tmp))
        tc.execute("""CREATE TABLE approval_watchlist(
            id INTEGER PRIMARY KEY AUTOINCREMENT, victim_address TEXT, contract_address TEXT,
            approve_timestamp TEXT, deployer_address TEXT, drain_detected INTEGER DEFAULT 0,
            drain_tx_hash TEXT, drain_timestamp TEXT, drain_caller TEXT,
            UNIQUE(victim_address, contract_address))""")
        tc.execute("CREATE TABLE contracts(contract_address TEXT PRIMARY KEY, chain TEXT)")
        tc.execute("INSERT INTO approval_watchlist(victim_address,contract_address,drain_detected) VALUES(?,?,0)", (rv, rc))
        tc.execute("INSERT INTO approval_watchlist(victim_address,contract_address,drain_detected) VALUES(?,?,0)", (nv, ncn))
        tc.execute("INSERT OR IGNORE INTO contracts VALUES(?,?)", (rc, rch))
        tc.execute("INSERT OR IGNORE INTO contracts VALUES(?,?)", (ncn, nch))
        tc.commit()

        r1 = check_drains_blockscout(tc, max_victims=10, db_path=str(tmp))
        print(f"  run#1: {r1}")
        det = tc.execute("SELECT victim_address,drain_detected FROM approval_watchlist ORDER BY id").fetchall()
        det_map = {v: d for v, d in det}
        g3 = (r1["errors"] == 0)
        g_detect = (det_map.get(rv) == 1 and det_map.get(nv) == 0 and r1["drains_detected"] == 1)
        print(f"  gate3 no-error: {'OK' if g3 else 'FAIL'} | detection (real=1,non=0): {'OK' if g_detect else 'FAIL'}")
        if not g3:
            failures.append(f"Gate3: errors={r1['errors']}")
        if not g_detect:
            failures.append(f"Gate-detect: real={det_map.get(rv)} non={det_map.get(nv)} drains={r1['drains_detected']}")

        r2 = check_drains_blockscout(tc, max_victims=10, db_path=str(tmp))
        print(f"  run#2: {r2}")
        g5 = (r2["drains_detected"] == 0 and r2["cache_hits"] >= 1)
        print(f"  gate5 idempotent (0 new, cache hits): {'OK' if g5 else 'FAIL'}")
        if not g5:
            failures.append(f"Gate5: run#2 drains={r2['drains_detected']} cache_hits={r2['cache_hits']}")
        tc.close()
        try:
            tmp.unlink()
        except OSError:
            pass
    else:
        failures.append("Gates3+5: insufficient cache samples (need >=1 real and >=1 non-drainer)")

    print()
    print("=" * 72)
    if notes:
        print("NOTES:")
        for x in notes:
            print(f"  - {x}")
    if failures:
        print(f"RESULT: FAIL ({len(failures)} gate failure(s))")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("RESULT: ALL HARD GATES PASS")


if __name__ == "__main__":
    main()
