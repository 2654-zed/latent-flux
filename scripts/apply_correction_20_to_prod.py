"""Apply Correction #20 dispositions to production surveillance.db.

Run against the production DB after sync/access is restored. This script is
idempotent — running it twice produces no additional changes (it checks for
the [CORRECTION #20] marker in watch_reason before modifying).

Usage (once production access is restored):
    python scripts/apply_correction_20_to_prod.py --db /app/surveillance/data/surveillance.db

What this does:
- Deactivates 10 watchlist rows where OLI tags identify the address as a CEX hot
  wallet, bridge, payment processor, or institutional deployer.
- Appends [CORRECTION #20] notes to 5 watchlist rows kept active (4 LOW-confidence
  project deployers pending second-source verification + 1 OLI-confirmed THORChain
  router).
- Updates 6 (or 7) rows in infrastructure_operator_candidates with status=
  'rejected_oli_correction_20'.

See `reports/correction_log.md` Correction #20 for the full rationale.
See `reports/blockscout_tag_audit_2026-05-09.csv` for the source audit.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


DEACTIVATIONS = [
    ('0x3304e22ddaa22bcdc5fca2269b418046ae7b566a', 'Binance 73 / Exchange'),
    ('0x39591e7c099a379fd7b349ebfecaeef439c40454', 'OKX 177 / Exchange'),
    ('0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda', 'OKX 137 / Exchange'),
    ('0xbaed383ede0e5d9d72430661f3285daa77e9439f', 'Bybit: Hot Wallet 6 / Exchange'),
    ('0xf70da97812cb96acdf810712aa562db8dfa3dbef', 'Relay: Solver / Relay Bridge'),
    ('0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb', 'Binance: Internal 2 / Exchange'),
    ('0x151b381058f91cf871e7ea1ee83c45326f61e96d', 'MoonPay 4 / Exchange'),
    ('0x45a318273749d6eb00f5f6ca3bc7cd3de26d642a', 'Owlto Finance: Bridge'),
    ('0xe4edb277e41dc89ab076a1f049f4a3efa700bce8', 'Orbiter Finance: Bridge 2'),
    ('0xbb50ce87be3443ed137df1dfdbf2fb0ca8c0a9e0', 'Circle: contract deployer'),
]

NOTE_ONLY = [
    ('0xd37bbe5744d730a1d98d8dc97c42f0ca46ad7146', 'THORChain: Router v4.1.1 (OLI confirms existing classification)'),
    ('0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8', 'Animoca: Deployer (LOW-confidence — pending second source)'),
    ('0xa2a01b4a68575280a2de45178e289da717bedb6f', 'Stabilize Finance: Deployer 2 (LOW-confidence — pending second source)'),
    ('0x147b8869655bc09f226955cc676ff78efe240ca8', 'Luchadores: Deployer (LOW-confidence — pending second source)'),
    ('0xc5d133296e17ba25df0409a6c31607bf3b78e3e3', 'CryptoCauses: Deployer (LOW-confidence — pending second source)'),
]

INFRA_REJECT = [
    '0x3304e22ddaa22bcdc5fca2269b418046ae7b566a',  # Binance
    '0x39591e7c099a379fd7b349ebfecaeef439c40454',  # OKX 177
    '0x4e3ae00e8323558fa5cac04b152238924aa31b60',  # MEXC
    '0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda',  # OKX 137
    '0xbaed383ede0e5d9d72430661f3285daa77e9439f',  # Bybit
    '0xf70da97812cb96acdf810712aa562db8dfa3dbef',  # Relay
    '0x80c67432656d59144ceff962e8faf8926599bcf8',  # Orbiter
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--db", required=True, help="Path to surveillance.db")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without modifying DB")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    marker = "[CORRECTION #20"

    n_deactivated = n_skipped = n_noted = n_infra = 0
    for addr, oli_label in DEACTIVATIONS:
        row = c.execute(
            "SELECT id, watch_reason, active FROM watchlist WHERE LOWER(address)=?",
            (addr.lower(),)
        ).fetchone()
        if row is None:
            continue
        id_, reason, active = row
        if reason and marker in reason:
            n_skipped += 1
            print(f"  SKIP (already marked): {addr}")
            continue
        new_reason = (reason or '') + f' [CORRECTION #20 (2026-05-09): OLI-tagged as {oli_label}]'
        if not args.dry_run:
            c.execute("UPDATE watchlist SET active=0, watch_reason=?, last_seen_date=? WHERE id=?",
                      (new_reason, now, id_))
        print(f"  DEACTIVATE: {addr} -> {oli_label}")
        n_deactivated += 1

    for addr, oli_label in NOTE_ONLY:
        row = c.execute(
            "SELECT id, watch_reason FROM watchlist WHERE LOWER(address)=?",
            (addr.lower(),)
        ).fetchone()
        if row is None:
            continue
        id_, reason = row
        if reason and marker in reason:
            n_skipped += 1
            print(f"  SKIP (already marked): {addr}")
            continue
        new_reason = (reason or '') + f' [CORRECTION #20 (2026-05-09): OLI-tagged as {oli_label}]'
        if not args.dry_run:
            c.execute("UPDATE watchlist SET watch_reason=?, last_seen_date=? WHERE id=?",
                      (new_reason, now, id_))
        print(f"  NOTE: {addr} -> {oli_label}")
        n_noted += 1

    for addr in INFRA_REJECT:
        row = c.execute(
            "SELECT notes, status FROM infrastructure_operator_candidates "
            "WHERE LOWER(funder_address)=?",
            (addr.lower(),)
        ).fetchone()
        if row is None:
            continue
        notes, status = row
        if status == 'rejected_oli_correction_20':
            n_skipped += 1
            print(f"  SKIP (already rejected): {addr}")
            continue
        new_notes = (notes or '') + ' [CORRECTION #20 (2026-05-09): OLI-tagged institutional, see correction log]'
        if not args.dry_run:
            c.execute(
                "UPDATE infrastructure_operator_candidates "
                "SET status='rejected_oli_correction_20', notes=?, last_checked=? "
                "WHERE LOWER(funder_address)=?",
                (new_notes, now, addr.lower())
            )
        print(f"  INFRA_REJECT: {addr}")
        n_infra += 1

    if not args.dry_run:
        conn.commit()
    conn.close()

    print(f"\nSummary: deactivated={n_deactivated}, noted={n_noted}, "
          f"infra-rejected={n_infra}, skipped-already-applied={n_skipped}")
    if args.dry_run:
        print("(dry-run — no changes written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
