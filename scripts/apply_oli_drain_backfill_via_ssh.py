"""Apply OLI drain-suppression backfill to production via railway ssh.

Companion script to the local-execution
`python -m surveillance.approval_drain_monitor --backfill-oli-suppression`
introduced 2026-05-10. Resets `drain_detected` (and the associated drain_*
columns) on `approval_watchlist` rows whose contract deployer is OLI-tagged
(severity != 'none') — institutional / project deployers whose contracts
should never have promoted drains via the heuristic detector.

Discovered gap: Animoca-deployed `0x752c5a95...` produced 4,587 phantom
drain rows on local; production carries the same rows. The fix is
non-destructive: only `drain_detected`, `drain_tx_hash`, `drain_timestamp`,
`drain_caller` are cleared. Approval rows remain.

Idempotency: re-running is safe — only `drain_detected=1` rows on the OLI-
tagged set are touched, and the second run finds none.

Precondition: railway CLI linked to blockchain@stellar-embrace.

Usage:
    python scripts/apply_oli_drain_backfill_via_ssh.py --verify-only
    python scripts/apply_oli_drain_backfill_via_ssh.py

After applying, follow with `python scripts/sync_prod_db.py` to refresh
the local DB from the corrected production state.
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess


# OLI-suppressed deployer set, hardcoded because production `oli_labels`
# table is empty (the Correction #20 backfill populated the local cache
# only — see CORRECTIONS.md #20 close note). Once a future commit syncs
# the OLI cache up to prod, this list can be replaced with a SQL
# selection against `oli_labels` directly.
#
# Restricted to HIGH (institutional/CEX/bridge) and LOW (project deployer)
# severities. The 3 `self-confirming` OLI tags (`0x4cfe37…`, `0xa70703…`,
# `0xd90e2f…`) are EXCLUDED because those are correctly-flagged adversarial
# deployers — their drains should remain detected.
OLI_SUPPRESSED = [
    "0x147b8869655bc09f226955cc676ff78efe240ca8",  # LOW — Luchadores
    "0x3304e22ddaa22bcdc5fca2269b418046ae7b566a",  # HIGH — Binance 73
    "0x39591e7c099a379fd7b349ebfecaeef439c40454",  # HIGH — OKX 177
    "0x4e3ae00e8323558fa5cac04b152238924aa31b60",  # HIGH — MEXC 15
    "0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8",  # LOW — Animoca
    "0x80c67432656d59144ceff962e8faf8926599bcf8",  # HIGH — Orbiter Finance
    "0xa2a01b4a68575280a2de45178e289da717bedb6f",  # LOW — Stabilize Finance
    "0xbaed383ede0e5d9d72430661f3285daa77e9439f",  # HIGH — Bybit Hot Wallet 6
    "0xc5d133296e17ba25df0409a6c31607bf3b78e3e3",  # LOW — CryptoCauses
    "0xd37bbe5744d730a1d98d8dc97c42f0ca46ad7146",  # HIGH — THORChain Router v4.1.1
    "0xe4edb277e41dc89ab076a1f049f4a3efa700bce8",  # HIGH — Orbiter Finance Bridge 2
    "0xf70da97812cb96acdf810712aa562db8dfa3dbef",  # HIGH — Relay Solver
    "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9",  # HIGH — Relay + Railgun
]

_S = repr(OLI_SUPPRESSED)

# Compact remote body — kept small so base64-encoded bootstrap stays under
# Windows cmd.exe's 8191-char command-line limit.
APPLY_SCRIPT = f'''import sqlite3
DB="/app/surveillance/data/surveillance.db"
S={_S}
ph=",".join("?"*len(S))
conn=sqlite3.connect(DB,timeout=30);c=conn.cursor()
n=c.execute(f"SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1 AND deployer_address IN ({{ph}})",S).fetchone()[0]
print(f"would reset: {{n}}")
c.execute(f"UPDATE approval_watchlist SET drain_detected=0, drain_tx_hash=NULL, drain_timestamp=NULL, drain_caller=NULL WHERE drain_detected=1 AND deployer_address IN ({{ph}})",S)
conn.commit();conn.close()
print(f"DONE reset={{n}}")
'''


VERIFY_SCRIPT = f'''import sqlite3
DB="/app/surveillance/data/surveillance.db"
S={_S}
ph=",".join("?"*len(S))
c=sqlite3.connect(DB,timeout=30).cursor()
n=c.execute(f"SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1 AND deployer_address IN ({{ph}})",S).fetchone()[0]
print(f"prod drain_detected=1 rows on OLI-suppressed-deployer contracts: {{n}}")
n2=c.execute(f"SELECT COUNT(*) FROM approval_watchlist WHERE deployer_address IN ({{ph}})",S).fetchone()[0]
print(f"prod total approval rows on OLI-suppressed-deployer contracts: {{n2}}")
print(f"--- per-deployer breakdown (drain_detected=1 only) ---")
for d in S:
 r=c.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1 AND deployer_address=?",(d,)).fetchone()[0]
 if r: print(f" {{d}} -> {{r}}")
'''


def _resolve_railway() -> tuple[str, list[str]]:
    railway = (
        shutil.which("railway.exe")
        or shutil.which("railway.cmd")
        or shutil.which("railway")
    )
    if not railway:
        raise SystemExit("railway CLI not found in PATH")
    if os.name == "nt" and railway.lower().endswith((".cmd", ".bat")):
        return railway, ["cmd.exe", "/c", railway]
    return railway, [railway]


def _check_linked() -> None:
    _, prefix = _resolve_railway()
    r = subprocess.run(prefix + ["status"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "blockchain" not in r.stdout or "stellar-embrace" not in r.stdout:
        raise SystemExit(
            "railway CLI not linked to blockchain@stellar-embrace. Run:\n"
            "  railway link --project blockchain && railway service stellar-embrace"
        )


def _run_remote(script: str) -> tuple[int, str, str]:
    _, prefix = _resolve_railway()
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    bootstrap = f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\'))"'
    if len(bootstrap) > 8000:
        raise SystemExit(
            f"bootstrap is {len(bootstrap)} chars, exceeds Windows cmd.exe limit. "
            f"Tighten the embedded script."
        )
    cmd = prefix + ["ssh", bootstrap]
    r = subprocess.run(cmd, capture_output=True, text=False, timeout=120)
    return (
        r.returncode,
        r.stdout.decode("utf-8", errors="replace"),
        r.stderr.decode("utf-8", errors="replace"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--verify-only", action="store_true",
        help="Count affected rows on production; do not modify",
    )
    args = ap.parse_args()

    _check_linked()
    script = VERIFY_SCRIPT if args.verify_only else APPLY_SCRIPT
    print(f"[oli-drain-backfill] {'verify' if args.verify_only else 'apply'}: "
          f"script {len(script)} bytes")
    rc, out, err = _run_remote(script)
    print(out)
    if err.strip():
        print("--- stderr ---")
        print(err[-2000:])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
