"""Apply Correction #20 dispositions to production via railway ssh.

This is the SSH-dispatched companion to `scripts/apply_correction_20_to_prod.py`.
The local-execution version is for running against any DB path with argparse;
this version specifically targets the production container's DB at
/app/surveillance/data/surveillance.db via railway ssh.

Why a separate script: the local version is 6.5 KB, which when base64-encoded
and wrapped exceeds Windows cmd.exe's 8191-character command-line limit. This
version is hand-compacted to fit (~3 KB → ~4.1 KB bootstrap). It must stay
functionally equivalent to apply_correction_20_to_prod.py — if you change the
DEACTIVATIONS / NOTE_ONLY / INFRA_REJECT lists in one, change them here too.

Idempotency: the remote script checks for `[CORRECTION #20` marker in
watch_reason / status before re-applying, so re-running this is a no-op when
production is already corrected.

Precondition: railway CLI must be linked to blockchain@stellar-embrace via
`railway link --project blockchain && railway service stellar-embrace`.

Usage:
    python scripts/apply_correction_20_via_ssh.py
    python scripts/apply_correction_20_via_ssh.py --verify-only

Verified 2026-05-10: applied successfully against production with output
"SUMMARY deact=10 note=5 infra=6 skip=0".
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys


# Compact apply script body — kept under ~3.5 KB so the base64-encoded
# bootstrap stays under cmd.exe's 8191-char command-line limit.
APPLY_SCRIPT = '''import sqlite3,sys
from datetime import datetime,timezone
DB="/app/surveillance/data/surveillance.db"
DEACT=[("0x3304e22ddaa22bcdc5fca2269b418046ae7b566a","Binance 73"),("0x39591e7c099a379fd7b349ebfecaeef439c40454","OKX 177"),("0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda","OKX 137"),("0xbaed383ede0e5d9d72430661f3285daa77e9439f","Bybit Hot Wallet 6"),("0xf70da97812cb96acdf810712aa562db8dfa3dbef","Relay Solver"),("0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb","Binance Internal 2"),("0x151b381058f91cf871e7ea1ee83c45326f61e96d","MoonPay 4"),("0x45a318273749d6eb00f5f6ca3bc7cd3de26d642a","Owlto Finance Bridge"),("0xe4edb277e41dc89ab076a1f049f4a3efa700bce8","Orbiter Finance Bridge 2"),("0xbb50ce87be3443ed137df1dfdbf2fb0ca8c0a9e0","Circle contract deployer")]
NOTE=[("0xd37bbe5744d730a1d98d8dc97c42f0ca46ad7146","THORChain Router v4.1.1 (OLI confirms)"),("0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8","Animoca Deployer (LOW pending 2nd src)"),("0xa2a01b4a68575280a2de45178e289da717bedb6f","Stabilize Finance Deployer 2 (LOW pending 2nd src)"),("0x147b8869655bc09f226955cc676ff78efe240ca8","Luchadores Deployer (LOW pending 2nd src)"),("0xc5d133296e17ba25df0409a6c31607bf3b78e3e3","CryptoCauses Deployer (LOW pending 2nd src)")]
INFRA=["0x3304e22ddaa22bcdc5fca2269b418046ae7b566a","0x39591e7c099a379fd7b349ebfecaeef439c40454","0x4e3ae00e8323558fa5cac04b152238924aa31b60","0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda","0xbaed383ede0e5d9d72430661f3285daa77e9439f","0xf70da97812cb96acdf810712aa562db8dfa3dbef","0x80c67432656d59144ceff962e8faf8926599bcf8"]
M="[CORRECTION #20"
conn=sqlite3.connect(DB);c=conn.cursor();now=datetime.now(timezone.utc).isoformat()
nd=nn=ni=ns=0
for a,l in DEACT:
 r=c.execute("SELECT id,watch_reason FROM watchlist WHERE LOWER(address)=?",(a.lower(),)).fetchone()
 if not r:continue
 if r[1] and M in r[1]:ns+=1;print(f"SKIP D {a}");continue
 c.execute("UPDATE watchlist SET active=0,watch_reason=?,last_seen_date=? WHERE id=?",((r[1] or "")+f" [CORRECTION #20 (2026-05-09): OLI-tagged as {l}]",now,r[0]));nd+=1;print(f"DEACT {a} -> {l}")
for a,l in NOTE:
 r=c.execute("SELECT id,watch_reason FROM watchlist WHERE LOWER(address)=?",(a.lower(),)).fetchone()
 if not r:continue
 if r[1] and M in r[1]:ns+=1;print(f"SKIP N {a}");continue
 c.execute("UPDATE watchlist SET watch_reason=?,last_seen_date=? WHERE id=?",((r[1] or "")+f" [CORRECTION #20 (2026-05-09): OLI-tagged as {l}]",now,r[0]));nn+=1;print(f"NOTE {a} -> {l}")
for a in INFRA:
 r=c.execute("SELECT notes,status FROM infrastructure_operator_candidates WHERE LOWER(funder_address)=?",(a.lower(),)).fetchone()
 if not r:continue
 if r[1]=="rejected_oli_correction_20":ns+=1;print(f"SKIP I {a}");continue
 c.execute("UPDATE infrastructure_operator_candidates SET status='rejected_oli_correction_20',notes=?,last_checked=? WHERE LOWER(funder_address)=?",((r[0] or "")+" [CORRECTION #20 (2026-05-09): OLI-tagged institutional]",now,a.lower()));ni+=1;print(f"INFRA {a}")
conn.commit();conn.close()
print(f"SUMMARY deact={nd} note={nn} infra={ni} skip={ns}")
'''


VERIFY_SCRIPT = '''import sqlite3
c=sqlite3.connect("/app/surveillance/data/surveillance.db").cursor()
print("=== Watchlist sample ===")
for a in ["0xbb50ce87be3443ed137df1dfdbf2fb0ca8c0a9e0","0x3304e22ddaa22bcdc5fca2269b418046ae7b566a","0xf70da97812cb96acdf810712aa562db8dfa3dbef","0x151b381058f91cf871e7ea1ee83c45326f61e96d"]:
 r=c.execute("SELECT priority,active,(watch_reason LIKE \\'%[CORRECTION #20%\\') FROM watchlist WHERE LOWER(address)=?",(a,)).fetchone()
 print(f"  {a}: priority={r[0]} active={r[1]} marked={bool(r[2]) if r else None}")
r=c.execute("SELECT COUNT(*) FROM infrastructure_operator_candidates WHERE status=\\'rejected_oli_correction_20\\'").fetchone()
print(f"infra rejected: {r[0]}")
r=c.execute("SELECT COUNT(*) FROM watchlist WHERE active=0 AND watch_reason LIKE \\'%[CORRECTION #20%\\'").fetchone()
print(f"deactivated by #20: {r[0]}")
r=c.execute("SELECT COUNT(*) FROM watchlist WHERE active=1 AND watch_reason LIKE \\'%[CORRECTION #20%\\'").fetchone()
print(f"noted-and-active by #20: {r[0]}")
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
    railway, prefix = _resolve_railway()
    r = subprocess.run(prefix + ["status"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "blockchain" not in r.stdout or "stellar-embrace" not in r.stdout:
        raise SystemExit(
            "railway CLI not linked to blockchain@stellar-embrace. Run:\n"
            "  railway link --project blockchain && railway service stellar-embrace"
        )


def _run_remote(script: str) -> tuple[int, str, str]:
    railway, prefix = _resolve_railway()
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    bootstrap = f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\'))"'
    if len(bootstrap) > 8000:
        raise SystemExit(
            f"bootstrap is {len(bootstrap)} chars, exceeds Windows cmd.exe limit (~8191). "
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
        help="Run only the verification probe; do not modify production",
    )
    args = ap.parse_args()

    _check_linked()
    script = VERIFY_SCRIPT if args.verify_only else APPLY_SCRIPT
    print(f"[apply-via-ssh] {'verify' if args.verify_only else 'apply'}: "
          f"script {len(script)} bytes")
    rc, out, err = _run_remote(script)
    print(out)
    if err.strip():
        print("--- stderr ---")
        print(err[-2000:])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
