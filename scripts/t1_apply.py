"""Task 1 APPLY — per-row OUT-leg reconciliation of the 45 migrated
drain-tainted contracts. 0 Alchemy CU (Blockscout only). Dry-run default.

Per-VICTIM test (not per-contract): a (victim, contract) drain row is REAL
iff the victim has >=1 outbound ERC-20 Transfer (from==victim) of the
contract token, per Blockscout address token-transfer history. Token key
is item.token.address_hash (confirmed by t1_probe_shape).

Reuses the audit_drain_legs cache from t1_decode_full (470 victims already
checked); fetches only the victims not yet cached (uncapped → full set).

Actions on --apply:
  1. RESET every drain row whose victim has NO outbound leg (phantom):
     drain_detected=0, null drain_tx_hash/timestamp/caller. Preserves the
     approval row itself.
  2. RESTORE contract unanalyzed->confirmed iff it retains >=2 real-victim
     drain rows after the reset (genuine drainer wrongly migrated by
     Correction #25). Annotate, preserve prior reason. Single-real-victim
     and zero-real contracts stay unanalyzed (flagged).

CLI:
  python scripts/t1_apply.py                      # dry-run local (full fetch)
  python scripts/t1_apply.py --apply              # apply local
  python scripts/t1_apply.py --db <path> [--apply]
"""
from __future__ import annotations
import argparse, json, sqlite3, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
BASE = {"base": "https://base.blockscout.com/api/v2",
        "arbitrum": "https://arbitrum.blockscout.com/api/v2",
        "optimism": "https://optimism.blockscout.com/api/v2"}
SLEEP = 0.12
MAX_PAGES = 10
RESTORE_MIN_REAL = 2

RESTORE_ANNOT = (
    "[RESTORED 2026-06-01 / Correction #27 Finding 4: on-chain-verified drainer "
    "({rv} victims with outbound token legs via Blockscout, 0 Alchemy CU). The "
    "Correction #25 migration to unanalyzed was a FALSE NEGATIVE. Prior reason follows.] | "
)


def get(url):
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0 (L3-apply)"})
    with urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def victim_has_outbound(base, victim, contract):
    """(n_out, n_in, err). n_out>0 => real drained victim."""
    url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}"
    n_out = n_in = pages = 0
    while url and pages < MAX_PAGES:
        try:
            d = get(url)
        except HTTPError as e:
            if e.code == 404:
                return n_out, n_in, None
            return n_out, n_in, f"http{e.code}"
        except (URLError, TimeoutError, OSError):
            return n_out, n_in, "neterr"
        for it in d.get("items", []):
            tok = ((it.get("token") or {}).get("address_hash") or "").lower()
            if tok and tok != contract.lower():
                continue
            frm = ((it.get("from") or {}).get("hash") or "").lower()
            to = ((it.get("to") or {}).get("hash") or "").lower()
            if frm == victim.lower():
                n_out += 1
            elif to == victim.lower():
                n_in += 1
        npp = d.get("next_page_params")
        if not npp:
            return n_out, n_in, None
        url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}&{urlencode(npp)}"
        pages += 1
        time.sleep(SLEEP)
    return n_out, n_in, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "surveillance" / "data" / "surveillance.db"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_drain_legs(
        victim TEXT, contract TEXT, n_out INTEGER, n_in INTEGER,
        truncated INTEGER, err TEXT, checked_at TEXT, PRIMARY KEY (victim, contract))""")
    conn.commit()

    mig = [r[0] for r in conn.execute(
        "SELECT contract_address FROM contracts WHERE confidence_tier='unanalyzed' "
        "AND confidence_reason LIKE '%Correction #25%'")]
    # (victim, contract) drain rows for migrated contracts
    pairs = conn.execute(f"""
        SELECT DISTINCT a.victim_address, a.contract_address, c.chain
        FROM approval_watchlist a JOIN contracts c ON c.contract_address=a.contract_address
        WHERE a.drain_detected=1 AND a.victim_address IS NOT NULL
          AND c.confidence_tier='unanalyzed' AND c.confidence_reason LIKE '%Correction #25%'""").fetchall()
    sys.stderr.write(f"migrated contracts: {len(mig)}; drain (victim,contract) pairs: {len(pairs)}\n")

    # resolve each pair's outbound status (cache-first)
    real = defaultdict(int)      # contract -> #real victims
    phantom_victims = []         # (victim, contract) to reset
    errs = 0
    cache_hits = fetched = 0
    for i, (victim, contract, chain) in enumerate(pairs):
        row = conn.execute("SELECT n_out, err FROM audit_drain_legs WHERE victim=? AND contract=?", (victim, contract)).fetchone()
        if row is not None and row[1] is None:
            n_out = row[0]; cache_hits += 1
        else:
            base = BASE.get(chain, BASE["base"])
            n_out, n_in, er = victim_has_outbound(base, victim, contract)
            conn.execute("INSERT OR REPLACE INTO audit_drain_legs VALUES (?,?,?,?,?,?,?)",
                         (victim, contract, n_out, n_in, 0, er, NOW))
            conn.commit(); fetched += 1; time.sleep(SLEEP)
            if er:
                errs += 1
                continue  # on error: do NOT treat as phantom (skip, leave row)
        if n_out and n_out > 0:
            real[contract] += 1
        else:
            phantom_victims.append((victim, contract))
        if (i + 1) % 200 == 0:
            sys.stderr.write(f"  {i+1}/{len(pairs)} cache={cache_hits} fetched={fetched} errs={errs}\n")

    restore = [c for c in mig if real.get(c, 0) >= RESTORE_MIN_REAL]
    keep = [c for c in set(x[1] for x in pairs) if real.get(c, 0) < RESTORE_MIN_REAL]

    L = []
    def p(s=""): L.append(str(s))
    p("=" * 76)
    p(f"T1 APPLY ({'APPLY' if args.apply else 'DRY-RUN'})  db={args.db}")
    p("=" * 76)
    p(f"drain pairs evaluated: {len(pairs)}  (cache={cache_hits} fetched={fetched} errs={errs})")
    p(f"phantom rows to reset: {len(phantom_victims)}")
    p(f"contracts to RESTORE (>= {RESTORE_MIN_REAL} real victims): {len(restore)}")
    p(f"contracts staying unanalyzed (<{RESTORE_MIN_REAL} real): {len(keep)}")
    p("")
    p("real-victim count per contract (post per-row test):")
    for c in sorted(real, key=lambda c: -real[c]):
        p(f"    {c}  real={real[c]}  -> {'RESTORE' if real[c] >= RESTORE_MIN_REAL else 'keep'}")

    if args.apply:
        nreset = 0
        for victim, contract in phantom_victims:
            cur = conn.execute("""UPDATE approval_watchlist SET drain_detected=0, drain_tx_hash=NULL,
                drain_timestamp=NULL, drain_caller=NULL
                WHERE victim_address=? AND contract_address=? AND drain_detected=1""", (victim, contract))
            nreset += cur.rowcount
        nrest = 0
        for contract in restore:
            row = conn.execute("SELECT confidence_tier, confidence_reason FROM contracts WHERE contract_address=?", (contract,)).fetchone()
            if not row or row[0] == "confirmed":
                continue
            annot = RESTORE_ANNOT.format(rv=real[contract])
            conn.execute("UPDATE contracts SET confidence_tier='confirmed', confidence_reason=?, last_updated=? WHERE contract_address=?",
                         (annot + (row[1] or ""), NOW, contract))
            nrest += 1
        conn.commit()
        p("")
        p(f"APPLIED: reset {nreset} phantom drain rows; restored {nrest} contracts to confirmed.")
        p(f"  post confirmed: {conn.execute(chr(0x53)+'ELECT COUNT(*) FROM contracts WHERE confidence_tier=' + repr('confirmed')).fetchone()[0]}")
        p(f"  post drain_detected=1: {conn.execute('SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1').fetchone()[0]}")
    else:
        p("")
        p("(DRY-RUN — no mutations. Re-run with --apply.)")

    out = ROOT / "reports" / "_t1_apply.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
