"""Validate the tx-INITIATOR drain discriminator before rebuilding the detector
(Correction #29). 0 Alchemy CU (Blockscout REST only).

Discriminator: a (victim, contract) is a REAL approval-drain iff the victim has
>=1 outbound ERC-20 Transfer of the contract token whose TX was initiated by
someone OTHER than the victim (tx.from != victim) -- i.e. a third-party
transferFrom. A victim-initiated swap/transfer (tx.from == victim) is a SALE,
NOT a drain. This is the gate the shipped n_out>0 method was missing.

Checks:
  NEGATIVE control -- OFC (0x752c5a95) approvers (known DEX sellers): the
    discriminator MUST score ~0 drains.
  POSITIVE/RATE -- a stratified random sample of pending approvers: report the
    true drain rate per tier and show example real drains (drainer + tx).

Usage: python scripts/validate_drain_initiator.py [--neg 25 --pos 50]
"""
from __future__ import annotations
import argparse, json, sqlite3, time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
BASES = {"base": "https://base.blockscout.com/api/v2",
         "arbitrum": "https://arbitrum.blockscout.com/api/v2",
         "optimism": "https://optimism.blockscout.com/api/v2"}
OFC = "0x752c5a95d202972e124390f30a50154409d3c858"
MAXLEGS = 12      # outbound legs checked per victim (newest-first)
MAXPAGES = 4
SLEEP = 0.1


def get(u):
    try:
        with urlopen(Request(u, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}), timeout=25) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"_err": str(e)}


def drain_check(base, victim, contract):
    """Return (verdict, drainer, collector, drain_tx, n_out, err).
    verdict in {'DRAIN','SALE','NONE'}. DRAIN iff an outbound leg's tx.from != victim."""
    vlow, clow = victim.lower(), contract.lower()
    url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}"
    legs = []  # (tx_hash, to) newest-first
    pages = 0
    while url and pages < MAXPAGES and len(legs) < MAXLEGS:
        d = get(url)
        if not isinstance(d, dict) or d.get("_err"):
            return ("ERR", None, None, None, len(legs), (d or {}).get("_err", "err"))
        for it in d.get("items", []):
            if ((it.get("from") or {}).get("hash", "") or "").lower() == vlow:
                tok = ((it.get("token") or {}).get("address_hash") or "").lower()
                if tok and tok != clow:
                    continue
                legs.append((it.get("transaction_hash") or it.get("tx_hash"),
                             ((it.get("to") or {}).get("hash", "") or "")))
        npp = d.get("next_page_params")
        if not npp:
            break
        url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}&{urlencode(npp)}"
        pages += 1
        time.sleep(SLEEP)
    if not legs:
        return ("NONE", None, None, None, 0, None)
    for txh, to in legs[:MAXLEGS]:
        if not txh:
            continue
        t = get(f"{base}/transactions/{txh}")
        if not isinstance(t, dict) or t.get("_err"):
            continue
        ini = ((t.get("from") or {}).get("hash", "") or "").lower()
        if ini and ini != vlow:
            return ("DRAIN", ini, to, txh, len(legs), None)
        time.sleep(SLEEP)
    return ("SALE", None, None, None, len(legs), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", type=int, default=25)
    ap.add_argument("--pos", type=int, default=50)
    args = ap.parse_args()
    ro = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    ro.row_factory = sqlite3.Row

    print("=" * 70)
    print(f"NEGATIVE CONTROL — {args.neg} OFC sellers (must score ~0 DRAIN)")
    print("=" * 70)
    negs = ro.execute("SELECT victim_address v FROM approval_watchlist WHERE contract_address=? LIMIT ?",
                      (OFC, args.neg)).fetchall()
    nd = defaultdict(int)
    for r in negs:
        verdict, drainer, col, txh, nout, err = drain_check(BASES["base"], r["v"], OFC)
        nd[verdict] += 1
        time.sleep(SLEEP)
    print("  verdicts:", dict(nd))
    print(f"  => DRAIN among OFC sellers: {nd['DRAIN']}/{sum(nd.values())} (expect ~0)")

    print()
    print("=" * 70)
    print(f"TRUE DRAIN RATE — {args.pos}/tier stratified random sample")
    print("=" * 70)
    examples = []
    for tier in ("confirmed", "suspected", "unanalyzed"):
        rows = ro.execute("""SELECT aw.victim_address v, aw.contract_address c, COALESCE(cc.chain,'base') ch
            FROM approval_watchlist aw LEFT JOIN contracts cc ON cc.contract_address=aw.contract_address
            WHERE aw.drain_detected=0 AND cc.confidence_tier=? ORDER BY RANDOM() LIMIT ?""", (tier, args.pos)).fetchall()
        vd = defaultdict(int)
        for r in rows:
            base = BASES.get(r["ch"], BASES["base"])
            verdict, drainer, col, txh, nout, err = drain_check(base, r["v"], r["c"])
            vd[verdict] += 1
            if verdict == "DRAIN" and len(examples) < 12:
                examples.append((tier, r["v"], r["c"], drainer, txh))
            time.sleep(SLEEP)
        cl = vd["DRAIN"] + vd["SALE"]
        print(f"  {tier:<11}: {dict(vd)}  => drain rate {100*vd['DRAIN']/cl:.0f}% ({vd['DRAIN']}/{cl})" if cl else f"  {tier}: {dict(vd)}")
    print()
    print("example REAL drains (victim -> drainer / tx):")
    for tier, v, c, drainer, txh in examples:
        print(f"  [{tier}] {v[:12]}.. on {c[:12]}.. drainer={(drainer or '')[:12]}.. tx={(txh or '')[:16]}")
    if not examples:
        print("  (none found in sample — real approval-drains are rare in this watchlist)")


if __name__ == "__main__":
    main()
