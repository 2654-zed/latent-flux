"""Task 1 decoder (READ-ONLY, 0 Alchemy CU) — full victim-outbound-leg test
across all 45 migrated drain-tainted contracts.

For each (victim, contract) with drain_detected=1, query the victim's
address-level ERC-20 transfer history filtered to the contract token
(Blockscout, free). The row is a REAL drain iff the victim appears as
`from` in >=1 Transfer of that token (i.e. their tokens left their wallet).
IN-only / none => phantom (distribution mislabel or Bug#19b over-credit).

Token address key is item.token.address_hash (confirmed by t1_probe_shape).

Caches every (victim,contract) result in audit_drain_legs so the run is
resumable and a later apply pass reuses it. NO mutation of corpus tables.

Per-contract verdict:
  REAL_DRAINER          real_victims >= 2
  DISTRIBUTION_MISLABEL  real_victims == 0  (all victims IN-only/none)
  MANUAL                 real_victims == 1  (borderline)

Outputs: reports/_t1_decode_full.txt  + .json (per-contract + counts)

CLI: python scripts/t1_decode_full.py [--db PATH] [--max-pages N] [--cap-victims N]
"""
from __future__ import annotations
import argparse, json, sqlite3, sys, time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
BASE = {"base": "https://base.blockscout.com/api/v2",
        "arbitrum": "https://arbitrum.blockscout.com/api/v2",
        "optimism": "https://optimism.blockscout.com/api/v2"}
SLEEP = 0.12


def get(url):
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0 (L3-t1)"})
    with urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def victim_legs(base, victim, contract, max_pages):
    """Return (n_out, n_in, truncated, err). n_out = Transfers with from==victim
    of the contract token; n_in = to==victim."""
    url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}"
    n_out = n_in = 0
    pages = 0
    while url and pages < max_pages:
        try:
            d = get(url)
        except HTTPError as e:
            if e.code == 404:
                return n_out, n_in, False, None
            return n_out, n_in, False, f"http{e.code}"
        except (URLError, TimeoutError, OSError):
            return n_out, n_in, False, "neterr"
        for it in d.get("items", []):
            frm = ((it.get("from") or {}).get("hash") or "").lower()
            to = ((it.get("to") or {}).get("hash") or "").lower()
            tok = ((it.get("token") or {}).get("address_hash") or "").lower()
            if tok and tok != contract.lower():
                continue
            if frm == victim.lower():
                n_out += 1
            elif to == victim.lower():
                n_in += 1
        npp = d.get("next_page_params")
        if not npp:
            return n_out, n_in, False, None
        url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}&{urlencode(npp)}"
        pages += 1
        time.sleep(SLEEP)
    return n_out, n_in, True, None  # truncated at max_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "surveillance" / "data" / "surveillance.db"))
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--cap-victims", type=int, default=0, help="0 = all victims")
    args = ap.parse_args()
    db = Path(args.db)
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_drain_legs(
        victim TEXT, contract TEXT, n_out INTEGER, n_in INTEGER,
        truncated INTEGER, err TEXT, checked_at TEXT,
        PRIMARY KEY (victim, contract))""")
    conn.commit()

    mig = [r[0] for r in conn.execute(
        "SELECT contract_address FROM contracts WHERE confidence_tier='unanalyzed' "
        "AND confidence_reason LIKE '%Correction #25%'")]
    # only those with drains
    contracts = []
    for a in mig:
        vs = [r[0] for r in conn.execute(
            "SELECT DISTINCT victim_address FROM approval_watchlist "
            "WHERE contract_address=? AND drain_detected=1 AND victim_address IS NOT NULL", (a,))]
        if vs:
            contracts.append((a, vs))
    total_victims = sum(len(v) for _, v in contracts)
    sys.stderr.write(f"contracts={len(contracts)} total_victims={total_victims}\n")

    done = 0
    per_contract = []
    for a, victims in contracts:
        chain = (conn.execute("SELECT chain FROM contracts WHERE contract_address=?", (a,)).fetchone() or ["base"])[0]
        base = BASE.get(chain, BASE["base"])
        if args.cap_victims:
            victims = victims[:args.cap_victims]
        real = inonly = none = err = trunc = 0
        for v in victims:
            cached = conn.execute("SELECT n_out, n_in, truncated, err FROM audit_drain_legs WHERE victim=? AND contract=?", (v, a)).fetchone()
            if cached is not None:
                n_out, n_in, tr, er = cached
            else:
                n_out, n_in, tr, er = victim_legs(base, v, a, args.max_pages)
                conn.execute("INSERT OR REPLACE INTO audit_drain_legs VALUES (?,?,?,?,?,?,?)",
                             (v, a, n_out, n_in, 1 if tr else 0, er, "2026-05-27"))
                conn.commit()
                time.sleep(SLEEP)
            if er:
                err += 1
            elif n_out and n_out > 0:
                real += 1
            elif n_in and n_in > 0:
                inonly += 1
            else:
                none += 1
            if tr:
                trunc += 1
            done += 1
            if done % 100 == 0:
                sys.stderr.write(f"  {done}/{total_victims}  ({a[:10]} real={real} in={inonly})\n")
        checked = len(victims)
        if real >= 2:
            verdict = "REAL_DRAINER"
        elif real == 0 and err == 0:
            verdict = "DISTRIBUTION_MISLABEL"
        else:
            verdict = "MANUAL"
        per_contract.append({"contract": a, "chain": chain, "victims_checked": checked,
                             "real": real, "in_only": inonly, "none": none,
                             "err": err, "truncated": trunc, "verdict": verdict})
        sys.stderr.write(f"DONE {a[:12]} {verdict} real={real}/{checked} in={inonly} none={none} err={err}\n")

    vc = Counter(d["verdict"] for d in per_contract)
    L = []
    def p(s=""): L.append(str(s))
    p("=" * 78)
    p(f"T1 FULL DECODE — {len(per_contract)} contracts, {total_victims} victims (0 Alchemy CU)")
    p("=" * 78)
    for k, v in vc.items():
        p(f"  {k:24s}: {v}")
    p("")
    restore_real = sum(d["real"] for d in per_contract if d["verdict"] == "REAL_DRAINER")
    p(f"  real victims on REAL_DRAINER contracts: {restore_real}")
    p(f"  total real victims (any contract):      {sum(d['real'] for d in per_contract)}")
    p("")
    p(f"  {'contract':44s} {'chain':9s} {'chk':>4s} {'real':>5s} {'in':>5s} {'none':>5s} {'err':>4s} {'tr':>4s}  verdict")
    for d in sorted(per_contract, key=lambda d: (d["verdict"], -d["real"])):
        p(f"  {d['contract']} {d['chain']:9s} {d['victims_checked']:>4} {d['real']:>5} "
          f"{d['in_only']:>5} {d['none']:>5} {d['err']:>4} {d['truncated']:>4}  {d['verdict']}")
    out = ROOT / "reports" / "_t1_decode_full.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(per_contract, indent=2), encoding="utf-8")
    print(f"wrote {out}  verdicts={dict(vc)}")


if __name__ == "__main__":
    main()
