"""Verify the OLI drain-suppression gate is not hiding real drains /
compromised named wallets. READ-ONLY on the DB; Blockscout only (0 Alchemy CU).

Two parts:
  1. Cross-reference all OLI-suppressed deployers (severity HIGH/LOW) against the
     contracts table + drain rows — is any of them actually adversarial? (The
     oli_labels table is the known-broken priority-#22 table: tag_count=0,
     entity=NULL, so suppression rests on severity alone with no real tags.)
  2. For the contract(s) that actually intersect pending suppressed drains,
     sample approvers and measure COLLECTOR CONCENTRATION of their outbound
     token legs. Diverse destinations => legitimate holders (suppression OK).
     Many victims -> few collectors => drain fingerprint (suppression hides a
     real drain / possible compromise) -> escalate.

Usage: python scripts/verify_oli_suppression.py [--sample 400]
"""
from __future__ import annotations
import argparse, json, sqlite3, sys, time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "surveillance" / "data" / "surveillance.db"
BASE = {"base": "https://base.blockscout.com/api/v2",
        "arbitrum": "https://arbitrum.blockscout.com/api/v2",
        "optimism": "https://optimism.blockscout.com/api/v2"}
SLEEP = 0.12
MAX_PAGES = 10


def outbound_collectors(base, victim, contract):
    """Return (n_out, collectors_list, err). collectors = to-addresses of
    transfers where from==victim and token==contract (the drain recipients)."""
    url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}"
    n_out = pages = 0
    collectors = []
    vlow, clow = victim.lower(), contract.lower()
    while url and pages < MAX_PAGES:
        req = Request(url, headers={"Accept": "application/json",
                                    "User-Agent": "Mozilla/5.0 (L3-oli-verify)"})
        try:
            with urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode())
        except HTTPError as e:
            if e.code == 404:
                return n_out, collectors, None
            return n_out, collectors, f"http{e.code}"
        except (URLError, TimeoutError, OSError):
            return n_out, collectors, "neterr"
        for it in d.get("items", []):
            tok = ((it.get("token") or {}).get("address_hash") or "").lower()
            if tok and tok != clow:
                continue
            frm = ((it.get("from") or {}).get("hash") or "").lower()
            to = ((it.get("to") or {}).get("hash") or "").lower()
            if frm == vlow:
                n_out += 1
                if to:
                    collectors.append(to)
        npp = d.get("next_page_params")
        if not npp:
            return n_out, collectors, None
        url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}&{urlencode(npp)}"
        pages += 1
        time.sleep(SLEEP)
    return n_out, collectors, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()
    ro = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    ro.row_factory = sqlite3.Row

    sup = [r[0] for r in ro.execute(
        "SELECT address FROM oli_labels WHERE severity IN ('HIGH','LOW')")]

    print("=" * 74)
    print("PART 1 — OLI-suppressed deployers cross-referenced against contracts")
    print("=" * 74)
    print(f"{'deployer':<44} {'sev':<5} {'#contracts':>10} {'tiers':<22} {'drain=1':>8} {'pend':>7}")
    for dep in sup:
        sev = ro.execute("SELECT severity FROM oli_labels WHERE address=?", (dep,)).fetchone()[0]
        rows = ro.execute("SELECT confidence_tier, COUNT(*) n FROM contracts WHERE deployer_address=? GROUP BY confidence_tier", (dep,)).fetchall()
        ncon = sum(r["n"] for r in rows)
        tiers = ",".join(f"{r['confidence_tier']}:{r['n']}" for r in rows) or "(none)"
        d1 = ro.execute("SELECT COUNT(*) FROM approval_watchlist WHERE deployer_address=? AND drain_detected=1", (dep,)).fetchone()[0]
        pend = ro.execute("SELECT COUNT(*) FROM approval_watchlist WHERE deployer_address=? AND drain_detected=0", (dep,)).fetchone()[0]
        flag = "  <-- has pending suppressed" if pend > 0 else ""
        print(f"{dep:<44} {sev:<5} {ncon:>10} {tiers:<22} {d1:>8} {pend:>7}{flag}")

    # Which contracts actually carry the pending suppressed drains?
    print()
    print("Contracts with pending OLI-suppressed drains:")
    ph = ",".join("?" * len(sup))
    targets = ro.execute(f"""
        SELECT aw.contract_address, c.chain, c.confidence_tier, COUNT(*) victims
        FROM approval_watchlist aw LEFT JOIN contracts c ON c.contract_address=aw.contract_address
        WHERE aw.drain_detected=0 AND aw.deployer_address IN ({ph})
        GROUP BY aw.contract_address ORDER BY victims DESC""", sup).fetchall()
    for r in targets:
        print(f"  {r['contract_address']} [{r['chain']}] tier={r['confidence_tier']} victims={r['victims']:,}")

    # ---- PART 2: collector concentration on the target contract(s) ----
    print()
    print("=" * 74)
    print(f"PART 2 — collector-concentration test (sample={args.sample} per contract)")
    print("=" * 74)
    for r in targets:
        contract = r["contract_address"]
        chain = r["chain"] or "base"
        base = BASE.get(chain, BASE["base"])
        victims = [x[0] for x in ro.execute(f"""
            SELECT aw.victim_address FROM approval_watchlist aw
            WHERE aw.drain_detected=0 AND aw.contract_address=? AND aw.deployer_address IN ({ph})
            ORDER BY aw.approve_timestamp LIMIT ?""", (contract, *sup, args.sample)).fetchall()]
        print(f"\ncontract {contract} [{chain}] — sampling {len(victims)} approvers")
        n_drained = errs = 0
        collector_victims = Counter()  # collector -> # distinct victims sending to it
        total_legs = 0
        for i, v in enumerate(victims):
            n_out, collectors, err = outbound_collectors(base, v, contract)
            if err:
                errs += 1
                continue
            if n_out > 0:
                n_drained += 1
                total_legs += n_out
                for c in set(collectors):
                    collector_victims[c] += 1
            time.sleep(SLEEP)
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(victims)} drained={n_drained} errs={errs}")
        checked = len(victims) - errs
        print(f"  RESULT: sampled={len(victims)} checked={checked} errs={errs}")
        print(f"          victims with outbound leg (n_out>0): {n_drained} "
              f"({(100*n_drained/checked if checked else 0):.1f}% of checked)")
        print(f"          total outbound legs: {total_legs}; distinct collectors: {len(collector_victims)}")
        if n_drained:
            topc = collector_victims.most_common(10)
            top3_share = sum(n for _, n in collector_victims.most_common(3)) / n_drained
            print(f"          collector concentration: distinct={len(collector_victims)} "
                  f"vs draining-victims={n_drained}  (top-3 collectors cover {100*top3_share:.1f}% of draining victims)")
            print("          top collectors (collector -> # draining victims):")
            for c, n in topc:
                print(f"            {c}  <- {n} victims")
            verdict = ("DRAIN FINGERPRINT (concentrated -> suppression hides a real drain / possible compromise)"
                       if top3_share >= 0.5 and n_drained >= 10
                       else "DIVERSE destinations -> consistent with legitimate holder activity (suppression OK)")
            print(f"          VERDICT: {verdict}")
        else:
            print("          VERDICT: no outbound legs in sample -> approvers not drained (suppression OK)")


if __name__ == "__main__":
    main()
