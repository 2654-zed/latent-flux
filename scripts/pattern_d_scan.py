"""Pattern D — Cross-chain reputation import.

Hypothesis: a deployer appears on Arbitrum or Optimism for the first time
with no prior on-chain history in our corpus, but the SAME address has
substantial history on Ethereum mainnet or Base that we haven't linked.

Approach:
1. Pull top 100 recently-appearing HIGH/CRITICAL risk deployers on
   Arbitrum + Optimism (by `deployers.first_seen >= 2026-04-01`,
   ordered by confirmed/suspected count).
2. For each, check via Etherscan free API whether the address has
   substantial activity on Ethereum mainnet.
3. Flag cross-chain imports: address has ≥ 100 Ethereum-mainnet txs
   with a `first_tx` date significantly earlier than our first-seen.

Etherscan free tier: 5 req/sec, 100k req/day. 100 addresses × 1 call each
= 100 calls, well within limits.

Returns a JSON summary suitable for inclusion in the report.
"""
import json, os, sqlite3, time, urllib.request, urllib.parse, sys

DB = "/app/surveillance/data/surveillance.db"
ETHERSCAN_V2 = "https://api.etherscan.io/v2/api"
API_KEY = os.environ.get("ETHERSCAN_V2_KEY") or os.environ.get("ARBISCAN_API_KEY", "")
SLEEP_SEC = 0.3  # 3.3 req/sec, under the 5/sec free-tier limit


def etherscan_first_tx(addr: str, chainid: int = 1) -> dict:
    """Return {'first_block': int|None, 'first_ts': iso|None} for the given
    address on the given chain. Uses Etherscan v2 multichain endpoint."""
    params = {
        "chainid": str(chainid),
        "module": "account",
        "action": "txlist",
        "address": addr,
        "startblock": "0",
        "endblock": "99999999",
        "page": "1",
        "offset": "1",
        "sort": "asc",
    }
    if API_KEY:
        params["apikey"] = API_KEY
    url = f"{ETHERSCAN_V2}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"err": f"fetch_failed: {e}"}
    # Etherscan API
    if data.get("status") == "0" and "No transactions" in (data.get("message") or ""):
        return {"count": 0, "first_block": None, "first_ts": None}
    if data.get("status") != "1":
        return {"err": f"api_msg: {data.get('message')}"}
    txs = data.get("result") or []
    if not txs:
        return {"count": 0, "first_block": None, "first_ts": None}
    first = txs[0]
    first_block = int(first.get("blockNumber", 0))
    first_ts = first.get("timeStamp", "")
    from datetime import datetime, timezone
    try:
        ts_iso = datetime.fromtimestamp(int(first_ts), tz=timezone.utc).isoformat()
    except Exception:
        ts_iso = None
    return {
        "first_block": first_block,
        "first_ts": ts_iso,
    }


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    # Top 100 recently-appearing risky deployers on arb/opt
    rows = c.execute(
        """SELECT d.deployer_address, d.first_seen, d.chain,
                  SUM(CASE WHEN c.confidence_tier='confirmed' THEN 1 ELSE 0 END) as conf,
                  SUM(CASE WHEN c.confidence_tier='suspected' THEN 1 ELSE 0 END) as susp,
                  COUNT(*) as total_contracts
           FROM deployers d
           JOIN contracts c ON c.deployer_address = d.deployer_address
           WHERE d.first_seen >= '2026-04-01'
             AND c.chain IN ('arbitrum', 'optimism')
           GROUP BY d.deployer_address
           HAVING conf >= 1 OR susp >= 3
           ORDER BY conf DESC, susp DESC
           LIMIT 100"""
    ).fetchall()
    print(f"high-risk recently-appearing deployers (arb/opt): {len(rows)}")

    cross_chain = []
    for i, r in enumerate(rows, 1):
        dep = r["deployer_address"]
        our_first = r["first_seen"]

        if i % 10 == 1:
            print(f"  [{i}/{len(rows)}] probing {dep[:18]}...")

        result = etherscan_first_tx(dep, chainid=1)
        time.sleep(SLEEP_SEC)

        eth_first = result.get("first_ts")
        if "err" in result:
            continue

        verdict = ""
        if eth_first and eth_first < our_first:
            # Has mainnet activity BEFORE we saw them on L2
            verdict = "CROSS_CHAIN_IMPORT"
        elif eth_first is None:
            verdict = "no_mainnet_activity"
        else:
            verdict = "mainnet_activity_after_l2"

        cross_chain.append({
            "deployer": dep,
            "our_first_seen": our_first,
            "our_chain": r["chain"],
            "confirmed": r["conf"],
            "suspected": r["susp"],
            "total": r["total_contracts"],
            "eth_first_tx": eth_first,
            "verdict": verdict,
        })

    print()
    print("=== Pattern D candidates (mainnet activity PRE-DATES our L2 first-seen) ===")
    imports = [x for x in cross_chain if x["verdict"] == "CROSS_CHAIN_IMPORT"]
    print(f"count: {len(imports)} of {len(cross_chain)} probed")
    for x in imports:
        gap_days = ""
        try:
            from datetime import datetime
            a = datetime.fromisoformat(x["eth_first_tx"].replace("Z","+00:00"))
            b = datetime.fromisoformat(x["our_first_seen"].replace("Z","+00:00"))
            gap_days = f"{(b-a).days}d"
        except Exception:
            pass
        print(f"  {x['deployer']}  our_chain={x['our_chain']}  "
              f"mainnet_first={x['eth_first_tx'][:10]}  l2_first={x['our_first_seen'][:10]}  "
              f"gap={gap_days}  conf={x['confirmed']}  susp={x['suspected']}")

    # No mainnet footprint (possible new identity or fresh wallet)
    print()
    print("=== Deployers with NO mainnet history (consistent with fresh-identity pattern) ===")
    fresh = [x for x in cross_chain if x["verdict"] == "no_mainnet_activity"]
    print(f"count: {len(fresh)}")
    for x in fresh[:15]:
        print(f"  {x['deployer']}  conf={x['confirmed']}  susp={x['suspected']}  total={x['total']}")

    # Summary JSON for the report
    summary = {
        "total_probed": len(cross_chain),
        "cross_chain_imports": len(imports),
        "fresh_identities": len(fresh),
        "mainnet_post_l2": sum(1 for x in cross_chain if x["verdict"] == "mainnet_activity_after_l2"),
    }
    print()
    print("Summary:", summary)


if __name__ == "__main__":
    main()
