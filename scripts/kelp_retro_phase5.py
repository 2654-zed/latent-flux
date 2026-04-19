"""Kelp retrospective Phase 5 — pre-attack anomaly probe on the Ethereum DVN.

Goal: characterize the DVN's on-chain signing activity in the days pre-attack.
A compromised DVN might show anomalous patterns: sudden drop/spike in
signing volume, attestation to unusual messages, dormancy followed by a
single critical attestation.

Budget: 10 RPC calls max.

Approach:
- eth_getTransactionCount(DVN, block_at_t-N) for several t-N values → baseline
  volume over time
- alchemy_getAssetTransfers fromAddress=DVN → sample of outgoing calls
- Does the DVN make frequent outbound txs (legitimate DVN signs on every
  eligible cross-chain message) or is it quiet (suspicious)?
"""
import json, os, urllib.request

ETH_DVN = "0x589dedbd617e0cbcb916a9223f4d1300c294236b"

ETH_RPC = os.environ.get("ETH_HTTP_URL")
if not ETH_RPC:
    wss = os.environ.get("ETH_WSS_URL", "")
    ETH_RPC = wss.replace("wss://", "https://") if wss else None

# Block timestamps around the attack (relative to attack_block=24908285)
# ETH block time ~12s → one day = 7200 blocks
ATTACK_BLOCK = 24908285
PROBE_BLOCKS = [
    ("attack_t-30d (block 24692285)", 24692285),
    ("attack_t-14d (block 24800285)", 24800285),
    ("attack_t-7d  (block 24858285)", 24858285),
    ("attack_t-3d  (block 24886285)", 24886285),
    ("attack_t-1d  (block 24901085)", 24901085),
    ("attack_block (block 24908285)", 24908285),
]


def rpc(method, params):
    req = urllib.request.Request(
        ETH_RPC, method="POST",
        data=json.dumps({"jsonrpc":"2.0","method":method,"params":params,"id":1}).encode(),
        headers={"Content-Type":"application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def main():
    if not ETH_RPC: return 1
    print(f"=== Kelp Phase 5 — Ethereum DVN activity baseline ===")
    print(f"DVN: {ETH_DVN}\n")

    print(f"{'checkpoint':40s}  {'nonce':>8}  delta")
    prev_nonce = None
    for label, blk in PROBE_BLOCKS:
        r = rpc("eth_getTransactionCount", [ETH_DVN, hex(blk)])
        nonce = int(r["result"], 16)
        delta = "" if prev_nonce is None else f"+{nonce - prev_nonce}"
        print(f"  {label:40s}  {nonce:>8}  {delta}")
        prev_nonce = nonce

    # Get the DVN's outbound activity sample — any txs immediately pre-attack?
    print()
    print("--- DVN outbound txs across attack window (last 20 before attack + attack-day) ---")
    resp = rpc("alchemy_getAssetTransfers", [{
        "fromBlock": hex(ATTACK_BLOCK - 50000),  # ~ 1 week before
        "toBlock": hex(ATTACK_BLOCK),
        "fromAddress": ETH_DVN,
        "category": ["external"],
        "order": "desc",
        "maxCount": "0x14",  # 20
        "withMetadata": True,
    }])
    transfers = resp.get("result", {}).get("transfers", [])
    for t in transfers[:20]:
        ts = (t.get("metadata") or {}).get("blockTimestamp", "?")
        blk = int(t.get("blockNum", "0x0"), 16)
        to = t.get("to", "?")
        val = t.get("value")
        asset = t.get("asset", "?")
        print(f"  blk={blk:>10}  {ts[:19]}  to={to}  {val} {asset}")

    print(f"\nDVN outbound tx sample count: {len(transfers)}")


if __name__ == "__main__":
    raise SystemExit(main())
