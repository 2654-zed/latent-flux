"""Kelp retrospective Phase 4 — attack recipient funding trace.

Traces 0x8B1b...0D3b backward via alchemy_getAssetTransfers to identify:
- When the address was first funded
- The source of initial funding (Tornado Cash per public reporting)
- Any interaction with monitored chains (Arbitrum / Base / Optimism) pre-attack

Budget: 20 RPC calls max. Should need only ~2-5 (one per relevant call type
plus a lookup on the funder if present).
"""
import json
import os
import urllib.request

RECIPIENT = "0x8B1b6c9A6DB1304000412dd21Ae6A70a82d60D3b"

# Ethereum mainnet RPC (Alchemy)
ETH_RPC = os.environ.get("ETH_HTTP_URL")
if not ETH_RPC:
    wss = os.environ.get("ETH_WSS_URL", "")
    ETH_RPC = wss.replace("wss://", "https://") if wss else None

TORNADO_CASH_ETH_ROUTER = "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b"
# Common Tornado Cash fixed-amount pools (0.1 / 1 / 10 / 100 ETH)
TORNADO_POOLS = {
    "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc": "Tornado 0.1 ETH",
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936": "Tornado 1 ETH",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf": "Tornado 10 ETH",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291": "Tornado 100 ETH",
}


def rpc(method, params):
    req = urllib.request.Request(
        ETH_RPC, method="POST",
        data=json.dumps({"jsonrpc":"2.0","method":method,"params":params,"id":1}).encode(),
        headers={"Content-Type":"application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def main():
    if not ETH_RPC:
        print("ERROR: no ETH RPC")
        return 1

    print(f"=== Kelp Phase 4 — attack recipient funding trace ===")
    print(f"target: {RECIPIENT}")
    print()

    # 1. Current balance + tx count
    bal_resp = rpc("eth_getBalance", [RECIPIENT, "latest"])
    eth_bal = int(bal_resp["result"], 16) / 1e18
    nonce_resp = rpc("eth_getTransactionCount", [RECIPIENT, "latest"])
    nonce = int(nonce_resp["result"], 16)
    print(f"current ETH balance: {eth_bal:.4f}")
    print(f"nonce (outbound txs): {nonce}")

    # 2. Earliest INBOUND transfers — this tells us when + how the address was funded
    inbound = rpc("alchemy_getAssetTransfers", [{
        "fromBlock": "0x0",
        "toBlock": "latest",
        "toAddress": RECIPIENT,
        "category": ["external", "erc20", "internal"],
        "order": "asc",
        "maxCount": "0xa",  # 10
        "withMetadata": True,
    }])
    transfers_in = inbound.get("result", {}).get("transfers", [])
    print(f"\n--- earliest inbound transfers ({len(transfers_in)}) ---")
    for t in transfers_in:
        ts = (t.get("metadata") or {}).get("blockTimestamp", "?")
        src = t.get("from", "?").lower()
        src_label = TORNADO_POOLS.get(src, "")
        if src.lower() == TORNADO_CASH_ETH_ROUTER:
            src_label = "Tornado Cash Router"
        blk = int(t.get("blockNum", "0x0"), 16)
        val = t.get("value")
        cat = t.get("category")
        asset = t.get("asset", "?")
        tag = f"  [{src_label}]" if src_label else ""
        print(f"  {ts[:19]}  block={blk}  {cat:8s}  from={src}  {val} {asset}{tag}")

    # 3. If first inbound source is in TORNADO_POOLS or Tornado router, verify
    tornado_match = False
    if transfers_in:
        first_src = transfers_in[0].get("from", "").lower()
        if first_src in TORNADO_POOLS or first_src == TORNADO_CASH_ETH_ROUTER:
            tornado_match = True
    print(f"\nFirst inbound from Tornado Cash: {tornado_match}")

    # 4. Outbound history — count unique destinations, look for monitored chain touches
    # The recipient is on Ethereum; its outbound activity IS the drain consumption.
    outbound = rpc("alchemy_getAssetTransfers", [{
        "fromBlock": "0x0",
        "toBlock": "latest",
        "fromAddress": RECIPIENT,
        "category": ["external", "erc20", "internal"],
        "order": "asc",
        "maxCount": "0x64",  # 100
        "withMetadata": True,
    }])
    transfers_out = outbound.get("result", {}).get("transfers", [])
    print(f"\n--- outbound transfers from recipient ({len(transfers_out)}) ---")
    destinations = set()
    for t in transfers_out[:20]:
        ts = (t.get("metadata") or {}).get("blockTimestamp", "?")
        dst = t.get("to", "?").lower()
        destinations.add(dst)
        blk = int(t.get("blockNum", "0x0"), 16)
        val = t.get("value")
        cat = t.get("category")
        asset = t.get("asset", "?")
        print(f"  {ts[:19]}  block={blk}  {cat:8s}  to={dst}  {val} {asset}")
    print(f"\nunique destinations (first 20 shown, {len(transfers_out)} total): {len(destinations)}")


if __name__ == "__main__":
    raise SystemExit(main())
