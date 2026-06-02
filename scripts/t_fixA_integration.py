"""Fix A integration gate — exercise the ACTUAL edited code path on prod.

Imports the real _fetch_block_receipts_http helper + runs one live Base block
(the chain that broke WS) through it, builds the map exactly as
_handle_block_header does, then cross-checks is_reverted vs per-tx receipts
for a sample. Proves the deployed code path (not a standalone reimpl) works
on Base over HTTP.

Run on prod:
  railway ssh "cd /app && python3 /app/scripts/t_fixA_integration.py"
"""
import asyncio, os, sys
sys.path.insert(0, "/app")
from web3 import AsyncWeb3
from web3.providers import WebSocketProvider
from surveillance.deployment_monitor import _fetch_block_receipts_http

BASE_WSS = os.environ.get("BASE_WSS_URL")
HTTP = BASE_WSS.replace("wss://", "https://") if BASE_WSS else None


def rev_from_blockreceipt(r):
    s = r.get("status")
    if isinstance(s, str): return int(s, 16) == 0
    if isinstance(s, int): return s == 0
    return None


async def main():
    if not HTTP:
        print("BASE_WSS_URL unset"); return
    async with AsyncWeb3(WebSocketProvider(BASE_WSS)) as w3:
        latest = await w3.eth.block_number
        bn = latest - 5
        # NEW path exactly as deployment_monitor uses it:
        items = await asyncio.to_thread(_fetch_block_receipts_http, HTTP, bn)
        bmap = {}
        for r in items:
            h = r.get("transactionHash")
            if h:
                bmap[h.lower()] = r
        print(f"base block {bn}: _fetch_block_receipts_http returned {len(items)} receipts, map size {len(bmap)}")
        # cross-check vs per-tx for a sample
        blk = await w3.eth.get_block(bn, full_transactions=True)
        sample = blk["transactions"][:25]
        mism = 0
        for tx in sample:
            h = tx["hash"].hex()
            if not h.startswith("0x"): h = "0x" + h
            old = await w3.eth.get_transaction_receipt(tx["hash"])
            old_rev = old["status"] == 0
            nr = bmap.get(h.lower())
            if nr is None:
                print(f"  FAIL: {h[:12]} missing from map"); mism += 1; continue
            if rev_from_blockreceipt(nr) != old_rev:
                print(f"  FAIL: {h[:12]} revert mismatch"); mism += 1
        print(f"sampled {len(sample)} txs, mismatches={mism}")
        print("VERDICT:", "PASS — deployed Base path correct over HTTP" if mism == 0 else "FAIL")

asyncio.run(main())
