"""Fix A pre-deploy gate: prove eth_getBlockReceipts revert-status parity
vs per-tx eth_getTransactionReceipt, across base/arbitrum/optimism.

For each chain: pick a recent block, fetch ALL receipts via
eth_getBlockReceipts (the new path) AND per-tx via getTransactionReceipt
(the old path) for a sample of txs, and assert the derived is_reverted
matches for every sampled tx. Also confirms the endpoint supports
eth_getBlockReceipts at all.

Run on prod (where the WSS env vars live):
  railway ssh "cd /app && python3 /app/scripts/t_fixA_parity.py"
Writes /app/reports/_fixA_parity.txt.
0 trading impact; ~ a few hundred CU total (one block's receipts x3 chains).
"""
import asyncio, json, os, sys
from pathlib import Path
from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

OUT = Path("/app/reports/_fixA_parity.txt")
CHAINS = {
    "base": os.environ.get("BASE_WSS_URL"),
    "arbitrum": os.environ.get("ARB_WSS_URL") or os.environ.get("ARBITRUM_WSS_URL"),
    "optimism": os.environ.get("OP_WSS_URL") or os.environ.get("OPTIMISM_WSS_URL"),
}
SAMPLE = 25  # txs per block to cross-check


def reverted_from_blockreceipt(r):
    s = r.get("status")
    if isinstance(s, str):
        return int(s, 16) == 0
    if isinstance(s, int):
        return s == 0
    return None


async def check_chain(name, wss):
    L = []
    if not wss:
        return f"{name}: WSS env not set — SKIP", True
    try:
        async with AsyncWeb3(WebSocketProvider(wss)) as w3:
            latest = await w3.eth.block_number
            bn = latest - 5  # a few blocks back for finality
            # NEW path: eth_getBlockReceipts
            raw = await w3.provider.make_request("eth_getBlockReceipts", [hex(bn)])
            items = raw.get("result") if isinstance(raw, dict) else None
            if not items:
                return f"{name}: eth_getBlockReceipts returned no result (raw keys={list(raw)[:3] if isinstance(raw,dict) else raw}) — UNSUPPORTED?", False
            bmap = {r["transactionHash"].lower(): r for r in items if r.get("transactionHash")}
            # OLD path: per-tx, on a sample
            block = await w3.eth.get_block(bn, full_transactions=True)
            txs = block["transactions"][:SAMPLE]
            mism = 0
            checked = 0
            for tx in txs:
                h = tx["hash"].hex()
                if not h.startswith("0x"):
                    h = "0x" + h
                old = await w3.eth.get_transaction_receipt(tx["hash"])
                old_rev = (old["status"] == 0)
                new_r = bmap.get(h.lower())
                if new_r is None:
                    return f"{name}: block {bn} tx {h[:12]} present in get_block but MISSING from getBlockReceipts map — keying bug", False
                new_rev = reverted_from_blockreceipt(new_r)
                checked += 1
                if old_rev != new_rev:
                    mism += 1
            ok = (mism == 0)
            return (f"{name}: block {bn}, receipts={len(items)}, sampled={checked}, "
                    f"mismatches={mism} -> {'PARITY OK' if ok else 'MISMATCH!'}"), ok
    except Exception as e:
        return f"{name}: ERROR {type(e).__name__}: {e}", False


async def main():
    lines = ["Fix A parity test — eth_getBlockReceipts vs per-tx getTransactionReceipt", "=" * 70]
    all_ok = True
    for name, wss in CHAINS.items():
        msg, ok = await check_chain(name, wss)
        lines.append("  " + msg)
        all_ok = all_ok and ok
    lines.append("")
    lines.append("VERDICT: " + ("ALL PARITY OK — safe to deploy Fix A" if all_ok else "FAIL — do not deploy"))
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

asyncio.run(main())
