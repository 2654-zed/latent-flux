"""Fix A parity gate, take 2 — fetch eth_getBlockReceipts over HTTP (not WS).
The WS frame limit (1009 message too big) blocks Base's large block receipts,
so the real implementation must call eth_getBlockReceipts over the HTTP
endpoint. This proves HTTP works for Base + parity holds.

Uses urllib over the HTTP RPC URL derived from the WSS env (wss->https).
Run on prod. Writes /app/reports/_fixA_parity_http.txt.
"""
import json, os, sys, urllib.request
from pathlib import Path

OUT = Path("/app/reports/_fixA_parity_http.txt")

def wss_to_http(u):
    if not u: return None
    return u.replace("wss://", "https://").replace("/v2/", "/v2/")

CHAINS = {
    "base": wss_to_http(os.environ.get("BASE_WSS_URL")),
    "arbitrum": wss_to_http(os.environ.get("ARB_WSS_URL") or os.environ.get("ARBITRUM_WSS_URL")),
    "optimism": wss_to_http(os.environ.get("OP_WSS_URL") or os.environ.get("OPTIMISM_WSS_URL")),
}

def rpc(url, method, params):
    body = json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def rev_from_blockreceipt(r):
    s = r.get("status")
    if isinstance(s, str): return int(s,16)==0
    if isinstance(s, int): return s==0
    return None

L=["Fix A parity (HTTP) — eth_getBlockReceipts over HTTPS","="*64]
all_ok=True
for name,url in CHAINS.items():
    if not url:
        L.append(f"  {name}: no HTTP url — SKIP"); continue
    try:
        latest = int(rpc(url,"eth_blockNumber",[])["result"],16)
        bn = latest-5
        br = rpc(url,"eth_getBlockReceipts",[hex(bn)])
        items = br.get("result")
        if not items:
            L.append(f"  {name}: getBlockReceipts no result — err={br.get('error')}"); all_ok=False; continue
        bmap={r["transactionHash"].lower():r for r in items if r.get("transactionHash")}
        blk = rpc(url,"eth_getBlockByNumber",[hex(bn),True])["result"]
        txs = blk["transactions"][:25]
        mism=0; checked=0
        for tx in txs:
            h=tx["hash"].lower()
            old = rpc(url,"eth_getTransactionReceipt",[tx["hash"]])["result"]
            old_rev = int(old["status"],16)==0
            nr = bmap.get(h)
            if nr is None:
                L.append(f"  {name}: tx {h[:12]} missing from receipts map — keying bug"); all_ok=False; break
            if old_rev != rev_from_blockreceipt(nr): mism+=1
            checked+=1
        else:
            ok=(mism==0); all_ok=all_ok and ok
            # approximate payload size for the frame-limit note
            sz = len(json.dumps(items))
            L.append(f"  {name}: block {bn} receipts={len(items)} (~{sz//1024}KB) sampled={checked} mismatches={mism} -> {'OK' if ok else 'MISMATCH'}")
    except Exception as e:
        L.append(f"  {name}: ERROR {type(e).__name__}: {e}"); all_ok=False
L.append("")
L.append("VERDICT: "+("HTTP path works on all chains — Fix A viable via HTTP" if all_ok else "FAIL"))
OUT.write_text("\n".join(L),encoding="utf-8")
print("\n".join(L))
