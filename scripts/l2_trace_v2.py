"""L2 trace v2 — uses only 'external' category (L2 Alchemy doesn't support 'internal')."""
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

TARGETS = [
    ("0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07", "optimism"),
    ("0x39591e7c099a379fd7b349ebfecaeef439c40454", "base"),
    ("0x8ca702323c341a8d46ee94a2abeddb08798ca10d", "base"),
    ("0x0e6e91775d24d34b90e0f3d806a90705f0199999", "base"),
]

CHAIN_WSS = {
    "base": "BASE_WSS_URL",
    "arbitrum": "ARB_WSS_URL",
    "optimism": "OP_WSS_URL",
}


def get_url(chain: str) -> str:
    wss = os.environ.get(CHAIN_WSS[chain], "")
    return wss.replace("wss://", "https://") if wss else ""


async def transfers(http_url: str, params: dict) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [params],
        }
        async with session.post(http_url, json=payload, timeout=30) as resp:
            data = await resp.json()
            if "error" in data:
                print(f"      RPC error: {data['error']}")
                return []
            return data.get("result", {}).get("transfers", [])


async def trace_l2(http_url: str, address: str) -> dict:
    # Inbound — external + erc20 (L2 doesn't have 'internal')
    inbound = await transfers(http_url, {
        "toAddress": address,
        "category": ["external", "erc20"],
        "withMetadata": True,
        "maxCount": "0x32",  # 50
        "order": "desc",
    })
    outbound = await transfers(http_url, {
        "fromAddress": address,
        "category": ["external", "erc20"],
        "withMetadata": True,
        "maxCount": "0x32",
        "order": "desc",
    })
    return {
        "address": address,
        "inbound": [
            {
                "from": t.get("from"),
                "to": t.get("to"),
                "value_eth": t.get("value"),
                "asset": t.get("asset"),
                "tx_hash": t.get("hash"),
                "block": int(t.get("blockNum", "0x0"), 16) if t.get("blockNum") else 0,
                "timestamp": (t.get("metadata") or {}).get("blockTimestamp"),
                "category": t.get("category"),
            } for t in inbound
        ],
        "outbound": [
            {
                "from": t.get("from"),
                "to": t.get("to"),
                "value_eth": t.get("value"),
                "asset": t.get("asset"),
                "tx_hash": t.get("hash"),
                "block": int(t.get("blockNum", "0x0"), 16) if t.get("blockNum") else 0,
                "timestamp": (t.get("metadata") or {}).get("blockTimestamp"),
                "category": t.get("category"),
            } for t in outbound
        ],
    }


async def main():
    out_dir = Path("/app/scripts/funder_traces_l2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for addr, chain in TARGETS:
        url = get_url(chain)
        if not url:
            print(f"  {addr} ({chain}): no RPC URL configured")
            continue
        print(f"  {addr} ({chain})...")
        try:
            r = await trace_l2(url, addr)
            (out_dir / f"{addr}_{chain}.json").write_text(json.dumps(r, indent=2))
            print(f"    inbound={len(r['inbound'])}  outbound={len(r['outbound'])}")
            # show top 5 inbound
            in_sorted = sorted(
                [t for t in r["inbound"] if t.get("value_eth")],
                key=lambda x: -float(x["value_eth"] or 0))
            for t in in_sorted[:5]:
                print(f"      IN  {t['timestamp'][:19] if t['timestamp'] else '-':<20} from={t['from']}  {t['value_eth']} {t['asset']}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
