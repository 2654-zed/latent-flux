"""One-shot L2 RPC trace for the 4 L2-only funders.

Reuses surveillance.eth_depth.trace_eth_mainnet (which is just an Alchemy
getAssetTransfers wrapper — chain-agnostic) by pointing it at L2 HTTP URLs.

Run on Railway:  railway ssh "python scripts/l2_trace_runner.py"
"""
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, '/app')
from surveillance.eth_depth import trace_eth_mainnet, ensure_tables

# Map each L2-only funder to its primary chain
TARGETS = [
    ("0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07", "optimism"),
    ("0x39591e7c099a379fd7b349ebfecaeef439c40454", "base"),
    ("0x8ca702323c341a8d46ee94a2abeddb08798ca10d", "base"),
    ("0x0e6e91775d24d34b90e0f3d806a90705f0199999", "base"),
]

CHAIN_URL_ENV = {
    "base": ("BASE_HTTP_URL", "BASE_WSS_URL"),
    "arbitrum": ("ARB_HTTP_URL", "ARB_WSS_URL"),
    "optimism": ("OP_HTTP_URL", "OP_WSS_URL"),
}


def get_url(chain: str) -> str:
    http_env, wss_env = CHAIN_URL_ENV[chain]
    url = os.environ.get(http_env, "")
    if not url:
        wss = os.environ.get(wss_env, "")
        url = wss.replace("wss://", "https://") if wss else ""
    return url


async def main():
    db_path = "/app/surveillance/data/surveillance.db"
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)

    out_dir = Path("/app/scripts/funder_traces_l2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for addr, chain in TARGETS:
        url = get_url(chain)
        if not url:
            print(f"  {addr} ({chain}): no RPC URL configured (skip)")
            continue
        print(f"  {addr} ({chain}): tracing...")
        try:
            result = await trace_eth_mainnet(url, addr, conn, max_hops=1)
            # tag the chain explicitly since trace_eth_mainnet hardcodes 'ethereum'
            result["chain"] = chain
            out_file = out_dir / f"{addr}_{chain}.json"
            out_file.write_text(json.dumps(result, indent=2))
            inb = len(result.get("inbound", []))
            out = len(result.get("outbound", []))
            br = len(result.get("bridge_activity", []))
            cx = len(result.get("cex_connections", []))
            hops = len(result.get("funding_hops", []))
            print(f"    inb={inb} out={out} bridge={br} cex={cx} hops={hops}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

    conn.close()
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
