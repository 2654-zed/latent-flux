"""Poll an explorer for outbound transactions from a target attacker EOA.

One-shot or loop mode. Emits one stdout line per new outbound transaction
observed since the previous check. Persists last-seen-hash to a state file
so multiple invocations don't double-emit.

Designed to be wrapped by Claude Code's Monitor tool (each stdout line is
a notification) or run as a cron job piped into Slack / email.

**Two backends:**

- `--backend blockscout` (default): uses Blockscout v2 REST API. Free, no
  API key. Coverage is sometimes incomplete on specific chains/addresses
  (notably observed 2026-05-10: Base Blockscout returned 0 items for an
  attacker EOA that basescan.org clearly showed had activity). Reliable
  for Eth / Arbitrum / Optimism in our experience.

- `--backend etherscan-v2`: uses Etherscan's unified V2 API across all
  chains it covers (Ethereum, Base, BSC, Arbitrum, Optimism, Polygon, …).
  Requires `ETHERSCAN_API_KEY` env var. Free tier supports Ethereum
  mainnet; many other chains require a paid plan ("Free API access is
  not supported for this chain" is the gating error on free keys).
  Recommended for Base and BSC reliability.

Initial use case: tracking `0xF7cFFC27732a5C9c4E2D592F3E33435F8dDb019A`, the
attacker EOA holding ~$172K stolen-key funds across Base / BSC / Ethereum
(2026-05-11 drain event documented in
`surveillance/data/cases/CASE_PRIVATE_KEY_DRAIN_F7CFFC27_20260511.md`).

Usage:
    python scripts/monitor_attacker_outflows.py \\
        --address 0xF7cFFC27732a5C9c4E2D592F3E33435F8dDb019A \\
        --chain base

    # Etherscan V2 (paid key, reliable on Base + BSC):
    export ETHERSCAN_API_KEY=...
    python scripts/monitor_attacker_outflows.py \\
        --address 0xF7cf... --chain base --backend etherscan-v2

    # Loop forever, poll every 60 seconds:
    python scripts/monitor_attacker_outflows.py --address 0x... --loop --interval 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CHAIN_HOSTS = {
    "base": "https://base.blockscout.com",
    "eth": "https://eth.blockscout.com",
    "arb": "https://arbitrum.blockscout.com",
    "op": "https://optimism.blockscout.com",
}

# Etherscan V2 unified API chain IDs. https://docs.etherscan.io/v2-migration
ETHERSCAN_V2_CHAIN_IDS = {
    "eth": 1,
    "op": 10,
    "bsc": 56,
    "polygon": 137,
    "arb": 42161,
    "base": 8453,
}

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"

STATE_DIR = Path.home() / ".cache" / "l3_attacker_monitor"


def _state_path(address: str, chain: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{chain}_{address.lower()}.json"


def _last_seen(address: str, chain: str) -> str | None:
    p = _state_path(address, chain)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("last_hash")
    except Exception:
        return None


def _save_last_seen(address: str, chain: str, tx_hash: str) -> None:
    _state_path(address, chain).write_text(json.dumps({"last_hash": tx_hash}))


def _fetch_blockscout(address: str, chain: str) -> list[dict]:
    host = CHAIN_HOSTS.get(chain)
    if not host:
        raise SystemExit(f"blockscout: unsupported chain {chain!r}; supported: {list(CHAIN_HOSTS)}")
    url = f"{host}/api/v2/addresses/{address}/transactions?filter=from"
    req = urllib.request.Request(url, headers={"User-Agent": "l3-attacker-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("items", []) or []


def _fetch_etherscan_v2(address: str, chain: str) -> list[dict]:
    chain_id = ETHERSCAN_V2_CHAIN_IDS.get(chain)
    if not chain_id:
        raise SystemExit(
            f"etherscan-v2: unsupported chain {chain!r}; supported: {list(ETHERSCAN_V2_CHAIN_IDS)}"
        )
    api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "etherscan-v2 backend requires ETHERSCAN_API_KEY env var. "
            "Set it via: export ETHERSCAN_API_KEY=..."
        )
    params = {
        "chainid": chain_id,
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": 50,
        "sort": "desc",
        "apikey": api_key,
    }
    url = f"{ETHERSCAN_V2_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "l3-attacker-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data.get("status") != "1":
        # Surface the error message clearly; common one: "Free API access is
        # not supported for this chain. Please upgrade your api plan ..."
        msg = data.get("message") or "unknown"
        result = data.get("result")
        if isinstance(result, str) and result:
            msg = f"{msg}: {result}"
        # Empty results return status=0 with "No transactions found" — treat as empty list.
        if "No transactions found" in str(data.get("result", "")):
            return []
        raise SystemExit(f"etherscan-v2 error: {msg}")
    raw = data.get("result", []) or []
    # Normalize to the Blockscout-v2 shape used by _format_event.
    items: list[dict] = []
    for t in raw:
        # Only outbound: where `from` matches our address (case-insensitive).
        if (t.get("from") or "").lower() != address.lower():
            continue
        items.append({
            "timestamp": _iso_from_unix(t.get("timeStamp")),
            "method": t.get("functionName") or t.get("methodId") or "transfer",
            "to": t.get("to"),
            "value": t.get("value", "0"),
            "hash": t.get("hash"),
        })
    return items


def _iso_from_unix(ts: str | int | None) -> str:
    if ts is None:
        return ""
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return str(ts)


def _fetch_recent_outbound(address: str, chain: str, backend: str) -> list[dict]:
    if backend == "blockscout":
        return _fetch_blockscout(address, chain)
    if backend == "etherscan-v2":
        return _fetch_etherscan_v2(address, chain)
    raise SystemExit(f"unknown backend {backend!r}")


def _format_event(tx: dict, address: str, chain: str) -> str:
    ts = tx.get("timestamp", "")
    method = tx.get("method") or (tx.get("decoded_input") or {}).get("method_call") or "transfer"
    to = (tx.get("to") or {}).get("hash") if isinstance(tx.get("to"), dict) else tx.get("to")
    value = tx.get("value", "0")
    # value is in wei; format as ETH for readability (or native currency on chain)
    try:
        eth = int(value) / 1e18
        value_fmt = f"{eth:.6f}"
    except Exception:
        value_fmt = str(value)
    tx_hash = tx.get("hash", "")
    return (
        f"OUTBOUND chain={chain} from={address[:10]}... "
        f"ts={ts} method={method} to={to} value={value_fmt} hash={tx_hash}"
    )


def _one_pass(address: str, chain: str, backend: str) -> int:
    """Single poll iteration. Emits new outbound txs. Returns count emitted."""
    items = _fetch_recent_outbound(address, chain, backend)
    if not items:
        return 0
    last = _last_seen(address, chain)
    # items are newest-first; emit only entries newer than last_seen, oldest-first
    new = []
    for tx in items:
        if tx.get("hash") == last:
            break
        new.append(tx)
    new.reverse()  # emit oldest-first so timeline reads chronologically
    for tx in new:
        print(_format_event(tx, address, chain), flush=True)
    if items:
        _save_last_seen(address, chain, items[0]["hash"])
    return len(new)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--address", required=True)
    ap.add_argument(
        "--chain", default="base",
        choices=sorted(set(CHAIN_HOSTS) | set(ETHERSCAN_V2_CHAIN_IDS)),
        help="Chain shortname. blockscout backend supports: %s. etherscan-v2 backend supports: %s." % (
            ", ".join(sorted(CHAIN_HOSTS)),
            ", ".join(sorted(ETHERSCAN_V2_CHAIN_IDS)),
        ),
    )
    ap.add_argument(
        "--backend", default="blockscout", choices=["blockscout", "etherscan-v2"],
        help="Explorer backend. etherscan-v2 requires ETHERSCAN_API_KEY env var.",
    )
    ap.add_argument("--loop", action="store_true", help="Poll forever")
    ap.add_argument("--interval", type=int, default=90, help="Poll interval seconds (loop mode)")
    ap.add_argument(
        "--reset", action="store_true",
        help="Clear last-seen state so the next pass treats all recent txs as new",
    )
    args = ap.parse_args()
    if args.reset:
        p = _state_path(args.address, args.chain)
        if p.exists():
            p.unlink()
        print(f"[reset] state cleared for {args.chain}:{args.address}", flush=True)
        if not args.loop:
            return 0
    if args.loop:
        while True:
            try:
                _one_pass(args.address, args.chain, args.backend)
            except Exception as e:
                print(f"ERROR {e}", flush=True)
            time.sleep(args.interval)
    else:
        _one_pass(args.address, args.chain, args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
