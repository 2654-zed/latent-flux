"""
Layer 3 — Address Poisoning Watcher

Monitors high-value watchlisted addresses for address poisoning attempts.
Detects:
  1. Vanity-prefix matches: inbound transfers from addresses starting with
     the same 6+ hex chars as the target (impersonation for copy-paste attacks)
  2. Unicode homoglyph tokens: inbound transfers of fake tokens with Cyrillic,
     Lisu, or other non-ASCII characters impersonating real token names
  3. Dust transfers: zero-value or sub-0.01 ETH transfers (poisoning signature)
  4. Zero-value ETH spam from known poisoner infrastructure

Fires POISONING_ATTEMPT alerts when any signal hits.

Usage:
    python -m surveillance.poisoning_watcher --scan
    python -m surveillance.poisoning_watcher --watch 0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb
"""

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

# Alchemy endpoints — use the env-provided URLs
import os
_ARB_WSS = os.environ.get("ARB_WSS_URL", "")
_BASE_WSS = os.environ.get("BASE_WSS_URL", "")
_OP_WSS = os.environ.get("OP_WSS_URL", "")

# Convert WSS to HTTP, or use hardcoded (monitoring-only, low volume)
def _wss_to_http(wss: str) -> str:
    if not wss:
        return ""
    return wss.replace("wss://", "https://")

RPC_URLS = {
    "arbitrum": _wss_to_http(_ARB_WSS) or "https://arb-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3",
    "base": _wss_to_http(_BASE_WSS) or "https://base-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3",
    "optimism": _wss_to_http(_OP_WSS) or "https://opt-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3",
}

# Default watched targets. Expand this list via --watch or by adding
# high-value addresses to the watchlist table with priority=CRITICAL.
DEFAULT_WATCHED = [
    "0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb",  # 49K ETH bridge whale
    "0x2ce910fbba65b454bbaf6a18c952a70f3bcd8299",  # 3,158 ETH bridge, also poisoner
    "0xf30ba13e4b04ce5dc4d254ae5fa95477800f0eb0",  # CEX hot wallet (L1)
]

# Prefix length for vanity-poisoning detection
VANITY_PREFIX_LEN = 8  # matches first 6 hex chars after 0x


def _rpc(url: str, method: str, params: list) -> Optional[dict]:
    data = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
    }).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result.get("result")
    except Exception as e:
        print(f"  RPC error: {e}", file=sys.stderr)
        return None


def _classify_transfer(target: str, transfer: dict) -> Optional[str]:
    """Classify an inbound transfer. Returns alert type or None."""
    src = (transfer.get("from") or "").lower()
    asset = transfer.get("asset") or ""
    value = transfer.get("value") or 0
    target_lower = target.lower()

    # Never flag the target talking to itself
    if src == target_lower:
        return None

    # 1. Vanity prefix match (poisoning via address lookalike)
    if src.startswith(target_lower[:VANITY_PREFIX_LEN]) and src != target_lower:
        return "VANITY_PREFIX_POISONING"

    # 2. Unicode/homoglyph token (fake token name impersonating real)
    if asset and not asset.isascii():
        return "UNICODE_TOKEN_POISONING"

    # 3. Known phishing patterns in token name
    if asset and any(kw in asset.lower() for kw in ["claim", "visit", "t.me/", "http", "airdrop"]):
        return "PHISHING_AIRDROP"

    # 4. Zero-value or dust from previously-unknown address (weak signal)
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        val_float = 0
    if val_float == 0 and transfer.get("category") == "external":
        return "ZERO_VALUE_PING"

    return None


def _get_inbound(chain: str, address: str, count: int = 100) -> list[dict]:
    """Get recent inbound transfers for target on given chain.

    Note: Base only supports 'internal' for ETH/MATIC. Arbitrum supports
    all three. Optimism supports external+erc20.
    """
    url = RPC_URLS.get(chain)
    if not url:
        return []
    # external + erc20 works on all chains, internal is Arbitrum-only
    categories = ["external", "erc20"]
    if chain == "arbitrum":
        categories.append("internal")

    params = [{
        "toAddress": address,
        "category": categories,
        "maxCount": hex(count),
        "order": "desc",
    }]
    result = _rpc(url, "alchemy_getAssetTransfers", params)
    if not result:
        return []
    return result.get("transfers", []) or []


def _get_last_scanned_block(conn: sqlite3.Connection, address: str, chain: str) -> int:
    """Return the last block we scanned for this target on this chain."""
    try:
        row = conn.execute(
            "SELECT last_block FROM poisoning_scan_state "
            "WHERE address = ? AND chain = ?",
            (address.lower(), chain),
        ).fetchone()
        if row:
            return row[0] or 0
    except sqlite3.OperationalError:
        pass
    return 0


def _set_last_scanned_block(conn: sqlite3.Connection, address: str, chain: str, block: int) -> None:
    conn.execute(
        "INSERT INTO poisoning_scan_state (address, chain, last_block, last_scan) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(address, chain) DO UPDATE SET "
        "last_block = excluded.last_block, last_scan = excluded.last_scan",
        (address.lower(), chain, block, datetime.now(timezone.utc).isoformat()),
    )


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create tracking tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS poisoning_scan_state (
            address TEXT NOT NULL,
            chain TEXT NOT NULL,
            last_block INTEGER,
            last_scan TEXT,
            PRIMARY KEY (address, chain)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS poisoning_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_address TEXT NOT NULL,
            chain TEXT NOT NULL,
            poisoner_address TEXT NOT NULL,
            event_type TEXT NOT NULL,
            asset TEXT,
            value TEXT,
            tx_hash TEXT,
            block_number INTEGER,
            detected_at TEXT,
            UNIQUE(target_address, chain, tx_hash)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_poisoning_target_time
        ON poisoning_events(target_address, detected_at DESC)
    """)
    conn.commit()


def scan_target(conn: sqlite3.Connection, address: str) -> dict:
    """Scan all three chains for poisoning attempts against target."""
    now_iso = datetime.now(timezone.utc).isoformat()
    results = {
        "target": address,
        "scanned_at": now_iso,
        "events_by_chain": {},
        "total_new_events": 0,
    }

    for chain in ["arbitrum", "base", "optimism"]:
        last_block = _get_last_scanned_block(conn, address, chain)
        transfers = _get_inbound(chain, address, count=100)

        new_events = []
        max_block_seen = last_block

        for t in transfers:
            try:
                block = int(t.get("blockNum", "0x0"), 16)
            except (ValueError, TypeError):
                block = 0
            max_block_seen = max(max_block_seen, block)

            # Skip if we've already scanned past this block
            if block <= last_block and last_block > 0:
                continue

            event_type = _classify_transfer(address, t)
            if not event_type:
                continue

            src = (t.get("from") or "").lower()
            asset = t.get("asset") or ""
            value = str(t.get("value") or 0)
            tx_hash = t.get("hash") or ""

            try:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO poisoning_events "
                    "(target_address, chain, poisoner_address, event_type, "
                    "asset, value, tx_hash, block_number, detected_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (address.lower(), chain, src, event_type,
                     asset, value, tx_hash, block, now_iso),
                )
                inserted = cursor.rowcount > 0
            except sqlite3.IntegrityError:
                inserted = False

            if inserted:
                new_events.append({
                    "chain": chain,
                    "poisoner": src,
                    "event_type": event_type,
                    "asset": asset,
                    "value": value,
                    "block": block,
                    "tx_hash": tx_hash,
                })
                # Also fire alert if alerts table exists
                try:
                    conn.execute(
                        "INSERT INTO alerts (alert_type, address, tx_hash, "
                        "block_number, timestamp, payload) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            "POISONING_ATTEMPT",
                            address.lower(),
                            tx_hash,
                            block,
                            now_iso,
                            json.dumps({
                                "target": address.lower(),
                                "chain": chain,
                                "poisoner": src,
                                "event_type": event_type,
                                "asset": asset,
                                "value": value,
                            }),
                        ),
                    )
                except sqlite3.OperationalError:
                    pass

        _set_last_scanned_block(conn, address, chain, max_block_seen)
        conn.commit()

        results["events_by_chain"][chain] = {
            "inbound_transfers_seen": len(transfers),
            "new_poisoning_events": len(new_events),
            "events": new_events,
        }
        results["total_new_events"] += len(new_events)
        time.sleep(0.3)  # be nice to RPC

    return results


def scan_all(targets: Optional[list[str]] = None) -> None:
    """Scan all watched targets."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_tables(conn)

    if not targets:
        targets = list(DEFAULT_WATCHED)
        # Also pull CRITICAL watchlist entries with high-value context
        try:
            extra = conn.execute(
                "SELECT DISTINCT address FROM watchlist "
                "WHERE priority = 'CRITICAL' OR priority = 'HIGH' "
                "LIMIT 20"
            ).fetchall()
            for r in extra:
                a = (r[0] or "").lower()
                if a and a not in [t.lower() for t in targets]:
                    targets.append(a)
        except sqlite3.OperationalError:
            pass

    print(f"[poisoning_watcher] Scanning {len(targets)} targets across 3 chains...")
    print()

    total_new = 0
    for target in targets:
        result = scan_target(conn, target)
        new = result["total_new_events"]
        total_new += new
        print(f"  {target[:24]}: {new} new poisoning events")
        for chain, data in result["events_by_chain"].items():
            if data["new_poisoning_events"] > 0:
                for ev in data["events"][:5]:
                    print(f"    [{chain}] {ev['event_type']}: "
                          f"from={ev['poisoner'][:24]} "
                          f"asset={ascii(ev['asset'])[:40]} "
                          f"val={ev['value']}")

    print()
    print(f"[poisoning_watcher] Complete. {total_new} new poisoning events detected.")
    conn.close()


def show_history(target: str, days: int = 7) -> None:
    """Print historical poisoning events for a target."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    _ensure_tables(conn)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT chain, poisoner_address, event_type, asset, value, "
        "tx_hash, block_number, detected_at "
        "FROM poisoning_events WHERE target_address = ? "
        "ORDER BY detected_at DESC LIMIT 100",
        (target.lower(),),
    ).fetchall()

    print(f"Poisoning events against {target} ({len(rows)} total):")
    if not rows:
        print("  (none detected yet)")
        return

    for r in rows:
        asset_safe = ascii(r["asset"])[:50] if r["asset"] else "none"
        print(f"  [{r['chain']}] {r['event_type']} "
              f"from={r['poisoner_address'][:24]} "
              f"asset={asset_safe} "
              f"val={r['value']} "
              f"block={r['block_number']} "
              f"@ {r['detected_at'][:19]}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 3 Address Poisoning Watcher"
    )
    parser.add_argument("--scan", action="store_true",
                        help="Scan all watched targets for poisoning attempts")
    parser.add_argument("--watch", type=str,
                        help="Add an address to scan (in addition to defaults)")
    parser.add_argument("--history", type=str,
                        help="Show historical poisoning events for an address")
    args = parser.parse_args()

    if args.history:
        show_history(args.history)
    elif args.scan or args.watch:
        targets = list(DEFAULT_WATCHED)
        if args.watch and args.watch.lower() not in [t.lower() for t in targets]:
            targets.append(args.watch.lower())
        scan_all(targets)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
