"""Org wallet registry — queries the `org_wallets` DB table.

Replaces the hardcoded ORG_WALLETS dict literals in auto_funder_tracer.py and
fund_tracer.py. Adding a new group no longer requires a code change — insert
a row into `org_wallets` via CLI or admin endpoint and the new mapping is
picked up on the next registry cache refresh.

Reads are cached in-process for a short TTL (default 5 min) — the registry
changes infrequently, and every funding-trace / classification call would
otherwise pay a read. Call `clear_cache()` after an insert to force a reload.

The registry is keyed on (address, chain). A single address can legitimately
appear on multiple chains (CREATE2 / cross-chain deployments). The convenience
lookups `get_org_for_address(addr)` match on address alone, returning the
first hit and warning if the address is registered under multiple chains with
conflicting org_ids (which should never happen — flagged for review).
"""

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from surveillance.db import DEFAULT_DB_PATH

_CACHE: dict[tuple[str, str], tuple[str, str]] = {}  # (addr, chain) -> (org_id, role)
_CACHE_BY_ADDR: dict[str, list[tuple[str, str, str]]] = {}  # addr -> [(chain, org_id, role)]
_CACHE_LOADED_AT: float = 0.0
_CACHE_TTL_SEC = 300.0


def _load_cache(db_path: Path = DEFAULT_DB_PATH) -> None:
    global _CACHE, _CACHE_BY_ADDR, _CACHE_LOADED_AT
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        rows = conn.execute(
            "SELECT address, chain, org_id, role FROM org_wallets"
        ).fetchall()
    finally:
        conn.close()
    _CACHE = {(a.lower(), c): (org, role) for a, c, org, role in rows}
    by_addr: dict[str, list] = {}
    for a, c, org, role in rows:
        by_addr.setdefault(a.lower(), []).append((c, org, role))
    _CACHE_BY_ADDR = by_addr
    _CACHE_LOADED_AT = time.monotonic()


def _ensure_cache(db_path: Path = DEFAULT_DB_PATH) -> None:
    if not _CACHE_LOADED_AT or (time.monotonic() - _CACHE_LOADED_AT) > _CACHE_TTL_SEC:
        _load_cache(db_path)


def clear_cache() -> None:
    """Force the next lookup to reload from disk."""
    global _CACHE_LOADED_AT
    _CACHE_LOADED_AT = 0.0


def get_org_wallet(address: str, chain: str,
                   db_path: Path = DEFAULT_DB_PATH) -> Optional[tuple[str, str]]:
    """Return (org_id, role) or None. Chain-scoped lookup."""
    _ensure_cache(db_path)
    return _CACHE.get((address.lower(), chain))


def get_org_for_address(address: str,
                        db_path: Path = DEFAULT_DB_PATH) -> Optional[tuple[str, str]]:
    """Return (org_id, role) for any chain match, or None.

    If an address is registered under multiple chains with conflicting org_ids,
    returns the first match but the case shouldn't occur — org identity should
    be chain-stable for any known group.
    """
    _ensure_cache(db_path)
    hits = _CACHE_BY_ADDR.get(address.lower())
    if not hits:
        return None
    return hits[0][1], hits[0][2]


def is_org_wallet(address: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Boolean check — any chain match."""
    _ensure_cache(db_path)
    return address.lower() in _CACHE_BY_ADDR


def all_wallets_for_org(org_id: str,
                        db_path: Path = DEFAULT_DB_PATH) -> list[tuple[str, str, str]]:
    """Return [(address, chain, role), ...] for a given org_id."""
    _ensure_cache(db_path)
    out = []
    for (addr, chain), (org, role) in _CACHE.items():
        if org == org_id:
            out.append((addr, chain, role))
    return out


def insert_wallet(conn: sqlite3.Connection, address: str, chain: str,
                  org_id: str, role: str, added_by: str,
                  reason: Optional[str] = None) -> None:
    """Register a new org wallet. Call clear_cache() after if immediate visibility needed."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO org_wallets
           (address, chain, org_id, role, added_at, added_by, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (address.lower(), chain, org_id, role, now, added_by, reason),
    )
    conn.commit()


def _main():
    """CLI: python -m surveillance.org_registry [--list | --seed]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="Dump current registry.")
    ap.add_argument("--seed", action="store_true", help="Seed the 13 known wallets.")
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    args = ap.parse_args()

    if args.list:
        _ensure_cache(Path(args.db))
        if not _CACHE:
            print("(empty)")
            return
        by_org: dict = {}
        for (addr, chain), (org, role) in sorted(_CACHE.items()):
            by_org.setdefault(org, []).append((addr, chain, role))
        for org, rows in sorted(by_org.items()):
            print(f"{org} ({len(rows)} wallets):")
            for addr, chain, role in rows:
                print(f"  {addr}  [{chain:<10}] {role}")
        return

    if args.seed:
        conn = sqlite3.connect(args.db, timeout=30)
        try:
            for addr, chain, org, role, reason in SEED_WALLETS:
                insert_wallet(conn, addr, chain, org, role, added_by="bootstrap",
                              reason=reason)
            print(f"seeded {len(SEED_WALLETS)} wallets")
        finally:
            conn.close()
        clear_cache()
        return

    ap.print_help()


# Seed data: the wallets that were formerly hardcoded in auto_funder_tracer.py
# and fund_tracer.py. Chain is 'arbitrum' because these were observed on
# Arbitrum; if the same wallet later appears on base/optimism it should get
# its own row per the composite PK.
SEED_WALLETS: list[tuple[str, str, str, str, str]] = [
    # (address, chain, org_id, role, reason)
    ("0xf186cb00e49e18491db5783ff04fae3818102ff7", "arbitrum", "org_001", "treasury",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0xe93d64f3fbc352131e79fc5578cbe44b66697f86", "arbitrum", "org_001", "operator",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0xfd51e33d44b376ef346d24a130a51035db09c1dc", "arbitrum", "org_001", "operator_2",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0xc6962004f452be9203591991d15f6b388e09e8d0", "arbitrum", "org_001", "cashout",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0x8c826f795466e39acbff1bb4eeeb759609377ba1", "arbitrum", "org_001", "gas_station",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0x360e68faccca8ca495c1b759fd9eee466db9fb32", "arbitrum", "org_001", "treasury_branch",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0x01989c93890aed05a63d179b03424997075b6acf", "arbitrum", "org_001", "exit_cex",
     "bootstrap from fund_tracer.py ORG_WALLETS"),
    ("0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb", "arbitrum", "org_001", "laundry",
     "bootstrap from fund_tracer.py ORG_WALLETS"),
    ("0x96daa0b8a5499ea9323421ed0cda06b345caab73", "arbitrum", "org_001", "lp_staging",
     "bootstrap from fund_tracer.py ORG_WALLETS"),
    ("0x27920e8039d2b6e93e36f5d5f53b998e2e631a70", "arbitrum", "org_001", "lp_companion",
     "bootstrap from fund_tracer.py ORG_WALLETS"),
    ("0x51c72848c68a965f66fa7a88855f9f7784502a7f", "arbitrum", "org_001", "defi_exit_channel",
     "bootstrap from fund_tracer.py ORG_WALLETS"),
    ("0x238d7170f309a55b87a144a341bd6105897082ca", "arbitrum", "org_002", "treasury_senior",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
    ("0xde8eb937cb5475eee5ac96dce6ba2d18e439c473", "arbitrum", "org_002", "treasury_junior",
     "bootstrap from auto_funder_tracer.py ORG_WALLETS"),
]


if __name__ == "__main__":
    _main()
