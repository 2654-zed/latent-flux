"""
Layer 3 -- Proxy Upgrade Watcher

For DELEGATECALL/proxy contracts, checks for implementation address changes.
Requires 1 RPC call per contract, so runs daily on suspected/confirmed only.

Detects the moment a benign proxy contract becomes a weapon by changing
its implementation to a malicious one.

Runs daily from the deployment monitor heartbeat loop.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger("surveillance.proxy_watcher")

# Standard proxy storage slots
# EIP-1967 implementation slot
EIP1967_IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
# EIP-1967 admin slot
EIP1967_ADMIN_SLOT = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
# OpenZeppelin Transparent Proxy impl slot (same as EIP-1967)
OZ_IMPL_SLOT = EIP1967_IMPL_SLOT


def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS proxy_implementations (
            contract_address TEXT NOT NULL,
            chain TEXT,
            implementation_address TEXT,
            admin_address TEXT,
            checked_at TEXT,
            previous_implementation TEXT,
            change_detected INTEGER DEFAULT 0,
            confidence_tier TEXT,
            deployer_address TEXT,
            PRIMARY KEY (contract_address, checked_at)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_proxy_impl_contract
        ON proxy_implementations(contract_address)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_proxy_impl_changed
        ON proxy_implementations(change_detected)
    """)


async def scan_proxies(conn: sqlite3.Connection, w3, chain: str, limit: int = 100) -> dict:
    """Check proxy contracts for implementation changes.

    Args:
        conn: SQLite connection (or QueueConnection)
        w3: AsyncWeb3 instance for RPC calls
        chain: Chain name
        limit: Max contracts to check per run (rate limit RPC calls)
    """
    ensure_table(conn)
    now = datetime.now(timezone.utc).isoformat()

    # Get suspected/confirmed DELEGATECALL contracts on this chain
    # Prioritize: confirmed first, then suspected with most callers
    candidates = conn.execute("""
        SELECT c.contract_address, c.deployer_address, c.confidence_tier
        FROM contracts c
        WHERE c.chain = ?
        AND c.confidence_tier IN ('suspected', 'confirmed')
        AND (c.bytecode_pattern_notes LIKE '%DELEGATECALL%'
             OR c.bytecode_pattern_notes LIKE '%proxy%'
             OR c.bytecode_pattern_notes LIKE '%upgradeable%')
        ORDER BY
            CASE c.confidence_tier WHEN 'confirmed' THEN 0 ELSE 1 END,
            c.detection_timestamp DESC
        LIMIT ?
    """, (chain, limit)).fetchall()

    checked = 0
    changes = 0
    errors = 0

    for c in candidates:
        addr = c["contract_address"]
        try:
            # Read EIP-1967 implementation slot
            impl_raw = await w3.eth.get_storage_at(addr, EIP1967_IMPL_SLOT)
            impl_addr = "0x" + impl_raw.hex()[-40:] if impl_raw else None

            # Normalize zero address
            if impl_addr and impl_addr == "0x" + "0" * 40:
                impl_addr = None

            # Read admin slot
            admin_raw = await w3.eth.get_storage_at(addr, EIP1967_ADMIN_SLOT)
            admin_addr = "0x" + admin_raw.hex()[-40:] if admin_raw else None
            if admin_addr and admin_addr == "0x" + "0" * 40:
                admin_addr = None

            # Check for previous implementation
            prev = conn.execute("""
                SELECT implementation_address FROM proxy_implementations
                WHERE contract_address = ?
                ORDER BY checked_at DESC LIMIT 1
            """, (addr,)).fetchone()

            prev_impl = prev["implementation_address"] if prev else None
            changed = prev_impl is not None and impl_addr is not None and prev_impl != impl_addr

            # Record
            conn.execute("""
                INSERT INTO proxy_implementations
                (contract_address, chain, implementation_address, admin_address,
                 checked_at, previous_implementation, change_detected,
                 confidence_tier, deployer_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (addr, chain, impl_addr, admin_addr, now,
                  prev_impl, 1 if changed else 0,
                  c["confidence_tier"], c["deployer_address"]))

            checked += 1

            if changed:
                changes += 1
                logger.warning(
                    "PROXY UPGRADE DETECTED: %s impl changed %s -> %s (tier=%s)",
                    addr[:18], prev_impl[:18] if prev_impl else "none",
                    impl_addr[:18] if impl_addr else "none",
                    c["confidence_tier"],
                )

                # Generate alert
                try:
                    from surveillance.alert_engine import insert_alert
                    insert_alert(
                        conn,
                        alert_type="PROXY_UPGRADE",
                        address=addr,
                        timestamp=now,
                        payload={
                            "contract": addr,
                            "chain": chain,
                            "deployer": c["deployer_address"],
                            "tier": c["confidence_tier"],
                            "old_implementation": prev_impl,
                            "new_implementation": impl_addr,
                            "admin": admin_addr,
                            "message": (
                                f"Proxy upgrade: {addr[:18]}... "
                                f"impl changed from {prev_impl[:18] if prev_impl else 'unknown'}... "
                                f"to {impl_addr[:18] if impl_addr else 'unknown'}... "
                                f"Tier: {c['confidence_tier']}"
                            ),
                        },
                    )
                except Exception as e:
                    logger.debug("Alert failed: %s", e)

        except Exception as e:
            errors += 1
            if checked < 3:  # Only log first few errors
                logger.debug("Proxy check failed for %s: %s", addr[:18], e)

    return {"checked": checked, "changes": changes, "errors": errors}


def get_upgrade_history(conn: sqlite3.Connection, address: str) -> list:
    """Get the implementation change history for a proxy contract."""
    ensure_table(conn)
    return [dict(r) for r in conn.execute("""
        SELECT * FROM proxy_implementations
        WHERE contract_address = ?
        ORDER BY checked_at DESC
    """, (address,)).fetchall()]
