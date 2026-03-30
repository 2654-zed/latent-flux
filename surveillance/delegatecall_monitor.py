"""
Layer 3 — DELEGATECALL Post-Deployment Monitor

Detects when contracts with DELEGATECALL capability change behavior
after deployment. Re-classifies bytecode and alerts on upgrades.

Detection: Contracts flagged with DELEGATECALL at deployment that have
subsequent activity are candidates for logic changes. On each scan,
fetch current bytecode (1 RPC call per candidate) and compare to cache.

Usage:
    python -m surveillance.delegatecall_monitor --check --dry-run  # candidates only, no RPC
    python -m surveillance.delegatecall_monitor --check             # live check with RPC
    python -m surveillance.delegatecall_monitor --watch             # 30-min loop
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
logger = logging.getLogger("surveillance.delegatecall_monitor")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _get_http(chain: str) -> str:
    urls = {
        "base": os.environ.get("BASE_WSS_URL", ""),
        "arbitrum": os.environ.get("ARB_WSS_URL", ""),
        "optimism": os.environ.get("OP_WSS_URL", ""),
    }
    wss = urls.get(chain, "")
    return wss.replace("wss://", "https://") if wss else ""


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upgrade_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address TEXT NOT NULL,
            chain TEXT,
            detected_at TEXT,
            trigger_type TEXT,
            old_code_hash TEXT,
            new_code_hash TEXT,
            old_tier TEXT,
            new_tier TEXT,
            old_patterns TEXT,
            new_patterns TEXT,
            bytecode_changed INTEGER,
            severity TEXT DEFAULT 'MEDIUM',
            details TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_upgrade_contract ON upgrade_events(contract_address)")
    conn.commit()


def find_candidates(conn) -> list:
    """Find DELEGATECALL contracts with activity that haven't been checked yet."""
    return conn.execute("""
        SELECT DISTINCT c.contract_address, c.chain, c.confidence_tier,
            c.bytecode_pattern_notes, c.deployer_address
        FROM contracts c
        JOIN transaction_events te ON te.contract_address = c.contract_address
        LEFT JOIN upgrade_events ue ON ue.contract_address = c.contract_address
        WHERE c.bytecode_pattern_notes LIKE '%DELEGATECALL%'
        AND ue.contract_address IS NULL
        GROUP BY c.contract_address
        HAVING COUNT(te.tx_hash) >= 3
    """).fetchall()


async def check_contract(session, http_url: str, contract: str) -> dict:
    """Fetch current bytecode via RPC. Returns code hash."""
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getCode",
        "params": [contract, "latest"]
    }
    try:
        async with session.post(http_url, json=payload, timeout=15) as resp:
            data = await resp.json()
            code = data.get("result", "0x")
            code_hex = code[2:] if code.startswith("0x") else code
            import hashlib
            code_hash = hashlib.sha256(bytes.fromhex(code_hex) if code_hex else b"").hexdigest()
            return {"code_hash": code_hash, "code_size": len(code_hex) // 2, "code": code_hex}
    except Exception as e:
        return {"error": str(e)}


def reclassify(bytecode_hex: str, contract_address: str) -> dict:
    """Re-run bytecode classifier on new bytecode."""
    from surveillance.deployment_monitor import Deployment
    from surveillance.bytecode_classifier import classify

    dep = Deployment(
        contract_address=contract_address,
        deployer_address="",
        block_number=0,
        timestamp_iso=_now(),
        tx_hash="",
        bytecode=bytecode_hex,
        gas_price_gwei=0,
        init_code_hash="",
        deployed_code_hash="",
        cache_hit="",
    )
    return classify(dep)


async def run_check(dry_run: bool = False):
    _load_env()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)

    candidates = find_candidates(conn)
    print(f"[delegatecall_monitor] Candidates: {len(candidates)} DELEGATECALL contracts with activity")

    if dry_run:
        print("\nDRY RUN — no RPC calls")
        for c in candidates[:20]:
            activity = conn.execute("SELECT COUNT(*) as h FROM transaction_events WHERE contract_address=?",
                                    (c["contract_address"],)).fetchone()["h"]
            print(f"  {c['contract_address'][:20]}... | {c['chain']} | {c['confidence_tier']} | {activity} hits | {(c['bytecode_pattern_notes'] or '')[:40]}")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")
        conn.close()
        return

    rpc_calls = 0
    changes = 0
    upgrades = 0

    async with aiohttp.ClientSession() as session:
        for c in candidates:
            http_url = _get_http(c["chain"])
            if not http_url:
                continue

            # Get cached bytecode hash
            cached = conn.execute(
                "SELECT code_hash FROM bytecode_cache WHERE source_contract=?",
                (c["contract_address"],)
            ).fetchone()
            old_hash = cached["code_hash"] if cached else None

            # Fetch current bytecode
            result = await check_contract(session, http_url, c["contract_address"])
            rpc_calls += 1

            if "error" in result:
                logger.warning("RPC error for %s: %s", c["contract_address"][:18], result["error"])
                continue

            new_hash = result["code_hash"]
            bytecode_changed = old_hash is not None and new_hash != old_hash
            code_empty = result["code_size"] == 0

            # Determine severity
            severity = "LOW"
            new_tier = c["confidence_tier"]
            new_patterns = c["bytecode_pattern_notes"]

            if code_empty:
                # Contract self-destructed!
                severity = "CRITICAL"
                new_tier = "destroyed"
                new_patterns = "SELFDESTRUCT executed — bytecode erased"
                print(f"  CRITICAL: {c['contract_address'][:18]}... SELF-DESTRUCTED (was {c['confidence_tier']})")
            elif bytecode_changed:
                changes += 1
                # Re-classify
                reclass = reclassify(result["code"], c["contract_address"])
                new_tier = reclass.get("confidence_tier", "unknown")
                new_patterns = reclass.get("bytecode_signals", {}).get("pattern_notes")

                if new_tier == "suspected" and c["confidence_tier"] == "unknown":
                    severity = "HIGH"
                    upgrades += 1
                    print(f"  HIGH: {c['contract_address'][:18]}... upgraded from clean to SUSPECTED")
                    print(f"    New patterns: {new_patterns[:80]}")
                elif new_tier != c["confidence_tier"]:
                    severity = "MEDIUM"
                    print(f"  MEDIUM: {c['contract_address'][:18]}... tier changed: {c['confidence_tier']} -> {new_tier}")
                else:
                    severity = "LOW"

            conn.execute("""
                INSERT INTO upgrade_events
                (contract_address, chain, detected_at, trigger_type, old_code_hash, new_code_hash,
                 old_tier, new_tier, old_patterns, new_patterns, bytecode_changed, severity, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c["contract_address"], c["chain"], _now(),
                "selfdestruct" if code_empty else "bytecode_change" if bytecode_changed else "no_change",
                old_hash, new_hash,
                c["confidence_tier"], new_tier,
                c["bytecode_pattern_notes"], new_patterns,
                1 if bytecode_changed or code_empty else 0,
                severity,
                json.dumps({"code_size": result["code_size"], "rpc_call": True}),
            ))

    conn.commit()

    print(f"\n[delegatecall_monitor] Complete.")
    print(f"  RPC calls: {rpc_calls}")
    print(f"  Bytecode changes: {changes}")
    print(f"  Clean->dangerous upgrades: {upgrades}")
    print(f"  Total candidates processed: {len(candidates)}")

    # Summary
    for r in conn.execute("SELECT severity, COUNT(*) as c FROM upgrade_events GROUP BY severity ORDER BY c DESC"):
        print(f"  {r['severity']}: {r['c']}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELEGATECALL post-deployment monitor")
    parser.add_argument("--check", action="store_true", help="Run one-time check")
    parser.add_argument("--dry-run", action="store_true", help="Show candidates only, no RPC")
    parser.add_argument("--watch", action="store_true", help="Continuous 30-min loop")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.check:
        asyncio.run(run_check(dry_run=args.dry_run))
    elif args.watch:
        import time
        while True:
            asyncio.run(run_check(dry_run=False))
            print(f"Next check in 30 minutes...")
            time.sleep(1800)
    else:
        parser.print_help()
