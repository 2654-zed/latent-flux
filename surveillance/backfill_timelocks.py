"""
One-time backfill: re-fetch bytecode for TIMESTAMP-gated contracts and
extract activation timestamps using the enhanced classifier.

Usage:
    python -m surveillance.backfill_timelocks [--limit 100] [--chain base]

Requires RPC access (reads bytecode via eth_getCode).
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

logger = logging.getLogger("surveillance.backfill_timelocks")

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


async def backfill(rpc_url: str, chain: str, limit: int = 100, db_path: str = None):
    db = db_path or str(DB_PATH)
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row

    from surveillance.bytecode_classifier import detect_timestamp_activation
    from surveillance.timelock_monitor import ensure_table

    ensure_table(conn)

    # Get TIMESTAMP contracts on this chain that don't have activation_timestamp in notes
    candidates = conn.execute("""
        SELECT contract_address, deployer_address, confidence_tier, bytecode_pattern_notes
        FROM contracts
        WHERE bytecode_pattern_notes LIKE '%TIMESTAMP%'
        AND bytecode_pattern_notes NOT LIKE '%activation_timestamp=%'
        AND chain = ?
        AND confidence_tier IN ('suspected', 'confirmed')
        ORDER BY detection_timestamp DESC
        LIMIT ?
    """, (chain, limit)).fetchall()

    logger.info("Backfilling %d TIMESTAMP contracts on %s", len(candidates), chain)

    w3 = AsyncWeb3(WebSocketProvider(rpc_url))
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())
    now_iso = now.isoformat()

    extracted = 0
    past = 0
    future = 0
    within_7d = 0
    errors = 0

    for i, c in enumerate(candidates):
        addr = c["contract_address"]
        try:
            code = await w3.eth.get_code(addr)
            if not code or code == b'' or code == b'\x00':
                continue

            bytecode_hex = code.hex()
            detected, notes = detect_timestamp_activation(bytecode_hex)

            if detected and "activation_timestamp=" in notes:
                ts_match = re.search(r'activation_timestamp=(\d+)', notes)
                if ts_match:
                    ts = int(ts_match.group(1))
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    delta_hours = (ts - now_ts) / 3600

                    status = "PENDING" if ts > now_ts else "ACTIVATED"
                    if ts <= now_ts:
                        past += 1
                    else:
                        future += 1
                        if delta_hours <= 168:  # 7 days
                            within_7d += 1

                    # Update contract notes
                    old_notes = c["bytecode_pattern_notes"] or ""
                    # Replace old TIMESTAMP note with enhanced one
                    new_notes = re.sub(
                        r'TIMESTAMP at 0x[0-9a-f]+ -> [A-Z]+ at 0x[0-9a-f]+ -> JUMPI at 0x[0-9a-f]+ -> REVERT at 0x[0-9a-f]+: time-locked activation trap in transfer context',
                        notes,
                        old_notes,
                    )
                    if new_notes == old_notes:
                        # Pattern didn't match for replacement, append
                        new_notes = old_notes + "; " + notes

                    conn.execute("""
                        UPDATE contracts SET bytecode_pattern_notes = ? WHERE contract_address = ?
                    """, (new_notes, addr))

                    # Update or insert timelock_countdowns
                    conn.execute("""
                        INSERT OR REPLACE INTO timelock_countdowns
                        (contract_address, chain, deployer_address, confidence_tier,
                         activation_timestamp, activation_iso, bytecode_evidence,
                         status, detected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (addr, chain, c["deployer_address"], c["confidence_tier"],
                          ts, dt.isoformat(), notes[:500], status, now_iso))

                    extracted += 1

                    if (extracted <= 5) or (delta_hours > 0 and delta_hours <= 168):
                        logger.info(
                            "%s: activation=%s (%s, %.1fh %s)",
                            addr[:18], dt.strftime('%Y-%m-%d %H:%M'),
                            status, abs(delta_hours),
                            "from now" if delta_hours > 0 else "ago",
                        )

        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.debug("Failed %s: %s", addr[:18], e)

        if (i + 1) % 50 == 0:
            conn.commit()
            logger.info("Progress: %d/%d checked, %d extracted", i + 1, len(candidates), extracted)

    conn.commit()
    conn.close()

    logger.info(
        "Backfill complete: %d/%d extracted. Past=%d Future=%d Within7d=%d Errors=%d",
        extracted, len(candidates), past, future, within_7d, errors,
    )
    return {
        "checked": len(candidates),
        "extracted": extracted,
        "past": past,
        "future": future,
        "within_7d": within_7d,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill timelock activation timestamps")
    parser.add_argument("--rpc", default=os.environ.get("BASE_WSS_URL") or os.environ.get("ARB_WSS_URL"),
                        help="WebSocket RPC URL")
    parser.add_argument("--chain", default="base", help="Chain to backfill")
    parser.add_argument("--limit", type=int, default=100, help="Max contracts to check")
    parser.add_argument("--db", default=None, help="Database path")
    args = parser.parse_args()

    if not args.rpc:
        print("ERROR: No RPC URL. Set BASE_WSS_URL or ARB_WSS_URL or use --rpc", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    result = asyncio.run(backfill(args.rpc, args.chain, args.limit, args.db))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
