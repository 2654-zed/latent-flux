"""
Layer 3 — Block Replayer for Historical Gap Recovery

Replays historical blocks to recover transaction events that were missed
during monitor outages. Scans blocks for interactions with watched contracts
and inserts missing transaction events.

Usage:
    python -m surveillance.block_replayer --chain base --from-block 43850000 --to-block 43860000
    python -m surveillance.block_replayer --chain base --from-time "2026-03-26T00:00" --to-time "2026-03-26T23:59"
    python -m surveillance.block_replayer --chain base --fill-gaps
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surveillance import db

logger = logging.getLogger("surveillance.replayer")

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _get_rpc(chain: str) -> str:
    if chain == "base":
        return os.environ.get("BASE_WSS_URL", "")
    elif chain == "arbitrum":
        return os.environ.get("ARB_WSS_URL", "")
    return ""


async def replay_blocks(chain: str, from_block: int, to_block: int,
                        batch_size: int = 50):
    """Replay a range of blocks, recovering missed transaction events."""
    rpc = _get_rpc(chain)
    if not rpc:
        print(f"No RPC URL for chain {chain}")
        return

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")

    # Get watched contracts (suspected + confirmed)
    watched = set(
        r[0] for r in conn.execute(
            "SELECT contract_address FROM contracts WHERE confidence_tier IN ('suspected', 'confirmed')"
        ).fetchall()
    )
    print(f"[replayer] Watched contracts: {len(watched):,}")

    # Get existing tx hashes to avoid duplicates
    existing_hashes = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT tx_hash FROM transaction_events WHERE timestamp >= ? AND timestamp <= ?",
            (f"2026-03-25T00:00:00", f"2026-03-28T00:00:00")
        ).fetchall()
    )
    print(f"[replayer] Existing tx hashes in window: {len(existing_hashes):,}")

    total_blocks = to_block - from_block + 1
    print(f"[replayer] Replaying {total_blocks:,} blocks ({from_block} -> {to_block}) on {chain}")

    recovered = 0
    skipped = 0
    blocks_processed = 0
    errors = 0

    async with AsyncWeb3(WebSocketProvider(rpc)) as w3:
        for start in range(from_block, to_block + 1, batch_size):
            end = min(start + batch_size - 1, to_block)

            for block_num in range(start, end + 1):
                try:
                    block = await w3.eth.get_block(block_num, full_transactions=True)
                    timestamp = datetime.fromtimestamp(block.timestamp, tz=timezone.utc)
                    ts_iso = timestamp.isoformat()

                    for tx in block.transactions:
                        tx_hash = tx.hash.hex() if hasattr(tx.hash, 'hex') else str(tx.hash)

                        # Skip if we already have this tx
                        if tx_hash in existing_hashes:
                            skipped += 1
                            continue

                        to_addr = tx.get("to")
                        if not to_addr:
                            continue  # contract creation, not interaction

                        to_lower = to_addr.lower() if isinstance(to_addr, str) else to_addr.hex().lower() if hasattr(to_addr, 'hex') else str(to_addr).lower()

                        if to_lower not in watched:
                            continue

                        # This tx interacts with a watched contract — get receipt
                        try:
                            receipt = await w3.eth.get_transaction_receipt(tx_hash)
                        except Exception:
                            errors += 1
                            continue

                        from_addr = tx["from"].lower() if isinstance(tx["from"], str) else tx["from"].hex().lower()
                        is_reverted = 1 if receipt.status == 0 else 0

                        # Extract selector
                        input_data = tx.get("input", b"")
                        if isinstance(input_data, bytes):
                            selector = input_data[:4].hex() if len(input_data) >= 4 else None
                        elif isinstance(input_data, str):
                            clean = input_data[2:] if input_data.startswith("0x") else input_data
                            selector = clean[:8] if len(clean) >= 8 else None
                        else:
                            selector = None

                        # Gas info
                        gas_price = tx.get("gasPrice", 0)
                        if isinstance(gas_price, int):
                            gas_gwei = gas_price / 1e9
                        else:
                            gas_gwei = 0

                        max_priority = tx.get("maxPriorityFeePerGas", 0)
                        if isinstance(max_priority, int):
                            priority_gwei = max_priority / 1e9
                        else:
                            priority_gwei = 0

                        value = tx.get("value", 0)
                        value_wei = str(value) if value and int(value) > 0 else None

                        try:
                            db.insert_transaction_event(
                                conn,
                                contract_address=to_lower,
                                interacting_address=from_addr,
                                function_selector=selector,
                                bot_tag=None,
                                gas_price_gwei=round(gas_gwei, 6),
                                max_priority_fee_gwei=round(priority_gwei, 6),
                                gas_pattern=None,
                                block_number=block_num,
                                timestamp=ts_iso,
                                is_reverted=is_reverted,
                                tx_hash=tx_hash,
                                value_wei=value_wei,
                            )
                            recovered += 1
                            existing_hashes.add(tx_hash)
                        except Exception as e:
                            if "UNIQUE" not in str(e):
                                errors += 1

                    blocks_processed += 1

                    if blocks_processed % 100 == 0:
                        conn.commit()
                        pct = blocks_processed / total_blocks * 100
                        print(f"  [{pct:.1f}%] block {block_num:,} | recovered={recovered} | skipped={skipped} | errors={errors}", flush=True)

                except Exception as e:
                    errors += 1
                    if errors % 10 == 0:
                        logger.warning("Block %d error: %s", block_num, e)

            conn.commit()

    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    conn.close()

    print(f"\n[replayer] Complete. Blocks: {blocks_processed:,} | Recovered: {recovered:,} | Skipped: {skipped:,} | Errors: {errors}")


async def fill_gaps(chain: str):
    """Auto-detect gaps in transaction_events and replay missing blocks."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    # Find hours with suspiciously low tx events compared to contracts deployed
    print("[replayer] Detecting coverage gaps...")
    gaps = conn.execute("""
        SELECT hour, contracts, events,
            CASE WHEN contracts > 50 AND events < contracts * 2 THEN 'GAP'
                 WHEN contracts > 20 AND events = 0 THEN 'BLACKOUT'
                 ELSE 'OK' END as status
        FROM (
            SELECT substr(c.detection_timestamp, 1, 13) as hour,
                COUNT(DISTINCT c.contract_address) as contracts,
                COALESCE((SELECT COUNT(*) FROM transaction_events te
                    WHERE substr(te.timestamp, 1, 13) = substr(c.detection_timestamp, 1, 13)), 0) as events
            FROM contracts c
            WHERE c.chain = ? AND c.detection_timestamp >= '2026-03-25T00:00:00'
            GROUP BY hour
        )
        WHERE status != 'OK'
        ORDER BY hour
    """, (chain,)).fetchall()

    if not gaps:
        print("[replayer] No gaps detected.")
        conn.close()
        return

    print(f"[replayer] Found {len(gaps)} gap hours:")
    for g in gaps:
        print(f"  {g['hour']} | {g['contracts']} contracts, {g['events']} events | {g['status']}")

    # Get block ranges for gap hours
    rpc = _get_rpc(chain)
    if not rpc:
        print(f"No RPC for {chain}")
        conn.close()
        return

    # Get block numbers from contracts table for gap hours
    block_ranges = []
    for g in gaps:
        hour = g['hour']
        r = conn.execute("""
            SELECT MIN(detection_block) as min_block, MAX(detection_block) as max_block
            FROM contracts WHERE chain = ? AND substr(detection_timestamp, 1, 13) = ?
        """, (chain, hour)).fetchone()
        if r and r['min_block'] and r['max_block']:
            # Add some padding
            block_ranges.append((r['min_block'] - 100, r['max_block'] + 100))

    conn.close()

    if not block_ranges:
        print("[replayer] Could not determine block ranges for gaps.")
        return

    # Merge overlapping ranges
    block_ranges.sort()
    merged = [block_ranges[0]]
    for start, end in block_ranges[1:]:
        if start <= merged[-1][1] + 1000:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_blocks = sum(e - s + 1 for s, e in merged)
    print(f"[replayer] Will replay {total_blocks:,} blocks across {len(merged)} ranges")

    for start, end in merged:
        print(f"\n[replayer] Range: {start:,} -> {end:,} ({end - start + 1:,} blocks)")
        await replay_blocks(chain, start, end)


if __name__ == "__main__":
    _load_env()

    parser = argparse.ArgumentParser(description="Block replayer for gap recovery")
    parser.add_argument("--chain", required=True, choices=["base", "arbitrum"])
    parser.add_argument("--from-block", type=int, help="Start block")
    parser.add_argument("--to-block", type=int, help="End block")
    parser.add_argument("--fill-gaps", action="store_true", help="Auto-detect and fill gaps")
    parser.add_argument("--batch", type=int, default=50, help="Blocks per batch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.fill_gaps:
        asyncio.run(fill_gaps(args.chain))
    elif args.from_block and args.to_block:
        asyncio.run(replay_blocks(args.chain, args.from_block, args.to_block, args.batch))
    else:
        parser.print_help()
