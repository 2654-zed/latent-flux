"""
Fix 3: Approval Value Sampling -- check allowance + balance for high-priority approvals.

Requires RPC. Budget: ~400 calls.

Usage:
    python -m surveillance.sample_approval_values --chain base --limit 200
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

logger = logging.getLogger("surveillance.approval_values")
DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

MAX_UINT256 = (1 << 256) - 1
# allowance(address,address) selector
ALLOWANCE_SIG = "0xdd62ed3e"
# balanceOf(address) selector
BALANCE_SIG = "0x70a08231"


def encode_allowance(owner: str, spender: str) -> str:
    return ALLOWANCE_SIG + owner.lower().replace("0x", "").zfill(64) + spender.lower().replace("0x", "").zfill(64)


def encode_balance(owner: str) -> str:
    return BALANCE_SIG + owner.lower().replace("0x", "").zfill(64)


async def sample(rpc_url: str, chain: str, limit: int = 200, db_path: str = None):
    db = db_path or str(DB_PATH)
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row

    from surveillance.blindspot_remediation import ensure_tables
    ensure_tables(conn)

    # Priority 1: Undrained approvals on confirmed contracts
    priority1 = conn.execute("""
        SELECT aw.victim_address as approver, aw.contract_address,
               c.chain, c.deployer_address
        FROM approval_watchlist aw
        JOIN contracts c ON aw.contract_address = c.contract_address
        WHERE c.confidence_tier = 'confirmed' AND aw.drain_detected = 0 AND c.chain = ?
        LIMIT ?
    """, (chain, limit // 2)).fetchall()

    # Priority 2: Top undrained collectors
    priority2 = conn.execute("""
        SELECT aw.victim_address as approver, aw.contract_address,
               c.chain, c.deployer_address
        FROM approval_watchlist aw
        JOIN contracts c ON aw.contract_address = c.contract_address
        WHERE aw.drain_detected = 0 AND c.chain = ?
        AND aw.contract_address IN (
            SELECT contract_address FROM approval_watchlist
            WHERE drain_detected = 0 GROUP BY contract_address
            HAVING COUNT(*) >= 10 ORDER BY COUNT(*) DESC LIMIT 20
        )
        AND aw.victim_address NOT IN (SELECT approver FROM approval_values)
        ORDER BY RANDOM()
        LIMIT ?
    """, (chain, limit // 2)).fetchall()

    all_targets = list(priority1) + list(priority2)
    logger.info("Sampling %d approvals on %s (P1=%d, P2=%d)",
                len(all_targets), chain, len(priority1), len(priority2))

    w3 = AsyncWeb3(WebSocketProvider(rpc_url))
    now = datetime.now(timezone.utc).isoformat()

    checked = 0
    max_approvals = 0
    zero_balance = 0
    total_risk_wei = 0
    errors = 0

    for t in all_targets:
        approver = t["approver"]
        contract = t["contract_address"]

        try:
            # We don't have token_address in approval_watchlist
            # The contract IS the token in most cases (ERC-20 approve on the token contract)
            token = contract

            # Check allowance
            allowance_data = encode_allowance(approver, contract)
            allowance_result = await w3.eth.call({
                "to": token,
                "data": allowance_data,
            })
            allowance_int = int(allowance_result.hex(), 16) if allowance_result else 0
            is_max = allowance_int >= MAX_UINT256 // 2  # treat anything huge as "max"

            # Check balance
            balance_data = encode_balance(approver)
            balance_result = await w3.eth.call({
                "to": token,
                "data": balance_data,
            })
            balance_int = int(balance_result.hex(), 16) if balance_result else 0

            risk = min(allowance_int, balance_int)

            if is_max:
                max_approvals += 1
            if balance_int == 0:
                zero_balance += 1
            total_risk_wei += risk

            conn.execute("""
                INSERT OR REPLACE INTO approval_values
                (approver, contract_address, token_address, chain,
                 allowance_raw, allowance_is_max, balance_raw,
                 value_at_risk_wei, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (approver, contract, token, chain,
                  hex(allowance_int), 1 if is_max else 0,
                  hex(balance_int), str(risk), now))

            checked += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.debug("Failed %s->%s: %s", approver[:12], contract[:12], e)

        if checked % 50 == 0 and checked > 0:
            conn.commit()
            logger.info("Progress: %d/%d checked, %d errors", checked, len(all_targets), errors)

    conn.commit()
    conn.close()

    result = {
        "checked": checked,
        "errors": errors,
        "max_uint256_approvals": max_approvals,
        "zero_balance_approvers": zero_balance,
        "total_value_at_risk_wei": str(total_risk_wei),
        "rpc_calls": checked * 2,  # 2 per approval (allowance + balance)
    }
    logger.info("Done: %s", json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description="Sample approval values via RPC")
    parser.add_argument("--rpc", default=os.environ.get("BASE_WSS_URL") or os.environ.get("ARB_WSS_URL"))
    parser.add_argument("--chain", default="base")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    if not args.rpc:
        print("ERROR: No RPC URL.", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    result = asyncio.run(sample(args.rpc, args.chain, args.limit, args.db))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
