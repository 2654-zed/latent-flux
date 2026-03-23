"""
Layer 3 — Automatic Funder Tracer

Traces the funding source for every new deployer at deployment time.
Runs as a sub-monitor inside the deployment pipeline.

For each new deployer:
1. Query first ETH inflow via alchemy_getAssetTransfers (1 API call)
2. Store funder address + amount + timestamp in deployers.funding_trail
3. Check if funder is a known org wallet -> flag deployer as org-linked
4. Check if funder funds 5+ other deployers -> flag as gas station pattern
5. Periodically batch hop-trace high-score deployers (funder's funder)

Read-only on-chain. Writes only to deployers.funding_trail.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Set

from web3 import AsyncWeb3

from surveillance import db

logger = logging.getLogger("surveillance.auto_funder")

# Known org wallets for instant flagging
ORG_WALLETS = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": ("org_001", "treasury"),
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": ("org_001", "operator"),
    "0xfd51e33d44b376ef346d24a130a51035db09c1dc": ("org_001", "operator_2"),
    "0xc6962004f452be9203591991d15f6b388e09e8d0": ("org_001", "cashout"),
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": ("org_001", "gas_station"),
    "0x360e68faccca8ca495c1b759fd9eee466db9fb32": ("org_001", "treasury_branch"),
    "0x238d7170f309a55b87a144a341bd6105897082ca": ("org_002", "treasury_senior"),
    "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473": ("org_002", "treasury_junior"),
}

# Gas station threshold: funder that funds N+ distinct deployers
GAS_STATION_THRESHOLD = 5


class AutoFunderTracer:
    """
    Traces funding source for new deployers.
    Plugs into the deployment monitor's block processing loop.
    """

    def __init__(self, conn: sqlite3.Connection, chain: str = "arbitrum"):
        self.conn = conn
        self.chain = chain
        self._traced: Set[str] = set()  # deployers already traced this session
        self._pending: list = []  # deployers queued for tracing
        self._rpc_url: Optional[str] = None
        self.traces_completed = 0
        self.org_links_found = 0
        self.gas_stations_found = 0

        # Load already-traced deployers to avoid re-tracing
        try:
            rows = conn.execute(
                "SELECT deployer_address FROM deployers WHERE funding_trail IS NOT NULL AND funding_trail != ''"
            ).fetchall()
            self._traced = set(r[0].lower() for r in rows)
        except Exception:
            pass

        logger.info("AutoFunderTracer initialized (%d already traced)", len(self._traced))

    def queue_deployer(self, deployer_address: str) -> None:
        """Queue a deployer for funder tracing."""
        addr = deployer_address.lower()
        if addr not in self._traced:
            self._pending.append(addr)

    async def process_pending(self, w3: AsyncWeb3) -> None:
        """Trace all pending deployers. Call after each block's deployments."""
        if not self._pending:
            return

        # Process up to 5 per block to stay within rate limits
        batch = self._pending[:5]
        self._pending = self._pending[5:]

        for addr in batch:
            if addr in self._traced:
                continue
            try:
                await self._trace_deployer(w3, addr)
                self._traced.add(addr)
            except Exception as e:
                logger.debug("Funder trace failed for %s: %s", addr[:14], e)

    async def _trace_deployer(self, w3: AsyncWeb3, deployer: str) -> None:
        """Trace the first ETH inflow to a deployer address."""
        # Use the web3 provider's HTTP endpoint for alchemy_getAssetTransfers
        # Build the raw JSON-RPC call
        provider = w3.provider
        result = await provider.make_request(
            "alchemy_getAssetTransfers",
            [{
                "fromBlock": "0x0",
                "toBlock": "latest",
                "toAddress": deployer,
                "category": ["external"],
                "maxCount": "0x1",
                "order": "asc",
                "withMetadata": True,
            }],
        )

        transfers = result.get("result", {}).get("transfers", [])
        if not transfers:
            # No external funding found — might be funded via internal tx or bridge
            self.conn.execute(
                "UPDATE deployers SET funding_trail = ? WHERE deployer_address = ?",
                (json.dumps({"note": "no external funding found", "traced_at": datetime.now(timezone.utc).isoformat()}), deployer),
            )
            self.conn.commit()
            self.traces_completed += 1
            return

        t = transfers[0]
        funder = (t.get("from") or "").lower()
        value = float(t.get("value") or 0)
        tx_hash = t.get("hash", "")
        ts = (t.get("metadata") or {}).get("blockTimestamp", "")
        block_hex = t.get("blockNum", "0x0")
        block = int(block_hex, 16) if isinstance(block_hex, str) else block_hex

        trail = {
            "funder": funder,
            "value_eth": value,
            "tx_hash": tx_hash,
            "timestamp": ts,
            "block": block,
            "chain": self.chain,
            "traced_at": datetime.now(timezone.utc).isoformat(),
        }

        # Check if funder is a known org wallet
        if funder in ORG_WALLETS:
            org_id, org_role = ORG_WALLETS[funder]
            trail["org_link"] = org_id
            trail["funder_role"] = org_role
            self.org_links_found += 1
            logger.warning(
                "ORG LINK: deployer %s funded by %s (%s:%s)",
                deployer[:14], funder[:14], org_id, org_role,
            )

        # Store the trail
        self.conn.execute(
            "UPDATE deployers SET funding_trail = ? WHERE deployer_address = ?",
            (json.dumps(trail), deployer),
        )
        self.conn.commit()
        self.traces_completed += 1

        # Check gas station pattern: does this funder fund many deployers?
        self._check_gas_station(funder)

    def _check_gas_station(self, funder: str) -> None:
        """Check if a funder has funded N+ distinct deployers (gas station pattern)."""
        try:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM deployers WHERE funding_trail LIKE ?",
                (f'%"{funder}"%',),
            ).fetchone()[0]

            if count >= GAS_STATION_THRESHOLD:
                # Check if already flagged
                existing = self.conn.execute(
                    "SELECT entity_type FROM deployers WHERE deployer_address = ?",
                    (funder,),
                ).fetchone()

                if existing and existing[0] not in ("unknown", None):
                    return  # Already classified

                self.gas_stations_found += 1
                logger.info(
                    "GAS STATION detected: %s funds %d deployers",
                    funder[:14], count,
                )

                # Ensure funder has a deployer row (it may not if it only funds, never deploys)
                self.conn.execute(
                    """INSERT OR IGNORE INTO deployers
                       (deployer_address, chain, first_seen, last_seen, total_contracts_deployed)
                       VALUES (?, ?, ?, ?, 0)""",
                    (funder, self.chain,
                     datetime.now(timezone.utc).isoformat(),
                     datetime.now(timezone.utc).isoformat()),
                )
                self.conn.execute(
                    "UPDATE deployers SET entity_type = 'gas_station', "
                    "deployment_pattern_notes = ? WHERE deployer_address = ? "
                    "AND (entity_type IS NULL OR entity_type = 'unknown')",
                    (f"Auto-detected gas station: funds {count}+ deployers", funder),
                )
                self.conn.commit()
        except Exception as e:
            logger.debug("Gas station check failed: %s", e)

    async def batch_hop_trace(self, w3: AsyncWeb3, min_score: float = 0.3, max_traces: int = 20) -> int:
        """
        Hop-trace high-score deployers: trace the funder's funder.
        Call periodically (e.g., every 1000 blocks).
        Returns number of hop traces completed.
        """
        # Find deployers with score >= min_score that have a funder but no hop trace
        rows = self.conn.execute(
            """SELECT deployer_address, funding_trail FROM deployers
               WHERE behavioral_score >= ?
                 AND funding_trail IS NOT NULL AND funding_trail != ''
                 AND funding_trail NOT LIKE '%hop_traced%'
               ORDER BY behavioral_score DESC LIMIT ?""",
            (min_score, max_traces),
        ).fetchall()

        traced = 0
        for row in rows:
            deployer = row[0]
            try:
                trail = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                continue

            funder = trail.get("funder")
            if not funder or funder.startswith("0x0000"):
                continue

            # Trace the funder's funder
            try:
                result = await w3.provider.make_request(
                    "alchemy_getAssetTransfers",
                    [{
                        "fromBlock": "0x0",
                        "toBlock": "latest",
                        "toAddress": funder,
                        "category": ["external"],
                        "maxCount": "0x3",
                        "order": "asc",
                        "withMetadata": True,
                    }],
                )
                hop_transfers = result.get("result", {}).get("transfers", [])

                trail["hop_traced"] = True
                trail["funder_funders"] = []
                for ht in hop_transfers[:3]:
                    ff = (ht.get("from") or "").lower()
                    fv = float(ht.get("value") or 0)
                    trail["funder_funders"].append({"address": ff, "value_eth": fv})

                    # Check if funder's funder is a known org
                    if ff in ORG_WALLETS:
                        org_id, org_role = ORG_WALLETS[ff]
                        trail["org_link_hop2"] = org_id
                        logger.warning(
                            "ORG LINK (2-hop): deployer %s -> funder %s -> %s (%s:%s)",
                            deployer[:14], funder[:14], ff[:14], org_id, org_role,
                        )

                self.conn.execute(
                    "UPDATE deployers SET funding_trail = ? WHERE deployer_address = ?",
                    (json.dumps(trail), deployer),
                )
                self.conn.commit()
                traced += 1

            except Exception as e:
                logger.debug("Hop trace failed for %s: %s", funder[:14], e)

            await asyncio.sleep(0.2)  # Rate limit

        if traced > 0:
            logger.info("Hop-traced %d deployers", traced)
        return traced
