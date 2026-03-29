"""
Layer 3 — Component 1: Contract Deployment Monitor

Watches every new contract deployment on Arbitrum in real time via WebSocket.
For each block, extracts all contract creation transactions (to == null),
records deployer address, contract address, bytecode, block number, and
timestamp, then passes each deployment to the bytecode classifier.

Read-only. Never interacts with or executes against any contract.

Usage:
    python -m surveillance.deployment_monitor --rpc wss://arb-mainnet.g.alchemy.com/v2/YOUR_KEY

Environment variable alternative:
    export ARB_WSS_URL=wss://arb-mainnet.g.alchemy.com/v2/YOUR_KEY
    python -m surveillance.deployment_monitor
"""

import argparse
import asyncio
import hashlib
import logging
import os
import signal
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

# Add project root to path for surveillance imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db

logger = logging.getLogger("surveillance.monitor")


def _sha256_hex(data: str) -> str:
    """SHA-256 hash of a hex string, returned as hex digest."""
    return hashlib.sha256(bytes.fromhex(data) if data else b"").hexdigest()


@dataclass
class Deployment:
    """A single contract deployment extracted from a block."""
    contract_address: str
    deployer_address: str
    block_number: int
    timestamp_iso: str
    tx_hash: str
    bytecode: str  # deployed bytecode (hex, no 0x prefix)
    gas_price_gwei: Optional[float]
    init_code_hash: str = ""     # SHA-256 of tx.input (creation code)
    deployed_code_hash: str = "" # SHA-256 of deployed bytecode
    cache_hit: str = ""          # "init" | "deployed" | "" (miss)


class DeploymentMonitor:
    """
    Async WebSocket monitor that subscribes to new Arbitrum blocks
    and extracts contract deployments.
    """

    def __init__(self, rpc_url: str, conn: sqlite3.Connection,
                 classifier=None, safe_deployers: set[str] | None = None,
                 velocity_threshold: int = 8, chain: str = "arbitrum"):
        """
        Args:
            rpc_url: WebSocket RPC endpoint (wss://...)
            conn: SQLite connection (from db.init_db)
            classifier: Optional callable(Deployment, conn) for bytecode
                        classification. If None, contracts are inserted
                        with confidence_tier='unknown'.
            safe_deployers: Set of deployer addresses to skip entirely
                            (known legitimate factories, noise reduction).
            velocity_threshold: If a deployer deploys more than this many
                                contracts in a single session, auto-flag
                                the deployer as priority.
            chain: Chain identifier stored in DB (arbitrum, base, etc.)
        """
        self.rpc_url = rpc_url
        self.conn = conn
        self.classifier = classifier
        self.safe_deployers = safe_deployers or set()
        self.velocity_threshold = velocity_threshold
        self.chain = chain
        self._running = False
        self._w3: Optional[AsyncWeb3] = None
        self._blocks_processed = 0
        self._deployments_found = 0
        # Deployment velocity: tracks per-deployer counts within this session
        self._session_deployer_counts: dict[str, int] = {}
        self._velocity_flagged: set[str] = set()
        # Bytecode cache stats for this session
        self._cache_hits_init = 0
        self._cache_hits_deployed = 0
        self._cache_misses = 0
        self._rpc_calls_saved = 0
        self._last_block: Optional[int] = None
        self._current_gap_id: Optional[int] = None
        # Sub-monitors (initialized lazily to avoid circular imports)
        self._selector_monitor = None
        self._revert_detector = None
        self._event_monitors = None
        self._funder_tracer = None

    def _init_sub_monitors(self) -> None:
        """Initialize selector monitor, revert detector, event monitors, and funder tracer."""
        if self._selector_monitor is None:
            from surveillance.selector_monitor import SelectorMonitor
            from surveillance.revert_cluster_detector import RevertClusterDetector
            from surveillance.event_monitors import EventMonitors
            from surveillance.auto_funder_tracer import AutoFunderTracer
            self._selector_monitor = SelectorMonitor(self.conn)
            self._revert_detector = RevertClusterDetector(self.conn)
            self._event_monitors = EventMonitors(self.conn, self.chain)
            self._funder_tracer = AutoFunderTracer(self.conn, self.chain)
            logger.info("Sub-monitors initialized: selector_monitor, revert_cluster_detector, event_monitors, auto_funder_tracer")

    async def start(self, max_retries: int = 0) -> None:
        """Connect to WebSocket and begin monitoring blocks.

        Auto-reconnects on connection loss with exponential backoff.
        Logs gaps to the database so missed blocks are auditable.
        Writes a heartbeat every 60 seconds.

        max_retries=0 means infinite retries (persistent service mode).
        """
        self._running = True
        attempt = 0

        while self._running and (max_retries == 0 or attempt <= max_retries):
            if attempt > 0:
                delay = min(2 ** attempt, 60)
                logger.warning(
                    "Reconnect attempt %d/%d in %ds...",
                    attempt, max_retries, delay,
                )
                await asyncio.sleep(delay)

            try:
                await self._run_connection(attempt)
                # Clean exit (stop() called) -- don't retry
                if not self._running:
                    break
            except asyncio.CancelledError:
                logger.info("Monitor cancelled")
                break
            except Exception as e:
                disconnect_time = datetime.now(timezone.utc).isoformat()
                reason = f"{type(e).__name__}: {e}"
                logger.error("Connection lost: %s", reason)

                gap_id = db.log_connection_gap(
                    self.conn, "deployment_monitor",
                    disconnect_time, self._last_block, reason,
                )
                self._current_gap_id = gap_id
                attempt += 1

        if max_retries > 0 and attempt > max_retries:
            logger.error(
                "Max retries (%d) exhausted -- monitor stopping", max_retries
            )

        self._running = False
        logger.info(
            "Monitor stopped -- processed %d blocks, found %d deployments",
            self._blocks_processed, self._deployments_found,
        )

    async def _run_connection(self, attempt: int) -> None:
        """Single WebSocket connection lifecycle."""
        logger.info("Connecting to %s", self.rpc_url)

        async with AsyncWeb3(WebSocketProvider(self.rpc_url)) as w3:
            self._w3 = w3

            chain_id = await w3.eth.chain_id
            latest = await w3.eth.block_number
            logger.info("Connected -- chain_id=%d, latest_block=%d", chain_id, latest)

            # Close any open gap from a prior disconnect
            if hasattr(self, "_current_gap_id") and self._current_gap_id:
                db.close_connection_gap(
                    self.conn, self._current_gap_id,
                    datetime.now(timezone.utc).isoformat(), latest,
                )
                self._current_gap_id = None
                # Reset retry counter on successful reconnect
                logger.info("Reconnected successfully -- gap closed, resuming")

            expected_chains = {"arbitrum": 42161, "base": 8453}
            expected = expected_chains.get(self.chain)
            if expected and chain_id != expected:
                logger.warning(
                    "Expected %s (%d), got chain_id=%d", self.chain, expected, chain_id
                )

            await w3.eth.subscribe("newHeads")
            logger.info("Subscribed to newHeads -- monitoring deployments")

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                async for msg in w3.socket.process_subscriptions():
                    if not self._running:
                        break
                    header = msg["result"]
                    await self._handle_block_header(w3, header)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self) -> None:
        """Write heartbeat to DB every 60 seconds. Checkpoint WAL every 15 min.
        Also prunes bytecode cache for clean/unknown contracts every 6 hours."""
        heartbeat_count = 0
        while self._running:
            try:
                db.write_heartbeat(
                    self.conn, f"deployment_monitor_{self.chain}",
                    self._blocks_processed, self._deployments_found,
                )
            except Exception as e:
                logger.warning("Heartbeat write failed: %s", e)

            heartbeat_count += 1

            # WAL checkpoint every 15 heartbeats (~15 min) using PASSIVE mode
            # PASSIVE doesn't require exclusive lock — safe with concurrent writers
            if heartbeat_count % 15 == 0:
                try:
                    result = self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                    logger.info("WAL checkpoint (PASSIVE): busy=%s, log=%s, checkpointed=%s",
                                result[0], result[1], result[2])
                    # If WAL is large (log > 5000 pages), try TRUNCATE
                    if result and result[1] and result[1] > 5000:
                        try:
                            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                            logger.info("WAL checkpoint (TRUNCATE) succeeded - WAL was large")
                        except Exception:
                            logger.info("WAL TRUNCATE failed (concurrent access) - PASSIVE was enough")
                except Exception as e:
                    logger.warning("WAL checkpoint failed: %s", e)

            # Prune bytecode cache every 360 heartbeats (~6 hours)
            # Remove cached bytecode for contracts with no trap patterns
            if heartbeat_count % 360 == 0:
                try:
                    cursor = self.conn.execute("""
                        DELETE FROM bytecode_cache WHERE source_contract IN (
                            SELECT bc.source_contract FROM bytecode_cache bc
                            LEFT JOIN contracts c ON c.contract_address = bc.source_contract
                            WHERE (c.confidence_tier = 'unknown' OR c.confidence_tier IS NULL)
                            AND bc.confidence_tier = 'unknown'
                            AND bc.hit_count = 0
                        )
                    """)
                    deleted = cursor.rowcount
                    self.conn.commit()
                    if deleted:
                        logger.info("Bytecode cache pruned: %d unknown entries removed", deleted)
                except Exception as e:
                    logger.warning("Bytecode cache prune failed: %s", e)

            # Self-test trap detection every 30 heartbeats (~30 min)
            # Finds contracts where 100% of interactions come from the deployer
            if heartbeat_count % 30 == 0:
                try:
                    # Ensure table exists
                    self.conn.execute("""
                        CREATE TABLE IF NOT EXISTS self_test_traps (
                            contract_address TEXT PRIMARY KEY,
                            deployer_address TEXT NOT NULL,
                            chain TEXT,
                            self_hits INTEGER DEFAULT 0,
                            self_reverts INTEGER DEFAULT 0,
                            external_hits INTEGER DEFAULT 0,
                            external_first_seen TEXT,
                            status TEXT DEFAULT 'SELF_TEST',
                            bytecode_patterns TEXT,
                            detected_at TEXT,
                            last_checked TEXT
                        )
                    """)

                    from datetime import datetime, timezone
                    now_iso = datetime.now(timezone.utc).isoformat()

                    # Find contracts where deployer is the ONLY caller, 10+ hits
                    new_self_tests = self.conn.execute("""
                        SELECT c.contract_address, c.deployer_address, c.chain,
                            c.bytecode_pattern_notes,
                            COUNT(*) as self_hits,
                            SUM(CASE WHEN te.is_reverted=1 THEN 1 ELSE 0 END) as self_reverts
                        FROM contracts c
                        JOIN transaction_events te ON te.contract_address = c.contract_address
                        LEFT JOIN self_test_traps st ON st.contract_address = c.contract_address
                        WHERE te.interacting_address = c.deployer_address
                        AND st.contract_address IS NULL
                        GROUP BY c.contract_address
                        HAVING self_hits >= 10
                        AND COUNT(DISTINCT te.interacting_address) = 1
                    """).fetchall()

                    for r in new_self_tests:
                        self.conn.execute("""
                            INSERT OR IGNORE INTO self_test_traps
                            (contract_address, deployer_address, chain, self_hits, self_reverts,
                             external_hits, status, bytecode_patterns, detected_at, last_checked)
                            VALUES (?, ?, ?, ?, ?, 0, 'SELF_TEST', ?, ?, ?)
                        """, (r["contract_address"], r["deployer_address"], r["chain"],
                              r["self_hits"], r["self_reverts"],
                              r["bytecode_pattern_notes"], now_iso, now_iso))
                        logger.info(
                            "SELF-TEST TRAP DETECTED: %s (deployer=%s, %d self-hits, %d reverts)",
                            r["contract_address"][:18], r["deployer_address"][:14],
                            r["self_hits"], r["self_reverts"])

                    # Update existing self-test traps — check for first external interaction
                    existing = self.conn.execute(
                        "SELECT contract_address, deployer_address FROM self_test_traps WHERE status = 'SELF_TEST'"
                    ).fetchall()

                    for st in existing:
                        ext = self.conn.execute("""
                            SELECT COUNT(*) as hits, MIN(timestamp) as first_seen
                            FROM transaction_events
                            WHERE contract_address = ? AND interacting_address != ?
                        """, (st["contract_address"], st["deployer_address"])).fetchone()

                        if ext and ext["hits"] and ext["hits"] > 0:
                            self.conn.execute("""
                                UPDATE self_test_traps
                                SET external_hits = ?, external_first_seen = ?,
                                    status = 'ARMED', last_checked = ?
                                WHERE contract_address = ?
                            """, (ext["hits"], ext["first_seen"], now_iso, st["contract_address"]))
                            logger.warning(
                                "SELF-TEST TRAP ARMED: %s has first external victim! %d external hits since %s",
                                st["contract_address"][:18], ext["hits"], ext["first_seen"])
                        else:
                            # Update self-hit count
                            self_count = self.conn.execute("""
                                SELECT COUNT(*) as hits,
                                    SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts
                                FROM transaction_events
                                WHERE contract_address = ? AND interacting_address = ?
                            """, (st["contract_address"], st["deployer_address"])).fetchone()
                            self.conn.execute("""
                                UPDATE self_test_traps
                                SET self_hits = ?, self_reverts = ?, last_checked = ?
                                WHERE contract_address = ?
                            """, (self_count["hits"], self_count["reverts"], now_iso,
                                  st["contract_address"]))

                    if new_self_tests:
                        self.conn.commit()
                        logger.info("Self-test scan: %d new traps detected", len(new_self_tests))

                except Exception as e:
                    logger.warning("Self-test trap scan failed: %s", e)

            # Approval drain monitor every 30 heartbeats (~30 min)
            # Tracks approve() calls on suspected contracts and watches for drains
            if heartbeat_count % 30 == 5:  # offset from self-test scan
                try:
                    from surveillance.approval_drain_monitor import scan_approvals, check_drains
                    r1 = scan_approvals(self.conn)
                    r2 = check_drains(self.conn)
                    if r1["new_approvals_tracked"] > 0:
                        logger.info("Approval monitor: %d new approvals tracked", r1["new_approvals_tracked"])
                    if r2["drains_detected"] > 0:
                        logger.warning("APPROVAL DRAIN DETECTED: %d victims drained!", r2["drains_detected"])
                except Exception as e:
                    logger.warning("Approval drain scan failed: %s", e)

            await asyncio.sleep(60)

    def stop(self) -> None:
        """Signal the monitor to stop after the current block."""
        self._running = False

    async def _handle_block_header(self, w3: AsyncWeb3, header) -> None:
        """Process a single block: fetch full block, extract deployments."""
        # Header fields from subscription are hex strings; from get_block they're ints.
        raw_number = header["number"]
        block_number = int(raw_number, 16) if isinstance(raw_number, str) else raw_number
        raw_ts = header["timestamp"]
        ts_int = int(raw_ts, 16) if isinstance(raw_ts, str) else raw_ts
        timestamp = datetime.fromtimestamp(ts_int, tz=timezone.utc)
        timestamp_iso = timestamp.isoformat()

        try:
            block = await w3.eth.get_block(block_number, full_transactions=True)
        except Exception as e:
            logger.error("Failed to fetch block %d: %s", block_number, e)
            return

        tx_count = len(block["transactions"])
        deployments = []

        for tx in block["transactions"]:
            # Contract creation: 'to' is None or empty
            if tx.get("to") is not None:
                continue

            # Get the receipt to find the deployed contract address
            try:
                receipt = await w3.eth.get_transaction_receipt(tx["hash"])
            except Exception as e:
                logger.warning(
                    "Failed to get receipt for tx %s: %s",
                    tx["hash"].hex(), e
                )
                continue

            # Failed deployment — skip
            if receipt["status"] != 1:
                continue

            contract_address = receipt.get("contractAddress")
            if not contract_address:
                continue

            # Gas price in gwei
            gas_price_wei = tx.get("gasPrice") or tx.get("maxFeePerGas")
            gas_price_gwei = float(gas_price_wei) / 1e9 if gas_price_wei else None

            # --- Tier 1 cache: hash init code (tx.input) ---
            init_code_raw = tx.get("input", b"")
            if isinstance(init_code_raw, bytes):
                init_code_hex = init_code_raw.hex()
            else:
                init_code_hex = str(init_code_raw).replace("0x", "")
            init_code_hash = _sha256_hex(init_code_hex) if init_code_hex else ""

            cached = db.cache_lookup(self.conn, init_code_hash) if init_code_hash else None
            if cached:
                # Full cache hit — skip eth_getCode entirely
                self._cache_hits_init += 1
                self._rpc_calls_saved += 1
                deployment = Deployment(
                    contract_address=contract_address.lower(),
                    deployer_address=tx["from"].lower(),
                    block_number=block_number,
                    timestamp_iso=timestamp_iso,
                    tx_hash=tx["hash"].hex(),
                    bytecode="",  # not fetched
                    gas_price_gwei=gas_price_gwei,
                    init_code_hash=init_code_hash,
                    deployed_code_hash="",
                    cache_hit="init",
                )
                deployment._cached_result = cached
                deployments.append(deployment)
                continue

            # Cache miss on init code — fetch deployed bytecode
            try:
                bytecode = await w3.eth.get_code(contract_address)
                bytecode_hex = bytecode.hex()
            except Exception as e:
                logger.warning(
                    "Failed to get bytecode for %s: %s", contract_address, e
                )
                bytecode_hex = ""

            deployed_code_hash = _sha256_hex(bytecode_hex) if bytecode_hex else ""

            # --- Tier 2 cache: hash deployed code ---
            cached = db.cache_lookup(self.conn, deployed_code_hash) if deployed_code_hash else None
            cache_hit = ""
            if cached:
                self._cache_hits_deployed += 1
                cache_hit = "deployed"
            else:
                self._cache_misses += 1

            deployment = Deployment(
                contract_address=contract_address.lower(),
                deployer_address=tx["from"].lower(),
                block_number=block_number,
                timestamp_iso=timestamp_iso,
                tx_hash=tx["hash"].hex(),
                bytecode=bytecode_hex,
                gas_price_gwei=gas_price_gwei,
                init_code_hash=init_code_hash,
                deployed_code_hash=deployed_code_hash,
                cache_hit=cache_hit,
            )
            if cached:
                deployment._cached_result = cached
            deployments.append(deployment)

        self._blocks_processed += 1
        self._deployments_found += len(deployments)
        self._last_block = block_number

        if deployments:
            logger.info(
                "Block %d — %d txs, %d deployments",
                block_number, tx_count, len(deployments),
            )

        for dep in deployments:
            self._record_deployment(dep)

        # Run sub-monitors on the full block
        self._init_sub_monitors()
        try:
            await self._selector_monitor.process_block(w3, block, timestamp_iso)
            await self._revert_detector.process_block(w3, block, timestamp_iso)
            await self._event_monitors.process_block(w3, block, timestamp_iso)
            # Flag CEX candidates every 500 blocks
            if block_number % 500 == 0:
                self._event_monitors.flag_cex_candidates()
        except Exception as e:
            logger.warning("Sub-monitor error on block %d: %s", block_number, e)

        # Queue new deployers for funder tracing
        try:
            for dep in deployments:
                self._funder_tracer.queue_deployer(dep.deployer_address)
            await self._funder_tracer.process_pending(w3)
            # Hop-trace high-score deployers every 1000 blocks
            if block_number % 1000 == 0:
                await self._funder_tracer.batch_hop_trace(w3)
        except Exception as e:
            logger.debug("Funder tracer error on block %d: %s", block_number, e)

        # Periodic heartbeat every 100 blocks
        if self._blocks_processed % 100 == 0:
            logger.info(
                "Heartbeat — %d blocks processed, %d total deployments",
                self._blocks_processed, self._deployments_found,
            )

    def _record_deployment(self, dep: Deployment) -> None:
        """
        Record a deployment to the database. Classification priority:
        1. Safe deployer -> skip entirely (noise reduction)
        2. Priority deployer -> auto-SUSPECTED via deployer_history
        3. Bytecode classifier -> pattern-based classification
        4. Fallback -> UNKNOWN
        """
        try:
            # Skip if already recorded (e.g., block replay after restart)
            if db.get_contract(self.conn, dep.contract_address):
                logger.debug("Skipping already-recorded %s", dep.contract_address)
                return

            # Check 1: safe deployer -> skip entirely
            if dep.deployer_address in self.safe_deployers:
                logger.debug(
                    "Skipping safe deployer %s contract %s",
                    dep.deployer_address[:18], dep.contract_address,
                )
                return

            # Check 2: priority deployer -> auto-SUSPECTED
            detection_method = "bytecode_pattern"
            tier = "unknown"
            reason = (
                f"New contract deployment detected at block {dep.block_number}, "
                f"tx {dep.tx_hash[:18]}... -- awaiting bytecode classification"
            )
            bytecode_signals = {}

            if db.is_priority_deployer(self.conn, dep.deployer_address):
                deployer_rec = db.get_deployer(self.conn, dep.deployer_address)
                notes = deployer_rec["deployment_pattern_notes"] if deployer_rec else ""
                detection_method = "deployer_history"
                tier = "suspected"
                reason = (
                    f"Auto-SUSPECTED: deployer {dep.deployer_address} is flagged "
                    f"as priority ({notes}). Contract deployed at block "
                    f"{dep.block_number}, tx {dep.tx_hash[:18]}..."
                )
                logger.info(
                    "PRIORITY DEPLOYER: %s -> auto-SUSPECTED %s",
                    dep.deployer_address[:18] + "...", dep.contract_address,
                )
            elif dep.cache_hit and hasattr(dep, "_cached_result"):
                # Check 3: bytecode cache hit — reuse prior classification
                cached = dep._cached_result
                tier = cached["confidence_tier"]
                reason = (
                    f"[CACHE {dep.cache_hit}] {cached['confidence_reason']} "
                    f"(cached from {cached['source_contract'][:18]}...)"
                )
                bytecode_signals = cached["bytecode_signals"]
                logger.debug(
                    "Cache hit (%s): %s -> tier=%s from %s",
                    dep.cache_hit, dep.contract_address, tier,
                    cached["source_contract"][:18],
                )
            elif self.classifier:
                # Check 4: bytecode classifier (full analysis)
                try:
                    result = self.classifier(dep)
                    tier = result.get("confidence_tier", "unknown")
                    reason = result.get("confidence_reason", reason)
                    bytecode_signals = result.get("bytecode_signals", {})
                    # Store result in cache under both hashes
                    for h in (dep.init_code_hash, dep.deployed_code_hash):
                        if h:
                            db.cache_store(
                                self.conn, h, tier, reason,
                                bytecode_signals, dep.contract_address,
                            )
                except Exception as e:
                    logger.error(
                        "Classifier failed for %s: %s", dep.contract_address, e
                    )

            db.insert_contract(
                self.conn,
                contract_address=dep.contract_address,
                deployer_address=dep.deployer_address,
                detection_method=detection_method,
                detection_timestamp=dep.timestamp_iso,
                detection_block=dep.block_number,
                confidence_tier=tier,
                confidence_reason=reason,
                chain=self.chain,
                has_asymmetric_transfer=bytecode_signals.get("has_asymmetric_transfer"),
                has_conditional_revert=bytecode_signals.get("has_conditional_revert"),
                has_unusual_fee_structure=bytecode_signals.get("has_unusual_fee_structure"),
                bytecode_pattern_notes=bytecode_signals.get("pattern_notes"),
            )
            logger.info(
                "Recorded: %s [%s/%s] deployer=%s block=%d",
                dep.contract_address, tier, detection_method,
                dep.deployer_address[:18] + "...", dep.block_number,
            )

            # Deployment velocity check
            self._check_velocity(dep.deployer_address)

        except Exception as e:
            logger.error(
                "DB insert failed for %s: %s", dep.contract_address, e
            )

    def _check_velocity(self, deployer_address: str) -> None:
        """
        Track per-deployer deployment count this session. If a deployer
        exceeds velocity_threshold, auto-flag as priority and upgrade
        all their UNKNOWN contracts to SUSPECTED.
        """
        count = self._session_deployer_counts.get(deployer_address, 0) + 1
        self._session_deployer_counts[deployer_address] = count

        if count <= self.velocity_threshold:
            return
        if deployer_address in self._velocity_flagged:
            return  # already flagged this session

        self._velocity_flagged.add(deployer_address)

        # Flag the deployer as priority
        existing = db.get_deployer(self.conn, deployer_address)
        old_notes = existing["deployment_pattern_notes"] if existing else None
        velocity_note = (
            f"PRIORITY: deployment velocity {count} contracts in current "
            f"session (threshold: {self.velocity_threshold}). "
            f"Auto-flagged by velocity detector."
        )
        if old_notes:
            velocity_note = f"{old_notes}; {velocity_note}"
        db.update_deployer_notes(self.conn, deployer_address, velocity_note)

        # Upgrade all UNKNOWN contracts from this deployer to SUSPECTED
        unknown_contracts = [
            c for c in db.get_contracts_by_deployer(self.conn, deployer_address)
            if c["confidence_tier"] == "unknown"
        ]
        for c in unknown_contracts:
            db.update_contract_confidence(
                self.conn,
                c["contract_address"],
                "suspected",
                f"Deployer velocity escalation: {deployer_address} deployed "
                f"{count}+ contracts this session (threshold: "
                f"{self.velocity_threshold}). Bulk upgrade from unknown.",
            )

        logger.warning(
            "VELOCITY FLAG: deployer %s hit %d deployments this session "
            "(threshold %d) -- flagged as priority, %d contracts upgraded "
            "to SUSPECTED",
            deployer_address[:18] + "...", count, self.velocity_threshold,
            len(unknown_contracts),
        )

    @property
    def blocks_processed(self) -> int:
        return self._blocks_processed

    @property
    def deployments_found(self) -> int:
        return self._deployments_found

    @property
    def velocity_flagged(self) -> set[str]:
        return self._velocity_flagged

    @property
    def cache_hits_init(self) -> int:
        return self._cache_hits_init

    @property
    def cache_hits_deployed(self) -> int:
        return self._cache_hits_deployed

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def rpc_calls_saved(self) -> int:
        return self._rpc_calls_saved

    @property
    def selector_events(self) -> int:
        return self._selector_monitor.events_logged if self._selector_monitor else 0

    @property
    def bots_tagged(self) -> int:
        return self._selector_monitor.bots_tagged if self._selector_monitor else 0

    @property
    def bot_candidates_found(self) -> int:
        return self._revert_detector.candidates_found if self._revert_detector else 0

    @property
    def bot_deployer_hits(self) -> int:
        return self._revert_detector.deployer_hits if self._revert_detector else 0


async def run_monitor(rpc_url: str, db_path: Optional[Path] = None,
                      classifier=None, chain: str = "arbitrum") -> None:
    """Entry point: initialize DB, start monitor, handle shutdown."""
    conn = db.init_db(db_path)

    monitor = DeploymentMonitor(rpc_url, conn, classifier, chain=chain)

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown():
        logger.info("Shutdown signal received")
        monitor.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await monitor.start()
    finally:
        conn.close()
        logger.info("Database connection closed")


def main():
    parser = argparse.ArgumentParser(
        description="Layer 3 — Contract Deployment Monitor"
    )
    parser.add_argument(
        "--rpc",
        default=os.environ.get("ARB_WSS_URL"),
        help="WebSocket RPC URL (or set ARB_WSS_URL env var)",
    )
    parser.add_argument(
        "--chain",
        default="arbitrum",
        help="Chain name: arbitrum, base, etc. Stored in DB chain field.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: surveillance/data/surveillance.db)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Disable bytecode classification (record all as unknown)",
    )
    args = parser.parse_args()

    if not args.rpc:
        print(
            "ERROR: No RPC URL provided.\n"
            "  --rpc wss://...\n"
            "  or set ARB_WSS_URL environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s [{args.chain}:%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting monitor for chain: %s", args.chain)

    db_path = Path(args.db) if args.db else None

    # Wire up bytecode classifier unless disabled
    classifier = None
    if not args.no_classify:
        from surveillance.bytecode_classifier import classify
        classifier = classify
        logger.info("Bytecode classifier enabled")

    try:
        asyncio.run(run_monitor(
            args.rpc, db_path, classifier=classifier, chain=args.chain,
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
