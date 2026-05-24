"""Layer 3 — per-method Alchemy CU instrumentation.

Wraps a web3.py provider's `make_request` to count every RPC call by
method + component + chain, and write batched per-30s aggregates to the
`rpc_call_log` SQLite table. Each row carries an estimated CU cost so
the daily totals are directly comparable to Alchemy's dashboard.

Designed to NEVER break the hot path:
  - Telemetry runs in-memory between flushes
  - Flush is best-effort (caught + logged on failure)
  - All exceptions in the wrapper layer propagate the original RPC error
    rather than the telemetry error

Why this exists: 2026-05-22/23 CU spike (1.1B -> 1.6B in ~36h, +500M)
was diagnosed against the DB but no per-method telemetry existed, so
the actual source (suspected: auto_funder_tracer.batch_hop_trace
non-idempotent retries) could not be confirmed. This module closes the
visibility gap.

CLI:
    python -m surveillance.rpc_telemetry --summary    # last 24h by method
    python -m surveillance.rpc_telemetry --top 20     # top 20 callers
    python -m surveillance.rpc_telemetry --since 2026-05-22T00:00 --until 2026-05-23T23:59
"""
from __future__ import annotations
import argparse
import logging
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("surveillance.rpc_telemetry")

# Alchemy CU cost table (as of 2026-05). Methods not listed default to 25.
# Source: https://docs.alchemy.com/reference/compute-units (cached locally)
ALCHEMY_CU_COSTS: dict[str, int] = {
    "eth_chainId": 0,
    "eth_blockNumber": 10,
    "eth_call": 26,
    "eth_estimateGas": 87,
    "eth_gasPrice": 19,
    "eth_getBalance": 19,
    "eth_getBlockByHash": 21,
    "eth_getBlockByNumber": 16,
    "eth_getCode": 19,
    "eth_getLogs": 75,
    "eth_getStorageAt": 17,
    "eth_getTransactionByHash": 17,
    "eth_getTransactionCount": 26,
    "eth_getTransactionReceipt": 15,
    "eth_sendRawTransaction": 250,
    "eth_subscribe": 20,
    "eth_unsubscribe": 10,
    "eth_syncing": 0,
    # Alchemy enhanced APIs
    "alchemy_getAssetTransfers": 150,
    "alchemy_getTokenBalances": 26,
    "alchemy_getTokenMetadata": 100,
    "alchemy_getTransactionReceipts": 250,
    "alchemy_getTokenAllowance": 19,
    "alchemy_pendingTransactions": 20,
    # Default for unknown methods
    "_default": 25,
}


def cu_estimate(method: str) -> int:
    return ALCHEMY_CU_COSTS.get(method, ALCHEMY_CU_COSTS["_default"])


_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS rpc_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    component TEXT NOT NULL,
    chain TEXT,
    method TEXT NOT NULL,
    n_calls INTEGER NOT NULL,
    cu_estimate INTEGER NOT NULL,
    n_errors INTEGER NOT NULL DEFAULT 0,
    avg_duration_ms INTEGER NOT NULL DEFAULT 0
)
"""

_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_rpc_log_ts ON rpc_call_log(ts)",
    "CREATE INDEX IF NOT EXISTS idx_rpc_log_method ON rpc_call_log(method)",
    "CREATE INDEX IF NOT EXISTS idx_rpc_log_component ON rpc_call_log(component)",
]


def ensure_rpc_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(_TABLE_DDL)
    for idx in _INDEX_DDL:
        conn.execute(idx)
    conn.commit()


class RpcTelemetry:
    """Buffer + flush per-method RPC counts to SQLite."""

    def __init__(self, db_path: str | Path, flush_interval: float = 30.0):
        self.db_path = str(db_path)
        self.flush_interval = flush_interval
        self._buf: dict[tuple[str, str, str], dict] = {}
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._enabled = True
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            ensure_rpc_log_table(conn)
            conn.close()
        except Exception as e:
            logger.warning("rpc_telemetry init failed (continuing disabled): %s", e)
            self._enabled = False

    def record(self, component: str, chain: str | None, method: str,
               ok: bool, duration_ms: int = 0) -> None:
        """Call-site hook. Never raises."""
        if not self._enabled:
            return
        try:
            key = (component, chain or "", method)
            with self._lock:
                d = self._buf.setdefault(key, {"ok": 0, "error": 0, "total_ms": 0})
                if ok:
                    d["ok"] += 1
                else:
                    d["error"] += 1
                d["total_ms"] += duration_ms
                if time.time() - self._last_flush > self.flush_interval:
                    self._flush_locked()
                    self._last_flush = time.time()
        except Exception as e:
            logger.warning("rpc_telemetry.record failed (continuing): %s", e)

    def flush(self) -> int:
        """Manual flush. Returns rows written."""
        if not self._enabled:
            return 0
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        if not self._buf:
            return 0
        rows = []
        ts = datetime.now(timezone.utc).isoformat()
        for (component, chain, method), counts in self._buf.items():
            n_calls = counts["ok"] + counts["error"]
            cu = cu_estimate(method) * n_calls
            avg_ms = counts["total_ms"] // n_calls if n_calls else 0
            rows.append((ts, component, chain or None, method, n_calls, cu,
                         counts["error"], avg_ms))
        self._buf.clear()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.executemany(
                "INSERT INTO rpc_call_log (ts, component, chain, method, n_calls, "
                "cu_estimate, n_errors, avg_duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            conn.close()
            return len(rows)
        except Exception as e:
            logger.warning("rpc_telemetry flush failed (%d rows lost): %s", len(rows), e)
            return 0


# Module-level singleton for components that don't want to manage their own
_GLOBAL: RpcTelemetry | None = None


def get_telemetry(db_path: str | Path | None = None) -> RpcTelemetry:
    """Return the process-wide telemetry instance."""
    global _GLOBAL
    if _GLOBAL is None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent / "data" / "surveillance.db"
        _GLOBAL = RpcTelemetry(db_path)
    return _GLOBAL


def wrap_async_provider(provider: Any, component: str, chain: str | None,
                        telemetry: RpcTelemetry | None = None) -> None:
    """Monkey-patch provider.make_request to record each call.

    Idempotent — if the provider is already wrapped (`_rpc_telemetry_wrapped`
    attribute set), this is a no-op.
    """
    if getattr(provider, "_rpc_telemetry_wrapped", False):
        return
    if telemetry is None:
        telemetry = get_telemetry()

    original_make_request = provider.make_request

    async def wrapped_make_request(method, params):
        t0 = time.monotonic()
        ok = False
        try:
            result = await original_make_request(method, params)
            # Detect JSON-RPC error in successful response
            ok = not (isinstance(result, dict) and "error" in result and result["error"])
            return result
        except Exception:
            ok = False
            raise
        finally:
            ms = int((time.monotonic() - t0) * 1000)
            telemetry.record(component, chain, method, ok=ok, duration_ms=ms)

    provider.make_request = wrapped_make_request
    provider._rpc_telemetry_wrapped = True
    logger.info("rpc_telemetry: wrapped provider for component=%s chain=%s",
                component, chain)


# --- query helpers (for the CLI + the /api/rpc/usage endpoint) ---

def summarize(conn: sqlite3.Connection, since: str | None = None,
              until: str | None = None, group_by: str = "method") -> list[dict]:
    """Aggregate rpc_call_log over a time window. group_by: 'method' | 'component' | 'chain'."""
    valid = {"method", "component", "chain"}
    if group_by not in valid:
        raise ValueError(f"group_by must be one of {valid}")
    where = []
    args = []
    if since:
        where.append("ts >= ?"); args.append(since)
    if until:
        where.append("ts <= ?"); args.append(until)
    sql = (
        f"SELECT {group_by}, SUM(n_calls) calls, SUM(cu_estimate) cu, "
        f"SUM(n_errors) errors, MAX(avg_duration_ms) max_avg_ms "
        f"FROM rpc_call_log "
        + (" WHERE " + " AND ".join(where) if where else "")
        + f" GROUP BY {group_by} ORDER BY cu DESC"
    )
    out = []
    for r in conn.execute(sql, args):
        out.append({
            group_by: r[0],
            "calls": r[1] or 0,
            "cu": r[2] or 0,
            "errors": r[3] or 0,
            "max_avg_ms": r[4] or 0,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(Path(__file__).resolve().parent / "data" / "surveillance.db"))
    ap.add_argument("--summary", action="store_true", help="show last-24h summary")
    ap.add_argument("--top", type=int, default=10, help="top N rows")
    ap.add_argument("--since", default=None, help="ISO timestamp lower bound")
    ap.add_argument("--until", default=None, help="ISO timestamp upper bound")
    ap.add_argument("--by", choices=["method", "component", "chain"], default="method")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}"); return 1

    conn = sqlite3.connect(args.db)
    ensure_rpc_log_table(conn)

    since = args.since
    if args.summary and not since:
        since = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=1)).isoformat()

    rows = summarize(conn, since=since, until=args.until, group_by=args.by)
    if not rows:
        print("No rpc_call_log rows in the requested window.")
        return 0

    total_calls = sum(r["calls"] for r in rows)
    total_cu = sum(r["cu"] for r in rows)
    print(f"=== RPC usage by {args.by} (since={since or 'all-time'} until={args.until or 'now'}) ===")
    print(f"  Total: {total_calls:,} calls, {total_cu:,} CUs")
    print()
    print(f"  {'group':40s}  {'calls':>12s}  {'CUs':>14s}  {'err':>5s}  {'max_avg_ms':>10s}")
    print("  " + "-" * 90)
    for r in rows[:args.top]:
        g = (r[args.by] or "")[:40]
        print(f"  {g:40s}  {r['calls']:>12,}  {r['cu']:>14,}  {r['errors']:>5,}  {r['max_avg_ms']:>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
