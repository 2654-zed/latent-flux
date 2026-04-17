"""
x402 write-failure diagnostic tracer.

Drop-in replacement for surveillance.db_writer.run() that wraps the same
single-writer loop with verbose logging for writes targeting x402 tables.
Every IntegrityError (currently swallowed at db_writer.py:109) is logged
with exception class, full SQL, parameters, and the target row's current
state in the database — so the true cause of x402_events staying empty
can be diagnosed from real failures rather than inferred from a snapshot.

Scope:
- Swallowing behavior is PRESERVED. Do not raise exceptions in the writer
  loop; that would crash the single-writer process and halt ingest for
  every chain. Instead, log every swallowed IntegrityError with full
  context so the operator can see what failed and why.
- Only x402-related writes are diagnosed (table names x402_events,
  x402_facilitators, x402_permit2_exposure). Full-table logging would
  bury the signal under routine duplicate-insert IntegrityErrors.
- Log file path defaults to logs/x402_diagnostics.log but can be
  overridden via X402_TRACE_LOG env var.

Deployment path:
    Replace the db_writer target in run_surveillance.py (line ~1576):
        _writer_proc = Process(
            target=db_writer.run,
            args=(DB_PATH, _write_queue),
        )
    with:
        from surveillance.diagnostics import x402_write_tracer
        _writer_proc = Process(
            target=x402_write_tracer.run,
            args=(DB_PATH, _write_queue),
        )
    or gate on env: if os.environ.get("X402_TRACE"): use tracer else writer.

After 24h, rotate the log and grep for 'INTEGRITY' or 'WRITE_FAILED':
    grep -c INTEGRITY logs/x402_diagnostics.log
    grep INTEGRITY logs/x402_diagnostics.log | head -20

Restore by reverting run_surveillance.py. The tracer does not change
any table schemas or write semantics — only the observability layer.
"""

import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from queue import Empty

_X402_TABLE_RE = re.compile(
    r"\b(x402_events|x402_facilitators|x402_permit2_exposure)\b",
    re.IGNORECASE,
)

_UPDATE_TABLE_RE = re.compile(r"UPDATE\s+(\w+)\s+SET", re.IGNORECASE)
_INSERT_TABLE_RE = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)", re.IGNORECASE)

_EMPTY = Empty


def _is_x402_write(sql: str) -> bool:
    return bool(_X402_TABLE_RE.search(sql))


def _extract_target_table(sql: str) -> str | None:
    m = _UPDATE_TABLE_RE.search(sql) or _INSERT_TABLE_RE.search(sql)
    return m.group(1) if m else None


def _sample_row_state(conn: sqlite3.Connection, sql: str,
                      params: tuple) -> str:
    """
    Best-effort probe of the target row state at the time of failure.
    Return a short description (or 'unavailable: <reason>') — never raise.
    """
    table = _extract_target_table(sql)
    if not table:
        return "unavailable: could not parse target table from sql"
    try:
        # For x402_facilitators we can look up by (address, chain) if params
        # include what looks like an address and a chain.
        if table == "x402_facilitators" and params:
            addr_candidates = [p for p in params
                               if isinstance(p, str) and p.startswith("0x")
                               and len(p) == 42]
            chain_candidates = [p for p in params
                                if isinstance(p, str)
                                and p in ("base", "arbitrum", "optimism", "ethereum")]
            if addr_candidates and chain_candidates:
                row = conn.execute(
                    "SELECT address, chain, classification, tx_count, "
                    "first_seen, last_seen FROM x402_facilitators "
                    "WHERE address = ? AND chain = ?",
                    (addr_candidates[0], chain_candidates[0]),
                ).fetchone()
                if row is None:
                    return f"no row for ({addr_candidates[0]}, {chain_candidates[0]})"
                return f"existing row: {dict(row)}"
            return "unavailable: could not identify (address, chain) in params"
        # For x402_events — look up by tx_hash if we can find one.
        if table == "x402_events" and params:
            tx_candidates = [p for p in params
                             if isinstance(p, str) and p.startswith("0x")
                             and len(p) == 66]
            if tx_candidates:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM x402_events WHERE tx_hash = ?",
                    (tx_candidates[0],),
                ).fetchone()[0]
                return f"existing x402_events rows for tx {tx_candidates[0]}: {cnt}"
            return "unavailable: no tx_hash-shaped param identified"
        return f"unavailable: no probe defined for table {table}"
    except Exception as probe_err:
        return f"probe failed: {type(probe_err).__name__}: {probe_err}"


def _truncate(s: str, n: int = 400) -> str:
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n} chars]"


def _format_params(params, max_items: int = 32) -> str:
    if not params:
        return "()"
    items = list(params)[:max_items]
    rendered = ", ".join(
        _truncate(repr(p), 120) for p in items
    )
    if len(params) > max_items:
        rendered += f", ...[+{len(params) - max_items} more]"
    return "(" + rendered + ")"


def _safe_commit(conn: sqlite3.Connection, logger: logging.Logger) -> None:
    try:
        conn.commit()
    except Exception as e:
        logger.error("COMMIT_FAILED: %s: %s", type(e).__name__, e)


def _force_checkpoint(conn: sqlite3.Connection, logger: logging.Logger) -> None:
    try:
        result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result and result[1] and result[1] > 0:
            logger.info("WAL_TRUNCATE: %d pages checkpointed", result[1])
    except Exception as e:
        logger.warning("WAL_TRUNCATE_FAILED: %s", e)


def _maybe_checkpoint(conn: sqlite3.Connection, logger: logging.Logger,
                      last: float, interval: float) -> None:
    if (time.monotonic() - last) >= interval:
        _force_checkpoint(conn, logger)


def run(db_path: str, write_queue, *,
        batch_size: int = 500,
        flush_interval: float = 2.0,
        checkpoint_interval: float = 120.0) -> None:
    """
    Drop-in replacement for surveillance.db_writer.run with x402-targeted
    verbose logging. Same arguments, same semantics — only logging differs.
    """
    log_path = Path(os.environ.get(
        "X402_TRACE_LOG",
        str(Path(db_path).parent.parent.parent / "logs" / "x402_diagnostics.log"),
    ))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("x402_write_tracer")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if run() is called more than once in-process
    if not logger.handlers:
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
        # Also emit to stderr so Railway logs capture it
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(
            "%(asctime)s [x402_tracer] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(sh)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")

    pending = 0
    last_flush = time.monotonic()
    last_checkpoint = time.monotonic()
    total_ops = 0
    x402_ops = 0
    x402_integrity = 0
    x402_other_err = 0

    logger.info("x402 write-tracer started (db=%s, log=%s)", db_path, log_path)

    while True:
        try:
            msg = write_queue.get(timeout=flush_interval)
        except _EMPTY:
            if pending > 0:
                _safe_commit(conn, logger)
                pending = 0
                last_flush = time.monotonic()
            _maybe_checkpoint(conn, logger, last_checkpoint, checkpoint_interval)
            last_checkpoint = time.monotonic()
            continue

        op = msg[0]

        if op == "shutdown":
            if pending > 0:
                _safe_commit(conn, logger)
            _force_checkpoint(conn, logger)
            conn.close()
            logger.info(
                "x402 write-tracer shut down (ops=%d, x402_ops=%d, "
                "x402_integrity=%d, x402_other_err=%d)",
                total_ops, x402_ops, x402_integrity, x402_other_err,
            )
            return

        if op == "flush":
            if pending > 0:
                _safe_commit(conn, logger)
                pending = 0
                last_flush = time.monotonic()
            continue

        sql = msg[1]
        params = msg[2]
        is_x402 = _is_x402_write(sql)

        try:
            if op == "execute":
                conn.execute(sql, params or ())
            elif op == "executemany":
                conn.executemany(sql, params or [])
            elif op == "executescript":
                conn.executescript(sql)
            pending += 1
            total_ops += 1
            if is_x402:
                x402_ops += 1
                logger.info(
                    "X402_OK %s | sql=%s",
                    op, _truncate(sql, 200),
                )
        except sqlite3.IntegrityError as e:
            # Don't raise — preserve production semantics — but log with
            # full context for any x402 write. For non-x402 writes,
            # integrity errors are routine OR IGNORE duplicates: stay quiet.
            if is_x402:
                x402_integrity += 1
                row_state = _sample_row_state(conn, sql, params or ())
                logger.error(
                    "X402_INTEGRITY %s class=%s | msg=%s | "
                    "sql=%s | params=%s | row_state=%s",
                    op,
                    type(e).__name__,
                    str(e),
                    _truncate(sql, 400),
                    _format_params(params),
                    row_state,
                )
        except Exception as e:
            if is_x402:
                x402_other_err += 1
                row_state = _sample_row_state(conn, sql, params or ())
                logger.error(
                    "X402_WRITE_FAILED %s class=%s | msg=%s | "
                    "sql=%s | params=%s | row_state=%s",
                    op,
                    type(e).__name__,
                    str(e),
                    _truncate(sql, 400),
                    _format_params(params),
                    row_state,
                )
            else:
                logger.error(
                    "WRITE_FAILED %s class=%s | msg=%s | sql=%s",
                    op, type(e).__name__, str(e), _truncate(sql, 200),
                )

        now = time.monotonic()
        if pending >= batch_size or (now - last_flush) >= flush_interval:
            _safe_commit(conn, logger)
            pending = 0
            last_flush = now

        if (now - last_checkpoint) >= checkpoint_interval:
            if pending > 0:
                _safe_commit(conn, logger)
                pending = 0
                last_flush = now
            _force_checkpoint(conn, logger)
            last_checkpoint = now

        # Periodic status line — every 5000 ops
        if total_ops and total_ops % 5000 == 0:
            logger.info(
                "STATUS total_ops=%d x402_ops=%d x402_integrity=%d "
                "x402_other_err=%d pending=%d",
                total_ops, x402_ops, x402_integrity, x402_other_err, pending,
            )


if __name__ == "__main__":
    # When run directly, exit with a usage note — the tracer must be
    # spawned as a Process by run_surveillance.py (it consumes from an
    # mp.Queue that only that entrypoint constructs).
    import sys
    print(__doc__, file=sys.stderr)
    sys.exit(1)
