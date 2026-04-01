"""
Layer 3 — Single-writer process for SQLite.

All database writes flow through this process via a multiprocessing.Queue.
This is the ONLY process that opens a read-write connection to the database.

Benefits:
- WAL checkpoint trivially succeeds (single writer, no contention)
- Batch commits reduce IO (flush every N ops or M seconds)
- No WAL bloat — TRUNCATE works because no other writers hold locks

Usage (from run_surveillance.py):
    from multiprocessing import Process, Queue
    q = Queue()
    writer = Process(target=db_writer.run, args=(DB_PATH, q))
    writer.start()
"""

import logging
import sqlite3
import time
from queue import Empty

logger = logging.getLogger("surveillance.db_writer")

# Sentinel for queue.get timeout (re-use stdlib Empty)
_EMPTY = Empty


def run(db_path: str, write_queue, *,
        batch_size: int = 500,
        flush_interval: float = 2.0,
        checkpoint_interval: float = 120.0) -> None:
    """
    Main loop for the writer process.

    Args:
        db_path: Path to the SQLite database file.
        write_queue: multiprocessing.Queue of (op, sql, params) tuples.
        batch_size: Commit after this many write operations.
        flush_interval: Commit after this many seconds even if batch_size not reached.
        checkpoint_interval: Run WAL TRUNCATE checkpoint every N seconds.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [writer:%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")    # manual checkpoints only
    conn.execute("PRAGMA synchronous=NORMAL")       # safe with WAL, faster
    conn.execute("PRAGMA busy_timeout=5000")

    pending = 0
    last_flush = time.monotonic()
    last_checkpoint = time.monotonic()
    total_ops = 0

    logger.info("DB writer started — exclusive writer for %s", db_path)

    while True:
        # ---- drain one message from the queue ----
        try:
            msg = write_queue.get(timeout=flush_interval)
        except _EMPTY:
            # Timeout — flush + maybe checkpoint
            if pending > 0:
                _safe_commit(conn)
                pending = 0
                last_flush = time.monotonic()
            _maybe_checkpoint(conn, last_checkpoint, checkpoint_interval)
            last_checkpoint = time.monotonic()
            continue

        op = msg[0]

        # ---- shutdown signal ----
        if op == "shutdown":
            if pending > 0:
                _safe_commit(conn)
            _force_checkpoint(conn)
            conn.close()
            logger.info("DB writer shut down cleanly (%d total ops)", total_ops)
            return

        # ---- flush signal (from QueueConnection.commit()) ----
        if op == "flush":
            if pending > 0:
                _safe_commit(conn)
                pending = 0
                last_flush = time.monotonic()
            continue

        # ---- execute write ----
        sql = msg[1]
        params = msg[2]

        try:
            if op == "execute":
                conn.execute(sql, params or ())
            elif op == "executemany":
                conn.executemany(sql, params or [])
            elif op == "executescript":
                conn.executescript(sql)
            pending += 1
            total_ops += 1
        except sqlite3.IntegrityError:
            pass  # OR IGNORE semantics — expected for duplicate inserts
        except Exception as e:
            logger.error("Write failed [%s]: %s | sql=%.80s", op, e, sql)

        # ---- batch commit ----
        now = time.monotonic()
        if pending >= batch_size or (now - last_flush) >= flush_interval:
            _safe_commit(conn)
            pending = 0
            last_flush = now

        # ---- periodic checkpoint ----
        if (now - last_checkpoint) >= checkpoint_interval:
            if pending > 0:
                _safe_commit(conn)
                pending = 0
                last_flush = now
            _force_checkpoint(conn)
            last_checkpoint = now


# -- helpers ------------------------------------------------------------------

def _safe_commit(conn: sqlite3.Connection) -> None:
    try:
        conn.commit()
    except Exception as e:
        logger.error("Commit failed: %s", e)


def _force_checkpoint(conn: sqlite3.Connection) -> None:
    try:
        result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result and result[1] and result[1] > 0:
            logger.info("WAL TRUNCATE: %d pages checkpointed", result[1])
    except Exception as e:
        logger.warning("WAL TRUNCATE failed: %s", e)


def _maybe_checkpoint(conn: sqlite3.Connection,
                      last: float, interval: float) -> None:
    if (time.monotonic() - last) >= interval:
        _force_checkpoint(conn)
