"""
Layer 3 — Queue-backed SQLite connection proxy.

Provides a sqlite3.Connection-compatible interface that routes all writes
to a multiprocessing.Queue (drained by db_writer.py) and all reads to a
local read-only connection. Monitors use this instead of a direct connection.

    queue_conn = QueueConnection(db_path, write_queue)
    queue_conn.execute("SELECT ...")   # → local RO connection
    queue_conn.execute("INSERT ...")   # → serialized to queue → writer process
    queue_conn.commit()                # → flush signal to writer
"""

import sqlite3
from multiprocessing import Queue


class _DummyCursor:
    """Returned for write operations routed to queue."""

    @property
    def lastrowid(self):
        return None

    @property
    def rowcount(self):
        return -1

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def fetchmany(self, n=1):
        return []

    def close(self):
        pass

    def __iter__(self):
        return iter([])


class QueueConnection:
    """
    Drop-in replacement for sqlite3.Connection.

    - SELECT / PRAGMA reads → local read-only connection
    - INSERT / UPDATE / DELETE → serialized to writer queue
    - commit() → flush signal so writer commits promptly
    """

    _WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
                       "ALTER", "REPLACE")
    _WRITE_PRAGMAS = ("WAL_CHECKPOINT", "JOURNAL_MODE", "WAL_AUTOCHECKPOINT",
                      "SYNCHRONOUS")

    def __init__(self, db_path: str, write_queue: Queue):
        self._db_path = str(db_path)
        self._queue = write_queue
        # Read-only connection for SELECT queries
        uri = f"file:{self._db_path}?mode=ro"
        self._ro = sqlite3.connect(uri, uri=True, timeout=30)
        self._ro.row_factory = sqlite3.Row

    # -- sqlite3.Connection interface ------------------------------------------

    def execute(self, sql: str, params=()):
        if self._is_write(sql):
            self._queue.put(("execute", sql, tuple(params) if params else ()))
            return _DummyCursor()
        return self._ro.execute(sql, params)

    def executemany(self, sql: str, params_list):
        self._queue.put(("executemany", sql, [tuple(p) for p in params_list]))
        return _DummyCursor()

    def executescript(self, script: str):
        self._queue.put(("executescript", script, None))

    def commit(self):
        self._queue.put(("flush", None, None))

    def close(self):
        try:
            self._ro.close()
        except Exception:
            pass

    @property
    def row_factory(self):
        return self._ro.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._ro.row_factory = value

    # -- Internal --------------------------------------------------------------

    def _is_write(self, sql: str) -> bool:
        stripped = sql.lstrip()
        upper10 = stripped[:10].upper()
        for prefix in self._WRITE_PREFIXES:
            if upper10.startswith(prefix):
                return True
        if upper10.startswith("PRAGMA"):
            sql_upper = stripped.upper()
            for wp in self._WRITE_PRAGMAS:
                if wp in sql_upper:
                    return True
            return False
        return False
