"""Remote-side production-DB sync script. Runs ON the Railway container,
performs a SQLite online backup, gzips it, base64-encodes it, and streams
the encoded bytes to stdout framed by start/end markers.

Invoked from `scripts/sync_prod_db.py` (the local wrapper) via:

    railway ssh python3 -c "import base64; exec(base64.b64decode('<this script base64-encoded>'))"

The base64 dispatch avoids shell-quoting nightmares for the script body.

Markers + base64 are required because railway ssh's transport:
  - merges stderr into stdout
  - does CRLF translation on binary bytes (verified 2026-05-09:
    1024 random bytes arrived as 1028 bytes — 4 stray \n→\r\n substitutions)
  - prepends/appends some banner content

base64 is the safest text-only encoding that survives all of that. Markers let
the local wrapper trim banner content cleanly.

Disk usage on the container during run:
  /tmp/<random>.db    — full backup, same size as production DB (~10.6 GB current)
  /tmp/<random>.db.gz — gzip-compressed (~50% of raw, depends on bytecode density)
  Both are unlinked at script exit.
"""
import base64
import gzip
import os
import sqlite3
import sys
import tempfile

PROD_DB = "/app/surveillance/data/surveillance.db"
MARKER_START = "===L3SYNC_PAYLOAD_START==="
MARKER_END = "===L3SYNC_PAYLOAD_END==="
B64_LINE_INPUT_BYTES = 48 * 1024  # 48KB binary -> 64KB base64 (newline at boundary)


def main() -> int:
    if not os.path.exists(PROD_DB):
        sys.stderr.write(f"PROD_DB not found at {PROD_DB}\n")
        return 2

    # 1. Online backup to a temp file (handles WAL + locks safely)
    src = sqlite3.connect(PROD_DB)
    backup_fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir="/tmp")
    backup_path = backup_fd.name
    backup_fd.close()
    dst = sqlite3.connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()
    backup_size = os.path.getsize(backup_path)
    sys.stderr.write(f"backup: {backup_size} bytes at {backup_path}\n")

    # 2. gzip the backup file
    gz_path = backup_path + ".gz"
    with open(backup_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
    os.unlink(backup_path)
    gz_size = os.path.getsize(gz_path)
    sys.stderr.write(f"gzip: {gz_size} bytes at {gz_path} ({gz_size * 100 // backup_size}% of raw)\n")

    # 3. base64-stream gz to stdout, framed
    sys.stdout.buffer.write((MARKER_START + "\n").encode())
    sys.stdout.buffer.flush()
    with open(gz_path, "rb") as f:
        while True:
            chunk = f.read(B64_LINE_INPUT_BYTES)
            if not chunk:
                break
            encoded = base64.b64encode(chunk)
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.write((MARKER_END + "\n").encode())
    sys.stdout.buffer.flush()

    os.unlink(gz_path)
    sys.stderr.write("done\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
