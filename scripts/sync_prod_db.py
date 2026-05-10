"""Local-side production-DB sync wrapper.

Invokes `scripts/sync_prod_db_remote.py` on the Railway container via
`railway ssh`, captures the framed base64 payload from stdout, streams
it through base64-decode → gunzip → temp file, validates the result is
a healthy SQLite database, and atomic-renames into place.

Why this exists: Railway production exposes no `/dump`-style HTTP
endpoint. Manual sync via `railway ssh` is the documented path. This
script automates that with safety (validation + atomic rename + backup
of prior DB) and streaming (no multi-GB in-memory buffering).

Usage:
    python scripts/sync_prod_db.py
    python scripts/sync_prod_db.py --out /custom/path/surveillance.db
    python scripts/sync_prod_db.py --dry-run   # don't replace local DB

Exit codes:
    0 success
    1 railway ssh failed
    2 marker frame not found in stdout
    3 SQLite integrity check failed on downloaded copy
"""
from __future__ import annotations

import argparse
import base64
import gzip
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_prod_db_remote.py"
DEFAULT_LOCAL_DB = REPO_ROOT / "surveillance" / "data" / "surveillance.db"
PROJECT = "blockchain"
SERVICE = "stellar-embrace"
MARKER_START = b"===L3SYNC_PAYLOAD_START==="
MARKER_END = b"===L3SYNC_PAYLOAD_END==="


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def stream_extract_payload(stream, dest_path: Path) -> tuple[int, int]:
    """Read line-by-line from `stream` (subprocess stdout), find the
    base64 payload between MARKER_START/MARKER_END, base64-decode it
    line-by-line, and write the decoded bytes to a temp file. Returns
    (base64_bytes_in, decoded_bytes_out).

    Streaming-decode keeps memory bounded regardless of payload size.
    """
    state = "before"
    b64_in = 0
    decoded_out = 0
    with open(dest_path, "wb") as f_out:
        for raw_line in stream:
            # Strip CR (Railway SSH does CRLF translation) and trailing newlines
            line = raw_line.rstrip(b"\r\n")
            if state == "before":
                if MARKER_START in line:
                    state = "in_payload"
                continue
            if state == "in_payload":
                if MARKER_END in line:
                    state = "after"
                    break
                if not line:
                    continue
                # Skip any non-base64 contamination; if line has stray content, abort
                if not re.fullmatch(rb"[A-Za-z0-9+/=]+", line):
                    sys.stderr.write(
                        f"WARN: non-base64 line inside payload (skipped): {line[:80]!r}\n"
                    )
                    continue
                b64_in += len(line)
                # Each line is 48KB binary worth of base64 = 65,536 chars (the
                # remote-side line size is exactly multiples of 4, so per-line
                # decode is safe).
                decoded = base64.b64decode(line)
                f_out.write(decoded)
                decoded_out += len(decoded)
    if state != "after":
        raise RuntimeError(
            f"Payload markers not properly framed (state={state}, "
            f"b64_in={b64_in}, decoded_out={decoded_out}). "
            f"Expected START + payload + END, but stream ended in state={state}."
        )
    return b64_in, decoded_out


def decompress_gz(gz_path: Path, db_path: Path) -> int:
    """Stream gunzip the gz file → db file. Returns final db size."""
    with gzip.open(gz_path, "rb") as f_in, open(db_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=4 * 1024 * 1024)
    return db_path.stat().st_size


def validate_sqlite(path: Path) -> dict:
    """Open the file as SQLite read-only, run integrity_check, return summary."""
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        deployers = conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
        contracts = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        try:
            tx = conn.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0]
        except sqlite3.OperationalError:
            tx = None
        try:
            heart = conn.execute(
                "SELECT monitor_name, last_seen FROM heartbeat ORDER BY last_seen DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            heart = None
    finally:
        conn.close()
    return {
        "integrity": integrity,
        "deployers": deployers,
        "contracts": contracts,
        "tx_events": tx,
        "heartbeat": heart,
    }


def run(args: argparse.Namespace) -> int:
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read & base64-encode the remote script
    remote_script = REMOTE_SCRIPT_PATH.read_text(encoding="utf-8")
    script_b64 = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    bootstrap = (
        f"python3 -c \"import base64; "
        f"exec(base64.b64decode('{script_b64}'))\""
    )

    # On Windows, subprocess.Popen with a list of args does not resolve PATH
    # the same way the shell does — it uses CreateProcess directly. Resolve
    # the railway binary via shutil.which to get a full path. If the resolved
    # binary is a .cmd/.bat (npm-installed railway is railway.CMD), Windows
    # CreateProcess cannot execute it directly; wrap with cmd.exe /c.
    railway_exe = (
        shutil.which("railway.exe")
        or shutil.which("railway.cmd")
        or shutil.which("railway")
    )
    if not railway_exe:
        sys.stderr.write(
            "[sync] FATAL: 'railway' CLI not found in PATH. "
            "Install via https://docs.railway.app/develop/cli or run "
            "`npm i -g @railway/cli`.\n"
        )
        return 1

    # Precondition: the linked-project context must already be set, because
    # `railway ssh -p/-s` flags expect project/service IDs (UUIDs), not the
    # human-readable names. We rely on `railway link --project <name>` having
    # been run interactively.
    status_cmd = (
        ["cmd.exe", "/c", railway_exe, "status"]
        if os.name == "nt" and railway_exe.lower().endswith((".cmd", ".bat"))
        else [railway_exe, "status"]
    )
    status = subprocess.run(status_cmd, capture_output=True, text=True)
    if status.returncode != 0 or PROJECT not in status.stdout or SERVICE not in status.stdout:
        sys.stderr.write(
            f"[sync] FATAL: railway CLI is not linked to {PROJECT}@{SERVICE}.\n"
            f"[sync] Run interactively: railway link --project {PROJECT} && "
            f"railway service {SERVICE}\n"
            f"[sync] Current `railway status`:\n{status.stdout or status.stderr}\n"
        )
        return 1

    if os.name == "nt" and railway_exe.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd.exe", "/c", railway_exe, "ssh", bootstrap]
    else:
        cmd = [railway_exe, "ssh", bootstrap]
    sys.stderr.write(f"[sync] running railway ssh against {SERVICE}@{PROJECT}\n")
    sys.stderr.write(f"[sync] railway binary: {railway_exe}\n")
    sys.stderr.write(f"[sync] remote script size: {len(remote_script)} bytes "
                     f"({len(script_b64)} chars base64)\n")

    # Use temp paths so we never partially overwrite the live DB
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        gz_path = tmp_dir / "snapshot.db.gz"
        new_db_path = tmp_dir / "snapshot.db"

        # Stream-extract base64 payload from railway ssh stdout into gz_path
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        try:
            b64_in, decoded_out = stream_extract_payload(proc.stdout, gz_path)
        finally:
            proc.wait()
            stderr_blob = proc.stderr.read().decode("utf-8", errors="replace")

        if proc.returncode != 0:
            sys.stderr.write(f"[sync] railway ssh exited {proc.returncode}\n")
            sys.stderr.write(f"[sync] stderr (last 2KB):\n{stderr_blob[-2000:]}\n")
            return 1

        sys.stderr.write(f"[sync] base64 received: {fmt_bytes(b64_in)}\n")
        sys.stderr.write(f"[sync] gz on disk: {fmt_bytes(gz_path.stat().st_size)}\n")
        sys.stderr.write(f"[sync] (remote stderr tail: {stderr_blob.strip()[-200:]!r})\n")

        # Decompress gz → SQLite db
        sys.stderr.write("[sync] decompressing...\n")
        db_size = decompress_gz(gz_path, new_db_path)
        sys.stderr.write(f"[sync] decompressed db: {fmt_bytes(db_size)}\n")

        # Validate
        sys.stderr.write("[sync] validating...\n")
        try:
            summary = validate_sqlite(new_db_path)
        except sqlite3.DatabaseError as e:
            sys.stderr.write(f"[sync] FAILED: not a valid SQLite database: {e}\n")
            return 3
        sys.stderr.write(f"[sync]   integrity_check: {summary['integrity']}\n")
        sys.stderr.write(f"[sync]   deployers: {summary['deployers']:,}\n")
        sys.stderr.write(f"[sync]   contracts: {summary['contracts']:,}\n")
        if summary["tx_events"] is not None:
            sys.stderr.write(f"[sync]   tx_events: {summary['tx_events']:,}\n")
        if summary["heartbeat"]:
            sys.stderr.write(f"[sync]   heartbeat: {summary['heartbeat']}\n")
        if summary["integrity"] != "ok":
            sys.stderr.write("[sync] INTEGRITY FAILED — refusing to replace local DB\n")
            return 3

        if args.dry_run:
            sys.stderr.write(f"[sync] dry-run: snapshot at {new_db_path} (will be deleted on exit)\n")
            return 0

        # Atomic-replace
        bak_path = out_path.with_suffix(out_path.suffix + ".bak")
        if out_path.exists():
            if bak_path.exists():
                bak_path.unlink()
            sys.stderr.write(f"[sync] backing up prior DB → {bak_path.name}\n")
            out_path.rename(bak_path)
        # rename() across a tempdir mountpoint may fail on Windows; use shutil.move which falls back to copy
        shutil.move(str(new_db_path), str(out_path))
        sys.stderr.write(f"[sync] done. {out_path}\n")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default=str(DEFAULT_LOCAL_DB),
                    help="Local DB path to replace (default: surveillance/data/surveillance.db)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate the snapshot but do not replace the local DB")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
