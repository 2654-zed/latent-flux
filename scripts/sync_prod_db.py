"""Local-side production-DB sync wrapper.

Architecture (v2, 2026-05-15):

Two-phase protocol via `scripts/sync_prod_db_remote.py`:

    PHASE 1 (prepare, single SSH call, ~5-10 min wall):
        railway ssh "python3 -c 'exec(...)' prepare"
        -> remote does SQLite backup + gzip to /tmp/l3sync_snapshot.db.gz
        -> stdout: READY:<size_bytes>:<sha256_hex>

    PHASE 2 (chunk loop, N SSH calls, ~10s each):
        for offset in range(0, size, CHUNK_SIZE):
            railway ssh "python3 -c 'exec(...)' chunk <offset> <chunk_size>"
            -> remote: open prepared gz, seek, read length bytes, base64 stream
            -> local: capture stdout, mmap-search markers, decode -> append to local gz

    PHASE 3 (cleanup, single SSH call, ~3s):
        railway ssh "python3 -c 'exec(...)' cleanup"

Why two-phase: empirically (2026-05-15) a single SSH call streaming the full
~4.4 GB base64 payload of an 11.6 GB production DB fails with
`Error: WebSocket error: tungstenite error` from the railway CLI. Smaller
streams (verified: 50 MB) and long-idle sessions (verified: 7.4 min) both
succeed. The cliff is total-volume-per-SSH-call. Chunking into ~100 MB
slices keeps each invocation comfortably below the cliff.

Each chunk is its own SSH session -> its own WebSocket -> size limits reset.
SHA-256 from prepare is verified against the locally-assembled gz before
gunzip, catching any chunk corruption or off-by-one stitching.

Usage:
    python scripts/sync_prod_db.py
    python scripts/sync_prod_db.py --out /custom/path/surveillance.db
    python scripts/sync_prod_db.py --dry-run         # validate, don't replace
    python scripts/sync_prod_db.py --chunk-mb 50     # smaller chunks (retry safety)
    python scripts/sync_prod_db.py --resume          # reuse prepared gz on container

Exit codes:
    0 success
    1 railway CLI not linked / not found
    2 marker frame not found in a chunk's output
    3 SQLite integrity check failed on downloaded copy
    4 SHA-256 mismatch between remote prepared gz and local assembled gz
    5 prepare phase failed
    6 chunk phase failed (after retries)
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
import time
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_prod_db_remote.py"
DEFAULT_LOCAL_DB = REPO_ROOT / "surveillance" / "data" / "surveillance.db"
PROJECT = "blockchain"
SERVICE = "stellar-embrace"
MARKER_START = b"===L3SYNC_PAYLOAD_START==="
MARKER_END = b"===L3SYNC_PAYLOAD_END==="

DEFAULT_CHUNK_BYTES = 100 * 1024 * 1024  # 100 MB binary -> ~133 MB base64
CHUNK_RETRIES = 2


def fmt_bytes(n: int) -> str:
    f: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def resolve_railway_exe() -> str | None:
    return (
        shutil.which("railway.exe")
        or shutil.which("railway.cmd")
        or shutil.which("railway")
    )


def build_ssh_cmd(railway_exe: str, remote_mode_args: list[str], script_b64: str) -> list[str]:
    """Build the cmd-line for `railway ssh` invoking the remote script in a
    specific mode. `remote_mode_args` is appended after the bootstrap so
    the remote sys.argv[1:] receives those tokens.
    """
    extra = (" " + " ".join(remote_mode_args)) if remote_mode_args else ""
    bootstrap = (
        f"python3 -c \"import base64; "
        f"exec(base64.b64decode('{script_b64}'))\"{extra}"
    )
    if os.name == "nt" and railway_exe.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", railway_exe, "ssh", bootstrap]
    return [railway_exe, "ssh", bootstrap]


def run_ssh(cmd: list[str], timeout: int = 1800) -> tuple[int, bytes, str]:
    """Run an SSH cmd, capturing stdout to bytes and stderr to str.
    Returns (returncode, stdout_bytes, stderr_str).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_chunks: list[bytes] = []
    try:
        # Drain stdout incrementally so the pipe never deadlocks
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk:
                break
            stdout_chunks.append(chunk)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    stderr_blob = proc.stderr.read().decode("utf-8", errors="replace")
    return proc.returncode, b"".join(stdout_chunks), stderr_blob


def parse_ready_line(stdout_bytes: bytes) -> tuple[int, str]:
    """Find a `READY:<size>:<sha256>` line in stdout (transport may have
    other banner lines). Return (size, sha256_hex).
    """
    text = stdout_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        m = re.match(r"^READY:(\d+):([0-9a-f]{64})\s*$", line.strip())
        if m:
            return int(m.group(1)), m.group(2)
    raise RuntimeError(
        f"READY line not found in remote output. Last 1KB: {text[-1024:]!r}"
    )


def extract_chunk_payload(stdout_bytes: bytes) -> bytes:
    """Given stdout from a `chunk` invocation, find MARKER_START / MARKER_END,
    strip non-base64 chars in the payload region, and return decoded bytes.
    """
    start_idx = stdout_bytes.find(MARKER_START)
    if start_idx < 0:
        head = stdout_bytes[:500].decode("utf-8", errors="replace")
        tail = stdout_bytes[-500:].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"MARKER_START not found in chunk output "
            f"({len(stdout_bytes):,} bytes captured).\n"
            f"Head: {head!r}\nTail: {tail!r}"
        )
    payload_start = start_idx + len(MARKER_START)
    end_idx = stdout_bytes.find(MARKER_END, payload_start)
    if end_idx < 0:
        tail = stdout_bytes[-500:].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"MARKER_END not found (start={start_idx}, "
            f"size={len(stdout_bytes):,}). Tail: {tail!r}"
        )
    payload_b64 = stdout_bytes[payload_start:end_idx]
    # Strip everything outside the base64 alphabet (banner content, CRLF)
    clean = re.sub(rb"[^A-Za-z0-9+/=]", b"", payload_b64)
    return base64.b64decode(clean)


def decompress_gz(gz_path: Path, db_path: Path) -> int:
    with gzip.open(gz_path, "rb") as f_in, open(db_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=4 * 1024 * 1024)
    return db_path.stat().st_size


def validate_sqlite(path: Path) -> dict:
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
                "SELECT monitor_name, last_seen FROM heartbeat "
                "ORDER BY last_seen DESC LIMIT 1"
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

    railway_exe = resolve_railway_exe()
    if not railway_exe:
        sys.stderr.write(
            "[sync] FATAL: 'railway' CLI not found in PATH. "
            "Install via https://docs.railway.app/develop/cli or run "
            "`npm i -g @railway/cli`.\n"
        )
        return 1

    # Precondition: linked-project context must already be set
    if os.name == "nt" and railway_exe.lower().endswith((".cmd", ".bat")):
        status_cmd = ["cmd.exe", "/c", railway_exe, "status"]
    else:
        status_cmd = [railway_exe, "status"]
    status = subprocess.run(status_cmd, capture_output=True, text=True)
    if (
        status.returncode != 0
        or PROJECT not in status.stdout
        or SERVICE not in status.stdout
    ):
        sys.stderr.write(
            f"[sync] FATAL: railway CLI is not linked to {PROJECT}@{SERVICE}.\n"
            f"[sync] Run interactively: railway link --project {PROJECT} && "
            f"railway service {SERVICE}\n"
            f"[sync] Current `railway status`:\n{status.stdout or status.stderr}\n"
        )
        return 1

    # Read & base64-encode the remote script once (reused across phases)
    remote_script = REMOTE_SCRIPT_PATH.read_text(encoding="utf-8")
    script_b64 = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    sys.stderr.write(
        f"[sync] remote script: {len(remote_script)} bytes "
        f"({len(script_b64)} chars base64)\n"
    )

    chunk_bytes = args.chunk_mb * 1024 * 1024

    # ---- PHASE 1: prepare (or skip via --resume) ----
    if args.resume:
        sys.stderr.write("[sync] phase 1: --resume requested, fetching sha256 of prepared gz...\n")
        cmd = build_ssh_cmd(railway_exe, ["sha256"], script_b64)
        rc, stdout_bytes, stderr_blob = run_ssh(cmd, timeout=120)
        if rc != 0:
            sys.stderr.write(
                f"[sync] sha256 phase failed (rc={rc}). stderr:\n{stderr_blob[-2000:]}\n"
            )
            return 5
    else:
        # ---- PHASE 1a: start prepare in detached background on container ----
        # Decoupled from SSH session lifetime — Railway has a wall-clock cap
        # (~4 min observed) on single SSH sessions that single-call prepare
        # cannot satisfy for a 12 GB DB. Background fork + poll instead.
        sys.stderr.write("[sync] phase 1a: starting detached prepare on container...\n")
        cmd = build_ssh_cmd(railway_exe, ["start"], script_b64)
        try:
            rc, stdout_bytes, stderr_blob = run_ssh(cmd, timeout=60)
        except subprocess.TimeoutExpired:
            sys.stderr.write("[sync] phase 1a: start command timed out (>60s)\n")
            return 5
        if rc != 0:
            sys.stderr.write(
                f"[sync] phase 1a failed (rc={rc}). stderr:\n{stderr_blob[-1500:]}\n"
                f"[sync] stdout: {stdout_bytes[-500:]!r}\n"
            )
            return 5
        out = stdout_bytes.decode("utf-8", errors="replace")
        if "STARTED" not in out and "ALREADY_READY" not in out:
            sys.stderr.write(f"[sync] phase 1a: unexpected response: {out!r}\n")
            return 5
        sys.stderr.write("[sync] phase 1a: prepare started.\n")

        # ---- PHASE 1b: poll status until READY ----
        # Each poll is a short SSH call (well under Railway's session cap).
        # The actual prepare runs decoupled in the daemon process.
        sys.stderr.write(
            "[sync] phase 1b: polling for READY (typical: 6-8 min for 12 GB)\n"
        )
        poll_interval = 30
        poll_max_wait = 30 * 60  # 30 min total cap
        t0 = time.time()
        stdout_bytes = b""
        while True:
            elapsed = time.time() - t0
            if elapsed > poll_max_wait:
                sys.stderr.write(
                    f"[sync] phase 1b: timed out after {elapsed:.0f}s "
                    f"waiting for READY\n"
                )
                return 5
            cmd_stat = build_ssh_cmd(railway_exe, ["status"], script_b64)
            try:
                s_rc, s_out, s_err = run_ssh(cmd_stat, timeout=60)
            except subprocess.TimeoutExpired:
                sys.stderr.write(f"[sync] phase 1b: status poll timed out at t={elapsed:.0f}s; retrying\n")
                time.sleep(poll_interval)
                continue
            out_text = s_out.decode("utf-8", errors="replace")
            # Emit progress to caller stderr (status lines + tail of log)
            for line in out_text.splitlines():
                if line.startswith("STATUS:") or line.startswith("  LOG:"):
                    sys.stderr.write(f"[sync]   t={elapsed:>5.0f}s  {line}\n")
            if s_rc == 0:
                # READY:size:sha — pass through to the existing parser
                stdout_bytes = s_out
                break
            if s_rc == 2:
                sys.stderr.write(f"[sync] phase 1b: ERROR from container\n{out_text}\n")
                return 5
            # s_rc == 3 (still running) — sleep and poll again
            time.sleep(poll_interval)
        sys.stderr.write(f"[sync] phase 1b: READY received after {time.time()-t0:.0f}s\n")

    try:
        gz_size, expected_sha = parse_ready_line(stdout_bytes)
    except RuntimeError as e:
        sys.stderr.write(f"[sync] phase 1: cannot parse READY line: {e}\n")
        return 5
    sys.stderr.write(
        f"[sync] phase 1: prepared gz = {fmt_bytes(gz_size)} ({gz_size:,} bytes), "
        f"sha256={expected_sha[:16]}...\n"
    )

    n_chunks = (gz_size + chunk_bytes - 1) // chunk_bytes
    sys.stderr.write(
        f"[sync] phase 2: streaming {n_chunks} chunks of "
        f"{fmt_bytes(chunk_bytes)} each\n"
    )

    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp_dir:
        tmp_dir_p = Path(tmp_dir)
        gz_path = tmp_dir_p / "snapshot.db.gz"
        new_db_path = tmp_dir_p / "snapshot.db"

        # ---- PHASE 2: chunk loop ----
        hasher = sha256()
        total_decoded = 0
        with open(gz_path, "wb") as gz_out:
            for i in range(n_chunks):
                offset = i * chunk_bytes
                length = min(chunk_bytes, gz_size - offset)
                last_err: Exception | None = None
                for attempt in range(CHUNK_RETRIES + 1):
                    chunk_cmd = build_ssh_cmd(
                        railway_exe,
                        ["chunk", str(offset), str(length)],
                        script_b64,
                    )
                    t0 = time.time()
                    try:
                        rc, ch_stdout, ch_stderr = run_ssh(chunk_cmd, timeout=600)
                    except subprocess.TimeoutExpired as e:
                        last_err = e
                        sys.stderr.write(
                            f"[sync]   chunk {i+1}/{n_chunks} attempt {attempt+1}: TIMED OUT\n"
                        )
                        continue
                    if rc != 0:
                        last_err = RuntimeError(
                            f"chunk rc={rc}, stderr (tail): {ch_stderr[-500:]!r}"
                        )
                        sys.stderr.write(
                            f"[sync]   chunk {i+1}/{n_chunks} attempt {attempt+1}: "
                            f"rc={rc} -- {ch_stderr[-200:]!r}\n"
                        )
                        continue
                    try:
                        decoded = extract_chunk_payload(ch_stdout)
                    except RuntimeError as e:
                        last_err = e
                        sys.stderr.write(
                            f"[sync]   chunk {i+1}/{n_chunks} attempt {attempt+1}: "
                            f"parse error: {e}\n"
                        )
                        continue
                    if len(decoded) != length:
                        last_err = RuntimeError(
                            f"chunk length mismatch: expected {length}, got {len(decoded)}"
                        )
                        sys.stderr.write(
                            f"[sync]   chunk {i+1}/{n_chunks} attempt {attempt+1}: "
                            f"{last_err}\n"
                        )
                        continue
                    # Success
                    gz_out.write(decoded)
                    hasher.update(decoded)
                    total_decoded += len(decoded)
                    elapsed = time.time() - t0
                    sys.stderr.write(
                        f"[sync]   chunk {i+1}/{n_chunks}: "
                        f"{fmt_bytes(len(decoded))} in {elapsed:.1f}s "
                        f"({fmt_bytes(total_decoded)}/{fmt_bytes(gz_size)})\n"
                    )
                    last_err = None
                    break
                if last_err is not None:
                    sys.stderr.write(
                        f"[sync] phase 2: chunk {i+1}/{n_chunks} failed after "
                        f"{CHUNK_RETRIES+1} attempts: {last_err}\n"
                    )
                    # best-effort cleanup
                    _try_cleanup(railway_exe, script_b64)
                    return 6

        # SHA-256 verification (catches any chunk-stitching error)
        local_sha = hasher.hexdigest()
        if local_sha != expected_sha:
            sys.stderr.write(
                f"[sync] phase 2: SHA-256 MISMATCH\n"
                f"[sync]   expected (remote prepare): {expected_sha}\n"
                f"[sync]   got (local assembled):     {local_sha}\n"
            )
            _try_cleanup(railway_exe, script_b64)
            return 4
        sys.stderr.write(f"[sync] phase 2: sha256 verified ({local_sha[:16]}...)\n")

        # Decompress
        sys.stderr.write("[sync] phase 3a: decompressing gz -> db...\n")
        db_size = decompress_gz(gz_path, new_db_path)
        sys.stderr.write(f"[sync] phase 3a: decompressed db = {fmt_bytes(db_size)}\n")

        # Validate
        sys.stderr.write("[sync] phase 3b: validating SQLite...\n")
        try:
            summary = validate_sqlite(new_db_path)
        except sqlite3.DatabaseError as e:
            sys.stderr.write(f"[sync] FAILED: not a valid SQLite database: {e}\n")
            _try_cleanup(railway_exe, script_b64)
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
            _try_cleanup(railway_exe, script_b64)
            return 3

        # ---- PHASE 4: remote cleanup (best-effort) ----
        _try_cleanup(railway_exe, script_b64)

        if args.dry_run:
            sys.stderr.write(
                f"[sync] dry-run: snapshot at {new_db_path} "
                f"(deleted on exit). Local DB unchanged.\n"
            )
            return 0

        # Atomic-replace
        bak_path = out_path.with_suffix(out_path.suffix + ".bak")
        if out_path.exists():
            if bak_path.exists():
                bak_path.unlink()
            sys.stderr.write(f"[sync] backing up prior DB -> {bak_path.name}\n")
            out_path.rename(bak_path)
        shutil.move(str(new_db_path), str(out_path))
        sys.stderr.write(f"[sync] done. {out_path}\n")

    return 0


def _try_cleanup(railway_exe: str, script_b64: str) -> None:
    try:
        cmd = build_ssh_cmd(railway_exe, ["cleanup"], script_b64)
        rc, _, stderr_blob = run_ssh(cmd, timeout=60)
        if rc == 0:
            sys.stderr.write("[sync] phase 4: remote cleanup ok\n")
        else:
            sys.stderr.write(
                f"[sync] phase 4: cleanup rc={rc} (non-fatal). stderr: {stderr_blob[-200:]!r}\n"
            )
    except Exception as e:
        sys.stderr.write(f"[sync] phase 4: cleanup failed (non-fatal): {e}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sync production surveillance.db to local via chunked railway ssh."
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_LOCAL_DB),
        help="Local DB path to replace (default: surveillance/data/surveillance.db)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the snapshot but do not replace the local DB",
    )
    ap.add_argument(
        "--chunk-mb",
        type=int,
        default=DEFAULT_CHUNK_BYTES // (1024 * 1024),
        help="Per-chunk binary size in MB (default 100). Smaller = more retry-safe.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip phase 1; reuse the prepared gz from a prior run.",
    )
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
