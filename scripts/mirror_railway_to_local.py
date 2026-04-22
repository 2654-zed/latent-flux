"""Mirror Railway's DB to local.

Flow:
  1. POST /admin/prepare-snapshot — Railway runs sqlite3.backup() to produce
     a consistent gzipped snapshot on-volume.
  2. GET /admin/download-chunk?offset=N&size=M (resumable) until the full
     mirror.gz is pulled to C:\\...\\Temp\\railway_mirror.gz.
  3. gunzip locally to railway_mirror.db.
  4. Atomic swap into surveillance/data/surveillance.db (previous local DB
     is preserved at surveillance/data/surveillance.db.prev).

Safety: local DB is never overwritten until the downloaded mirror passes
integrity_check. The old file is renamed, not deleted.

CLI:
    python scripts/mirror_railway_to_local.py
    python scripts/mirror_railway_to_local.py --no-swap  # download only
    python scripts/mirror_railway_to_local.py --resume   # continue prior partial
"""
import argparse
import gzip
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://spypy.up.railway.app"
TOKEN = "jfwufhnsuisfj"
CHUNK = 4 * 1024 * 1024  # 4 MB

MIRROR_GZ = Path(r"C:\Users\jason\AppData\Local\Temp\railway_mirror.gz")
MIRROR_DB = Path(r"C:\Users\jason\AppData\Local\Temp\railway_mirror.db")
LOCAL_DB = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
LOCAL_DB_PREV = LOCAL_DB.with_suffix(".db.prev")


def _post(path: str) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=b"",
        headers={"Authorization": f"Bearer {TOKEN}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def prepare_snapshot() -> dict:
    """Trigger server-side snapshot. Returns {'gz_size_bytes': N, ...}."""
    url = f"{BASE_URL}/admin/prepare-snapshot?token={TOKEN}"
    req = urllib.request.Request(url, method="GET")
    # This can take 30-120s on a 1.8 GB DB while sqlite3.backup() runs + gzip
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def download_chunk(offset: int, size: int, attempts: int = 6) -> tuple[bytes, int]:
    """Return (body, total_size). Retries on transient network errors."""
    url = f"{BASE_URL}/admin/download-chunk?token={TOKEN}&offset={offset}&size={size}"
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=180) as resp:
                total = int(resp.headers.get("X-Total-Size", "-1"))
                body = resp.read()
                return body, total
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            backoff = min(3 * (i + 1), 15)
            print(f"    attempt {i+1}/{attempts} err: {e}; backoff {backoff}s",
                  flush=True)
            time.sleep(backoff)
    raise RuntimeError(f"download-chunk failed after {attempts} attempts at offset={offset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-swap", action="store_true",
                    help="Download + decompress only; don't swap into surveillance/data/.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing partial railway_mirror.gz if present.")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="Skip prepare-snapshot (use the existing on-server mirror.gz).")
    args = ap.parse_args()

    # Step 1: prepare snapshot
    if args.skip_prepare:
        print("skipping prepare-snapshot", flush=True)
        # Probe total via a zero-size chunk
        _, total = download_chunk(0, 1)
        expected_size = total
        print(f"existing mirror.gz size: {expected_size:,} bytes", flush=True)
    else:
        print("requesting snapshot preparation (Railway runs online backup + gzip)...",
              flush=True)
        t0 = time.time()
        result = prepare_snapshot()
        print(f"prepare result: {json.dumps(result, indent=2)}", flush=True)
        expected_size = int(result.get("gz_size_bytes", 0))
        print(f"snapshot ready in {time.time()-t0:.1f}s  gz_size={expected_size:,}",
              flush=True)

    if expected_size <= 0:
        print("ABORT: invalid expected_size", flush=True)
        sys.exit(1)

    # Step 2: chunked download (resumable)
    start_offset = 0
    if args.resume and MIRROR_GZ.exists():
        start_offset = MIRROR_GZ.stat().st_size
        if start_offset >= expected_size:
            print(f"already fully downloaded ({start_offset:,} bytes)", flush=True)
        else:
            print(f"resuming from offset {start_offset:,}", flush=True)

    if start_offset == 0 and MIRROR_GZ.exists():
        MIRROR_GZ.unlink()

    t0 = time.time()
    with open(MIRROR_GZ, "ab" if start_offset > 0 else "wb") as f:
        offset = start_offset
        while offset < expected_size:
            remaining = expected_size - offset
            req_size = min(CHUNK, remaining)
            body, total = download_chunk(offset, req_size)
            if total != expected_size:
                print(f"WARN: server total_size changed {expected_size} -> {total}",
                      flush=True)
                expected_size = total
            if len(body) == 0:
                print(f"WARN: server returned empty chunk at offset={offset}", flush=True)
                break
            f.write(body)
            offset += len(body)
            rate = (offset - start_offset) / max(time.time() - t0, 0.001) / (1024 * 1024)
            pct = 100 * offset / expected_size
            eta = (expected_size - offset) / max(rate * 1024 * 1024, 1)
            print(f"  {offset:>12,}/{expected_size:,} ({pct:5.1f}%)  rate={rate:.2f}MB/s  eta={eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\ndownload complete in {elapsed:.1f}s  total={offset:,} bytes", flush=True)

    if offset != expected_size:
        print(f"WARN: downloaded {offset:,} != expected {expected_size:,}", flush=True)

    # Step 3: decompress
    print(f"decompressing to {MIRROR_DB}...", flush=True)
    with gzip.open(MIRROR_GZ, "rb") as fin, open(MIRROR_DB, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)
    print(f"decompressed: {MIRROR_DB.stat().st_size:,} bytes", flush=True)

    # Step 4: integrity check
    print("running integrity check on mirror...", flush=True)
    conn = sqlite3.connect(str(MIRROR_DB))
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    n_contracts = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    n_deployers = conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
    n_traps = conn.execute("SELECT COUNT(*) FROM trap_events").fetchone()[0]
    latest = conn.execute("SELECT MAX(detection_timestamp) FROM contracts").fetchone()[0]
    conn.close()
    print(f"  integrity: {integrity}")
    print(f"  contracts={n_contracts:,}  deployers={n_deployers:,}  trap_events={n_traps:,}")
    print(f"  latest detection: {latest}")

    if integrity != "ok":
        print("ABORT: integrity check failed, not swapping", flush=True)
        sys.exit(1)

    # Step 5: atomic swap (optional)
    if args.no_swap:
        print(f"--no-swap given; mirror left at {MIRROR_DB}", flush=True)
        return

    if LOCAL_DB.exists():
        print(f"renaming current local DB to {LOCAL_DB_PREV}", flush=True)
        if LOCAL_DB_PREV.exists():
            LOCAL_DB_PREV.unlink()
        LOCAL_DB.rename(LOCAL_DB_PREV)

    # Also sweep WAL/SHM so SQLite starts fresh on next open
    for ext in ("-wal", "-shm"):
        p = Path(str(LOCAL_DB) + ext)
        if p.exists():
            p.unlink()

    shutil.copy2(MIRROR_DB, LOCAL_DB)
    print(f"swapped in: {LOCAL_DB} = {LOCAL_DB.stat().st_size:,} bytes", flush=True)
    print(f"prior local DB preserved at: {LOCAL_DB_PREV}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
