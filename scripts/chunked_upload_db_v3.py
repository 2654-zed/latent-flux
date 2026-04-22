"""Chunked upload v3 — handles partial writes via truncate-staging endpoint.

On network error:
  1. Probe server staging_size.
  2. If server is beyond where this chunk started, the connection drop left
     partial data on the server. Call /admin/truncate-staging to roll back
     to the pre-chunk size.
  3. Retry the chunk.

Small 2 MB chunks and longer timeouts, to minimize time-in-flight per chunk.
"""
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://spypy.up.railway.app"
TOKEN = "jfwufhnsuisfj"
CHUNK = 2 * 1024 * 1024  # 2 MB — smaller chunks = less drift on partial drops
SRC = Path(r"C:\Users\jason\AppData\Local\Temp\surveillance.db.gz")


def probe_staging_size() -> int:
    """Zero-byte append (offset != 0 so mode='ab'); read back current size."""
    req = urllib.request.Request(
        f"{BASE_URL}/admin/upload-chunk",
        data=b"",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
            "X-Chunk-Offset": "1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("staging_size", -1)
    except Exception as e:
        print(f"    probe err: {e}", flush=True)
        return -1


def truncate_staging(target_size: int) -> int:
    """Return new staging size after truncate, or -1 on error."""
    req = urllib.request.Request(
        f"{BASE_URL}/admin/truncate-staging",
        data=b"",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
            "X-Target-Size": str(target_size),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            return body.get("staging_size", -1)
    except urllib.error.HTTPError as e:
        print(f"    truncate HTTP {e.code}: {e.read().decode(errors='replace')[:200]}", flush=True)
    except Exception as e:
        print(f"    truncate err: {e}", flush=True)
    return -1


def send_chunk(buf: bytes, sent_before: int, attempt: int = 0) -> bool:
    """Send one chunk; return True on success (server size matches expected)."""
    if attempt > 3:
        print(f"    attempts exhausted; ABORT", flush=True)
        return False

    offset = sent_before  # server uses 0 as truncate flag; any other value is append
    expected_after = sent_before + len(buf)

    # If offset == 0 and sent_before == 0, server will truncate+write. Otherwise append.
    req = urllib.request.Request(
        f"{BASE_URL}/admin/upload-chunk",
        data=buf,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
            "X-Chunk-Offset": str(offset),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode())
        srv_size = body.get("staging_size", -1)
        if srv_size == expected_after:
            return True
        # Size mismatch after success response — shouldn't happen, but recover
        print(f"    size mismatch after success: expected {expected_after}, server={srv_size}",
              flush=True)
        if srv_size > sent_before:
            trunc_to = truncate_staging(sent_before)
            if trunc_to == sent_before:
                return send_chunk(buf, sent_before, attempt + 1)
        return False
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"    network err: {e}; recovering...", flush=True)
        time.sleep(3)
        probe = probe_staging_size()
        if probe == expected_after:
            # Chunk actually landed; keep going
            return True
        if probe == sent_before:
            # Nothing changed on server; safe to retry
            return send_chunk(buf, sent_before, attempt + 1)
        if probe > sent_before and probe < expected_after:
            # Partial write — roll back and retry
            print(f"    partial write detected: server={probe}, expected {sent_before}",
                  flush=True)
            trunc_to = truncate_staging(sent_before)
            if trunc_to == sent_before:
                return send_chunk(buf, sent_before, attempt + 1)
            return False
        # Drifted beyond expected_after — unrecoverable
        print(f"    server drifted: {probe} vs expected [{sent_before}..{expected_after}]",
              flush=True)
        return False


def main():
    total = SRC.stat().st_size
    print(f"uploading {SRC}  size={total:,} bytes  chunk={CHUNK:,}", flush=True)

    # Truncate staging to 0 first to ensure clean start
    print("truncating staging to 0 (clean start)...", flush=True)
    truncate_staging(0)

    sent = 0
    chunk_idx = 0
    start = time.time()
    last_print = 0

    with open(SRC, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            if not send_chunk(buf, sent):
                print(f"\nABORT at chunk {chunk_idx}", flush=True)
                sys.exit(1)
            sent += len(buf)
            chunk_idx += 1
            # Rate-limit progress prints to every 10 MB or 10s
            if sent - last_print >= 10 * 1024 * 1024 or time.time() - start - last_print / 1e6 >= 10:
                rate = sent / max(time.time() - start, 0.001) / (1024 * 1024)
                pct = 100 * sent / total
                print(f"  {sent:>12,}/{total:,} ({pct:5.1f}%)  rate={rate:.1f}MB/s  chunk={chunk_idx}",
                      flush=True)
                last_print = sent

    elapsed = time.time() - start
    print(f"\nupload complete in {elapsed:.1f}s. finalizing...", flush=True)

    fin_req = urllib.request.Request(
        f"{BASE_URL}/admin/finalize-upload",
        data=b"",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(fin_req, timeout=600) as resp:
            print("finalize:", resp.read().decode(), flush=True)
    except urllib.error.HTTPError as e:
        print(f"finalize HTTP {e.code}: {e.read().decode(errors='replace')[:500]}", flush=True)
    except Exception as e:
        print(f"finalize err: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
