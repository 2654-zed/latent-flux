"""Chunked upload v4 — resume + robust probe retry.

Improvements over v3:
  - At startup, query current staging_size and resume from that offset.
    A dead client process no longer starts over from zero.
  - Probe retries up to 6 times with increasing backoff before giving up,
    so transient 502s / 500s during recovery don't fatally abort.
  - Truncate-staging retries similarly.
  - If the resume point is > 0 and the server file ends mid-chunk (i.e.,
    not on a 2 MB boundary), the script truncates the staging to the
    nearest multiple of CHUNK below the current size before resuming.
"""
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://spypy.up.railway.app"
TOKEN = "jfwufhnsuisfj"
CHUNK = 2 * 1024 * 1024
SRC = Path(r"C:\Users\jason\AppData\Local\Temp\surveillance.db.gz")


def _post(path: str, body: bytes, headers: dict, timeout: int = 120):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
            **headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def probe_staging_size(attempts: int = 6) -> int:
    for i in range(attempts):
        try:
            body = _post("/admin/upload-chunk", b"", {"X-Chunk-Offset": "1"}, timeout=60)
            return body.get("staging_size", -1)
        except Exception as e:
            backoff = min(3 * (i + 1), 15)
            print(f"    probe attempt {i+1}/{attempts} err: {e}; backoff {backoff}s",
                  flush=True)
            time.sleep(backoff)
    return -1


def truncate_staging(target: int, attempts: int = 6) -> int:
    for i in range(attempts):
        try:
            body = _post("/admin/truncate-staging", b"",
                         {"X-Target-Size": str(target)}, timeout=60)
            return body.get("staging_size", -1)
        except Exception as e:
            backoff = min(3 * (i + 1), 15)
            print(f"    truncate attempt {i+1}/{attempts} err: {e}; backoff {backoff}s",
                  flush=True)
            time.sleep(backoff)
    return -1


def send_chunk(buf: bytes, sent_before: int, attempt: int = 0) -> bool:
    if attempt > 5:
        return False
    expected_after = sent_before + len(buf)
    try:
        body = _post("/admin/upload-chunk", buf,
                     {"X-Chunk-Offset": str(sent_before)}, timeout=120)
        srv = body.get("staging_size", -1)
        if srv == expected_after:
            return True
        # Unexpected but success-shaped response; treat as drift
        print(f"    unexpected size after success: srv={srv}, expected={expected_after}",
              flush=True)
        trunc = truncate_staging(sent_before)
        if trunc != sent_before:
            return False
        return send_chunk(buf, sent_before, attempt + 1)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"    chunk err at sent_before={sent_before}: {e}", flush=True)
        time.sleep(3)
        probe = probe_staging_size()
        if probe == expected_after:
            return True
        if probe == sent_before:
            return send_chunk(buf, sent_before, attempt + 1)
        if probe > sent_before:
            # Partial write; roll back and retry
            trunc = truncate_staging(sent_before)
            if trunc != sent_before:
                return False
            return send_chunk(buf, sent_before, attempt + 1)
        print(f"    probe returned {probe}; unrecoverable from sent_before={sent_before}",
              flush=True)
        return False


def main():
    total = SRC.stat().st_size
    print(f"uploading {SRC}  size={total:,} bytes  chunk={CHUNK:,}", flush=True)

    # Resume: probe current staging, align to CHUNK boundary, seek file
    current = probe_staging_size()
    if current < 0:
        print("could not probe staging size; ABORT", flush=True)
        sys.exit(1)
    print(f"current staging on server: {current:,} bytes", flush=True)

    # Align to chunk boundary below current (truncate forward if needed)
    aligned = (current // CHUNK) * CHUNK
    if aligned != current:
        print(f"aligning staging from {current} to {aligned} (chunk boundary)", flush=True)
        if truncate_staging(aligned) != aligned:
            print("truncate to alignment failed; ABORT", flush=True)
            sys.exit(1)

    sent = aligned
    if sent == total:
        print("staging already complete; jumping straight to finalize", flush=True)
    else:
        start = time.time()
        last_print = sent
        with open(SRC, "rb") as f:
            f.seek(sent)
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                if not send_chunk(buf, sent):
                    print(f"\nABORT at sent={sent}", flush=True)
                    sys.exit(1)
                sent += len(buf)
                if sent - last_print >= 20 * 1024 * 1024 or sent == total:
                    rate = (sent - aligned) / max(time.time() - start, 0.001) / (1024 * 1024)
                    pct = 100 * sent / total
                    print(f"  {sent:>12,}/{total:,} ({pct:5.1f}%)  rate={rate:.2f}MB/s",
                          flush=True)
                    last_print = sent
        elapsed = time.time() - start
        print(f"\nupload span complete in {elapsed:.1f}s. finalizing...", flush=True)

    # Finalize
    try:
        body = _post("/admin/finalize-upload", b"", {}, timeout=600)
        print("finalize:", json.dumps(body, indent=2), flush=True)
    except urllib.error.HTTPError as e:
        print(f"finalize HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
              flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"finalize err: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
