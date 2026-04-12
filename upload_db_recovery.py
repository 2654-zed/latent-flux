"""Upload a local DB to Railway via chunked upload.

Usage: python upload_db_recovery.py [db_file.gz]
"""
import os
import sys
import time
import urllib.request
import urllib.error
import json

BASE_URL = os.environ.get("RAILWAY_URL", "https://spypy.up.railway.app")
TOKEN = os.environ.get("ADMIN_TOKEN")
if not TOKEN:
    TOKEN = input("ADMIN_TOKEN: ").strip()

CHUNK_SIZE = 5_000_000  # 5MB per chunk


def upload_chunk(data, offset):
    """Send a single chunk."""
    req = urllib.request.Request(
        f"{BASE_URL}/admin/upload-chunk",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
            "X-Chunk-Offset": str(offset),
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def finalize():
    """Tell Railway to decompress and replace DB."""
    req = urllib.request.Request(
        f"{BASE_URL}/admin/finalize-upload",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "surveillance/data/surveillance_minimal.db.gz"
    if not os.path.exists(src):
        print(f"File not found: {src}")
        sys.exit(1)

    file_size = os.path.getsize(src)
    total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Uploading {src} ({file_size / 1024 / 1024:.1f} MB) in {total_chunks} chunks")

    with open(src, "rb") as f:
        offset = 0
        chunk_num = 0
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            chunk_num += 1
            for attempt in range(3):
                try:
                    result = upload_chunk(data, offset)
                    print(f"  Chunk {chunk_num}/{total_chunks}: {result.get('staging_size', 0):,} bytes staged")
                    break
                except Exception as e:
                    print(f"  Chunk {chunk_num} attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        print("FATAL: giving up")
                        sys.exit(1)
            offset += len(data)
            time.sleep(0.2)  # Be nice

    print(f"\nAll chunks uploaded. Finalizing...")
    try:
        result = finalize()
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Finalize failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
