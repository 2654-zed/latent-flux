"""Remote-side sync. See scripts/sync_prod_db.py for full architecture.

Two-phase: prepare (backup+gzip to fixed path; print READY:size:sha256),
then chunk <offset> <length> (stream base64 of that slice). cleanup removes
the file. sha256 re-emits READY without re-running prepare.

Compact form (kept under ~5 KB so its base64 fits the Windows cmd.exe 8191
char arg limit when wrapped by `python3 -c "exec(base64.b64decode('...'))"`).
"""
import base64, gzip, hashlib, os, sqlite3, sys, tempfile

P = "/app/surveillance/data/surveillance.db"
G = "/tmp/l3sync_snapshot.db.gz"
MS = "===L3SYNC_PAYLOAD_START==="
ME = "===L3SYNC_PAYLOAD_END==="
B = 48 * 1024


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1024 * 1024)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def prepare():
    if not os.path.exists(P):
        sys.stderr.write(f"PROD_DB not found at {P}\n")
        return 2
    if os.path.exists(G):
        os.unlink(G)
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir="/tmp")
    bp = fd.name
    fd.close()
    s = sqlite3.connect(P)
    d = sqlite3.connect(bp)
    s.backup(d)
    d.close()
    s.close()
    bs = os.path.getsize(bp)
    sys.stderr.write(f"backup: {bs} bytes\n")
    with open(bp, "rb") as fi, gzip.open(G, "wb", compresslevel=6) as fo:
        while True:
            c = fi.read(1024 * 1024)
            if not c:
                break
            fo.write(c)
    os.unlink(bp)
    gs = os.path.getsize(G)
    sys.stderr.write(f"gzip: {gs} bytes ({gs * 100 // bs}% of raw)\n")
    h = _sha(G)
    sys.stderr.write(f"sha256: {h}\n")
    sys.stdout.write(f"READY:{gs}:{h}\n")
    sys.stdout.flush()
    return 0


def chunk(off, ln):
    if not os.path.exists(G):
        sys.stderr.write(f"gz not found at {G}\n")
        return 2
    gs = os.path.getsize(G)
    if off < 0 or off > gs:
        sys.stderr.write(f"invalid offset {off} (gz_size={gs})\n")
        return 2
    end = min(off + ln, gs)
    n = end - off
    sys.stderr.write(f"chunk: off={off} len={n} gz_size={gs}\n")
    sys.stdout.buffer.write((MS + "\n").encode())
    sys.stdout.buffer.flush()
    rem = n
    with open(G, "rb") as f:
        f.seek(off)
        while rem > 0:
            need = min(B, rem)
            c = f.read(need)
            if not c:
                break
            rem -= len(c)
            sys.stdout.buffer.write(base64.b64encode(c))
            sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.write((ME + "\n").encode())
    sys.stdout.buffer.flush()
    sys.stderr.write(f"chunk done: wrote {n - rem} bytes\n")
    return 0


def cleanup():
    if os.path.exists(G):
        os.unlink(G)
        sys.stderr.write(f"cleaned: {G}\n")
    sys.stdout.write("CLEANED\n")
    sys.stdout.flush()
    return 0


def sha_only():
    if not os.path.exists(G):
        sys.stderr.write(f"gz not found at {G}\n")
        return 2
    gs = os.path.getsize(G)
    h = _sha(G)
    sys.stdout.write(f"READY:{gs}:{h}\n")
    sys.stdout.flush()
    return 0


def main():
    a = sys.argv[1:]
    if not a:
        return prepare()
    m = a[0]
    if m == "prepare":
        return prepare()
    if m == "chunk":
        return chunk(int(a[1]), int(a[2])) if len(a) == 3 else 2
    if m == "cleanup":
        return cleanup()
    if m == "sha256":
        return sha_only()
    sys.stderr.write(f"unknown mode: {m}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
