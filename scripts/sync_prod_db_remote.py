"""Remote-side sync. See scripts/sync_prod_db.py for full architecture.

Two-phase: prepare (backup+gzip to fixed path; print READY:size:sha256),
then chunk <offset> <length> (stream base64 of that slice). cleanup removes
the file. sha256 re-emits READY without re-running prepare.

Compact form (kept under ~6 KB so its base64 fits the Windows cmd.exe 8191
char arg limit when wrapped by `python3 -c "exec(base64.b64decode('...'))"`).

2026-05-17: prepare() now emits periodic stderr heartbeats during the long
backup() + gzip phases. Without them, Railway's WebSocket idle-timeout
kills the SSH session after ~5 min — diagnosed when a 12 GB DB sync failed
at the 5.8-min mark with rc=1.
"""
import base64, gzip, hashlib, os, sqlite3, sys, tempfile, time

P = "/app/surveillance/data/surveillance.db"
G = "/tmp/l3sync_snapshot.db.gz"
MS = "===L3SYNC_PAYLOAD_START==="
ME = "===L3SYNC_PAYLOAD_END==="
B = 48 * 1024
# Heartbeat cadence: emit stderr line every N seconds during long ops to
# defeat Railway's WebSocket idle-timeout (~5 min observed empirically).
HB_SEC = 30


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

    # Backup with progress callback throttled to HB_SEC for keepalive.
    # pages=10000 makes the callback fire every 10K-page batch; the
    # callback further throttles to time-based intervals.
    _last_hb = [time.time()]
    def _backup_hb(status, remaining, total):
        now = time.time()
        if now - _last_hb[0] >= HB_SEC:
            done = total - remaining
            sys.stderr.write(f"backup_hb: {done}/{total} pages "
                             f"({100 * done // max(total, 1)}%)\n")
            sys.stderr.flush()
            _last_hb[0] = now
    s.backup(d, pages=10000, progress=_backup_hb)
    d.close()
    s.close()
    bs = os.path.getsize(bp)
    sys.stderr.write(f"backup: {bs} bytes\n")
    sys.stderr.flush()

    # Gzip with heartbeat every HB_SEC of wall-clock time.
    _hb = time.time()
    written = 0
    with open(bp, "rb") as fi, gzip.open(G, "wb", compresslevel=6) as fo:
        while True:
            c = fi.read(1024 * 1024)
            if not c:
                break
            fo.write(c)
            written += len(c)
            if time.time() - _hb >= HB_SEC:
                sys.stderr.write(f"gzip_hb: {written}/{bs} bytes "
                                 f"({100 * written // max(bs, 1)}%)\n")
                sys.stderr.flush()
                _hb = time.time()
    os.unlink(bp)
    gs = os.path.getsize(G)
    sys.stderr.write(f"gzip: {gs} bytes ({gs * 100 // bs}% of raw)\n")
    sys.stderr.flush()

    # sha256 with heartbeat per HB_SEC for large gz files.
    _hb = time.time()
    read = 0
    h = hashlib.sha256()
    with open(G, "rb") as f:
        while True:
            c = f.read(1024 * 1024)
            if not c:
                break
            h.update(c)
            read += len(c)
            if time.time() - _hb >= HB_SEC:
                sys.stderr.write(f"sha_hb: {read}/{gs} bytes "
                                 f"({100 * read // max(gs, 1)}%)\n")
                sys.stderr.flush()
                _hb = time.time()
    digest = h.hexdigest()
    sys.stderr.write(f"sha256: {digest}\n")
    sys.stderr.flush()
    sys.stdout.write(f"READY:{gs}:{digest}\n")
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
