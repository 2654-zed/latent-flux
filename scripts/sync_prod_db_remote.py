"""Remote-side sync. See scripts/sync_prod_db.py for full architecture.

Modes:
  start    fork prepare into detached daemon, print STARTED
  status   print STATUS:READY:size:sha | STATUS:RUNNING | STATUS:ERROR:msg
  chunk    stream a slice of the prepared gz file
  cleanup  remove prepared + status files
  sha256   re-emit READY without re-running prepare

Background prepare: Railway imposes a wall-clock cap (~4 min observed) on a
single `railway ssh` session regardless of activity. v1/v2/v3 all died
mid-prepare. Fix: double-fork + setsid so the prepare survives parent
disconnect; local script polls status via short SSH calls.

Compact form fits the 8191-char cmd.exe budget.
"""
import base64, gzip, hashlib, os, sqlite3, sys, tempfile, time

P = "/app/surveillance/data/surveillance.db"
G = "/tmp/l3sync_snapshot.db.gz"
ST = "/tmp/l3sync_status"
LG = "/tmp/l3sync_log"
MS = "===L3SYNC_PAYLOAD_START==="
ME = "===L3SYNC_PAYLOAD_END==="
B = 48 * 1024


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(1024 * 1024)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def _wst(s):
    with open(ST, "w") as f:
        f.write(s + "\n")


def _log(s):
    with open(LG, "a") as f:
        f.write(f"[{int(time.time())}] {s}\n")


def _do_prepare():
    try:
        if not os.path.exists(P):
            _wst(f"ERROR:PROD_DB not found at {P}")
            return 2
        if os.path.exists(G):
            os.unlink(G)
        _log("backup start")
        fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir="/tmp")
        bp = fd.name
        fd.close()
        s = sqlite3.connect(P)
        d = sqlite3.connect(bp)
        s.backup(d)
        d.close()
        s.close()
        bs = os.path.getsize(bp)
        _log(f"backup done {bs}")
        with open(bp, "rb") as fi, gzip.open(G, "wb", compresslevel=6) as fo:
            while True:
                c = fi.read(1024 * 1024)
                if not c:
                    break
                fo.write(c)
        os.unlink(bp)
        gs = os.path.getsize(G)
        _log(f"gzip done {gs}")
        h = _sha(G)
        _log(f"sha {h}")
        _wst(f"READY:{gs}:{h}")
        return 0
    except Exception as e:
        _wst(f"ERROR:{type(e).__name__}:{e}")
        return 2


def start():
    if os.path.exists(ST):
        with open(ST) as f:
            v = f.read().strip()
        if v.startswith("READY:"):
            sys.stdout.write(f"ALREADY_READY {v}\n")
            sys.stdout.flush()
            return 0
        os.unlink(ST)
    if os.path.exists(LG):
        os.unlink(LG)
    _wst("RUNNING")
    p1 = os.fork()
    if p1 == 0:
        os.setsid()
        p2 = os.fork()
        if p2 == 0:
            try:
                os.close(0); os.close(1); os.close(2)
            except OSError:
                pass
            os._exit(_do_prepare())
        os._exit(0)
    os.waitpid(p1, 0)
    sys.stdout.write("STARTED\n")
    sys.stdout.flush()
    return 0


def status():
    if not os.path.exists(ST):
        sys.stdout.write("STATUS:NO_PREPARE\n")
        sys.stdout.flush()
        return 2
    with open(ST) as f:
        v = f.read().strip()
    sys.stdout.write(f"STATUS:{v}\n")
    if os.path.exists(LG):
        try:
            with open(LG) as f:
                ls = f.readlines()
            for ln in ls[-5:]:
                sys.stdout.write(f"  LOG: {ln.rstrip()}\n")
        except OSError:
            pass
    sys.stdout.flush()
    if v.startswith("READY:"):
        return 0
    if v.startswith("ERROR:"):
        return 2
    return 3


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
    for f in (G, ST, LG):
        if os.path.exists(f):
            try:
                os.unlink(f)
                sys.stderr.write(f"cleaned: {f}\n")
            except OSError as e:
                sys.stderr.write(f"could not unlink {f}: {e}\n")
    sys.stdout.write("CLEANED\n")
    sys.stdout.flush()
    return 0


def sha_only():
    if not os.path.exists(G):
        sys.stderr.write(f"gz not found at {G}\n")
        return 2
    gs = os.path.getsize(G)
    sys.stdout.write(f"READY:{gs}:{_sha(G)}\n")
    sys.stdout.flush()
    return 0


def main():
    a = sys.argv[1:]
    if not a:
        sys.stderr.write("usage: start|status|chunk OFF LEN|cleanup|sha256\n")
        return 2
    m = a[0]
    if m == "start": return start()
    if m == "status": return status()
    if m == "chunk": return chunk(int(a[1]), int(a[2])) if len(a) == 3 else 2
    if m == "cleanup": return cleanup()
    if m == "sha256": return sha_only()
    sys.stderr.write(f"unknown mode: {m}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
