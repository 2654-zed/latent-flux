"""Fast delta sync for the trading software's Tier-1 tables.

The trading H9 paper-trade path reads only:
  org_transfer_events  (THE H9 transfer-role-entropy signal)
  liquidity_events     (info-lens 2nd source)
  poisoning_events     (graph lens only, optional, tiny)

All three are id INTEGER PRIMARY KEY AUTOINCREMENT, append-only per L3's
immutable-record charter (verified: max-id row == max-timestamp row), so
the complete + correct delta is simply `WHERE id > local_max(id)` per table.
No UPDATE / changed-row handling needed.

This replaces the 3-hour 17 GB full clone for keeping trading fed. Typical
daily delta = a few hundred K rows = seconds.

Modes:
  --from-db PATH   pull the delta from an existing SQLite (e.g. a freshly
                   synced surveillance_new.db) — 0 prod load, pure local.
  --from-prod      pull the delta from Railway prod via `railway ssh`
                   sqlite3 .dump of just the new rows (small payload).
  (default target is the live trading DB surveillance/data/surveillance.db)

Safe: only INSERTs new rows into the 3 tables. Never drops/updates. Uses a
busy_timeout so it coexists with a running trading reader.

CLI:
  python scripts/delta_sync_trading.py --from-db surveillance/data/surveillance_new.db
  python scripts/delta_sync_trading.py --from-prod
  python scripts/delta_sync_trading.py --from-db ... --dry-run
"""
from __future__ import annotations
import argparse, sqlite3, subprocess, sys, time, shutil, base64, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "surveillance" / "data" / "surveillance.db"
PROD_DB = "/app/surveillance/data/surveillance.db"
TABLES = ["org_transfer_events", "liquidity_events", "poisoning_events"]


def local_max_ids(conn):
    out = {}
    for t in TABLES:
        try:
            out[t] = conn.execute(f"SELECT COALESCE(MAX(id),0) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            out[t] = None  # table absent
    return out


def delta_from_db(target, src_path, dry_run):
    conn = sqlite3.connect(target, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    maxids = local_max_ids(conn)
    conn.execute("ATTACH ? AS src", (str(src_path),))
    report = {}
    total = 0
    for t in TABLES:
        lm = maxids.get(t)
        if lm is None:
            report[t] = "target table absent — skipped"
            continue
        avail = conn.execute(f"SELECT COUNT(*) FROM src.{t} WHERE id > ?", (lm,)).fetchone()[0]
        report[t] = {"local_max": lm, "new_rows": avail}
        total += avail
        if not dry_run and avail:
            conn.execute(f"INSERT INTO {t} SELECT * FROM src.{t} WHERE id > ?", (lm,))
    if not dry_run:
        conn.commit()
    conn.execute("DETACH src")
    # post counts
    for t in TABLES:
        if isinstance(report[t], dict):
            report[t]["target_max_after"] = conn.execute(f"SELECT COALESCE(MAX(id),0) FROM {t}").fetchone()[0]
    conn.close()
    return report, total


def fetch_prod_delta(maxids):
    """Pull only new rows from prod as SQL INSERT statements via railway ssh
    sqlite3 .dump-style export, base64-framed to survive the transport.
    Returns a path to a local .sql file."""
    exe = shutil.which("railway.cmd") or shutil.which("railway")
    if not exe:
        raise RuntimeError("railway CLI not found")
    # Build a prod-side python that dumps new rows of each table as INSERTs,
    # gzip+base64 to stdout. Small payload (only the delta).
    py = (
        "import sqlite3,gzip,base64,io,sys\n"
        f"c=sqlite3.connect('{PROD_DB}',timeout=30)\n"
        "buf=io.StringIO()\n"
        f"maxids={maxids!r}\n"
        f"for t in {TABLES!r}:\n"
        "    lm=maxids.get(t)\n"
        "    if lm is None: continue\n"
        "    try:\n"
        "        cur=c.execute(f'SELECT * FROM {t} WHERE id > ?',(lm,))\n"
        "    except sqlite3.OperationalError: continue\n"
        "    cols=[d[0] for d in cur.description]\n"
        "    for row in cur:\n"
        "        vals=','.join('NULL' if v is None else (repr(v) if not isinstance(v,(int,float)) else str(v)) for v in row)\n"
        "        buf.write(f'INSERT INTO {t} ({chr(44).join(cols)}) VALUES ({vals});\\n')\n"
        "sys.stdout.write(base64.b64encode(gzip.compress(buf.getvalue().encode())).decode())\n"
    )
    b64 = base64.b64encode(py.encode()).decode()
    bootstrap = "python3 -c \"import base64;exec(base64.b64decode('" + b64 + "'))\""
    cmd = ["cmd.exe", "/c", exe, "ssh", bootstrap]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"prod delta fetch failed: {r.stderr[-500:]}")
    import gzip
    data = gzip.decompress(base64.b64decode(r.stdout.strip()))
    f = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
    f.write(data); f.close()
    return f.name


def delta_from_prod(target, dry_run):
    conn = sqlite3.connect(target, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    maxids = local_max_ids(conn)
    conn.close()
    sql_path = fetch_prod_delta(maxids)
    n_lines = sum(1 for _ in open(sql_path, encoding="utf-8"))
    report = {"insert_statements": n_lines, "maxids_before": maxids}
    if not dry_run and n_lines:
        conn = sqlite3.connect(target, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(open(sql_path, encoding="utf-8").read())
        conn.commit()
        for t in TABLES:
            try:
                report[f"{t}_max_after"] = conn.execute(f"SELECT COALESCE(MAX(id),0) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                pass
        conn.close()
    return report, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(TARGET))
    ap.add_argument("--from-db", default=None, help="source SQLite path (e.g. surveillance_new.db)")
    ap.add_argument("--from-prod", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.target).exists():
        print(f"target not found: {args.target}"); return 1

    t0 = time.time()
    if args.from_db:
        report, total = delta_from_db(args.target, Path(args.from_db), args.dry_run)
        print(f"=== DELTA from {args.from_db} ({'DRY-RUN' if args.dry_run else 'APPLIED'}) ===")
    elif args.from_prod:
        report, total = delta_from_prod(args.target, args.dry_run)
        print(f"=== DELTA from PROD ({'DRY-RUN' if args.dry_run else 'APPLIED'}) ===")
    else:
        print("specify --from-db PATH or --from-prod"); return 1

    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  total new rows: {total}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
