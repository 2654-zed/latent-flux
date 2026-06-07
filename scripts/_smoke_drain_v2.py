"""Smoke test for the rebuilt tx-initiator check_drains_blockscout (Correction #29).
Seeds a temp DB with known OFC sellers (must NOT be flagged) and runs the real
module function. 0 Alchemy CU."""
import sqlite3, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "surveillance" / "data" / "surveillance.db"
from surveillance.approval_drain_monitor import check_drains_blockscout, _blockscout_drain_check, BLOCKSCOUT_BASE

OFC = "0x752c5a95d202972e124390f30a50154409d3c858"
ro = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
sellers = [r[0] for r in ro.execute(
    "SELECT victim_address FROM approval_watchlist WHERE contract_address=? LIMIT 6", (OFC,))]
print("seeding", len(sellers), "known OFC sellers (expect 0 DRAIN, all SALE)")

tmp = Path(tempfile.gettempdir()) / "l3_drain_v2_smoke.db"
if tmp.exists(): tmp.unlink()
tc = sqlite3.connect(str(tmp))
tc.execute("""CREATE TABLE approval_watchlist(id INTEGER PRIMARY KEY AUTOINCREMENT,
  victim_address TEXT, contract_address TEXT, approve_timestamp TEXT, deployer_address TEXT,
  drain_detected INTEGER DEFAULT 0, drain_tx_hash TEXT, drain_timestamp TEXT, drain_caller TEXT,
  UNIQUE(victim_address, contract_address))""")
tc.execute("CREATE TABLE contracts(contract_address TEXT PRIMARY KEY, chain TEXT)")
tc.execute("INSERT OR IGNORE INTO contracts VALUES(?,?)", (OFC, "base"))
for v in sellers:
    tc.execute("INSERT INTO approval_watchlist(victim_address,contract_address,drain_detected) VALUES(?,?,0)", (v, OFC))
tc.commit()

r = check_drains_blockscout(tc, max_victims=10, db_path=str(tmp))
print("result:", r)
ok = (r["drains_detected"] == 0 and r["sales"] >= 1 and r["errors"] == 0)
print("NEGATIVE CONTROL (OFC sellers not flagged):", "PASS" if ok else "FAIL")

# idempotency: second run = all verdict-cache hits, 0 new
r2 = check_drains_blockscout(tc, max_victims=10, db_path=str(tmp))
print("run#2:", r2, "| idempotent:", "PASS" if (r2["drains_detected"] == 0 and r2["cache_hits"] >= 1) else "FAIL")
tc.close(); tmp.unlink()
print("RESULT:", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
