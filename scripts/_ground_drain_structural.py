"""Grounding queries for the Blockscout-drain build. READ-ONLY, 0 Alchemy CU.
Confirms (with real local-DB numbers) the two claims the build rests on:
  1. transferFrom (23b872dd) events are near-absent in transaction_events
     (the structural blind spot — drains target token contracts not in the
     watched set, so the tx_events-join Method 1 cannot see them).
  2. The pending-approval backlog is large and the join finds ~0 matches.
Also sanity-checks row-type behaviour with/without row_factory so the
"regression" narrative can be settled with evidence, not assertion.
"""
import sqlite3, time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
print(f"db={DB}  exists={DB.exists()}  size={DB.stat().st_size/1e9:.1f}GB" if DB.exists() else f"db MISSING {DB}")

# RO connection (mirror prod read path)
ro = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)

def q1(sql, p=()):
    t = time.time()
    v = ro.execute(sql, p).fetchone()[0]
    return v, time.time() - t

# --- Claim 1: transferFrom presence in tx_events ---
n_tf, dt = q1("SELECT COUNT(*) FROM transaction_events WHERE function_selector='23b872dd'")
print(f"[1] transferFrom(23b872dd) rows in transaction_events: {n_tf:,}  ({dt:.1f}s)")

n_tf_recent, dt = q1(
    "SELECT COUNT(*) FROM transaction_events WHERE function_selector='23b872dd' "
    "AND timestamp > '2026-05-29'")
print(f"    of which timestamp>2026-05-29 (last ~7d): {n_tf_recent:,}  ({dt:.1f}s)")

n_tf_nonrev, dt = q1(
    "SELECT COUNT(*) FROM transaction_events WHERE function_selector='23b872dd' AND is_reverted=0")
print(f"    non-reverted transferFroms total: {n_tf_nonrev:,}  ({dt:.1f}s)")

# --- Claim 2: pending backlog + join match count ---
n_pending, dt = q1("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=0")
print(f"[2] pending approvals (drain_detected=0): {n_pending:,}  ({dt:.1f}s)")

n_drained, _ = q1("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1")
print(f"    already drain_detected=1: {n_drained:,}")

# How many pending have ANY matching non-reverted transferFrom on the contract
# after their approval (what Method 1 would catch). Bounded sample for speed.
t = time.time()
match = ro.execute("""
    SELECT COUNT(*) FROM approval_watchlist aw
    WHERE aw.drain_detected=0
      AND EXISTS (
        SELECT 1 FROM transaction_events te
        WHERE te.contract_address = aw.contract_address
          AND te.function_selector='23b872dd' AND te.is_reverted=0
          AND te.timestamp > aw.approve_timestamp)
""").fetchone()[0]
print(f"    pending with a Method-1 (tx_events) match: {match:,}  ({time.time()-t:.1f}s)")

# --- audit_drain_legs cache state (reused by the new detector) ---
try:
    n_cache, _ = q1("SELECT COUNT(*) FROM audit_drain_legs")
    n_cache_out, _ = q1("SELECT COUNT(*) FROM audit_drain_legs WHERE n_out>0")
    n_cache_clean, _ = q1("SELECT COUNT(*) FROM audit_drain_legs WHERE err IS NULL")
    print(f"[cache] audit_drain_legs rows: {n_cache:,}  (n_out>0: {n_cache_out:,}; err IS NULL: {n_cache_clean:,})")
except sqlite3.OperationalError as e:
    print(f"[cache] audit_drain_legs: {e}")

# --- chain coverage for pending (needed to pick the Blockscout base URL) ---
print("[chains] pending approvals by contract chain:")
for chain, c in ro.execute("""
    SELECT COALESCE(c.chain,'(null)') ch, COUNT(*) n
    FROM approval_watchlist aw LEFT JOIN contracts c ON c.contract_address=aw.contract_address
    WHERE aw.drain_detected=0 GROUP BY ch ORDER BY n DESC"""):
    print(f"    {chain}: {c:,}")

# --- row-type evidence (settles the 'regression' question) ---
plain = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
r_plain = plain.execute("SELECT victim_address, contract_address FROM approval_watchlist LIMIT 1").fetchone()
print(f"[rowtype] no row_factory -> type={type(r_plain).__name__}; dict-access works? ", end="")
try:
    _ = r_plain["victim_address"]; print("YES")
except TypeError as e:
    print(f"NO ({e})")
rowf = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); rowf.row_factory = sqlite3.Row
r_row = rowf.execute("SELECT victim_address, contract_address FROM approval_watchlist LIMIT 1").fetchone()
print(f"[rowtype] row_factory=Row -> type={type(r_row).__name__}; dict-access works? ", end="")
try:
    _ = r_row["victim_address"]; print("YES")
except TypeError as e:
    print(f"NO ({e})")
print("done")
