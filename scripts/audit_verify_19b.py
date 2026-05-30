"""Verify whether the 1,520-victim single-tx drain is real (multicall) or
Bug #19b over-crediting. Inspect the actual transaction_events row(s) for
that tx_hash: a real mass-drain would show ONE tx touching the contract;
Bug #19b would show the drain credited from a single transferFrom whose
`from` is one address, fanned out to 1,520 approval rows."""
import sqlite3
from pathlib import Path
DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT = Path(__file__).resolve().parent.parent / "reports" / "_audit_19b_verify.txt"
c = sqlite3.connect(DB)
L=[]
def p(s=""): L.append(str(s))

# The worst tx
tx = "cf2fed47275687febeefb5b1"  # prefix only; find full
full = c.execute("""SELECT DISTINCT drain_tx_hash FROM approval_watchlist
  WHERE drain_tx_hash LIKE ? LIMIT 1""", (tx+"%",)).fetchone()
p(f"worst tx (prefix {tx}): full={full[0] if full else None}")
if full:
    txh = full[0]
    # How is it stored in transaction_events?
    te = c.execute("""SELECT tx_hash, contract_address, interacting_address, function_selector, is_reverted, timestamp
      FROM transaction_events WHERE tx_hash=? OR tx_hash=? LIMIT 5""",
      (txh, txh if txh.startswith("0x") else "0x"+txh)).fetchall()
    p(f"\ntransaction_events rows for this tx_hash: {len(te)}")
    for r in te:
        p(f"    contract={r[1][:20]} caller={str(r[2])[:20]} selector={r[3]} reverted={r[4]} ts={r[5]}")
    # The approval rows: do they all share one drain_caller? one approve window?
    agg = c.execute("""SELECT drain_caller, COUNT(*) n, COUNT(DISTINCT victim_address) v,
      MIN(approve_timestamp), MAX(approve_timestamp)
      FROM approval_watchlist WHERE drain_tx_hash=? GROUP BY drain_caller""", (txh,)).fetchall()
    p(f"\napproval_watchlist rows for this drain_tx_hash, grouped by drain_caller:")
    for r in agg:
        p(f"    caller={str(r[0])[:20]} rows={r[1]} victims={r[2]} approve_window=[{str(r[3])[:19]} .. {str(r[4])[:19]}]")
    # Verdict logic
    p("\nINTERPRETATION:")
    p("  If transaction_events shows ONE transferFrom (selector 23b872dd) for this tx,")
    p("  but approval_watchlist credits 1,520 victims to it, that's Bug #19b over-crediting:")
    p("  the drain-detector matched ANY transferFrom on the contract and stamped it onto")
    p("  EVERY pending approver, rather than the single victim whose tokens actually moved.")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
