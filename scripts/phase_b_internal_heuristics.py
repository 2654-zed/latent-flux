"""Phase B — internal heuristics on confirmed-tier residual after Phase A.

Per reports/confirmed_tier_audit_plan.md Phase B:

  Apply purely-internal FP heuristics to the residual NEEDS_REVIEW + EDGE
  populations (no new API calls):

    - Recidivist deployer (>=2 confirmed) + not institutionally-tagged
      -> LIKELY_TP (keep confirmed; do NOT downgrade)

    - Self-loop / BACKFILL reason + solo deployer + behavioral-only
      -> LIKELY_FP_WEAK (candidate to downgrade)

    - Drain/tx ratio >= 30 -> BUG_19B_SUSPECT (residual from-matching
      bug; flag for manual review, do not auto-downgrade)

    - Verified ERC-20 with <10 holders (the EDGE cases from Phase A)
      -> STILL_NEEDS_REVIEW (manual source-code inspection required)

    - All others -> NEEDS_REVIEW (Phase C work)

Input: reports/confirmed_tier_audit_2026-05-22.csv (Phase A CSV)
       + local DB for internal joins (deployer recidivism, drain ratios,
       bytecode_cache presence)

Output: reports/confirmed_tier_audit_phase_b_2026-05-22.csv with phase_b
        verdict column appended.

Phase B does NOT migrate anything. It produces classifications. The
migration decisions (Phase D follow-up) are reviewed by the user.

CLI:
    python scripts/phase_b_internal_heuristics.py
"""
from __future__ import annotations
import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_2026-05-22.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_b_2026-05-22.csv"


def load_internal_signals(conn: sqlite3.Connection) -> dict:
    """Compute internal signals once, joined by contract_address.

    Returns dict keyed by contract_address with fields:
        deployer_confirmed_count (int)  — # other confirmed contracts by same deployer
        has_bytecode_cache (bool)
        bytecode_flags (str)            — 'asym/rev/fee' format
        drain_count (int)
        drain_tx_count (int)
        drain_ratio (float)             — drain_count / drain_tx_count
        reason_class (str)              — 'self_loop' | 'backfill' | 'behavioral' | 'other'
        deployer_oli_tag_count (int)    — count of OLI tags on deployer (Bug #22 affected — likely 0)
    """
    print("  loading internal signals...")

    # Recidivism: how many other confirmed contracts does each deployer have?
    # Phase D already moved 116 contracts to unanalyzed, so recidivism is computed
    # against the POST-MIGRATION confirmed population.
    recidivism = {}
    for r in conn.execute(
        "SELECT deployer_address, COUNT(*) "
        "FROM contracts WHERE confidence_tier='confirmed' AND deployer_address IS NOT NULL "
        "GROUP BY deployer_address"
    ):
        recidivism[r[0]] = r[1]

    # Bytecode_cache presence + flags
    bytecode_signals = {}
    for r in conn.execute(
        "SELECT c.contract_address, b.bytecode_signals "
        "FROM contracts c LEFT JOIN bytecode_cache b ON b.code_hash = c.deployed_code_hash "
        "WHERE c.confidence_tier IN ('confirmed', 'unanalyzed')"  # both for post-migration cross-check
    ):
        if r[1]:
            import json as _json
            try:
                d = _json.loads(r[1])
                flags = "/".join([
                    "asym" if d.get("has_asymmetric_transfer") else "-",
                    "rev" if d.get("has_conditional_revert") else "-",
                    "fee" if d.get("has_unusual_fee_structure") else "-",
                ])
                bytecode_signals[r[0]] = flags
            except Exception:
                bytecode_signals[r[0]] = "parse_err"
        else:
            bytecode_signals[r[0]] = None  # no cache row

    # Drain count + tx count per contract (using current state — Phase 0 backfill applied)
    drain_stats = {}
    for r in conn.execute("""
        SELECT contract_address,
               COUNT(*) AS drains,
               COUNT(DISTINCT drain_tx_hash) AS txs
        FROM approval_watchlist
        WHERE drain_detected = 1 AND drain_tx_hash IS NOT NULL
        GROUP BY contract_address
    """):
        ratio = r[1] / r[2] if r[2] else 0.0
        drain_stats[r[0]] = {"drains": r[1], "txs": r[2], "ratio": ratio}

    # Reason class + deployer OLI tag count
    contract_meta = {}
    for r in conn.execute(
        "SELECT contract_address, confidence_reason, deployer_address "
        "FROM contracts WHERE confidence_tier='confirmed'"
    ):
        reason = (r[1] or "").lower()
        if reason.startswith("self-loop"):
            cls = "self_loop"
        elif reason.startswith("backfill"):
            cls = "backfill"
        elif "behavioral confirmation" in reason:
            cls = "behavioral"
        else:
            cls = "other"
        contract_meta[r[0]] = {"reason_class": cls, "deployer_address": r[2]}

    # Deployer OLI tag count (likely all zero per Bug #22, but check anyway)
    try:
        deployer_oli = {}
        for r in conn.execute(
            "SELECT address, COUNT(*) FROM oli_labels GROUP BY address"
        ):
            deployer_oli[r[0]] = r[1]
    except sqlite3.OperationalError:
        deployer_oli = {}

    # Combine into per-contract record
    out = {}
    # Iterate over all currently-confirmed contracts
    for addr in contract_meta:
        m = contract_meta[addr]
        deployer = m["deployer_address"]
        recidivism_n = recidivism.get(deployer, 0) - 1  # subtract self
        out[addr] = {
            "deployer_address": deployer,
            "deployer_confirmed_count": max(recidivism_n, 0),
            "has_bytecode_cache": bytecode_signals.get(addr) is not None,
            "bytecode_flags": bytecode_signals.get(addr) or "",
            "drain_count": drain_stats.get(addr, {}).get("drains", 0),
            "drain_tx_count": drain_stats.get(addr, {}).get("txs", 0),
            "drain_ratio": drain_stats.get(addr, {}).get("ratio", 0.0),
            "reason_class": m["reason_class"],
            "deployer_oli_tag_count": deployer_oli.get(deployer, 0),
        }
    return out


def classify_phase_b(row: dict, signals: dict) -> str:
    """Apply Phase B heuristics.
    Returns one of:
      LIKELY_TP_RECIDIVIST   — recidivist deployer, keep
      LIKELY_FP_WEAK         — self-loop / BACKFILL solo, candidate downgrade
      BUG_19B_SUSPECT        — high drain/tx ratio, investigate
      STILL_NEEDS_REVIEW     — Phase A EDGE case, Phase C manual review needed
      NEEDS_REVIEW           — residual, Phase C
    """
    addr = row["contract_address"]
    s = signals.get(addr, {})

    # Phase A LIKELY_FP cases already migrated — don't touch
    if row["preliminary_verdict"] == "LIKELY_FP":
        return "ALREADY_MIGRATED"
    # Phase A LIKELY_TP — keep
    if row["preliminary_verdict"] == "LIKELY_TP":
        return "LIKELY_TP_PHASE_A"

    recidivism_n = s.get("deployer_confirmed_count", 0)
    oli_tags = s.get("deployer_oli_tag_count", 0)
    drain_ratio = s.get("drain_ratio", 0.0)
    drain_txs = s.get("drain_tx_count", 0)
    reason_class = s.get("reason_class", "other")

    # Bug 19b suspect — high ratio, residual from-matching bug
    if drain_ratio >= 30 and drain_txs >= 1:
        return "BUG_19B_SUSPECT"

    # Phase A EDGE: verified+ERC20 with <10 holders
    try:
        hc = int(str(row.get("holders_count") or "").replace(",", "")) if row.get("holders_count") else 0
    except Exception:
        hc = 0
    if row.get("is_verified") == "True" and row.get("token_type") and hc < 10:
        return "STILL_NEEDS_REVIEW"

    # Recidivism — strong TP signal IF no institutional tag
    if recidivism_n >= 2 and oli_tags == 0:
        return "LIKELY_TP_RECIDIVIST"

    # Self-loop / BACKFILL with solo deployer — weak evidence
    if reason_class in ("self_loop", "backfill") and recidivism_n == 0:
        return "LIKELY_FP_WEAK"

    return "NEEDS_REVIEW"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    print(f"Phase B internal heuristics")
    print(f"  DB:     {args.db}")
    print(f"  input:  {args.input}")
    print(f"  output: {args.output}")
    print()

    conn = sqlite3.connect(args.db)
    signals = load_internal_signals(conn)
    print(f"  signals loaded for {len(signals)} currently-confirmed contracts")

    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    print(f"  audit CSV: {len(rows)} rows")

    # Output columns = input + Phase B columns
    out_columns = list(rows[0].keys()) + [
        "deployer_confirmed_count", "has_bytecode_cache", "bytecode_flags",
        "drain_count", "drain_tx_count", "drain_ratio", "reason_class",
        "deployer_oli_tag_count", "phase_b_verdict"
    ]
    from collections import Counter
    counts = Counter()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_columns)
        w.writeheader()
        for r in rows:
            addr = r["contract_address"]
            s = signals.get(addr, {})
            r["deployer_confirmed_count"] = s.get("deployer_confirmed_count", "")
            r["has_bytecode_cache"] = s.get("has_bytecode_cache", "")
            r["bytecode_flags"] = s.get("bytecode_flags", "")
            r["drain_count"] = s.get("drain_count", "")
            r["drain_tx_count"] = s.get("drain_tx_count", "")
            r["drain_ratio"] = f"{s.get('drain_ratio', 0.0):.2f}" if s.get("drain_ratio") else ""
            r["reason_class"] = s.get("reason_class", "")
            r["deployer_oli_tag_count"] = s.get("deployer_oli_tag_count", "")
            verdict = classify_phase_b(r, signals)
            r["phase_b_verdict"] = verdict
            counts[verdict] += 1
            w.writerow(r)

    print()
    print(f"  Phase B verdict counts:")
    total = sum(counts.values())
    for k, v in counts.most_common():
        print(f"    {k:30s}: {v:>5,}  ({100*v/total:.1f}%)")
    print()
    print(f"  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
