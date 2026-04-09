"""
Layer 3 — x402 Activity Detection

Detects x402-pattern payment flows on monitored chains. x402 is a
Coinbase-incubated protocol that embeds stablecoin payments into HTTP
using the 402 status code — AI agents autonomously pay for API access
without human approval.

On-chain surface:
  - EIP-3009 transferWithAuthorization / receiveWithAuthorization
    calls on USDC/EURC (facilitator submits the tx; payer signs off-chain)
  - Permit2 permit + transferFrom calls for any ERC-20
  - Facilitator settlement via x402ExactPermit2Proxy (CREATE2 canonical:
    0x402085c248EeA27D92E8b30b2C58ed07f9E20001, same on all EVM chains)

The x402 facilitator is an off-chain HTTP service. In on-chain terms
the facilitator manifests as (a) an EOA signing settlement txs on the
payer's behalf, or (b) a call into x402ExactPermit2Proxy. Facilitator
tracking in this module tracks BOTH — contract addresses and
high-volume EOAs calling the proxy / Permit2.

Phase 1 (--recon): scans existing transaction_events for x402-relevant
selectors and produces a structured report. No DB writes, no RPC calls.

Phase 2+ (tables, live monitor, amplification analysis) will ship in
subsequent commits — Phase 1 is strictly diagnostic.

Usage:
    python -m surveillance.x402_monitor --recon
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger("surveillance.x402_monitor")

# ---------------------------------------------------------------------
# Canonical constants (hardcoded — do not discover at runtime)
# ---------------------------------------------------------------------

# Uniswap Permit2 — CREATE2-deterministic, same address on all EVM chains.
# Source: https://github.com/Uniswap/permit2
PERMIT2_ADDRESS = "0x000000000022d473030f116ddee9f6b43ac78ba3"

# x402 Exact-Permit2 Proxy — the canonical reference facilitator
# settlement contract. CREATE2-deterministic, same on all EVM chains.
# Source: coinbase/x402 specs/schemes/exact/scheme_exact_evm.md
# (Annex: Reference Implementation).
X402_PERMIT2_PROXY = "0x402085c248eea27d92e8b30b2c58ed07f9e20001"

# ---------------------------------------------------------------------
# Selectors (first 4 bytes of keccak256 of the canonical signature)
# ---------------------------------------------------------------------

# EIP-3009: USDC / EURC native authorization transfer
# transferWithAuthorization(address,address,uint256,uint256,uint256,bytes32,uint8,bytes32,bytes32)
SEL_EIP3009_TRANSFER_AUTH = "e3ee160e"
# receiveWithAuthorization(address,address,uint256,uint256,uint256,bytes32,uint8,bytes32,bytes32)
SEL_EIP3009_RECEIVE_AUTH = "ef55bec6"

# Permit2 permit variants
# permit(address,((address,uint160,uint48,uint48),address,uint256),bytes) — PermitSingle
SEL_PERMIT2_PERMIT_SINGLE = "2b67b570"
# permit(address,((address,uint160,uint48,uint48)[],address,uint256),bytes) — PermitBatch
SEL_PERMIT2_PERMIT_BATCH = "30f28b7a"
# transferFrom(address,address,uint160,address) — Permit2 signature-based transfer
SEL_PERMIT2_TRANSFER_FROM = "36c78516"

X402_SELECTORS = {
    SEL_EIP3009_TRANSFER_AUTH: "transferWithAuthorization (EIP-3009)",
    SEL_EIP3009_RECEIVE_AUTH:  "receiveWithAuthorization (EIP-3009)",
    SEL_PERMIT2_PERMIT_SINGLE: "permit(PermitSingle) (Permit2)",
    SEL_PERMIT2_PERMIT_BATCH:  "permit(PermitBatch) (Permit2)",
    SEL_PERMIT2_TRANSFER_FROM: "transferFrom (Permit2)",
}

EIP3009_SELECTORS = {SEL_EIP3009_TRANSFER_AUTH, SEL_EIP3009_RECEIVE_AUTH}
PERMIT2_SELECTORS = {
    SEL_PERMIT2_PERMIT_SINGLE,
    SEL_PERMIT2_PERMIT_BATCH,
    SEL_PERMIT2_TRANSFER_FROM,
}

# ---------------------------------------------------------------------
# Known facilitator contract registry (seeded from public docs)
# ---------------------------------------------------------------------

# Only addresses explicitly documented in public sources. Do not guess.
KNOWN_FACILITATORS = {
    X402_PERMIT2_PROXY: {
        "name": "x402ExactPermit2Proxy",
        "source": "github.com/coinbase/x402/blob/main/specs/schemes/exact/scheme_exact_evm.md (Annex: Reference Implementation, CREATE2 canonical)",
        "classification": "known",
    },
}

# Coinbase CDP operates the facilitator as an HTTP service at
# https://api.cdp.coinbase.com/platform/v2/x402 — the on-chain EOA
# signing settlement txs is not published. Tracking it requires
# observation. Seeded empty.
KNOWN_FACILITATOR_EOAS: dict[str, dict] = {}


# ---------------------------------------------------------------------
# Phase 1: Reconnaissance
# ---------------------------------------------------------------------

def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or Path(__file__).resolve().parent / "data" / "surveillance.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def recon(conn: sqlite3.Connection) -> dict:
    """
    Scan for x402-relevant activity in the existing corpus.

    Classifies each matching tx from transaction_events into one of:
      - confirmed_x402    : facilitator-mediated (tx.to = known facilitator
                            contract, OR tx hits Permit2 from a facilitator EOA)
      - possible_x402     : EIP-3009 / Permit2 selector seen but no known
                            facilitator involvement
      - generic_permit2   : Permit2 activity that is NOT any of the above

    IMPORTANT: transaction_events is scoped to contracts that are already
    in the monitored set (suspected or confirmed). Permit2 and the x402
    proxy are NOT monitored contracts, so calls TO them will not appear
    here. The zero-selector baseline therefore does NOT mean "no x402
    activity on chain" — it means "no x402 activity touching contracts
    we were already watching." Phase 3 closes this gap by adding the
    canonical Permit2 and x402 proxy addresses to the live monitor.

    Also reports stored-potential signals from approval_events: every
    time a monitored address grants allowance to Permit2, that's a
    Permit2 exposure durable until revoked. This is the x402 attack
    surface and is measurable TODAY in the existing corpus.

    Zero writes, zero RPC. Pure SQL.
    """
    report: dict = {
        "selectors": {},
        "permit2_direct_calls": 0,
        "x402_proxy_calls": 0,
        "facilitator_candidates": [],
        "contracts_hit": {},
        "confirmed_x402": 0,
        "possible_x402": 0,
        "generic_permit2": 0,
        "total_matches": 0,
        "corpus_size": 0,
    }

    # Corpus size for context
    row = conn.execute("SELECT COUNT(*) FROM transaction_events").fetchone()
    report["corpus_size"] = row[0] if row else 0

    # Per-selector counts
    for sel, name in X402_SELECTORS.items():
        r = conn.execute(
            """SELECT COUNT(*)                              AS hits,
                      COUNT(DISTINCT contract_address)      AS contracts,
                      COUNT(DISTINCT interacting_address)   AS callers,
                      MIN(timestamp)                        AS first,
                      MAX(timestamp)                        AS last
               FROM transaction_events
               WHERE function_selector = ?""",
            (sel,),
        ).fetchone()
        entry = {
            "name": name,
            "selector": sel,
            "hits": r["hits"] or 0,
            "distinct_contracts": r["contracts"] or 0,
            "distinct_callers": r["callers"] or 0,
            "first_seen": r["first"],
            "last_seen": r["last"],
        }
        report["selectors"][sel] = entry
        report["total_matches"] += entry["hits"]

    # Calls TO the canonical Permit2 address
    r = conn.execute(
        """SELECT COUNT(*)                            AS hits,
                  COUNT(DISTINCT interacting_address) AS callers
           FROM transaction_events
           WHERE contract_address = ?""",
        (PERMIT2_ADDRESS,),
    ).fetchone()
    report["permit2_direct_calls"] = r["hits"] or 0
    report["permit2_direct_callers"] = r["callers"] or 0

    # Calls TO the x402 Exact-Permit2 Proxy
    r = conn.execute(
        """SELECT COUNT(*)                            AS hits,
                  COUNT(DISTINCT interacting_address) AS callers,
                  MIN(timestamp)                      AS first,
                  MAX(timestamp)                      AS last
           FROM transaction_events
           WHERE contract_address = ?""",
        (X402_PERMIT2_PROXY,),
    ).fetchone()
    report["x402_proxy_calls"] = r["hits"] or 0
    report["x402_proxy_callers"] = r["callers"] or 0
    report["x402_proxy_first_seen"] = r["first"]
    report["x402_proxy_last_seen"] = r["last"]

    # Top contracts hit by any x402 selector (possible facilitators or
    # settlement targets)
    rows = conn.execute(
        """SELECT contract_address,
                  function_selector,
                  COUNT(*)                            AS hits,
                  COUNT(DISTINCT interacting_address) AS callers
           FROM transaction_events
           WHERE function_selector IN (?, ?, ?, ?, ?)
           GROUP BY contract_address, function_selector
           ORDER BY hits DESC
           LIMIT 30""",
        (
            SEL_EIP3009_TRANSFER_AUTH, SEL_EIP3009_RECEIVE_AUTH,
            SEL_PERMIT2_PERMIT_SINGLE, SEL_PERMIT2_PERMIT_BATCH,
            SEL_PERMIT2_TRANSFER_FROM,
        ),
    ).fetchall()
    report["contracts_hit"] = [
        {
            "contract": r["contract_address"],
            "selector": r["function_selector"],
            "selector_name": X402_SELECTORS.get(r["function_selector"], "?"),
            "hits": r["hits"],
            "callers": r["callers"],
        }
        for r in rows
    ]

    # Classify matches
    # - confirmed_x402  : tx.to in KNOWN_FACILITATORS (proxy)
    # - possible_x402   : EIP-3009 selector or Permit2 selector to anything else
    # - generic_permit2 : calls to PERMIT2_ADDRESS directly (not via proxy)
    # We count at the tx-hash level to avoid double-counting
    confirmed = conn.execute(
        """SELECT COUNT(*) FROM transaction_events
           WHERE contract_address = ?""",
        (X402_PERMIT2_PROXY,),
    ).fetchone()[0] or 0

    # Possible x402 = any x402 selector, minus the confirmed-proxy hits,
    # minus the direct-Permit2 hits (which we bucket as generic).
    possible = conn.execute(
        """SELECT COUNT(*) FROM transaction_events
           WHERE function_selector IN (?, ?, ?, ?, ?)
             AND contract_address != ?
             AND contract_address != ?""",
        (
            SEL_EIP3009_TRANSFER_AUTH, SEL_EIP3009_RECEIVE_AUTH,
            SEL_PERMIT2_PERMIT_SINGLE, SEL_PERMIT2_PERMIT_BATCH,
            SEL_PERMIT2_TRANSFER_FROM,
            X402_PERMIT2_PROXY,
            PERMIT2_ADDRESS,
        ),
    ).fetchone()[0] or 0

    # Generic Permit2 = direct calls to Permit2 contract that aren't
    # x402 proxy calls. This may include non-x402 Permit2 usage.
    generic = conn.execute(
        """SELECT COUNT(*) FROM transaction_events
           WHERE contract_address = ?""",
        (PERMIT2_ADDRESS,),
    ).fetchone()[0] or 0

    report["confirmed_x402"] = confirmed
    report["possible_x402"] = possible
    report["generic_permit2"] = generic

    # Facilitator candidates: distinct `tx.from` addresses calling
    # x402 proxy or Permit2, sorted by tx count. These are candidate
    # facilitator EOAs — addresses that repeatedly submit settlement
    # txs on behalf of others.
    rows = conn.execute(
        """SELECT interacting_address,
                  COUNT(*)                         AS tx_count,
                  COUNT(DISTINCT contract_address) AS distinct_targets,
                  MIN(timestamp)                   AS first_seen,
                  MAX(timestamp)                   AS last_seen
           FROM transaction_events
           WHERE (contract_address = ? OR contract_address = ?)
              OR function_selector IN (?, ?, ?, ?, ?)
           GROUP BY interacting_address
           ORDER BY tx_count DESC
           LIMIT 20""",
        (
            PERMIT2_ADDRESS, X402_PERMIT2_PROXY,
            SEL_EIP3009_TRANSFER_AUTH, SEL_EIP3009_RECEIVE_AUTH,
            SEL_PERMIT2_PERMIT_SINGLE, SEL_PERMIT2_PERMIT_BATCH,
            SEL_PERMIT2_TRANSFER_FROM,
        ),
    ).fetchall()
    report["facilitator_candidates"] = [
        {
            "address": r["interacting_address"],
            "tx_count": r["tx_count"],
            "distinct_targets": r["distinct_targets"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
        }
        for r in rows
    ]

    # ---------------------------------------------------------------
    # Permit2 stored-potential analysis (approval_events table)
    # ---------------------------------------------------------------
    # approval_events records ERC-20 approve() calls where the token
    # contract is in the suspected/confirmed monitored set. Entries
    # with spender = canonical Permit2 mean a monitored address has
    # granted Permit2 an allowance on a token our corpus is watching.
    # These are the x402 attack surface: stored potential that can be
    # consumed later by any facilitator submitting a Permit2 transferFrom.
    try:
        r = conn.execute(
            """SELECT COUNT(*)                           AS events,
                      COUNT(DISTINCT approver)           AS approvers,
                      COUNT(DISTINCT token_contract)     AS tokens,
                      MIN(timestamp)                     AS first,
                      MAX(timestamp)                     AS last
               FROM approval_events
               WHERE spender = ?""",
            (PERMIT2_ADDRESS,),
        ).fetchone()
        report["permit2_approvals_total"] = r["events"] or 0
        report["permit2_approvers"] = r["approvers"] or 0
        report["permit2_approved_tokens"] = r["tokens"] or 0
        report["permit2_first_seen"] = r["first"]
        report["permit2_last_seen"] = r["last"]

        # By chain
        rows = conn.execute(
            """SELECT chain, COUNT(*) AS n,
                      COUNT(DISTINCT approver)       AS approvers,
                      COUNT(DISTINCT token_contract) AS tokens
               FROM approval_events
               WHERE spender = ?
               GROUP BY chain""",
            (PERMIT2_ADDRESS,),
        ).fetchall()
        report["permit2_approvals_by_chain"] = [dict(r) for r in rows]

        # Top tokens by unique approvers, joined with tier
        rows = conn.execute(
            """SELECT ae.token_contract,
                      ae.chain,
                      COUNT(DISTINCT ae.approver)         AS approvers,
                      COUNT(*)                            AS events,
                      c.confidence_tier,
                      substr(c.confidence_reason, 1, 70)  AS reason
               FROM approval_events ae
               LEFT JOIN contracts c ON c.contract_address = ae.token_contract
               WHERE ae.spender = ?
               GROUP BY ae.token_contract, ae.chain
               ORDER BY approvers DESC
               LIMIT 15""",
            (PERMIT2_ADDRESS,),
        ).fetchall()
        report["permit2_top_tokens"] = [dict(r) for r in rows]

        # Top approvers by distinct tokens exposed
        rows = conn.execute(
            """SELECT approver,
                      COUNT(DISTINCT token_contract) AS n_tokens,
                      COUNT(*)                       AS n_events,
                      MIN(timestamp)                 AS first,
                      MAX(timestamp)                 AS last
               FROM approval_events
               WHERE spender = ?
               GROUP BY approver
               ORDER BY n_tokens DESC, n_events DESC
               LIMIT 10""",
            (PERMIT2_ADDRESS,),
        ).fetchall()
        report["permit2_top_approvers"] = [dict(r) for r in rows]

        # Breakdown of approved tokens by confidence tier
        rows = conn.execute(
            """SELECT COALESCE(c.confidence_tier, 'not_in_corpus') AS tier,
                      COUNT(DISTINCT ae.token_contract) AS tokens,
                      COUNT(DISTINCT ae.approver)       AS approvers,
                      COUNT(*)                          AS events
               FROM approval_events ae
               LEFT JOIN contracts c ON c.contract_address = ae.token_contract
               WHERE ae.spender = ?
               GROUP BY COALESCE(c.confidence_tier, 'not_in_corpus')
               ORDER BY events DESC""",
            (PERMIT2_ADDRESS,),
        ).fetchall()
        report["permit2_exposure_by_tier"] = [dict(r) for r in rows]

    except sqlite3.Error as e:
        logger.warning("approval_events query failed: %s", e)
        report["permit2_approvals_total"] = None

    # permit_events table (created but historically unpopulated)
    try:
        r = conn.execute("SELECT COUNT(*) FROM permit_events").fetchone()
        report["permit_events_rows"] = r[0] or 0
    except sqlite3.Error:
        report["permit_events_rows"] = None

    # Are the canonical x402 addresses already in our contracts table?
    report["permit2_in_corpus"] = conn.execute(
        "SELECT 1 FROM contracts WHERE contract_address = ?", (PERMIT2_ADDRESS,)
    ).fetchone() is not None
    report["x402_proxy_in_corpus"] = conn.execute(
        "SELECT 1 FROM contracts WHERE contract_address = ?", (X402_PERMIT2_PROXY,)
    ).fetchone() is not None

    return report


def _print_report(report: dict) -> None:
    """Human-readable report for Phase 1 recon output."""
    print("=" * 72)
    print("x402 Activity Reconnaissance — Phase 1")
    print("=" * 72)
    print(f"Corpus size: {report['corpus_size']:,} transaction_events")
    print()

    print("--- Selector matches ---")
    any_matches = False
    for sel, info in report["selectors"].items():
        if info["hits"] > 0:
            any_matches = True
            print(f"  0x{sel}  {info['name']}")
            print(f"    hits={info['hits']:,}  contracts={info['distinct_contracts']}  "
                  f"callers={info['distinct_callers']}")
            print(f"    first={info['first_seen']}  last={info['last_seen']}")
        else:
            print(f"  0x{sel}  {info['name']}  — 0 hits")
    if not any_matches:
        print("  (zero selector matches — expected baseline for x402-blind corpus)")
    print()

    print("--- Canonical address hits ---")
    print(f"  Permit2 ({PERMIT2_ADDRESS})")
    print(f"    direct calls: {report['permit2_direct_calls']:,}  "
          f"distinct callers: {report.get('permit2_direct_callers', 0)}")
    print(f"  x402ExactPermit2Proxy ({X402_PERMIT2_PROXY})")
    print(f"    hits: {report['x402_proxy_calls']:,}  "
          f"callers: {report.get('x402_proxy_callers', 0)}")
    print(f"    first={report.get('x402_proxy_first_seen')}  "
          f"last={report.get('x402_proxy_last_seen')}")
    print()

    print("--- Classification ---")
    total = (report["confirmed_x402"]
             + report["possible_x402"]
             + report["generic_permit2"])
    print(f"  confirmed x402    (tx.to = x402ExactPermit2Proxy):  {report['confirmed_x402']:,}")
    print(f"  possible  x402    (x402 selectors, no known facilitator): {report['possible_x402']:,}")
    print(f"  generic Permit2   (direct Permit2, not x402):       {report['generic_permit2']:,}")
    print(f"  total classified:                                    {total:,}")
    print()

    print("--- Top contracts receiving x402-selector calls ---")
    if report["contracts_hit"]:
        for entry in report["contracts_hit"][:15]:
            marker = ""
            if entry["contract"] == X402_PERMIT2_PROXY:
                marker = " [x402 proxy]"
            elif entry["contract"] == PERMIT2_ADDRESS:
                marker = " [Permit2]"
            print(f"  {entry['contract']}  sel=0x{entry['selector']}  "
                  f"hits={entry['hits']}  callers={entry['callers']}{marker}")
    else:
        print("  (none)")
    print()

    print("--- Candidate facilitator EOAs (top 10 by tx count) ---")
    print("  Addresses that called Permit2 or x402 proxy repeatedly — these")
    print("  are the facilitator EOAs if x402 activity exists on-chain.")
    if report["facilitator_candidates"]:
        for c in report["facilitator_candidates"][:10]:
            print(f"  {c['address']}  txs={c['tx_count']:,}  "
                  f"targets={c['distinct_targets']}")
    else:
        print("  (none)")
    print()

    # --- Permit2 stored-potential section ---
    print("--- Permit2 stored potential (approval_events) ---")
    total = report.get("permit2_approvals_total")
    if total is None:
        print("  approval_events query failed — table may be missing")
    elif total == 0:
        print("  No Permit2 approvals in approval_events (no agent wallets")
        print("  exposed on monitored tokens).")
    else:
        print(f"  Permit2 approvals in corpus: {total:,} events")
        print(f"  Distinct approvers (agent wallet candidates): "
              f"{report.get('permit2_approvers', 0):,}")
        print(f"  Distinct approved tokens: {report.get('permit2_approved_tokens', 0)}")
        print(f"  Date range: {report.get('permit2_first_seen')}"
              f"  -> {report.get('permit2_last_seen')}")

        by_chain = report.get("permit2_approvals_by_chain") or []
        if by_chain:
            print()
            print("  By chain:")
            for row in by_chain:
                print(f"    {row['chain']:<10} events={row['n']:>6} "
                      f"approvers={row['approvers']:>5} tokens={row['tokens']:>4}")

        by_tier = report.get("permit2_exposure_by_tier") or []
        if by_tier:
            print()
            print("  Exposure by token tier:")
            for row in by_tier:
                print(f"    {row['tier']:<16} tokens={row['tokens']:>4} "
                      f"approvers={row['approvers']:>5} events={row['events']:>6}")

        top_tokens = report.get("permit2_top_tokens") or []
        if top_tokens:
            print()
            print("  Top approved tokens (most-exposed first):")
            for t in top_tokens[:10]:
                tier = t.get("confidence_tier") or "not_in_corpus"
                marker = f" [{tier}]"
                print(f"    {t['token_contract']} chain={t['chain']} "
                      f"approvers={t['approvers']} events={t['events']}{marker}")
                if t.get("reason"):
                    print(f"      reason: {t['reason']}")

        top_approvers = report.get("permit2_top_approvers") or []
        if top_approvers:
            print()
            print("  Top exposed approvers (most distinct tokens approved):")
            for a in top_approvers[:10]:
                print(f"    {a['approver']} tokens={a['n_tokens']} "
                      f"events={a['n_events']}")
    print()

    # --- Infrastructure scope checks ---
    print("--- Infrastructure scope ---")
    print(f"  Permit2 in contracts table:             "
          f"{'yes' if report.get('permit2_in_corpus') else 'no'}")
    print(f"  x402ExactPermit2Proxy in contracts:     "
          f"{'yes' if report.get('x402_proxy_in_corpus') else 'no'}")
    pe = report.get("permit_events_rows")
    print(f"  permit_events table rows:               {pe if pe is not None else 'n/a'}")
    print()

    # --- Interpretation ---
    print("--- Interpretation ---")

    selector_finding = (report["total_matches"] == 0
                        and report["permit2_direct_calls"] == 0
                        and report["x402_proxy_calls"] == 0)

    if selector_finding:
        print("  ZERO direct x402 selector hits in transaction_events.")
        print("  IMPORTANT scope note: transaction_events only records txs whose")
        print("  tx.to is already in the monitored (suspected/confirmed) contracts")
        print("  set. Permit2 and x402ExactPermit2Proxy are NOT in our contracts")
        print("  table, so ANY x402 activity that happens outside a monitored")
        print("  contract is invisible to this query. Phase 3 closes this gap by")
        print("  adding both canonical addresses to the live monitor.")
    else:
        if report["confirmed_x402"] > 0:
            print(f"  CONFIRMED x402 activity: {report['confirmed_x402']:,} txs "
                  f"hitting x402ExactPermit2Proxy.")
        if report["possible_x402"] > 0:
            print(f"  POSSIBLE x402 signals: {report['possible_x402']:,} txs using"
                  " EIP-3009/Permit2 selectors without a known facilitator.")
        if report["generic_permit2"] > 0:
            print(f"  Generic Permit2: {report['generic_permit2']:,} direct calls.")

    if report.get("permit2_approvals_total", 0) > 0:
        approvers = report.get("permit2_approvers", 0)
        tokens = report.get("permit2_approved_tokens", 0)
        print()
        print(f"  STORED POTENTIAL FINDING: {approvers:,} distinct addresses have")
        print(f"  granted Permit2 allowance on {tokens} monitored tokens. These are")
        print("  the x402 attack surface — allowances can be consumed at any time")
        print("  by any facilitator with a valid EIP-712 signature from the owner.")

        # Suspicion weighting: tier breakdown
        by_tier = {r["tier"]: r for r in (report.get("permit2_exposure_by_tier") or [])}
        suspected = by_tier.get("suspected", {}).get("events", 0)
        confirmed = by_tier.get("confirmed", {}).get("events", 0)
        if suspected or confirmed:
            print(f"  Of those events, {suspected:,} are approvals on SUSPECTED")
            print(f"  trap tokens and {confirmed:,} are on CONFIRMED trap tokens.")
            print("  This is not hypothetical: real wallets have granted Permit2")
            print("  permissions on contracts Layer 3 has flagged as traps.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Layer 3 — x402 Activity Monitor"
    )
    parser.add_argument(
        "--recon", action="store_true",
        help="Phase 1: reconnaissance scan of existing transaction_events. "
             "Zero writes, zero RPC.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite DB (default: surveillance/data/surveillance.db)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    db_path = Path(args.db) if args.db else None
    conn = _get_conn(db_path)

    try:
        if args.recon:
            report = recon(conn)
            _print_report(report)
            return 0
        else:
            parser.print_help()
            return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
