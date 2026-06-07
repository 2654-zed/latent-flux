"""
Layer 3 — Approval Drain Monitor

Tracks the deferred exploitation pattern:
1. Victim calls approve() on a trap contract (looks clean, 0% revert)
2. Later, the operator calls transferFrom() to drain the approved tokens
3. The drain may come from a different address than the deployer

This monitor:
- Tracks all approve() calls on suspected/self-test contracts
- Builds a watchlist of (victim, approved_contract) pairs
- Monitors for transferFrom() calls that match pending approvals
- Alerts when drains begin (the approval trap fires)

Runs as a periodic scan in the heartbeat loop (no API calls).

Usage:
    python -m surveillance.approval_drain_monitor --scan
    python -m surveillance.approval_drain_monitor --watchlist
"""

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

logger = logging.getLogger("surveillance.approval_drain")

# Blockscout free REST endpoints — 0 Alchemy CU. Used by the authoritative
# victim-outbound-leg drain detector (check_drains_blockscout). The tx_events
# join (check_drains) is structurally blind to most drains because drain
# transferFrom() calls target the TOKEN contract, which is usually not in the
# watched set and so never enters transaction_events. Grounded 2026-06-05:
# of 54,996 pending approvals only 2 had a tx_events match.
BLOCKSCOUT_BASE = {
    "base": "https://base.blockscout.com/api/v2",
    "arbitrum": "https://arbitrum.blockscout.com/api/v2",
    "optimism": "https://optimism.blockscout.com/api/v2",
}
_BS_MAX_PAGES = 10
_BS_SLEEP = 0.12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _oli_suppressed_deployers(conn: sqlite3.Connection) -> set:
    # Contract deployers publicly attributed (Open Labels Initiative tags via
    # Blockscout metadata) as known-legitimate institutions or projects.
    # Drain detector cannot distinguish their legitimate batch/distribution
    # flows from extraction by shape alone, so drain_detected promotion is
    # gated. Mirrors the entity_classifier OLI redirect landed in
    # Correction #20.
    #
    # `self-confirming` is excluded because that severity tier means the OLI
    # tag agrees with our adversarial classification (scam, phishing, drain
    # hub). Those drains are legitimate detections — keep them.
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT address FROM oli_labels "
                "WHERE severity IN ('HIGH', 'LOW')"
            )
        }
    except sqlite3.OperationalError:
        # oli_labels table not yet migrated in this DB
        return set()


def ensure_tables(conn: sqlite3.Connection):
    """Create approval monitoring tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            victim_address TEXT NOT NULL,
            contract_address TEXT NOT NULL,
            approve_tx_hash TEXT,
            approve_timestamp TEXT,
            approve_block INTEGER,
            contract_tier TEXT,
            is_self_test_trap INTEGER DEFAULT 0,
            deployer_address TEXT,
            drain_detected INTEGER DEFAULT 0,
            drain_tx_hash TEXT,
            drain_timestamp TEXT,
            drain_caller TEXT,
            logged_at TEXT,
            UNIQUE(victim_address, contract_address)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approval_victim ON approval_watchlist(victim_address)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approval_contract ON approval_watchlist(contract_address)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approval_drain ON approval_watchlist(drain_detected)
    """)
    conn.commit()


def scan_approvals(conn: sqlite3.Connection) -> dict:
    """
    Scan for approve() calls on suspected contracts.
    Build/update the approval watchlist.
    """
    ensure_tables(conn)
    now = _now()

    # Find approve() calls on suspected/confirmed contracts not yet in watchlist
    new_approvals = conn.execute("""
        SELECT te.interacting_address as victim,
               te.contract_address,
               te.tx_hash,
               te.timestamp,
               te.block_number,
               c.confidence_tier,
               c.deployer_address,
               CASE WHEN st.contract_address IS NOT NULL THEN 1 ELSE 0 END as is_self_test
        FROM transaction_events te
        JOIN contracts c ON c.contract_address = te.contract_address
        LEFT JOIN self_test_traps st ON st.contract_address = te.contract_address
        LEFT JOIN approval_watchlist aw ON aw.victim_address = te.interacting_address
            AND aw.contract_address = te.contract_address
        WHERE te.function_selector = '095ea7b3'
        AND c.confidence_tier IN ('suspected', 'confirmed')
        AND te.interacting_address != c.deployer_address
        AND aw.id IS NULL
    """).fetchall()

    added = 0
    for a in new_approvals:
        # Tuple-index by position rather than dict-subscript. Defensive: it
        # works whether or not the caller set a row_factory. Every PRODUCTION
        # caller does set sqlite3.Row (QueueConnection._ro in the heartbeat
        # path, db.init_db in standalone, the CLI below), so dict access would
        # also work there — but a bare `sqlite3.connect()` (tests, ad-hoc
        # scripts) returns plain tuples on which dict access raises TypeError.
        # Position-indexing removes that foot-gun. Column order is fixed by the
        # SELECT above:
        #   victim, contract_address, tx_hash, timestamp, block_number,
        #   confidence_tier, deployer_address, is_self_test
        (victim, contract_address, tx_hash, timestamp, block_number,
         confidence_tier, deployer_address, is_self_test) = a
        try:
            conn.execute("""
                INSERT OR IGNORE INTO approval_watchlist
                (victim_address, contract_address, approve_tx_hash, approve_timestamp,
                 approve_block, contract_tier, is_self_test_trap, deployer_address, logged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (victim, contract_address, tx_hash,
                  timestamp, block_number, confidence_tier,
                  is_self_test, deployer_address, now))
            added += 1
        except Exception:
            pass

    conn.commit()
    return {"new_approvals_tracked": added}


def check_drains(conn: sqlite3.Connection) -> dict:
    """
    Check if any watched approvals have been drained.

    Looks for:
    1. transferFrom() calls on watched contracts
    2. Any token transfer FROM a victim TO the deployer or unknown collector
    3. Contract interactions by the deployer AFTER approvals came in

    OLI gate: skip rows whose contract was deployed by an OLI-tagged
    institutional/project address. Mirrors the entity_classifier redirect
    introduced in Correction #20. See `_oli_suppressed_deployers`.
    """
    ensure_tables(conn)
    now = _now()
    drains_found = 0
    suppressed = _oli_suppressed_deployers(conn)
    skipped = 0

    # Method 1: transferFrom() on watched contracts.
    #
    # NOTE: this is the LEGACY fast path, retained for reference and manual
    # CLI use. It is NO LONGER called from the heartbeat loop — the
    # authoritative detector is check_drains_blockscout() below, which is not
    # blind to the common case. Method 1 can only ever see a drain whose
    # transferFrom targets a WATCHED contract; almost all real drains target a
    # token contract outside the watched set, so they never enter
    # transaction_events. Grounded 2026-06-05: of 54,996 pending approvals,
    # exactly 2 had a Method-1 match. Method 1 also lacks per-victim
    # from-matching (Bug #19b over-credit risk), which is the other reason the
    # precise Blockscout per-victim test supersedes it.
    #
    # Rows are tuple-indexed by position (defensive — works with or without a
    # row_factory). Column order is fixed by the SELECTs below.
    pending = conn.execute("""
        SELECT aw.victim_address, aw.contract_address, aw.deployer_address,
               aw.approve_timestamp
        FROM approval_watchlist aw
        WHERE aw.drain_detected = 0
    """).fetchall()

    for p in pending:
        victim_address, contract_address, deployer_address, approve_timestamp = p
        if deployer_address and deployer_address in suppressed:
            skipped += 1
            continue
        # Check for transferFrom on this contract after the approval.
        # IMPORTANT (Correction #24 / Bug #19, 2026-05-21): filter to
        # successful (non-reverted) transactions. Reverted transferFroms
        # MUST NOT be credited as drains — they moved zero tokens.
        # Before this filter was added, the OFC token (0x752c5a95) produced
        # 4,587 phantom drain rows from 3 failed transferFrom transactions.
        drain = conn.execute("""
            SELECT te.tx_hash, te.timestamp, te.interacting_address as caller
            FROM transaction_events te
            WHERE te.contract_address = ?
            AND te.function_selector = '23b872dd'
            AND te.is_reverted = 0
            AND te.timestamp > ?
            LIMIT 1
        """, (contract_address, approve_timestamp)).fetchone()

        if drain:
            drain_tx_hash, drain_timestamp, drain_caller = drain
            conn.execute("""
                UPDATE approval_watchlist
                SET drain_detected = 1, drain_tx_hash = ?, drain_timestamp = ?,
                    drain_caller = ?
                WHERE victim_address = ? AND contract_address = ?
            """, (drain_tx_hash, drain_timestamp, drain_caller,
                  victim_address, contract_address))
            drains_found += 1

    # Method 2: DISABLED 2026-05-27 (Correction #27 / Bug #19b).
    # ------------------------------------------------------------------
    # This was the primary drain OVER-CREDIT source. Its UPDATE clause
    #   WHERE contract_address = ? AND drain_detected = 0
    # stamped drain_detected=1 onto EVERY pending approver of a contract
    # whenever the deployer made ANY non-approve/non-transfer call, with
    # no check that a specific victim's tokens actually moved. The Phase 0
    # is_reverted filter (Correction #24) reduced but did not fix this:
    # a single SUCCESSFUL non-transferFrom call still credited all
    # approvers. Dark-window audit attributed the largest phantom drain
    # rows to this method (e.g. a single call crediting 1,520 "victims").
    #
    # A custom-drain detector cannot be made precise without decoding the
    # tx's ERC-20 Transfer logs and crediting only addresses whose balance
    # moved. Re-introduce later as a weak `drain_suspected` signal gated by
    # a Transfer-log verification pass before ever setting drain_detected=1.
    #
    # Per CLAUDE.md "conservative over aggressive": disabling produces
    # false negatives (missed custom drains) but removes false positives
    # (phantom victim credits) — the correct trade for a credibility metric.
    #
    # Method 1 (standard transferFrom) remains active. It also lacks
    # per-victim from-matching, but its UPDATE is scoped to a single
    # victim_address, bounding the over-credit. From-matching for Method 1
    # is a tracked resume-action.
    # ------------------------------------------------------------------
    _ = suppressed  # still used by Method 1 above; Method 2 body removed

    conn.commit()
    return {"drains_detected": drains_found, "oli_suppressed_skips": skipped}


def _blockscout_outbound(base: str, victim: str, contract: str,
                         sleep: float = _BS_SLEEP):
    """Victim-outbound-leg test via Blockscout (0 Alchemy CU).

    Returns (n_out, n_in, last_out_tx, last_out_to, last_out_ts, err).

    A (victim, contract) approval row is a REAL drain iff the victim has >=1
    ERC-20 Transfer of the contract token with from==victim in their Blockscout
    address token-transfer history (n_out > 0). Inbound-only (n_in>0, n_out==0)
    is distribution/airdrop, NOT a drain (the FIRE/OFC lesson — never credit a
    drain on n_in alone).

    last_out_* describe the most-recent outbound leg (Blockscout returns items
    newest-first), used to populate drain_tx_hash / drain_timestamp /
    drain_caller. They are best-effort: None if n_out==0, and may be None for
    cache-hit drains (the cache stores only the n_out verdict, not the tx).

    err is None on success; a short string ("http500", "neterr", ...) on fetch
    failure — the caller leaves the row pending and retries on the next cycle.

    Mirrors scripts/t1_apply.py::victim_has_outbound (validated on 5,174
    victim/contract pairs, 0 errors) and extends it to surface the latest leg.
    Token key is item.token.address_hash (NOT .address). Pages via
    next_page_params, capped at _BS_MAX_PAGES (matches the proven primitive).
    """
    url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}"
    n_out = n_in = pages = 0
    last_tx = last_to = last_ts = None
    vlow, clow = victim.lower(), contract.lower()
    while url and pages < _BS_MAX_PAGES:
        req = Request(url, headers={"Accept": "application/json",
                                    "User-Agent": "Mozilla/5.0 (L3-drain)"})
        try:
            with urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode())
        except HTTPError as e:
            if e.code == 404:  # address never transacted this token => no drain
                return n_out, n_in, last_tx, last_to, last_ts, None
            return n_out, n_in, last_tx, last_to, last_ts, f"http{e.code}"
        except (URLError, TimeoutError, OSError):
            return n_out, n_in, last_tx, last_to, last_ts, "neterr"
        for it in d.get("items", []):
            tok = ((it.get("token") or {}).get("address_hash") or "").lower()
            if tok and tok != clow:
                continue
            frm = ((it.get("from") or {}).get("hash") or "").lower()
            to = ((it.get("to") or {}).get("hash") or "").lower()
            if frm == vlow:
                n_out += 1
                if last_tx is None:  # newest-first => first seen is most recent
                    last_tx = it.get("tx_hash") or it.get("transaction_hash")
                    last_to = to or None
                    last_ts = it.get("timestamp")
            elif to == vlow:
                n_in += 1
        npp = d.get("next_page_params")
        if not npp:
            return n_out, n_in, last_tx, last_to, last_ts, None
        url = (f"{base}/addresses/{victim}/token-transfers"
               f"?type=ERC-20&token={contract}&{urlencode(npp)}")
        pages += 1
        time.sleep(sleep)
    return n_out, n_in, last_tx, last_to, last_ts, None


# Recognized victim-initiated swap/router methods. A token-transfer leg whose
# top-level tx method matches one of these is a VICTIM SWAP (validated 2026-06-06:
# every swap-method leg sampled had tx.from == victim), so it is skipped without a
# tx fetch. Every other leg (transferFrom / transfer / custom selector / unknown)
# is tx-checked definitively. Substring match, lowercased.
_SWAP_METHOD_HINTS = ("swap", "exactinput", "exactoutput", "multicall",
                      "execute", "transferandmulticall", "fillorder", "settle",
                      "unoswap", "clipperswap")


def _tx_initiator(base: str, tx_hash: str):
    """Return the lowercased tx sender (msg.sender / tx.from), or None on error."""
    try:
        req = Request(f"{base}/transactions/{tx_hash}",
                      headers={"Accept": "application/json",
                               "User-Agent": "Mozilla/5.0 (L3-drain)"})
        with urlopen(req, timeout=25) as r:
            t = json.loads(r.read().decode())
        return ((t.get("from") or {}).get("hash", "") or "").lower() or None
    except Exception:
        return None


def _blockscout_drain_check(base: str, victim: str, contract: str,
                            sleep: float = _BS_SLEEP, max_legs: int = 10):
    """tx-INITIATOR drain test (Correction #29). 0 Alchemy CU.

    A (victim, contract) is a REAL approval-drain iff the victim has an outbound
    ERC-20 Transfer of the contract token whose TRANSACTION was initiated by a
    third party (tx.from != victim) -- a drainer's transferFrom. A victim-initiated
    swap/transfer (tx.from == victim) is a SALE, not a drain. n_out>0 ALONE is NOT
    a drain (it counts every seller -- the retired method's ~98% FP bug); this
    initiator gate is what it lacked.

    Returns (verdict, drainer, collector, drain_tx, n_out, err):
      verdict in {'DRAIN','SALE','NONE','ERR'}.
      DRAIN -> drainer=tx.from (the actual drainer), collector=leg.to, drain_tx=hash.

    Fast-path: legs whose top-level tx method is a recognized victim swap are
    skipped without a tx fetch. Non-swap legs are tx-checked definitively.
    """
    vlow, clow = victim.lower(), contract.lower()
    url = f"{base}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}"
    legs = []  # (tx_hash, to, method) newest-first
    pages = 0
    while url and pages < _BS_MAX_PAGES and len(legs) < max_legs:
        req = Request(url, headers={"Accept": "application/json",
                                    "User-Agent": "Mozilla/5.0 (L3-drain)"})
        try:
            with urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode())
        except HTTPError as e:
            if e.code == 404:
                return ("NONE", None, None, None, 0, None)
            return ("ERR", None, None, None, len(legs), f"http{e.code}")
        except (URLError, TimeoutError, OSError):
            return ("ERR", None, None, None, len(legs), "neterr")
        for it in d.get("items", []):
            if ((it.get("from") or {}).get("hash", "") or "").lower() != vlow:
                continue
            tok = ((it.get("token") or {}).get("address_hash") or "").lower()
            if tok and tok != clow:
                continue
            legs.append((it.get("transaction_hash") or it.get("tx_hash"),
                         ((it.get("to") or {}).get("hash", "") or ""),
                         (it.get("method") or "")))
        npp = d.get("next_page_params")
        if not npp:
            break
        url = (f"{base}/addresses/{victim}/token-transfers"
               f"?type=ERC-20&token={contract}&{urlencode(npp)}")
        pages += 1
        time.sleep(sleep)
    if not legs:
        return ("NONE", None, None, None, 0, None)
    n_out = len(legs)
    nonswap_checked = errs = 0
    for txh, to, method in legs:
        m = (method or "").lower()
        if any(h in m for h in _SWAP_METHOD_HINTS):
            continue  # victim-initiated swap leg -> not a drain
        if not txh:
            continue
        ini = _tx_initiator(base, txh)
        time.sleep(sleep)
        if ini is None:
            errs += 1
            continue
        nonswap_checked += 1
        if ini != vlow:
            return ("DRAIN", ini, (to.lower() if to else None), txh, n_out, None)
    if nonswap_checked == 0 and errs > 0:
        return ("ERR", None, None, None, n_out, "tx_fetch_failed")
    return ("SALE", None, None, None, n_out, None)


def check_drains_blockscout(conn, max_victims=400, sleep: float = _BS_SLEEP,
                            db_path: str = None) -> dict:
    """tx-INITIATOR-gated approval-drain detection (Correction #29). 0 Alchemy CU.

    Supersedes the retired n_out>0 victim-outbound-leg method, which conflated
    drains with legitimate DEX sales (~98% false positives — see Correction #29).
    A victim is drained iff they have an outbound leg of the contract token whose
    tx was initiated by a third party (tx.from != victim); a victim-initiated
    swap/transfer is a SALE, not a drain. See _blockscout_drain_check.

    Caches:
      * audit_drain_legs (n_out) is reused as a FREE pre-filter — n_out==0 means
        no outbound legs => NONE without any fetch.
      * drain_initiator_verdicts persists the tx-initiator verdict so re-runs are
        incremental (a victim is decided once).

    Threading / budget / OLI gate unchanged from the prior version (RO reads via a
    thread-local connection; writes via conn; OLI suppression retained; at most
    max_victims NEW drain-checks per call, None = unbounded backfill). NOTE: the
    heartbeat wiring is currently PAUSED (Correction #29) — this runs via the CLI
    --drain-scan-all until the operator re-enables a live detector.

    Returns {drains_detected, checked, sales, none, cache_hits, errors,
             oli_suppressed_skips, budget_exhausted}.
    """
    path = db_path or getattr(conn, "_db_path", None) or str(DB_PATH)

    conn.execute("""CREATE TABLE IF NOT EXISTS audit_drain_legs(
        victim TEXT, contract TEXT, n_out INTEGER, n_in INTEGER,
        truncated INTEGER, err TEXT, checked_at TEXT,
        PRIMARY KEY (victim, contract))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS drain_initiator_verdicts(
        victim TEXT, contract TEXT, verdict TEXT, drainer TEXT, collector TEXT,
        drain_tx TEXT, checked_at TEXT, PRIMARY KEY (victim, contract))""")
    conn.commit()

    if not hasattr(conn, "_queue"):
        try:
            conn.execute("PRAGMA busy_timeout=30000")
        except Exception:
            pass

    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    try:
        ro.execute("PRAGMA busy_timeout=30000")
        suppressed = _oli_suppressed_deployers(ro)
        nout_cache = {}
        try:
            for v, c, n_out, err in ro.execute(
                    "SELECT victim, contract, n_out, err FROM audit_drain_legs"):
                nout_cache[(v, c)] = (n_out, err)
        except sqlite3.OperationalError:
            pass
        verdict_cache = {}
        try:
            for v, c, verdict, drainer, collector, drain_tx in ro.execute(
                    "SELECT victim, contract, verdict, drainer, collector, drain_tx "
                    "FROM drain_initiator_verdicts"):
                verdict_cache[(v, c)] = (verdict, drainer, collector, drain_tx)
        except sqlite3.OperationalError:
            pass
        pending = ro.execute("""
            SELECT aw.victim_address, aw.contract_address, aw.deployer_address,
                   COALESCE(c.chain, 'base')
            FROM approval_watchlist aw
            LEFT JOIN contracts c ON c.contract_address = aw.contract_address
            WHERE aw.drain_detected = 0
        """).fetchall()
    finally:
        ro.close()

    drains = checked = cache_hits = errors = skipped = fetched = 0
    sales = none = writes = 0
    budget_exhausted = False
    now = _now()

    for victim, contract, deployer, chain in pending:
        if deployer and deployer in suppressed:
            skipped += 1
            continue
        key = (victim, contract)
        vc = verdict_cache.get(key)
        if vc is not None and vc[0] in ("DRAIN", "SALE", "NONE"):
            verdict, drainer, collector, drain_tx = vc
            cache_hits += 1
        else:
            nc = nout_cache.get(key)
            if nc is not None and nc[1] is None and (nc[0] or 0) == 0:
                # free pre-filter: no outbound legs => cannot be a drain
                verdict, drainer, collector, drain_tx = "NONE", None, None, None
                conn.execute("INSERT OR REPLACE INTO drain_initiator_verdicts VALUES (?,?,?,?,?,?,?)",
                             (victim, contract, verdict, None, None, None, now))
                writes += 1
            else:
                if max_victims is not None and fetched >= max_victims:
                    budget_exhausted = True
                    break
                base = BLOCKSCOUT_BASE.get(chain, BLOCKSCOUT_BASE["base"])
                verdict, drainer, collector, drain_tx, _n_out, err = _blockscout_drain_check(
                    base, victim, contract, sleep=sleep)
                fetched += 1
                if verdict == "ERR":
                    errors += 1
                    if fetched % 100 == 0:
                        conn.commit()
                    continue  # leave pending; retry next run
                conn.execute("INSERT OR REPLACE INTO drain_initiator_verdicts VALUES (?,?,?,?,?,?,?)",
                             (victim, contract, verdict, drainer, collector, drain_tx, now))
                writes += 1
                time.sleep(sleep)
        if verdict == "DRAIN":
            conn.execute("""
                UPDATE approval_watchlist
                SET drain_detected = 1, drain_tx_hash = ?, drain_timestamp = NULL,
                    drain_caller = ?
                WHERE victim_address = ? AND contract_address = ?
                  AND drain_detected = 0
            """, (drain_tx, drainer, victim, contract))
            drains += 1
            writes += 1
        elif verdict == "SALE":
            sales += 1
        else:
            none += 1
        checked += 1
        if writes >= 200:
            conn.commit()
            writes = 0

    conn.commit()
    return {
        "drains_detected": drains,
        "checked": checked,
        "sales": sales,
        "none": none,
        "cache_hits": cache_hits,
        "errors": errors,
        "oli_suppressed_skips": skipped,
        "budget_exhausted": budget_exhausted,
    }


def backfill_oli_suppression(conn: sqlite3.Connection) -> dict:
    """Reset drain_detected on rows whose contract was deployed by an
    OLI-tagged institutional/project address.

    These rows promoted before the OLI gate was wired in (the gap surfaced
    on 2026-05-10 — Animoca-deployed `0x752c5a95...` produced 4,587 phantom
    drain rows from two callers). The contracts themselves likely remain
    on the suspected/confirmed lists (separate bytecode-classifier signal);
    only the drain promotion is rolled back.

    The fix is non-destructive in that approval rows are retained — only
    `drain_detected`, `drain_tx_hash`, `drain_timestamp`, `drain_caller` are
    cleared. The bytecode classifier and confidence_tier fields are untouched.
    """
    ensure_tables(conn)
    suppressed = _oli_suppressed_deployers(conn)
    if not suppressed:
        return {"reset": 0, "suppressed_deployers": 0}

    placeholders = ",".join("?" * len(suppressed))
    args = list(suppressed)
    affected = conn.execute(
        f"""
        SELECT COUNT(*) FROM approval_watchlist
        WHERE drain_detected = 1 AND deployer_address IN ({placeholders})
        """,
        args,
    ).fetchone()[0]
    conn.execute(
        f"""
        UPDATE approval_watchlist
        SET drain_detected = 0,
            drain_tx_hash = NULL,
            drain_timestamp = NULL,
            drain_caller = NULL
        WHERE drain_detected = 1 AND deployer_address IN ({placeholders})
        """,
        args,
    )
    conn.commit()
    return {"reset": affected, "suppressed_deployers": len(suppressed)}


def get_summary(conn: sqlite3.Connection) -> dict:
    """Get current approval watchlist statistics."""
    ensure_tables(conn)

    total = conn.execute("SELECT COUNT(*) FROM approval_watchlist").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=0").fetchone()[0]
    drained = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
    self_test = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE is_self_test_trap=1").fetchone()[0]

    unique_victims = conn.execute("SELECT COUNT(DISTINCT victim_address) FROM approval_watchlist").fetchone()[0]
    unique_contracts = conn.execute("SELECT COUNT(DISTINCT contract_address) FROM approval_watchlist").fetchone()[0]

    return {
        "total_tracked": total,
        "pending_drain": pending,
        "drain_detected": drained,
        "on_self_test_traps": self_test,
        "unique_victims": unique_victims,
        "unique_contracts": unique_contracts,
    }


def print_watchlist(conn: sqlite3.Connection):
    """Print the current watchlist status."""
    ensure_tables(conn)
    summary = get_summary(conn)

    print(f"[approval_drain] Watchlist: {summary['total_tracked']} tracked | {summary['pending_drain']} pending | {summary['drain_detected']} drained")
    print(f"  Unique victims: {summary['unique_victims']} | Contracts: {summary['unique_contracts']} | On self-test traps: {summary['on_self_test_traps']}")

    # Top contracts by pending approvals
    print()
    print("Top contracts by pending approvals:")
    for r in conn.execute("""
        SELECT contract_address, deployer_address, COUNT(*) as pending,
            is_self_test_trap,
            MIN(approve_timestamp) as first_approve,
            MAX(approve_timestamp) as last_approve
        FROM approval_watchlist WHERE drain_detected=0
        GROUP BY contract_address ORDER BY pending DESC LIMIT 15
    """):
        st = " [SELF-TEST]" if r["is_self_test_trap"] else ""
        print(f"  {r['contract_address'][:18]}... | {r['pending']:>4} pending | deployer={r['deployer_address'][:12]}... | {r['first_approve'][:10]} to {r['last_approve'][:10]}{st}")

    # Any drains detected?
    drained = conn.execute("""
        SELECT contract_address, drain_caller, drain_timestamp, COUNT(*) as victims_drained
        FROM approval_watchlist WHERE drain_detected=1
        GROUP BY contract_address ORDER BY victims_drained DESC LIMIT 10
    """).fetchall()
    if drained:
        print()
        print("DRAINS DETECTED:")
        for r in drained:
            print(f"  {r['contract_address'][:18]}... | {r['victims_drained']} victims drained | caller={r['drain_caller'][:14]}... | {r['drain_timestamp'][:16]}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approval drain monitor")
    parser.add_argument("--scan", action="store_true", help="Scan for new approvals + check drains")
    parser.add_argument("--watchlist", action="store_true", help="Show current watchlist")
    parser.add_argument(
        "--backfill-oli-suppression",
        action="store_true",
        help="Reset drain_detected on rows whose contract deployer is OLI-tagged",
    )
    parser.add_argument(
        "--drain-scan-all",
        action="store_true",
        help="Out-of-band Blockscout victim-leg drain backfill over ALL pending "
             "approvals (0 Alchemy CU). Unbounded; clears the backlog faster "
             "than the per-heartbeat cap. Safe to re-run (cache-incremental).",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    if args.scan:
        r1 = scan_approvals(conn)
        print(f"[approval_drain] New approvals tracked: {r1['new_approvals_tracked']}")
        r2 = check_drains(conn)
        print(
            f"[approval_drain] Drains detected: {r2['drains_detected']} "
            f"| OLI-suppressed skips: {r2.get('oli_suppressed_skips', 0)}"
        )
        print()
        print_watchlist(conn)
    elif args.watchlist:
        print_watchlist(conn)
    elif args.backfill_oli_suppression:
        r = backfill_oli_suppression(conn)
        print(
            f"[approval_drain] OLI backfill: reset {r['reset']} drain_detected rows "
            f"across {r['suppressed_deployers']} suppressed deployers"
        )
    elif args.drain_scan_all:
        logging.basicConfig(level=logging.INFO)
        print("[approval_drain] Blockscout drain backfill (0 Alchemy CU) — "
              "this can take a while over the full pending backlog...")
        r = check_drains_blockscout(conn, max_victims=None)
        print(
            f"[approval_drain] tx-initiator drain scan complete: "
            f"DRAINS={r['drains_detected']} sales={r.get('sales',0)} "
            f"none={r.get('none',0)} checked={r['checked']} "
            f"cache_hits={r['cache_hits']} errors={r['errors']} "
            f"oli_skips={r['oli_suppressed_skips']} "
            f"budget_exhausted={r['budget_exhausted']}"
        )
    else:
        parser.print_help()

    conn.close()
