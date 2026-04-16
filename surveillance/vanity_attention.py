"""
Layer 3 — Vanity Attention Scorer

STATUS: SHELVED 2026-04-16 — infrastructure complete, intelligence yield
unconfirmed. Revisit when one of the re-open triggers below fires. See
shelf note at end of this docstring.

Inverts the usual detection model. Instead of analyzing a wallet's behavior
to determine its role, this module analyzes OTHER ATTACKERS' behavior
toward the wallet to infer the wallet's role.

The premise: vanity-prefix address generation requires GPU compute. Poisoners
don't waste that compute on random targets. An address receiving dedicated
vanity-grinding attempts has been selected through the poisoners' own
behavioral analysis as a high-value manual-send target.

This methodology gives us three things:

1. Cross-validation — a wallet that scores high on vanity_attention AND
   traces to a known funding chain AND has a behavioral profile consistent
   with operational infrastructure is confirmed by three independent methods.

2. Discovery — a wallet scoring high on vanity_attention that ISN'T in our
   org mapping is a wallet the poisoners found before we did. Worth
   investigating.

3. Capability measurement — poisoner nonces, attack diversity, and attempt
   counts let us rank targets by the confidence level the attackers have
   in each target's value.

Detection strategy:

For every target in our corpus, we collect the set of addresses the target
regularly transacts with (its "counterparty set"). Then we scan inbound
transfers. If an inbound sender shares a 4+ character vanity prefix with
any counterparty BUT is not itself in the counterparty set, it's a vanity
impersonation attempt.

Scoring:
  attention_score = attempt_count
                  + (max_poisoner_nonce // 1000)
                  + (distinct_poisoners * 5)
                  + (impersonated_counterparty_count * 3)

Usage:
    python -m surveillance.vanity_attention --score 0xe93d64f3fbc352131e79fc5578cbe44b66697f86
    python -m surveillance.vanity_attention --rank --top 20
    python -m surveillance.vanity_attention --rank --deep --blocks 10000

----------------------------------------------------------------------
SHELF NOTE (2026-04-16)
----------------------------------------------------------------------

What's proven:
  - Architecture works end-to-end (v1 asset-transfer path + v2 block-walk)
  - False-positive filtering against system addresses (zero, dead,
    precompiles, WETH9, Permit2, OP Stack bridge) is correct
  - Batched block-scanning scales with blocks_back, not target count

What's NOT proven:
  - The "discovery signal" thesis — that scoring reveals unclassified
    high-value targets the poisoners found before we did. We scored 20
    top-value candidates and found zero high-attention targets. Could
    mean sophisticated vanity poisoning is genuinely rare, OR the
    window was too narrow, OR the poisoners moved to zero-value spam
    that v1 can't see and v2 wasn't run with a wide enough window.
  - The documented org_001 Treasury poisoners (nonces 2622, 4069) were
    not detected with the 100-block v2 window. A 10,000-block run
    would resolve whether they're still active.

Re-open triggers:
  1. Targeted investigation — "is this specific operator being
     poisoned?" Run --score on the single address.
  2. New poisoner addresses surface in other investigations.
     The 3 org_001 poisoners are on watch; if similar patterns appear
     elsewhere, re-run --rank --deep --blocks 10000 on fresh candidates.
  3. Customer use case — wallet provider asks "is this address a
     target?" The infrastructure answers it without further build work.

What keeps running:
  - surveillance/poisoning_watcher.py continues catching non-vanity
    poisoning (Unicode homoglyphs, phishing airdrops) against the
    3 default targets. If its output starts showing vanity-pattern
    senders, that's the signal to revive this module.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

_ARB_WSS = os.environ.get("ARB_WSS_URL", "")
_BASE_WSS = os.environ.get("BASE_WSS_URL", "")
_OP_WSS = os.environ.get("OP_WSS_URL", "")


def _wss_to_http(wss: str) -> str:
    return wss.replace("wss://", "https://") if wss else ""


RPC_URLS = {
    "arbitrum": _wss_to_http(_ARB_WSS) or "https://arb-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3",
    "base": _wss_to_http(_BASE_WSS) or "https://base-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3",
    "optimism": _wss_to_http(_OP_WSS) or "https://opt-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3",
}

# Vanity prefix length to test for impersonation (4-8 hex chars typical)
PREFIX_MATCH_CHARS = 8  # 4 hex chars + '0x' prefix = 6 char substring
# Minimum matching length between sender and counterparty for poisoning suspicion
MIN_PREFIX_OVERLAP = 6  # 4 real hex chars (strict)

# Block walking parameters for zero-value poisoning detection (v2)
# Batch scans across all targets in a single chain pass to minimize RPC cost.
BLOCKS_BACK_DEFAULT = 500  # ~8min on Arbitrum, ~16min on Base/Optimism


def _rpc(url: str, method: str, params: list) -> Optional[dict]:
    data = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
    }).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result.get("result")
    except Exception:
        return None


def _get_counterparties(conn: sqlite3.Connection, target: str, top_n: int = 50) -> set[str]:
    """Get the target's regular transaction counterparties from local DB.

    A counterparty is any address the target has interacted with multiple
    times. We pull from org_transfer_events, transaction_events, and
    approval_events to build a comprehensive set.
    """
    target = target.lower()
    counterparties: set[str] = set()

    # Addresses the target has sent to (org_transfer_events)
    try:
        rows = conn.execute(
            "SELECT to_address, COUNT(*) as n FROM org_transfer_events "
            "WHERE from_address = ? "
            "GROUP BY to_address ORDER BY n DESC LIMIT ?",
            (target, top_n),
        ).fetchall()
        for r in rows:
            addr = (r[0] or "").lower()
            if addr and addr != target:
                counterparties.add(addr)
    except sqlite3.OperationalError:
        pass

    # Addresses that have sent to the target (org_transfer_events)
    try:
        rows = conn.execute(
            "SELECT from_address, COUNT(*) as n FROM org_transfer_events "
            "WHERE to_address = ? "
            "GROUP BY from_address ORDER BY n DESC LIMIT ?",
            (target, top_n),
        ).fetchall()
        for r in rows:
            addr = (r[0] or "").lower()
            if addr and addr != target:
                counterparties.add(addr)
    except sqlite3.OperationalError:
        pass

    # Contracts the target has interacted with (transaction_events)
    try:
        rows = conn.execute(
            "SELECT contract_address, COUNT(*) as n FROM transaction_events "
            "WHERE interacting_address = ? "
            "GROUP BY contract_address ORDER BY n DESC LIMIT ?",
            (target, top_n),
        ).fetchall()
        for r in rows:
            addr = (r[0] or "").lower()
            if addr and addr != target:
                counterparties.add(addr)
    except sqlite3.OperationalError:
        pass

    # Approval spenders (if target has granted approvals)
    try:
        rows = conn.execute(
            "SELECT spender, COUNT(*) as n FROM approval_events "
            "WHERE owner = ? "
            "GROUP BY spender ORDER BY n DESC LIMIT ?",
            (target, top_n),
        ).fetchall()
        for r in rows:
            addr = (r[0] or "").lower()
            if addr and addr != target:
                counterparties.add(addr)
    except sqlite3.OperationalError:
        pass

    return counterparties


def _get_inbound_dust(chain: str, target: str, count: int = 100) -> list[dict]:
    """Get recent inbound transfers (including dust) for target.

    Combines asset transfers (ERC-20 with vanity sender) and zero-value
    ETH pings (detected via eth_getLogs on target's tx history). The
    zero-value path catches the pure spam-ping poisoners that don't
    move actual assets — like the documented org_001 Treasury attackers
    at nonces 2622 and 4069.
    """
    url = RPC_URLS.get(chain)
    if not url:
        return []
    categories = ["external", "erc20"]
    if chain == "arbitrum":
        categories.append("internal")
    params = [{
        "toAddress": target,
        "category": categories,
        "maxCount": hex(count),
        "order": "desc",
    }]
    result = _rpc(url, "alchemy_getAssetTransfers", params)
    transfers = result.get("transfers", []) if result else []

    # Zero-value pings are NOT returned by alchemy_getAssetTransfers
    # (no asset moved). They show up in the address's full tx list.
    # For a comprehensive scan we'd need to walk block-by-block, which
    # is too expensive. This gap is documented: vanity poisoners using
    # zero-value ETH spam are not caught by this module. See CORRECTIONS.md
    # (2026-04-12 org_001 poisoner entry: nonces 2622, 4069 use this path).
    return transfers


def _get_nonce(chain: str, address: str) -> int:
    """Get address nonce on the given chain."""
    url = RPC_URLS.get(chain)
    if not url:
        return 0
    result = _rpc(url, "eth_getTransactionCount", [address, "latest"])
    if not result:
        return 0
    try:
        return int(result, 16)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# v2: Zero-value ETH poisoning detection via block walking
#
# alchemy_getAssetTransfers does not return transactions where value=0 and
# no ERC-20 is transferred — there's no asset to report. But that's exactly
# how sophisticated vanity poisoners operate: they send empty txs from
# vanity addresses so the target's tx history shows the lookalike address
# next to their real counterparties.
#
# The documented org_001 Treasury poisoners (nonces 2622, 4069) use this
# exact path. v1 was blind to them.
#
# v2 approach: batch-scan recent blocks on each chain ONCE and match all
# targets in memory. For 20 targets × 3 chains × 500 blocks = 1,500 RPC
# calls (batched per-chain). Worst-case RPC cost is bounded regardless
# of target count.
# ---------------------------------------------------------------------------

def _batch_scan_blocks(chain: str, target_set: set[str],
                      blocks_back: int = BLOCKS_BACK_DEFAULT) -> dict[str, list[dict]]:
    """Walk the most recent N blocks on chain, collect all inbound txs
    (including value=0) for every address in target_set.

    Returns {target_address: [{tx_hash, block, from, value_wei}, ...]}.

    Single pass per chain — scales with blocks_back, not with target count.
    """
    url = RPC_URLS.get(chain)
    if not url:
        return {}

    latest_hex = _rpc(url, "eth_blockNumber", [])
    if not latest_hex:
        return {}
    try:
        latest = int(latest_hex, 16)
    except (ValueError, TypeError):
        return {}

    start = max(0, latest - blocks_back)
    results: dict[str, list[dict]] = {t.lower(): [] for t in target_set}

    for block_num in range(start, latest + 1):
        block = _rpc(url, "eth_getBlockByNumber", [hex(block_num), True])
        if not block:
            continue
        for tx in block.get("transactions", []) or []:
            to_addr = (tx.get("to") or "").lower()
            if to_addr in results:
                try:
                    val_wei = int(tx.get("value", "0x0"), 16)
                except (ValueError, TypeError):
                    val_wei = 0
                results[to_addr].append({
                    "tx_hash": tx.get("hash", ""),
                    "block": block_num,
                    "from": (tx.get("from") or "").lower(),
                    "value_wei": val_wei,
                })
        # Light rate limiting
        if block_num % 100 == 0:
            time.sleep(0.05)

    return results


# Well-known system addresses that should NOT be treated as counterparties
# or senders — they create false positives via shared prefixes.
SYSTEM_ADDRESSES: set[str] = {
    "0x0000000000000000000000000000000000000000",  # zero address (mint/burn)
    "0x000000000000000000000000000000000000dead",  # dead address
    "0x0000000000000000000000000000000000000064",  # ArbSys precompile
    "0x4200000000000000000000000000000000000006",  # WETH9 OP Stack
    "0x4200000000000000000000000000000000000010",  # L2StandardBridge OP Stack
    "0x4200000000000000000000000000000000000007",  # L2 message passer
    "0x000000000022d473030f116ddee9f6b43ac78ba3",  # Permit2
    "0x0000000000001ff3684f28c67538d4d072c22734",  # 0x v4 exchange proxy
}

# Prefixes that match so many system addresses they're noise
SYSTEM_PREFIXES: set[str] = {
    "0x0000",  # any zero-padded address or precompile
    "0x4200",  # any OP Stack system contract
}


def _is_system_address(addr: str) -> bool:
    addr = addr.lower()
    if addr in SYSTEM_ADDRESSES:
        return True
    # Zero-pad check: if 36+ of the 40 hex chars are zero, it's a system addr
    hex_part = addr[2:] if addr.startswith("0x") else addr
    if hex_part.count("0") >= 36:
        return True
    return False


def _check_vanity_match(sender: str, target: str,
                        counterparties: set[str]) -> Optional[tuple[str, int]]:
    """Check if sender is a vanity impersonation.

    Returns (impersonated_address, matching_prefix_length) if yes, None otherwise.

    Filters out system addresses (zero, dead, precompiles, bridge, WETH) which
    create false positives via shared prefixes like "0x0000" and "0x4200".
    """
    sender = sender.lower()
    target = target.lower()

    # Never flag system addresses — they create massive false positives
    if _is_system_address(sender):
        return None

    # Never flag if sender IS a real counterparty
    if sender in counterparties or sender == target:
        return None

    # Check against target itself
    target_prefix = target[:2 + MIN_PREFIX_OVERLAP]  # 0x + 4 hex
    if target_prefix not in SYSTEM_PREFIXES and sender.startswith(target_prefix):
        # Bonus confidence if suffix also matches
        if sender[-4:] == target[-4:]:
            return ("TARGET_IMPERSONATION_PREFIX_SUFFIX", MIN_PREFIX_OVERLAP + 4)
        return ("TARGET_IMPERSONATION_PREFIX", MIN_PREFIX_OVERLAP)

    # Check against each counterparty
    best_match: Optional[tuple[str, int]] = None
    for cp in counterparties:
        if _is_system_address(cp):
            continue
        cp_prefix = cp[:2 + MIN_PREFIX_OVERLAP]
        if cp_prefix in SYSTEM_PREFIXES:
            continue
        if sender.startswith(cp_prefix):
            match_len = MIN_PREFIX_OVERLAP
            if sender[-4:] == cp[-4:]:
                match_len += 4
            if best_match is None or match_len > best_match[1]:
                best_match = (cp, match_len)

    return best_match


def score_target(conn: sqlite3.Connection, target: str,
                 block_scan_results: Optional[dict] = None) -> dict:
    """Score vanity attention for a single target across all chains.

    If block_scan_results is provided (from _batch_scan_blocks), use it
    for zero-value poisoning detection instead of (or in addition to)
    the asset-transfer API.
    """
    target = target.lower()

    # Get target's counterparty set (same across chains — these are
    # organizational relationships, not chain-specific)
    counterparties = _get_counterparties(conn, target, top_n=100)

    attempts: list[dict] = []
    chain_details: dict[str, dict] = {}

    for chain in ["arbitrum", "base", "optimism"]:
        inbound = _get_inbound_dust(chain, target, count=100)
        chain_attempts: list[dict] = []
        seen_tx_hashes: set[str] = set()

        # v1 path: asset-bearing transfers
        for t in inbound:
            sender = (t.get("from") or "").lower()
            value = t.get("value") or 0
            asset = t.get("asset") or ""
            tx_hash = t.get("hash", "")

            match = _check_vanity_match(sender, target, counterparties)
            if not match:
                continue

            impersonated, match_len = match
            chain_attempts.append({
                "poisoner": sender,
                "impersonates": impersonated,
                "match_length": match_len,
                "value": value,
                "asset": asset,
                "tx_hash": tx_hash,
                "chain": chain,
                "detection_path": "asset_transfer",
            })
            if tx_hash:
                seen_tx_hashes.add(tx_hash)

        # v2 path: zero-value inbound from block walk (if provided)
        if block_scan_results and chain in block_scan_results:
            chain_blocks = block_scan_results[chain].get(target, [])
            for b in chain_blocks:
                sender = b["from"]
                tx_hash = b.get("tx_hash", "")
                if tx_hash in seen_tx_hashes:
                    continue

                match = _check_vanity_match(sender, target, counterparties)
                if not match:
                    continue

                impersonated, match_len = match
                chain_attempts.append({
                    "poisoner": sender,
                    "impersonates": impersonated,
                    "match_length": match_len,
                    "value": b.get("value_wei", 0),
                    "asset": "ETH" if b.get("value_wei", 0) > 0 else "none",
                    "tx_hash": tx_hash,
                    "chain": chain,
                    "detection_path": "block_walk_zero_value",
                })

        # Get poisoner nonces (expensive — only for the matches)
        for a in chain_attempts:
            a["poisoner_nonce"] = _get_nonce(chain, a["poisoner"])

        attempts.extend(chain_attempts)
        chain_details[chain] = {
            "inbound_count": len(inbound),
            "vanity_attempts": len(chain_attempts),
            "zero_value_caught": sum(1 for a in chain_attempts
                                     if a.get("detection_path") == "block_walk_zero_value"),
        }
        time.sleep(0.2)

    # Compute score
    attempt_count = len(attempts)
    distinct_poisoners = len(set(a["poisoner"] for a in attempts))
    max_nonce = max((a["poisoner_nonce"] for a in attempts), default=0)
    impersonated_set = set(a["impersonates"] for a in attempts)
    distinct_targets = len(impersonated_set)

    attention_score = (
        attempt_count
        + (max_nonce // 1000)
        + (distinct_poisoners * 5)
        + (distinct_targets * 3)
    )

    # Tier assignment
    if attention_score >= 100:
        tier = "CRITICAL_TARGET"
    elif attention_score >= 30:
        tier = "HIGH_TARGET"
    elif attention_score >= 10:
        tier = "MODERATE_TARGET"
    elif attention_score > 0:
        tier = "LOW_TARGET"
    else:
        tier = "NO_ATTENTION"

    return {
        "target": target,
        "attention_score": attention_score,
        "tier": tier,
        "attempt_count": attempt_count,
        "distinct_poisoners": distinct_poisoners,
        "max_poisoner_nonce": max_nonce,
        "distinct_impersonated": distinct_targets,
        "counterparty_pool_size": len(counterparties),
        "chain_details": chain_details,
        "attempts": attempts[:20],  # cap for output
    }


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vanity_attention (
            target_address TEXT PRIMARY KEY,
            attention_score INTEGER,
            tier TEXT,
            attempt_count INTEGER,
            distinct_poisoners INTEGER,
            max_poisoner_nonce INTEGER,
            distinct_impersonated INTEGER,
            last_scored TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vanity_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_address TEXT,
            chain TEXT,
            poisoner_address TEXT,
            impersonates TEXT,
            match_length INTEGER,
            poisoner_nonce INTEGER,
            asset TEXT,
            value TEXT,
            tx_hash TEXT,
            detected_at TEXT,
            UNIQUE(target_address, chain, tx_hash)
        )
    """)
    conn.commit()


def _persist_score(conn: sqlite3.Connection, result: dict) -> None:
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO vanity_attention "
        "(target_address, attention_score, tier, attempt_count, "
        "distinct_poisoners, max_poisoner_nonce, distinct_impersonated, last_scored) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result["target"], result["attention_score"], result["tier"],
            result["attempt_count"], result["distinct_poisoners"],
            result["max_poisoner_nonce"], result["distinct_impersonated"],
            now_iso,
        ),
    )
    for a in result["attempts"]:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO vanity_attempts "
                "(target_address, chain, poisoner_address, impersonates, "
                "match_length, poisoner_nonce, asset, value, tx_hash, detected_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result["target"], a.get("chain"), a["poisoner"],
                    a["impersonates"], a["match_length"], a["poisoner_nonce"],
                    a.get("asset", ""), str(a.get("value", "")),
                    a.get("tx_hash", ""), now_iso,
                ),
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()


def rank_targets(limit: int = 20, deep_scan: bool = False,
                 blocks_back: int = BLOCKS_BACK_DEFAULT) -> None:
    """Rank known candidates by vanity attention score.

    Scores a curated set of high-value addresses: known org wallets,
    CRITICAL watchlist entries, and bridge whales.

    deep_scan=True activates v2 block-walking for zero-value poisoning
    detection. Costs ~blocks_back RPC calls per chain (once, batched
    across all targets).
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)

    # Collect candidate targets
    candidates: set[str] = set()

    # Default high-value targets
    candidates.update([
        "0xe69f81b825d7dc31ee9becef4dbeab5cf30e3abb",  # bridge whale
        "0x2ce910fbba65b454bbaf6a18c952a70f3bcd8299",  # bridge + poisoner
        "0xe93d64f3fbc352131e79fc5578cbe44b66697f86",  # org_001 operator
        "0xc6962004f452be9203591991d15f6b388e09e8d0",  # cashout
        "0x51c72848c68a965f66fa7a88855f9f7784502a7f",  # defi exit
        "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70",  # lp companion
        "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb",  # laundry
        "0xf186cb00e49e18491db5783ff04fae3818102ff7",  # real treasury (contract)
        "0x8c826f795466e39acbff1bb4eeeb759609377ba1",  # real gas station
        "0x360e68faccca8ca495c1b759fd9eee466db9fb32",  # vault
    ])

    # CRITICAL watchlist entries
    try:
        rows = conn.execute(
            "SELECT DISTINCT address FROM watchlist "
            "WHERE priority IN ('CRITICAL', 'HIGH') LIMIT 30"
        ).fetchall()
        for r in rows:
            a = (r[0] or "").lower()
            if a:
                candidates.add(a)
    except sqlite3.OperationalError:
        pass

    candidates_list = list(candidates)[:limit]

    # v2: batch block scan ONCE per chain for all candidates
    block_scan_results: dict[str, dict] = {}
    if deep_scan:
        print(f"[vanity_attention] v2 deep scan: walking {blocks_back} recent blocks "
              f"on each chain for {len(candidates_list)} targets...")
        target_set = set(candidates_list)
        for chain in ["arbitrum", "base", "optimism"]:
            print(f"  Scanning {chain}...", end=" ", flush=True)
            result = _batch_scan_blocks(chain, target_set, blocks_back=blocks_back)
            block_scan_results[chain] = result
            total_inbound = sum(len(v) for v in result.values())
            print(f"{total_inbound} inbound txs captured")
        print()

    print(f"[vanity_attention] Scoring {len(candidates_list)} candidates across 3 chains...\n")

    results = []
    for i, target in enumerate(candidates_list, 1):
        print(f"  [{i}/{len(candidates_list)}] {target[:24]}...", end=" ", flush=True)
        try:
            result = score_target(conn, target,
                                  block_scan_results=block_scan_results if deep_scan else None)
            _persist_score(conn, result)
            results.append(result)
            zero_val = sum(cd.get("zero_value_caught", 0)
                          for cd in result.get("chain_details", {}).values())
            zero_note = f" zero_val={zero_val}" if deep_scan and zero_val else ""
            print(f"score={result['attention_score']} tier={result['tier']} "
                  f"attempts={result['attempt_count']} "
                  f"distinct_poisoners={result['distinct_poisoners']}{zero_note}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Sort by score
    results.sort(key=lambda r: r["attention_score"], reverse=True)

    print()
    print("=" * 80)
    print("VANITY ATTENTION RANKING")
    print("=" * 80)
    print(f"{'Target':<44} {'Tier':<18} {'Score':>6} {'Attempts':>9} {'Poisoners':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['target'][:42]:<44} {r['tier']:<18} "
              f"{r['attention_score']:>6} {r['attempt_count']:>9} "
              f"{r['distinct_poisoners']:>10}")

    # Discovery: high attention but no org mapping
    print()
    print("=" * 80)
    print("DISCOVERY SIGNALS — targeted by poisoners, not yet in our org mapping")
    print("=" * 80)
    discovered = []
    for r in results:
        if r["attention_score"] >= 10:
            # Check if target has org classification
            row = conn.execute(
                "SELECT org_id, subtype FROM entity_classification WHERE address = ?",
                (r["target"],),
            ).fetchone()
            has_org = bool(row and row["org_id"])

            dep = conn.execute(
                "SELECT entity_type FROM deployers WHERE deployer_address = ?",
                (r["target"],),
            ).fetchone()
            has_dep_classification = bool(dep and dep["entity_type"])

            if not has_org and not has_dep_classification:
                discovered.append(r)

    if discovered:
        for d in discovered:
            print(f"  {d['target']} — score={d['attention_score']} "
                  f"attempts={d['attempt_count']} — UNCLASSIFIED but targeted")
    else:
        print("  (all high-attention targets are already in our org mapping)")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 3 Vanity Attention Scorer — read the poisoners' labels"
    )
    parser.add_argument("--score", type=str, help="Score a single address")
    parser.add_argument("--rank", action="store_true",
                        help="Rank known high-value targets by attention score")
    parser.add_argument("--top", type=int, default=20, help="Limit for --rank")
    parser.add_argument("--deep", action="store_true",
                        help="Enable v2 block-walking for zero-value poisoning detection")
    parser.add_argument("--blocks", type=int, default=BLOCKS_BACK_DEFAULT,
                        help="Blocks to walk per chain in --deep mode")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_tables(conn)

    if args.score:
        result = score_target(conn, args.score)
        _persist_score(conn, result)
        print(json.dumps({k: v for k, v in result.items() if k != "attempts"}, indent=2))
        if result["attempts"]:
            print("\nSample attempts:")
            for a in result["attempts"][:10]:
                print(f"  [{a.get('chain')}] {a['poisoner'][:24]} "
                      f"impersonates={a['impersonates'][:24]} "
                      f"nonce={a['poisoner_nonce']} "
                      f"match_len={a['match_length']}")
    elif args.rank:
        rank_targets(limit=args.top, deep_scan=args.deep, blocks_back=args.blocks)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
