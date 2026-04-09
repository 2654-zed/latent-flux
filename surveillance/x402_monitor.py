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
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger("surveillance.x402_monitor")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

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
#
# Two registries: the CREATE2 settlement proxy (same address on all
# chains) and the public facilitator EOA list from facilitators.x402.watch
# (community-maintained directory of self-registered operators).
KNOWN_FACILITATORS = {
    X402_PERMIT2_PROXY: {
        "name": "x402ExactPermit2Proxy",
        "source": "github.com/coinbase/x402/blob/main/specs/schemes/exact/scheme_exact_evm.md (Annex: Reference Implementation, CREATE2 canonical)",
        "classification": "known",
    },
}

# Facilitator operator EOAs by chain, sourced 2026-04-09 from
# https://facilitators.x402.watch/. Key is (address_lower, chain).
# Operator is the public brand; use classification='known' when seeding.
X402_WATCH_REGISTRY: dict[tuple[str, str], str] = {
    # --- Coinbase (Base) ---
    ("0xdbdf3d8ed80f84c35d01c6c9f9271761bad90ba6", "base"): "Coinbase",
    ("0x9aae2b0d1b9dc55ac9bab9556f9a26cb64995fb9", "base"): "Coinbase",
    ("0x3a70788150c7645a21b95b7062ab1784d3cc2104", "base"): "Coinbase",
    ("0x708e57b6650a9a741ab39cae1969ea1d2d10eca1", "base"): "Coinbase",
    ("0xce82eeec8e98e443ec34fda3c3e999cbe4cb6ac2", "base"): "Coinbase",
    ("0x7f6d822467df2a85f792d4508c5722ade96be056", "base"): "Coinbase",
    ("0x001ddabba5782ee48842318bd9ff4008647c8d9c", "base"): "Coinbase",
    ("0x9c09faa49c4235a09677159ff14f17498ac48738", "base"): "Coinbase",
    ("0xcbb10c30a9a72fae9232f41cbbd566a097b4e03a", "base"): "Coinbase",
    ("0x9fb2714af0a84816f5c6322884f2907e33946b88", "base"): "Coinbase",
    # --- Questflow (Base) ---
    ("0x724efafb051f17ae824afcdf3c0368ae312da264", "base"): "Questflow",
    ("0xa9a54ef09fc8b86bc747cec6ef8d6e81c38c6180", "base"): "Questflow",
    ("0x4638bc811c93bf5e60deed32325e93505f681576", "base"): "Questflow",
    ("0xd7d91a42dfadd906c5b9ccde7226d28251e4cd0f", "base"): "Questflow",
    ("0x4544b535938b67d2a410a98a7e3b0f8f68921ca7", "base"): "Questflow",
    ("0x59e8014a3b884392fbb679fe461da07b18c1ff81", "base"): "Questflow",
    ("0xe6123e6b389751c5f7e9349f3d626b105c1fe618", "base"): "Questflow",
    ("0xf70e7cb30b132fab2a0a5e80d41861aa133ea21b", "base"): "Questflow",
    ("0x90da501fdbec74bb0549100967eb221fed79c99b", "base"): "Questflow",
    ("0xce7819f0b0b871733c933d1f486533bab95ec47b", "base"): "Questflow",
    # --- Heurist (Base) ---
    ("0xb578b7db22581507d62bdbeb85e06acd1be09e11", "base"): "Heurist",
    ("0x021cc47adeca6673def958e324ca38023b80a5be", "base"): "Heurist",
    ("0x3f61093f61817b29d9556d3b092e67746af8cdfd", "base"): "Heurist",
    ("0x290d8b8edcafb25042725cb9e78bcac36b8865f8", "base"): "Heurist",
    ("0x612d72dc8402bba997c61aa82ce718ea23b2df5d", "base"): "Heurist",
    ("0x1fc230ee3c13d0d520d49360a967dbd1555c8326", "base"): "Heurist",
    ("0x48ab4b0af4ddc2f666a3fcc43666c793889787a3", "base"): "Heurist",
    ("0xd97c12726dcf994797c981d31cfb243d231189fb", "base"): "Heurist",
    ("0x90d5e567017f6c696f1916f4365dd79985fce50f", "base"): "Heurist",
    # --- X402rs (Base) ---
    ("0xd8dfc729cbd05381647eb5540d756f4f8ad63eec", "base"): "X402rs",
    ("0x76eee8f0acabd6b49f1cc4e9656a0c8892f3332e", "base"): "X402rs",
    ("0x97d38aa5de015245dcca76305b53abe6da25f6a5", "base"): "X402rs",
    ("0x0168f80e035ea68b191faf9bfc12778c87d92008", "base"): "X402rs",
    ("0x5e437bee4321db862ac57085ea5eb97199c0ccc5", "base"): "X402rs",
    ("0xc19829b32324f116ee7f80d193f99e445968499a", "base"): "X402rs",
    # --- PayAI (Base) ---
    ("0xc6699d2aada6c36dfea5c248dd70f9cb0235cb63", "base"): "PayAI",
    ("0xb2bd29925cbbcea7628279c91945ca5b98bf371b", "base"): "PayAI",
    ("0x25659315106580ce2a787ceec5efb2d347b539c9", "base"): "PayAI",
    ("0xb8f41cb13b1f213da1e94e1b742ec1323235c48f", "base"): "PayAI",
    ("0xe575fa51af90957d66fab6d63355f1ed021b887b", "base"): "PayAI",
    # --- CodeNut (Base) ---
    ("0x8d8fa42584a727488eeb0e29405ad794a105bb9b", "base"): "CodeNut",
    ("0x87af99356d774312b73018b3b6562e1ae0e018c9", "base"): "CodeNut",
    ("0x65058cf664d0d07f68b663b0d4b4f12a5e331a38", "base"): "CodeNut",
    ("0x88e13d4c764a6c840ce722a0a3765f55a85b327e", "base"): "CodeNut",
    # --- AurraCloud (Base) ---
    ("0x222c4367a2950f3b53af260e111fc3060b0983ff", "base"): "AurraCloud",
    ("0xb70c4fe126de09bd292fe3d1e40c6d264ca6a52a", "base"): "AurraCloud",
    ("0xd348e724e0ef36291a28dfeccf692399b0e179f8", "base"): "AurraCloud",
    # --- OpenX402 (Base) ---
    ("0x97316fa4730bc7d3b295234f8e4d04a0a4c093e8", "base"): "OpenX402",
    ("0x97db9b5291a218fc77198c285cefdc943ef74917", "base"): "OpenX402",
    # --- Single-address operators (Base) ---
    ("0x742d35cc6634c0532925a3b844bc9e7595f0bee4", "base"): "KAMIYO",
    ("0x80c08de1a05df2bd633cf520754e40fde3c794d3", "base"): "Thirdweb",
    ("0x279e08f711182c79ba6d09669127a426228a4653", "base"): "Daydreams",
    ("0x103040545ac5031a11e8c03dd11324c7333a13c7", "base"): "Ultravioleta DAO",
    ("0xfe0920a0a7f0f8a1ec689146c30c3bbef439bf8a", "base"): "Mogami",
    ("0x73b2b8df52fbe7c40fe78db52e3dffdd5db5ad07", "base"): "402104",
    ("0x3be45f576696a2fd5a93c1330cd19f1607ab311d", "base"): "xEcho",
    ("0x80735b3f7808e2e229ace880dbe85e80115631ca", "base"): "Virtuals Protocol",
}
X402_WATCH_SOURCE = "facilitators.x402.watch (community registry, pulled 2026-04-09)"

# Coinbase CDP operates the facilitator as an HTTP service at
# https://api.cdp.coinbase.com/platform/v2/x402 — the on-chain EOA
# signing settlement txs is not published. Tracking it requires
# observation. Seeded empty.
KNOWN_FACILITATOR_EOAS: dict[str, dict] = {}


# ---------------------------------------------------------------------
# Phase 2: Schema + classification seeding
# ---------------------------------------------------------------------

# Chains the monitor is active on. x402 proxy is CREATE2-deterministic
# so the same address appears on every EVM chain.
SEEDED_CHAINS = ("base", "arbitrum", "optimism")


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create x402 tables if missing. Idempotent — safe to call on every
    monitor startup.

    All three tables are owned by this module. We do not extend db.py.
    Entity classification taxonomy values (x402_facilitator, x402_agent,
    x402_resource_server) are data-only additions to the existing
    entity_classification table — no schema change required.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS x402_events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            chain               TEXT    NOT NULL,
            tx_hash             TEXT    NOT NULL,
            block_number        INTEGER NOT NULL,
            timestamp           TEXT    NOT NULL,
            facilitator_address TEXT,
            payer_address       TEXT,
            payee_address       TEXT,
            token_contract      TEXT,
            token_symbol        TEXT,
            amount              REAL,
            x402_type           TEXT CHECK (x402_type IN ('eip3009', 'permit2')),
            confidence          TEXT CHECK (confidence IN ('confirmed', 'suspected')),
            selector            TEXT,
            created_at          TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_x402_events_facilitator
            ON x402_events(facilitator_address);
        CREATE INDEX IF NOT EXISTS idx_x402_events_payer
            ON x402_events(payer_address);
        CREATE INDEX IF NOT EXISTS idx_x402_events_chain_ts
            ON x402_events(chain, timestamp);
        CREATE INDEX IF NOT EXISTS idx_x402_events_tx_hash
            ON x402_events(tx_hash);

        CREATE TABLE IF NOT EXISTS x402_facilitators (
            address         TEXT    NOT NULL,
            chain           TEXT    NOT NULL,
            name            TEXT,
            classification  TEXT CHECK (classification IN ('known', 'unknown', 'rogue')),
            source          TEXT,
            first_seen      TEXT,
            last_seen       TEXT,
            tx_count        INTEGER NOT NULL DEFAULT 0,
            total_volume    REAL    NOT NULL DEFAULT 0,
            created_at      TEXT    NOT NULL,
            PRIMARY KEY (address, chain)
        );
        CREATE INDEX IF NOT EXISTS idx_x402_facilitators_classification
            ON x402_facilitators(classification);

        CREATE TABLE IF NOT EXISTS x402_permit2_exposure (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_address       TEXT    NOT NULL,
            spender_address     TEXT    NOT NULL,
            token_contract      TEXT    NOT NULL,
            chain               TEXT    NOT NULL,
            allowance_amount    TEXT,
            expiration          INTEGER,
            first_seen          TEXT    NOT NULL,
            last_seen           TEXT    NOT NULL,
            created_at          TEXT    NOT NULL,
            UNIQUE(owner_address, spender_address, token_contract, chain)
        );
        CREATE INDEX IF NOT EXISTS idx_x402_exposure_owner
            ON x402_permit2_exposure(owner_address);
        CREATE INDEX IF NOT EXISTS idx_x402_exposure_spender
            ON x402_permit2_exposure(spender_address);
        CREATE INDEX IF NOT EXISTS idx_x402_exposure_token
            ON x402_permit2_exposure(token_contract);
    """)
    conn.commit()


def _seed_facilitators(conn: sqlite3.Connection) -> int:
    """Seed the x402_facilitators registry.

    Two sources:
      1. KNOWN_FACILITATORS (CREATE2 proxy, same on every chain)
      2. X402_WATCH_REGISTRY (chain-specific EOAs from the public
         facilitators.x402.watch directory)

    For addresses already present as 'unknown' (auto-registered from
    observation), upgrade them to 'known' when they match a registry
    entry. This converts prior unclassified sightings into attributed
    facilitator rows without losing their tx_count or first/last_seen
    observation data.

    Returns number of rows inserted or upgraded.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    changed = 0

    # Source 1: CREATE2 proxy on all chains
    for addr, meta in KNOWN_FACILITATORS.items():
        for chain in SEEDED_CHAINS:
            cur = conn.execute(
                """INSERT OR IGNORE INTO x402_facilitators
                   (address, chain, name, classification, source,
                    first_seen, last_seen, tx_count, total_volume, created_at)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, 0, ?)""",
                (addr, chain, meta["name"], meta["classification"],
                 meta["source"], now),
            )
            if cur.rowcount:
                changed += 1

    # Source 2: x402.watch registry (chain-specific EOAs)
    for (addr, chain), operator in X402_WATCH_REGISTRY.items():
        cur = conn.execute(
            """INSERT INTO x402_facilitators
               (address, chain, name, classification, source,
                first_seen, last_seen, tx_count, total_volume, created_at)
               VALUES (?, ?, ?, 'known', ?, NULL, NULL, 0, 0, ?)
               ON CONFLICT(address, chain)
               DO UPDATE SET
                   classification = 'known',
                   name = CASE
                       WHEN x402_facilitators.classification = 'known'
                           THEN x402_facilitators.name
                       ELSE excluded.name
                   END,
                   source = CASE
                       WHEN x402_facilitators.classification = 'known'
                           THEN x402_facilitators.source
                       ELSE excluded.source
                   END""",
            (addr, chain, operator, X402_WATCH_SOURCE, now),
        )
        if cur.rowcount:
            changed += 1
    conn.commit()
    return changed


def _seed_known_selectors(conn: sqlite3.Connection) -> int:
    """Register the 5 x402 selectors in the known_selectors reference
    table. Idempotent via INSERT OR IGNORE. Returns rows inserted.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    entries = [
        (SEL_EIP3009_TRANSFER_AUTH,
         "x402",
         "transferWithAuthorization(address,address,uint256,uint256,uint256,bytes32,uint8,bytes32,bytes32)",
         "EIP-3009 authorization transfer. Used by USDC/EURC for gasless "
         "off-chain-signed payments. Primary settlement path for x402 when "
         "the payment token implements EIP-3009."),
        (SEL_EIP3009_RECEIVE_AUTH,
         "x402",
         "receiveWithAuthorization(address,address,uint256,uint256,uint256,bytes32,uint8,bytes32,bytes32)",
         "EIP-3009 receiver-pulls-authorization variant. Less common than "
         "transferWithAuthorization but valid x402 settlement selector."),
        (SEL_PERMIT2_PERMIT_SINGLE,
         "permit2",
         "permit(address,((address,uint160,uint48,uint48),address,uint256),bytes)",
         "Uniswap Permit2 PermitSingle. Used by x402 for non-EIP-3009 "
         "tokens via the Permit2 universal approval contract. Signed "
         "off-chain by the owner, submitted on-chain by the facilitator."),
        (SEL_PERMIT2_PERMIT_BATCH,
         "permit2",
         "permit(address,((address,uint160,uint48,uint48)[],address,uint256),bytes)",
         "Uniswap Permit2 PermitBatch. Multi-token variant. Same x402 role "
         "as PermitSingle but authorizes several tokens in one call."),
        (SEL_PERMIT2_TRANSFER_FROM,
         "permit2",
         "transferFrom(address,address,uint160,address)",
         "Uniswap Permit2 signature-based transferFrom. This is the "
         "consumption selector — facilitators call this to actually move "
         "funds after a Permit2 allowance has been granted. Seeing this "
         "selector with a monitored owner address is the direct x402 "
         "drain signal."),
    ]
    inserted = 0
    for sel, tag, decoded, notes in entries:
        cur = conn.execute(
            """INSERT OR IGNORE INTO known_selectors
               (function_selector, tag, decoded_name, notes, created)
               VALUES (?, ?, ?, ?, ?)""",
            (sel, tag, decoded, notes, now),
        )
        if cur.rowcount:
            inserted += 1
    conn.commit()
    return inserted


def init_schema(conn: sqlite3.Connection) -> dict:
    """Run all Phase 2 setup steps. Idempotent. Returns a summary."""
    _ensure_tables(conn)
    facilitators_inserted = _seed_facilitators(conn)
    selectors_inserted = _seed_known_selectors(conn)

    # Collect verification data
    summary: dict = {
        "tables": {},
        "facilitators_inserted": facilitators_inserted,
        "selectors_inserted": selectors_inserted,
    }

    for table in ("x402_events", "x402_facilitators", "x402_permit2_exposure"):
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        summary["tables"][table] = {
            "columns": [(c[1], c[2]) for c in cols],
            "row_count": count,
        }

    # Dump facilitator rows and seeded selector rows for verification
    fac_rows = conn.execute(
        """SELECT address, chain, name, classification, source
           FROM x402_facilitators ORDER BY chain"""
    ).fetchall()
    summary["facilitator_rows"] = [dict(r) for r in fac_rows]

    sel_rows = conn.execute(
        """SELECT function_selector, tag, decoded_name
           FROM known_selectors
           WHERE function_selector IN (?, ?, ?, ?, ?)
           ORDER BY function_selector""",
        (SEL_EIP3009_TRANSFER_AUTH, SEL_EIP3009_RECEIVE_AUTH,
         SEL_PERMIT2_PERMIT_SINGLE, SEL_PERMIT2_PERMIT_BATCH,
         SEL_PERMIT2_TRANSFER_FROM),
    ).fetchall()
    summary["selector_rows"] = [dict(r) for r in sel_rows]

    return summary


def _print_init_summary(summary: dict) -> None:
    print("=" * 72)
    print("x402 monitor — Phase 2 schema + classification seed")
    print("=" * 72)
    print()
    for table, info in summary["tables"].items():
        print(f"TABLE {table}  (rows={info['row_count']})")
        for name, col_type in info["columns"]:
            print(f"    {name:<22} {col_type}")
        print()
    print(f"Facilitators inserted this run: {summary['facilitators_inserted']}")
    print(f"Selectors inserted this run:    {summary['selectors_inserted']}")
    print()
    print("x402_facilitators rows (seeded):")
    for row in summary["facilitator_rows"]:
        print(f"  {row['chain']:<10} {row['address']}  [{row['classification']}]")
        print(f"              name={row['name']}")
        print(f"              source={row['source']}")
    print()
    print("known_selectors rows (x402):")
    for row in summary["selector_rows"]:
        print(f"  0x{row['function_selector']}  {row['tag']:<8} {row['decoded_name']}")
    print()


# ---------------------------------------------------------------------
# Phase 3: Calldata decoders
# ---------------------------------------------------------------------

def _word(params: str, i: int) -> Optional[str]:
    start = i * 64
    w = params[start:start + 64]
    return w if len(w) == 64 else None


def _addr_from_word(w: Optional[str]) -> Optional[str]:
    if not w:
        return None
    return "0x" + w[-40:]


def _uint_from_word(w: Optional[str]) -> Optional[int]:
    if not w:
        return None
    try:
        return int(w, 16)
    except ValueError:
        return None


def decode_eip3009(calldata: str) -> Optional[dict]:
    """Decode EIP-3009 transferWithAuthorization / receiveWithAuthorization.

    Signature (both variants share the same layout for the first 3 args):
        fn(address from, address to, uint256 value,
           uint256 validAfter, uint256 validBefore,
           bytes32 nonce, uint8 v, bytes32 r, bytes32 s)

    Returns {"from", "to", "value"} or None on malformed input.
    """
    if not calldata or len(calldata) < 10:
        return None
    params = calldata[10:] if calldata.startswith("0x") else calldata[8:]
    w_from = _word(params, 0)
    w_to = _word(params, 1)
    w_val = _word(params, 2)
    if not w_from or not w_to or not w_val:
        return None
    return {
        "from": _addr_from_word(w_from),
        "to": _addr_from_word(w_to),
        "value": _uint_from_word(w_val),
    }


def decode_permit2_transfer_from(calldata: str) -> Optional[dict]:
    """Decode Permit2.transferFrom(from, to, amount, token).

    Returns {"from", "to", "amount", "token"} or None.
    """
    if not calldata or len(calldata) < 10:
        return None
    params = calldata[10:] if calldata.startswith("0x") else calldata[8:]
    w_from = _word(params, 0)
    w_to = _word(params, 1)
    w_amt = _word(params, 2)
    w_tok = _word(params, 3)
    if not all((w_from, w_to, w_amt, w_tok)):
        return None
    return {
        "from": _addr_from_word(w_from),
        "to": _addr_from_word(w_to),
        "amount": _uint_from_word(w_amt),
        "token": _addr_from_word(w_tok),
    }


def decode_permit2_permit_single(calldata: str) -> Optional[dict]:
    """Decode Permit2.permit(owner, PermitSingle, sig) best-effort.

    Signature:
        permit(address owner,
               ((address token, uint160 amount, uint48 expiration, uint48 nonce),
                address spender,
                uint256 sigDeadline),
               bytes signature)

    ABI layout (all fixed):
        word 0  : owner
        word 1  : details.token
        word 2  : details.amount | expiration | nonce  (packed uint160/uint48/uint48 — amount in upper bits)
        word 3  : spender
        word 4  : sigDeadline
        word 5  : offset to bytes signature

    Returns {"owner", "token", "amount", "expiration", "nonce", "spender"} or None.
    """
    if not calldata or len(calldata) < 10:
        return None
    params = calldata[10:] if calldata.startswith("0x") else calldata[8:]
    w_owner = _word(params, 0)
    w_token = _word(params, 1)
    w_packed = _word(params, 2)
    w_spender = _word(params, 3)
    if not all((w_owner, w_token, w_packed, w_spender)):
        return None
    try:
        packed = int(w_packed, 16)
        # Lower 48 bits = nonce, next 48 = expiration, upper 160 = amount
        nonce = packed & ((1 << 48) - 1)
        expiration = (packed >> 48) & ((1 << 48) - 1)
        amount = packed >> 96
    except ValueError:
        return None
    return {
        "owner": _addr_from_word(w_owner),
        "token": _addr_from_word(w_token),
        "amount": amount,
        "expiration": expiration,
        "nonce": nonce,
        "spender": _addr_from_word(w_spender),
    }


# ---------------------------------------------------------------------
# Phase 3: Live monitor
# ---------------------------------------------------------------------

class X402Monitor:
    """
    Live x402 activity monitor. Attaches to the deployment_monitor block
    processing loop via process_block(w3, block, timestamp_iso).

    For each transaction in a block, checks:

      (a) tx.to = known facilitator contract (x402ExactPermit2Proxy)
          AND selector is an x402 settlement selector
          -> record confirmed x402_event

      (b) tx.to = canonical Permit2 address
          AND selector is a Permit2 consumption/permit selector
          -> record suspected x402_event; if unknown facilitator EOA
             is calling it, emit X402_FACILITATOR_UNKNOWN

      (c) selector = EIP-3009 transferWithAuthorization / receiveWithAuthorization
          (tx.to = token contract)
          -> decode; record suspected x402_event; tx.from is the
             facilitator EOA candidate

      (d) Permit2 transferFrom with decoded payer in x402_permit2_exposure
          -> emit X402_AGENT_DRAIN

    Also periodically syncs x402_permit2_exposure from approval_events
    so stored-potential tracking stays current without duplicating
    decode logic across monitors.
    """

    # Periodic sync / cache refresh cadence
    SYNC_EVERY_BLOCKS = 200
    CACHE_REFRESH_BLOCKS = 500

    def __init__(self, conn: sqlite3.Connection, chain: str = "base"):
        self.conn = conn
        self.chain = chain
        _ensure_tables(conn)
        _seed_facilitators(conn)
        _seed_known_selectors(conn)

        # Caches
        self._known_facilitators: Set[str] = set()
        self._exposed_owners: Set[str] = set()  # payers with active Permit2 allowances
        self._refresh_caches()

        self.events_logged = 0
        self.exposures_added = 0
        self.alerts_generated = 0
        logger.info(
            "X402Monitor initialized for %s  known_facilitators=%d  exposed_owners=%d",
            chain, len(self._known_facilitators), len(self._exposed_owners),
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def _refresh_caches(self) -> None:
        """Reload known-facilitator and exposed-owner sets."""
        try:
            rows = self.conn.execute(
                "SELECT address FROM x402_facilitators "
                "WHERE classification = 'known'"
            ).fetchall()
            self._known_facilitators = {r[0].lower() for r in rows}
        except sqlite3.Error as e:
            logger.warning("facilitator cache refresh failed: %s", e)

        try:
            rows = self.conn.execute(
                "SELECT DISTINCT owner_address FROM x402_permit2_exposure"
            ).fetchall()
            self._exposed_owners = {r[0].lower() for r in rows}
        except sqlite3.Error as e:
            logger.warning("exposure cache refresh failed: %s", e)

    # ------------------------------------------------------------------
    # Block processing
    # ------------------------------------------------------------------
    async def process_block(self, w3, block: dict, timestamp_iso: str) -> None:
        """Scan every tx in the block for x402-relevant calldata."""
        block_number = block["number"]

        # Periodic sync / cache refresh
        if block_number % self.SYNC_EVERY_BLOCKS == 0:
            try:
                added = self.sync_exposure_from_approvals()
                if added:
                    logger.info(
                        "x402: synced %d new Permit2 exposures from approval_events",
                        added,
                    )
            except Exception as e:
                logger.warning("x402 exposure sync failed: %s", e)

        if block_number % self.CACHE_REFRESH_BLOCKS == 0:
            self._refresh_caches()

        for tx in block.get("transactions", []):
            to_addr = (tx.get("to") or "").lower()
            from_addr = (tx.get("from") or "").lower()
            if not to_addr or not from_addr:
                continue

            input_data = tx.get("input") or "0x"
            if isinstance(input_data, bytes):
                input_data = "0x" + input_data.hex()
            if len(input_data) < 10:
                continue
            selector = input_data[2:10].lower()

            if selector not in X402_SELECTORS:
                continue

            tx_hash = tx.get("hash")
            if hasattr(tx_hash, "hex"):
                tx_hash = tx_hash.hex()
            elif isinstance(tx_hash, bytes):
                tx_hash = tx_hash.hex()
            else:
                tx_hash = str(tx_hash or "")
            if tx_hash and not tx_hash.startswith("0x"):
                tx_hash = "0x" + tx_hash

            self._handle_x402_tx(
                tx_hash=tx_hash,
                block_number=block_number,
                timestamp_iso=timestamp_iso,
                from_addr=from_addr,
                to_addr=to_addr,
                selector=selector,
                input_data=input_data,
            )

    def _handle_x402_tx(self, *, tx_hash: str, block_number: int,
                        timestamp_iso: str, from_addr: str, to_addr: str,
                        selector: str, input_data: str) -> None:
        """Classify and record a single x402-selector transaction."""
        # Determine facilitator and confidence
        facilitator = None
        confidence = "suspected"
        unknown_facilitator = False

        if to_addr in self._known_facilitators:
            facilitator = to_addr
            confidence = "confirmed"
        elif to_addr == PERMIT2_ADDRESS:
            # Permit2 call — facilitator is tx.from (EOA)
            facilitator = from_addr
            confidence = "suspected"
            if from_addr not in self._known_facilitators:
                unknown_facilitator = True
        else:
            # Token contract being hit with an EIP-3009 selector — the
            # facilitator is tx.from; to_addr is the token.
            if selector in EIP3009_SELECTORS:
                facilitator = from_addr
                confidence = "suspected"
                if from_addr not in self._known_facilitators:
                    unknown_facilitator = True
            else:
                # Permit2 selector to an unrelated contract — anomalous.
                facilitator = from_addr
                confidence = "suspected"
                unknown_facilitator = True

        # Decode payer / payee / token / amount
        payer = None
        payee = None
        token_contract = None
        amount = None
        x402_type = "permit2"

        if selector in EIP3009_SELECTORS:
            x402_type = "eip3009"
            d = decode_eip3009(input_data)
            if d:
                payer = d["from"]
                payee = d["to"]
                amount = d["value"]
                token_contract = to_addr  # tx.to is the token for EIP-3009
        elif selector == SEL_PERMIT2_TRANSFER_FROM:
            d = decode_permit2_transfer_from(input_data)
            if d:
                payer = d["from"]
                payee = d["to"]
                amount = d["amount"]
                token_contract = d["token"]
        elif selector == SEL_PERMIT2_PERMIT_SINGLE:
            d = decode_permit2_permit_single(input_data)
            if d:
                payer = d["owner"]
                payee = d["spender"]
                token_contract = d["token"]
                amount = d["amount"]
                # A permit call = adding stored potential. Write to
                # x402_permit2_exposure as well as x402_events.
                self._upsert_exposure(
                    owner=d["owner"], spender=d["spender"],
                    token=d["token"], chain=self.chain,
                    allowance=str(d["amount"]) if d["amount"] is not None else None,
                    expiration=d["expiration"], ts=timestamp_iso,
                )

        # Insert x402_event row
        try:
            self.conn.execute(
                """INSERT INTO x402_events
                   (chain, tx_hash, block_number, timestamp,
                    facilitator_address, payer_address, payee_address,
                    token_contract, token_symbol, amount,
                    x402_type, confidence, selector, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.chain, tx_hash, block_number, timestamp_iso,
                 facilitator, payer, payee,
                 token_contract, None, amount,
                 x402_type, confidence, selector, _now_iso()),
            )
            self.events_logged += 1
        except Exception as e:
            logger.debug("x402_events insert failed: %s", e)
            return

        # Bump facilitator stats
        if facilitator:
            try:
                self.conn.execute(
                    """UPDATE x402_facilitators
                       SET tx_count   = tx_count + 1,
                           last_seen  = ?,
                           first_seen = COALESCE(first_seen, ?)
                       WHERE address = ? AND chain = ?""",
                    (timestamp_iso, timestamp_iso, facilitator, self.chain),
                )
                # If facilitator was unknown, record it as such
                if unknown_facilitator:
                    self.conn.execute(
                        """INSERT OR IGNORE INTO x402_facilitators
                           (address, chain, name, classification, source,
                            first_seen, last_seen, tx_count, total_volume, created_at)
                           VALUES (?, ?, ?, 'unknown', 'observed',
                                   ?, ?, 1, 0, ?)""",
                        (facilitator, self.chain, None,
                         timestamp_iso, timestamp_iso, _now_iso()),
                    )
            except Exception:
                pass

        # Alerts
        if unknown_facilitator:
            self._alert(
                "X402_FACILITATOR_UNKNOWN",
                facilitator or to_addr,
                tx_hash, block_number, timestamp_iso,
                {
                    "facilitator": facilitator,
                    "selector": selector,
                    "selector_name": X402_SELECTORS.get(selector),
                    "tx_to": to_addr,
                    "payer": payer,
                    "chain": self.chain,
                    "message": (
                        f"Unknown facilitator {facilitator} called "
                        f"{X402_SELECTORS.get(selector, selector)} on {self.chain}"
                    ),
                },
            )

        # Drain check: Permit2 transferFrom from an address we track
        if selector == SEL_PERMIT2_TRANSFER_FROM and payer:
            if payer.lower() in self._exposed_owners:
                self._alert(
                    "X402_AGENT_DRAIN",
                    payer,
                    tx_hash, block_number, timestamp_iso,
                    {
                        "payer": payer,
                        "payee": payee,
                        "token": token_contract,
                        "amount": amount,
                        "facilitator": facilitator,
                        "chain": self.chain,
                        "message": (
                            f"Permit2.transferFrom pulled from exposed payer "
                            f"{payer[:18]}... on {self.chain}"
                        ),
                    },
                )

        try:
            self.conn.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Exposure management
    # ------------------------------------------------------------------
    def _upsert_exposure(self, *, owner: str, spender: str, token: str,
                         chain: str, allowance: Optional[str],
                         expiration: Optional[int], ts: str) -> bool:
        """Insert or update a row in x402_permit2_exposure.
        Returns True if a new row was inserted (not a last_seen update).
        """
        if not owner or not spender or not token:
            return False
        try:
            cur = self.conn.execute(
                """INSERT INTO x402_permit2_exposure
                   (owner_address, spender_address, token_contract, chain,
                    allowance_amount, expiration,
                    first_seen, last_seen, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(owner_address, spender_address, token_contract, chain)
                   DO UPDATE SET
                       last_seen = excluded.last_seen,
                       allowance_amount = COALESCE(excluded.allowance_amount, allowance_amount),
                       expiration = COALESCE(excluded.expiration, expiration)""",
                (owner.lower(), spender.lower(), token.lower(), chain,
                 allowance, expiration, ts, ts, _now_iso()),
            )
            inserted = cur.rowcount > 0
            if inserted:
                self.exposures_added += 1
                self._exposed_owners.add(owner.lower())
                # First-time exposure alert
                self._alert(
                    "X402_PERMIT2_EXPOSURE", owner, None, None, ts,
                    {
                        "owner": owner,
                        "spender": spender,
                        "token": token,
                        "chain": chain,
                        "message": (
                            f"New Permit2 exposure: {owner[:18]}... granted "
                            f"allowance on {token[:18]}... to spender "
                            f"{spender[:18]}... on {chain}"
                        ),
                    },
                )
            return inserted
        except sqlite3.Error as e:
            logger.debug("exposure upsert failed: %s", e)
            return False

    def sync_exposure_from_approvals(self) -> int:
        """Pull any Permit2 approvals from approval_events into
        x402_permit2_exposure. Idempotent via UNIQUE constraint.

        This is the bridge between the existing approval_events stream
        (written by event_monitors.py when a suspected token gets an
        approve() call) and the x402-specific exposure table. We do NOT
        duplicate the decode logic — approval_events is already correct,
        we just project the rows where spender = Permit2.

        Used for both historical backfill (--backfill CLI) and periodic
        live sync from process_block.

        Returns count of new rows inserted.
        """
        try:
            rows = self.conn.execute(
                """SELECT chain, token_contract, approver, spender,
                          MIN(timestamp) AS first_ts,
                          MAX(timestamp) AS last_ts
                   FROM approval_events
                   WHERE spender = ?
                   GROUP BY chain, token_contract, approver""",
                (PERMIT2_ADDRESS,),
            ).fetchall()
        except sqlite3.Error as e:
            logger.warning("approval_events query failed: %s", e)
            return 0

        added = 0
        for r in rows:
            chain = r[0] or self.chain
            token = r[1]
            owner = r[2]
            spender = r[3]
            first_ts = r[4]
            last_ts = r[5]
            if not token or not owner or not spender:
                continue
            try:
                cur = self.conn.execute(
                    """INSERT INTO x402_permit2_exposure
                       (owner_address, spender_address, token_contract, chain,
                        allowance_amount, expiration,
                        first_seen, last_seen, created_at)
                       VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                       ON CONFLICT(owner_address, spender_address, token_contract, chain)
                       DO UPDATE SET
                           last_seen = CASE
                               WHEN excluded.last_seen > last_seen
                               THEN excluded.last_seen
                               ELSE last_seen
                           END""",
                    (owner.lower(), spender.lower(), token.lower(), chain,
                     first_ts, last_ts, _now_iso()),
                )
                if cur.rowcount > 0:
                    added += 1
                    self.exposures_added += 1
                    self._exposed_owners.add(owner.lower())
            except sqlite3.Error:
                continue
        try:
            self.conn.commit()
        except Exception:
            pass
        return added

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def _alert(self, alert_type: str, address: Optional[str],
               tx_hash: Optional[str], block_number: Optional[int],
               timestamp: str, payload: dict) -> None:
        try:
            self.conn.execute(
                """INSERT INTO alerts
                   (alert_type, address, tx_hash, block_number, timestamp,
                    payload, false_positive)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (alert_type, address or "", tx_hash, block_number,
                 timestamp, json.dumps(payload)),
            )
            self.alerts_generated += 1
            logger.warning("ALERT: %s %s", alert_type, (address or "")[:18])
        except sqlite3.Error as e:
            logger.debug("alert insert failed: %s", e)


# ---------------------------------------------------------------------
# Phase 4: Trust amplification analysis
# ---------------------------------------------------------------------
#
# Mirrors the methodology in surveillance.trust_amplification.py but
# applied to x402 payer flow instead of Uniswap-router caller flow.
#
# For each contract that appears in x402_events as a payee, compute:
#   x402_callers_per_day = distinct_payers / span_days
#
# Compare to the bytecode-family average callers/day (same helper as
# trust_amplification.py — we replicate it here rather than importing
# to keep the two modules independent, per the spec invariant to not
# modify trust_amplification.py).
#
# amplification_factor = x402_callers_per_day / family_avg_callers_per_day
#
# If factor > 2.0, emit X402_TRUST_AMPLIFICATION alert. Sample-size
# warnings are returned in the report so downstream consumers can
# apply their own confidence filters.
#
# Insufficient-data path: if x402_events has < MIN_EVENTS_FOR_ANALYSIS
# rows total, return {"status": "insufficient_data"} without computing
# per-contract statistics. This is the expected baseline behavior when
# no live facilitator settlements have been captured yet.

MIN_EVENTS_FOR_ANALYSIS = 10
MIN_EVENTS_PER_CONTRACT = 5
AMPLIFICATION_ALERT_THRESHOLD = 2.0


def _get_family_avg_callers_per_day(
    conn: sqlite3.Connection, contract_address: str,
) -> Optional[float]:
    """Return the average callers/day across sibling contracts in the
    same bytecode family, or None if the contract has no family or
    the family has no comparable activity.

    Replicated from surveillance.trust_amplification.py:_get_family_avg.
    The two modules are deliberately kept independent.
    """
    fam = conn.execute(
        "SELECT family_id FROM bytecode_family_members "
        "WHERE contract_address = ?",
        (contract_address,),
    ).fetchone()
    if not fam:
        return None

    siblings = conn.execute(
        "SELECT bfm.contract_address "
        "FROM bytecode_family_members bfm WHERE bfm.family_id = ?",
        (fam[0],),
    ).fetchall()
    if not siblings:
        return None

    total_cpd = 0.0
    count = 0
    for s in siblings:
        row = conn.execute(
            """SELECT COUNT(DISTINCT interacting_address) AS callers,
                      JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp))
                        AS span_days
               FROM transaction_events
               WHERE contract_address = ?""",
            (s[0],),
        ).fetchone()
        if row and row[0] and row[1] and row[1] > 0:
            total_cpd += row[0] / row[1]
            count += 1

    return (total_cpd / count) if count else None


def amplification(conn: sqlite3.Connection,
                  emit_alerts: bool = True) -> dict:
    """Compute x402 trust amplification for contracts that have
    received at least MIN_EVENTS_PER_CONTRACT x402 events.

    Returns a structured report. Emits X402_TRUST_AMPLIFICATION
    alerts when the per-contract factor exceeds the threshold.

    Zero RPC, pure SQLite.
    """
    _ensure_tables(conn)

    total_events = conn.execute(
        "SELECT COUNT(*) FROM x402_events"
    ).fetchone()[0] or 0
    distinct_payees = conn.execute(
        "SELECT COUNT(DISTINCT payee_address) FROM x402_events "
        "WHERE payee_address IS NOT NULL"
    ).fetchone()[0] or 0

    report: dict = {
        "status": "ok",
        "x402_events_total": total_events,
        "distinct_payees": distinct_payees,
        "threshold": AMPLIFICATION_ALERT_THRESHOLD,
        "min_events_per_contract": MIN_EVENTS_PER_CONTRACT,
        "min_events_for_analysis": MIN_EVENTS_FOR_ANALYSIS,
        "results": [],
        "alerts_emitted": 0,
        "warnings": [],
    }

    if total_events < MIN_EVENTS_FOR_ANALYSIS:
        report["status"] = "insufficient_data"
        report["message"] = (
            f"Only {total_events} x402_events in the corpus "
            f"(minimum {MIN_EVENTS_FOR_ANALYSIS} required for analysis). "
            "The detector is active and will start producing amplification "
            "results the moment facilitator settlements are captured. "
            "No amplification comparison performed."
        )
        return report

    # Find payee contracts with enough x402 events to analyze
    rows = conn.execute(
        """SELECT payee_address,
                  COUNT(*)                           AS events,
                  COUNT(DISTINCT payer_address)      AS payers,
                  JULIANDAY(MAX(timestamp))
                    - JULIANDAY(MIN(timestamp))      AS span_days,
                  MIN(timestamp)                     AS first_seen,
                  MAX(timestamp)                     AS last_seen,
                  SUM(CASE WHEN confidence = 'confirmed' THEN 1 ELSE 0 END)
                    AS confirmed_events
           FROM x402_events
           WHERE payee_address IS NOT NULL
             AND confidence IN ('confirmed','suspected')
           GROUP BY payee_address
           HAVING events >= ?
           ORDER BY payers DESC""",
        (MIN_EVENTS_PER_CONTRACT,),
    ).fetchall()

    if not rows:
        report["warnings"].append(
            f"No payee contracts have reached {MIN_EVENTS_PER_CONTRACT} "
            "x402 events yet. Amplification analysis skipped."
        )
        return report

    for r in rows:
        addr = r["payee_address"]
        events = r["events"]
        payers = r["payers"]
        span = max(r["span_days"] or 1.0 / 24.0, 1.0 / 24.0)  # >= 1 hour
        x402_cpd = round(payers / span, 3)

        family_avg = _get_family_avg_callers_per_day(conn, addr)
        if family_avg and family_avg > 0:
            factor = round(x402_cpd / family_avg, 2)
            comparator_status = "ok"
        else:
            factor = None
            comparator_status = "no_family_baseline"

        # Sample-size warning
        sample_warning = None
        if events < 20:
            sample_warning = f"small_sample (events={events})"

        result = {
            "payee_address": addr,
            "x402_events": events,
            "confirmed_events": r["confirmed_events"],
            "distinct_payers": payers,
            "span_days": round(span, 3),
            "x402_callers_per_day": x402_cpd,
            "family_avg_callers_per_day": (
                round(family_avg, 3) if family_avg is not None else None
            ),
            "amplification_factor": factor,
            "comparator_status": comparator_status,
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "sample_warning": sample_warning,
        }
        report["results"].append(result)

        # Alert
        if emit_alerts and factor is not None and factor > AMPLIFICATION_ALERT_THRESHOLD:
            try:
                payload = {
                    "payee_address": addr,
                    "x402_events": events,
                    "distinct_payers": payers,
                    "x402_callers_per_day": x402_cpd,
                    "family_avg_callers_per_day": round(family_avg, 3),
                    "amplification_factor": factor,
                    "threshold": AMPLIFICATION_ALERT_THRESHOLD,
                    "sample_warning": sample_warning,
                    "message": (
                        f"x402 trust amplification: {addr[:18]}... receives "
                        f"{x402_cpd:.2f} x402 payers/day vs family baseline "
                        f"{family_avg:.2f}/day (factor={factor}x)"
                    ),
                }
                conn.execute(
                    """INSERT INTO alerts
                       (alert_type, address, tx_hash, block_number,
                        timestamp, payload, false_positive)
                       VALUES ('X402_TRUST_AMPLIFICATION', ?, NULL, NULL,
                               ?, ?, 0)""",
                    (addr, _now_iso(), json.dumps(payload)),
                )
                report["alerts_emitted"] += 1
            except sqlite3.Error as e:
                logger.debug("amplification alert insert failed: %s", e)

    try:
        conn.commit()
    except Exception:
        pass

    return report


def _print_amplification_report(report: dict) -> None:
    print("=" * 72)
    print("x402 monitor -- Phase 4 trust amplification analysis")
    print("=" * 72)
    print(f"Total x402 events in corpus:   {report['x402_events_total']:,}")
    print(f"Distinct payee contracts:      {report['distinct_payees']:,}")
    print(f"Analysis minimum:              {report['min_events_for_analysis']} "
          f"events total, {report['min_events_per_contract']} per contract")
    print(f"Alert threshold:               {report['threshold']}x")
    print()

    if report["status"] == "insufficient_data":
        print("STATUS: insufficient data")
        print()
        print(f"  {report['message']}")
        print()
        print("  This is the expected baseline. The x402 monitor is wired")
        print("  into the live block loop. On the next deploy, any x402")
        print("  settlement captured will start accumulating into x402_events,")
        print("  and this analysis will produce real amplification factors.")
        print()
        return

    if not report["results"]:
        print("STATUS: no qualifying contracts")
        for w in report["warnings"]:
            print(f"  {w}")
        print()
        return

    print(f"STATUS: ok  ({len(report['results'])} contracts analyzed, "
          f"{report['alerts_emitted']} alerts emitted)")
    print()
    print("Per-contract results:")
    print(f"  {'payee':<44} {'events':>7} {'payers':>7} {'x402/d':>8} "
          f"{'fam/d':>8} {'factor':>7}")
    for r in report["results"]:
        payee = r["payee_address"] or "?"
        payee = payee[:42]
        fam = (f"{r['family_avg_callers_per_day']:>8.2f}"
               if r["family_avg_callers_per_day"] is not None
               else f"{'n/a':>8}")
        fac = (f"{r['amplification_factor']:>7.2f}"
               if r["amplification_factor"] is not None
               else f"{'n/a':>7}")
        warn_marker = " !" if r["sample_warning"] else ""
        print(f"  {payee:<44} {r['x402_events']:>7} {r['distinct_payers']:>7} "
              f"{r['x402_callers_per_day']:>8.2f} {fam} {fac}{warn_marker}")

    # Comparator coverage
    no_family = sum(1 for r in report["results"]
                    if r["comparator_status"] == "no_family_baseline")
    if no_family:
        print()
        print(f"  ! {no_family} contract(s) have no bytecode family "
              "baseline (payee may be an EOA or novel contract).")
    small = sum(1 for r in report["results"] if r["sample_warning"])
    if small:
        print(f"  ! {small} contract(s) flagged with small-sample warning "
              f"(events < 20)")
    print()


def backfill(conn: sqlite3.Connection, chain: str = "base") -> dict:
    """Phase 3 backfill: populate x402_permit2_exposure from the
    existing approval_events rows where spender = canonical Permit2.

    Zero RPC. Idempotent.
    """
    _ensure_tables(conn)
    monitor = X402Monitor(conn, chain=chain)

    # Count how many exposure rows exist before
    before = conn.execute(
        "SELECT COUNT(*) FROM x402_permit2_exposure"
    ).fetchone()[0] or 0
    # sync uses INSERT ... ON CONFLICT DO UPDATE which inflates rowcount
    # (both inserts and updates report rowcount=1). Use the table-count
    # delta instead for an accurate new-row measurement.
    monitor.sync_exposure_from_approvals()
    after = conn.execute(
        "SELECT COUNT(*) FROM x402_permit2_exposure"
    ).fetchone()[0] or 0
    added = max(0, after - before)

    summary = {
        "approval_events_with_permit2_spender": conn.execute(
            "SELECT COUNT(*) FROM approval_events WHERE spender = ?",
            (PERMIT2_ADDRESS,),
        ).fetchone()[0] or 0,
        "distinct_exposures_before": before,
        "new_exposures_inserted": added,
        "distinct_exposures_after": after,
        "distinct_owners": conn.execute(
            "SELECT COUNT(DISTINCT owner_address) FROM x402_permit2_exposure"
        ).fetchone()[0] or 0,
        "distinct_tokens": conn.execute(
            "SELECT COUNT(DISTINCT token_contract) FROM x402_permit2_exposure"
        ).fetchone()[0] or 0,
        "by_chain": [
            dict(r) for r in conn.execute(
                """SELECT chain,
                          COUNT(*) AS rows,
                          COUNT(DISTINCT owner_address) AS owners,
                          COUNT(DISTINCT token_contract) AS tokens
                   FROM x402_permit2_exposure GROUP BY chain"""
            ).fetchall()
        ],
    }
    return summary


def _print_backfill_summary(summary: dict) -> None:
    print("=" * 72)
    print("x402 monitor -- Phase 3 backfill (approval_events -> x402_permit2_exposure)")
    print("=" * 72)
    print(f"approval_events rows with Permit2 spender: "
          f"{summary['approval_events_with_permit2_spender']:,}")
    print(f"Exposures before:        {summary['distinct_exposures_before']:,}")
    print(f"New exposures inserted:  {summary['new_exposures_inserted']:,}")
    print(f"Exposures after:         {summary['distinct_exposures_after']:,}")
    print(f"Distinct owners:         {summary['distinct_owners']:,}")
    print(f"Distinct tokens:         {summary['distinct_tokens']}")
    print()
    print("By chain:")
    for row in summary["by_chain"]:
        print(f"  {row['chain']:<10} rows={row['rows']:>5} "
              f"owners={row['owners']:>5} tokens={row['tokens']:>4}")
    print()


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
        "--init-schema", action="store_true",
        help="Phase 2: create x402 tables and seed facilitator + selector "
             "registries. Idempotent.",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Phase 3: populate x402_permit2_exposure from existing "
             "approval_events (Permit2 spender). Idempotent, zero RPC.",
    )
    parser.add_argument(
        "--amplification", action="store_true",
        help="Phase 4: compute x402 trust amplification for payee "
             "contracts vs bytecode-family baselines. Reports "
             "insufficient-data when x402_events is empty.",
    )
    parser.add_argument(
        "--chain", default="base",
        help="Chain label for backfilled rows (default: base)",
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
        if args.init_schema:
            summary = init_schema(conn)
            _print_init_summary(summary)
            return 0
        if args.backfill:
            summary = backfill(conn, chain=args.chain)
            _print_backfill_summary(summary)
            return 0
        if args.amplification:
            report = amplification(conn)
            _print_amplification_report(report)
            return 0
        if args.recon:
            report = recon(conn)
            _print_report(report)
            return 0
        parser.print_help()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
