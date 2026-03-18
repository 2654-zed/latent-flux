"""
Layer 3 — Component 2: Bytecode Classifier

Static analysis of deployed EVM bytecode to detect trap patterns.
No execution, no simulation — pattern matching on raw opcodes only.

Conservative by design: when in doubt, returns UNKNOWN. A false negative
is acceptable; a false positive destroys credibility permanently.

The classifier receives a Deployment dataclass (from the deployment monitor)
and returns a classification dict consumed by db.insert_contract().
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surveillance.deployment_monitor import Deployment

logger = logging.getLogger("surveillance.classifier")


# =====================================================================
# EVM opcode constants (single-byte hex)
# =====================================================================
OP_STOP = "00"
OP_CALLER = "33"          # msg.sender
OP_ORIGIN = "32"          # tx.origin
OP_TIMESTAMP = "42"       # block.timestamp
OP_NUMBER = "43"          # block.number
OP_SLOAD = "54"           # storage read
OP_SSTORE = "55"          # storage write
OP_JUMPI = "57"           # conditional jump
OP_REVERT = "fd"          # revert
OP_SELFDESTRUCT = "ff"    # selfdestruct
OP_DELEGATECALL = "f4"    # delegatecall
OP_CALL = "f1"            # external call
OP_CALLVALUE = "34"       # msg.value
OP_EQ = "14"              # equality comparison
OP_ISZERO = "15"          # zero check
OP_GT = "11"              # greater than
OP_LT = "10"              # less than
OP_MUL = "02"             # multiply
OP_DIV = "04"             # divide
OP_SHA3 = "20"            # keccak256
OP_MSTORE = "52"          # memory store

# ERC-20 function selectors (first 4 bytes of keccak256)
SEL_TRANSFER = "a9059cbb"          # transfer(address,uint256)
SEL_TRANSFER_FROM = "23b872dd"     # transferFrom(address,address,uint256)
SEL_APPROVE = "095ea7b3"           # approve(address,uint256)
SEL_BALANCE_OF = "70a08231"        # balanceOf(address)

# Flash loan callback selectors
SEL_UNISWAP_V2_CALL = "10d1e85c"  # uniswapV2Call(address,uint256,uint256,bytes)
SEL_UNISWAP_V3_CALLBACK = "fa461e33"  # uniswapV3SwapCallback(int256,int256,bytes)


# =====================================================================
# Pattern detectors
#
# Each returns (detected: bool, notes: str).
# Notes explain WHAT was found and WHERE (byte offset) — this becomes
# the provenance trail in confidence_reason.
# =====================================================================

def _find_all_offsets(bytecode_hex: str, opcode: str) -> list[int]:
    """Find byte offsets of an opcode in raw hex. Each hex char = half byte."""
    offsets = []
    i = 0
    bc = bytecode_hex.lower()
    while i < len(bc) - 1:
        if bc[i:i+2] == opcode:
            offsets.append(i // 2)  # byte offset
        # Skip PUSH data: PUSH1..PUSH32 = 0x60..0x7f
        op = int(bc[i:i+2], 16)
        if 0x60 <= op <= 0x7f:
            push_bytes = op - 0x5f
            i += 2 + push_bytes * 2  # skip opcode + pushed data
        else:
            i += 2
    return offsets


def _opcodes_near(bytecode_hex: str, anchor_op: str, target_op: str,
                  window_bytes: int = 30) -> list[tuple[int, int]]:
    """
    Find pairs where target_op appears within window_bytes AFTER anchor_op.
    Returns list of (anchor_offset, target_offset).
    """
    anchors = _find_all_offsets(bytecode_hex, anchor_op)
    targets = set(_find_all_offsets(bytecode_hex, target_op))
    pairs = []
    for a in anchors:
        for t in targets:
            if 0 < t - a <= window_bytes:
                pairs.append((a, t))
    return pairs


def detect_asymmetric_transfer(bytecode_hex: str) -> tuple[bool, str]:
    """
    CALLER opcode followed by EQ + JUMPI near a REVERT — indicates
    "if msg.sender != owner, revert" gating on transfer.

    Pattern: CALLER ... EQ ... JUMPI ... REVERT within a transfer selector context.
    """
    # Must contain a transfer selector to be relevant
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    # Look for CALLER -> EQ within 10 bytes, then JUMPI within 10 more
    caller_eq = _opcodes_near(bytecode_hex, OP_CALLER, OP_EQ, window_bytes=10)
    if not caller_eq:
        return False, ""

    # For each CALLER->EQ pair, look for JUMPI nearby
    for caller_off, eq_off in caller_eq:
        jumpi_near = _opcodes_near(
            bytecode_hex[eq_off*2:], "00", OP_JUMPI, window_bytes=10
        )
        # Actually search from eq_off for JUMPI
        jumpi_offsets = _find_all_offsets(bytecode_hex, OP_JUMPI)
        for j in jumpi_offsets:
            if 0 < j - eq_off <= 15:
                # Now check for REVERT within 50 bytes of JUMPI
                revert_offsets = _find_all_offsets(bytecode_hex, OP_REVERT)
                for r in revert_offsets:
                    if 0 < r - j <= 50:
                        return True, (
                            f"CALLER at 0x{caller_off:x} -> EQ at 0x{eq_off:x} -> "
                            f"JUMPI at 0x{j:x} -> REVERT at 0x{r:x}: "
                            f"conditional revert gated on msg.sender in transfer context"
                        )

    return False, ""


def detect_blacklist_check(bytecode_hex: str) -> tuple[bool, str]:
    """
    SLOAD of address-keyed mapping followed by conditional REVERT.

    Pattern: CALLER/ORIGIN -> SLOAD -> ISZERO/JUMPI -> REVERT.
    This is the "blacklisted[msg.sender]" pattern.
    """
    # Need both CALLER and SLOAD
    caller_sload = _opcodes_near(bytecode_hex, OP_CALLER, OP_SLOAD, window_bytes=20)
    origin_sload = _opcodes_near(bytecode_hex, OP_ORIGIN, OP_SLOAD, window_bytes=20)
    pairs = caller_sload + origin_sload

    if not pairs:
        return False, ""

    for src_off, sload_off in pairs:
        # Look for ISZERO or JUMPI after SLOAD
        jumpi_offsets = _find_all_offsets(bytecode_hex, OP_JUMPI)
        for j in jumpi_offsets:
            if 0 < j - sload_off <= 15:
                revert_offsets = _find_all_offsets(bytecode_hex, OP_REVERT)
                for r in revert_offsets:
                    if 0 < r - j <= 50:
                        src_name = "CALLER" if src_off in [p[0] for p in caller_sload] else "ORIGIN"
                        return True, (
                            f"{src_name} at 0x{src_off:x} -> SLOAD at 0x{sload_off:x} -> "
                            f"JUMPI at 0x{j:x} -> REVERT at 0x{r:x}: "
                            f"address-keyed storage read gating transfer (blacklist pattern)"
                        )

    return False, ""


def detect_tx_origin_conditional(bytecode_hex: str) -> tuple[bool, str]:
    """
    tx.origin (ORIGIN opcode) used in conditional logic — a strong
    honeypot signal. Legitimate contracts almost never branch on tx.origin.

    Pattern: ORIGIN -> EQ/JUMPI in transfer context.
    """
    origin_offsets = _find_all_offsets(bytecode_hex, OP_ORIGIN)
    if not origin_offsets:
        return False, ""

    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    # ORIGIN followed by EQ or JUMPI
    for off in origin_offsets:
        eq_offsets = _find_all_offsets(bytecode_hex, OP_EQ)
        jumpi_offsets = _find_all_offsets(bytecode_hex, OP_JUMPI)

        for eq in eq_offsets:
            if 0 < eq - off <= 12:
                for j in jumpi_offsets:
                    if 0 < j - eq <= 8:
                        return True, (
                            f"ORIGIN at 0x{off:x} -> EQ at 0x{eq:x} -> JUMPI at 0x{j:x}: "
                            f"tx.origin conditional in transfer context (buy-not-sell pattern)"
                        )

    return False, ""


def detect_callback_trap(bytecode_hex: str) -> tuple[bool, str]:
    """
    Flash loan callback selector present without a corresponding outbound
    CALL after it — the callback doesn't repay.

    Pattern: callback selector present + fewer than expected CALL opcodes
    in the callback region.
    """
    has_v2_callback = SEL_UNISWAP_V2_CALL in bytecode_hex
    has_v3_callback = SEL_UNISWAP_V3_CALLBACK in bytecode_hex

    if not has_v2_callback and not has_v3_callback:
        return False, ""

    callback_name = "uniswapV2Call" if has_v2_callback else "uniswapV3SwapCallback"
    callback_sel = SEL_UNISWAP_V2_CALL if has_v2_callback else SEL_UNISWAP_V3_CALLBACK

    # Find the callback selector offset
    sel_idx = bytecode_hex.find(callback_sel)
    if sel_idx < 0:
        return False, ""

    # Look at the 200 bytes after the selector for CALL opcodes
    region_start = sel_idx
    region_end = min(sel_idx + 400, len(bytecode_hex))  # 200 bytes = 400 hex chars
    region = bytecode_hex[region_start:region_end]

    call_count = len(_find_all_offsets(region, OP_CALL))
    revert_count = len(_find_all_offsets(region, OP_REVERT))

    # A legitimate callback should have at least 1 outbound CALL (the repayment).
    # If there are 0 CALLs but REVERT is present, suspicious.
    if call_count == 0 and revert_count > 0:
        return True, (
            f"{callback_name} selector at 0x{sel_idx//2:x}: "
            f"callback handler has {revert_count} REVERT(s) but 0 outbound CALLs "
            f"in next 200 bytes — incomplete repayment pattern"
        )

    return False, ""


def detect_hidden_fee(bytecode_hex: str) -> tuple[bool, str]:
    """
    Fee-on-transfer anomaly: multiple SLOAD -> arithmetic -> SSTORE sequences
    near transfer selectors, indicating hidden fee deductions.

    We look for an unusually high density of SSTORE ops near transfer code,
    which suggests value is being siphoned during transfers.
    """
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    # Count SSTORE ops in the entire contract
    sstore_offsets = _find_all_offsets(bytecode_hex, OP_SSTORE)
    sload_offsets = _find_all_offsets(bytecode_hex, OP_SLOAD)

    # Legitimate ERC-20: ~2 SSTOREs per transfer (sender balance, receiver balance)
    # Honeypot with hidden fees: 4+ SSTOREs (extra writes for fee collection,
    # owner balance, fee accumulator, etc.)
    # We also check for SLOAD density — fee logic reads extra storage slots.
    total_bytes = len(bytecode_hex) // 2

    if total_bytes == 0:
        return False, ""

    sstore_density = len(sstore_offsets) / (total_bytes / 100)  # per 100 bytes
    sload_density = len(sload_offsets) / (total_bytes / 100)

    # Threshold: >3 SSTOREs per 100 bytes in a transfer-containing contract
    # is abnormal. Most ERC-20s have ~0.5-1.5.
    if sstore_density > 3.0 and sload_density > 4.0:
        return True, (
            f"High storage density near transfer code: "
            f"{len(sstore_offsets)} SSTOREs ({sstore_density:.1f}/100B), "
            f"{len(sload_offsets)} SLOADs ({sload_density:.1f}/100B) "
            f"in {total_bytes}B contract — hidden fee structure suspected"
        )

    return False, ""


def detect_selfdestruct(bytecode_hex: str) -> tuple[bool, str]:
    """
    SELFDESTRUCT in a token contract is a strong rug-pull signal.
    Legitimate ERC-20s never need self-destruct capability.
    """
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    offsets = _find_all_offsets(bytecode_hex, OP_SELFDESTRUCT)
    if offsets:
        offset_str = ", ".join(f"0x{o:x}" for o in offsets)
        return True, (
            f"SELFDESTRUCT at {offset_str} in ERC-20 transfer context: "
            f"legitimate tokens do not self-destruct"
        )

    return False, ""


def detect_delegatecall_in_token(bytecode_hex: str) -> tuple[bool, str]:
    """
    DELEGATECALL in a token contract — the implementation can be swapped
    post-deployment to change transfer behavior. Common in upgradeable
    honeypots: deploy clean, upgrade to trap.
    """
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    offsets = _find_all_offsets(bytecode_hex, OP_DELEGATECALL)
    if offsets:
        offset_str = ", ".join(f"0x{o:x}" for o in offsets)
        return True, (
            f"DELEGATECALL at {offset_str} in ERC-20 context: "
            f"upgradeable token — transfer logic can be changed post-deployment"
        )

    return False, ""


def detect_timestamp_activation(bytecode_hex: str) -> tuple[bool, str]:
    """
    Time-lock delayed activation. TIMESTAMP/NUMBER -> GT/LT -> ISZERO -> JUMPI -> REVERT.
    Contract allows transfers freely until a specific block/timestamp, then restricts.
    """
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    for time_op, time_name in [(OP_TIMESTAMP, "TIMESTAMP"), (OP_NUMBER, "NUMBER")]:
        for cmp_op, cmp_name in [(OP_GT, "GT"), (OP_LT, "LT")]:
            pairs = _opcodes_near(bytecode_hex, time_op, cmp_op, window_bytes=20)
            for time_off, cmp_off in pairs:
                jumpi_offsets = _find_all_offsets(bytecode_hex, OP_JUMPI)
                for j in jumpi_offsets:
                    if 0 < j - cmp_off <= 15:
                        revert_offsets = _find_all_offsets(bytecode_hex, OP_REVERT)
                        for r in revert_offsets:
                            if 0 < r - j <= 50:
                                return True, (
                                    f"{time_name} at 0x{time_off:x} -> {cmp_name} at 0x{cmp_off:x} -> "
                                    f"JUMPI at 0x{j:x} -> REVERT at 0x{r:x}: "
                                    f"time-locked activation trap in transfer context"
                                )
    return False, ""


def detect_origin_eoa_gate(bytecode_hex: str) -> tuple[bool, str]:
    """
    tx.origin == msg.sender check blocks smart contract callers (bots, routers)
    while allowing retail EOA victims through. ORIGIN -> CALLER -> EQ pattern.
    """
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    # Pattern: ORIGIN followed by CALLER within 5 bytes, then EQ
    origin_caller = _opcodes_near(bytecode_hex, OP_ORIGIN, OP_CALLER, window_bytes=5)
    if not origin_caller:
        return False, ""

    for origin_off, caller_off in origin_caller:
        eq_offsets = _find_all_offsets(bytecode_hex, OP_EQ)
        for eq in eq_offsets:
            if 0 < eq - caller_off <= 5:
                jumpi_offsets = _find_all_offsets(bytecode_hex, OP_JUMPI)
                for j in jumpi_offsets:
                    if 0 < j - eq <= 10:
                        return True, (
                            f"ORIGIN at 0x{origin_off:x} -> CALLER at 0x{caller_off:x} -> "
                            f"EQ at 0x{eq:x} -> JUMPI at 0x{j:x}: "
                            f"tx.origin == msg.sender EOA gate - blocks bots/routers, "
                            f"allows retail EOA victims"
                        )
    return False, ""


def detect_obfuscated_fee(bytecode_hex: str) -> tuple[bool, str]:
    """
    Hidden tax via KECCAK256-keyed storage. MSTORE -> SHA3 -> SLOAD -> JUMPI -> MUL/DIV.
    The KECCAK256 output is used as a storage pointer for a blacklist/fee mapping.
    If the result gates arithmetic on the transfer amount, a concealed fee is present.
    """
    has_transfer = SEL_TRANSFER in bytecode_hex or SEL_TRANSFER_FROM in bytecode_hex
    if not has_transfer:
        return False, ""

    # Look for SHA3 -> SLOAD -> JUMPI -> MUL or DIV chain
    sha3_sload = _opcodes_near(bytecode_hex, OP_SHA3, OP_SLOAD, window_bytes=15)
    if not sha3_sload:
        return False, ""

    for sha3_off, sload_off in sha3_sload:
        jumpi_offsets = _find_all_offsets(bytecode_hex, OP_JUMPI)
        for j in jumpi_offsets:
            if 0 < j - sload_off <= 20:
                # Check for MUL or DIV after the JUMPI (fee calculation)
                for arith_op, arith_name in [(OP_MUL, "MUL"), (OP_DIV, "DIV")]:
                    arith_offsets = _find_all_offsets(bytecode_hex, arith_op)
                    for a in arith_offsets:
                        if 0 < a - j <= 40:
                            return True, (
                                f"SHA3 at 0x{sha3_off:x} -> SLOAD at 0x{sload_off:x} -> "
                                f"JUMPI at 0x{j:x} -> {arith_name} at 0x{a:x}: "
                                f"KECCAK256-keyed storage lookup gates arithmetic on "
                                f"transfer amount - obfuscated fee-on-transfer"
                            )
    return False, ""


# =====================================================================
# Pattern registry
# =====================================================================

@dataclass
class PatternResult:
    name: str
    detected: bool
    notes: str


# Maps pattern name -> (detector function, bytecode_signal field name or None)
PATTERN_REGISTRY: list[tuple[str, callable, str | None]] = [
    ("asymmetric_transfer", detect_asymmetric_transfer, "has_asymmetric_transfer"),
    ("blacklist_check", detect_blacklist_check, "has_conditional_revert"),
    ("tx_origin_conditional", detect_tx_origin_conditional, "has_asymmetric_transfer"),
    ("callback_trap", detect_callback_trap, None),
    ("hidden_fee", detect_hidden_fee, "has_unusual_fee_structure"),
    ("selfdestruct", detect_selfdestruct, None),
    ("delegatecall_in_token", detect_delegatecall_in_token, None),
    ("timestamp_activation", detect_timestamp_activation, None),
    ("origin_eoa_gate", detect_origin_eoa_gate, "has_asymmetric_transfer"),
    ("obfuscated_fee", detect_obfuscated_fee, "has_unusual_fee_structure"),
]

# Minimum number of patterns that must fire to upgrade to SUSPECTED.
# Conservative: require at least 1 strong signal.
MIN_PATTERNS_FOR_SUSPECTED = 1


# =====================================================================
# Main classifier entry point
# =====================================================================

def classify(deployment: Deployment) -> dict:
    """
    Classify a deployment's bytecode. Returns a dict compatible with
    the deployment monitor's expected format:

        {
            "confidence_tier": "unknown" | "suspected",
            "confidence_reason": str,  # non-empty, with byte offsets
            "bytecode_signals": {
                "has_asymmetric_transfer": bool | None,
                "has_conditional_revert": bool | None,
                "has_unusual_fee_structure": bool | None,
                "pattern_notes": str | None,
            }
        }
    """
    bytecode = deployment.bytecode
    if not bytecode or len(bytecode) < 10:
        return {
            "confidence_tier": "unknown",
            "confidence_reason": (
                f"Contract at block {deployment.block_number} has minimal bytecode "
                f"({len(bytecode) // 2}B) — no patterns to analyze"
            ),
            "bytecode_signals": {},
        }

    # Run all detectors
    results: list[PatternResult] = []
    for name, detector, _ in PATTERN_REGISTRY:
        try:
            detected, notes = detector(bytecode)
            results.append(PatternResult(name=name, detected=detected, notes=notes))
        except Exception as e:
            logger.warning("Detector %s failed on %s: %s",
                           name, deployment.contract_address, e)
            results.append(PatternResult(name=name, detected=False, notes=""))

    fired = [r for r in results if r.detected]

    # Build bytecode_signals dict
    signals: dict = {
        "has_asymmetric_transfer": None,
        "has_conditional_revert": None,
        "has_unusual_fee_structure": None,
        "pattern_notes": None,
    }
    for name, _, signal_field in PATTERN_REGISTRY:
        if signal_field is None:
            continue
        for r in results:
            if r.name == name:
                if r.detected:
                    signals[signal_field] = True
                elif signals[signal_field] is None:
                    signals[signal_field] = False

    # Combine all fired pattern notes
    all_notes = "; ".join(r.notes for r in fired if r.notes)
    if all_notes:
        signals["pattern_notes"] = all_notes

    # Classification decision
    if len(fired) >= MIN_PATTERNS_FOR_SUSPECTED:
        pattern_names = ", ".join(r.name for r in fired)
        reason = (
            f"Bytecode exhibits {len(fired)} trap pattern(s) [{pattern_names}]: "
            f"{all_notes}"
        )
        return {
            "confidence_tier": "suspected",
            "confidence_reason": reason,
            "bytecode_signals": signals,
        }

    # No patterns fired — UNKNOWN with honest explanation
    checked_names = ", ".join(r.name for r in results)
    return {
        "confidence_tier": "unknown",
        "confidence_reason": (
            f"Bytecode analyzed ({len(bytecode) // 2}B) — "
            f"checked [{checked_names}], no trap patterns detected"
        ),
        "bytecode_signals": signals,
    }
