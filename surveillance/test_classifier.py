"""
Tests for the bytecode classifier.

Each test constructs a minimal EVM bytecode hex string containing
the exact opcode pattern a detector looks for, then verifies detection.
Also tests that clean bytecode does NOT trigger false positives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import Optional

from surveillance.bytecode_classifier import (
    _find_all_offsets,
    _opcodes_near,
    detect_asymmetric_transfer,
    detect_blacklist_check,
    detect_tx_origin_conditional,
    detect_callback_trap,
    detect_hidden_fee,
    detect_selfdestruct,
    detect_delegatecall_in_token,
    classify,
    SEL_TRANSFER,
    SEL_TRANSFER_FROM,
    SEL_UNISWAP_V2_CALL,
    OP_CALLER,
    OP_ORIGIN,
    OP_EQ,
    OP_JUMPI,
    OP_REVERT,
    OP_SLOAD,
    OP_SSTORE,
    OP_SELFDESTRUCT,
    OP_DELEGATECALL,
    OP_CALL,
)


# Helper: mock Deployment
@dataclass
class MockDeployment:
    contract_address: str = "0xtest"
    deployer_address: str = "0xdeployer"
    block_number: int = 100
    timestamp_iso: str = "2026-03-16T00:00:00+00:00"
    tx_hash: str = "0xabc"
    bytecode: str = ""
    gas_price_gwei: Optional[float] = None


def pad(hex_str: str, total_bytes: int = 0) -> str:
    """Pad hex string with 00s (STOP opcode) for alignment."""
    if total_bytes > 0:
        target_len = total_bytes * 2
        if len(hex_str) < target_len:
            hex_str += "00" * ((target_len - len(hex_str)) // 2)
    return hex_str


# =====================================================================
# _find_all_offsets
# =====================================================================

def test_find_all_offsets_basic():
    # bytecode: STOP CALLER STOP CALLER STOP
    bc = "00" + "33" + "00" + "33" + "00"
    offsets = _find_all_offsets(bc, "33")
    assert offsets == [1, 3], f"Expected [1, 3], got {offsets}"


def test_find_all_offsets_skips_push_data():
    # PUSH1 0x33 should NOT be detected as CALLER
    # 0x60 = PUSH1, followed by 1 byte of data (0x33)
    bc = "60" + "33" + "00" + "33"  # PUSH1 0x33, STOP, CALLER
    offsets = _find_all_offsets(bc, "33")
    assert offsets == [3], f"Expected [3], got {offsets}"


def test_find_all_offsets_push32():
    # PUSH32 (0x7f) followed by 32 bytes of data
    bc = "7f" + "33" * 32 + "33"  # PUSH32 <32 bytes of 0x33>, then real CALLER
    offsets = _find_all_offsets(bc, "33")
    assert offsets == [33], f"Expected [33], got {offsets}"


# =====================================================================
# detect_asymmetric_transfer
# =====================================================================

def test_asymmetric_transfer_detected():
    # Build: transfer selector ... CALLER EQ JUMPI ... REVERT
    bc = SEL_TRANSFER + "00" * 10 + OP_CALLER + "00" * 3 + OP_EQ + "00" * 2 + OP_JUMPI + "00" * 10 + OP_REVERT + "00" * 20
    detected, notes = detect_asymmetric_transfer(bc)
    assert detected, f"Expected detection, got notes: {notes}"
    assert "CALLER" in notes
    assert "REVERT" in notes


def test_asymmetric_transfer_no_transfer_selector():
    # Same pattern but no transfer selector — should not fire
    bc = "00" * 10 + OP_CALLER + "00" * 3 + OP_EQ + "00" * 2 + OP_JUMPI + "00" * 10 + OP_REVERT + "00" * 20
    detected, notes = detect_asymmetric_transfer(bc)
    assert not detected


def test_asymmetric_transfer_clean_erc20():
    # A simple bytecode with transfer selector but no CALLER→EQ→JUMPI→REVERT chain
    bc = SEL_TRANSFER + "00" * 200
    detected, notes = detect_asymmetric_transfer(bc)
    assert not detected


# =====================================================================
# detect_blacklist_check
# =====================================================================

def test_blacklist_detected():
    # CALLER → SLOAD → JUMPI → REVERT
    bc = OP_CALLER + "00" * 5 + OP_SLOAD + "00" * 3 + OP_JUMPI + "00" * 10 + OP_REVERT + "00" * 20
    detected, notes = detect_blacklist_check(bc)
    assert detected, f"Expected detection, got notes: {notes}"
    assert "blacklist" in notes.lower() or "SLOAD" in notes


def test_blacklist_with_origin():
    # ORIGIN → SLOAD → JUMPI → REVERT
    bc = OP_ORIGIN + "00" * 5 + OP_SLOAD + "00" * 3 + OP_JUMPI + "00" * 10 + OP_REVERT + "00" * 20
    detected, notes = detect_blacklist_check(bc)
    assert detected


def test_blacklist_clean():
    bc = "00" * 200
    detected, notes = detect_blacklist_check(bc)
    assert not detected


# =====================================================================
# detect_tx_origin_conditional
# =====================================================================

def test_tx_origin_detected():
    # Transfer selector + ORIGIN → EQ → JUMPI
    bc = SEL_TRANSFER + "00" * 10 + OP_ORIGIN + "00" * 3 + OP_EQ + "00" * 2 + OP_JUMPI + "00" * 20
    detected, notes = detect_tx_origin_conditional(bc)
    assert detected
    assert "ORIGIN" in notes


def test_tx_origin_no_transfer():
    bc = "00" * 10 + OP_ORIGIN + "00" * 3 + OP_EQ + "00" * 2 + OP_JUMPI + "00" * 20
    detected, notes = detect_tx_origin_conditional(bc)
    assert not detected


# =====================================================================
# detect_callback_trap
# =====================================================================

def test_callback_trap_detected():
    # uniswapV2Call selector, followed by REVERT but no CALL
    bc = SEL_UNISWAP_V2_CALL + "00" * 20 + OP_REVERT + "00" * 100
    detected, notes = detect_callback_trap(bc)
    assert detected
    assert "uniswapV2Call" in notes


def test_callback_with_repayment():
    # uniswapV2Call selector, followed by CALL (legitimate repayment)
    bc = SEL_UNISWAP_V2_CALL + "00" * 20 + OP_CALL + "00" * 20 + OP_REVERT + "00" * 100
    detected, notes = detect_callback_trap(bc)
    assert not detected, "Should not flag callback with outbound CALL"


def test_callback_no_selector():
    bc = "00" * 200 + OP_REVERT
    detected, notes = detect_callback_trap(bc)
    assert not detected


# =====================================================================
# detect_hidden_fee
# =====================================================================

def test_hidden_fee_detected():
    # Transfer selector + very high SSTORE/SLOAD density
    # 50 bytes with 5 SSTOREs and 7 SLOADs
    bc = SEL_TRANSFER
    for _ in range(7):
        bc += OP_SLOAD + "00"
    for _ in range(5):
        bc += OP_SSTORE + "00"
    # Keep it short to push density high
    detected, notes = detect_hidden_fee(bc)
    assert detected, f"Expected detection with high storage density"
    assert "SSTORE" in notes or "density" in notes.lower()


def test_hidden_fee_normal_erc20():
    # Transfer selector + low SSTORE density (normal ERC-20)
    bc = SEL_TRANSFER + "00" * 400 + OP_SSTORE + "00" * 50 + OP_SSTORE + "00" * 50
    detected, notes = detect_hidden_fee(bc)
    assert not detected, "Normal ERC-20 should not trigger hidden fee"


# =====================================================================
# detect_selfdestruct
# =====================================================================

def test_selfdestruct_detected():
    bc = SEL_TRANSFER + "00" * 50 + OP_SELFDESTRUCT + "00" * 20
    detected, notes = detect_selfdestruct(bc)
    assert detected
    assert "SELFDESTRUCT" in notes


def test_selfdestruct_no_transfer():
    # SELFDESTRUCT in non-token contract — not flagged
    bc = "00" * 50 + OP_SELFDESTRUCT + "00" * 20
    detected, notes = detect_selfdestruct(bc)
    assert not detected


# =====================================================================
# detect_delegatecall_in_token
# =====================================================================

def test_delegatecall_detected():
    bc = SEL_TRANSFER + "00" * 50 + OP_DELEGATECALL + "00" * 20
    detected, notes = detect_delegatecall_in_token(bc)
    assert detected
    assert "DELEGATECALL" in notes


def test_delegatecall_no_transfer():
    bc = "00" * 50 + OP_DELEGATECALL + "00" * 20
    detected, notes = detect_delegatecall_in_token(bc)
    assert not detected


# =====================================================================
# classify (full pipeline)
# =====================================================================

def test_classify_suspected():
    # Build bytecode that triggers asymmetric_transfer
    bc = SEL_TRANSFER + "00" * 10 + OP_CALLER + "00" * 3 + OP_EQ + "00" * 2 + OP_JUMPI + "00" * 10 + OP_REVERT + "00" * 50
    dep = MockDeployment(bytecode=bc)
    result = classify(dep)
    assert result["confidence_tier"] == "suspected"
    assert len(result["confidence_reason"]) > 0
    assert "asymmetric_transfer" in result["confidence_reason"]
    assert result["bytecode_signals"]["has_asymmetric_transfer"] is True


def test_classify_unknown_clean():
    bc = "00" * 200
    dep = MockDeployment(bytecode=bc)
    result = classify(dep)
    assert result["confidence_tier"] == "unknown"
    assert "no trap patterns" in result["confidence_reason"]


def test_classify_unknown_empty():
    dep = MockDeployment(bytecode="")
    result = classify(dep)
    assert result["confidence_tier"] == "unknown"
    assert "minimal bytecode" in result["confidence_reason"]


def test_classify_multiple_patterns():
    # Trigger both tx_origin and selfdestruct
    bc = (
        SEL_TRANSFER + "00" * 10
        + OP_ORIGIN + "00" * 3 + OP_EQ + "00" * 2 + OP_JUMPI + "00" * 20
        + OP_SELFDESTRUCT + "00" * 50
    )
    dep = MockDeployment(bytecode=bc)
    result = classify(dep)
    assert result["confidence_tier"] == "suspected"
    # Should mention multiple patterns
    reason = result["confidence_reason"]
    assert "tx_origin" in reason or "selfdestruct" in reason


def test_classify_confidence_reason_never_empty():
    """Every classification must produce a non-empty confidence_reason."""
    test_cases = [
        "",           # empty
        "00",         # minimal
        "00" * 100,   # clean
        SEL_TRANSFER + OP_CALLER + OP_EQ + OP_JUMPI + OP_REVERT + "00" * 50,  # suspected
    ]
    for bc in test_cases:
        dep = MockDeployment(bytecode=bc)
        result = classify(dep)
        assert len(result["confidence_reason"]) > 0, (
            f"Empty confidence_reason for bytecode len {len(bc)}"
        )


# =====================================================================
# Run all tests
# =====================================================================

if __name__ == "__main__":
    import inspect

    tests = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        sys.exit(1)
