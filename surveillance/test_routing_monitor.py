"""
Tests for the 1inch routing monitor.

All HTTP calls are mocked -- no real API requests.
Tests verify: token registry cross-referencing, routing_presence updates,
routing anomaly detection logic, rate limit handling, and DB state transitions.
"""

import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surveillance import db
from surveillance.routing_monitor import RoutingMonitor, USDC_ARB


def _make_conn() -> tuple[sqlite3.Connection, Path]:
    """Create a temp DB with schema applied."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp = Path(f.name)
    f.close()
    conn = db.init_db(tmp)
    return conn, tmp


def _insert_test_contract(conn, address="0xaaa", tier="unknown",
                          routing=0, deployer="0xdep1"):
    db.insert_contract(
        conn,
        contract_address=address,
        deployer_address=deployer,
        detection_method="bytecode_pattern",
        detection_timestamp="2026-03-16T00:00:00+00:00",
        detection_block=100,
        confidence_tier=tier,
        confidence_reason=f"Test contract {address} at block 100",
    )


# =====================================================================
# Token registry cross-reference
# =====================================================================

def test_routing_hit_updates_db():
    """When a contract appears in 1inch registry, routing_presence is set."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    # Mock the token list response
    mock_tokens = {
        "0xaaa": {"symbol": "TRAP", "name": "TrapToken"},
        "0xbbb": {"symbol": "SAFE", "name": "SafeToken"},
    }

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_fetch_token_list", return_value=mock_tokens):
            await monitor._check_token_registry(mock_session)

    asyncio.run(run())

    contract = db.get_contract(conn, "0xaaa")
    assert contract["routing_presence"] == 1, "routing_presence should be 1"
    assert contract["routing_first_seen"] is not None
    assert monitor.routing_hits == 1

    conn.close()
    tmp.unlink()


def test_no_hit_leaves_routing_false():
    """Contracts not in 1inch registry keep routing_presence=0."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    # Token list doesn't contain 0xaaa
    mock_tokens = {
        "0xbbb": {"symbol": "SAFE", "name": "SafeToken"},
    }

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_fetch_token_list", return_value=mock_tokens):
            await monitor._check_token_registry(mock_session)

    asyncio.run(run())

    contract = db.get_contract(conn, "0xaaa")
    assert contract["routing_presence"] == 0
    assert monitor.routing_hits == 0

    conn.close()
    tmp.unlink()


def test_case_insensitive_matching():
    """Address matching should be case-insensitive."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")  # lowercase in DB

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    # 1inch returns checksummed (mixed-case) address
    mock_tokens = {
        "0xAAA": {"symbol": "TRAP", "name": "TrapToken"},
    }

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_fetch_token_list", return_value=mock_tokens):
            await monitor._check_token_registry(mock_session)

    asyncio.run(run())

    contract = db.get_contract(conn, "0xaaa")
    assert contract["routing_presence"] == 1

    conn.close()
    tmp.unlink()


def test_already_routed_not_rechecked():
    """Contracts already marked routing_presence=1 should be skipped."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")
    # Manually set routing
    db.update_contract_routing(conn, "0xaaa", "2026-03-15T00:00:00+00:00")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    mock_tokens = {
        "0xaaa": {"symbol": "TRAP", "name": "TrapToken"},
    }

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_fetch_token_list", return_value=mock_tokens):
            await monitor._check_token_registry(mock_session)

    asyncio.run(run())

    # Should not increment because it was already routed
    assert monitor.routing_hits == 0

    conn.close()
    tmp.unlink()


def test_token_list_failure_handled():
    """If token list fetch fails, cycle continues without crashing."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_fetch_token_list", return_value=None):
            await monitor._check_token_registry(mock_session)

    asyncio.run(run())

    contract = db.get_contract(conn, "0xaaa")
    assert contract["routing_presence"] == 0

    conn.close()
    tmp.unlink()


# =====================================================================
# Routing anomaly detection
# =====================================================================

def test_anomaly_detected_no_quote():
    """If 1inch refuses to quote a suspected+routed token, flag anomaly."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa", tier="suspected")
    db.update_contract_routing(conn, "0xaaa", "2026-03-15T00:00:00+00:00")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    async def run():
        mock_session = AsyncMock()
        # _get_quote returns None -> token is unquotable
        with patch.object(monitor, "_get_quote", return_value=None):
            await monitor._check_routing_anomalies(mock_session)

    asyncio.run(run())

    contract = db.get_contract(conn, "0xaaa")
    assert "ROUTING ANOMALY" in contract["confidence_reason"]
    assert monitor.anomalies_found == 1

    conn.close()
    tmp.unlink()


def test_no_anomaly_when_quotable():
    """If 1inch returns a valid quote, no anomaly flagged."""
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa", tier="suspected")
    db.update_contract_routing(conn, "0xaaa", "2026-03-15T00:00:00+00:00")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_get_quote", return_value={"dstAmount": "1000000"}):
            await monitor._check_routing_anomalies(mock_session)

    asyncio.run(run())

    contract = db.get_contract(conn, "0xaaa")
    assert "ROUTING ANOMALY" not in contract["confidence_reason"]
    assert monitor.anomalies_found == 0

    conn.close()
    tmp.unlink()


def test_anomaly_only_checks_suspected_routed():
    """Anomaly checks only run for SUSPECTED contracts with routing_presence."""
    conn, tmp = _make_conn()
    # Unknown tier -- should be skipped
    _insert_test_contract(conn, "0xaaa", tier="unknown")
    db.update_contract_routing(conn, "0xaaa", "2026-03-15T00:00:00+00:00")

    # Suspected but no routing -- should be skipped
    _insert_test_contract(conn, "0xbbb", tier="suspected")

    monitor = RoutingMonitor("test-key", conn, poll_interval=1)

    call_count = 0

    async def mock_quote(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return None

    async def run():
        mock_session = AsyncMock()
        with patch.object(monitor, "_get_quote", side_effect=mock_quote):
            await monitor._check_routing_anomalies(mock_session)

    asyncio.run(run())

    # Neither should have been checked
    assert call_count == 0

    conn.close()
    tmp.unlink()


# =====================================================================
# DB helper functions
# =====================================================================

def test_get_all_contract_addresses():
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")
    _insert_test_contract(conn, "0xbbb", deployer="0xdep2")

    addrs = db.get_all_contract_addresses(conn)
    assert addrs == {"0xaaa", "0xbbb"}

    conn.close()
    tmp.unlink()


def test_get_contracts_without_routing():
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa")
    _insert_test_contract(conn, "0xbbb", deployer="0xdep2")
    db.update_contract_routing(conn, "0xbbb", "2026-03-15T00:00:00+00:00")

    unrouted = db.get_contracts_without_routing(conn)
    assert len(unrouted) == 1
    assert unrouted[0]["contract_address"] == "0xaaa"

    conn.close()
    tmp.unlink()


def test_get_suspected_contracts():
    conn, tmp = _make_conn()
    _insert_test_contract(conn, "0xaaa", tier="unknown")
    _insert_test_contract(conn, "0xbbb", tier="suspected", deployer="0xdep2")
    _insert_test_contract(conn, "0xccc", tier="suspected", deployer="0xdep3")

    suspected = db.get_suspected_contracts(conn)
    assert len(suspected) == 2
    addrs = {c["contract_address"] for c in suspected}
    assert addrs == {"0xbbb", "0xccc"}

    conn.close()
    tmp.unlink()


# =====================================================================
# Run
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
