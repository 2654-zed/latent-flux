"""Tests for surveillance.regime_monitor.

Verifies that the first production-side flux_manifold consumer behaves
correctly. Uses an in-memory SQLite with synthetic daily-aggregate data
shaped to trigger a known changepoint.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from surveillance.regime_monitor import (
    RegimeMonitor, SignalDef, DEFAULT_HAZARD_RATE, DEFAULT_THRESHOLD,
)


@pytest.fixture
def db() -> sqlite3.Connection:
    """In-memory DB with a synthetic source table for the test signal."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE synthetic_observations (
            day   TEXT NOT NULL PRIMARY KEY,
            value REAL NOT NULL
        );
    """)
    conn.commit()
    yield conn
    conn.close()


def _seed_observations(db: sqlite3.Connection, values: list[float]) -> None:
    """Insert (day, value) starting 2026-01-01."""
    from datetime import date, timedelta
    start = date(2026, 1, 1)
    for i, v in enumerate(values):
        day = (start + timedelta(days=i)).isoformat()
        db.execute(
            "INSERT INTO synthetic_observations(day, value) VALUES (?, ?)",
            (day, v),
        )
    db.commit()


def _make_signal(prior_mu: float = 10.0, prior_var: float = 4.0) -> SignalDef:
    return SignalDef(
        name="synthetic",
        query="SELECT day, value FROM synthetic_observations ORDER BY day",
        prior_mu=prior_mu,
        prior_var=prior_var,
        notes="test signal",
    )


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_regime_monitor_detects_obvious_changepoint(db):
    """Feed 30 days of ~mean=10 noise, then 30 days of ~mean=100 noise.
    The detector must flag the regime change."""
    import random
    rng = random.Random(42)
    pre = [10.0 + rng.gauss(0, 1) for _ in range(30)]
    post = [100.0 + rng.gauss(0, 5) for _ in range(30)]
    _seed_observations(db, pre + post)

    monitor = RegimeMonitor(db, signals=[_make_signal()])
    new_alerts = monitor.scan()

    assert len(new_alerts) >= 1, \
        f"detector must flag the obvious mean-shift; got {len(new_alerts)} alerts"
    # The first alert should land somewhere in the post-window (the
    # detector lags by a few observations as it accumulates evidence)
    first_alert_day = new_alerts[0].observation_date
    assert first_alert_day >= "2026-01-31", \
        f"first changepoint should fire in post-window; got {first_alert_day}"


def test_regime_monitor_silent_on_stationary_series(db):
    """Pure noise with stable mean must NOT produce changepoints."""
    import random
    rng = random.Random(42)
    stationary = [10.0 + rng.gauss(0, 1) for _ in range(60)]
    _seed_observations(db, stationary)

    monitor = RegimeMonitor(db, signals=[_make_signal()])
    new_alerts = monitor.scan()

    # Bayesian detectors can have rare spurious fires; budget at most 1
    # over 60 observations. If our detector fires more often than that
    # on pure noise, the priors / threshold need tuning.
    assert len(new_alerts) <= 1, \
        f"stationary series should produce ≤1 alert; got {len(new_alerts)}"


def test_regime_monitor_writes_to_regime_alerts_table(db):
    """The scan() call must persist alerts to the regime_alerts table."""
    import random
    rng = random.Random(42)
    _seed_observations(
        db,
        [10.0 + rng.gauss(0, 1) for _ in range(20)]
        + [200.0 + rng.gauss(0, 5) for _ in range(20)]
    )

    monitor = RegimeMonitor(db, signals=[_make_signal()])
    new_alerts = monitor.scan()

    rows = db.execute("SELECT * FROM regime_alerts").fetchall()
    assert len(rows) == len(new_alerts), \
        "regime_alerts table row count must match returned alerts"
    assert all(row["signal_name"] == "synthetic" for row in rows)
    assert all(row["detector_threshold"] == DEFAULT_THRESHOLD for row in rows)
    assert all(row["hazard_rate"] == DEFAULT_HAZARD_RATE for row in rows)


def test_regime_monitor_is_idempotent(db):
    """Running scan() twice produces no duplicate rows (UNIQUE constraint)."""
    import random
    rng = random.Random(42)
    _seed_observations(
        db,
        [10.0 + rng.gauss(0, 1) for _ in range(20)]
        + [200.0 + rng.gauss(0, 5) for _ in range(20)]
    )

    monitor = RegimeMonitor(db, signals=[_make_signal()])
    first_run = monitor.scan()
    n_first = len(db.execute("SELECT * FROM regime_alerts").fetchall())
    assert n_first == len(first_run)

    # Second run: no new rows should be written
    second_run = monitor.scan()
    n_second = len(db.execute("SELECT * FROM regime_alerts").fetchall())
    assert n_second == n_first, \
        f"second run must produce no new rows; was {n_first}, now {n_second}"
    assert second_run == [], "second run's returned new_alerts must be empty"


def test_regime_monitor_skips_signals_with_missing_source_table(db):
    """If a signal's source table doesn't exist, the monitor must SKIP
    that signal and continue with the others — not crash the scan."""
    import random
    rng = random.Random(42)
    _seed_observations(
        db,
        [10.0 + rng.gauss(0, 1) for _ in range(20)]
        + [200.0 + rng.gauss(0, 5) for _ in range(20)]
    )

    missing_signal = SignalDef(
        name="missing_table",
        query="SELECT day, value FROM nonexistent_table",
        prior_mu=10.0,
        prior_var=4.0,
        notes="references a table that doesn't exist",
    )
    # missing_signal first, real signal second → ensures we don't bail on first error
    monitor = RegimeMonitor(db, signals=[missing_signal, _make_signal()])
    new_alerts = monitor.scan()

    # We should still get alerts from the real signal
    assert any(a.signal_name == "synthetic" for a in new_alerts), \
        "real signal must still be scanned even if a prior signal failed"
    assert not any(a.signal_name == "missing_table" for a in new_alerts), \
        "missing-table signal must produce no alerts"
