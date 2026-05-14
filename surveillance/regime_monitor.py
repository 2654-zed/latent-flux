"""Layer 3 regime monitor — operational-tempo changepoint detection.

The first production-side consumer of `flux_manifold` primitives. Feeds
daily aggregate signals from the surveillance corpus through a Bayesian
online changepoint detector (`flux_manifold.changepoint.BayesianChangePoint`)
and records detected regime shifts to the `regime_alerts` table.

Rationale
─────────
Layer 3's existing detectors operate per-contract: bytecode_classifier on
deployment, OLI guardrail on classification, drainer-spawn-hub iteration
tracking. None of them surface *operational-tempo* changes — when an
operator pauses, when a new pattern emerges in the wave, when victim
pools shift.

These signals are visible only at the daily-aggregate level. Regime
monitor adds that surface.

This module also resolves the long-standing UNK-024: the README has
claimed "Latent Flux primitives power Layer 3's analysis layer" since
2026-04-16 without that claim being backed by any code. Regime monitor
is the first concrete instantiation of that claim. See
`memory/UNKNOWNS.md` UNK-024 RESOLVED and `memory/DECISIONS.md` ADR-006
for context.

Signals (V1)
────────────
- new_deployers_total           : COUNT(*) FROM deployers GROUP BY DATE(first_seen)
- confirmed_traps_per_day       : COUNT(*) FROM contracts WHERE confidence_tier='confirmed' GROUP BY DATE
- suspected_traps_per_day       : COUNT(*) FROM contracts WHERE confidence_tier='suspected' GROUP BY DATE
- watchlist_promotions_per_day  : COUNT(*) FROM watchlist GROUP BY DATE(added_date)
- approval_events_per_day       : COUNT(*) FROM approval_watchlist GROUP BY DATE(approve_timestamp)
- trap_event_victims_per_day    : COUNT(DISTINCT bot_address) FROM trap_events GROUP BY DATE(timestamp)

Each signal gets its own BayesianChangePoint instance. Currently they
share one hazard_rate (1/30 = expected 30-day regime persistence) — this
is a tunable. Per-signal priors are seeded from corpus history.

Usage
─────
    # One-shot: scan corpus, write new alerts, print summary
    python -m surveillance.regime_monitor

    # Programmatic
    from surveillance.regime_monitor import RegimeMonitor
    rm = RegimeMonitor(conn)
    new_alerts = rm.scan()
    for alert in new_alerts:
        print(alert)

Idempotency
───────────
The `regime_alerts` table has a UNIQUE(signal_name, observation_date)
constraint. Re-running on already-scanned data produces no duplicates;
INSERT OR IGNORE silently skips. Safe to schedule daily.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from flux_manifold.changepoint import BayesianChangePoint


# ─────────────────────────────────────────────────────────────────────
# Signal definitions
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalDef:
    """A daily-aggregate signal definition.

    Attributes:
        name: machine-readable identifier (stored in regime_alerts.signal_name)
        query: SQL that returns (observation_date, observed_value) rows
        prior_mu: Normal-Gamma prior on signal mean
        prior_var: Normal-Gamma prior on signal variance (1/precision)
        notes: human-readable description
    """
    name: str
    query: str
    prior_mu: float
    prior_var: float
    notes: str


# Default hazard rate: 1/30 → expected regime persistence ~30 days.
# Higher (e.g. 1/7) makes the detector more eager; lower (1/90) more conservative.
DEFAULT_HAZARD_RATE = 1.0 / 30.0

# Default detection threshold: declare changepoint when P(CP) > 0.5
DEFAULT_THRESHOLD = 0.5


# Daily-aggregate signals. Priors are seeded from rough corpus averages;
# they self-adjust quickly under Bayesian updates so exact values don't
# matter much beyond ensuring numerical stability.
SIGNALS: list[SignalDef] = [
    SignalDef(
        name="new_deployers_total",
        query=(
            "SELECT DATE(first_seen) AS day, COUNT(*) AS n "
            "FROM deployers "
            "WHERE first_seen IS NOT NULL "
            "GROUP BY day "
            "ORDER BY day"
        ),
        prior_mu=200.0,
        prior_var=10000.0,
        notes="Count of new deployers per day across all chains.",
    ),
    SignalDef(
        name="confirmed_traps_per_day",
        query=(
            "SELECT DATE(detection_timestamp) AS day, COUNT(*) AS n "
            "FROM contracts "
            "WHERE confidence_tier='confirmed' "
            "GROUP BY day "
            "ORDER BY day"
        ),
        prior_mu=5.0,
        prior_var=25.0,
        notes="Confirmed-tier trap detections per day.",
    ),
    SignalDef(
        name="suspected_traps_per_day",
        query=(
            "SELECT DATE(detection_timestamp) AS day, COUNT(*) AS n "
            "FROM contracts "
            "WHERE confidence_tier='suspected' "
            "GROUP BY day "
            "ORDER BY day"
        ),
        prior_mu=300.0,
        prior_var=10000.0,
        notes="Suspected-tier trap detections per day.",
    ),
    SignalDef(
        name="watchlist_additions_per_day",
        query=(
            "SELECT DATE(added_date) AS day, COUNT(*) AS n "
            "FROM watchlist "
            "WHERE added_date IS NOT NULL "
            "GROUP BY day "
            "ORDER BY day"
        ),
        prior_mu=2.0,
        prior_var=10.0,
        notes="New watchlist promotions per day.",
    ),
    SignalDef(
        name="approval_events_per_day",
        query=(
            "SELECT DATE(approve_timestamp) AS day, COUNT(*) AS n "
            "FROM approval_watchlist "
            "WHERE approve_timestamp IS NOT NULL "
            "GROUP BY day "
            "ORDER BY day"
        ),
        prior_mu=1000.0,
        prior_var=250000.0,
        notes="New approval-watchlist entries per day (victim approvals).",
    ),
    SignalDef(
        name="trap_event_victims_per_day",
        query=(
            "SELECT DATE(timestamp) AS day, COUNT(DISTINCT bot_address) AS n "
            "FROM trap_events "
            "WHERE timestamp IS NOT NULL "
            "GROUP BY day "
            "ORDER BY day"
        ),
        prior_mu=20.0,
        prior_var=400.0,
        notes="Unique trapped bot addresses per day (victims appearing in trap_events).",
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RegimeAlert:
    """One detected changepoint. Mirrors the regime_alerts table schema."""
    signal_name: str
    observation_date: str
    observed_value: float
    cp_probability: float
    detector_threshold: float
    hazard_rate: float
    notes: str = ""


class RegimeMonitor:
    """Scan corpus daily-aggregate signals for Bayesian changepoints.

    The monitor is stateless across runs: each `scan()` replays all
    historical observations through fresh BayesianChangePoint detectors,
    then writes any new (per UNIQUE-constraint) alerts.

    Tradeoff:
      + No persistence complexity (the BCP internal state is non-trivial)
      + Re-runs are idempotent (UNIQUE constraint blocks dupes)
      + If we improve the detector, prior history gets reanalyzed automatically
      - Cost grows linearly with history length (negligible for daily ticks
        over weeks; tolerable for months; would need optimization at years)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        hazard_rate: float = DEFAULT_HAZARD_RATE,
        threshold: float = DEFAULT_THRESHOLD,
        signals: list[SignalDef] | None = None,
    ):
        self.conn = conn
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        self.signals = signals if signals is not None else SIGNALS

    def _stream_observations(self, signal: SignalDef) -> Iterator[tuple[str, float]]:
        """Yield (date, value) tuples for this signal, in chronological order.

        Uses positional access so callers can name columns whatever they like.
        Contract: the signal's query must return exactly (date, value).
        """
        cursor = self.conn.execute(signal.query)
        for row in cursor:
            day = row[0]
            value = row[1]
            if day is None:
                continue
            yield (str(day), float(value))

    def _scan_one_signal(self, signal: SignalDef) -> list[RegimeAlert]:
        """Process one signal's full history; return detected changepoints."""
        prior_kappa = 1.0  # 1 pseudo-observation for the mean
        prior_alpha = 1.0  # weakly-informative Gamma shape
        # prior_var = beta/(alpha*kappa) → beta = prior_var * alpha * kappa
        prior_beta = max(signal.prior_var * prior_alpha * prior_kappa, 1e-6)

        detector = BayesianChangePoint(
            hazard_rate=self.hazard_rate,
            threshold=self.threshold,
            prior_mu=signal.prior_mu,
            prior_kappa=prior_kappa,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            max_run=500,
        )

        alerts: list[RegimeAlert] = []
        for day, value in self._stream_observations(signal):
            # update() returns raw P(r<10) which is naturally near 1.0
            # during burn-in (before the detector has ~20 observations
            # of history). is_changepoint() applies the full gating:
            #   - MAP run length < 10 (regime is "young")
            #   - t > 20 (enough history accumulated)
            #   - cp_prob > threshold
            # See flux_manifold/changepoint.py:164.
            prob = detector.update(value)
            if detector.is_changepoint(threshold=self.threshold):
                alerts.append(RegimeAlert(
                    signal_name=signal.name,
                    observation_date=day,
                    observed_value=value,
                    cp_probability=prob,
                    detector_threshold=self.threshold,
                    hazard_rate=self.hazard_rate,
                    notes=signal.notes,
                ))
        return alerts

    def scan(self) -> list[RegimeAlert]:
        """Process all signals; write detected alerts to regime_alerts table.

        Returns the list of alerts WRITTEN this run (excludes those that
        were already present in the table due to the UNIQUE constraint).
        """
        # Ensure the regime_alerts table exists. If init_db has been run,
        # this is a no-op; if running against a fresh in-memory DB (e.g.
        # tests), create it inline.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_alerts (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name        TEXT    NOT NULL,
                observation_date   TEXT    NOT NULL,
                observed_value     REAL    NOT NULL,
                cp_probability     REAL    NOT NULL,
                detector_threshold REAL    NOT NULL,
                hazard_rate        REAL    NOT NULL,
                detected_at        TEXT    NOT NULL,
                notes              TEXT,
                UNIQUE(signal_name, observation_date)
            )
        """)

        all_alerts: list[RegimeAlert] = []
        new_alerts: list[RegimeAlert] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for signal in self.signals:
            try:
                signal_alerts = self._scan_one_signal(signal)
            except sqlite3.OperationalError as e:
                # Most likely cause: source table doesn't exist in this DB.
                # Log and skip the signal rather than failing the whole scan.
                print(f"  [skip] {signal.name}: {e}", file=sys.stderr)
                continue
            all_alerts.extend(signal_alerts)
            for alert in signal_alerts:
                cursor = self.conn.execute(
                    """INSERT OR IGNORE INTO regime_alerts
                       (signal_name, observation_date, observed_value,
                        cp_probability, detector_threshold, hazard_rate,
                        detected_at, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (alert.signal_name, alert.observation_date,
                     alert.observed_value, alert.cp_probability,
                     alert.detector_threshold, alert.hazard_rate,
                     now_iso, alert.notes),
                )
                if cursor.rowcount > 0:
                    new_alerts.append(alert)

        self.conn.commit()
        return new_alerts

    def list_alerts(self, signal_name: str | None = None) -> list[dict]:
        """Read alerts from the table. Optional filter by signal_name."""
        if signal_name:
            cursor = self.conn.execute(
                "SELECT * FROM regime_alerts WHERE signal_name = ? "
                "ORDER BY observation_date DESC",
                (signal_name,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM regime_alerts ORDER BY observation_date DESC"
            )
        return [dict(row) for row in cursor]


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH),
                    help="Path to surveillance.db")
    ap.add_argument("--hazard-rate", type=float, default=DEFAULT_HAZARD_RATE,
                    help="Bayesian hazard rate (1/expected_run_length); "
                         f"default {DEFAULT_HAZARD_RATE:.4f}")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Changepoint probability threshold; default {DEFAULT_THRESHOLD}")
    ap.add_argument("--list", action="store_true",
                    help="Print existing alerts table contents and exit")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    monitor = RegimeMonitor(conn, hazard_rate=args.hazard_rate,
                             threshold=args.threshold)

    if args.list:
        alerts = monitor.list_alerts()
        print(f"regime_alerts: {len(alerts)} total")
        for a in alerts[:50]:
            print(f"  {a['observation_date']} {a['signal_name']:32s} "
                  f"value={a['observed_value']:>10.1f} "
                  f"P(CP)={a['cp_probability']:.3f}")
        return 0

    print(f"Scanning {len(monitor.signals)} signals from {db_path}...",
          file=sys.stderr)
    new_alerts = monitor.scan()
    print(f"  {len(new_alerts)} new regime alerts written.", file=sys.stderr)
    for a in new_alerts:
        print(f"  ALERT  {a.observation_date} {a.signal_name:32s} "
              f"value={a.observed_value:>10.1f} P(CP)={a.cp_probability:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
