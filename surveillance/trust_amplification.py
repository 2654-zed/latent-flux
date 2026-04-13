"""Trust Amplification — identifies contracts whose traffic is router-dominated.

For contracts with 50+ callers, measures what percentage of calls route through
selector 3593564c (Uniswap Universal Router execute()). Contracts with high
router dominance (>80%) are using trusted infrastructure as their victim
delivery mechanism — the router IS the attack surface.

Compares callers/day against bytecode family average to compute an
amplification factor. Generates TRUST_AMPLIFICATION alerts for contracts
crossing the router dominance threshold.

Can run as:
  - One-shot: python -m surveillance.trust_amplification --analyze
  - Periodic: called from deployment_monitor heartbeat loop
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

ROUTER_SELECTOR = "3593564c"

# Known router/aggregator contract addresses. When these show up as
# interacting_address, callers arrived via routing infrastructure.
# NOTE: As of 2026-04-11, router addresses do NOT appear as
# interacting_address in transaction_events because the selector_monitor
# records tx.from (the bot/EOA), not the intermediate router. The
# router_callers count is therefore always 0 in the current pipeline.
#
# The REAL signal for trust amplification is CALLER DIVERSITY:
# router-delivered contracts have many one-time callers (retail users
# routed by the aggregator), while bot-targeted contracts have few
# callers calling many times. The caller_diversity_ratio
# (distinct_callers / total_events) captures this without needing
# internal transaction tracing.
ROUTER_ADDRESSES = {
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 Router
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 Router2
}

# Caller diversity threshold. Ratio of distinct_callers / total_events.
# High ratio (>0.5) = many unique callers, each interacting once
# (retail-like, router-delivered). Low ratio (<0.1) = few callers
# making many calls (bot-like, direct interaction).
CALLER_DIVERSITY_THRESHOLD = 0.5

# Router dominance threshold — contracts above this are flagged.
# Currently unused because router_callers is always 0 in the pipeline.
# Kept for future use when internal-call tracing is added.
ROUTER_DOMINANCE_THRESHOLD = 80.0

# Minimum callers to be considered (avoids noise on low-traffic contracts)
MIN_CALLERS = 50

logger = logging.getLogger("surveillance.trust_amplification")


def get_connection() -> sqlite3.Connection:
    """Return a connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create trust_amplification table if missing. Idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trust_amplification (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address            TEXT NOT NULL UNIQUE,
            total_callers               INTEGER,
            router_callers              INTEGER,
            router_percentage           REAL,
            callers_per_day             REAL,
            bytecode_family             TEXT,
            family_avg_callers_per_day  REAL,
            amplification_factor        REAL,
            revert_rate                 REAL,
            first_seen                  TEXT,
            last_updated                TEXT,
            alert_level                 TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ta_router_pct
            ON trust_amplification(router_percentage);
        CREATE INDEX IF NOT EXISTS idx_ta_alert
            ON trust_amplification(alert_level);
    """)
    conn.commit()


def _get_family_avg(conn: sqlite3.Connection, contract_address: str) -> Optional[float]:
    """Get the avg callers/day for the bytecode family this contract belongs to."""
    fam = conn.execute(
        "SELECT family_id FROM bytecode_family_members WHERE contract_address = ?",
        (contract_address,)
    ).fetchone()
    if not fam:
        return None

    siblings = conn.execute("""
        SELECT bfm.contract_address
        FROM bytecode_family_members bfm
        WHERE bfm.family_id = ?
    """, (fam["family_id"],)).fetchall()

    if not siblings:
        return None

    total_cpd = 0.0
    count = 0
    for s in siblings:
        row = conn.execute("""
            SELECT COUNT(DISTINCT interacting_address) as callers,
                   JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp)) as span_days
            FROM transaction_events
            WHERE contract_address = ?
        """, (s["contract_address"],)).fetchone()
        if row and row["callers"] and row["span_days"] and row["span_days"] > 0:
            total_cpd += row["callers"] / row["span_days"]
            count += 1

    return round(total_cpd / count, 2) if count else None


def analyze(conn: sqlite3.Connection, emit_alerts: bool = True,
            quiet: bool = False) -> dict:
    """Analyze trust amplification for all qualifying contracts.

    Returns a summary dict. When emit_alerts=True, generates
    TRUST_AMPLIFICATION alerts for contracts with router_percentage
    above ROUTER_DOMINANCE_THRESHOLD.
    """
    ensure_table(conn)
    if not quiet:
        print("[trust_amplification] Finding contracts with 50+ callers...")

    contracts = conn.execute("""
        SELECT contract_address,
               COUNT(DISTINCT interacting_address) as total_callers,
               COUNT(*) as total_tx,
               JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp)) as span_days,
               ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100, 1) as revert_rate,
               MIN(timestamp) as first_seen
        FROM transaction_events
        GROUP BY contract_address
        HAVING COUNT(DISTINCT interacting_address) >= 50
    """).fetchall()

    if not quiet:
        print(f"  Found {len(contracts)} contracts with 50+ callers")

    conn.execute("DELETE FROM trust_amplification")
    inserted = 0
    router_dominated = 0
    alerts_emitted = 0

    for c in contracts:
        addr = c["contract_address"]
        # Count calls through router selector
        router_row = conn.execute("""
            SELECT COUNT(*) as router_calls,
                   COUNT(DISTINCT interacting_address) as router_callers
            FROM transaction_events
            WHERE contract_address = ? AND function_selector = ?
        """, (addr, ROUTER_SELECTOR)).fetchone()

        router_callers = router_row["router_callers"] if router_row else 0
        router_pct = round(router_callers / c["total_callers"] * 100, 1) if c["total_callers"] else 0

        span = max(c["span_days"] or 1, 1)
        callers_per_day = round(c["total_callers"] / span, 2)

        # Get family avg
        family_avg = _get_family_avg(conn, addr)

        # Look up family
        try:
            fam_row = conn.execute(
                "SELECT family_id FROM bytecode_family_members WHERE contract_address = ?",
                (addr,)
            ).fetchone()
            family_id = fam_row["family_id"] if fam_row else None
        except sqlite3.OperationalError:
            family_id = None

        if family_avg and family_avg > 0:
            amplification = round(callers_per_day / family_avg, 2)
        else:
            amplification = 1.0
            family_avg = callers_per_day  # self-baseline

        # Caller diversity ratio — the proxy for router-mediated delivery.
        # High ratio = many distinct one-time callers (retail traffic).
        # Low ratio = few bot callers making repeat calls.
        caller_diversity = round(c["total_callers"] / c["total_tx"], 4) if c["total_tx"] else 0

        # Alert level — three independent dimensions:
        # 1. Amplification factor (vs family baseline)
        # 2. Router dominance (% of callers via Universal Router — currently always 0)
        # 3. Caller diversity (high = retail-like, router-delivered population)
        if amplification > 10 and c["total_callers"] >= 100:
            alert = "CRITICAL"
        elif amplification > 5 and c["total_callers"] >= 50:
            alert = "WARNING"
        elif router_pct >= ROUTER_DOMINANCE_THRESHOLD and c["total_callers"] >= MIN_CALLERS:
            alert = "WARNING"
        elif (caller_diversity >= CALLER_DIVERSITY_THRESHOLD
              and c["total_callers"] >= MIN_CALLERS
              and c["revert_rate"] is not None
              and c["revert_rate"] < 10):
            # High caller diversity + low revert rate + many callers =
            # looks like retail traffic hitting a camouflaged trap.
            # This is the trust amplification signal without needing
            # internal transaction tracing.
            alert = "WARNING"
        elif amplification > 2:
            alert = "INFO"
        else:
            alert = None

        conn.execute("""
            INSERT OR REPLACE INTO trust_amplification
                (contract_address, total_callers, router_callers, router_percentage,
                 callers_per_day, bytecode_family, family_avg_callers_per_day,
                 amplification_factor, revert_rate, first_seen, last_updated, alert_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
        """, (addr, c["total_callers"], router_callers, router_pct,
              callers_per_day, family_id, family_avg, amplification,
              c["revert_rate"], c["first_seen"], alert))
        inserted += 1

        if router_pct >= ROUTER_DOMINANCE_THRESHOLD or caller_diversity >= CALLER_DIVERSITY_THRESHOLD:
            router_dominated += 1

        # Emit alert for trust-amplified contracts (router-dominated OR
        # high caller diversity with low revert rate)
        if (emit_alerts and alert in ("CRITICAL", "WARNING")
                and (router_pct >= ROUTER_DOMINANCE_THRESHOLD
                     or caller_diversity >= CALLER_DIVERSITY_THRESHOLD)):
            try:
                payload = json.dumps({
                    "contract": addr,
                    "router_percentage": router_pct,
                    "total_callers": c["total_callers"],
                    "callers_per_day": callers_per_day,
                    "amplification_factor": amplification,
                    "revert_rate": c["revert_rate"],
                    "alert_level": alert,
                    "epistemic_tag": "assessed",
                    "message": (
                        f"Trust amplification: {addr} has "
                        f"{caller_diversity:.0%} caller diversity, "
                        f"{router_pct}% router selector dominance, "
                        f"{c['revert_rate']:.1f}% revert rate across "
                        f"{c['total_callers']:,} callers."
                    ),
                })
                # Check if alert already exists for this contract
                existing = conn.execute(
                    "SELECT 1 FROM alerts WHERE alert_type = 'TRUST_AMPLIFICATION' "
                    "AND address = ?", (addr,)
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO alerts (alert_type, address, timestamp, "
                        "payload, false_positive) VALUES (?, ?, datetime('now'), ?, 0)",
                        ("TRUST_AMPLIFICATION", addr, payload),
                    )
                    alerts_emitted += 1
                    logger.warning(
                        "TRUST_AMPLIFICATION: %s router=%s%% callers=%s",
                        addr, router_pct, c["total_callers"],
                    )
            except Exception as e:
                logger.debug("trust_amplification alert failed: %s", e)

        if alert and not quiet:
            print(f"  [{alert}] {addr} amp={amplification}x "
                  f"router={router_pct}% callers={c['total_callers']}")

    conn.commit()

    summary = {
        "contracts_analyzed": inserted,
        "router_dominated": router_dominated,
        "alerts_emitted": alerts_emitted,
    }

    if not quiet:
        print(f"[trust_amplification] Done. {inserted} contracts analyzed, "
              f"{router_dominated} router-dominated (>{ROUTER_DOMINANCE_THRESHOLD}%), "
              f"{alerts_emitted} alerts emitted.")
        for level in ("CRITICAL", "WARNING", "INFO"):
            cnt = conn.execute(
                "SELECT COUNT(*) FROM trust_amplification WHERE alert_level = ?", (level,)
            ).fetchone()[0]
            if cnt:
                print(f"  {level}: {cnt}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trust Amplification Analyzer")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    args = parser.parse_args()

    if args.analyze:
        conn = get_connection()
        analyze(conn)
        conn.close()
    else:
        parser.print_help()
