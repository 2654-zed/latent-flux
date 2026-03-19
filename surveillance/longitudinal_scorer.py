"""
Longitudinal Behavioral Scoring — The Card Counter Detector

Scores deployers based on accumulated behavioral signals over time,
not single-event thresholds. Designed to catch patient operators who
stay below every individual detection threshold but show systematic
extraction patterns in aggregate.

Score range: 0.0 (benign) to 1.0+ (strong evidence of systematic extraction)
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --------------- Scoring weights ---------------

# Positive signals (suspicious)
W_MULTI_SESSION = 0.3        # Deployed in 3+ distinct calendar days
W_USDC_FLOW = 0.2            # Any USDC flow to/from deployed contracts
W_RETURN_RATIO = 0.3         # Withdrew more USDC than deposited
W_PATIENCE = 0.2             # Active across >7 days
W_TEMPLATE_REUSE = 0.2       # Same bytecode template in 3+ contracts
W_BOT_INTERACTION = 0.1      # Contracts attracted bot activity
W_GAS_STATION_FUNDING = 0.2  # Funded by high-nonce gas station

# Negative signals (likely legitimate)
W_ZERO_FLOWS = -0.3          # No financial flows at all
W_SINGLE_SESSION = -0.2      # Active in only 1 day
W_VERIFIED = -0.2            # Has verified/proxy contracts
W_IRREGULAR_TIMING = -0.1    # Long gaps, looks experimental


@dataclass
class ScoreBreakdown:
    """Tracks which signals fired and their contribution."""
    signals: dict = field(default_factory=dict)
    raw_score: float = 0.0

    def add(self, name: str, weight: float, detail: str = ""):
        self.signals[name] = {"weight": weight, "detail": detail}
        self.raw_score += weight

    @property
    def final_score(self) -> float:
        return max(0.0, min(1.5, self.raw_score))

    def to_json(self) -> str:
        return json.dumps({
            "score": round(self.final_score, 3),
            "signals": self.signals,
        }, default=str)


def _distinct_active_days(conn: sqlite3.Connection, deployer: str) -> int:
    """Count distinct calendar days this deployer had contract deployments."""
    row = conn.execute(
        """SELECT COUNT(DISTINCT DATE(detection_timestamp)) as days
           FROM contracts WHERE deployer_address = ?""",
        (deployer,),
    ).fetchone()
    return row[0] if row else 0


def _activity_span_days(conn: sqlite3.Connection, deployer: str) -> float:
    """Days between first and last seen."""
    row = conn.execute(
        "SELECT julianday(last_seen) - julianday(first_seen) FROM deployers WHERE deployer_address = ?",
        (deployer,),
    ).fetchone()
    return row[0] if row and row[0] else 0.0


def _has_usdc_flows(conn: sqlite3.Connection, deployer: str) -> tuple[bool, float, float]:
    """
    Check if any contracts from this deployer have USDC-related transaction events.
    Returns (has_flow, total_in_estimate, total_out_estimate).

    Since we don't track token amounts directly, we use transaction_events
    as a proxy: approve (095ea7b3) and transfer (a9059cbb) selectors
    on this deployer's contracts indicate token flow.
    """
    # Count approve/transfer events on this deployer's contracts
    row = conn.execute(
        """SELECT
            COUNT(CASE WHEN te.function_selector IN ('095ea7b3','a9059cbb','23b872dd') THEN 1 END) as token_events,
            COUNT(*) as total_events
           FROM transaction_events te
           JOIN contracts c ON te.contract_address = c.contract_address
           WHERE c.deployer_address = ?""",
        (deployer,),
    ).fetchone()
    token_events = row[0] if row else 0
    return (token_events > 0, float(token_events), 0.0)


def _template_reuse_count(conn: sqlite3.Connection, deployer: str) -> int:
    """
    Count how many of this deployer's contracts share a bytecode template.
    Uses bytecode_cache: if multiple contracts from same deployer map to
    same code_hash, that's template reuse.
    """
    # Get all contracts from this deployer
    contracts = conn.execute(
        "SELECT contract_address FROM contracts WHERE deployer_address = ?",
        (deployer,),
    ).fetchall()
    if len(contracts) < 3:
        return 0

    addrs = [r[0] for r in contracts]
    # Check bytecode_cache for shared hashes
    placeholders = ",".join("?" * len(addrs))
    rows = conn.execute(
        f"""SELECT code_hash, COUNT(*) as cnt
            FROM bytecode_cache
            WHERE source_contract IN ({placeholders})
            GROUP BY code_hash
            HAVING cnt >= 3""",
        addrs,
    ).fetchall()
    # source_contract is only the first contract that produced this hash,
    # so also check hit_count for templates originating from this deployer
    if rows:
        return max(r[1] for r in rows)

    # Alternative: check if any cache entry sourced from this deployer's
    # contract has high hit_count (meaning other contracts reused it)
    row = conn.execute(
        f"""SELECT MAX(hit_count) FROM bytecode_cache
            WHERE source_contract IN ({placeholders})""",
        addrs,
    ).fetchone()
    max_hits = row[0] if row and row[0] else 0
    return max_hits + 1 if max_hits >= 2 else 0


def _has_bot_interactions(conn: sqlite3.Connection, deployer: str) -> tuple[bool, int]:
    """Check if any contracts from this deployer received bot interactions."""
    row = conn.execute(
        """SELECT COUNT(DISTINCT te.interacting_address)
           FROM transaction_events te
           JOIN contracts c ON te.contract_address = c.contract_address
           WHERE c.deployer_address = ?
             AND te.interacting_address != ?""",
        (deployer, deployer),
    ).fetchone()
    count = row[0] if row else 0
    return (count > 0, count)


def _is_gas_station_funded(conn: sqlite3.Connection, deployer: str) -> tuple[bool, str]:
    """
    Check if deployer was funded by a gas station (high-nonce funder).
    We approximate this by checking the funding_trail JSON for known
    gas station addresses or funders that funded many deployers.
    """
    row = conn.execute(
        "SELECT funding_trail FROM deployers WHERE deployer_address = ?",
        (deployer,),
    ).fetchone()
    if not row or not row[0]:
        return (False, "")

    try:
        trail = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return (False, "")

    funder = trail.get("funding_source") or trail.get("funder") or ""
    if not funder:
        return (False, "")

    # Check how many other deployers this same funder funded
    count = conn.execute(
        """SELECT COUNT(*) FROM deployers
           WHERE funding_trail LIKE ? AND deployer_address != ?""",
        (f'%{funder}%', deployer),
    ).fetchone()[0]

    # If the funder funded 5+ other deployers, it's a gas station
    return (count >= 5, funder)


def _has_verified_contracts(conn: sqlite3.Connection, deployer: str) -> bool:
    """Check if any contracts from this deployer are verified or proxies."""
    row = conn.execute(
        """SELECT COUNT(*) FROM contract_verification cv
           JOIN contracts c ON cv.contract_address = c.contract_address
           WHERE c.deployer_address = ? AND cv.is_proxy = 1""",
        (deployer,),
    ).fetchone()
    return row[0] > 0 if row else False


def _has_zero_financial_flows(conn: sqlite3.Connection, deployer: str) -> bool:
    """Check if this deployer's contracts have zero transaction events."""
    row = conn.execute(
        """SELECT COUNT(*) FROM transaction_events te
           JOIN contracts c ON te.contract_address = c.contract_address
           WHERE c.deployer_address = ?""",
        (deployer,),
    ).fetchone()
    return row[0] == 0 if row else True


def score_deployer(conn: sqlite3.Connection, deployer: str) -> ScoreBreakdown:
    """Calculate the longitudinal behavioral score for a single deployer."""
    sb = ScoreBreakdown()

    # --- Positive signals ---

    # Multi-session activity
    active_days = _distinct_active_days(conn, deployer)
    if active_days >= 3:
        sb.add("multi_session", W_MULTI_SESSION, f"{active_days} distinct active days")

    # USDC / token flow
    has_flow, token_events, _ = _has_usdc_flows(conn, deployer)
    if has_flow:
        sb.add("usdc_flow", W_USDC_FLOW, f"{int(token_events)} token-related tx events")

    # Return ratio — we can't measure exact USDC in/out from on-chain data
    # in the DB alone, but if deployer interacts with own contracts (approve
    # then transfer pattern), that suggests extraction
    deployer_self_interactions = conn.execute(
        """SELECT COUNT(*) FROM transaction_events te
           JOIN contracts c ON te.contract_address = c.contract_address
           WHERE c.deployer_address = ? AND te.interacting_address = ?
             AND te.function_selector IN ('a9059cbb','23b872dd')""",
        (deployer, deployer),
    ).fetchone()[0]
    if deployer_self_interactions > 0:
        sb.add("return_ratio", W_RETURN_RATIO,
               f"deployer called transfer/transferFrom on own contracts {deployer_self_interactions} times")

    # Patience signal
    span = _activity_span_days(conn, deployer)
    if span > 7:
        sb.add("patience", W_PATIENCE, f"active span: {span:.1f} days")

    # Template reuse
    reuse = _template_reuse_count(conn, deployer)
    if reuse >= 3:
        sb.add("template_reuse", W_TEMPLATE_REUSE, f"bytecode template reused across {reuse} contracts")

    # Bot interaction
    has_bots, bot_count = _has_bot_interactions(conn, deployer)
    if has_bots:
        sb.add("bot_interaction", W_BOT_INTERACTION, f"{bot_count} distinct bot addresses interacted")

    # Gas station funding
    is_gs, funder = _is_gas_station_funded(conn, deployer)
    if is_gs:
        sb.add("gas_station_funding", W_GAS_STATION_FUNDING,
               f"funded by {funder[:18]}... (shared with 5+ other deployers)")

    # --- Negative signals ---

    # Zero financial flows
    if _has_zero_financial_flows(conn, deployer):
        sb.add("zero_flows", W_ZERO_FLOWS, "no transaction events on any contracts")

    # Single session
    if active_days <= 1:
        sb.add("single_session", W_SINGLE_SESSION, "active in only 1 day")

    # Verified contracts
    if _has_verified_contracts(conn, deployer):
        sb.add("verified", W_VERIFIED, "has verified or proxy contracts")

    # Irregular timing (long gaps suggest experimentation)
    if span > 0 and active_days > 0:
        contracts_deployed = conn.execute(
            "SELECT total_contracts_deployed FROM deployers WHERE deployer_address = ?",
            (deployer,),
        ).fetchone()
        total = contracts_deployed[0] if contracts_deployed else 0
        if total > 0:
            deploy_rate = total / max(span, 1)
            # Very low rate + long span = experimental
            if deploy_rate < 0.3 and span > 14:
                sb.add("irregular_timing", W_IRREGULAR_TIMING,
                       f"{total} contracts over {span:.0f} days = {deploy_rate:.2f}/day")

    return sb


def score_all_deployers(db_path: Optional[Path] = None) -> dict:
    """
    Score all deployers and persist results to the database.
    Returns summary statistics.
    """
    from surveillance.db import DEFAULT_DB_PATH
    db_path = db_path or DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row

    # Ensure columns exist
    try:
        conn.execute("SELECT behavioral_score FROM deployers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE deployers ADD COLUMN behavioral_score REAL DEFAULT 0.0")
        conn.commit()

    try:
        conn.execute("SELECT score_breakdown FROM deployers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE deployers ADD COLUMN score_breakdown TEXT")
        conn.commit()

    # Get all deployers
    deployers = conn.execute(
        "SELECT deployer_address FROM deployers"
    ).fetchall()

    scored = 0
    high = 0     # > 0.5
    critical = 0  # > 0.7
    top_scorers = []

    for row in deployers:
        addr = row[0]
        sb = score_deployer(conn, addr)
        score = sb.final_score
        breakdown_json = sb.to_json()

        conn.execute(
            "UPDATE deployers SET behavioral_score = ?, score_breakdown = ? WHERE deployer_address = ?",
            (round(score, 4), breakdown_json, addr),
        )
        scored += 1

        if score > 0.5:
            high += 1
        if score > 0.7:
            critical += 1

        top_scorers.append((addr, score, breakdown_json))

    conn.commit()

    # Sort and trim
    top_scorers.sort(key=lambda x: -x[1])
    top_10 = top_scorers[:10]

    # Find card counters: high score but NOT already caught by velocity detector
    card_counters = conn.execute(
        """SELECT d.deployer_address, d.behavioral_score, d.score_breakdown,
                  d.total_contracts_deployed, d.entity_type,
                  GROUP_CONCAT(DISTINCT c.confidence_tier) as tiers
           FROM deployers d
           LEFT JOIN contracts c ON d.deployer_address = c.deployer_address
           WHERE d.behavioral_score > 0.5
             AND d.deployer_address NOT IN (
                 SELECT deployer_address FROM contracts
                 WHERE confidence_tier = 'suspected'
                 GROUP BY deployer_address
                 HAVING COUNT(*) >= 5
             )
           GROUP BY d.deployer_address
           ORDER BY d.behavioral_score DESC"""
    ).fetchall()

    conn.close()

    return {
        "scored": scored,
        "high": high,
        "critical": critical,
        "top_10": [(addr, score, json.loads(bd)) for addr, score, bd in top_10],
        "card_counters": [dict(r) for r in card_counters],
    }


def print_report(results: dict) -> None:
    """Print a human-readable scoring report."""
    print("=" * 70)
    print("LONGITUDINAL BEHAVIORAL SCORING REPORT")
    print("=" * 70)
    print()
    print(f"Deployers scored:     {results['scored']}")
    print(f"High (>0.5):          {results['high']}")
    print(f"Critical (>0.7):      {results['critical']}")
    print()

    print("--- TOP 10 DEPLOYERS BY SCORE ---")
    for addr, score, breakdown in results["top_10"]:
        tier = "CRITICAL" if score > 0.7 else "HIGH" if score > 0.5 else "MEDIUM" if score > 0.3 else "LOW"
        print(f"\n  {addr}")
        print(f"  Score: {score:.3f} [{tier}]")
        for sig_name, sig_data in breakdown.get("signals", {}).items():
            w = sig_data["weight"]
            sign = "+" if w > 0 else ""
            print(f"    {sign}{w:.1f}  {sig_name}: {sig_data.get('detail', '')}")

    if results["card_counters"]:
        print()
        print("=" * 70)
        print("CARD COUNTERS — High score, NOT caught by velocity detector")
        print("=" * 70)
        for cc in results["card_counters"]:
            addr = cc["deployer_address"]
            score = cc["behavioral_score"]
            contracts = cc["total_contracts_deployed"]
            tiers = cc.get("tiers", "?")
            entity = cc.get("entity_type", "unknown")
            print(f"\n  {addr}")
            print(f"  Score: {score:.3f} | Contracts: {contracts} | Tiers: {tiers} | Entity: {entity}")
            try:
                bd = json.loads(cc["score_breakdown"])
                for sig_name, sig_data in bd.get("signals", {}).items():
                    w = sig_data["weight"]
                    sign = "+" if w > 0 else ""
                    print(f"    {sign}{w:.1f}  {sig_name}: {sig_data.get('detail', '')}")
            except (json.JSONDecodeError, TypeError):
                pass
    else:
        print("\nNo card counters found (all high-scorers already flagged by velocity detector)")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = score_all_deployers()
    print_report(results)
