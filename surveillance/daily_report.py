"""
Layer 3 — Automated Daily Report Generator

Generates a comprehensive daily intelligence report from the surveillance database.
Designed to run via cron at a set time each day (e.g., 06:00 UTC).

Usage:
    python -m surveillance.daily_report
    python -m surveillance.daily_report --date 2026-03-23

Output: data/cases/DAILY_REPORT_YYYY-MM-DD.md
"""

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
CASES_DIR = Path(__file__).resolve().parent / "data" / "cases"

ORG_LABELS = {
    "org_001": "DELEGATECALL Trap Network",
    "org_002": "tx.origin Campaign",
    "org_003": "Ghost Fee-Skimmer Network",
}

ORG_CASHOUTS = {
    "org_001": "0xc6962004f452be9203591991d15f6b388e09e8d0",
}

ORG_DEPLOYERS = {
    "org_001": [
        "0xe93d64f3fbc352131e79fc5578cbe44b66697f86",
        "0xfd51e33d44b376ef346d24a130a51035db09c1dc",
    ],
}


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def generate_report(target_date=None):
    """Generate the daily report for a given date (defaults to yesterday)."""
    conn = get_connection()
    now = datetime.now(timezone.utc)

    if target_date:
        day_start = f"{target_date}T00:00:00"
        day_end = f"{target_date}T23:59:59"
        report_date = target_date
    else:
        yesterday = now - timedelta(days=1)
        report_date = yesterday.strftime("%Y-%m-%d")
        day_start = f"{report_date}T00:00:00"
        day_end = f"{report_date}T23:59:59"

    # Previous day for comparison
    prev_date = (datetime.fromisoformat(report_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_start = f"{prev_date}T00:00:00"
    prev_end = f"{prev_date}T23:59:59"

    md = f"# DAILY INTELLIGENCE REPORT — {report_date}\n"
    md += f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
    md += f"**Period:** {day_start[:10]} 00:00 UTC to 23:59 UTC\n\n---\n\n"

    # ── CORPUS GROWTH ──
    today_c = conn.execute("SELECT COUNT(*) FROM contracts WHERE detection_timestamp BETWEEN ? AND ?", (day_start, day_end)).fetchone()[0]
    prev_c = conn.execute("SELECT COUNT(*) FROM contracts WHERE detection_timestamp BETWEEN ? AND ?", (prev_start, prev_end)).fetchone()[0]
    total_c = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]

    today_tx = conn.execute("SELECT COUNT(*) FROM transaction_events WHERE timestamp BETWEEN ? AND ?", (day_start, day_end)).fetchone()[0]
    prev_tx = conn.execute("SELECT COUNT(*) FROM transaction_events WHERE timestamp BETWEEN ? AND ?", (prev_start, prev_end)).fetchone()[0]

    today_rev = conn.execute("SELECT COUNT(*) FROM transaction_events WHERE timestamp BETWEEN ? AND ? AND is_reverted=1", (day_start, day_end)).fetchone()[0]
    rev_pct = (today_rev / today_tx * 100) if today_tx > 0 else 0
    prev_rev = conn.execute("SELECT COUNT(*) FROM transaction_events WHERE timestamp BETWEEN ? AND ? AND is_reverted=1", (prev_start, prev_end)).fetchone()[0]
    prev_rev_pct = (prev_rev / prev_tx * 100) if prev_tx > 0 else 0

    today_dep = conn.execute("SELECT COUNT(DISTINCT deployer_address) FROM contracts WHERE detection_timestamp BETWEEN ? AND ?", (day_start, day_end)).fetchone()[0]
    total_dep = conn.execute("SELECT COUNT(*) FROM deployers").fetchone()[0]
    total_bots = conn.execute("SELECT COUNT(*) FROM bot_candidates").fetchone()[0]

    c_delta = today_c - prev_c
    c_sign = "+" if c_delta >= 0 else ""
    tx_delta = today_tx - prev_tx
    tx_sign = "+" if tx_delta >= 0 else ""
    rev_delta = rev_pct - prev_rev_pct
    rev_sign = "+" if rev_delta >= 0 else ""

    md += "## Corpus Growth\n\n"
    md += "| Metric | Today | Previous | Change | Total |\n"
    md += "|---|---|---|---|---|\n"
    md += f"| Contracts deployed | {today_c:,} | {prev_c:,} | {c_sign}{c_delta:,} | {total_c:,} |\n"
    md += f"| TX events | {today_tx:,} | {prev_tx:,} | {tx_sign}{tx_delta:,} | — |\n"
    md += f"| Revert rate | {rev_pct:.1f}% | {prev_rev_pct:.1f}% | {rev_sign}{rev_delta:.1f}% | — |\n"
    md += f"| Active deployers | {today_dep:,} | — | — | {total_dep:,} |\n"
    md += f"| Bots tracked | — | — | — | {total_bots:,} |\n\n"

    # Chain split
    md += "### Chain Split\n| Chain | Contracts | TX Events |\n|---|---|---|\n"
    for r in conn.execute("""
        SELECT chain, COUNT(*) FROM contracts
        WHERE detection_timestamp BETWEEN ? AND ? GROUP BY chain
    """, (day_start, day_end)):
        tx_chain = conn.execute("""
            SELECT COUNT(*) FROM transaction_events te
            JOIN contracts c ON te.contract_address = c.contract_address
            WHERE te.timestamp BETWEEN ? AND ? AND c.chain = ?
        """, (day_start, day_end, r[0])).fetchone()[0]
        md += f"| {r[0]} | {r[1]:,} | {tx_chain:,} |\n"
    md += "\n"

    # ── HOT CONTRACTS ──
    md += "## Hot Contracts (by interaction volume)\n\n"
    md += "| Contract | Chain | Hits | Callers | Rev% | Assessment |\n"
    md += "|---|---|---|---|---|---|\n"
    for r in conn.execute("""
        SELECT te.contract_address, c.chain,
               COUNT(*) as hits,
               COUNT(DISTINCT te.interacting_address) as callers,
               ROUND(SUM(CASE WHEN te.is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1) as rp,
               c.bytecode_pattern_notes, c.confidence_tier
        FROM transaction_events te
        JOIN contracts c ON te.contract_address = c.contract_address
        WHERE te.timestamp BETWEEN ? AND ?
        GROUP BY te.contract_address ORDER BY hits DESC LIMIT 10
    """, (day_start, day_end)):
        tier = r["confidence_tier"]
        if r["rp"] < 5 and r["callers"] > 50:
            assess = "Camouflaged"
        elif r["rp"] > 50:
            assess = "Overt trap"
        elif tier == "confirmed":
            assess = "CONFIRMED"
        else:
            assess = tier
        md += f"| `{r[0][:14]}...` | {r[1]} | {r[2]:,} | {r[3]:,} | {r[4]}% | {assess} |\n"
    md += "\n"

    # ── CAMOUFLAGE RATIO ──
    camo = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN rp < 10 THEN 1 ELSE 0 END) as camouflaged,
               SUM(CASE WHEN rp > 50 THEN 1 ELSE 0 END) as overt
        FROM (
            SELECT ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1) as rp
            FROM transaction_events WHERE timestamp BETWEEN ? AND ?
            GROUP BY contract_address HAVING COUNT(*) >= 10
        )
    """, (day_start, day_end)).fetchone()
    if camo and camo[0] > 0:
        md += "## Camouflage Ratio\n\n"
        md += f"Of {camo[0]} active contracts (10+ interactions today):\n"
        md += f"- **{camo[1]}** camouflaged (<10% revert) — {camo[1]/camo[0]*100:.0f}%\n"
        md += f"- **{camo[2]}** overt traps (>50% revert) — {camo[2]/camo[0]*100:.0f}%\n\n"

    # ── ORGANIZATION ACTIVITY ──
    md += "## Organization Activity\n\n"
    for org_id, label in ORG_LABELS.items():
        md += f"### {org_id}: {label}\n"

        # Deployments
        if org_id in ORG_DEPLOYERS:
            ops = ORG_DEPLOYERS[org_id]
            ph = ",".join("?" * len(ops))
            dep_count = conn.execute(
                f"SELECT COUNT(*) FROM contracts WHERE deployer_address IN ({ph}) AND detection_timestamp BETWEEN ? AND ?",
                ops + [day_start, day_end],
            ).fetchone()[0]
            md += f"- Deployments today: {dep_count}\n"

        if org_id == "org_002":
            o2 = conn.execute("""
                SELECT COUNT(*) FROM contracts
                WHERE bytecode_pattern_notes LIKE '%tx.origin%'
                  AND detection_timestamp BETWEEN ? AND ?
            """, (day_start, day_end)).fetchone()[0]
            md += f"- tx.origin deployments today: {o2}\n"

        if org_id == "org_003":
            o3_deps = conn.execute("""
                SELECT COUNT(DISTINCT te.interacting_address)
                FROM transaction_events te
                JOIN contracts c ON te.contract_address = c.contract_address
                WHERE c.deployer_address IN (
                    SELECT deployer_address FROM deployers WHERE entity_type = 'org_003_ghost_deployer'
                ) AND te.timestamp BETWEEN ? AND ?
            """, (day_start, day_end)).fetchone()[0]
            md += f"- Victims interacting today: {o3_deps}\n"

        # Extraction events
        extractions = conn.execute(
            "SELECT event_id, total_usd_moved FROM extraction_events WHERE observed_at BETWEEN ? AND ?",
            (day_start, day_end),
        ).fetchall()
        if extractions:
            for e in extractions:
                md += f"- **{e[0]}**: ${e[1]:,.0f} extracted\n"
        md += "\n"

    # ── NEW DEPLOYERS ──
    md += "## Notable New Deployers\n\n"
    md += "| Deployer | Chain | Contracts | Type |\n|---|---|---|---|\n"
    for r in conn.execute("""
        SELECT deployer_address, chain, total_contracts_deployed, entity_type
        FROM deployers WHERE first_seen BETWEEN ? AND ?
          AND total_contracts_deployed >= 5
        ORDER BY total_contracts_deployed DESC LIMIT 10
    """, (day_start, day_end)):
        md += f"| `{r[0][:16]}...` | {r[1]} | {r[2]} | {r[3]} |\n"
    md += "\n"

    # ── CROSS-CHAIN ──
    xc = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT deployer_address FROM contracts
            GROUP BY deployer_address HAVING COUNT(DISTINCT chain) > 1
        )
    """).fetchone()[0]
    md += f"## Cross-Chain Deployers: {xc} total\n\n"

    # ── LONGITUDINAL SCORING ──
    high = conn.execute("SELECT COUNT(*) FROM deployers WHERE behavioral_score > 0.5").fetchone()[0]
    med = conn.execute("SELECT COUNT(*) FROM deployers WHERE behavioral_score > 0.3").fetchone()[0]
    md += "## Longitudinal Scoring\n\n"
    md += f"- HIGH (>0.5): {high} deployers\n"
    md += f"- MEDIUM+ (>0.3): {med} deployers\n"

    top = conn.execute("""
        SELECT deployer_address, behavioral_score FROM deployers
        WHERE behavioral_score > 0.4 ORDER BY behavioral_score DESC LIMIT 5
    """).fetchall()
    if top:
        md += "\nTop scorers:\n"
        for r in top:
            md += f"- `{r[0][:16]}...` — {r[1]}\n"
    md += "\n"

    # ── SYSTEM HEALTH ──
    md += "## System Health\n\n"
    md += "| Monitor | Last Heartbeat | Deployments |\n|---|---|---|\n"
    for r in conn.execute("SELECT component, timestamp, deployments FROM heartbeat ORDER BY timestamp DESC"):
        md += f"| {r[0]} | {r[1][:19]} | {r[2]:,} |\n"

    gaps = conn.execute(
        "SELECT COUNT(*) FROM connection_gaps WHERE disconnect BETWEEN ? AND ?",
        (day_start, day_end),
    ).fetchone()[0]
    md += f"\nConnection gaps today: {gaps}\n\n"

    # ── CONFIRMED CONTRACTS ──
    confirmed = conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'confirmed'").fetchone()[0]
    suspected = conn.execute("SELECT COUNT(*) FROM contracts WHERE confidence_tier = 'suspected'").fetchone()[0]
    md += f"## Detection Summary\n\n"
    md += f"- Confirmed traps: {confirmed}\n"
    md += f"- Suspected: {suspected:,}\n"
    md += f"- Extraction events documented: {conn.execute('SELECT COUNT(*) FROM extraction_events').fetchone()[0]}\n"
    md += f"- Total USD traced: ${sum(r[0] for r in conn.execute('SELECT total_usd_moved FROM extraction_events').fetchall()):,.0f}\n\n"

    # ── WATCHLIST ──
    md += "## Active Watchlist\n\n"

    # Bot_A
    bc = conn.execute("SELECT total_revert_count, last_seen FROM bot_candidates WHERE address LIKE '0x84792c2a%'").fetchone()
    if bc:
        md += f"- **Bot_A (0x84792c2a)**: {bc[0]:,} reverts, last seen {bc[1][:19]}. "
        md += "MEV whale R&D sidecar ($5.75M vault). "
        md += "Refueled Mar 23 — operator still active.\n"

    # Dragon
    dragon = conn.execute("SELECT total_contracts_deployed FROM deployers WHERE deployer_address = '0x2e20b26172a8625c33097288075920a6210a8233'").fetchone()
    if dragon:
        md += f"- **Dragon (0x2e20b2)**: {dragon[0]:,} pre-staged contracts. Still approving to PancakeSwap. Balance ~0 ETH.\n"

    # Open exposures
    exp = conn.execute("SELECT COUNT(*) FROM live_exposures WHERE status = 'open'").fetchone()[0]
    md += f"- **Open exposures**: {exp}\n\n"

    md += "---\n*Generated by surveillance/daily_report.py*\n"

    conn.close()

    # Save
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CASES_DIR / f"DAILY_REPORT_{report_date}.md"
    filepath.write_text(md, encoding="utf-8")
    print(f"Daily report generated: {filepath}")
    return filepath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate daily intelligence report")
    parser.add_argument("--date", default=None, help="Date to report on (YYYY-MM-DD, default: yesterday)")
    args = parser.parse_args()
    generate_report(target_date=args.date)
