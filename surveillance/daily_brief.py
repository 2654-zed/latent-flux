"""
Layer 3 — Daily Intelligence Brief

Synthesizes all upgrade modules into a single comprehensive report.
The "President's Daily Brief" for the on-chain ecosystem.

Usage:
    python -m surveillance.daily_brief --generate
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
BRIEFS_DIR = Path(__file__).resolve().parent / "data" / "briefs"


def generate_brief(target_date: str = None) -> Path:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc)

    if target_date:
        day = target_date
    else:
        day = now.strftime("%Y-%m-%d")

    prev = (datetime.fromisoformat(day) - timedelta(days=1)).strftime("%Y-%m-%d")

    # Totals
    total_c = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    total_tx = conn.execute("SELECT COUNT(*) FROM transaction_events").fetchone()[0]
    total_ent = conn.execute("SELECT COUNT(*) FROM entity_classification").fetchone()[0]

    # Today vs yesterday from daily_metrics
    today_m = conn.execute("SELECT * FROM daily_metrics WHERE date = ?", (day,)).fetchone()
    prev_m = conn.execute("SELECT * FROM daily_metrics WHERE date = ?", (prev,)).fetchone()

    md = f"# LAYER 3 DAILY INTELLIGENCE BRIEF\n"
    md += f"**Date:** {day}\n"
    md += f"**Coverage:** Base + Arbitrum\n"
    md += f"**Corpus:** {total_c:,} contracts | {total_tx:,} events | {total_ent:,} entities classified\n\n---\n\n"

    # Key metrics
    md += "## Key Metrics (vs Yesterday)\n\n"
    md += "| Metric | Today | Yesterday | Trend |\n|---|---|---|---|\n"

    if today_m and prev_m:
        def trend(t, p):
            if t is None or p is None or p == 0:
                return "—"
            d = ((t - p) / p) * 100
            return f"{'+'if d>=0 else ''}{d:.0f}%"

        md += f"| New contracts | {today_m['new_contracts']:,} | {prev_m['new_contracts']:,} | {trend(today_m['new_contracts'], prev_m['new_contracts'])} |\n"
        md += f"| TX events | {today_m['total_tx_events']:,} | {prev_m['total_tx_events']:,} | {trend(today_m['total_tx_events'], prev_m['total_tx_events'])} |\n"
        md += f"| Revert rate | {today_m['revert_rate']:.1f}% | {prev_m['revert_rate']:.1f}% | {'up' if today_m['revert_rate'] > prev_m['revert_rate'] else 'down'} |\n"
        md += f"| Unique callers | {today_m['unique_callers']:,} | {prev_m['unique_callers']:,} | {trend(today_m['unique_callers'], prev_m['unique_callers'])} |\n"
    elif today_m:
        md += f"| New contracts | {today_m['new_contracts']:,} | — | — |\n"
        md += f"| TX events | {today_m['total_tx_events']:,} | — | — |\n"
        md += f"| Revert rate | {today_m['revert_rate']:.1f}% | — | — |\n"
    else:
        md += "| No daily_metrics data for today | — | — | — |\n"
    md += "\n"

    # Prediction scorecard
    md += "## Prediction Scorecard\n\n"
    scored = conn.execute("SELECT * FROM predictions WHERE target_date <= ? AND actual_value IS NOT NULL ORDER BY scored_at DESC LIMIT 5", (day,)).fetchall()
    if scored:
        md += "| Metric | Predicted | Actual | Hit |\n|---|---|---|---|\n"
        for p in scored:
            md += f"| {p['metric_name']} | {p['predicted_value']:.1f} | {p['actual_value']:.1f} | {'YES' if p['hit'] else 'NO'} |\n"
    else:
        md += "No scored predictions yet.\n"
    md += "\n"

    # New predictions
    md += "## Predictions (48h window)\n\n"
    preds = conn.execute("SELECT * FROM predictions WHERE target_date > ? AND actual_value IS NULL ORDER BY target_date", (day,)).fetchall()
    if preds:
        md += "| Target Date | Metric | Predicted | Range |\n|---|---|---|---|\n"
        for p in preds:
            md += f"| {p['target_date']} | {p['metric_name']} | {p['predicted_value']:.1f} | {p['predicted_range_low']:.0f}-{p['predicted_range_high']:.0f} |\n"
    else:
        md += "No pending predictions.\n"
    md += "\n"

    # Trust amplification alerts
    md += "## Trust Amplification Alerts\n\n"
    alerts = conn.execute("SELECT * FROM trust_amplification WHERE alert_level IN ('CRITICAL','WARNING') ORDER BY amplification_factor DESC LIMIT 5").fetchall()
    if alerts:
        md += "| Contract | Callers | Router % | Amplification | Alert |\n|---|---|---|---|---|\n"
        for a in alerts:
            md += f"| `{a['contract_address'][:16]}...` | {a['total_callers']:,} | {a['router_percentage']:.0f}% | {a['amplification_factor']:.1f}x | {a['alert_level']} |\n"
    else:
        md += "No trust amplification alerts.\n"
    md += "\n"

    # Camouflage watch
    md += "## Camouflage Watch\n\n"
    camo = conn.execute("SELECT * FROM camouflage_metrics WHERE date = ? AND chain IS NULL", (day,)).fetchone()
    if not camo:
        camo = conn.execute("SELECT * FROM camouflage_metrics ORDER BY date DESC LIMIT 1").fetchone()
    if camo:
        md += f"- Active contracts: {camo['total_active_contracts']}\n"
        md += f"- Camouflaged (<10% revert): {camo['camouflaged_count']} ({camo['camouflage_ratio']:.1f}%)\n"
        md += f"- Overt (>50% revert): {camo['overt_count']}\n"
        md += f"- Total camouflaged victims: {camo['total_camouflaged_victims']:,}\n"
    else:
        md += "No camouflage data.\n"
    md += "\n"

    # Callback trap watch — mine existing data for V3/V2 callback abuse
    md += "## Callback Trap Watch\n\n"
    cb_traps = conn.execute("""
        SELECT c.contract_address, c.chain, c.deployer_address, c.bytecode_pattern_notes,
               COUNT(te.tx_hash) as hits,
               COUNT(DISTINCT te.interacting_address) as victims,
               SUM(CASE WHEN te.is_reverted=1 THEN 1 ELSE 0 END) as reverts,
               ROUND(SUM(CASE WHEN te.is_reverted=1 THEN 1.0 ELSE 0 END)/NULLIF(COUNT(*),0)*100, 1) as rev_pct,
               d.funding_trail
        FROM contracts c
        LEFT JOIN transaction_events te ON te.contract_address = c.contract_address
            AND te.timestamp >= ? AND te.timestamp < ?
        LEFT JOIN deployers d ON d.deployer_address = c.deployer_address
        WHERE c.bytecode_pattern_notes LIKE '%callback%'
        GROUP BY c.contract_address
        HAVING hits > 0
        ORDER BY hits DESC LIMIT 10
    """, (f"{day}T00:00:00", f"{day}T23:59:59")).fetchall()
    if cb_traps:
        md += "| Contract | Chain | Hits | Victims | Rev% | Deployer | Org Link |\n|---|---|---|---|---|---|---|\n"
        for t in cb_traps:
            org = ""
            if t['funding_trail']:
                import json as _json
                try:
                    trail = _json.loads(t['funding_trail'])
                    org = trail.get('org_link', '')
                except Exception:
                    pass
            md += f"| `{t['contract_address'][:16]}...` | {t['chain']} | {t['hits']:,} | {t['victims']:,} | {t['rev_pct']:.0f}% | `{t['deployer_address'][:12]}...` | {org} |\n"
        total_victims = sum(t['victims'] for t in cb_traps)
        md += f"\nCallback traps active today: {len(cb_traps)} | Total victims: {total_victims:,}\n"
    else:
        md += "No callback trap activity today.\n"
    md += "\n"

    # Top bytecode families
    md += "## Bytecode Family Watch\n\n"
    fams = conn.execute("SELECT * FROM bytecode_families WHERE is_cross_deployer=1 ORDER BY member_count DESC LIMIT 5").fetchall()
    if fams:
        md += "| Family | Members | Deployers | Victims | Avg Rev% |\n|---|---|---|---|---|\n"
        for f in fams:
            md += f"| `{f['family_id'][:20]}...` | {f['member_count']:,} | {f['unique_deployers']:,} | {f['total_victims']:,} | {f['avg_revert_rate']:.1f}% |\n"
    md += "\n"

    # Top anomalies
    md += "## Behavioral Anomalies (Top 5 z-scores)\n\n"
    anoms = conn.execute("SELECT * FROM behavioral_anomalies ORDER BY z_score DESC LIMIT 5").fetchall()
    if anoms:
        md += "| Address | Type | Metric | Value | Mean | Z-score |\n|---|---|---|---|---|---|\n"
        for a in anoms:
            md += f"| `{a['address'][:16]}...` | {a['entity_type']} | {a['metric_name']} | {a['observed_value']:.1f} | {a['baseline_mean']:.1f} | {a['z_score']:.1f} |\n"
    md += "\n"

    # Diamond model summary
    md += "## Diamond Model Index\n\n"
    diamonds = conn.execute("SELECT case_id, adversary_type, infrastructure_delivery, victim_count, confidence FROM diamond_model ORDER BY victim_count DESC").fetchall()
    if diamonds:
        md += "| Case | Adversary | Delivery | Victims | Confidence |\n|---|---|---|---|---|\n"
        for d in diamonds:
            md += f"| {d['case_id']} | {d['adversary_type']} | {d['infrastructure_delivery']} | {d['victim_count']:,} | {d['confidence']} |\n"
    md += "\n"

    # System health
    md += "## System Health\n\n"
    for r in conn.execute("SELECT component, timestamp, deployments FROM heartbeat ORDER BY timestamp DESC"):
        md += f"- {r['component']}: {r['timestamp'][:19]} (d={r['deployments']})\n"

    gaps = conn.execute("SELECT COUNT(*) FROM connection_gaps WHERE disconnect > ?", (f"{day}T00:00:00",)).fetchone()[0]
    md += f"- Connection gaps today: {gaps}\n"

    # Strategy lifecycle monitor
    md += "## Strategy Lifecycle Monitor\n\n"
    try:
        lifecycle = conn.execute("SELECT * FROM strategy_lifecycle ORDER BY saturation_index DESC").fetchall()
        if lifecycle:
            md += "| Strategy | Bots | Traps | Saturation | Stage | Profit | Avg Rev |\n|---|---|---|---|---|---|---|\n"
            for sl in lifecycle:
                md += f"| {sl['strategy_type']} | {sl['active_bots']} | {sl['trap_count']} | {sl['saturation_index']:.3f} | {sl['lifecycle_stage']} | {sl['avg_bot_profitability']} | {sl['avg_bot_revert_rate']:.1f}% |\n"
        else:
            md += "No strategy lifecycle data. Run `strategy_fingerprint --lifecycle`.\n"
    except Exception:
        md += "Strategy lifecycle table not yet created.\n"
    md += "\n"

    # Bait detection summary
    md += "## Bait Detection Summary\n\n"
    try:
        bait = conn.execute("""
            SELECT bait_type, target_strategy, COUNT(*) as traps,
                SUM(bot_victims) as bot_victims,
                ROUND(AVG(bot_revert_rate), 1) as avg_bot_rev
            FROM bait_profiles GROUP BY bait_type, target_strategy ORDER BY traps DESC
        """).fetchall()
        if bait:
            md += "| Bait Type | Target Strategy | Traps | Bot Victims | Bot Rev% |\n|---|---|---|---|---|\n"
            for b in bait:
                md += f"| {b['bait_type']} | {b['target_strategy']} | {b['traps']} | {b['bot_victims']} | {b['avg_bot_rev']:.1f}% |\n"
        else:
            md += "No bait profiles. Run `strategy_fingerprint --profile-baits`.\n"
    except Exception:
        md += "Bait profiles table not yet created.\n"
    md += "\n"

    # Vanity address activity
    md += "## Vanity Address Watch\n\n"
    try:
        vanity = conn.execute("""
            SELECT vt.label, vt.pattern_type, COUNT(DISTINCT vt.address) as addresses,
                vt.address_type
            FROM vanity_tags vt
            GROUP BY vt.label, vt.address_type
            ORDER BY addresses DESC LIMIT 15
        """).fetchall()
        if vanity:
            md += "| Label | Type | Addresses | Role |\n|---|---|---|---|\n"
            for v in vanity:
                md += f"| {v['label']} | {v['pattern_type']} | {v['addresses']} | {v['address_type']} |\n"
        else:
            md += "No vanity addresses tagged. Run `vanity_detector --scan`.\n"
    except Exception:
        md += "Vanity tags table not yet created.\n"
    md += "\n"

    md += "\n---\n*Generated by surveillance/daily_brief.py*\n"

    conn.close()

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = BRIEFS_DIR / f"DAILY_BRIEF_{day}.md"
    filepath.write_text(md, encoding="utf-8")
    print(f"[daily_brief] Generated: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    if args.generate or not args.date:
        generate_brief(target_date=args.date)
