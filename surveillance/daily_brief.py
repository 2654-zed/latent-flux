"""
Layer 3 — Daily Intelligence Brief

Synthesizes all upgrade modules into a single comprehensive report.
The "President's Daily Brief" for the on-chain ecosystem.

Usage:
    python -m surveillance.daily_brief --generate          # sync from prod + generate
    python -m surveillance.daily_brief --generate --no-sync # local only (offline)
"""

import argparse
import json
import os
import sqlite3
import urllib.parse
import urllib.request
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
BRIEFS_DIR = Path(__file__).resolve().parent / "data" / "briefs"

# Production sync settings
RAILWAY_URL = os.environ.get("RAILWAY_URL", "https://spypy.up.railway.app")
SYNC_TOKEN = os.environ.get("ADMIN_TOKEN", "")
SYNC_BATCH = 5000

# Tables to sync for the brief (in dependency order).
# Only tables the brief actually reads — not the full list from sync_railway_db.py.
BRIEF_TABLES = [
    "deployers", "contracts", "trap_events", "transaction_events",
    "bot_candidates", "alerts", "watchlist", "watchlist_hits",
    "trust_amplification", "approval_watchlist", "self_test_traps",
    "bytecode_families", "bytecode_family_members",
    "heartbeat", "connection_gaps", "liquidity_events",
    "approval_events", "bridge_events", "pair_creation_events",
    "org_transfer_events", "x402_facilitators", "x402_events",
    "x402_permit2_exposure",
]


def _sync_table(local_conn, table, token):
    """Pull a single table from production /dump endpoint into local DB."""
    local_cols = [r[1] for r in local_conn.execute(f"PRAGMA table_info([{table}])").fetchall()]
    if not local_cols:
        return 0

    offset = 0
    total_synced = 0
    while True:
        params = urllib.parse.urlencode({
            "token": token, "table": table, "offset": offset, "limit": SYNC_BATCH,
        })
        url = f"{RAILWAY_URL}/dump?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception:
            break

        rows = data.get("rows", [])
        if not rows:
            break

        remote_cols = set(rows[0].keys())
        sync_cols = [c for c in local_cols if c in remote_cols]
        if not sync_cols:
            break

        col_names = ", ".join(f"[{c}]" for c in sync_cols)
        placeholders = ", ".join("?" * len(sync_cols))
        sql = f"INSERT OR REPLACE INTO [{table}] ({col_names}) VALUES ({placeholders})"

        for row in rows:
            vals = [row.get(c) for c in sync_cols]
            try:
                local_conn.execute(sql, vals)
            except sqlite3.IntegrityError:
                try:
                    local_conn.execute(
                        f"INSERT OR IGNORE INTO [{table}] ({col_names}) VALUES ({placeholders})",
                        vals,
                    )
                except Exception:
                    pass

        total_synced += len(rows)
        offset += len(rows)

        if len(rows) < SYNC_BATCH:
            break
        time.sleep(0.15)

    local_conn.commit()
    return total_synced


def sync_from_production(token: str = None) -> dict:
    """Pull latest data from Railway production into local DB.

    Returns a dict of {table: rows_synced}.
    """
    tok = token or SYNC_TOKEN
    if not tok:
        print("[daily_brief] No ADMIN_TOKEN — skipping production sync.")
        print("  Set ADMIN_TOKEN env var or pass --token.")
        return {}

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")

    # Quick connectivity check
    try:
        url = f"{RAILWAY_URL}/dump?{urllib.parse.urlencode({'token': tok, 'table': 'heartbeat', 'offset': 0, 'limit': 1})}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            json.loads(resp.read())
    except Exception as e:
        print(f"[daily_brief] Cannot reach production: {e}")
        conn.close()
        return {}

    print(f"[daily_brief] Syncing from {RAILWAY_URL}...")
    results = {}
    for table in BRIEF_TABLES:
        synced = _sync_table(conn, table, tok)
        if synced > 0:
            print(f"  {table}: {synced:,} rows")
        results[table] = synced

    conn.execute("PRAGMA foreign_keys=ON")
    conn.close()
    total = sum(results.values())
    print(f"[daily_brief] Sync complete: {total:,} rows across {sum(1 for v in results.values() if v > 0)} tables.")
    return results


def generate_brief(target_date: str = None, sync: bool = True, token: str = None) -> Path:
    # Sync from production first (unless --no-sync)
    if sync:
        sync_from_production(token=token)

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
    md += f"**Coverage:** Base + Arbitrum + Optimism\n"
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

    # Watchlist alerts
    md += "## Watchlist Alerts\n\n"
    try:
        wl_hits = conn.execute("""
            SELECT wh.hit_type, wh.contract_address, wh.chain, wh.timestamp,
                   w.entity_name, w.priority, w.watch_reason
            FROM watchlist_hits wh
            JOIN watchlist w ON wh.watchlist_id = w.id
            WHERE wh.alerted = 0
            ORDER BY CASE w.priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END,
                     wh.timestamp DESC
            LIMIT 20
        """).fetchall()
        if wl_hits:
            md += f"**{len(wl_hits)} new hits:**\n\n"
            md += "| Priority | Entity | Type | Chain | Contract | Time |\n|---|---|---|---|---|---|\n"
            for h in wl_hits:
                md += f"| {h['priority']} | {h['entity_name']} | {h['hit_type']} | {h['chain']} | `{h['contract_address'][:16]}...` | {h['timestamp'][:16]} |\n"
            conn.execute("UPDATE watchlist_hits SET alerted = 1 WHERE alerted = 0")
            conn.commit()
        else:
            md += "No new watchlist activity.\n"
        # Summary
        total_w = conn.execute("SELECT COUNT(*) FROM watchlist WHERE active=1").fetchone()[0]
        total_h = conn.execute("SELECT COUNT(*) FROM watchlist_hits").fetchone()[0]
        md += f"\nActive watches: {total_w} | Total hits: {total_h}\n"
    except Exception:
        md += "Watchlist not yet initialized.\n"
    md += "\n"

    # Trust amplification alerts
    md += "## Trust Amplification Alerts\n"
    md += "*Epistemic: MIXED — amplification factor is arithmetic (D), bytecode family baseline is inferential (I)*\n\n"
    alerts = conn.execute("SELECT * FROM trust_amplification WHERE alert_level IN ('CRITICAL','WARNING') ORDER BY amplification_factor DESC LIMIT 5").fetchall()
    if alerts:
        md += "| Contract | Callers | Router % | Amplification | Alert |\n|---|---|---|---|---|\n"
        for a in alerts:
            md += f"| `{a['contract_address'][:16]}...` | {a['total_callers']:,} | {a['router_percentage']:.0f}% | {a['amplification_factor']:.1f}x | {a['alert_level']} |\n"
    else:
        md += "No trust amplification alerts.\n"
    md += "\n"

    # Camouflage watch
    md += "## Camouflage Watch\n"
    md += "*Epistemic: DEDUCTIVE — revert rates computed from on-chain receipts. 10% threshold is a policy choice (4.5x victim multiplier empirical).*\n\n"
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
    md += "## Diamond Model Index\n"
    md += "*Epistemic: INFERENTIAL — organizational attribution via funding-chain + behavioral pattern matching.*\n\n"
    diamonds = conn.execute("SELECT case_id, adversary_type, infrastructure_delivery, victim_count, confidence, capability_anti_forensic FROM diamond_model ORDER BY victim_count DESC").fetchall()
    if diamonds:
        md += "| Case | Adversary | Delivery | Victims | Confidence | Anti-Forensic Capabilities |\n|---|---|---|---|---|---|\n"
        for d in diamonds:
            af = d['capability_anti_forensic'] if d['capability_anti_forensic'] else "none"
            try:
                af_list = json.loads(af) if af != "none" else []
                af_str = ", ".join(af_list) if af_list else "none"
            except Exception:
                af_str = af
            md += f"| {d['case_id']} | {d['adversary_type']} | {d['infrastructure_delivery']} | {d['victim_count']:,} | {d['confidence']} | {af_str} |\n"
    md += "\n"

    # Anti-forensic capability tiers (org_001 three-tier model)
    md += "### Anti-Forensic Capability Tiers (org_001)\n\n"
    md += "| Layer | Technique | Target | Effect |\n|---|---|---|---|\n"
    md += "| Transaction | Custom selector drains (`e37136db`) | Log-based forensic tools | Zero log events; extraction invisible at event-log level |\n"
    md += "| Victim | Unicode WETH impersonation | Human inspection of token names | Unicode characters mimic legitimate token symbols |\n"
    md += "| **Intelligence** | **Vanity address spoofing** (7-char prefix) | **Organizational monitoring & chain analysis** | **Spoofed addresses pass truncated-display matching in dashboards and analyst review** |\n"
    md += "\n*Intelligence-layer capability is the highest counter-intelligence sophistication observed in the corpus.*\n\n"

    # System health
    md += "## System Health\n\n"
    for r in conn.execute("SELECT component, timestamp, deployments FROM heartbeat ORDER BY timestamp DESC"):
        md += f"- {r['component']}: {r['timestamp'][:19]} (d={r['deployments']})\n"

    gaps = conn.execute("SELECT COUNT(*) FROM connection_gaps WHERE disconnect > ?", (f"{day}T00:00:00",)).fetchone()[0]
    md += f"- Connection gaps today: {gaps}\n"

    # Strategy lifecycle monitor
    md += "## Strategy Lifecycle Monitor\n"
    md += "*Epistemic: INFERENTIAL — strategy classification is rule-based selector matching; lifecycle stage is a model prediction.*\n\n"
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

    # Self-test trap early warning
    md += "## Zero-Day Trap Watch (Self-Test Detection)\n"
    md += "*Epistemic: MIXED — self-test detection is deductive (deployer is sole caller, on-chain fact). \"Zero-day\" label implies future activation intent (inferential).*\n\n"
    try:
        self_tests = conn.execute("""
            SELECT st.*,
                ROUND(CAST(st.self_reverts AS REAL) / NULLIF(st.self_hits, 0) * 100, 0) as rev_pct
            FROM self_test_traps st
            ORDER BY CASE st.status WHEN 'ARMED' THEN 0 ELSE 1 END, st.self_hits DESC
            LIMIT 10
        """).fetchall()
        if self_tests:
            armed = [s for s in self_tests if s['status'] == 'ARMED']
            testing = [s for s in self_tests if s['status'] == 'SELF_TEST']
            if armed:
                md += f"**{len(armed)} ARMED** (first external victim detected):\n\n"
                md += "| Contract | Deployer | Self-Hits | Rev% | External | First Victim |\n|---|---|---|---|---|---|\n"
                for s in armed:
                    md += f"| `{s['contract_address'][:16]}...` | `{s['deployer_address'][:12]}...` | {s['self_hits']:,} | {s['rev_pct']:.0f}% | {s['external_hits']} | {s['external_first_seen'][:16] if s['external_first_seen'] else '?'} |\n"
                md += "\n"
            if testing:
                md += f"**{len(testing)} in R&D** (deployer-only interaction):\n\n"
                md += "| Contract | Deployer | Self-Hits | Rev% | Patterns |\n|---|---|---|---|---|\n"
                for s in testing:
                    md += f"| `{s['contract_address'][:16]}...` | `{s['deployer_address'][:12]}...` | {s['self_hits']:,} | {s['rev_pct']:.0f}% | {(s['bytecode_patterns'] or 'none')[:40]} |\n"
        else:
            md += "No self-test traps detected.\n"
    except Exception:
        md += "Self-test trap table not yet created.\n"
    md += "\n"

    # Approval drain watchlist
    md += "## Approval Drain Watchlist\n"
    md += "*Epistemic: MIXED — approval counts and drain detections are deductive (on-chain tx evidence). \"Pending drain\" framing implies future action (predictive).*\n\n"
    try:
        aw_total = conn.execute("SELECT COUNT(*) FROM approval_watchlist").fetchone()[0]
        aw_pending = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=0").fetchone()[0]
        aw_drained = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE drain_detected=1").fetchone()[0]
        aw_victims = conn.execute("SELECT COUNT(DISTINCT victim_address) FROM approval_watchlist").fetchone()[0]
        aw_self_test = conn.execute("SELECT COUNT(*) FROM approval_watchlist WHERE is_self_test_trap=1").fetchone()[0]
        md += f"Tracked approvals: **{aw_total}** | Pending drain: **{aw_pending}** | Drained: **{aw_drained}** | Unique victims: **{aw_victims}** | On self-test traps: **{aw_self_test}**\n\n"

        # Top contracts by pending
        top_pending = conn.execute("""
            SELECT contract_address, deployer_address, COUNT(*) as pending, is_self_test_trap
            FROM approval_watchlist WHERE drain_detected=0
            GROUP BY contract_address ORDER BY pending DESC LIMIT 5
        """).fetchall()
        if top_pending:
            md += "**Top pending approval targets:**\n\n"
            md += "| Contract | Deployer | Pending Victims | Self-Test |\n|---|---|---|---|\n"
            for t in top_pending:
                st = "YES" if t['is_self_test_trap'] else ""
                md += f"| `{t['contract_address'][:16]}...` | `{t['deployer_address'][:12]}...` | {t['pending']} | {st} |\n"
            md += "\n"

        # Any drains?
        drains = conn.execute("""
            SELECT contract_address, drain_caller, drain_timestamp, COUNT(*) as victims
            FROM approval_watchlist WHERE drain_detected=1
            GROUP BY contract_address ORDER BY victims DESC LIMIT 5
        """).fetchall()
        if drains:
            md += "**DRAINS DETECTED:**\n\n"
            md += "| Contract | Drain Caller | Victims | When |\n|---|---|---|---|\n"
            for d in drains:
                md += f"| `{d['contract_address'][:16]}...` | `{d['drain_caller'][:14]}...` | {d['victims']} | {d['drain_timestamp'][:16]} |\n"
    except Exception:
        md += "Approval watchlist table not yet created.\n"
    md += "\n"

    # Infrastructure events
    md += "## Infrastructure Events\n\n"
    try:
        events = conn.execute("""
            SELECT event_date, event_type, provider, chain, description, impact_metric, impact_pct
            FROM infra_events ORDER BY event_date DESC LIMIT 5
        """).fetchall()
        if events:
            md += "| Date | Type | Provider | Chain | Impact | Description |\n|---|---|---|---|---|---|\n"
            for e in events:
                impact = f"{e['impact_pct']:+.0f}% {e['impact_metric']}" if e['impact_pct'] else ""
                md += f"| {e['event_date']} | {e['event_type']} | {e['provider']} | {e['chain']} | {impact} | {e['description'][:60]}... |\n"
        else:
            md += "No infrastructure events logged.\n"
    except Exception:
        md += "Infrastructure events table not yet created.\n"
    md += "\n"

    # x402 activity summary
    md += "## x402 Activity\n\n"
    try:
        # Event counts in the reporting day
        today_events = conn.execute(
            "SELECT COUNT(*) FROM x402_events WHERE timestamp LIKE ?",
            (day + "%",),
        ).fetchone()[0] or 0
        total_events = conn.execute(
            "SELECT COUNT(*) FROM x402_events"
        ).fetchone()[0] or 0

        # Exposure snapshot
        exp_total = conn.execute(
            "SELECT COUNT(*) FROM x402_permit2_exposure"
        ).fetchone()[0] or 0
        exp_owners = conn.execute(
            "SELECT COUNT(DISTINCT owner_address) FROM x402_permit2_exposure"
        ).fetchone()[0] or 0
        exp_tokens = conn.execute(
            "SELECT COUNT(DISTINCT token_contract) FROM x402_permit2_exposure"
        ).fetchone()[0] or 0

        # Facilitator breakdown
        fac_known = conn.execute(
            "SELECT COUNT(*) FROM x402_facilitators WHERE classification = 'known'"
        ).fetchone()[0] or 0
        fac_unknown = conn.execute(
            "SELECT COUNT(*) FROM x402_facilitators WHERE classification = 'unknown'"
        ).fetchone()[0] or 0
        fac_rogue = conn.execute(
            "SELECT COUNT(*) FROM x402_facilitators WHERE classification = 'rogue'"
        ).fetchone()[0] or 0

        # Today's x402 alerts
        alert_rows = conn.execute("""
            SELECT alert_type, COUNT(*) AS c
            FROM alerts
            WHERE alert_type LIKE 'X402_%'
              AND timestamp LIKE ?
            GROUP BY alert_type
            ORDER BY c DESC
        """, (day + "%",)).fetchall()

        md += f"- x402 events captured today: **{today_events:,}** (lifetime: {total_events:,})\n"
        md += f"- Permit2 exposure rows: **{exp_total:,}** "
        md += f"across **{exp_owners:,}** owners and {exp_tokens} tokens\n"
        md += f"- Facilitators: {fac_known} known, {fac_unknown} unknown, {fac_rogue} rogue\n"

        if alert_rows:
            md += "- Alerts today:\n"
            for r in alert_rows:
                md += f"  - `{r['alert_type']}`: {r['c']}\n"

        if total_events == 0 and exp_total == 0:
            md += "- _No x402 activity detected in reporting period._\n"
        elif total_events == 0 and exp_total > 0:
            md += ("- _Stored-potential only: live x402 settlements not yet "
                   "observed on monitored chains. Exposure rows are the "
                   "attack surface sitting on-chain, waiting for a facilitator "
                   "to consume them._\n")

        # Top exposed owners
        top_owners = conn.execute("""
            SELECT owner_address, COUNT(DISTINCT token_contract) AS n_tokens,
                   MAX(last_seen) AS last_seen
            FROM x402_permit2_exposure
            GROUP BY owner_address
            ORDER BY n_tokens DESC
            LIMIT 5
        """).fetchall()
        if top_owners:
            md += "\n**Top exposed owners (most Permit2 allowances granted):**\n\n"
            md += "| Owner | Distinct tokens | Last activity |\n|---|---|---|\n"
            for r in top_owners:
                md += (f"| `{r['owner_address'][:18]}...` | "
                       f"{r['n_tokens']} | {r['last_seen']} |\n")
    except Exception as e:
        md += f"x402 tables not yet initialized: {e}\n"
    md += "\n"

    md += "\n---\n*Generated by surveillance/daily_brief.py*\n"

    conn.close()

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = BRIEFS_DIR / f"DAILY_BRIEF_{day}.md"
    filepath.write_text(md, encoding="utf-8")
    print(f"[daily_brief] Generated: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer 3 Daily Intelligence Brief — syncs from production then generates."
    )
    parser.add_argument("--generate", action="store_true", help="Generate the brief")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--no-sync", action="store_true", help="Skip production sync (use local DB only)")
    parser.add_argument("--sync-only", action="store_true", help="Sync from production without generating")
    parser.add_argument("--token", default=None, help="ADMIN_TOKEN for production sync")
    args = parser.parse_args()
    if args.sync_only:
        sync_from_production(token=args.token)
    elif args.generate or not args.date:
        generate_brief(target_date=args.date, sync=not args.no_sync, token=args.token)
