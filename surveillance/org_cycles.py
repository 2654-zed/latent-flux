"""
Organization Activity Cycle Analysis for Layer 3 Surveillance.

Analyzes temporal patterns for tracked organizations:
active/dormant cycles, time-of-day patterns, funding->deployment lag,
deployment->cashout lag, and predicts next active windows.

Usage:
    python -m surveillance.org_cycles
    python -m surveillance.org_cycles --gap-threshold 4
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
CASES_DIR = Path(__file__).resolve().parent / "data" / "cases"

DEFAULT_GAP_HOURS = 6

# Org definitions — addresses grouped by org
ORGS = {
    "org_001": {
        "label": "DELEGATECALL Trap Network (Arbitrum + Base)",
        "roles": {
            "0xf186cb00e49e18491db5783ff04fae3818102ff7": "treasury",
            "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "operator",
            "0xfd51e33d44b376ef346d24a130a51035db09c1dc": "operator_2",
            "0xc6962004f452be9203591991d15f6b388e09e8d0": "cashout",
            "0x01989c93890aed05a63d179b03424997075b6acf": "exit_cex",
            "0x8c826f795466e39acbff1bb4eeeb759609377ba1": "gas_station",
            "0x360e68faccca8ca495c1b759fd9eee466db9fb32": "treasury_branch",
            "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "laundry",
            "0x96daa0b8a5499ea9323421ed0cda06b345caab73": "lp_staging",
            "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70": "lp_companion",
            "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "defi_exit_channel",
        },
        "operators": [
            "0xe93d64f3fbc352131e79fc5578cbe44b66697f86",
            "0xfd51e33d44b376ef346d24a130a51035db09c1dc",
        ],
        "cashout_addrs": [
            "0xc6962004f452be9203591991d15f6b388e09e8d0",
        ],
        "treasury_addrs": [
            "0xf186cb00e49e18491db5783ff04fae3818102ff7",
            "0x360e68faccca8ca495c1b759fd9eee466db9fb32",
        ],
    },
    "org_002": {
        "label": "tx.origin Trap Campaign (Base-only)",
        "roles": {
            "0x238d7170f309a55b87a144a341bd6105897082ca": "treasury_senior",
            "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473": "treasury_junior",
        },
        "operators": [],  # 43 disposable deployers, not tracked individually
        "cashout_addrs": [],
        "treasury_addrs": [
            "0x238d7170f309a55b87a144a341bd6105897082ca",
            "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473",
        ],
    },
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp to datetime."""
    if not ts_str:
        return None
    try:
        # Handle various formats
        clean = ts_str.replace("+00:00", "").replace("Z", "").strip()
        if "T" in clean:
            return datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
        return None
    except (ValueError, TypeError):
        return None


def get_org_events(conn: sqlite3.Connection, org_id: str) -> List[Dict[str, Any]]:
    """Pull all timestamped events for an org."""
    org = ORGS.get(org_id)
    if not org:
        return []

    events = []
    all_addrs = set(org["roles"].keys())

    # 1. DEPLOYMENT events — contracts created by org operators
    operators = org["operators"]
    if operators:
        placeholders = ",".join("?" * len(operators))
        for r in conn.execute(
            f"SELECT deployer_address, contract_address, detection_timestamp, chain "
            f"FROM contracts WHERE deployer_address IN ({placeholders}) ORDER BY detection_timestamp",
            operators,
        ):
            ts = _parse_ts(r["detection_timestamp"])
            if ts:
                events.append({
                    "timestamp": ts,
                    "event_type": "deployment",
                    "detail": f"{r['deployer_address'][:14]}... deployed {r['contract_address'][:14]}... on {r['chain']}",
                    "address": r["deployer_address"],
                })

    # For org_002: identify deployers via tx.origin bytecode pattern
    # These are disposable deployers not tracked by funding_trail in the DB
    if org_id == "org_002":
        for r in conn.execute(
            """SELECT deployer_address, contract_address, detection_timestamp, chain
               FROM contracts
               WHERE bytecode_pattern_notes LIKE '%tx.origin%'
               ORDER BY detection_timestamp""",
        ):
            ts = _parse_ts(r["detection_timestamp"])
            if ts:
                events.append({
                    "timestamp": ts,
                    "event_type": "deployment",
                    "detail": f"{r['deployer_address'][:14]}... deployed {r['contract_address'][:14]}... on {r['chain']}",
                    "address": r["deployer_address"],
                })

        # Interactions on tx.origin contracts
        for r in conn.execute(
            """SELECT te.timestamp, te.interacting_address, te.contract_address, te.is_reverted
               FROM transaction_events te
               JOIN contracts c ON te.contract_address = c.contract_address
               WHERE c.bytecode_pattern_notes LIKE '%tx.origin%'
               ORDER BY te.timestamp""",
        ):
            ts = _parse_ts(r["timestamp"])
            if ts:
                events.append({
                    "timestamp": ts,
                    "event_type": "interaction",
                    "detail": f"{r['interacting_address'][:14]}... hit {r['contract_address'][:14]}... {'(REVERT)' if r['is_reverted'] else '(OK)'}",
                    "address": r["interacting_address"],
                })

    # 2. INTERACTION events — tx_events on org-linked contracts
    if operators:
        placeholders = ",".join("?" * len(operators))
        for r in conn.execute(
            f"""SELECT te.timestamp, te.interacting_address, te.contract_address, te.is_reverted
                FROM transaction_events te
                JOIN contracts c ON te.contract_address = c.contract_address
                WHERE c.deployer_address IN ({placeholders})
                ORDER BY te.timestamp""",
            operators,
        ):
            ts = _parse_ts(r["timestamp"])
            if ts:
                events.append({
                    "timestamp": ts,
                    "event_type": "interaction",
                    "detail": f"{r['interacting_address'][:14]}... hit {r['contract_address'][:14]}... {'(REVERT)' if r['is_reverted'] else '(OK)'}",
                    "address": r["interacting_address"],
                })

    # 3. CASHOUT events from alerts table
    cashout_addrs = org.get("cashout_addrs", [])
    for addr in cashout_addrs:
        for r in conn.execute(
            "SELECT timestamp, alert_type, tx_hash FROM alerts WHERE address = ? AND COALESCE(false_positive,0)=0 ORDER BY timestamp",
            (addr,),
        ):
            ts = _parse_ts(r["timestamp"])
            if ts:
                events.append({
                    "timestamp": ts,
                    "event_type": "cashout",
                    "detail": f"{r['alert_type']} on {addr[:14]}...",
                    "address": addr,
                })

    # 4. EXTRACTION events (documented live observations)
    if org_id == "org_001":
        for r in conn.execute("SELECT event_id, observed_at, total_usd_moved, summary FROM extraction_events"):
            ts = _parse_ts(r["observed_at"])
            if ts:
                events.append({
                    "timestamp": ts,
                    "event_type": "cashout",
                    "detail": f"{r['event_id']}: ${r['total_usd_moved']:,.0f} moved",
                    "address": "extraction_event",
                })

    # 5. FUNDING events from funding_trail on org deployers
    for addr in all_addrs:
        dep = conn.execute("SELECT funding_trail FROM deployers WHERE deployer_address = ?", (addr,)).fetchone()
        if dep and dep["funding_trail"]:
            try:
                trail = json.loads(dep["funding_trail"])
                ts = _parse_ts(trail.get("timestamp", ""))
                funder = trail.get("funder") or trail.get("funding_source") or "unknown"
                if ts:
                    events.append({
                        "timestamp": ts,
                        "event_type": "funding",
                        "detail": f"{funder[:14]}... funded {addr[:14]}... with {trail.get('value_eth', '?')} ETH",
                        "address": funder,
                    })
            except (json.JSONDecodeError, TypeError):
                pass

    # Deduplicate by timestamp+type+detail
    seen = set()
    unique = []
    for e in events:
        key = (e["timestamp"].isoformat(), e["event_type"], e["detail"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return sorted(unique, key=lambda e: e["timestamp"])


def identify_sessions(events: List[Dict], gap_hours: float = DEFAULT_GAP_HOURS) -> List[List[Dict]]:
    """Cluster events into sessions separated by gaps > gap_hours."""
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    sessions = []
    current = [sorted_events[0]]

    for event in sorted_events[1:]:
        gap = (event["timestamp"] - current[-1]["timestamp"]).total_seconds() / 3600
        if gap > gap_hours:
            sessions.append(current)
            current = [event]
        else:
            current.append(event)

    sessions.append(current)
    return sessions


def analyze_hourly(events: List[Dict]) -> Dict[str, Dict[int, int]]:
    """Count events per UTC hour by type."""
    hours = defaultdict(lambda: defaultdict(int))
    for e in events:
        hours[e["event_type"]][e["timestamp"].hour] += 1
    return hours


def analyze_dow(events: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Count events per day of week by type."""
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days = defaultdict(lambda: defaultdict(int))
    for e in events:
        day = names[e["timestamp"].weekday()]
        days[day][e["event_type"]] += 1
    return days


def calculate_lag(events: List[Dict], from_type: str, to_type: str) -> List[Dict]:
    """For each from_type event, find the next to_type event and calc lag."""
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    lags = []
    for i, e in enumerate(sorted_events):
        if e["event_type"] != from_type:
            continue
        for j in range(i + 1, len(sorted_events)):
            if sorted_events[j]["event_type"] == to_type:
                lag = (sorted_events[j]["timestamp"] - e["timestamp"]).total_seconds() / 3600
                lags.append({
                    "from": e, "to": sorted_events[j],
                    "lag_hours": round(lag, 2),
                })
                break
    return lags


def predict_next(sessions: List[List[Dict]]) -> Tuple[Optional[datetime], str]:
    """Predict next active window from session timing."""
    if len(sessions) < 2:
        return None, "INSUFFICIENT DATA (need 2+ sessions)"

    starts = [s[0]["timestamp"] for s in sessions]
    gaps = [(starts[i] - starts[i-1]).total_seconds() / 3600 for i in range(1, len(starts))]

    avg = sum(gaps) / len(gaps)
    std = (sum((g - avg) ** 2 for g in gaps) / len(gaps)) ** 0.5
    cv = std / avg if avg > 0 else 999

    predicted = starts[-1] + timedelta(hours=avg)

    if len(gaps) < 3:
        conf = f"LOW ({len(gaps)} cycles observed)"
    elif cv < 0.3:
        conf = f"HIGH (regular, CV={cv:.1%})"
    elif cv < 0.6:
        conf = f"MEDIUM (CV={cv:.1%})"
    else:
        conf = f"LOW (irregular, CV={cv:.1%})"

    return predicted, conf


def render_heatmap(hourly: Dict, types: List[str]) -> str:
    """ASCII heatmap of activity by UTC hour."""
    lines = ["```"]
    lines.append("Hour:  " + " ".join(f"{h:02d}" for h in range(24)))
    for t in types:
        counts = [hourly.get(t, {}).get(h, 0) for h in range(24)]
        mx = max(counts) if max(counts) > 0 else 1
        syms = []
        for c in counts:
            if c == 0:
                syms.append(" .")
            elif c / mx > 0.7:
                syms.append(" #")
            elif c / mx > 0.3:
                syms.append(" o")
            else:
                syms.append(" .")
        lines.append(f"{t[:5]:>5}: " + "".join(syms))
    lines.append("```")
    return "\n".join(lines)


def generate_report(org_id: Optional[str] = None, gap_hours: float = DEFAULT_GAP_HOURS) -> Path:
    """Generate the full cycle analysis report."""
    conn = get_connection()
    now = datetime.now(timezone.utc)

    orgs_to_analyze = {org_id: ORGS[org_id]} if org_id and org_id in ORGS else ORGS

    md = f"# ORGANIZATION ACTIVITY CYCLE ANALYSIS\n"
    md += f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
    md += f"**Session gap threshold:** {gap_hours} hours\n"
    md += f"**Organizations analyzed:** {len(orgs_to_analyze)}\n\n---\n\n"

    all_summaries = {}

    for oid, org in orgs_to_analyze.items():
        events = get_org_events(conn, oid)
        if not events:
            md += f"## {oid}: {org['label']}\n\nNo events found.\n\n---\n\n"
            continue

        types_present = sorted(set(e["event_type"] for e in events))
        type_counts = defaultdict(int)
        for e in events:
            type_counts[e["event_type"]] += 1

        ts_range = f"{events[0]['timestamp'].strftime('%Y-%m-%d %H:%M')} to {events[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')}"

        md += f"## {oid}: {org['label']}\n\n"
        md += f"**Events:** {len(events)} | **Range:** {ts_range}\n"
        md += f"**Event types:** " + ", ".join(f"{t}: {type_counts[t]}" for t in types_present) + "\n\n"

        # Hourly heatmap
        hourly = analyze_hourly(events)
        md += "### Activity Timeline (UTC hours)\n"
        md += render_heatmap(hourly, types_present)
        md += "\n\n"

        # Peak hours per type
        md += "### Peak Hours\n| Event Type | Peak Hour (UTC) | Events at Peak |\n|---|---|---|\n"
        for t in types_present:
            counts = hourly.get(t, {})
            if counts:
                peak_h = max(counts, key=counts.get)
                md += f"| {t} | {peak_h:02d}:00 | {counts[peak_h]} |\n"
        md += "\n"

        # Day of week
        dow = analyze_dow(events)
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        md += "### Day-of-Week Pattern\n| Day |"
        for t in types_present:
            md += f" {t} |"
        md += " Total |\n|---|"
        for _ in types_present:
            md += "---|"
        md += "---|\n"
        for day in day_order:
            row_total = sum(dow.get(day, {}).get(t, 0) for t in types_present)
            if row_total == 0:
                continue
            md += f"| {day} |"
            for t in types_present:
                md += f" {dow.get(day, {}).get(t, 0)} |"
            md += f" {row_total} |\n"
        md += "\n"

        # Sessions
        sessions = identify_sessions(events, gap_hours)
        md += f"### Activity Sessions ({len(sessions)} detected)\n"
        md += "| # | Start (UTC) | End (UTC) | Duration | Events | Types |\n|---|---|---|---|---|---|\n"

        session_durations = []
        session_gaps = []
        for i, sess in enumerate(sessions):
            start = sess[0]["timestamp"]
            end = sess[-1]["timestamp"]
            dur = (end - start).total_seconds() / 3600
            session_durations.append(dur)
            etypes = set(e["event_type"] for e in sess)
            md += f"| {i+1} | {start.strftime('%b %d %H:%M')} | {end.strftime('%b %d %H:%M')} | {dur:.1f}h | {len(sess)} | {'+'.join(sorted(etypes))} |\n"

            if i > 0:
                gap = (start - sessions[i-1][-1]["timestamp"]).total_seconds() / 3600
                session_gaps.append(gap)

        md += "\n"
        if session_durations:
            md += f"**Average session duration:** {sum(session_durations)/len(session_durations):.1f} hours\n"
        if session_gaps:
            md += f"**Average gap between sessions:** {sum(session_gaps)/len(session_gaps):.1f} hours\n"
        md += f"**Average events per session:** {len(events)/len(sessions):.0f}\n\n"

        # Funding -> Deployment lag
        fund_deploy_lags = calculate_lag(events, "funding", "deployment")
        if fund_deploy_lags:
            md += "### Funding -> Deployment Lag\n"
            md += "| Funding Event | First Deploy After | Lag (hours) |\n|---|---|---|\n"
            for lag in fund_deploy_lags[:10]:
                md += f"| {lag['from']['timestamp'].strftime('%b %d %H:%M')} | {lag['to']['timestamp'].strftime('%b %d %H:%M')} | {lag['lag_hours']:.2f} |\n"
            avg_fd = sum(l["lag_hours"] for l in fund_deploy_lags) / len(fund_deploy_lags)
            med_fd = sorted(l["lag_hours"] for l in fund_deploy_lags)[len(fund_deploy_lags)//2]
            md += f"\n**Average:** {avg_fd:.2f} hours | **Median:** {med_fd:.2f} hours\n\n"

        # Deployment -> Cashout lag
        deploy_cash_lags = calculate_lag(events, "deployment", "cashout")
        if deploy_cash_lags:
            md += "### Deployment -> Cashout Lag\n"
            md += "| Deploy Event | First Cashout After | Lag (hours) |\n|---|---|---|\n"
            for lag in deploy_cash_lags[:10]:
                md += f"| {lag['from']['timestamp'].strftime('%b %d %H:%M')} | {lag['to']['timestamp'].strftime('%b %d %H:%M')} | {lag['lag_hours']:.2f} |\n"
            avg_dc = sum(l["lag_hours"] for l in deploy_cash_lags) / len(deploy_cash_lags)
            md += f"\n**Average:** {avg_dc:.2f} hours\n\n"

        # Prediction
        predicted, confidence = predict_next(sessions)
        md += "### Cycle Assessment\n"
        if predicted:
            md += f"**Predicted next active window:** {predicted.strftime('%Y-%m-%d %H:%M UTC')}\n"
            md += f"**Confidence:** {confidence}\n"
        else:
            md += f"**Prediction:** {confidence}\n"
        md += "\n---\n\n"

        # Store for comparison
        all_summaries[oid] = {
            "sessions": len(sessions),
            "avg_duration": round(sum(session_durations)/len(session_durations), 1) if session_durations else 0,
            "avg_gap": round(sum(session_gaps)/len(session_gaps), 1) if session_gaps else 0,
            "peak_hour": max(range(24), key=lambda h: sum(hourly.get(t, {}).get(h, 0) for t in types_present)) if events else "N/A",
            "fund_deploy_lag": round(sum(l["lag_hours"] for l in fund_deploy_lags)/len(fund_deploy_lags), 2) if fund_deploy_lags else "N/A",
            "deploy_cash_lag": round(sum(l["lag_hours"] for l in deploy_cash_lags)/len(deploy_cash_lags), 2) if deploy_cash_lags else "N/A",
            "predicted": predicted.strftime("%Y-%m-%d %H:%M UTC") if predicted else "N/A",
            "confidence": confidence,
            "total_events": len(events),
        }

    # Cross-org comparison
    if len(all_summaries) > 1:
        md += "## Cross-Org Comparison\n\n"
        md += "| Metric |"
        for oid in all_summaries:
            md += f" {oid} |"
        md += "\n|---|"
        for _ in all_summaries:
            md += "---|"
        md += "\n"

        metrics = [
            ("Sessions observed", "sessions"),
            ("Avg session duration", "avg_duration"),
            ("Avg gap between sessions", "avg_gap"),
            ("Peak UTC hour", "peak_hour"),
            ("Funding -> deploy lag", "fund_deploy_lag"),
            ("Deploy -> cashout lag", "deploy_cash_lag"),
            ("Total events", "total_events"),
            ("Predicted next window", "predicted"),
            ("Prediction confidence", "confidence"),
        ]
        for label, key in metrics:
            md += f"| {label} |"
            for oid in all_summaries:
                val = all_summaries[oid].get(key, "N/A")
                if isinstance(val, float):
                    md += f" {val:.1f}h |"
                else:
                    md += f" {val} |"
            md += "\n"

    conn.close()

    # Save
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    ts = now.strftime("%Y%m%d_%H%M%S")
    filepath = CASES_DIR / f"ORG_CYCLES_{ts}.md"
    filepath.write_text(md, encoding="utf-8")
    print(f"Report generated: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze org activity cycles")
    parser.add_argument("--org", default=None, help="Specific org to analyze")
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_HOURS)
    args = parser.parse_args()
    generate_report(org_id=args.org, gap_hours=args.gap_threshold)
