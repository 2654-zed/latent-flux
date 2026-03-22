"""
Case File Generator for Layer 3 Surveillance System.

Usage:
    python -m surveillance.case_file 0x9da33e... --chain base
    python -m surveillance.case_file 0xb15e7a... --chain arbitrum
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
CASES_DIR = Path(__file__).resolve().parent / "data" / "cases"

ORG_001_ADDRS = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": "treasury",
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "operator",
    "0xfd51e33d44b376ef346d24a130a51035db09c1dc": "operator_2",
    "0xc6962004f452be9203591991d15f6b388e09e8d0": "cashout",
    "0x01989c93890aed05a63d179b03424997075b6acf": "exit_cex",
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": "gas_station",
    "0x360e68faccca8ca495c1b759fd9eee466db9fb32": "treasury_branch",
    "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "laundry",
    "0x96daa0b8a5499ea9323421ed0cda06b345caab73": "lp_staging",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "defi_exit_channel",
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_address(conn: sqlite3.Connection, prefix: str) -> Optional[str]:
    """Resolve a partial address prefix to full address."""
    r = conn.execute(
        "SELECT contract_address FROM contracts WHERE contract_address LIKE ?",
        (prefix.lower() + "%",),
    ).fetchone()
    return r[0] if r else None


def get_contract_info(conn: sqlite3.Connection, address: str) -> Dict[str, Any]:
    """Pull all surveillance data for a contract."""
    r = conn.execute("SELECT * FROM contracts WHERE contract_address = ?", (address,)).fetchone()
    if not r:
        return {}
    d = dict(r)

    # Parse trap mechanisms from bytecode_pattern_notes
    mechanisms = []
    notes = d.get("bytecode_pattern_notes") or ""
    for part in notes.split(";"):
        part = part.strip()
        if not part:
            continue
        if "conditional revert" in part.lower():
            mechanisms.append({"name": "Conditional Revert", "pattern": part, "effect": "Reverts transfers for non-privileged callers"})
        elif "tx.origin" in part.lower():
            mechanisms.append({"name": "Buy-Not-Sell Gate", "pattern": part, "effect": "Allows buying but blocks selling for most users"})
        elif "fee-on-transfer" in part.lower() or "obfuscated fee" in part.lower():
            mechanisms.append({"name": "Obfuscated Fee Extraction", "pattern": part, "effect": "Silently skims a percentage on every transfer"})
        elif "selfdestruct" in part.lower():
            mechanisms.append({"name": "Self-Destruct Capability", "pattern": part, "effect": "Contract can be destroyed post-deployment, erasing evidence"})
        elif "delegatecall" in part.lower():
            mechanisms.append({"name": "Upgradeable Logic (DELEGATECALL)", "pattern": part, "effect": "Transfer logic can be changed post-deployment"})
        elif "timestamp" in part.lower() and "activation" in part.lower():
            mechanisms.append({"name": "Time-Locked Activation", "pattern": part, "effect": "Trap activates after a specific timestamp"})
        else:
            mechanisms.append({"name": "Other Pattern", "pattern": part, "effect": "See pattern details"})

    d["trap_mechanisms"] = mechanisms

    # Infer contract type
    if any("fee" in m["name"].lower() for m in mechanisms):
        d["contract_type"] = "Fee-Skimming Token"
    elif any("buy-not-sell" in m["name"].lower() for m in mechanisms):
        d["contract_type"] = "Honeypot (Buy-Not-Sell)"
    elif any("delegatecall" in m["name"].lower() for m in mechanisms):
        d["contract_type"] = "Upgradeable Proxy Trap"
    elif any("self-destruct" in m["name"].lower() for m in mechanisms):
        d["contract_type"] = "Ephemeral Trap"
    elif mechanisms:
        d["contract_type"] = "Multi-Mechanism Trap"
    else:
        d["contract_type"] = "Unknown"

    # Bytecode cache
    cache = conn.execute(
        "SELECT code_hash, bytecode_signals FROM bytecode_cache WHERE source_contract = ?",
        (address,),
    ).fetchone()
    d["code_hash"] = cache["code_hash"] if cache else "N/A (not in cache)"
    d["bytecode_signals"] = cache["bytecode_signals"] if cache else "{}"

    return d


def get_deployer_info(conn: sqlite3.Connection, deployer_address: str) -> Dict[str, Any]:
    """Pull deployer profile."""
    r = conn.execute("SELECT * FROM deployers WHERE deployer_address = ?", (deployer_address,)).fetchone()
    d = dict(r) if r else {"deployer_address": deployer_address}

    # Funding trail
    funding = {}
    trail = d.get("funding_trail")
    if trail:
        try:
            funding = json.loads(trail)
        except (json.JSONDecodeError, TypeError):
            pass
    d["funding"] = funding

    # Other contracts from this deployer
    others = conn.execute(
        "SELECT contract_address, confidence_tier, bytecode_pattern_notes, detection_timestamp FROM contracts WHERE deployer_address = ? ORDER BY detection_timestamp",
        (deployer_address,),
    ).fetchall()
    d["linked_contracts"] = [dict(o) for o in others]
    d["total_deployments"] = len(others)

    # Behavioral score breakdown
    bd = {}
    if d.get("score_breakdown"):
        try:
            bd = json.loads(d["score_breakdown"])
        except (json.JSONDecodeError, TypeError):
            pass
    d["score_detail"] = bd

    return d


def get_traffic_analysis(conn: sqlite3.Connection, address: str) -> Dict[str, Any]:
    """Pull call counts, callers, selectors, reverts."""
    row = conn.execute("""
        SELECT COUNT(*) as total_calls,
               COUNT(DISTINCT interacting_address) as unique_callers,
               SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts,
               MIN(timestamp) as first_interaction,
               MAX(timestamp) as last_interaction
        FROM transaction_events WHERE contract_address = ?
    """, (address,)).fetchone()

    total = row["total_calls"]
    reverts = row["reverts"] or 0
    revert_rate = (reverts / total * 100) if total > 0 else 0

    # Selector breakdown
    selectors = []
    for r in conn.execute("""
        SELECT function_selector, COUNT(*) as calls,
               COUNT(DISTINCT interacting_address) as callers,
               SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts
        FROM transaction_events WHERE contract_address = ?
        GROUP BY function_selector ORDER BY calls DESC
    """, (address,)):
        sr = (r["reverts"] / r["calls"] * 100) if r["calls"] > 0 else 0
        ks = conn.execute("SELECT tag FROM known_selectors WHERE function_selector = ?", (r["function_selector"],)).fetchone()
        tag = ks[0] if ks else "unknown"
        selectors.append({
            "selector": r["function_selector"],
            "calls": r["calls"],
            "callers": r["callers"],
            "revert_rate": round(sr, 1),
            "tag": tag,
        })

    # Known bots among callers
    bot_count = conn.execute("""
        SELECT COUNT(*) FROM bot_candidates WHERE address IN (
            SELECT DISTINCT interacting_address FROM transaction_events WHERE contract_address = ?
        )
    """, (address,)).fetchone()[0]

    return {
        "total_calls": total,
        "unique_callers": row["unique_callers"] or 0,
        "reverts": reverts,
        "revert_rate": round(revert_rate, 1),
        "first_interaction": row["first_interaction"] or "N/A",
        "last_interaction": row["last_interaction"] or "N/A",
        "selectors": selectors,
        "known_bots": bot_count,
    }


def get_org_links(conn: sqlite3.Connection, deployer_address: str) -> Dict[str, Any]:
    """Check organizational connections."""
    dep = conn.execute(
        "SELECT entity_type, deployment_pattern_notes FROM deployers WHERE deployer_address = ?",
        (deployer_address,),
    ).fetchone()

    entity_type = dep["entity_type"] if dep else "unknown"
    notes = dep["deployment_pattern_notes"] if dep else ""

    # Check against known org addresses
    org_role = ORG_001_ADDRS.get(deployer_address.lower())

    # Check pattern_matches
    pm = conn.execute(
        "SELECT pattern_type, confidence, status FROM pattern_matches WHERE address = ?",
        (deployer_address,),
    ).fetchone()

    # Check if any linked deployer is in an org
    linked_orgs = []
    funding = conn.execute(
        "SELECT funding_trail FROM deployers WHERE deployer_address = ?",
        (deployer_address,),
    ).fetchone()
    if funding and funding["funding_trail"]:
        try:
            trail = json.loads(funding["funding_trail"])
            funder = (trail.get("funder") or trail.get("funding_source") or "").lower()
            if funder in ORG_001_ADDRS:
                linked_orgs.append({"org": "org_001", "connection": f"funded by {ORG_001_ADDRS[funder]} ({funder[:14]}...)"})
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "entity_type": entity_type,
        "org_role": org_role,
        "pattern_match": dict(pm) if pm else None,
        "linked_orgs": linked_orgs,
        "notes": notes or "N/A",
    }


def get_timeline(conn: sqlite3.Connection, address: str, deployer_address: str) -> List[Dict]:
    """Build chronological event timeline."""
    events = []

    # Deployer funding
    dep = conn.execute("SELECT first_seen, funding_trail FROM deployers WHERE deployer_address = ?", (deployer_address,)).fetchone()
    if dep and dep["funding_trail"]:
        try:
            trail = json.loads(dep["funding_trail"])
            ts = trail.get("timestamp", dep["first_seen"])
            funder = trail.get("funder") or trail.get("funding_source") or "unknown"
            eth = trail.get("value_eth", "?")
            events.append({"timestamp": ts[:19] if ts else "?", "event": f"Deployer funded with {eth} ETH from {funder[:18]}..."})
        except (json.JSONDecodeError, TypeError):
            pass

    # Contract deployment
    c = conn.execute("SELECT detection_timestamp, detection_block FROM contracts WHERE contract_address = ?", (address,)).fetchone()
    if c:
        events.append({"timestamp": c["detection_timestamp"][:19], "event": f"Contract deployed at block {c['detection_block']}"})

    # First and last interactions
    first = conn.execute("SELECT MIN(timestamp) as ts, interacting_address FROM transaction_events WHERE contract_address = ?", (address,)).fetchone()
    last = conn.execute("SELECT MAX(timestamp) as ts FROM transaction_events WHERE contract_address = ?", (address,)).fetchone()
    if first and first["ts"]:
        events.append({"timestamp": first["ts"][:19], "event": f"First interaction from {first['interacting_address'][:18]}..."})
    if last and last["ts"] and last["ts"] != (first["ts"] if first else None):
        count = conn.execute("SELECT COUNT(DISTINCT interacting_address) FROM transaction_events WHERE contract_address = ?", (address,)).fetchone()[0]
        events.append({"timestamp": last["ts"][:19], "event": f"Last interaction ({count} unique callers total)"})

    # Confirmation if exists
    conf = conn.execute("SELECT confirmation_timestamp FROM contracts WHERE contract_address = ? AND confirmation_timestamp IS NOT NULL", (address,)).fetchone()
    if conf:
        events.append({"timestamp": conf["confirmation_timestamp"][:19], "event": "Escalated to CONFIRMED"})

    events.sort(key=lambda x: x["timestamp"])
    return events


def classify_threat(victim_count: int, revert_rate: float, trap_count: int, deployer_history: int) -> Dict[str, str]:
    """Assign risk ratings."""
    if victim_count > 500:
        victim_rating = "CRITICAL"
    elif victim_count > 100:
        victim_rating = "HIGH"
    elif victim_count > 20:
        victim_rating = "MODERATE"
    else:
        victim_rating = "LOW"

    if trap_count >= 3:
        sophistication = "HIGH"
    elif trap_count >= 2:
        sophistication = "MODERATE"
    elif trap_count >= 1:
        sophistication = "LOW"
    else:
        sophistication = "NONE DETECTED"

    if revert_rate < 15 and trap_count > 0:
        camouflage = "HIGH"
    elif revert_rate < 30 and trap_count > 0:
        camouflage = "MODERATE"
    else:
        camouflage = "LOW"

    if deployer_history >= 5:
        deployer_risk = "HIGH"
    elif deployer_history >= 2:
        deployer_risk = "MODERATE"
    else:
        deployer_risk = "LOW"

    ratings = [victim_rating, sophistication, camouflage, deployer_risk]
    if "CRITICAL" in ratings:
        overall = "CRITICAL"
    elif ratings.count("HIGH") >= 2:
        overall = "HIGH"
    elif "HIGH" in ratings:
        overall = "MODERATE-HIGH"
    else:
        overall = "MODERATE"

    return {
        "victim_count": victim_rating,
        "sophistication": sophistication,
        "camouflage": camouflage,
        "deployer_history": deployer_risk,
        "overall": overall,
    }


def build_markdown(
    contract: Dict, deployer: Dict, traffic: Dict,
    org: Dict, timeline: List[Dict], threat: Dict, chain: str,
) -> str:
    """Assemble all sections into the final Markdown document."""
    addr = contract["contract_address"]
    short = addr[:10]
    tier = contract.get("confidence_tier", "unknown").upper()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    first_seen = contract.get("detection_timestamp", "N/A")[:19]
    last_activity = traffic.get("last_interaction", "N/A")[:19]

    mechanisms = contract.get("trap_mechanisms", [])
    trap_count = len(mechanisms)

    # Executive summary
    summary_parts = []
    if traffic["unique_callers"] > 0:
        summary_parts.append(f"{traffic['unique_callers']} unique addresses interacted with this contract")
    if traffic["revert_rate"] > 0:
        summary_parts.append(f"{traffic['revert_rate']}% revert rate")
    if trap_count > 0:
        summary_parts.append(f"{trap_count} trap mechanism(s) detected in bytecode")
    summary = ". ".join(summary_parts) + "." if summary_parts else "No interaction data available."

    md = f"""# CASE FILE: {short}
**Classification:** {tier}
**Generated:** {now}
**Surveillance Period:** {first_seen} -> {last_activity}

---

## Executive Summary
{contract.get('contract_type', 'Unknown')} contract on {chain.title()}. {summary} Deployer `{contract.get('deployer_address', 'N/A')[:18]}...` has deployed {deployer.get('total_deployments', 'N/A')} contracts total. Threat level: **{threat['overall']}**.

---

## Contract Identity
| Field | Value |
|---|---|
| Address | `{addr}` |
| Chain | {chain.title()} |
| Deployed | Block {contract.get('detection_block', 'N/A')}, {first_seen} |
| Deployer | `{contract.get('deployer_address', 'N/A')}` |
| Confidence Tier | {tier} |
| Detection Method | {contract.get('detection_method', 'N/A')} |
| Contract Type | {contract.get('contract_type', 'Unknown')} |
| Bytecode Hash | `{contract.get('code_hash', 'N/A')}` |

---

## Trap Mechanisms
"""
    if mechanisms:
        for i, m in enumerate(mechanisms, 1):
            md += f"""
### Mechanism {i}: {m['name']}
**Pattern:** {m['pattern']}
**Effect:** {m['effect']}
**Detection Method:** Static bytecode analysis (opcode pattern matching)
"""
    else:
        md += "\nNo trap mechanisms detected in bytecode analysis.\n"

    # Bytecode signals
    signals = contract.get("bytecode_signals", "{}")
    try:
        sig = json.loads(signals) if isinstance(signals, str) else signals
        if sig:
            md += "\n### Bytecode Signal Summary\n"
            md += "| Signal | Value |\n|---|---|\n"
            for k, v in sig.items():
                md += f"| {k} | {v} |\n"
    except (json.JSONDecodeError, TypeError):
        pass

    md += f"""
---

## Traffic Analysis
| Metric | Value |
|---|---|
| Total Calls | {traffic['total_calls']:,} |
| Unique Callers | {traffic['unique_callers']:,} |
| Revert Rate | {traffic['revert_rate']}% ({traffic['reverts']:,} reverted) |
| Known Bots Among Callers | {traffic['known_bots']} |
| First Interaction | {traffic['first_interaction'][:19] if traffic['first_interaction'] != 'N/A' else 'N/A'} |
| Last Interaction | {traffic['last_interaction'][:19] if traffic['last_interaction'] != 'N/A' else 'N/A'} |

### Function Selector Breakdown
| Selector | Calls | Unique Callers | Revert Rate | Likely Function |
|---|---|---|---|---|
"""
    for sel in traffic["selectors"]:
        md += f"| `{sel['selector']}` | {sel['calls']:,} | {sel['callers']:,} | {sel['revert_rate']}% | {sel['tag']} |\n"

    md += f"""
---

## Token Flows
Token flow data requires on-chain verification via Alchemy `getAssetTransfers` API. Not available from surveillance database alone. See investigation scripts for manually traced flows.

---

## Deployer Profile
| Field | Value |
|---|---|
| Address | `{deployer.get('deployer_address', 'N/A')}` |
| Chain | {deployer.get('chain', 'N/A')} |
| Total Deployments | {deployer.get('total_deployments', 'N/A')} |
| First Seen | {str(deployer.get('first_seen', 'N/A'))[:19]} |
| Last Seen | {str(deployer.get('last_seen', 'N/A'))[:19]} |
| Entity Type | {deployer.get('entity_type', 'unknown')} |
| Behavioral Score | {deployer.get('behavioral_score', 'N/A')} |
| Funding Source | {deployer.get('funding', {}).get('funder', 'N/A')} |
| Funding Amount | {deployer.get('funding', {}).get('value_eth', 'N/A')} ETH |

### Linked Contracts
| Address | Tier | Patterns | Deployed |
|---|---|---|---|
"""
    for lc in deployer.get("linked_contracts", []):
        lc_addr = lc["contract_address"]
        lc_tier = lc.get("confidence_tier", "?")
        lc_notes = (lc.get("bytecode_pattern_notes") or "none")[:60]
        lc_ts = str(lc.get("detection_timestamp", "?"))[:19]
        marker = " **(this contract)**" if lc_addr == addr else ""
        md += f"| `{lc_addr[:18]}...`{marker} | {lc_tier} | {lc_notes} | {lc_ts} |\n"

    md += f"""
---

## Organizational Links
"""
    if org["org_role"]:
        md += f"**Organization:** org_001\n**Role:** {org['org_role']}\n"
    elif org["linked_orgs"]:
        for lo in org["linked_orgs"]:
            md += f"**Organization:** {lo['org']}\n**Connection:** {lo['connection']}\n"
    elif org["entity_type"] not in ("unknown", None):
        md += f"**Entity Type:** {org['entity_type']}\n**Notes:** {org['notes']}\n"
    else:
        md += "No known organizational links. Independent operator or unclassified.\n"

    if org["pattern_match"]:
        pm = org["pattern_match"]
        md += f"\n**Pattern Match:** {pm.get('pattern_type', 'N/A')} (confidence: {pm.get('confidence', 'N/A')}, status: {pm.get('status', 'N/A')})\n"

    md += f"""
---

## Kill Chain Timeline
| Timestamp (UTC) | Event |
|---|---|
"""
    for ev in timeline:
        md += f"| {ev['timestamp']} | {ev['event']} |\n"

    if not timeline:
        md += "| N/A | No timeline events available |\n"

    md += f"""
---

## Risk Assessment
| Factor | Rating | Notes |
|---|---|---|
| Victim Count | {threat['victim_count']} | {traffic['unique_callers']:,} unique addresses |
| Trap Sophistication | {threat['sophistication']} | {trap_count} layered mechanism(s) |
| Revert Camouflage | {threat['camouflage']} | {traffic['revert_rate']}% revert looks {'normal' if traffic['revert_rate'] < 20 else 'suspicious'} |
| Deployer History | {threat['deployer_history']} | {deployer.get('total_deployments', 0)} contracts from this deployer |
| Active Status | {'ACTIVE' if traffic['last_interaction'] != 'N/A' else 'UNKNOWN'} | Last interaction: {traffic['last_interaction'][:10] if traffic['last_interaction'] != 'N/A' else 'N/A'} |

**Overall Threat Level: {threat['overall']}**

---

## Evidence Hashes
| Evidence | SHA256 |
|---|---|
| Contract bytecode | `{contract.get('code_hash', 'requires on-chain verification')}` |
"""

    md += f"""
---

## Recommended Actions
- [ ] Escalate to CONFIRMED if not already
- [ ] Submit to Chainabuse.com with this case file
- [ ] Submit to ScamSniffer database
- [ ] Monitor deployer `{contract.get('deployer_address', 'N/A')[:18]}...` for next deployment (current nonce: {deployer.get('total_deployments', '?')})
- [ ] Cross-reference deployer on {'Base' if chain == 'arbitrum' else 'Arbitrum'} for linked operations
- [ ] Trace token flows via Alchemy getAssetTransfers for USD extraction estimate
"""
    return md


def generate_case_file(address: str, chain: str = "base") -> Path:
    """Generate the full case file."""
    conn = get_connection()
    try:
        full_addr = _resolve_address(conn, address)
        if not full_addr:
            print(f"Contract not found in surveillance database: {address}")
            return None

        contract = get_contract_info(conn, full_addr)
        if not contract:
            print(f"No contract data for: {full_addr}")
            return None

        deployer_addr = contract["deployer_address"]
        deployer = get_deployer_info(conn, deployer_addr)
        traffic = get_traffic_analysis(conn, full_addr)
        org = get_org_links(conn, deployer_addr)
        timeline = get_timeline(conn, full_addr, deployer_addr)
        threat = classify_threat(
            traffic["unique_callers"],
            traffic["revert_rate"],
            len(contract.get("trap_mechanisms", [])),
            deployer.get("total_deployments", 0),
        )

        md = build_markdown(contract, deployer, traffic, org, timeline, threat, chain)

        CASES_DIR.mkdir(parents=True, exist_ok=True)
        short = full_addr[:10].lower()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"CASE_{short}_{chain}_{ts}.md"
        filepath = CASES_DIR / filename

        # Write and append self-hash
        file_hash = hashlib.sha256(md.encode()).hexdigest()
        md += f"| This case file | `{file_hash}` |\n"
        filepath.write_text(md, encoding="utf-8")

        print(f"Case file generated: {filepath}")
        print(f"  Contract: {full_addr}")
        print(f"  Threat Level: {threat['overall']}")
        print(f"  Victims: {traffic['unique_callers']}")
        print(f"  Trap Mechanisms: {len(contract.get('trap_mechanisms', []))}")
        return filepath

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surveillance case file")
    parser.add_argument("address", help="Contract address (full or prefix)")
    parser.add_argument("--chain", default="base", help="Chain (base or arbitrum)")
    args = parser.parse_args()
    generate_case_file(args.address, args.chain)
