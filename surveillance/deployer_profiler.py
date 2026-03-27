"""
Layer 3 — Deployer Behavioral Profiling

Builds a composite identity fingerprint for each deployer based on:
1. Timezone (deployment hour distribution → peak activity window)
2. Gas fingerprint (gas price range and variance)
3. Technique mix (which trap patterns they use)
4. Chain preference (Base vs Arbitrum ratio)
5. Template diversity (how many bytecode families they use)
6. Deployment cadence (contracts per day, burst vs steady)
7. Operational style (selector count, interaction patterns)

These dimensions persist across wallet rotation — a deployer who burns
their wallet and starts fresh will still deploy at the same hours, use
the same gas settings, favor the same chain, and reach for the same
trap techniques.

Usage:
    python -m surveillance.deployer_profiler --profile-all
    python -m surveillance.deployer_profiler --find-similar 0xADDRESS
    python -m surveillance.deployer_profiler --cluster
"""

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------

def extract_profile(conn, deployer_address: str) -> dict:
    """Extract all behavioral features for a single deployer."""

    profile = {"deployer_address": deployer_address}

    # Basic stats
    d = conn.execute(
        "SELECT * FROM deployers WHERE deployer_address=?",
        (deployer_address,)
    ).fetchone()
    if not d:
        return profile

    profile["chain"] = d["chain"]
    profile["total_contracts"] = d["total_contracts_deployed"]

    # --- 1. Timezone / Work Schedule ---
    hours = conn.execute("""
        SELECT CAST(substr(detection_timestamp, 12, 2) AS INTEGER) as hour, COUNT(*) as cnt
        FROM contracts WHERE deployer_address=?
        GROUP BY hour ORDER BY cnt DESC
    """, (deployer_address,)).fetchall()

    hour_dist = [0] * 24
    total_deploys = 0
    for h in hours:
        if h["hour"] is not None:
            hour_dist[h["hour"]] = h["cnt"]
            total_deploys += h["cnt"]

    if total_deploys > 0:
        # Peak hour and concentration
        peak_hour = max(range(24), key=lambda i: hour_dist[i])
        peak_count = hour_dist[peak_hour]
        concentration = peak_count / total_deploys  # 1.0 = all in one hour

        # Find the 4-hour window with most activity (timezone proxy)
        best_window_start = 0
        best_window_count = 0
        for start in range(24):
            window_count = sum(hour_dist[(start + i) % 24] for i in range(4))
            if window_count > best_window_count:
                best_window_count = window_count
                best_window_start = start

        window_pct = best_window_count / total_deploys

        # Infer timezone
        # Map peak 4h window midpoint to likely timezone
        mid = (best_window_start + 2) % 24
        if 6 <= mid <= 10:
            tz_guess = "asia_morning"
        elif 11 <= mid <= 15:
            tz_guess = "europe_business"
        elif 16 <= mid <= 20:
            tz_guess = "americas_afternoon"
        elif 21 <= mid or mid <= 1:
            tz_guess = "americas_evening"
        else:
            tz_guess = "night_shift"

        profile["peak_hour"] = peak_hour
        profile["hour_concentration"] = round(concentration, 3)
        profile["active_window_start"] = best_window_start
        profile["active_window_pct"] = round(window_pct, 3)
        profile["timezone_guess"] = tz_guess
        profile["hour_distribution"] = hour_dist
    else:
        profile["peak_hour"] = None
        profile["hour_concentration"] = 0
        profile["active_window_start"] = None
        profile["active_window_pct"] = 0
        profile["timezone_guess"] = "unknown"
        profile["hour_distribution"] = hour_dist

    # --- 2. Gas Fingerprint ---
    gas = conn.execute("""
        SELECT ROUND(AVG(te.gas_price_gwei), 6) as avg_gas,
               ROUND(MIN(te.gas_price_gwei), 6) as min_gas,
               ROUND(MAX(te.gas_price_gwei), 6) as max_gas,
               COUNT(*) as gas_samples
        FROM transaction_events te
        WHERE te.interacting_address=?
    """, (deployer_address,)).fetchone()

    profile["gas_avg"] = gas["avg_gas"]
    profile["gas_min"] = gas["min_gas"]
    profile["gas_max"] = gas["max_gas"]
    profile["gas_samples"] = gas["gas_samples"] or 0
    if gas["max_gas"] and gas["min_gas"] and gas["min_gas"] > 0:
        profile["gas_variance_ratio"] = round(gas["max_gas"] / gas["min_gas"], 3)
    else:
        profile["gas_variance_ratio"] = None

    # --- 3. Technique Mix ---
    techniques = conn.execute("""
        SELECT
            SUM(CASE WHEN bytecode_pattern_notes LIKE '%tx.origin%' THEN 1 ELSE 0 END) as tx_origin,
            SUM(CASE WHEN bytecode_pattern_notes LIKE '%callback%' THEN 1 ELSE 0 END) as callback,
            SUM(CASE WHEN bytecode_pattern_notes LIKE '%fee%' THEN 1 ELSE 0 END) as fee,
            SUM(CASE WHEN bytecode_pattern_notes LIKE '%DELEGATECALL%' THEN 1 ELSE 0 END) as delegatecall,
            SUM(CASE WHEN bytecode_pattern_notes LIKE '%CALLER%' THEN 1 ELSE 0 END) as caller_check,
            SUM(CASE WHEN bytecode_pattern_notes IS NULL OR bytecode_pattern_notes = '' THEN 1 ELSE 0 END) as clean,
            COUNT(*) as total
        FROM contracts WHERE deployer_address=?
    """, (deployer_address,)).fetchone()

    total_c = techniques["total"] or 1
    technique_vec = {
        "tx_origin": techniques["tx_origin"] or 0,
        "callback": techniques["callback"] or 0,
        "fee": techniques["fee"] or 0,
        "delegatecall": techniques["delegatecall"] or 0,
        "caller_check": techniques["caller_check"] or 0,
        "clean": techniques["clean"] or 0,
    }
    # Normalize to ratios
    technique_ratios = {k: round(v / total_c, 3) for k, v in technique_vec.items()}
    profile["technique_counts"] = technique_vec
    profile["technique_ratios"] = technique_ratios

    active_techniques = [k for k, v in technique_vec.items() if v > 0 and k != "clean"]
    profile["technique_count"] = len(active_techniques)
    profile["primary_technique"] = max(
        [(k, v) for k, v in technique_vec.items() if k != "clean"],
        key=lambda x: x[1]
    )[0] if active_techniques else "none"

    # --- 4. Chain Preference ---
    chains = conn.execute("""
        SELECT chain, COUNT(*) as cnt FROM contracts
        WHERE deployer_address=? GROUP BY chain
    """, (deployer_address,)).fetchall()

    chain_counts = {r["chain"]: r["cnt"] for r in chains}
    total_chain = sum(chain_counts.values()) or 1
    base_ratio = chain_counts.get("base", 0) / total_chain
    profile["base_ratio"] = round(base_ratio, 3)
    profile["chain_preference"] = (
        "base_only" if base_ratio == 1.0
        else "arb_only" if base_ratio == 0.0
        else "mixed"
    )

    # --- 5. Template Diversity ---
    families = conn.execute("""
        SELECT COUNT(DISTINCT family_id) as families, COUNT(*) as members
        FROM bytecode_family_members WHERE deployer=?
    """, (deployer_address,)).fetchone()

    profile["bytecode_families"] = families["families"] or 0
    profile["family_members"] = families["members"] or 0

    # --- 6. Deployment Cadence ---
    dates = conn.execute("""
        SELECT DATE(detection_timestamp) as day, COUNT(*) as cnt
        FROM contracts WHERE deployer_address=?
        GROUP BY day ORDER BY day
    """, (deployer_address,)).fetchall()

    if len(dates) > 0:
        daily_counts = [r["cnt"] for r in dates]
        profile["active_days"] = len(dates)
        profile["avg_contracts_per_day"] = round(sum(daily_counts) / len(daily_counts), 1)
        profile["max_contracts_per_day"] = max(daily_counts)
        profile["deployment_style"] = (
            "burst" if len(dates) <= 2 and sum(daily_counts) > 10
            else "steady" if len(dates) >= 4
            else "sporadic"
        )
    else:
        profile["active_days"] = 0
        profile["avg_contracts_per_day"] = 0
        profile["max_contracts_per_day"] = 0
        profile["deployment_style"] = "unknown"

    # --- 7. Operational Style ---
    ops = conn.execute("""
        SELECT COUNT(DISTINCT function_selector) as selectors_used,
               COUNT(DISTINCT contract_address) as contracts_interacted,
               COUNT(*) as total_interactions,
               SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) as reverts
        FROM transaction_events WHERE interacting_address=?
    """, (deployer_address,)).fetchone()

    profile["selectors_used"] = ops["selectors_used"] or 0
    profile["contracts_interacted"] = ops["contracts_interacted"] or 0
    profile["total_interactions"] = ops["total_interactions"] or 0
    profile["interaction_revert_rate"] = round(
        (ops["reverts"] or 0) / max(ops["total_interactions"] or 1, 1) * 100, 1
    )

    # --- Org link ---
    trail = d["funding_trail"]
    if trail:
        try:
            t = json.loads(trail)
            profile["org_link"] = t.get("org_link", "")
            profile["funder"] = t.get("funder", "")
        except Exception:
            profile["org_link"] = ""
            profile["funder"] = ""
    else:
        profile["org_link"] = ""
        profile["funder"] = ""

    return profile


# ---------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------

def compute_similarity(p1: dict, p2: dict) -> dict:
    """
    Compute behavioral similarity between two deployer profiles.
    Returns a score 0-1 for each dimension and a composite score.
    """
    scores = {}

    # 1. Timezone similarity (compare hour distributions)
    h1 = p1.get("hour_distribution", [0] * 24)
    h2 = p2.get("hour_distribution", [0] * 24)
    t1 = sum(h1) or 1
    t2 = sum(h2) or 1
    # Cosine similarity on hour distributions
    dot = sum(h1[i] / t1 * h2[i] / t2 for i in range(24))
    mag1 = math.sqrt(sum((h1[i] / t1) ** 2 for i in range(24)))
    mag2 = math.sqrt(sum((h2[i] / t2) ** 2 for i in range(24)))
    scores["timezone"] = round(dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0, 3)

    # 2. Gas fingerprint similarity
    g1 = p1.get("gas_avg")
    g2 = p2.get("gas_avg")
    if g1 and g2 and g1 > 0 and g2 > 0:
        ratio = min(g1, g2) / max(g1, g2)
        scores["gas"] = round(ratio, 3)
    else:
        scores["gas"] = 0

    # 3. Technique similarity (cosine on technique ratio vectors)
    tr1 = p1.get("technique_ratios", {})
    tr2 = p2.get("technique_ratios", {})
    keys = set(list(tr1.keys()) + list(tr2.keys()))
    dot = sum(tr1.get(k, 0) * tr2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(tr1.get(k, 0) ** 2 for k in keys))
    mag2 = math.sqrt(sum(tr2.get(k, 0) ** 2 for k in keys))
    scores["technique"] = round(dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0, 3)

    # 4. Chain preference
    b1 = p1.get("base_ratio", 0.5)
    b2 = p2.get("base_ratio", 0.5)
    scores["chain"] = round(1.0 - abs(b1 - b2), 3)

    # 5. Deployment cadence similarity
    avg1 = p1.get("avg_contracts_per_day", 0)
    avg2 = p2.get("avg_contracts_per_day", 0)
    if avg1 > 0 and avg2 > 0:
        scores["cadence"] = round(min(avg1, avg2) / max(avg1, avg2), 3)
    else:
        scores["cadence"] = 0

    # 6. Scale similarity (total contracts)
    c1 = p1.get("total_contracts", 0)
    c2 = p2.get("total_contracts", 0)
    if c1 > 0 and c2 > 0:
        scores["scale"] = round(min(c1, c2) / max(c1, c2), 3)
    else:
        scores["scale"] = 0

    # Composite score (weighted)
    weights = {
        "timezone": 0.25,
        "gas": 0.15,
        "technique": 0.25,
        "chain": 0.10,
        "cadence": 0.15,
        "scale": 0.10,
    }
    composite = sum(scores.get(k, 0) * w for k, w in weights.items())
    scores["composite"] = round(composite, 3)

    return scores


# ---------------------------------------------------------------
# Main operations
# ---------------------------------------------------------------

def profile_all():
    """Profile all deployers with 3+ contracts."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS deployer_profiles (
            deployer_address TEXT PRIMARY KEY,
            total_contracts INTEGER,
            chain_preference TEXT,
            base_ratio REAL,
            timezone_guess TEXT,
            peak_hour INTEGER,
            active_window_start INTEGER,
            active_window_pct REAL,
            hour_concentration REAL,
            gas_avg REAL,
            gas_min REAL,
            gas_max REAL,
            gas_variance_ratio REAL,
            primary_technique TEXT,
            technique_count INTEGER,
            technique_ratios TEXT,
            bytecode_families INTEGER,
            active_days INTEGER,
            avg_contracts_per_day REAL,
            max_contracts_per_day INTEGER,
            deployment_style TEXT,
            selectors_used INTEGER,
            contracts_interacted INTEGER,
            interaction_revert_rate REAL,
            org_link TEXT,
            funder TEXT,
            hour_distribution TEXT,
            profiled_at TEXT
        )
    """)

    deployers = conn.execute(
        "SELECT deployer_address FROM deployers WHERE total_contracts_deployed >= 3"
    ).fetchall()

    profiled = 0
    for row in deployers:
        addr = row["deployer_address"]
        p = extract_profile(conn, addr)
        if p.get("total_contracts", 0) < 3:
            continue

        conn.execute("""
            INSERT OR REPLACE INTO deployer_profiles
            (deployer_address, total_contracts, chain_preference, base_ratio,
             timezone_guess, peak_hour, active_window_start, active_window_pct,
             hour_concentration, gas_avg, gas_min, gas_max, gas_variance_ratio,
             primary_technique, technique_count, technique_ratios,
             bytecode_families, active_days, avg_contracts_per_day,
             max_contracts_per_day, deployment_style, selectors_used,
             contracts_interacted, interaction_revert_rate,
             org_link, funder, hour_distribution, profiled_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            addr, p.get("total_contracts"), p.get("chain_preference"), p.get("base_ratio"),
            p.get("timezone_guess"), p.get("peak_hour"), p.get("active_window_start"),
            p.get("active_window_pct"), p.get("hour_concentration"),
            p.get("gas_avg"), p.get("gas_min"), p.get("gas_max"), p.get("gas_variance_ratio"),
            p.get("primary_technique"), p.get("technique_count"),
            json.dumps(p.get("technique_ratios", {})),
            p.get("bytecode_families"), p.get("active_days"),
            p.get("avg_contracts_per_day"), p.get("max_contracts_per_day"),
            p.get("deployment_style"), p.get("selectors_used"),
            p.get("contracts_interacted"), p.get("interaction_revert_rate"),
            p.get("org_link"), p.get("funder"),
            json.dumps(p.get("hour_distribution", [])),
            _now()
        ))
        profiled += 1

    conn.commit()

    # Summary
    print(f"[deployer_profiler] Profiled {profiled} deployers")
    print()

    # Timezone distribution
    print("Timezone distribution:")
    for r in conn.execute("""
        SELECT timezone_guess, COUNT(*) as c FROM deployer_profiles
        GROUP BY timezone_guess ORDER BY c DESC
    """):
        print(f"  {r['timezone_guess']:25} | {r['c']:>5} deployers")

    # Chain preference
    print()
    print("Chain preference:")
    for r in conn.execute("""
        SELECT chain_preference, COUNT(*) as c FROM deployer_profiles
        GROUP BY chain_preference ORDER BY c DESC
    """):
        print(f"  {r['chain_preference']:15} | {r['c']:>5} deployers")

    # Deployment style
    print()
    print("Deployment style:")
    for r in conn.execute("""
        SELECT deployment_style, COUNT(*) as c FROM deployer_profiles
        GROUP BY deployment_style ORDER BY c DESC
    """):
        print(f"  {r['deployment_style']:15} | {r['c']:>5} deployers")

    # Primary technique
    print()
    print("Primary technique:")
    for r in conn.execute("""
        SELECT primary_technique, COUNT(*) as c FROM deployer_profiles
        GROUP BY primary_technique ORDER BY c DESC
    """):
        print(f"  {r['primary_technique']:15} | {r['c']:>5} deployers")

    conn.close()


def find_similar(target_address: str, top_n: int = 10):
    """Find deployers most similar to a target address."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    # Get target profile
    target = extract_profile(conn, target_address)
    if target.get("total_contracts", 0) < 1:
        print(f"[deployer_profiler] Target {target_address} not found or no contracts")
        conn.close()
        return

    # Compare against all profiled deployers
    candidates = conn.execute(
        "SELECT deployer_address FROM deployer_profiles WHERE deployer_address != ?",
        (target_address,)
    ).fetchall()

    similarities = []
    for row in candidates:
        candidate = extract_profile(conn, row["deployer_address"])
        scores = compute_similarity(target, candidate)
        similarities.append({
            "address": row["deployer_address"],
            "scores": scores,
            "composite": scores["composite"],
            "chain": candidate.get("chain_preference"),
            "contracts": candidate.get("total_contracts"),
            "timezone": candidate.get("timezone_guess"),
            "technique": candidate.get("primary_technique"),
            "org": candidate.get("org_link"),
        })

    similarities.sort(key=lambda x: -x["composite"])

    print(f"[deployer_profiler] Top {top_n} similar to {target_address[:18]}...")
    print(f"  Target: {target.get('total_contracts')} contracts | {target.get('timezone_guess')} | {target.get('chain_preference')} | {target.get('primary_technique')} | org={target.get('org_link')}")
    print()
    print(f"  {'Address':22} | {'Score':>5} | {'TZ':>5} | {'Gas':>4} | {'Tech':>4} | {'Chain':>5} | {'Cad.':>4} | {'Contracts':>9} | {'Timezone':20} | {'Technique':12} | {'Org'}")
    print("  " + "-" * 140)

    for s in similarities[:top_n]:
        sc = s["scores"]
        print(f"  {s['address'][:22]} | {sc['composite']:.3f} | {sc['timezone']:.2f} | {sc['gas']:.2f} | {sc['technique']:.2f} | {sc['chain']:.2f} | {sc['cadence']:.2f} | {s['contracts']:>9} | {s['timezone']:20} | {s['technique']:12} | {s['org']}")

    conn.close()


def cluster_deployers():
    """Find groups of deployers with high mutual similarity."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS deployer_similarity (
            deployer_a TEXT NOT NULL,
            deployer_b TEXT NOT NULL,
            composite_score REAL,
            timezone_score REAL,
            gas_score REAL,
            technique_score REAL,
            chain_score REAL,
            cadence_score REAL,
            computed_at TEXT,
            PRIMARY KEY (deployer_a, deployer_b)
        )
    """)

    # Only compare deployers with 5+ contracts for meaningful profiles
    deployers = conn.execute(
        "SELECT deployer_address FROM deployer_profiles WHERE total_contracts >= 5"
    ).fetchall()

    profiles = {}
    for row in deployers:
        profiles[row["deployer_address"]] = extract_profile(conn, row["deployer_address"])

    addrs = list(profiles.keys())
    high_matches = []
    compared = 0

    for i in range(len(addrs)):
        for j in range(i + 1, len(addrs)):
            # Skip same-org comparisons (we already know those)
            org_i = profiles[addrs[i]].get("org_link", "")
            org_j = profiles[addrs[j]].get("org_link", "")
            if org_i and org_j and org_i == org_j:
                continue

            scores = compute_similarity(profiles[addrs[i]], profiles[addrs[j]])
            compared += 1

            if scores["composite"] >= 0.7:
                high_matches.append({
                    "a": addrs[i], "b": addrs[j],
                    "scores": scores,
                    "org_a": org_i, "org_b": org_j,
                })

                conn.execute("""
                    INSERT OR REPLACE INTO deployer_similarity
                    (deployer_a, deployer_b, composite_score, timezone_score,
                     gas_score, technique_score, chain_score, cadence_score, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    addrs[i], addrs[j], scores["composite"],
                    scores["timezone"], scores["gas"], scores["technique"],
                    scores["chain"], scores["cadence"], _now()
                ))

    conn.commit()

    print(f"[deployer_profiler] Compared {compared} pairs, found {len(high_matches)} high-similarity matches (>=0.7)")
    print()

    high_matches.sort(key=lambda x: -x["scores"]["composite"])
    for m in high_matches[:20]:
        sc = m["scores"]
        org_str = ""
        if m["org_a"]:
            org_str += f" a={m['org_a']}"
        if m["org_b"]:
            org_str += f" b={m['org_b']}"
        print(f"  {m['a'][:16]}... <-> {m['b'][:16]}... | composite={sc['composite']:.3f} | tz={sc['timezone']:.2f} gas={sc['gas']:.2f} tech={sc['technique']:.2f} chain={sc['chain']:.2f}{org_str}")

    conn.close()


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployer behavioral profiling")
    parser.add_argument("--profile-all", action="store_true", help="Profile all deployers with 3+ contracts")
    parser.add_argument("--find-similar", type=str, help="Find deployers similar to ADDRESS")
    parser.add_argument("--cluster", action="store_true", help="Find high-similarity deployer pairs")
    parser.add_argument("--top", type=int, default=10, help="Number of results for find-similar")
    args = parser.parse_args()

    if args.profile_all:
        profile_all()
    elif args.find_similar:
        find_similar(args.find_similar, args.top)
    elif args.cluster:
        cluster_deployers()
    else:
        parser.print_help()
