"""Bytecode Families — clusters contracts by bytecode similarity.

Tier 1: Exact match on first 100 chars of bytecode_pattern_notes.
Tier 2: Combination of has_unusual_fee_structure + has_asymmetric_transfer + has_conditional_revert.

Stores families in bytecode_families and members in bytecode_family_members.
"""

import argparse
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _family_id(prefix: str, key: str) -> str:
    """Generate a deterministic family_id from a prefix and key."""
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"{prefix}-{h}"


def _get_contract_stats(conn: sqlite3.Connection, contract_address: str) -> dict[str, Any]:
    """Get revert rate and unique callers for a contract from transaction_events."""
    row = conn.execute("""
        SELECT COUNT(DISTINCT interacting_address) as callers,
               ROUND(SUM(CASE WHEN is_reverted=1 THEN 1.0 ELSE 0 END)/MAX(COUNT(*),1)*100, 1) as revert_rate
        FROM transaction_events
        WHERE contract_address = ?
    """, (contract_address,)).fetchone()
    if row and row["callers"]:
        return {"callers": row["callers"], "revert_rate": row["revert_rate"] or 0.0}
    return {"callers": 0, "revert_rate": 0.0}


def cluster(conn: sqlite3.Connection) -> None:
    """Run both tiers of clustering and store results."""
    # Clear existing data for re-runs
    conn.execute("DELETE FROM bytecode_family_members")
    conn.execute("DELETE FROM bytecode_families")
    conn.commit()

    families: dict[str, dict] = {}
    members: list[dict] = []

    # ---- Tier 1: Group by first 100 chars of bytecode_pattern_notes ----
    print("[bytecode_families] Tier 1: clustering by bytecode_pattern_notes prefix...")
    tier1_rows = conn.execute("""
        SELECT contract_address, deployer_address, chain, detection_timestamp,
               bytecode_pattern_notes
        FROM contracts
        WHERE bytecode_pattern_notes IS NOT NULL AND bytecode_pattern_notes != ''
    """).fetchall()

    tier1_groups: dict[str, list] = {}
    for r in tier1_rows:
        key = r["bytecode_pattern_notes"][:100]
        tier1_groups.setdefault(key, []).append(r)

    for key, group in tier1_groups.items():
        if len(group) < 2:
            continue
        fid = _family_id("T1", key)
        deployers = set()
        total_victims = 0
        revert_rates = []
        first_seen = None

        for r in group:
            stats = _get_contract_stats(conn, r["contract_address"])
            deployers.add(r["deployer_address"])
            total_victims += stats["callers"]
            revert_rates.append(stats["revert_rate"])
            ts = r["detection_timestamp"]
            if ts and (first_seen is None or ts < first_seen):
                first_seen = ts

            members.append({
                "family_id": fid,
                "contract_address": r["contract_address"],
                "deployer": r["deployer_address"],
                "chain": r["chain"],
                "deployment_timestamp": r["detection_timestamp"],
                "bytecode_length": len(r["bytecode_pattern_notes"] or ""),
                "revert_rate": stats["revert_rate"],
                "unique_callers": stats["callers"],
            })

        avg_revert = round(sum(revert_rates) / len(revert_rates), 1) if revert_rates else 0
        families[fid] = {
            "family_id": fid,
            "family_name": f"Tier1-{key[:40]}",
            "detection_tier": "T1-bytecode_notes",
            "member_count": len(group),
            "unique_deployers": len(deployers),
            "representative_bytecode_prefix": key[:100],
            "selector_fingerprint": None,
            "avg_revert_rate": avg_revert,
            "total_victims": total_victims,
            "first_seen": first_seen,
            "last_updated": first_seen,
            "is_cross_deployer": 1 if len(deployers) > 1 else 0,
            "notes": None,
        }

    print(f"  Tier 1: {len(families)} families from {sum(f['member_count'] for f in families.values())} contracts")

    # ---- Tier 2: Group by flag combination ----
    print("[bytecode_families] Tier 2: clustering by flag combinations...")
    tier2_rows = conn.execute("""
        SELECT contract_address, deployer_address, chain, detection_timestamp,
               COALESCE(has_unusual_fee_structure, 0) as fee,
               COALESCE(has_asymmetric_transfer, 0) as asym,
               COALESCE(has_conditional_revert, 0) as crev
        FROM contracts
        WHERE contract_address NOT IN (
            SELECT contract_address FROM contracts
            WHERE bytecode_pattern_notes IS NOT NULL AND bytecode_pattern_notes != ''
        )
    """).fetchall()

    tier2_groups: dict[str, list] = {}
    for r in tier2_rows:
        key = f"fee={r['fee']}|asym={r['asym']}|crev={r['crev']}"
        tier2_groups.setdefault(key, []).append(r)

    t2_count = 0
    for key, group in tier2_groups.items():
        if len(group) < 2:
            continue
        fid = _family_id("T2", key)
        if fid in families:
            continue
        deployers = set()
        total_victims = 0
        revert_rates = []
        first_seen = None

        for r in group:
            stats = _get_contract_stats(conn, r["contract_address"])
            deployers.add(r["deployer_address"])
            total_victims += stats["callers"]
            revert_rates.append(stats["revert_rate"])
            ts = r["detection_timestamp"]
            if ts and (first_seen is None or ts < first_seen):
                first_seen = ts

            members.append({
                "family_id": fid,
                "contract_address": r["contract_address"],
                "deployer": r["deployer_address"],
                "chain": r["chain"],
                "deployment_timestamp": r["detection_timestamp"],
                "bytecode_length": 0,
                "revert_rate": stats["revert_rate"],
                "unique_callers": stats["callers"],
            })

        avg_revert = round(sum(revert_rates) / len(revert_rates), 1) if revert_rates else 0
        families[fid] = {
            "family_id": fid,
            "family_name": f"Tier2-{key}",
            "detection_tier": "T2-flag_combo",
            "member_count": len(group),
            "unique_deployers": len(deployers),
            "representative_bytecode_prefix": key,
            "selector_fingerprint": None,
            "avg_revert_rate": avg_revert,
            "total_victims": total_victims,
            "first_seen": first_seen,
            "last_updated": first_seen,
            "is_cross_deployer": 1 if len(deployers) > 1 else 0,
            "notes": None,
        }
        t2_count += 1

    print(f"  Tier 2: {t2_count} families")

    # ---- Store results ----
    print(f"[bytecode_families] Inserting {len(families)} families and {len(members)} members...")
    for f in families.values():
        conn.execute("""
            INSERT OR REPLACE INTO bytecode_families
                (family_id, family_name, detection_tier, member_count, unique_deployers,
                 representative_bytecode_prefix, selector_fingerprint, avg_revert_rate,
                 total_victims, first_seen, last_updated, is_cross_deployer, notes)
            VALUES (:family_id, :family_name, :detection_tier, :member_count, :unique_deployers,
                    :representative_bytecode_prefix, :selector_fingerprint, :avg_revert_rate,
                    :total_victims, :first_seen, :last_updated, :is_cross_deployer, :notes)
        """, f)

    for m in members:
        conn.execute("""
            INSERT OR REPLACE INTO bytecode_family_members
                (family_id, contract_address, deployer, chain, deployment_timestamp,
                 bytecode_length, revert_rate, unique_callers)
            VALUES (:family_id, :contract_address, :deployer, :chain, :deployment_timestamp,
                    :bytecode_length, :revert_rate, :unique_callers)
        """, m)

    conn.commit()
    print(f"[bytecode_families] Done. {len(families)} families, {len(members)} members stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bytecode Family Clustering")
    parser.add_argument("--cluster", action="store_true", help="Run clustering")
    args = parser.parse_args()

    if args.cluster:
        conn = get_connection()
        cluster(conn)
        conn.close()
    else:
        parser.print_help()
