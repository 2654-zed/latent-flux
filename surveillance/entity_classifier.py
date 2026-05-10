"""
Layer 3 — Entity Classification System.

Classifies every address in the surveillance database into a structured
taxonomy: category/subtype with confidence levels and org attribution.

Usage:
    python -m surveillance.entity_classifier --all
    python -m surveillance.entity_classifier --import-existing
    python -m surveillance.entity_classifier --classify-unknown
    python -m surveillance.entity_classifier --address 0x...
    python -m surveillance.entity_classifier --report
    python -m surveillance.entity_classifier --stats
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"
CASES_DIR = Path(__file__).resolve().parent / "data" / "cases"

# ──────────────────────────────────────────────────────
# Confidence hierarchy — never overwrite higher with lower
# ──────────────────────────────────────────────────────
CONFIDENCE_RANK = {
    "CONFIRMED": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}

# ──────────────────────────────────────────────────────
# Entity type mapping from deployers.entity_type
# ──────────────────────────────────────────────────────
ENTITY_TYPE_MAP = {
    "treasury": ("CRIMINAL", "org_treasury", None),
    "treasury_branch": ("CRIMINAL", "org_treasury", None),
    "operator": ("CRIMINAL", "org_deployer", None),
    "cashout": ("CRIMINAL", "org_cashout", None),
    "cex_deposit": ("CRIMINAL", "org_exit_ramp", None),
    "laundry": ("CRIMINAL", "org_laundry", None),
    "laundry_candidate": ("CRIMINAL", "org_laundry", None),
    "lp_staging": ("CRIMINAL", "org_laundry", None),
    "lp_companion": ("CRIMINAL", "org_laundry", None),
    "gas_station": ("INFRASTRUCTURE", "gas_station", None),
    "org_002_treasury_senior": ("CRIMINAL", "org_treasury", "org_002"),
    "org_002_treasury_junior": ("CRIMINAL", "org_treasury", "org_002"),
    "org_003_ghost_deployer": ("CRIMINAL", "org_deployer", "org_003"),
    "trap_inventory_operator": ("CRIMINAL", "trap_inventory", None),
    "infrastructure_parasite": ("CRIMINAL", "infrastructure_parasite", None),
    "known_attacker": ("CRIMINAL", "known_attacker", None),
    "ground_truth": ("REFERENCE", "ground_truth", None),
    "mev_bot_factory": ("COMMERCIAL", "mev_factory", None),
}

# ──────────────────────────────────────────────────────
# Known addresses from investigations
# ──────────────────────────────────────────────────────
KNOWN_ADDRESSES = {
    "0xa45b5130f36cdca45667738e2a258ab09f4a5f7f": ("INSTITUTIONAL", "mev_vault", "MEDIUM"),
    "0x9ebed688a923060d3d713cb27d356ced4fa005c5": ("INDIVIDUAL", "bot_operator", "MEDIUM"),
    "0x260790b15bbae85c85987cbb0a533e3c2ce9ca53": ("INDIVIDUAL", "bot_operator", "MEDIUM"),
    "0xd2903bd69321496c9af3f72bd812ab8d3688209f": ("COMMERCIAL", "token_factory", "MEDIUM"),
    "0xd88b345dc5c9888a54c34a186bfafcd016807d07": ("COMMERCIAL", "lp_manager", "MEDIUM"),
    "0x604bb5788bc356f40d7526022dad3a00cb7643bb": ("COMMERCIAL", "lp_manager", "MEDIUM"),
    "0x0860cd84cc10e3263e3393550258b82d28f1d93f": ("COMMERCIAL", "lp_manager", "MEDIUM"),
    "0x84792c2a999c0c12a0a0e00fbadb090cb2f84ad6": ("BOT", "rd_bot", "HIGH"),
    "0xb38e8c17e38363af6ebdcb3dae12e0243582891d": ("INFRASTRUCTURE", "gas_station", "MEDIUM"),
}


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_table(conn):
    """Create entity_classification table if it does not exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entity_classification (
            address             TEXT    NOT NULL PRIMARY KEY,
            category            TEXT    NOT NULL,
            subtype             TEXT    NOT NULL,
            confidence          TEXT    NOT NULL DEFAULT 'LOW'
                                CHECK (confidence IN ('LOW', 'MEDIUM', 'HIGH', 'CONFIRMED')),
            org_id              TEXT,
            source              TEXT    NOT NULL,
            first_classified    TEXT    NOT NULL,
            last_updated        TEXT    NOT NULL,
            notes               TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ec_category ON entity_classification(category);
        CREATE INDEX IF NOT EXISTS idx_ec_subtype ON entity_classification(subtype);
        CREATE INDEX IF NOT EXISTS idx_ec_org ON entity_classification(org_id);
        CREATE INDEX IF NOT EXISTS idx_ec_confidence ON entity_classification(confidence);
    """)
    conn.commit()


# Trap-class subtypes that the OLI guardrail blocks for institutionally-tagged
# addresses. Per Correction #20, these are the typology classifications that
# behavioral/topology signals can spuriously assign to CEX hot wallets, bridge
# solvers, payment processors, and institutional deployers. If an OLI HIGH-severity
# tag is present, the classification is downgraded to a COMMERCIAL/institutional row
# instead of the requested adversarial-class assignment.
_OLI_GUARDED_TRAP_SUBTYPES = frozenset({
    "trap_contract", "trap_deployer", "trap_inventory", "infrastructure_parasite",
    "mev_factory", "bot_operator", "rd_bot", "known_attacker",
    "org_laundry", "org_cashout", "org_001_shadow_cex_exit", "org_001_shadow_lp_staging",
    # PSO + ISO subtypes (typology-promotion paths most affected by Correction #20)
    "pristine_solo_operator", "infrastructure_scale_operator",
    "single_purpose_funder", "drainer_spawn_hub",
})


def classify_address(conn, address, category, subtype, confidence, source,
                     org_id=None, notes=None):
    """
    Insert or update an entity classification.

    CONFIDENCE PROTECTION: never overwrite HIGH/CONFIRMED with MEDIUM/LOW.

    OLI GUARDRAIL (added 2026-05-09 per Correction #20): if `subtype` is in
    `_OLI_GUARDED_TRAP_SUBTYPES` AND the address has a HIGH-severity OLI tag
    (institutional / CEX / bridge / payment processor), the classification is
    redirected to ('COMMERCIAL', 'institutional_oli_tagged', confidence='HIGH')
    with the OLI primary entity preserved in `notes`. This prevents the systematic
    mislabel pattern that Correction #20 documented (e.g., Binance hot wallet
    promoted to "infrastructure_scale_drainer_spawn_hub", Circle deployer promoted
    to "pristine_solo_industrial_operator"). The guardrail short-circuits at the
    classify_address boundary so all callers (import_from_deployers,
    classify_by_deployment_pattern, watchlist promoters, etc.) inherit the check
    without needing per-caller modification.

    Lookup uses `surveillance.oli_enrichment.is_known_legitimate` against the
    `oli_labels` cache. If the cache is empty for this address, the guardrail
    is silent (no fetch is triggered from inside classify_address — calling code
    should run `oli_enrichment.enrich_address(conn, addr)` ahead of time for
    new-promotion paths).
    """
    address = address.lower().strip()
    now = datetime.now(timezone.utc).isoformat()

    # OLI guardrail — redirect adversarial-class classifications when the address
    # has an institutional public tag.
    if subtype in _OLI_GUARDED_TRAP_SUBTYPES:
        try:
            from surveillance.oli_enrichment import is_known_legitimate
            oli_row = is_known_legitimate(conn, address, chain_id=1)
        except Exception:
            oli_row = None
        if oli_row is not None:
            entity = oli_row.get("primary_entity") or oli_row.get("primary_tag_name") or "OLI-tagged"
            redirect_notes = (
                f"[OLI guardrail (Correction #20)] Original classification request: "
                f"category={category} subtype={subtype} source={source} confidence={confidence}. "
                f"Redirected to COMMERCIAL/institutional_oli_tagged because address carries "
                f"OLI HIGH-severity tag '{oli_row.get('primary_tag_name')}' (entity={entity})."
            )
            if notes:
                redirect_notes = redirect_notes + " || Original notes: " + notes
            category = "COMMERCIAL"
            subtype = "institutional_oli_tagged"
            confidence = "HIGH"
            source = source + "+oli_redirect"
            notes = redirect_notes

    existing = conn.execute(
        "SELECT confidence FROM entity_classification WHERE address = ?",
        (address,)
    ).fetchone()

    if existing:
        existing_rank = CONFIDENCE_RANK.get(existing["confidence"], 0)
        new_rank = CONFIDENCE_RANK.get(confidence, 0)
        if new_rank < existing_rank:
            # Do not downgrade confidence
            return False
        conn.execute("""
            UPDATE entity_classification
            SET category = ?, subtype = ?, confidence = ?, org_id = ?,
                source = ?, last_updated = ?, notes = ?
            WHERE address = ?
        """, (category, subtype, confidence, org_id, source, now, notes, address))
    else:
        conn.execute("""
            INSERT INTO entity_classification
                (address, category, subtype, confidence, org_id, source,
                 first_classified, last_updated, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (address, category, subtype, confidence, org_id, source, now, now, notes))

    return True


# ──────────────────────────────────────────────────────
# Import functions
# ──────────────────────────────────────────────────────

def import_from_deployers(conn):
    """Read deployers.entity_type and map to classification taxonomy."""
    print("\n[1] Importing from deployers.entity_type...")
    rows = conn.execute(
        "SELECT deployer_address, entity_type, chain, funding_trail FROM deployers "
        "WHERE entity_type IS NOT NULL AND entity_type != 'unknown' AND entity_type != ''"
    ).fetchall()

    imported = 0
    skipped = 0
    for row in rows:
        entity_type = row["entity_type"]
        address = row["deployer_address"]

        if entity_type in ENTITY_TYPE_MAP:
            category, subtype, org_id_override = ENTITY_TYPE_MAP[entity_type]
        else:
            # Unknown entity_type — skip
            skipped += 1
            continue

        # Try to extract org_id from fund_tracer ORG_WALLETS or funding_trail
        org_id = org_id_override
        if org_id is None:
            # Check if address is in ORG_WALLETS from fund_tracer
            try:
                from surveillance.fund_tracer import ORG_WALLETS
                label = ORG_WALLETS.get(address.lower())
                if label and ":" in label:
                    org_id = label.split(":")[0]
            except ImportError:
                pass

        notes = f"deployers.entity_type={entity_type}"
        if row["funding_trail"]:
            try:
                trail = json.loads(row["funding_trail"])
                if isinstance(trail, dict) and trail.get("funder"):
                    notes += f", funded_by={trail['funder'][:18]}..."
            except (json.JSONDecodeError, TypeError):
                pass

        if classify_address(conn, address, category, subtype, "HIGH",
                            "deployers_import", org_id=org_id, notes=notes):
            imported += 1

    conn.commit()
    print(f"    Imported: {imported}, Skipped (unmapped): {skipped}")
    return imported


def import_bot_candidates(conn):
    """Import all bot_candidates as BOT/unclassified_bot."""
    print("\n[2] Importing bot_candidates...")
    rows = conn.execute(
        "SELECT address, total_revert_count, is_deployer, selector_cluster "
        "FROM bot_candidates"
    ).fetchall()

    imported = 0
    for row in rows:
        notes_parts = [f"reverts={row['total_revert_count']}"]
        if row["is_deployer"]:
            notes_parts.append("is_deployer=true")
        if row["selector_cluster"]:
            notes_parts.append(f"cluster={row['selector_cluster']}")

        if classify_address(conn, row["address"], "BOT", "unclassified_bot",
                            "MEDIUM", "bot_candidates_import",
                            notes=", ".join(notes_parts)):
            imported += 1

    conn.commit()
    print(f"    Imported: {imported} of {len(rows)} bot candidates")
    return imported


def import_fund_tracer_registries(conn):
    """Import known address registries from surveillance.fund_tracer."""
    print("\n[3] Importing fund_tracer registries...")
    try:
        from surveillance.fund_tracer import (
            CEX_HOT_WALLETS, DEX_ROUTERS, DEX_POOLS, BRIDGE_CONTRACTS, MIXERS
        )
    except ImportError:
        print("    ERROR: Could not import from surveillance.fund_tracer")
        return 0

    imported = 0

    for addr, label in CEX_HOT_WALLETS.items():
        if classify_address(conn, addr, "INFRASTRUCTURE", "cex_hot_wallet",
                            "CONFIRMED", "fund_tracer_registry",
                            notes=f"exchange={label}"):
            imported += 1

    for addr, label in DEX_ROUTERS.items():
        if classify_address(conn, addr, "INFRASTRUCTURE", "dex_router",
                            "CONFIRMED", "fund_tracer_registry",
                            notes=f"protocol={label}"):
            imported += 1

    for addr, label in DEX_POOLS.items():
        if classify_address(conn, addr, "INFRASTRUCTURE", "dex_pool",
                            "CONFIRMED", "fund_tracer_registry",
                            notes=f"pool={label}"):
            imported += 1

    for addr, label in BRIDGE_CONTRACTS.items():
        if classify_address(conn, addr, "INFRASTRUCTURE", "bridge",
                            "CONFIRMED", "fund_tracer_registry",
                            notes=f"bridge={label}"):
            imported += 1

    for addr, label in MIXERS.items():
        if classify_address(conn, addr, "INFRASTRUCTURE", "mixer",
                            "CONFIRMED", "fund_tracer_registry",
                            notes=f"mixer={label}"):
            imported += 1

    conn.commit()
    print(f"    Imported: {imported} infrastructure addresses")
    return imported


def import_known_addresses(conn):
    """Import hardcoded addresses from investigations."""
    print("\n[4] Importing known investigation addresses...")
    imported = 0

    for addr, (category, subtype, confidence) in KNOWN_ADDRESSES.items():
        if classify_address(conn, addr, category, subtype, confidence,
                            "investigation_hardcoded",
                            notes="manual investigation finding"):
            imported += 1

    conn.commit()
    print(f"    Imported: {imported} known addresses")
    return imported


# ──────────────────────────────────────────────────────
# Heuristic classifiers
# ──────────────────────────────────────────────────────

def classify_by_deployment_pattern(conn):
    """Classify deployers by contract deployment patterns."""
    print("\n[5] Classifying by deployment pattern...")
    classified = 0

    # Heuristic 1: Deployer with 50+ contracts, all dormant -> COMMERCIAL/token_factory
    rows = conn.execute("""
        SELECT d.deployer_address, d.total_contracts_deployed,
               COUNT(c.contract_address) AS contract_count
        FROM deployers d
        LEFT JOIN contracts c ON c.deployer_address = d.deployer_address
        WHERE d.total_contracts_deployed >= 50
        GROUP BY d.deployer_address
    """).fetchall()

    for row in rows:
        addr = row["deployer_address"]
        # Check if already classified with higher confidence
        existing = conn.execute(
            "SELECT confidence FROM entity_classification WHERE address = ?",
            (addr,)
        ).fetchone()
        if existing and CONFIDENCE_RANK.get(existing["confidence"], 0) >= CONFIDENCE_RANK["LOW"]:
            continue

        # Check if contracts are dormant (no recent transaction_events)
        active = conn.execute("""
            SELECT COUNT(*) as cnt FROM transaction_events te
            JOIN contracts c ON c.contract_address = te.contract_address
            WHERE c.deployer_address = ?
        """, (addr,)).fetchone()

        if active["cnt"] == 0:
            if classify_address(conn, addr, "COMMERCIAL", "token_factory", "LOW",
                                "deployment_pattern_heuristic",
                                notes=f"50+ contracts ({row['total_contracts_deployed']}), all dormant"):
                classified += 1

    # Heuristic 2: Deployer with 10+ contracts, >40% revert, 50+ victims -> CRIMINAL/trap_deployer
    deployers = conn.execute("""
        SELECT d.deployer_address, d.total_contracts_deployed
        FROM deployers d
        WHERE d.total_contracts_deployed >= 10
    """).fetchall()

    for dep in deployers:
        addr = dep["deployer_address"]
        existing = conn.execute(
            "SELECT confidence FROM entity_classification WHERE address = ?",
            (addr,)
        ).fetchone()
        if existing and CONFIDENCE_RANK.get(existing["confidence"], 0) >= CONFIDENCE_RANK["MEDIUM"]:
            continue

        stats = conn.execute("""
            SELECT
                COUNT(*) AS total_events,
                SUM(CASE WHEN is_reverted = 1 THEN 1 ELSE 0 END) AS revert_count,
                COUNT(DISTINCT interacting_address) AS unique_interactors
            FROM transaction_events te
            JOIN contracts c ON c.contract_address = te.contract_address
            WHERE c.deployer_address = ?
        """, (addr,)).fetchone()

        if stats["total_events"] and stats["total_events"] > 0:
            revert_pct = stats["revert_count"] / stats["total_events"]
            if revert_pct > 0.4 and stats["unique_interactors"] >= 50:
                if classify_address(conn, addr, "CRIMINAL", "trap_deployer", "MEDIUM",
                                    "deployment_pattern_heuristic",
                                    notes=f"revert_rate={revert_pct:.1%}, victims={stats['unique_interactors']}"):
                    classified += 1

    conn.commit()
    print(f"    Classified: {classified} by deployment pattern")
    return classified


def classify_confirmed_traps(conn):
    """All contracts with confidence_tier='confirmed' -> CRIMINAL/trap_contract."""
    print("\n[6] Classifying confirmed trap contracts...")
    rows = conn.execute("""
        SELECT contract_address, deployer_address, confidence_reason
        FROM contracts
        WHERE confidence_tier = 'confirmed'
    """).fetchall()

    classified = 0
    for row in rows:
        # Look up deployer's org_id if available
        org_id = None
        dep_class = conn.execute(
            "SELECT org_id FROM entity_classification WHERE address = ?",
            (row["deployer_address"],)
        ).fetchone()
        if dep_class:
            org_id = dep_class["org_id"]

        if classify_address(conn, row["contract_address"], "CRIMINAL", "trap_contract",
                            "CONFIRMED", "confirmed_trap_import", org_id=org_id,
                            notes=f"deployer={row['deployer_address'][:18]}..., "
                                  f"reason={row['confidence_reason'][:60]}"):
            classified += 1

    conn.commit()
    print(f"    Classified: {classified} confirmed trap contracts")
    return classified


def classify_self_operating(conn):
    """Contracts where deployer is >80% of callers -> COMMERCIAL/private_infrastructure."""
    print("\n[7] Classifying self-operating contracts...")
    contracts = conn.execute("""
        SELECT c.contract_address, c.deployer_address
        FROM contracts c
        WHERE c.contract_address NOT IN (
            SELECT address FROM entity_classification
            WHERE confidence IN ('HIGH', 'CONFIRMED')
        )
    """).fetchall()

    classified = 0
    for row in contracts:
        stats = conn.execute("""
            SELECT
                COUNT(*) AS total_calls,
                SUM(CASE WHEN interacting_address = ? THEN 1 ELSE 0 END) AS deployer_calls
            FROM transaction_events
            WHERE contract_address = ?
        """, (row["deployer_address"], row["contract_address"])).fetchone()

        if stats["total_calls"] and stats["total_calls"] >= 5:
            deployer_pct = stats["deployer_calls"] / stats["total_calls"]
            if deployer_pct > 0.8:
                if classify_address(conn, row["contract_address"],
                                    "COMMERCIAL", "private_infrastructure", "LOW",
                                    "self_operating_heuristic",
                                    notes=f"deployer_call_pct={deployer_pct:.1%}, "
                                          f"total_calls={stats['total_calls']}"):
                    classified += 1

    conn.commit()
    print(f"    Classified: {classified} self-operating contracts")
    return classified


# ──────────────────────────────────────────────────────
# Reporting functions
# ──────────────────────────────────────────────────────

def get_stats(conn):
    """Return classification statistics."""
    stats = {}

    total = conn.execute("SELECT COUNT(*) AS cnt FROM entity_classification").fetchone()
    stats["total_classified"] = total["cnt"]

    # By category
    cats = conn.execute(
        "SELECT category, COUNT(*) AS cnt FROM entity_classification GROUP BY category ORDER BY cnt DESC"
    ).fetchall()
    stats["by_category"] = {r["category"]: r["cnt"] for r in cats}

    # By subtype
    subs = conn.execute(
        "SELECT subtype, COUNT(*) AS cnt FROM entity_classification GROUP BY subtype ORDER BY cnt DESC"
    ).fetchall()
    stats["by_subtype"] = {r["subtype"]: r["cnt"] for r in subs}

    # By confidence
    confs = conn.execute(
        "SELECT confidence, COUNT(*) AS cnt FROM entity_classification GROUP BY confidence ORDER BY cnt DESC"
    ).fetchall()
    stats["by_confidence"] = {r["confidence"]: r["cnt"] for r in confs}

    # By source
    sources = conn.execute(
        "SELECT source, COUNT(*) AS cnt FROM entity_classification GROUP BY source ORDER BY cnt DESC"
    ).fetchall()
    stats["by_source"] = {r["source"]: r["cnt"] for r in sources}

    # Org breakdown
    orgs = conn.execute(
        "SELECT org_id, COUNT(*) AS cnt FROM entity_classification "
        "WHERE org_id IS NOT NULL GROUP BY org_id ORDER BY cnt DESC"
    ).fetchall()
    stats["by_org"] = {r["org_id"]: r["cnt"] for r in orgs}

    return stats


def lookup(conn, address):
    """Look up classification for a single address."""
    address = address.lower().strip()
    row = conn.execute(
        "SELECT * FROM entity_classification WHERE address = ?",
        (address,)
    ).fetchone()
    if row:
        return dict(row)
    return None


def generate_report(conn):
    """Generate a full entity classification report as markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stats = get_stats(conn)

    md = f"""# ENTITY CLASSIFICATION REPORT
**Generated:** {now}
**Total classified entities:** {stats['total_classified']}

---

## Category Breakdown

| Category | Count | % |
|---|---|---|
"""
    for cat, cnt in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        pct = round(cnt / stats["total_classified"] * 100, 1) if stats["total_classified"] else 0
        md += f"| {cat} | {cnt} | {pct}% |\n"

    md += "\n---\n\n## Subtype Detail\n\n| Subtype | Count | % |\n|---|---|---|\n"
    for sub, cnt in sorted(stats["by_subtype"].items(), key=lambda x: -x[1]):
        pct = round(cnt / stats["total_classified"] * 100, 1) if stats["total_classified"] else 0
        md += f"| {sub} | {cnt} | {pct}% |\n"

    md += "\n---\n\n## Confidence Distribution\n\n| Confidence | Count | % |\n|---|---|---|\n"
    for conf, cnt in sorted(stats["by_confidence"].items(),
                            key=lambda x: -CONFIDENCE_RANK.get(x[0], 0)):
        pct = round(cnt / stats["total_classified"] * 100, 1) if stats["total_classified"] else 0
        md += f"| {conf} | {cnt} | {pct}% |\n"

    md += "\n---\n\n## Classification Sources\n\n| Source | Count |\n|---|---|\n"
    for src, cnt in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        md += f"| {src} | {cnt} |\n"

    if stats["by_org"]:
        md += "\n---\n\n## Organization Attribution\n\n| Org ID | Entities |\n|---|---|\n"
        for org, cnt in sorted(stats["by_org"].items(), key=lambda x: -x[1]):
            md += f"| {org} | {cnt} |\n"

    # Criminal entities detail
    criminals = conn.execute("""
        SELECT address, subtype, confidence, org_id, notes
        FROM entity_classification
        WHERE category = 'CRIMINAL'
        ORDER BY confidence DESC, subtype
    """).fetchall()

    if criminals:
        md += "\n---\n\n## CRIMINAL Entities Detail\n\n"
        md += "| Address | Subtype | Confidence | Org | Notes |\n"
        md += "|---|---|---|---|---|\n"
        for r in criminals:
            addr_short = r["address"][:18] + "..."
            md += f"| {addr_short} | {r['subtype']} | {r['confidence']} | {r['org_id'] or '-'} | {(r['notes'] or '')[:50]} |\n"

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = CASES_DIR / f"ENTITY_CLASSIFICATION_{ts}.md"
    filepath.write_text(md, encoding="utf-8")
    print(f"\nReport saved: {filepath}")
    return filepath


def print_stats(stats):
    """Pretty-print stats to console."""
    print(f"\n{'='*60}")
    print(f"  ENTITY CLASSIFICATION STATS")
    print(f"{'='*60}")
    print(f"  Total classified: {stats['total_classified']}")

    print(f"\n  By Category:")
    for cat, cnt in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"    {cat:25s} {cnt:>5}")

    print(f"\n  By Subtype:")
    for sub, cnt in sorted(stats["by_subtype"].items(), key=lambda x: -x[1]):
        print(f"    {sub:25s} {cnt:>5}")

    print(f"\n  By Confidence:")
    for conf, cnt in sorted(stats["by_confidence"].items(),
                            key=lambda x: -CONFIDENCE_RANK.get(x[0], 0)):
        print(f"    {conf:25s} {cnt:>5}")

    print(f"\n  By Source:")
    for src, cnt in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        print(f"    {src:30s} {cnt:>5}")

    if stats.get("by_org"):
        print(f"\n  By Organization:")
        for org, cnt in sorted(stats["by_org"].items(), key=lambda x: -x[1]):
            print(f"    {org:25s} {cnt:>5}")

    print(f"{'='*60}")


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def run_imports(conn):
    """Run all import functions."""
    import_from_deployers(conn)
    import_bot_candidates(conn)
    import_fund_tracer_registries(conn)
    import_known_addresses(conn)


def run_classifiers(conn):
    """Run all heuristic classifiers."""
    classify_by_deployment_pattern(conn)
    classify_confirmed_traps(conn)
    classify_self_operating(conn)


def main():
    parser = argparse.ArgumentParser(
        description="Layer 3 Entity Classification System"
    )
    parser.add_argument("--import-existing", action="store_true",
                        help="Import entities from deployers, bot_candidates, fund_tracer")
    parser.add_argument("--classify-unknown", action="store_true",
                        help="Run heuristic classifiers on unclassified entities")
    parser.add_argument("--address", type=str,
                        help="Look up classification for a specific address")
    parser.add_argument("--report", action="store_true",
                        help="Generate full classification report")
    parser.add_argument("--stats", action="store_true",
                        help="Show classification statistics")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline: import + classify + stats + report")

    args = parser.parse_args()

    conn = get_connection()
    ensure_table(conn)

    if args.all:
        print("="*60)
        print("  ENTITY CLASSIFICATION — FULL PIPELINE")
        print("="*60)
        run_imports(conn)
        run_classifiers(conn)
        stats = get_stats(conn)
        print_stats(stats)
        generate_report(conn)
        conn.close()
        return

    if args.import_existing:
        run_imports(conn)

    if args.classify_unknown:
        run_classifiers(conn)

    if args.address:
        result = lookup(conn, args.address)
        if result:
            print(f"\nEntity Classification for {args.address}:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"\nNo classification found for {args.address}")

    if args.stats:
        stats = get_stats(conn)
        print_stats(stats)

    if args.report:
        generate_report(conn)

    conn.close()


if __name__ == "__main__":
    main()
