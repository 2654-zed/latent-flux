"""
Layer 3 — Stored Potential Risk Scoring Model

Computes a composite risk score for flagged contracts based on:
  stored_potential = approval_scope + capability + deployer_risk + org_context
                   + observation_capability
  risk_score = (stored_potential * volatility) / max(realized_value, 1)

observation_capability was added 2026-04-20 (Correction #12). Prior to that,
the framework named five primitives but the scoring code computed four;
observer-class contracts (oracles, routers with broad visibility) scored
MINIMAL despite being at maximum stored potential per the framework.

High stored_potential + low realized_value + high volatility = HIGHEST RISK.
Contracts that have accumulated permissions/capabilities but haven't extracted
value yet are the most dangerous.

Read-only analytics on existing surveillance data. Creates no new tables.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

# Permit2 constants
MAX_UINT160 = (1 << 160) - 1
MAX_UINT48 = (1 << 48) - 1

# Risk tier thresholds
TIER_CRITICAL = 50
TIER_HIGH = 20
TIER_MEDIUM = 8
TIER_LOW = 3


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _risk_tier(score: float) -> str:
    if score >= TIER_CRITICAL:
        return "CRITICAL"
    if score >= TIER_HIGH:
        return "HIGH"
    if score >= TIER_MEDIUM:
        return "MEDIUM"
    if score >= TIER_LOW:
        return "LOW"
    return "MINIMAL"


# ------------------------------------------------------------------
# Component 1: Approval Scope Score (0-25)
# ------------------------------------------------------------------

def _compute_approval_scope(conn: sqlite3.Connection, address: str) -> tuple[int, dict]:
    """Score based on Permit2 / ERC-20 approvals targeting this contract."""
    components: dict[str, Any] = {"approvals_found": 0, "max_tier": "none"}
    addr = address.lower()
    best_score = 0

    # Check approval_watchlist: contract is the spender
    if _table_exists(conn, "approval_watchlist"):
        rows = conn.execute(
            "SELECT contract_tier, drain_detected FROM approval_watchlist "
            "WHERE contract_address = ?",
            (addr,),
        ).fetchall()
        components["approval_watchlist_rows"] = len(rows)
        for r in rows:
            tier = (r["contract_tier"] or "").lower()
            if tier == "confirmed":
                best_score = max(best_score, 25)
                components["max_tier"] = "unlimited_no_expiry"
            elif tier == "suspected":
                best_score = max(best_score, 20)
                components["max_tier"] = components.get("max_tier") or "unlimited_with_expiry"

    # Check approval_events: contract is the spender
    if _table_exists(conn, "approval_events"):
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM approval_events WHERE spender = ?",
            (addr,),
        ).fetchone()
        count = rows["cnt"] if rows else 0
        components["approval_events_count"] = count
        if count > 0 and best_score < 15:
            # Without on-chain value data, use count as a heuristic
            if count >= 10:
                best_score = max(best_score, 15)
            elif count >= 5:
                best_score = max(best_score, 10)
            else:
                best_score = max(best_score, 5)

    # Check live_exposures: contract is the approved_contract
    if _table_exists(conn, "live_exposures"):
        rows = conn.execute(
            "SELECT approval_amount, status FROM live_exposures "
            "WHERE approved_contract = ?",
            (addr,),
        ).fetchall()
        components["live_exposures_count"] = len(rows)
        for r in rows:
            try:
                amount = int(r["approval_amount"])
            except (ValueError, TypeError):
                amount = 0
            if amount >= MAX_UINT160:
                best_score = max(best_score, 25)
                components["max_tier"] = "unlimited_no_expiry"
            elif amount > 10_000 * 10**18:
                best_score = max(best_score, 15)
            elif amount > 1_000 * 10**18:
                best_score = max(best_score, 10)
            elif amount > 0:
                best_score = max(best_score, 5)

    components["approvals_found"] = max(
        components.get("approval_watchlist_rows", 0),
        components.get("approval_events_count", 0),
        components.get("live_exposures_count", 0),
    )
    return min(best_score, 25), components


# ------------------------------------------------------------------
# Component 2: Capability Score (0-25)
# ------------------------------------------------------------------

def _compute_capability(conn: sqlite3.Connection, address: str) -> tuple[int, dict]:
    """Score based on bytecode signals from contracts and bytecode_cache."""
    components: dict[str, Any] = {}
    addr = address.lower()
    score = 0

    # Check if deployed_code_hash column exists
    try:
        conn.execute("SELECT deployed_code_hash FROM contracts LIMIT 0")
        has_code_hash_col = True
    except sqlite3.OperationalError:
        has_code_hash_col = False

    if has_code_hash_col:
        row = conn.execute(
            "SELECT has_asymmetric_transfer, has_conditional_revert, "
            "has_unusual_fee_structure, bytecode_pattern_notes, "
            "confidence_tier, deployed_code_hash "
            "FROM contracts WHERE contract_address = ?",
            (addr,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT has_asymmetric_transfer, has_conditional_revert, "
            "has_unusual_fee_structure, bytecode_pattern_notes, "
            "confidence_tier "
            "FROM contracts WHERE contract_address = ?",
            (addr,),
        ).fetchone()

    if row is None:
        return 0, {"reason": "contract not found"}

    if row["has_asymmetric_transfer"]:
        score += 8
        components["has_asymmetric_transfer"] = True
    if row["has_conditional_revert"]:
        score += 7
        components["has_conditional_revert"] = True
    if row["has_unusual_fee_structure"]:
        score += 5
        components["has_unusual_fee_structure"] = True

    # Check for DELEGATECALL in bytecode_pattern_notes
    notes = row["bytecode_pattern_notes"] or ""
    notes_upper = notes.upper()
    if "DELEGATECALL" in notes_upper:
        score += 10
        components["delegatecall_in_token"] = True

    # Also check bytecode_cache for additional signals
    code_hash = row["deployed_code_hash"] if has_code_hash_col else None
    if code_hash and _table_exists(conn, "bytecode_cache"):
        cache_row = conn.execute(
            "SELECT bytecode_signals, confidence_tier FROM bytecode_cache "
            "WHERE code_hash = ?",
            (code_hash,),
        ).fetchone()
        if cache_row:
            try:
                signals = json.loads(cache_row["bytecode_signals"] or "{}")
            except (json.JSONDecodeError, TypeError):
                signals = {}
            if signals.get("delegatecall_in_token") and "delegatecall_in_token" not in components:
                score += 10
                components["delegatecall_in_token"] = True

    # Admin-freeze capability (blacklist-like stored potential)
    #
    # A contract with conditional_revert (address-keyed SLOAD gating
    # transfer) AND non-renounced ownership has the admin-freeze
    # primitive: the admin can freeze any holder's tokens by writing
    # to the blacklist mapping. This is the same topological primitive
    # as USDT's blacklist function — a capability held by the admin
    # against ALL holders.
    #
    # The classifier detects the CHECK side (SLOAD -> JUMPI -> REVERT).
    # The WRITE side (admin populates the mapping) is inferred from
    # non-renounced ownership — if the admin key is live, the write
    # capability exists.
    #
    # This doesn't score HIGHER than other capability signals — it
    # scores ALONGSIDE them. A contract with both asymmetric_transfer
    # (trap mechanism) and admin_freeze (admin control) has two
    # independent risk surfaces.
    has_admin_freeze = False
    if row["has_conditional_revert"]:
        # Check if ownership is NOT burned (admin key is live)
        ownership_live = True
        if "RENOUNCEOWNERSHIP" in notes_upper.replace(" ", ""):
            ownership_live = False
        # Check for blacklist-specific pattern in notes
        is_blacklist_pattern = "blacklist" in notes.lower() or "owner_check" in notes.lower()

        if ownership_live and is_blacklist_pattern:
            has_admin_freeze = True
            components["has_admin_freeze"] = True

    # Confidence tier bonus
    tier = (row["confidence_tier"] or "").lower()
    if tier == "confirmed":
        score += 5
        components["confidence_bonus"] = 5
    elif tier == "suspected":
        score += 3
        components["confidence_bonus"] = 3

    return min(score, 25), components


# ------------------------------------------------------------------
# Component 3: Deployer Risk Score (0-25)
# ------------------------------------------------------------------

def _compute_deployer_risk(conn: sqlite3.Connection, address: str) -> tuple[int, dict]:
    """Score based on deployer history and behavior."""
    components: dict[str, Any] = {}
    addr = address.lower()
    score = 0

    # Get deployer for this contract
    row = conn.execute(
        "SELECT deployer_address FROM contracts WHERE contract_address = ?",
        (addr,),
    ).fetchone()
    if row is None:
        return 0, {"reason": "contract not found"}

    deployer = row["deployer_address"]
    components["deployer_address"] = deployer

    # Deployer record
    dep = conn.execute(
        "SELECT total_contracts_deployed, entity_type FROM deployers "
        "WHERE deployer_address = ?",
        (deployer,),
    ).fetchone()

    if dep is None:
        return 0, {"reason": "deployer not found"}

    # Full deployer record for age/funding checks
    dep_full = conn.execute(
        "SELECT first_seen, funding_trail FROM deployers "
        "WHERE deployer_address = ?",
        (deployer,),
    ).fetchone()

    # Confirmed trap count (derived -- schema note says these are computed at read time)
    confirmed = conn.execute(
        "SELECT COUNT(*) as cnt FROM contracts "
        "WHERE deployer_address = ? AND confidence_tier = 'confirmed'",
        (deployer,),
    ).fetchone()
    confirmed_count = confirmed["cnt"] if confirmed else 0
    components["confirmed_trap_count"] = confirmed_count
    if confirmed_count > 0:
        score += 10

    # Suspected trap count
    suspected = conn.execute(
        "SELECT COUNT(*) as cnt FROM contracts "
        "WHERE deployer_address = ? AND confidence_tier = 'suspected'",
        (deployer,),
    ).fetchone()
    suspected_count = suspected["cnt"] if suspected else 0
    components["suspected_trap_count"] = suspected_count
    if suspected_count > 5:
        score += 5

    # Total contracts deployed
    total = dep["total_contracts_deployed"] or 0
    components["total_contracts_deployed"] = total
    if total > 20:
        score += 5

    # Entity type check (org membership)
    entity_type = (dep["entity_type"] or "").lower()
    components["entity_type"] = entity_type
    if "org_" in entity_type:
        score += 5

    # Deployment velocity flag: check deployer_profiles for burst style
    if _table_exists(conn, "deployer_profiles"):
        profile = conn.execute(
            "SELECT deployment_style, max_contracts_per_day FROM deployer_profiles "
            "WHERE deployer_address = ?",
            (deployer,),
        ).fetchone()
        if profile:
            style = (profile["deployment_style"] or "").lower()
            max_per_day = profile["max_contracts_per_day"] or 0
            if style == "burst" or max_per_day >= 10:
                score += 3
                components["velocity_flag"] = True

    # Deployer age TTL boost — new deployers funded by CEX hot wallets
    # are high risk even with zero history. This addresses the 32% blind
    # spot where confirmed traps from first-time deployers score LOW.
    first_seen = dep_full["first_seen"] if dep_full else None
    if first_seen:
        try:
            from datetime import datetime, timezone, timedelta
            seen_dt = datetime.fromisoformat(first_seen.replace("+00:00", "+00:00"))
            if seen_dt.tzinfo is None:
                seen_dt = seen_dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - seen_dt).total_seconds() / 3600
            components["deployer_age_hours"] = round(age_hours, 1)

            if age_hours < 24:
                components["deployer_is_new"] = True
                # Check if funded by a high-nonce address (CEX hot wallet pattern)
                funding_trail = dep_full["funding_trail"] if dep_full else None
                funder_nonce = None
                if funding_trail:
                    try:
                        trail = json.loads(funding_trail)
                        funder_addr = trail.get("funder", "")
                        if funder_addr:
                            # Check funder's deployer record for nonce hints,
                            # or check if funder is a known gas_station
                            funder_dep = conn.execute(
                                "SELECT total_contracts_deployed, entity_type "
                                "FROM deployers WHERE deployer_address = ?",
                                (funder_addr.lower(),),
                            ).fetchone()
                            if funder_dep:
                                funder_total = funder_dep["total_contracts_deployed"] or 0
                                funder_type = (funder_dep["entity_type"] or "").lower()
                                # High-volume funder (>100 contracts funded = likely gas station/CEX)
                                if funder_total > 100 or "gas_station" in funder_type:
                                    score += 10
                                    components["new_deployer_cex_funded"] = True
                                    components["funder_address"] = funder_addr
                                    components["funder_total_contracts"] = funder_total
                            # Also check entity_classification
                            if not components.get("new_deployer_cex_funded") and _table_exists(conn, "entity_classification"):
                                ec = conn.execute(
                                    "SELECT subtype FROM entity_classification WHERE address = ?",
                                    (funder_addr.lower(),),
                                ).fetchone()
                                if ec and "gas_station" in (ec["subtype"] or "").lower():
                                    score += 10
                                    components["new_deployer_cex_funded"] = True
                                    components["funder_address"] = funder_addr
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass
        except (ValueError, TypeError):
            pass

    return min(score, 25), components


# ------------------------------------------------------------------
# Component 4: Org Context Score (0-25)
# ------------------------------------------------------------------

def _compute_org_context(conn: sqlite3.Connection, address: str) -> tuple[int, dict]:
    """Score based on organizational associations."""
    components: dict[str, Any] = {}
    addr = address.lower()
    score = 0

    # Get deployer
    row = conn.execute(
        "SELECT deployer_address, deployer_funding_source FROM contracts "
        "WHERE contract_address = ?",
        (addr,),
    ).fetchone()
    if row is None:
        return 0, {"reason": "contract not found"}

    deployer = row["deployer_address"]
    funder = row["deployer_funding_source"]

    # Check entity_classification for org links on both contract and deployer
    org_id = None
    if _table_exists(conn, "entity_classification"):
        for check_addr in [addr, deployer]:
            ec = conn.execute(
                "SELECT org_id, category, subtype FROM entity_classification "
                "WHERE address = ?",
                (check_addr,),
            ).fetchone()
            if ec and ec["org_id"]:
                org_id = ec["org_id"]
                components["org_id"] = org_id
                components["classified_address"] = check_addr
                break

    # Check deployer entity_type for org link
    if not org_id:
        dep = conn.execute(
            "SELECT entity_type FROM deployers WHERE deployer_address = ?",
            (deployer,),
        ).fetchone()
        if dep:
            entity_type = (dep["entity_type"] or "").lower()
            if "org_" in entity_type:
                org_id = entity_type
                components["org_id_from_deployer"] = entity_type

    # Check deployer_profiles for org_link
    if not org_id and _table_exists(conn, "deployer_profiles"):
        profile = conn.execute(
            "SELECT org_link FROM deployer_profiles WHERE deployer_address = ?",
            (deployer,),
        ).fetchone()
        if profile and profile["org_link"]:
            org_id = profile["org_link"]
            components["org_id_from_profile"] = org_id

    # Score based on org
    if org_id:
        if "org_001" in str(org_id).lower():
            score += 15
            components["org_001_link"] = True
        else:
            score += 10
            components["other_org_link"] = True

    # Check if deployer funded by gas_station entity
    gas_station_funded = False
    if funder:
        if _table_exists(conn, "entity_classification"):
            ec = conn.execute(
                "SELECT subtype FROM entity_classification WHERE address = ?",
                (funder.lower(),),
            ).fetchone()
            if ec and "gas_station" in (ec["subtype"] or "").lower():
                gas_station_funded = True

    # Also check deployer's entity_type in deployers table for gas_station funding
    if not gas_station_funded:
        dep_row = conn.execute(
            "SELECT funding_sources, entity_type FROM deployers "
            "WHERE deployer_address = ?",
            (deployer,),
        ).fetchone()
        if dep_row:
            # Check if any funding source is a known gas_station
            try:
                sources = json.loads(dep_row["funding_sources"] or "[]")
            except (json.JSONDecodeError, TypeError):
                sources = []
            if _table_exists(conn, "entity_classification"):
                for src in sources:
                    ec = conn.execute(
                        "SELECT subtype FROM entity_classification WHERE address = ?",
                        (src.lower(),),
                    ).fetchone()
                    if ec and "gas_station" in (ec["subtype"] or "").lower():
                        gas_station_funded = True
                        break

    if gas_station_funded:
        score += 5
        components["gas_station_funded"] = True

    # Check if part of a cross-deployer bytecode family
    if _table_exists(conn, "bytecode_family_members"):
        family_row = conn.execute(
            "SELECT family_id FROM bytecode_family_members "
            "WHERE contract_address = ?",
            (addr,),
        ).fetchone()
        if family_row:
            # Check if family spans multiple deployers
            family_id = family_row["family_id"]
            deployer_count = conn.execute(
                "SELECT COUNT(DISTINCT deployer) as cnt FROM bytecode_family_members "
                "WHERE family_id = ?",
                (family_id,),
            ).fetchone()
            if deployer_count and deployer_count["cnt"] > 1:
                score += 3
                components["cross_deployer_family"] = family_id

    # Also check alerts table for org associations
    if _table_exists(conn, "alerts"):
        try:
            alert_rows = conn.execute(
                "SELECT alert_type, payload FROM alerts "
                "WHERE address = ? AND false_positive = 0 "
                "ORDER BY timestamp DESC LIMIT 10",
                (addr,),
            ).fetchall()
        except sqlite3.OperationalError:
            # false_positive column may not exist yet
            alert_rows = conn.execute(
                "SELECT alert_type, payload FROM alerts "
                "WHERE address = ? "
                "ORDER BY timestamp DESC LIMIT 10",
                (addr,),
            ).fetchall()
        for ar in alert_rows:
            payload = ar["payload"] or ""
            if "org_001" in payload.lower() and not components.get("org_001_link"):
                score += 15
                components["org_001_from_alert"] = True
                break
            elif "org_" in payload.lower() and not components.get("other_org_link"):
                score += 10
                components["other_org_from_alert"] = True
                break

    return min(score, 25), components


# ------------------------------------------------------------------
# Component 5: Realized Value (0-25)
# ------------------------------------------------------------------

def _compute_realized_value(conn: sqlite3.Connection, address: str) -> tuple[int, dict]:
    """Score based on value already extracted via on-chain transfers.

    Gaming resistance: excludes value sent by deployer-adjacent wallets
    (deployer itself, funder, same-org wallets). Only counts value from
    external callers — the actual victim/user traffic that represents
    real extraction.
    """
    components: dict[str, Any] = {}
    addr = address.lower()

    # Build exclusion set: deployer + funder + org wallets
    # These are "self-deposits" that shouldn't count as realized value
    exclude_addrs = set()
    contract_row = conn.execute(
        "SELECT deployer_address, deployer_funding_source FROM contracts "
        "WHERE contract_address = ?",
        (addr,),
    ).fetchone()
    if contract_row:
        deployer = contract_row["deployer_address"]
        if deployer:
            exclude_addrs.add(deployer.lower())
        funder = contract_row["deployer_funding_source"]
        if funder:
            exclude_addrs.add(funder.lower())
        # Also exclude other contracts from the same deployer
        siblings = conn.execute(
            "SELECT contract_address FROM contracts WHERE deployer_address = ?",
            (deployer,),
        ).fetchall()
        for s in siblings:
            exclude_addrs.add(s["contract_address"].lower())
        # Check funding_trail for additional funders
        dep_row = conn.execute(
            "SELECT funding_trail FROM deployers WHERE deployer_address = ?",
            (deployer,),
        ).fetchone()
        if dep_row and dep_row["funding_trail"]:
            try:
                trail = json.loads(dep_row["funding_trail"])
                trail_funder = trail.get("funder", "")
                if trail_funder:
                    exclude_addrs.add(trail_funder.lower())
            except (json.JSONDecodeError, TypeError):
                pass

    components["excluded_addresses"] = len(exclude_addrs)

    if not _table_exists(conn, "transaction_events"):
        return 0, {"reason": "transaction_events table not found"}

    # Check if value_wei column exists
    has_value = True
    try:
        conn.execute("SELECT value_wei FROM transaction_events LIMIT 0")
    except sqlite3.OperationalError:
        has_value = False

    if not has_value:
        # Fall back to trap_events loss estimates (these are already external victims)
        row = conn.execute(
            "SELECT SUM(loss_estimate_usd) as total_usd, COUNT(*) as cnt "
            "FROM trap_events WHERE trap_contract_address = ?",
            (addr,),
        ).fetchone()
        total_usd = row["total_usd"] if row and row["total_usd"] else 0
        components["source"] = "trap_events"
        components["total_usd"] = total_usd
        components["event_count"] = row["cnt"] if row else 0
    else:
        # Sum value_wei from EXTERNAL callers only (exclude deployer-adjacent)
        if exclude_addrs:
            placeholders = ",".join("?" * len(exclude_addrs))
            exclude_clause = f"AND interacting_address NOT IN ({placeholders})"
            params = [addr] + list(exclude_addrs)
        else:
            exclude_clause = ""
            params = [addr]

        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM transaction_events "
            f"WHERE contract_address = ? AND value_wei IS NOT NULL "
            f"AND value_wei != '' AND value_wei != '0' {exclude_clause}",
            params,
        ).fetchone()
        tx_count = row["cnt"] if row else 0
        components["source"] = "transaction_events_external_only"
        components["value_tx_count"] = tx_count

        rows = conn.execute(
            f"SELECT value_wei FROM transaction_events "
            f"WHERE contract_address = ? AND value_wei IS NOT NULL "
            f"AND value_wei != '' AND value_wei != '0' {exclude_clause}",
            params,
        ).fetchall()
        total_wei = 0
        for r in rows:
            try:
                total_wei += int(r["value_wei"])
            except (ValueError, TypeError):
                pass

        # Convert wei to approximate USD (use 1 ETH ~ $2000 as rough estimate)
        total_eth = total_wei / 10**18
        total_usd = total_eth * 2000
        components["total_wei"] = str(total_wei)
        components["total_eth"] = round(total_eth, 4)
        components["total_usd_approx"] = round(total_usd, 2)

    # Also factor in drain_value_scanner results
    if _table_exists(conn, "drain_values"):
        drain = conn.execute(
            "SELECT SUM(amount_human) as total FROM drain_values "
            "WHERE contract_address = ?",
            (addr,),
        ).fetchone()
        if drain and drain["total"]:
            # amount_human is token amounts, not USD; use as rough proxy
            drain_val = drain["total"]
            total_usd = max(total_usd, drain_val)
            components["drain_value_amount"] = drain_val

    # Scale to 0-25
    if total_usd >= 100_000:
        score = 25
    elif total_usd >= 10_000:
        score = 20
    elif total_usd >= 1_000:
        score = 15
    elif total_usd >= 100:
        score = 10
    elif total_usd > 0:
        score = 5
    else:
        score = 0

    components["score_usd_basis"] = round(total_usd, 2)
    return score, components


# ------------------------------------------------------------------
# Component 5: Observation Capability Score (0-25)
# ------------------------------------------------------------------

_OBS_BYTECODE_MARKERS = ("CALLER", "TIMESTAMP", "TXORIGIN", "TX.ORIGIN", "ORIGIN")
_OBS_INFRA_ROLES = ("router", "aggregator", "bundler", "oracle", "endpoint",
                    "messagetransmitter", "transmitter", "relay", "relayer")


def _compute_observation_capability(conn: sqlite3.Connection, address: str) -> tuple[int, dict]:
    """Score based on what user-activity context this contract can observe.

    Exception-as-rule audit P2 finding: the framework names five primitives
    (position, permissions, trust bindings, mutability, observation) but the
    scoring model only computes four. A read-only oracle with broad visibility
    into user intent should be at maximum stored potential per the framework's
    definition; without this component it scored MINIMAL. See Correction #12.

    Signals (signals D, E deferred from the original spec):
      A — bytecode pattern notes contain CALLER/TIMESTAMP/TXORIGIN markers
      B — log-scaled distinct EOA count (breadth of observation)
      C — present in infrastructure_registry as router/aggregator/bundler/oracle
      edge — +1 if the contract interacts with Permit2 (sees signed-intent flow)
    """
    components: dict[str, Any] = {}
    addr = address.lower()
    score = 0

    row = conn.execute(
        "SELECT bytecode_pattern_notes FROM contracts WHERE contract_address = ?",
        (addr,),
    ).fetchone()
    if row is None:
        return 0, {"reason": "contract not found"}

    # Signal A — bytecode observation markers (0-8)
    notes_upper = (row["bytecode_pattern_notes"] or "").upper()
    matched = [m for m in _OBS_BYTECODE_MARKERS if m in notes_upper]
    if matched:
        score += 8
        components["bytecode_observation_markers"] = matched

    # Signal B — distinct EOA count, log-scaled (0-8)
    eoa_row = conn.execute(
        "SELECT COUNT(DISTINCT interacting_address) AS n "
        "FROM transaction_events WHERE contract_address = ?",
        (addr,),
    ).fetchone()
    distinct_eoas = int(eoa_row["n"] or 0) if eoa_row else 0
    components["distinct_eoas"] = distinct_eoas
    if distinct_eoas >= 10000:
        score += 8
        components["eoa_tier"] = "10k+"
    elif distinct_eoas >= 1000:
        score += 6
        components["eoa_tier"] = "1k-10k"
    elif distinct_eoas >= 100:
        score += 4
        components["eoa_tier"] = "100-1k"
    elif distinct_eoas >= 10:
        score += 2
        components["eoa_tier"] = "10-100"
    else:
        components["eoa_tier"] = "<10"

    # Signal C — infrastructure registry as observation-capable role (0-8)
    if _table_exists(conn, "infrastructure_registry"):
        infra = conn.execute(
            "SELECT classification, notes FROM infrastructure_registry "
            "WHERE address = ?",
            (addr,),
        ).fetchone()
        if infra:
            text = " ".join(
                str(v or "").lower() for v in (infra["classification"], infra["notes"])
            )
            matched_role = [r for r in _OBS_INFRA_ROLES if r in text]
            if matched_role:
                score += 8
                components["infrastructure_role_match"] = matched_role

    # Edge — Permit2 interaction bump (+1)
    if _table_exists(conn, "approval_watchlist"):
        permit2 = conn.execute(
            "SELECT 1 FROM approval_watchlist "
            "WHERE contract_address = ? LIMIT 1",
            (addr,),
        ).fetchone()
        if permit2:
            score += 1
            components["permit2_seen"] = True

    return min(score, 25), components


# ------------------------------------------------------------------
# Component 6: Volatility Multiplier
# ------------------------------------------------------------------

def _compute_volatility(conn: sqlite3.Connection, address: str) -> tuple[float, dict]:
    """Compute volatility multiplier based on delegatecall and timestamp gates."""
    components: dict[str, Any] = {}
    addr = address.lower()
    volatility = 1.0

    row = conn.execute(
        "SELECT bytecode_pattern_notes, deployer_address FROM contracts "
        "WHERE contract_address = ?",
        (addr,),
    ).fetchone()
    if row is None:
        return 1.0, {"reason": "contract not found"}

    notes = (row["bytecode_pattern_notes"] or "").upper()

    # Check for DELEGATECALL
    has_delegatecall = "DELEGATECALL" in notes
    components["has_delegatecall"] = has_delegatecall

    # Check if ownership is renounced
    ownership_renounced = False
    if "RENOUNCEOWNERSHIP" in notes.replace(" ", "").upper():
        ownership_renounced = True
    # Also check upgrade_events for ownership status
    if _table_exists(conn, "upgrade_events"):
        upgrade = conn.execute(
            "SELECT details FROM upgrade_events "
            "WHERE contract_address = ? ORDER BY detected_at DESC LIMIT 1",
            (addr,),
        ).fetchone()
        if upgrade and upgrade["details"]:
            details = (upgrade["details"] or "").lower()
            if "renounce" in details or "owner = 0x000" in details:
                ownership_renounced = True
    components["ownership_renounced"] = ownership_renounced

    # Check for timestamp-gated logic
    has_timestamp_gate = False
    # TIMESTAMP opcode (0x42) patterns in bytecode notes
    if "TIMESTAMP" in notes and ("GATE" in notes or "BLOCK" in notes or "CONDITION" in notes):
        has_timestamp_gate = True
    # Check bytecode_cache signals
    try:
        code_hash_row = conn.execute(
            "SELECT deployed_code_hash FROM contracts WHERE contract_address = ?",
            (addr,),
        ).fetchone()
    except sqlite3.OperationalError:
        code_hash_row = None
    if code_hash_row and code_hash_row["deployed_code_hash"] and _table_exists(conn, "bytecode_cache"):
        cache = conn.execute(
            "SELECT bytecode_signals FROM bytecode_cache WHERE code_hash = ?",
            (code_hash_row["deployed_code_hash"],),
        ).fetchone()
        if cache:
            try:
                signals = json.loads(cache["bytecode_signals"] or "{}")
            except (json.JSONDecodeError, TypeError):
                signals = {}
            if signals.get("timestamp_dependence") or signals.get("time_gate"):
                has_timestamp_gate = True

    # Check timelock_countdowns for approaching gates
    if _table_exists(conn, "timelock_countdowns"):
        tl = conn.execute(
            "SELECT COUNT(*) as cnt FROM timelock_countdowns "
            "WHERE contract_address = ?",
            (addr,),
        ).fetchone()
        if tl and tl["cnt"] > 0:
            has_timestamp_gate = True
            components["timelock_countdown"] = True

    components["has_timestamp_gate"] = has_timestamp_gate

    # Check for SELFDESTRUCT (opcode 0xff) — contract can erase evidence
    has_selfdestruct = "SELFDESTRUCT" in notes or "SUICIDE" in notes
    components["has_selfdestruct"] = has_selfdestruct

    # Check for CREATE2 (opcode 0xf5) — contract can redeploy at same address
    has_create2 = "CREATE2" in notes
    components["has_create2"] = has_create2

    # Apply multipliers -- use the highest if multiple apply
    delegatecall_vol = 1.0
    timestamp_vol = 1.0
    selfdestruct_vol = 1.0

    if has_delegatecall and not ownership_renounced:
        delegatecall_vol = 2.5
    if has_timestamp_gate:
        timestamp_vol = 2.0
    if has_selfdestruct:
        selfdestruct_vol = 3.0
        # SELFDESTRUCT + CREATE2 = metamorphic contract (destroy + redeploy)
        if has_create2:
            selfdestruct_vol = 3.5
            components["metamorphic_pattern"] = True

    volatility = max(delegatecall_vol, timestamp_vol, selfdestruct_vol)
    components["delegatecall_volatility"] = delegatecall_vol
    components["timestamp_volatility"] = timestamp_vol
    components["selfdestruct_volatility"] = selfdestruct_vol

    return volatility, components


# ------------------------------------------------------------------
# Main scoring interface
# ------------------------------------------------------------------

class RiskScorer:
    """Computes the stored_potential risk score for a contract address."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def score(self, contract_address: str) -> dict[str, Any]:
        """Compute the full risk breakdown for a single contract."""
        return score_contract(self.conn, contract_address)

    def batch(self, limit: int = 100) -> list[dict[str, Any]]:
        """Score the top N contracts by stored_potential."""
        return batch_score(self.conn, limit=limit)

    def score_family(self, family_id: str) -> list[dict[str, Any]]:
        """Score all members of a bytecode family."""
        return score_family(self.conn, family_id)


def score_contract(conn: sqlite3.Connection, contract_address: str) -> dict[str, Any]:
    """
    Compute the stored_potential risk score for a single contract.

    Returns a dict with the full breakdown.
    """
    addr = contract_address.lower()

    # Verify contract exists
    exists = conn.execute(
        "SELECT 1 FROM contracts WHERE contract_address = ?", (addr,)
    ).fetchone()
    if exists is None:
        return {
            "contract_address": addr,
            "error": "contract not found in database",
            "risk_score": 0.0,
            "risk_tier": "MINIMAL",
        }

    # Compute all components
    approval_score, approval_comp = _compute_approval_scope(conn, addr)
    capability_score, capability_comp = _compute_capability(conn, addr)
    deployer_score, deployer_comp = _compute_deployer_risk(conn, addr)
    org_score, org_comp = _compute_org_context(conn, addr)
    observation_score, observation_comp = _compute_observation_capability(conn, addr)
    realized, realized_comp = _compute_realized_value(conn, addr)
    volatility, volatility_comp = _compute_volatility(conn, addr)

    stored_potential = (approval_score + capability_score + deployer_score
                        + org_score + observation_score)
    risk_score = (stored_potential * volatility) / max(realized, 1)

    return {
        "contract_address": addr,
        "stored_potential": stored_potential,
        "approval_scope_score": approval_score,
        "capability_score": capability_score,
        "deployer_risk_score": deployer_score,
        "org_context_score": org_score,
        "observation_capability_score": observation_score,
        "realized_value": realized,
        "volatility": volatility,
        "risk_score": round(risk_score, 2),
        "risk_tier": _risk_tier(risk_score),
        "components": {
            "approval_scope": approval_comp,
            "capability": capability_comp,
            "deployer_risk": deployer_comp,
            "org_context": org_comp,
            "observation_capability": observation_comp,
            "realized_value": realized_comp,
            "volatility": volatility_comp,
        },
    }


def batch_score(conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    """
    Score the top N contracts ordered by stored_potential (descending).

    First does a lightweight pass to estimate stored_potential, then
    fully scores the top candidates.
    """
    # Get all contract addresses with basic signals for pre-ranking
    rows = conn.execute("""
        SELECT c.contract_address,
               COALESCE(c.has_asymmetric_transfer, 0)
               + COALESCE(c.has_conditional_revert, 0)
               + COALESCE(c.has_unusual_fee_structure, 0) as signal_count,
               c.confidence_tier,
               c.deployer_address
        FROM contracts c
        ORDER BY signal_count DESC, c.detection_timestamp DESC
    """).fetchall()

    # Pre-filter: take a wider net than requested to account for re-ranking
    candidates = rows[: limit * 3] if len(rows) > limit * 3 else rows

    results = []
    for row in candidates:
        result = score_contract(conn, row["contract_address"])
        results.append(result)

    # Sort by stored_potential descending, then risk_score descending
    results.sort(key=lambda r: (r.get("stored_potential", 0), r.get("risk_score", 0)), reverse=True)
    return results[:limit]


def score_family(conn: sqlite3.Connection, family_id: str) -> list[dict[str, Any]]:
    """Score all members of a bytecode family."""
    if not _table_exists(conn, "bytecode_family_members"):
        return []

    rows = conn.execute(
        "SELECT contract_address FROM bytecode_family_members WHERE family_id = ?",
        (family_id,),
    ).fetchall()

    results = []
    for row in rows:
        result = score_contract(conn, row["contract_address"])
        results.append(result)

    results.sort(key=lambda r: r.get("risk_score", 0), reverse=True)
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _print_result(result: dict[str, Any]) -> None:
    """Pretty-print a single scoring result."""
    tier = result.get("risk_tier", "?")
    tier_markers = {
        "CRITICAL": "!!!",
        "HIGH": "!! ",
        "MEDIUM": "!  ",
        "LOW": ".  ",
        "MINIMAL": "   ",
    }
    marker = tier_markers.get(tier, "   ")
    print(f"  {marker} {result['contract_address']}")
    print(f"      stored_potential: {result.get('stored_potential', 0):>3}/100  "
          f"(approval={result.get('approval_scope_score', 0)} "
          f"capability={result.get('capability_score', 0)} "
          f"deployer={result.get('deployer_risk_score', 0)} "
          f"org={result.get('org_context_score', 0)})")
    print(f"      realized_value:   {result.get('realized_value', 0):>3}/25   "
          f"volatility: {result.get('volatility', 1.0):.1f}x")
    print(f"      risk_score:       {result.get('risk_score', 0):>7.2f}  "
          f"tier: {tier}")
    if result.get("error"):
        print(f"      ERROR: {result['error']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stored Potential Risk Scorer — Layer 3 surveillance analytics"
    )
    parser.add_argument("--address", type=str, help="Score a single contract address")
    parser.add_argument("--top", type=int, default=0, help="Score top N contracts by stored_potential")
    parser.add_argument("--family", type=str, help="Score all members of a bytecode family")
    parser.add_argument("--db", type=str, help="Path to surveillance database")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    conn = _get_conn(db_path)

    if args.address:
        result = score_contract(conn, args.address)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n[risk_scoring] Score for {args.address}:\n")
            _print_result(result)

    elif args.top > 0:
        results = batch_score(conn, limit=args.top)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n[risk_scoring] Top {args.top} contracts by stored_potential:\n")
            for i, r in enumerate(results, 1):
                print(f"  #{i}")
                _print_result(r)

    elif args.family:
        results = score_family(conn, args.family)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n[risk_scoring] Family {args.family} ({len(results)} members):\n")
            for r in results:
                _print_result(r)

    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
