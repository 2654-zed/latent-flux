"""
Layer 3 — Diamond Model Integration

Populates the diamond_model table from existing case files and investigations.
Each entry maps to the four Diamond Model vertices: Adversary, Capability,
Infrastructure, Victim.

Usage:
    python -m surveillance.diamond_model --import-cases
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

CASES = [
    {
        "case_id": "org_001",
        "adversary_id": "org_001",
        "adversary_type": "organization",
        "adversary_timezone": "americas (UTC-5 to UTC-8)",
        "adversary_operational_pattern": "night_shift",
        "capability_bytecode_family": "delegatecall_proxy",
        "capability_trap_mechanisms": json.dumps(["delegatecall_upgradeable", "conditional_revert"]),
        "capability_camouflage_rating": "MEDIUM",
        "capability_anti_forensic": json.dumps(["unicode_weth_impersonation", "multi_exit_channel"]),
        "infrastructure_chains": json.dumps(["arbitrum", "base"]),
        "infrastructure_wallets": json.dumps({
            "treasury": "0xf186cb00e49e18491db5783ff04fae3818102ff7",
            "operator": "0xe93d64f3fbc352131e79fc5578cbe44b66697f86",
            "cashout": "0xc6962004f452be9203591991d15f6b388e09e8d0",
            "exit_cex": "0x01989c93890aed05a63d179b03424997075b6acf",
        }),
        "infrastructure_delivery": "direct_deployment",
        "infrastructure_exit_channels": json.dumps(["cex_deposit", "defi_router", "lp_staging"]),
        "victim_count": 3400,
        "victim_type": "bots",
        "victim_return_rate": None,
        "victim_overlap_with": json.dumps([]),
        "confidence": "CONFIRMED",
    },
    {
        "case_id": "org_002",
        "adversary_id": "org_002",
        "adversary_type": "organization",
        "adversary_timezone": "asia_or_automated",
        "adversary_operational_pattern": "continuous_marathon",
        "capability_bytecode_family": "T1-d5351e977044",
        "capability_trap_mechanisms": json.dumps(["tx_origin_conditional", "buy_not_sell"]),
        "capability_camouflage_rating": "MEDIUM",
        "capability_anti_forensic": json.dumps(["disposable_deployers", "one_contract_per_deployer"]),
        "infrastructure_chains": json.dumps(["base"]),
        "infrastructure_wallets": json.dumps({
            "treasury_senior": "0x238d7170f309a55b87a144a341bd6105897082ca",
            "treasury_junior": "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473",
        }),
        "infrastructure_delivery": "direct_deployment",
        "infrastructure_exit_channels": json.dumps(["unknown"]),
        "victim_count": 1660,
        "victim_type": "mixed",
        "victim_return_rate": None,
        "victim_overlap_with": json.dumps([]),
        "confidence": "CONFIRMED",
    },
    {
        "case_id": "org_003",
        "adversary_id": "org_003",
        "adversary_type": "organization",
        "adversary_timezone": "unknown",
        "adversary_operational_pattern": "unknown",
        "capability_bytecode_family": "T1-sha3_sload_fee",
        "capability_trap_mechanisms": json.dumps(["obfuscated_fee_on_transfer", "sha3_sload_jumpi_div"]),
        "capability_camouflage_rating": "HIGH",
        "capability_anti_forensic": json.dumps(["ghost_deployers", "invisible_funding", "zero_traceable_inflows"]),
        "infrastructure_chains": json.dumps(["base"]),
        "infrastructure_wallets": json.dumps({"deployers": "6 ghost addresses"}),
        "infrastructure_delivery": "direct_deployment",
        "infrastructure_exit_channels": json.dumps(["unknown"]),
        "victim_count": 727,
        "victim_type": "mixed",
        "victim_return_rate": None,
        "victim_overlap_with": json.dumps([]),
        "confidence": "HIGH",
    },
    {
        "case_id": "parasite_0xd4624228",
        "adversary_id": "0xe8e0c4883d7196a7de87a6489f6da58212dbe813",
        "adversary_type": "individual",
        "adversary_timezone": "unknown",
        "adversary_operational_pattern": "one_shot",
        "capability_bytecode_family": "multi_interface_impersonation",
        "capability_trap_mechanisms": json.dumps(["asymmetric_buy_not_sell", "obfuscated_fee", "multi_interface_37_selectors"]),
        "capability_camouflage_rating": "CRITICAL",
        "capability_anti_forensic": json.dumps(["unicode_weth_impersonation", "multi_interface_impersonation", "disposable_deployer"]),
        "infrastructure_chains": json.dumps(["base"]),
        "infrastructure_wallets": json.dumps({
            "deployer": "0xe8e0c4883d7196a7de87a6489f6da58212dbe813",
            "collection_1": "0xd462be33c46d84a0ce702103336f2fc290dcf159",
            "collection_2": "0xe502b1568aba07040a4580717e3399297067c50e",
            "collection_3": "0x07bd23d6ae11e61450ea74c4d96e21f3946eacb6",
        }),
        "infrastructure_delivery": "router_injected",
        "infrastructure_exit_channels": json.dumps(["direct_weth_withdrawal"]),
        "victim_count": 2910,
        "victim_type": "retail_users",
        "victim_return_rate": 78.0,
        "victim_overlap_with": json.dumps([]),
        "confidence": "CONFIRMED",
    },
]


def import_cases():
    conn = sqlite3.connect(str(DB_PATH))
    now = datetime.now(timezone.utc).isoformat()

    for case in CASES:
        conn.execute("""
            INSERT OR REPLACE INTO diamond_model
            (case_id, adversary_id, adversary_type, adversary_timezone,
             adversary_operational_pattern, capability_bytecode_family,
             capability_trap_mechanisms, capability_camouflage_rating,
             capability_anti_forensic, infrastructure_chains,
             infrastructure_wallets, infrastructure_delivery,
             infrastructure_exit_channels, victim_count, victim_type,
             victim_return_rate, victim_overlap_with, created_at, updated_at, confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            case["case_id"], case["adversary_id"], case["adversary_type"],
            case["adversary_timezone"], case["adversary_operational_pattern"],
            case["capability_bytecode_family"], case["capability_trap_mechanisms"],
            case["capability_camouflage_rating"], case["capability_anti_forensic"],
            case["infrastructure_chains"], case["infrastructure_wallets"],
            case["infrastructure_delivery"], case["infrastructure_exit_channels"],
            case["victim_count"], case["victim_type"], case["victim_return_rate"],
            case["victim_overlap_with"], now, now, case["confidence"],
        ))

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM diamond_model").fetchone()[0]
    conn.close()
    print(f"[diamond_model] Imported {len(CASES)} cases. Total: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-cases", action="store_true")
    args = parser.parse_args()
    if args.import_cases:
        import_cases()
