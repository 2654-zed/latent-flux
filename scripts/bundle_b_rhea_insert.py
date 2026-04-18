"""
Bundle B — extraction_events schema migration + EXTRACTION_004 (Rhea Finance).

Assumes the db.py migration has already run (adds chain + monitored_chain
columns, backfills existing rows as ethereum_l2_mixed). Inserts Rhea as
chain='near', monitored_chain=0.

Idempotent: INSERT OR IGNORE on event_id. Safe to re-run.

See reports/extraction_event_004_rhea_finance.md for the full incident
analysis and methodology notes.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

SUMMARY = (
    "Rhea Finance (formerly Burrow Finance) on NEAR Protocol. $18.4M drained "
    "2026-04-16 08:22-09:42 UTC via margin-trading slippage aggregation bug "
    "combined with fake-token oracle manipulation. Subject Wallet funded via "
    "cross-chain onboarding (intents.near). 423+ implicit accounts used as "
    "distribution infrastructure. 55 intermediary accounts deleted across 3 "
    "coordinated waves with Subject Wallet as sole beneficiary. ~45% recovery: "
    "$3.29M USDT frozen by Tether in attacker wallet + $1.05M USDT frozen in "
    "NEAR Intents + $3.36M USDC + $1.56M NEAR voluntarily returned ($8.26M "
    "total). Attack tx: 44tWhQmmkTJgchgFVkYpPrgyKvaH7wRLu1jZWXD3Du1x."
)

RAW_TX = {
    "attack_tx": "44tWhQmmkTJgchgFVkYpPrgyKvaH7wRLu1jZWXD3Du1x",
    "subject_wallet_creation": "2026-04-15T06:53:00+00:00",
    "subject_wallet_funding_source": "intents.near",
    "intermediary_accounts_total": "423+",
    "intermediary_accounts_deleted": 55,
    "mca_infrastructure": [
        "rhea000453.multica.near",
        "rhea000462.multica.near",
        "rhea000505.multica.near",
    ],
    "first_mca_activation": "rhea000453 storage_deposit on lst.rhealab.near @ 2026-04-14T03:49:00+00:00",
    "main_exploit_window": "2026-04-16T08:22/09:42 UTC",
    "fake_token_pool_ids_ref_finance": [8528, 8538],
    "affected_code": "burrowland/margin_trading.rs#L102",
    "recovery": {
        "tether_freeze_attacker_wallet_usdt": 3291000,
        "tether_freeze_near_intents_usdt": 1053000,
        "voluntary_return_usdc": 3359000,
        "voluntary_return_near": 1564000,
        "total_recovered_usd_approx": 8257000,
    },
}

NOTES = """ATTACK CATEGORY (Tier B interpretation): Compositional harm via oracle manipulation. Primary mechanism: fake token price manipulation via controlled liquidity pools. Secondary mechanism: slippage protection aggregation bug in margin trading. Parent pattern: oracle manipulation + lending protocol exploit (Drift copycat family).

ROOT CAUSE (Tier A from handoff): Burrow Protocol margin trading aggregated min_amount_out values across swap actions without accounting for intermediary token reuse between steps. Each individual min_amount_out was correctly implemented. Aggregation logic was correctly implemented. Swap execution was correctly implemented. The vulnerability was in the semantic gap between what the code computed (sum of step-level slippage tolerances) and what the protocol designers intended it to bound (end-to-end output quantity). Affected code: burrowland/margin_trading.rs#L102.

CROSS-CHAIN CORRELATION (Tier B): 15 days after Drift Protocol exploit ($285M, 2026-04-01, Solana). Same attack family: fake token + manipulated oracle + lending protocol drain. Different chain (NEAR vs Solana). Different trigger (code aggregation bug vs compromised multisig authority). Confirms Strategy Lifecycle prediction: EARLY -> ARMS_RACE transition within 15 days of public demonstration.

METHODOLOGY NOTES (Tier B, framework implications):
- NEAR account deletion pattern: 55 intermediary accounts deleted with asset transfer to hub. NEAR-specific architectural capability enabling ephemeral organizational infrastructure. Layer 3 persistent-wallet assumption breaks on NEAR; methodology adaptation required if NEAR expansion is considered.
- Cross-chain identity laundering (Pattern D validation): Subject Wallet funded from intents.near, creating no direct on-chain link to attacker pre-NEAR identity. Confirms Pattern D from behavioral laundering framework.
- Sybil infrastructure at scale: 423+ counterparty addresses used as operational wallets. Distribution from single hub wallet, deletion after use. auto_funder_tracer would identify hub on EVM chains; on NEAR, deletion removes evidence.
- Bytecode-equivalent detection gap: Fake tokens lacking NEP-141 metadata methods = NEAR equivalent of 'token without standard ERC-20 interface' -- detection pattern EVM classifier catches but isn't portable to NEAR Wasm.

RECOVERY COMPARISON (Tier B): Drift: ~$247.5M recovery vs $285M loss (~87% coverage, per handoff: Tether $127.5M + Solana Foundation $20M + $100M credit facility). Rhea: $8.2M recovery vs $18.4M loss (~45% coverage). Pattern: recovery rate depends heavily on stablecoin issuer intervention and project treasury voluntary return; varies widely by event.

TIER A (DEDUCTIVE) claims preserved from handoff:
- Attack transaction hash verified: 44tWhQmmkTJgchgFVkYpPrgyKvaH7wRLu1jZWXD3Du1x
- Affected code path: burrowland/margin_trading.rs#L102
- Tether froze specific USDT amounts (verifiable on-chain)
- Account deletion pattern confirmed via NEAR explorer (55 accounts deleted)
- Fake tokens deployed on implicit account addresses (verifiable)

TIER B (INFERENTIAL) claims labeled as such:
- Attack methodology interpretation
- Connection to Drift attack family (pattern similarity, not provenance)
- Strategy lifecycle classification (ARMS_RACE)
- Attribution of coordinated behavior across 423 counterparty addresses to single operator"""


def main(argv):
    db_path = Path(argv[1]) if len(argv) > 1 else DB_PATH
    print(f"DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        # Confirm schema migration landed
        cols = [c[1] for c in conn.execute("PRAGMA table_info(extraction_events)")]
        assert "chain" in cols and "monitored_chain" in cols, (
            "Schema migration missing; run db.init_db first"
        )
        print(f"extraction_events cols include chain/monitored_chain: OK")

        existing = conn.execute(
            "SELECT 1 FROM extraction_events WHERE event_id = 'EXTRACTION_004'"
        ).fetchone()
        if existing:
            print("EXTRACTION_004 already present — INSERT OR IGNORE will no-op.")

        conn.execute(
            """INSERT OR IGNORE INTO extraction_events (
                event_id, event_type, observed_at, documented_at,
                summary, raw_transactions, total_usd_moved, nodes_active,
                notes, chain, monitored_chain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "EXTRACTION_004",
                "oracle_manipulation_lending_exploit",
                "2026-04-16T08:22:00+00:00",
                "2026-04-18T00:00:00+00:00",
                SUMMARY,
                json.dumps(RAW_TX),
                18400000.0,
                423,
                NOTES,
                "near",
                0,
            ),
        )
        conn.commit()

        print("\nextraction_events snapshot:")
        for r in conn.execute(
            "SELECT event_id, event_type, chain, monitored_chain, total_usd_moved "
            "FROM extraction_events ORDER BY id"
        ):
            amt = f"${r[4]:,.0f}" if r[4] else "n/a"
            print(f"  {r[0]:18s}  {r[1]:38s}  {r[2]:20s}  monitored={r[3]}  {amt}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
