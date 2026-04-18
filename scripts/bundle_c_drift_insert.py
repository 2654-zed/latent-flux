"""
Bundle C — EXTRACTION_005 (Drift Protocol, Solana, 2026-04-01).

Parent event for EXTRACTION_004 (Rhea Finance, NEAR). Same attack family
(oracle_manipulation_lending_exploit), 15-day interval between
demonstrations confirming the Strategy Lifecycle EARLY -> ARMS_RACE
transition.

Reuses the extraction_events schema migration from Bundle B (chain +
monitored_chain columns). Idempotent via INSERT OR IGNORE on event_id.

Public post-mortem sources for the Tier A claims:
- Drift Protocol incident report (April 2026)
- Elliptic attribution report (DPRK linkage)
- TRM Labs attack timeline
- Drift governance transaction history (Solana explorer)
Deck summaries in l3-narrative/Drift_Heist_Analysis.pptx.

Specific transaction signatures are well-documented in public
post-mortems but not transcribed here; raw_transactions references
the event rather than claiming specific sigs from memory.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

SUMMARY = (
    "Drift Protocol (Solana) governance takeover + vault drain. 2026-04-01 "
    "16:05 UTC. $285M stolen in ~12 minutes across two transactions four "
    "slots apart. Attack composed correctly-functioning components: "
    "durable-nonce pre-signed transactions (from 2 of 5 Security Council "
    "members) + Drift's 2026-03-27 governance change reducing threshold "
    "3/5 -> 2/5 with zero timelock + CarbonVote Token (CVT) accepted as "
    "collateral after $500 Raydium liquidity pool + wash-traded $1 price "
    "history. Laundering via Jupiter -> deBridge -> Wormhole -> Ethereum -> "
    "Tornado Cash. Attributed to DPRK by Elliptic and TRM Labs. Recovery "
    "~$247.5M: Tether freeze $127.5M + Solana Foundation $20M + $100M "
    "revenue credit facility. Parent event for EXTRACTION_004 (Rhea); "
    "same attack family, 15-day interval."
)

RAW_TX = {
    "execution_window_utc": "2026-04-01T16:05:00+00:00",
    "execution_duration_minutes": 12,
    "execution_tx_count": 2,
    "execution_tx_spacing_slots": 4,
    "attributed_threat_actor": "DPRK (per Elliptic, TRM Labs)",
    "staging_timeline": {
        "2026-03-11": "Attacker staging begins (infrastructure, token mfg, social engineering)",
        "2026-03-23": "4 durable nonce accounts created; 2 tied to Security Council member signatures",
        "2026-03-23": "CarbonVote Token (CVT) deployed; 750M supply minted",
        "2026-03-23_to_2026-03-27": "CVT wash-traded on $500 Raydium pool to ~$1 price history",
        "2026-03-27": "Drift governance change: threshold 3/5 -> 2/5, zero timelock",
        "2026-03-30": "Re-phished updated council member nonce (quorum restored under new config)",
        "2026-04-01T16:05": "Execution: admin takeover + vault drain",
        "2026-04-01T16:20": "Laundering begins",
    },
    "oracle_manipulation_mechanism": {
        "fake_token": "CarbonVote Token (CVT)",
        "fake_token_supply": 750000000,
        "seed_liquidity_usd": 500,
        "seed_venue": "Raydium",
        "manipulated_price_target_usd": 1.0,
        "drift_oracle_behavior": "Accepted Raydium pool price as real; CVT admitted as collateral at hundreds of millions in implied value",
    },
    "governance_mechanism": {
        "multisig_before": "3/5 with timelock",
        "multisig_after": "2/5 with zero timelock (as of 2026-03-27)",
        "pre_signed_sigs_held_by_attacker": 2,
        "pre_sign_technique": "Durable nonce accounts (Solana native; transaction stays valid indefinitely vs ~90s normal expiry)",
    },
    "laundering_path": [
        "Jupiter (Solana DEX aggregator)",
        "deBridge",
        "Wormhole",
        "Ethereum mainnet",
        "Tornado Cash",
    ],
    "recovery_sources": {
        "tether_usdt_freeze": 127500000,
        "solana_foundation_grant": 20000000,
        "revenue_credit_facility": 100000000,
        "total_recovered_usd": 247500000,
    },
    "affected_tvl_usd": 550000000,
    "amount_stolen_usd": 285000000,
}

NOTES = """ATTACK CATEGORY (Tier B interpretation): Compositional harm at protocol governance layer. Primary mechanism: oracle manipulation via controlled liquidity pool. Secondary mechanism: pre-signed durable-nonce governance takeover under post-reduction threshold. Parent pattern of EXTRACTION_004 (Rhea Finance); both use fake-token + manipulated-oracle + lending-protocol-drain composition.

ROOT CAUSE (Tier A from public post-mortems): Every component executed exactly as designed. The durable nonce system kept transactions valid indefinitely (working as designed). The multisig accepted 2 of 5 valid signatures (threshold was met). The oracle reported the price it observed (price was real on Raydium). The vault contracts executed authorized withdrawals (admin key was valid). The security audits confirmed code was sound (Trail of Bits + ClawSecure pre-incident). The harm emerged from the composition of correctly-executing components, each trusting the one before it.

The governance change on 2026-03-27 (3/5 -> 2/5 multisig with zero timelock) was the phase transition. Pre-2026-03-27: system scored safe by Drift's own governance design. Post-2026-03-27: same protocol, same code, same team, but the attacker now had quorum with their pre-signed nonces. Drift's Adversarial Topology score (per our pre-hindsight simulation in reports/drift_prehindsight_simulation.md) reached 9/10 on the same date — five days before execution.

CROSS-CHAIN CORRELATION (Tier B): 15 days after Drift, Rhea Finance on NEAR replicates the composition pattern with different triggers (code aggregation bug in Rhea's margin trading vs. pre-signed governance in Drift). See EXTRACTION_004. Confirms Strategy Lifecycle EARLY -> ARMS_RACE transition within ~2 weeks of public demonstration. Expect further copies within 30 days per the lifecycle model.

METHODOLOGY NOTES (Tier B, framework implications):
- Durable nonces are the canonical 'stored potential in governance' example. Signed in Week -3, executable indefinitely, irrevocable by signer once the nonce account exists. The approval-that-never-expires of Permit2 at the wallet-user level translated to governance-multisig at the protocol level.
- Pre-hindsight behavioral scoring (reports/drift_prehindsight_simulation.md) scored Drift 9/10 on 2026-03-27 based purely on structural-tension signals: governance threshold reduced, timelock removed, fresh multisig wallet creations. The framework identifies the loaded-state correctly even when no malicious action has occurred.
- Post-mortem confirms the commercial-recovery asymmetry thesis: Tether alone accounted for $127.5M of recovery; centralized stablecoin issuer intervention capability is a competitive vector between USDC and USDT. Rhea's 45% recovery vs Drift's ~87% recovery supports the pattern.

TIER A (DEDUCTIVE) claims from public post-mortems:
- Attack execution window: 2026-04-01 16:05-16:17 UTC, two transactions four Solana slots apart
- Amount stolen: $285M (verifiable from Drift vault balance delta)
- TVL at incident: ~$550M
- Tether freeze: $127.5M USDT (verifiable on-chain)
- Multisig threshold change: 3/5 -> 2/5 on 2026-03-27 (verifiable on Solana explorer; Drift governance tx history)
- Timelock removal on 2026-03-27 (same tx; verifiable)
- CVT deployment tx (verifiable; specific sig in Drift incident report)
- Laundering path: Jupiter -> deBridge -> Wormhole -> Ethereum -> Tornado Cash (verifiable per DeBridge and Wormhole tx logs)
- DPRK attribution per Elliptic + TRM Labs (public reports)

TIER B (INFERENTIAL) claims labeled as such:
- Attack family classification (oracle_manipulation_lending_exploit)
- Connection to EXTRACTION_004 Rhea Finance through pattern similarity
- Strategy Lifecycle ARMS_RACE classification
- Economic recovery pattern interpretation (% coverage as commercial signal)
- Durable nonce framing as 'governance-layer stored potential'
- Pre-hindsight scoring significance

CROSS-REFERENCES:
- reports/drift_simulation.md — initial analysis document
- reports/drift_prehindsight_simulation.md — 9/10 score 5 days pre-execution
- reports/post_drift_impact.md — downstream ecosystem effects
- l3-narrative/Drift_Heist_Analysis.pptx — narrative/pitch materials
- EXTRACTION_004 — Rhea Finance copycat, 15 days later"""


def main(argv):
    db_path = Path(argv[1]) if len(argv) > 1 else DB_PATH
    print(f"DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(extraction_events)")]
        assert "chain" in cols and "monitored_chain" in cols, (
            "Schema migration missing; run db.init_db first (Bundle B)"
        )

        existing = conn.execute(
            "SELECT 1 FROM extraction_events WHERE event_id = 'EXTRACTION_005'"
        ).fetchone()
        if existing:
            print("EXTRACTION_005 already present — INSERT OR IGNORE will no-op.")

        conn.execute(
            """INSERT OR IGNORE INTO extraction_events (
                event_id, event_type, observed_at, documented_at,
                summary, raw_transactions, total_usd_moved, nodes_active,
                notes, chain, monitored_chain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "EXTRACTION_005",
                "oracle_manipulation_lending_exploit",
                "2026-04-01T16:05:00+00:00",
                "2026-04-18T00:00:00+00:00",
                SUMMARY,
                json.dumps(RAW_TX),
                285000000.0,
                5,  # Security Council members involved; not directly comparable to 423 Rhea nodes
                NOTES,
                "solana",
                0,
            ),
        )
        conn.commit()

        print("\nextraction_events full snapshot:")
        for r in conn.execute(
            "SELECT event_id, event_type, chain, monitored_chain, total_usd_moved, "
            "observed_at FROM extraction_events ORDER BY observed_at"
        ):
            amt = f"${r[4]:>13,.0f}" if r[4] else " n/a"
            print(f"  {r[5][:10]}  {r[0]:18s}  {r[1]:38s}  {r[2]:20s}  monitored={r[3]}  {amt}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
