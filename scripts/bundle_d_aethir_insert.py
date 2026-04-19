"""
Bundle D — EXTRACTION_006 (Aethir OFTAdapter admin compromise, BNB Chain).

Precursor case study to EXTRACTION_008 (Kelp) — same OFT-adapter attack
family, different compromise mechanism (EOA key vs DVN configuration).

Chain: BNB Chain (monitored_chain=0).

Draft-only. Not executed. Approval-gated same as EXTRACTION_004/005.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

SUMMARY = (
    "Aethir AethirOFTAdapter (LayerZero OFT adapter) on BNB Chain. "
    "2026-04-09. Attacker gained admin privileges over the adapter via "
    "EOA private-key compromise; the adapter was owned by a single EOA "
    "with no multisig and no timelock. Once admin, the attacker drained "
    "bridged assets from the adapter and bridged out to TRON via "
    "Symbiosis Finance. Gross drain reported ~$400K+, net user loss "
    "<$90K after recovery actions. Precursor case to EXTRACTION_008 "
    "(Kelp, 2026-04-18) — same OFT-adapter attack surface, different "
    "compromise mechanism (key vs DVN configuration)."
)

RAW_TX = {
    "attack_date": "2026-04-09",
    "protocol": "Aethir",
    "adapter_contract": "AethirOFTAdapter (LayerZero OFT adapter pattern)",
    "chain": "bnb",
    "compromise_mechanism": "EOA private key compromise (admin ownership)",
    "pre_attack_state": {
        "adapter_owner_type": "EOA",
        "multisig_present": False,
        "timelock_present": False,
    },
    "laundering_path": [
        "drained BNB Chain adapter balances",
        "bridged out to TRON via Symbiosis Finance",
    ],
    "amount_gross_usd": 400000,
    "amount_net_user_loss_usd": 90000,
    "source": "dev.to writeup 2026-04, developer post-mortem",
    "attack_family": "oft_adapter_admin_compromise",
    "sibling_event_id": "EXTRACTION_008",
}

NOTES = """ATTACK CATEGORY (Tier B interpretation): Compositional harm via operational-control compromise. The bytecode of AethirOFTAdapter behaved exactly as designed — the adapter mints/burns correctly, the ownership-transfer admin function was implemented per the OFT standard. The failure was that the admin authority was a single EOA with no multisig wrapper and no timelock. Layer 3's stored-potential framework would score this contract at VERY HIGH pre-attack: maximum capability (mint authority), single-point-of-failure permissions (one private key), maximum trust binding (users trust Aethir brand), zero constraint (no timelock). The attacker did not exploit a code defect; they acquired the key.

ROOT CAUSE (Tier A from developer post-mortem): Single-EOA ownership of a pooled-custody bridge adapter. Per the dev.to writeup (2026-04-09): 'Ownership was changed for the AethirOFTAdapter Adapter. The hacker immediately got admin privileges and can now change this Adapter contract in whatever manner they choose... The legitimate owner was just an eoa, leading to the conclusion for now that it's a private key compromise attack... The protocol had no multisig. They had no time wait mechanism. In 2026, this is not an acceptable level of operational security.'

CROSS-CHAIN CORRELATION (Tier B): Precursor to EXTRACTION_008 (KelpDAO, 2026-04-18). Both exploits are in the OFT-adapter attack family:
- Same architectural class: pooled-custody adapter bridging LST/LRT assets across LayerZero
- Same stored-potential characteristics: maximum capability + weak operational constraint
- Different trigger: Aethir = compromised admin EOA key; Kelp = misconfigured DVN set (1-of-1 on both chains)
- Same outcome: unauthorized mint/transfer on destination chain, drained adapter, bridged out to non-EVM chain (TRON / Ethereum -> off-ramp)

The 9-day gap between Aethir and Kelp is inside the Strategy Lifecycle EARLY->ARMS_RACE window (~15 days observed in EXTRACTION_005 -> EXTRACTION_004). OFT-adapter attacks are the emerging family of April 2026, paralleling the oracle-manipulation-lending family confirmed by Drift/Rhea.

METHODOLOGY NOTES (Tier B, framework implications):
- Stored-potential model: Aethir's adapter was pre-mint-authorized across the bridge. The gap between 'permission is granted' and 'permission is used' is exactly the stored-potential surface the framework measures. Pre-attack score would have been CRITICAL.
- Our methodology doesn't monitor BNB Chain; detection here would have required either (a) BNB Chain ingest extension, (b) infrastructure_registry coverage of major LayerZero adapters, or (c) governance-structure monitoring of adapter ownership patterns. All three are non-trivial extensions.
- The pattern matches Pattern F (advisor-parasite) in one dimension only: users don't realize the stored potential exists until the sweep fires. Unlike Pattern F, Aethir isn't a relationship-of-trust model; it's a single-event infrastructure compromise.

RECOVERY COMPARISON (Tier B):
- Aethir: gross ~$400K, net user loss ~$90K (~77% coverage, likely project-treasury backstop)
- Rhea (EXTRACTION_004): ~45% recovery
- Drift (EXTRACTION_005): ~87% recovery (Tether + Solana Foundation + credit facility)

TIER A (DEDUCTIVE) claims from public sources:
- Attack date 2026-04-09
- AethirOFTAdapter contract compromised
- Admin ownership was a single EOA (verifiable on-chain: Ownership-transferred event history)
- No multisig, no timelock (verifiable: contract does not reference multisig or timelock modules)
- Attacker bridged out via Symbiosis Finance (verifiable: Symbiosis withdrawal tx history)
- Laundering destination chain: TRON

TIER B (INFERENTIAL) claims labeled as such:
- Attack-family classification (OFT-adapter admin compromise)
- Connection to EXTRACTION_008 Kelp as sibling event
- Stored-potential framework applicability
- Methodology-implication statements above"""


def main(argv):
    db_path = Path(argv[1]) if len(argv) > 1 else DB_PATH
    print(f"DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(extraction_events)")]
        assert "chain" in cols and "monitored_chain" in cols, (
            "Schema migration missing; run Bundle B migration first"
        )

        existing = conn.execute(
            "SELECT 1 FROM extraction_events WHERE event_id = 'EXTRACTION_006'"
        ).fetchone()
        if existing:
            print("EXTRACTION_006 already present — INSERT OR IGNORE will no-op.")

        conn.execute(
            """INSERT OR IGNORE INTO extraction_events (
                event_id, event_type, observed_at, documented_at,
                summary, raw_transactions, total_usd_moved, nodes_active,
                notes, chain, monitored_chain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "EXTRACTION_006",
                "oft_adapter_admin_compromise",
                "2026-04-09T00:00:00+00:00",
                "2026-04-18T00:00:00+00:00",
                SUMMARY,
                json.dumps(RAW_TX),
                400000.0,
                1,  # one adapter contract
                NOTES,
                "bnb",
                0,
            ),
        )
        conn.commit()

        print("\nextraction_events snapshot:")
        for r in conn.execute(
            "SELECT event_id, event_type, chain, monitored_chain, total_usd_moved, observed_at "
            "FROM extraction_events ORDER BY observed_at"
        ):
            amt = f"${r[4]:>13,.0f}" if r[4] else "       n/a"
            print(f"  {r[5][:10]}  {r[0]:18s}  {r[1]:40s}  {r[2]:20s}  monitored={r[3]}  {amt}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
