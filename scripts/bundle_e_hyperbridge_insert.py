"""
Bundle E — EXTRACTION_007 (Hyperbridge Token Gateway MMR proof verification bypass).

Cross-chain verification-code exploit. Distinct from Kelp (configuration
failure) and Aethir (admin-key compromise) in that this one is a CODE-level
bug in the proof verification logic.

Chain: Ethereum (monitored_chain=1 — this is on our monitored chains).

Draft-only. Not executed.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

SUMMARY = (
    "Hyperbridge Token Gateway cross-chain proof verification bypass. "
    "2026-04-13. Merkle Mountain Range (MMR) proof validation bug in "
    "HandlerV1.handlePostRequests allowed an out-of-bounds leaf index to "
    "pass validation, permitting the attacker to mint approximately 1B "
    "bridged DOT tokens on Ethereum without a corresponding burn on the "
    "source chain. Two-phase attack: initial ~245 ETH extraction via an "
    "early-test message, then the ~1B DOT mint. Initial on-chain loss "
    "estimate $237K; reconciliation may revise higher once bridged-DOT "
    "holders assess fair-value impact. Affected code: HandlerV1."
    "handlePostRequests — cross-chain verification logic that, when "
    "defeated, allows unauthorized minting on the destination chain."
)

RAW_TX = {
    "attack_date": "2026-04-13",
    "protocol": "Hyperbridge",
    "contract": "HandlerV1 / Token Gateway (Ethereum deployment)",
    "affected_code": "HandlerV1.handlePostRequests — MMR leaf-index validation",
    "verification_mechanism": "Merkle Mountain Range (MMR) proof",
    "vulnerability_shape": "out_of_bounds_leaf_index_bypass_validation",
    "chains_affected_as_target": ["ethereum", "base", "bnb", "arbitrum"],
    "two_phase_attack": {
        "phase_1": "~245 ETH extracted via early-test message (smaller scope)",
        "phase_2": "~1B bridged DOT minted on Ethereum without source burn",
    },
    "initial_usd_estimate": 237000,
    "reconciliation_pending": True,
    "attack_family": "cross_chain_proof_verification_bypass",
    "sibling_event_ids": ["EXTRACTION_006", "EXTRACTION_008"],
    "source": "Hyperbridge public incident statement + repository commit history",
}

NOTES = """ATTACK CATEGORY (Tier B interpretation): Code-level cross-chain verification bypass. Unlike Aethir (EXTRACTION_006: compromised operational control) and Kelp (EXTRACTION_008: misconfigured security layer), Hyperbridge is an actual code bug in proof verification logic. The Merkle Mountain Range proof format's leaf-index check allowed out-of-bounds values to skip validation; an attacker with access to the cross-chain messaging interface could submit a crafted proof that the verification function accepted despite being invalid, enabling unauthorized destination-chain minting.

This is the one exploit in the April-2026 cross-chain cluster that WOULD have been catchable by a traditional code audit — the bug is in verifiable code paths at a specific function. Hyperbridge's codebase was open-source; the bug was in a committed function. Whether an audit actually reviewed HandlerV1.handlePostRequests at the depth needed to catch an out-of-bounds check is unknown; the empirical fact is the bug made it to production.

ROOT CAUSE (Tier A from repo + incident statement): HandlerV1.handlePostRequests accepted MMR proofs where the leaf-index parameter was outside the valid range for the committed tree. Valid proofs should reference leaves at indices within the current MMR size; out-of-bounds indices should fail early validation. The guard was missing or misapplied, so crafted proofs bypassed the check and the handler treated them as authenticated.

CROSS-CHAIN CORRELATION (Tier B): Part of the April-2026 cross-chain infrastructure exploitation cluster with EXTRACTION_006 (Aethir, admin compromise) and EXTRACTION_008 (Kelp, DVN misconfiguration). All three attack the bridge-adjacent surface:
- Aethir: operational layer (who holds the admin key)
- Hyperbridge: code layer (does the verification function catch invalid input)
- Kelp: configuration layer (is the validator set configured with sufficient redundancy)

Three orthogonal attack vectors against the same structural target — pooled-custody cross-chain messaging — within 9 days. The rapid succession matches the Strategy Lifecycle EARLY->ARMS_RACE transition.

METHODOLOGY NOTES (Tier B, framework implications):
- The MMR bypass is outside our bytecode classifier's detection surface — the vulnerability is in Solidity logic for cross-chain proof verification, not in the trap-signature space our classifier targets. Detection would require audit-style code review.
- However, the downstream ECONOMIC SIGNAL was detectable: a ~1B-token mint on Ethereum with no corresponding source-chain burn is a balance-accounting anomaly. If we indexed bridged-asset mint events and compared against known source-chain burns (cross-chain conservation check), this attack would have been observable within minutes of execution. Not built; flagged as architectural extension for future consideration.
- The 'bridges are now the dominant attack surface' thesis from this month's three-incident cluster is a commercial-framing asset. Layer 3's positioning should reflect that traditional bytecode classification isn't the only detection surface — cross-chain balance-accounting is a distinct methodology that could add meaningful coverage.

TIER A (DEDUCTIVE) claims from public sources:
- Attack date 2026-04-13
- Affected contract: HandlerV1 in the Hyperbridge Token Gateway deployment
- Vulnerability: MMR proof leaf-index bypass in handlePostRequests function
- Target chain: Ethereum (also impacts Base, BNB, Arbitrum as bridging destinations)
- Initial loss estimate $237K (verifiable via post-mortem Hyperbridge statement)
- Two-phase attack structure (verifiable via Ethereum tx history of the exploit window)

TIER B (INFERENTIAL) claims labeled as such:
- Attack-family classification (cross_chain_proof_verification_bypass)
- Grouping with EXTRACTION_006 and 008 as the April-2026 cross-chain cluster
- Commercial-framing implications
- Methodology-extension proposals (mint-event conservation check)

CROSS-REFERENCES:
- EXTRACTION_006 (Aethir) — same attack family, different mechanism
- EXTRACTION_008 (Kelp) — same attack family, different mechanism
- reports/behavioral_laundering_detection_scope.md — Pattern D validated by cross-chain attacks this month"""


def main(argv):
    db_path = Path(argv[1]) if len(argv) > 1 else DB_PATH
    print(f"DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(extraction_events)")]
        assert "chain" in cols and "monitored_chain" in cols, (
            "Schema migration missing"
        )

        existing = conn.execute(
            "SELECT 1 FROM extraction_events WHERE event_id = 'EXTRACTION_007'"
        ).fetchone()
        if existing:
            print("EXTRACTION_007 already present — INSERT OR IGNORE will no-op.")

        conn.execute(
            """INSERT OR IGNORE INTO extraction_events (
                event_id, event_type, observed_at, documented_at,
                summary, raw_transactions, total_usd_moved, nodes_active,
                notes, chain, monitored_chain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "EXTRACTION_007",
                "cross_chain_proof_verification_bypass",
                "2026-04-13T00:00:00+00:00",
                "2026-04-18T00:00:00+00:00",
                SUMMARY,
                json.dumps(RAW_TX),
                237000.0,
                4,  # chains affected as bridging destinations
                NOTES,
                "ethereum",
                1,  # monitored_chain=1: Hyperbridge Token Gateway deployment on ETH
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
