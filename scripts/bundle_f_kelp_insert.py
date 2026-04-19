"""
Bundle F — EXTRACTION_008 (KelpDAO LayerZero DVN configuration failure).

Largest exploit of April 2026. Configuration-layer failure: DVN set was
1-of-1 on BOTH Unichain (source) and Ethereum (destination). Catastrophic
configuration was publicly visible on-chain via getConfig() for weeks
prior to exploitation.

Chain: Ethereum (destination, monitored_chain=1). Source: Unichain (not in
our monitored set). Downstream propagation to Arbitrum via Aave V3 deposit
(WITHIN our monitored set).

Draft-only. Not executed.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

SUMMARY = (
    "KelpDAO rsETH LayerZero OFT adapter, 2026-04-18. ~$292M drained via "
    "forged LayerZero cross-chain message from Unichain (source, srcEid "
    "30320) to Ethereum (destination). Root cause: DVN (Decentralized "
    "Verifier Network) configuration was 1-of-1 on BOTH chains — a single "
    "required DVN with zero optional DVNs and optionalDVNThreshold=0 — vs "
    "LayerZero documented best practice of >=2 required DVNs plus optional "
    "threshold. Attacker (fresh recipient 0x8B1b...0D3b, Tornado-funded per "
    "public reporting) received 116,500 rsETH (~18% of circulating supply) "
    "at nonce 308 GUID 0x3f4510...3b4e. A follow-on 40,000 rsETH attempt at "
    "nonce 309 was blocked by Kelp's TransfersBlocked response. Stolen rsETH "
    "was deposited to Aave V3 on Ethereum AND Arbitrum as collateral, against "
    "which ~$236M WETH was borrowed. Post-attack the collateral was unbacked, "
    "creating bad debt on Aave V3. Configuration-layer failure — not a code "
    "bug, not an admin-key compromise; a publicly-observable architectural "
    "choice that removed cross-chain message redundancy."
)

RAW_TX = {
    "attack_date": "2026-04-18",
    "protocol": "KelpDAO (rsETH)",
    "attack_tx_hash": "0x1ae232da212c45f35c1525f851e4c41d529bf18af862d9ce9fd40bf709db4222",
    "attack_block_eth": 24908285,
    "pre_state_block_eth": 24908284,
    "entry_call": "EndpointV2.lzReceive",
    "gas_used": 94456,
    "chains": {
        "source": "unichain",
        "source_eid": 30320,
        "destination": "ethereum",
        "propagation_observed_on": ["ethereum", "arbitrum"],
    },
    "contracts": {
        "kelp_oft_adapter_ethereum": "0x85d456b2dff1fd8245387c0bfb64dfb700e98ef3",
        "ethereum_required_dvn": "0x589dedbd617e0cbcb916a9223f4d1300c294236b",
        "unichain_required_dvn": "0x282b3386571f7f794450d5789911a9804fa346b4",
        "ethereum_endpoint_receive_library": "0xc02ab410f0734efa3f14628780e6e695156024c2",
        "unichain_endpoint_send_library": "0xc39161c743d0307eb9bcc9fef03eeb9dc4802de7",
        "attack_recipient": "0x8b1b6c9a6db1304000412dd21ae6a70a82d60d3b",
    },
    "drains": {
        "nonce_308_succeeded": {
            "guid": "0x3f4510d855cf3a805fec59daafae640d290749b7bf1e5450f91b5fb0018b3b4e",
            "amountSD_hex": "0x1b1ff0ed00",
            "amount_rsETH": 116500,
        },
        "nonce_309_blocked": {
            "guid": "0x19073f141ef29ea2eb2c52046e60942a928b2106651e622b73c68e27c969cfe6",
            "amount_rsETH": 40000,
            "blocked_by": "TransfersBlocked response",
            "blocked_at_utc": "2026-04-19T18:23:11",
        },
    },
    "dvn_configuration_at_attack_time": {
        "source_chain_unichain": {
            "requiredDVNCount": 1,
            "optionalDVNCount": 0,
            "optionalDVNThreshold": 0,
        },
        "destination_chain_ethereum": {
            "requiredDVNCount": 1,
            "optionalDVNCount": 0,
            "optionalDVNThreshold": 0,
        },
        "layerzero_best_practice": ">=2 required DVNs + optional threshold >=1",
        "publicly_observable_via_getConfig_pre_attack": True,
    },
    "downstream_propagation": {
        "aave_v3_collateral_deposit_ethereum": True,
        "aave_v3_collateral_deposit_arbitrum": True,
        "weth_borrowed_usd_approx": 236000000,
        "aave_bad_debt_created": True,
    },
    "amount_stolen_usd_approx": 292000000,
    "attack_family": "cross_chain_dvn_verification_failure",
    "sibling_event_ids": ["EXTRACTION_006", "EXTRACTION_007"],
    "source_material": [
        "https://github.com/DK27ss/KelpDAO-294m-PoC (PoC + deductive evidence)",
        "Blockaid public statement (2026-04-18)",
        "LayerZero network response statement",
    ],
}

NOTES = """ATTACK CATEGORY (Tier B interpretation): Configuration-layer failure in a cross-chain bridge architecture. No code bug. No compromised admin key. The DVN set was configured as 1-of-1 on both chains, a deliberate architectural choice that removed Byzantine fault tolerance from the cross-chain message verification. When one validator was compromised (or its signing key was obtained) the attacker could forge a valid-appearing cross-chain mint authorization with single-point effort.

This matches EXTRACTION_005 (Drift) at the configuration layer rather than the governance layer: both are 'stored potential via removed constraint,' both are publicly observable on-chain well before exploitation. The `CLAUDE.md` core interpretive rule applies directly: maximum capability (mint authority), maximum trust binding (Kelp + LayerZero brands), maximum victim exposure (rsETH holders), zero constraint (1-of-1 DVN). Pre-exploit stored-potential tier: CRITICAL. The exploit required zero code defects, zero surprise inputs — just an attacker willing to compromise one validator in a system that accepted 1-of-1 validation.

ROOT CAUSE (Tier A from repo + public statements): Kelp's rsETH OFT adapter required single-DVN signature on both source burn and destination mint. LayerZero documented best practice is >=2 required DVNs with optional threshold >=1. Kelp's deployed configuration had requiredDVNCount=1, optionalDVNCount=0, optionalDVNThreshold=0 on both sides (readable via EndpointV2.getConfig with configType=2). Configuration was stable across the observation window prior to the attack.

CROSS-CHAIN CORRELATION (Tier B): Third event in the April-2026 cross-chain infrastructure cluster:
- EXTRACTION_006 (Aethir, 2026-04-09, BNB, $400K): operational layer compromise (EOA key)
- EXTRACTION_007 (Hyperbridge, 2026-04-13, Ethereum, $237K): code layer bypass (MMR leaf-index)
- EXTRACTION_008 (Kelp, 2026-04-18, Ethereum, $292M): configuration layer failure (1-of-1 DVN)

Three orthogonal attack vectors against pooled-custody cross-chain infrastructure in 9 days. Aggregate cluster loss: ~$292.6M. Kelp alone exceeds the other two combined by 2 orders of magnitude. The cluster demonstrates the 'bridges are the dominant attack surface' thesis for April 2026 and validates the framework-level observation that cross-chain failures come in code, configuration, and operational flavors — traditional audits catch only the code flavor.

METHODOLOGY NOTES (Tier B, framework implications):
- Kelp is the textbook 'stored potential via architectural choice' case. The detection surface is DVN configuration enumeration, not bytecode classification. LayerZero DVN configs are on-chain readable via EndpointV2.getConfig.
- A surveillance module that enumerates DVN requirements for each LayerZero-connected OApp and flags requiredDVNCount=1 configs would have flagged Kelp as CRITICAL pre-exploit with zero ambiguity.
- Layer 3 currently does not monitor Ethereum or LayerZero OApp configurations. Retrospective replay scoped as `reports/kelp_retrospective_replay.md` will test what signals our methodology COULD have caught had it covered the relevant surface.
- Downstream Arbitrum leg (Aave V3 rsETH deposit, WETH borrow) touched our monitored chain; that is the one observation point where our existing pipeline should have captured attacker activity. Retrospective Phase 7 tests whether it did.

RECOVERY STATUS (as of 2026-04-18 documentation): Kelp + LayerZero coordination ongoing. Aave bad debt creates secondary recovery complexity. rsETH is a yield-bearing LRT, not a stablecoin; no Circle/Tether freeze path. Recovery likely requires protocol-native action.

TIER A (DEDUCTIVE) claims:
- Attack tx hash 0x1ae232...db4222 at Ethereum block 24908285 (verifiable)
- Kelp OFT adapter 0x85d4...8Ef3 (verifiable)
- Required DVNs 0x589d...236b (ETH) and 0x282b...46b4 (Unichain) (verifiable)
- DVN configuration 1-of-1 on both chains, with requiredDVNCount=1, optionalDVNCount=0, optionalDVNThreshold=0 (verifiable via historical getConfig call)
- 116,500 rsETH drained at nonce 308; 40,000 rsETH at nonce 309 blocked by TransfersBlocked
- Attack recipient 0x8B1b...0D3b received the drained rsETH (verifiable)
- Downstream Aave V3 deposits on Ethereum and Arbitrum (verifiable)
- ~$236M WETH borrowed against unbacked rsETH collateral (verifiable)

TIER B (INFERENTIAL) claims labeled as such:
- Attack-family classification (cross_chain_dvn_verification_failure)
- Grouping with 006/007 as cluster
- Stored-potential pre-exploit score framing (CRITICAL)
- Detection-module viability proposal (DVN configuration enumeration)
- Commercial-dataset framing

CROSS-REFERENCES:
- EXTRACTION_006 (Aethir) — precursor case, same attack family, 9 days earlier
- EXTRACTION_007 (Hyperbridge) — code-layer variant of cross-chain verification failure
- EXTRACTION_005 (Drift) — parallel 'stored potential via removed constraint' at governance layer
- reports/kelp_retrospective_replay.md — deep retrospective (scoped separately; 50-RPC-call budget) answering 'what signals would Layer 3 have caught if it monitored Ethereum and LayerZero OApp configs'
- reports/behavioral_laundering_detection_scope.md Pattern D — cross-chain attack surface is the dominant April-2026 vector"""


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
            "SELECT 1 FROM extraction_events WHERE event_id = 'EXTRACTION_008'"
        ).fetchone()
        if existing:
            print("EXTRACTION_008 already present — INSERT OR IGNORE will no-op.")

        conn.execute(
            """INSERT OR IGNORE INTO extraction_events (
                event_id, event_type, observed_at, documented_at,
                summary, raw_transactions, total_usd_moved, nodes_active,
                notes, chain, monitored_chain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "EXTRACTION_008",
                "cross_chain_dvn_verification_failure",
                "2026-04-18T00:00:00+00:00",
                "2026-04-18T00:00:00+00:00",
                SUMMARY,
                json.dumps(RAW_TX),
                292000000.0,
                2,  # source chain + destination chain adapter pair
                NOTES,
                "ethereum",
                1,
            ),
        )
        conn.commit()

        print("\nextraction_events full snapshot:")
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
