"""
Bundle A — infrastructure_registry seed with Circle CCTP v2 contracts.

Idempotent: uses INSERT OR IGNORE so repeat runs don't duplicate.
Intended to run against local first, then Railway production via
`railway ssh 'python -m scripts.bundle_a_circle_bridge_insert'`.

Scope is intentionally narrow — 12 rows, all Circle CCTP v2 canonical
contracts (CREATE2-deterministic, same address on every EVM chain).
CCTP v1 is out of scope for this seed. Consumer-wrapper contracts
(if they ever exist as distinct addresses) are out of scope.

See reports/circle_bridge_infrastructure.md for the full Phase 1
findings and epistemic tagging.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

VERIF_SOURCE = (
    "https://developers.circle.com/cctp/evm-smart-contracts "
    "(fetched 2026-04-18)"
)

ROWS = [
    # (address, chain, classification, notes)
    ("0x28b5a0e9c621a5badaa536219b3a228c8168cf5d", "base",     "circle_cctp_token_messenger_v2",
     "CCTP v2 TokenMessenger. User-facing depositForBurn entry point. "
     "Stored potential: VERY HIGH (see reports/circle_bridge_infrastructure.md). "
     "Epistemic: deductive from Circle docs."),
    ("0x28b5a0e9c621a5badaa536219b3a228c8168cf5d", "arbitrum", "circle_cctp_token_messenger_v2",
     "CCTP v2 TokenMessenger. Same CREATE2 address on all chains."),
    ("0x28b5a0e9c621a5badaa536219b3a228c8168cf5d", "optimism", "circle_cctp_token_messenger_v2",
     "CCTP v2 TokenMessenger. Same CREATE2 address on all chains."),

    ("0x81d40f21f12a8f0e3252bccb954d722d4c464b64", "base",     "circle_cctp_message_transmitter_v2",
     "CCTP v2 MessageTransmitter. Cross-chain message relay. "
     "Circle-controlled upgradeable proxy (Tier B mutability claim)."),
    ("0x81d40f21f12a8f0e3252bccb954d722d4c464b64", "arbitrum", "circle_cctp_message_transmitter_v2",
     "CCTP v2 MessageTransmitter. Same CREATE2 address on all chains."),
    ("0x81d40f21f12a8f0e3252bccb954d722d4c464b64", "optimism", "circle_cctp_message_transmitter_v2",
     "CCTP v2 MessageTransmitter. Same CREATE2 address on all chains."),

    ("0xfd78ee919681417d192449715b2594ab58f5d002", "base",     "circle_cctp_token_minter_v2",
     "CCTP v2 TokenMinter. Mint/burn authority over USDC on destination chain. "
     "Maximum-capability node in the stablecoin ecosystem."),
    ("0xfd78ee919681417d192449715b2594ab58f5d002", "arbitrum", "circle_cctp_token_minter_v2",
     "CCTP v2 TokenMinter. Same CREATE2 address on all chains."),
    ("0xfd78ee919681417d192449715b2594ab58f5d002", "optimism", "circle_cctp_token_minter_v2",
     "CCTP v2 TokenMinter. Same CREATE2 address on all chains."),

    ("0xec546b6b005471ecf012e5af77fbec07e0fd8f78", "base",     "circle_cctp_message_v2",
     "CCTP v2 Message library. Message encoding/validation helpers."),
    ("0xec546b6b005471ecf012e5af77fbec07e0fd8f78", "arbitrum", "circle_cctp_message_v2",
     "CCTP v2 Message library. Same CREATE2 address on all chains."),
    ("0xec546b6b005471ecf012e5af77fbec07e0fd8f78", "optimism", "circle_cctp_message_v2",
     "CCTP v2 Message library. Same CREATE2 address on all chains."),
]


def main(argv):
    db_path = Path(argv[1]) if len(argv) > 1 else DB_PATH
    print(f"DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM infrastructure_registry WHERE classification LIKE 'circle_cctp_%'"
        )
        before = cur.fetchone()[0]
        print(f"Circle CCTP rows before: {before}")

        for addr, chain, cls, notes in ROWS:
            conn.execute(
                """INSERT OR IGNORE INTO infrastructure_registry
                   (address, chain, classification, verified_at,
                    verification_source, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (addr, chain, cls, "2026-04-18", VERIF_SOURCE, notes),
            )
        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*) FROM infrastructure_registry WHERE classification LIKE 'circle_cctp_%'"
        ).fetchone()[0]
        print(f"Circle CCTP rows after:  {after}")
        print(f"Inserted: {after - before} (expected 12 on first run, 0 on re-run)")
        print()
        print("Registry contents:")
        for r in conn.execute(
            "SELECT address, chain, classification FROM infrastructure_registry "
            "ORDER BY classification, chain"
        ):
            print(f"  {r[2]:40s}  {r[1]:10s}  {r[0]}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
