"""Kelp retrospective Phase 2 — infrastructure_registry retrospective entries.

Adds six Kelp-related contracts to infrastructure_registry with
classification='retrospective_*' naming convention. These are NOT
forward-watch entries — they document what WOULD have been in the
registry had Layer 3 monitored Ethereum / LayerZero OApp configs.

Schema decision per Phase 1 pre-flight: use classification-string
convention rather than adding a new `retrospective` column. The notes
field carries the explicit "retrospective reference — not a forward-
watch entry" label. Zero schema churn.

Scope decision: include Unichain DVN even though we don't monitor
Unichain. infrastructure_registry becomes the authoritative list of
"contracts we know are architecturally interesting" regardless of
whether we have active ingest on that chain. Scoping to monitored-only
would artificially limit the registry's documentary value.

Zero RPC. Idempotent INSERT OR IGNORE.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"

VERIF_SOURCE = (
    "https://github.com/DK27ss/KelpDAO-294m-PoC (PoC repo + tx 0x1ae232...db4222); "
    "Blockaid statement 2026-04-18; LayerZero network response"
)

ROWS = [
    (
        "0x85d456b2dff1fd8245387c0bfb64dfb700e98ef3", "ethereum",
        "retrospective_kelp_oft_adapter",
        "Kelp rsETH OFT adapter (LayerZero). Drained 2026-04-18 via "
        "forged lzReceive. Retrospective reference entry; Layer 3 did "
        "not monitor Ethereum at attack time. See EXTRACTION_008.",
    ),
    (
        "0x589dedbd617e0cbcb916a9223f4d1300c294236b", "ethereum",
        "retrospective_kelp_dvn_ethereum",
        "Kelp required DVN (Ethereum destination). 1-of-1 required set "
        "at attack time, publicly observable via getConfig(configType=2). "
        "Retrospective reference entry.",
    ),
    (
        "0x282b3386571f7f794450d5789911a9804fa346b4", "unichain",
        "retrospective_kelp_dvn_unichain",
        "Kelp required DVN (Unichain source). 1-of-1 required set at "
        "attack time. Unichain is NOT in Layer 3 active monitoring "
        "(srcEid 30320). Retrospective reference entry.",
    ),
    (
        "0xc02ab410f0734efa3f14628780e6e695156024c2", "ethereum",
        "retrospective_layerzero_endpoint_receive_lib",
        "LayerZero Endpoint receive library (Ethereum). Referenced in "
        "attack tx; infrastructure-scope reference entry.",
    ),
    (
        "0xc39161c743d0307eb9bcc9fef03eeb9dc4802de7", "unichain",
        "retrospective_layerzero_endpoint_send_lib",
        "LayerZero Endpoint send library (Unichain). Referenced in "
        "attack source leg; infrastructure-scope reference entry.",
    ),
    (
        "0x8b1b6c9a6db1304000412dd21ae6a70a82d60d3b", "ethereum",
        "retrospective_kelp_attack_recipient",
        "Kelp attack recipient. Fresh address per public reporting; "
        "received 116,500 rsETH at nonce 308. This is the operator "
        "address, not infrastructure; included here ONLY for retrospective "
        "cross-referencing. Do NOT treat as 'known-legit infrastructure.' "
        "Entry type distinct from other 5: this is a flagged adversarial "
        "address on a non-monitored chain (Ethereum), preserved as "
        "retrospective evidence.",
    ),
]


def main(argv):
    db_path = Path(argv[1]) if len(argv) > 1 else DB_PATH
    print(f"DB: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    try:
        before = conn.execute(
            "SELECT COUNT(*) FROM infrastructure_registry "
            "WHERE classification LIKE 'retrospective_kelp%' "
            "OR classification LIKE 'retrospective_layerzero%'"
        ).fetchone()[0]
        print(f"retrospective_kelp/layerzero rows before: {before}")

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
            "SELECT COUNT(*) FROM infrastructure_registry "
            "WHERE classification LIKE 'retrospective_kelp%' "
            "OR classification LIKE 'retrospective_layerzero%'"
        ).fetchone()[0]
        print(f"retrospective_kelp/layerzero rows after:  {after}")
        print(f"inserted: {after - before} (expected 6 on first run)")
        print()
        print("Registry entries (retrospective + circle_cctp):")
        for r in conn.execute(
            "SELECT classification, chain, address "
            "FROM infrastructure_registry "
            "WHERE classification LIKE 'circle_cctp_%' "
            "OR classification LIKE 'retrospective_%' "
            "ORDER BY classification, chain"
        ):
            print(f"  {r[0]:48s}  {r[1]:10s}  {r[2]}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
