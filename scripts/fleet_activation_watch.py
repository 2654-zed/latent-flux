"""Fleet activation watch — report current dormant-activation state for a
deployer address. Used to confirm Attack-2 (Dormant Fleet + Proxy Upgrade)
profile candidates from the compositional zero-days catalog.

Usage:
    python scripts/fleet_activation_watch.py 0x<deployer>

Emits: fleet size, activations so far, timing, and a simple confirmation
signal — if N activations fire within a 48-hour window, that's the
Attack-2 shape worth escalating.

Default target: 0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e (209-contract
c0ffee-family fleet flagged 2026-04-18 after 2 activations in ~10.5 hours).
"""
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = "/app/surveillance/data/surveillance.db"
DEFAULT_DEPLOYER = "0xc0ffeefeed8b9d271445cf5d1d24d74d2ca4235e"
ATTACK2_THRESHOLD = 3  # >= N activations from the same fleet in 48h = escalate


def report(db_path: str, deployer: str) -> int:
    deployer = deployer.lower()
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row

    # Deployer row
    dep = c.execute(
        "SELECT * FROM deployers WHERE LOWER(deployer_address) = ?", (deployer,)
    ).fetchone()
    if not dep:
        print(f"deployer {deployer} not in corpus")
        return 1

    total = c.execute(
        "SELECT COUNT(*) FROM contracts WHERE LOWER(deployer_address) = ?",
        (deployer,),
    ).fetchone()[0]
    conf = c.execute(
        "SELECT COUNT(*) FROM contracts WHERE LOWER(deployer_address) = ? "
        "AND confidence_tier = 'confirmed'",
        (deployer,),
    ).fetchone()[0]
    susp = c.execute(
        "SELECT COUNT(*) FROM contracts WHERE LOWER(deployer_address) = ? "
        "AND confidence_tier = 'suspected'",
        (deployer,),
    ).fetchone()[0]
    unk = c.execute(
        "SELECT COUNT(*) FROM contracts WHERE LOWER(deployer_address) = ? "
        "AND confidence_tier = 'unknown'",
        (deployer,),
    ).fetchone()[0]

    print(f"=== fleet watch: {deployer} ===")
    print(f"  deployer first_seen={dep['first_seen']}  last_seen={dep['last_seen']}")
    print(f"  entity_type={dep['entity_type']}  pattern_notes={dep['deployment_pattern_notes']}")
    print(f"  fleet total={total}  confirmed={conf}  suspected={susp}  unknown={unk}")

    # Activation alerts
    activations = c.execute(
        "SELECT timestamp, tx_hash, payload FROM alerts "
        "WHERE alert_type = 'DORMANT_ACTIVATION' AND LOWER(address) = ? "
        "ORDER BY timestamp",
        (deployer,),
    ).fetchall()
    print(f"\n  DORMANT_ACTIVATION alerts for this deployer: {len(activations)}")
    for r in activations:
        try:
            p = json.loads(r["payload"] or "{}")
            nc = p.get("newly_activated")
            ta = p.get("total_active")
            fs = p.get("fleet_size")
        except Exception:
            nc = ta = fs = "?"
        print(f"    {r['timestamp'][:19]}  newly_activated={nc}  total_active={ta}  "
              f"fleet_size={fs}  tx={r['tx_hash']}")

    # 48-hour rolling check
    if len(activations) >= 1:
        from datetime import datetime, timezone, timedelta
        try:
            latest_ts = datetime.fromisoformat(activations[-1]["timestamp"].replace("Z", "+00:00"))
            window_start = (latest_ts - timedelta(hours=48)).isoformat()
            in_window = [a for a in activations if a["timestamp"] >= window_start]
            print(f"\n  activations in last 48h window (ending {activations[-1]['timestamp'][:19]}): {len(in_window)}")
            if len(in_window) >= ATTACK2_THRESHOLD:
                print(f"  *** ATTACK-2 THRESHOLD MET ({ATTACK2_THRESHOLD}+ activations in 48h) — escalate ***")
            else:
                gap_to_threshold = ATTACK2_THRESHOLD - len(in_window)
                print(f"  below Attack-2 threshold; {gap_to_threshold} more activation(s) in 48h would trigger")
        except Exception as e:
            print(f"  window analysis failed: {e}")

    return 0


def main(argv):
    db = DEFAULT_DB
    dep = DEFAULT_DEPLOYER
    if len(argv) >= 2:
        dep = argv[1]
    if len(argv) >= 3:
        db = argv[2]
    return report(db, dep)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
