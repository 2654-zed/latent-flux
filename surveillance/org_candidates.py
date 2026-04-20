"""Novel-org candidate detector.

Exception-as-rule audit (P1) finding: org classification is a hardcoded
13-wallet allowlist. A novel group operating with different timezone / gas
fingerprint / funding pattern would never surface as an org candidate — it
would just generate unlinked `suspected` contracts forever.

This module scans `deployers` for clusters that look organizational but have
no `org_wallets` entry. Emits rows to `org_candidates` for review. Two rules
stacked cheaply on top of existing columns:

  Rule A: SHARED_FUNDING_CLUSTER
    Deployers whose funding_trail / funding_sources overlap — i.e., three
    or more deployers funded by the same upstream address or the same mixer
    / CEX hot wallet. Weak signal on its own (legitimate clusters exist
    too: faucets, exchange deposit-to-deploy flows) but strong in
    combination with Rule B.

  Rule B: GAS_FINGERPRINT_NEAR_EQUAL
    Deployers whose `typical_gas_price_gwei` fall within a tight band
    (default +/- 0.05 gwei) AND deploy within a short window (default 72h)
    AND share funding source with ≥2 siblings.

A cluster promotes to `org_candidates` only if it satisfies both A AND B,
has size >= 3, and no member appears in `org_wallets`. Clusters touching
any known org_wallet are skipped — those belong to an existing org.

Tier B inferential. Emits rows for human review, not for auto-classification.

CLI:
    python -m surveillance.org_candidates --dry-run
    python -m surveillance.org_candidates --apply
    python -m surveillance.org_candidates --apply --min-size 5 --window-hours 48
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "surveillance.db"

DEFAULT_MIN_SIZE = 3
DEFAULT_GAS_TOLERANCE_GWEI = 0.05
DEFAULT_WINDOW_HOURS = 72


def _known_org_addresses(conn: sqlite3.Connection) -> set:
    """Return lowercase set of every address in org_wallets."""
    rows = conn.execute("SELECT address FROM org_wallets").fetchall()
    return {r[0].lower() for r in rows}


def _parse_funding_list(raw: str) -> list[str]:
    """funding_sources column is a JSON list (sometimes empty)."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(v, list):
        return []
    return [str(x).lower() for x in v]


def compute_clusters(conn: sqlite3.Connection, *,
                     min_size: int = DEFAULT_MIN_SIZE,
                     gas_tolerance_gwei: float = DEFAULT_GAS_TOLERANCE_GWEI,
                     window_hours: int = DEFAULT_WINDOW_HOURS,
                     ) -> list[dict]:
    """Group deployers by shared funding source + tight gas band. Return candidate clusters."""
    known = _known_org_addresses(conn)

    # Pull every deployer with a funding source AND a gas fingerprint
    rows = conn.execute(
        """
        SELECT deployer_address, chain, funding_sources, typical_gas_price_gwei,
               first_seen, last_seen, entity_type
        FROM deployers
        WHERE funding_sources IS NOT NULL
          AND funding_sources != ''
          AND funding_sources != '[]'
          AND typical_gas_price_gwei IS NOT NULL
        """
    ).fetchall()

    # Group by each unique funding-source address encountered
    by_funder: dict[str, list[dict]] = {}
    for r in rows:
        addr = r[0].lower()
        if addr in known:
            continue  # skip already-classified wallets
        funders = _parse_funding_list(r[2])
        for f in funders:
            f_low = f.lower()
            if f_low in known:
                continue  # don't seed clusters from known org treasury
            by_funder.setdefault(f_low, []).append({
                "deployer": addr,
                "chain": r[1],
                "funding_source": f_low,
                "gas": float(r[3]),
                "first_seen": r[4],
                "last_seen": r[5],
                "entity_type": r[6],
            })

    # For each funder group, find sub-clusters within the gas band + time window
    clusters: list[dict] = []
    window_sec = window_hours * 3600
    for funder, members in by_funder.items():
        if len(members) < min_size:
            continue
        # sort by gas so banding is linear-time
        members.sort(key=lambda m: m["gas"])
        i = 0
        while i < len(members):
            band_start_gas = members[i]["gas"]
            j = i
            while j < len(members) and members[j]["gas"] - band_start_gas <= gas_tolerance_gwei:
                j += 1
            band = members[i:j]
            if len(band) >= min_size:
                # filter band by time window — only keep members whose first_seen
                # falls within +/- window from the median
                ts = []
                for m in band:
                    try:
                        ts.append(datetime.fromisoformat(m["first_seen"].replace("Z", "+00:00")))
                    except (ValueError, AttributeError):
                        pass
                if len(ts) >= min_size:
                    ts.sort()
                    median = ts[len(ts) // 2]
                    in_window = [
                        m for m, t in zip(band, ts)
                        if abs((t - median).total_seconds()) <= window_sec
                    ]
                    if len(in_window) >= min_size:
                        gases = [m["gas"] for m in in_window]
                        chains = sorted({m["chain"] for m in in_window})
                        clusters.append({
                            "funding_source": funder,
                            "members": in_window,
                            "size": len(in_window),
                            "gas_mean": sum(gases) / len(gases),
                            "gas_span": max(gases) - min(gases),
                            "chain": chains[0] if len(chains) == 1 else ",".join(chains),
                            "first_seen": min(m["first_seen"] for m in in_window),
                            "last_seen": max(m["last_seen"] for m in in_window),
                        })
            i = j if j > i else i + 1

    # Dedupe — if the same set of deployers surfaces under two different funders
    seen: set = set()
    out = []
    for c in clusters:
        key = tuple(sorted(m["deployer"] for m in c["members"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    out.sort(key=lambda c: (-c["size"], c["gas_mean"]))
    return out


def _candidate_id(cluster: dict) -> str:
    """Stable ID from the sorted member list."""
    members = ",".join(sorted(m["deployer"] for m in cluster["members"]))
    h = hashlib.sha256(members.encode()).hexdigest()[:12]
    return f"orgcand_{h}"


def apply_clusters(conn: sqlite3.Connection, clusters: list[dict]) -> dict:
    """Insert / update org_candidates rows for each cluster."""
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    refreshed = 0
    for c in clusters:
        cid = _candidate_id(c)
        existing = conn.execute(
            "SELECT id, status FROM org_candidates WHERE candidate_id = ?", (cid,)
        ).fetchone()
        member_json = json.dumps([m["deployer"] for m in c["members"]])
        if existing is None:
            conn.execute(
                """INSERT INTO org_candidates
                   (candidate_id, cluster_size, deployer_addresses,
                    shared_funding_source, shared_gas_fingerprint,
                    shared_chain, first_seen, last_seen, detected_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (cid, c["size"], member_json, c["funding_source"],
                 c["gas_mean"], c["chain"], c["first_seen"], c["last_seen"], now),
            )
            inserted += 1
        else:
            conn.execute(
                """UPDATE org_candidates SET last_seen = ?, detected_at = ?,
                       cluster_size = ?, deployer_addresses = ?
                       WHERE candidate_id = ?""",
                (c["last_seen"], now, c["size"], member_json, cid),
            )
            refreshed += 1
    conn.commit()
    return {"inserted": inserted, "refreshed": refreshed, "scanned": len(clusters)}


def main():
    ap = argparse.ArgumentParser(description="Novel-org candidate detector.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    ap.add_argument("--gas-tolerance-gwei", type=float, default=DEFAULT_GAS_TOLERANCE_GWEI)
    ap.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    ap.add_argument("--db", type=str, default=str(DB_PATH))
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("Pass --dry-run or --apply.")

    conn = sqlite3.connect(args.db, timeout=30)
    clusters = compute_clusters(
        conn, min_size=args.min_size,
        gas_tolerance_gwei=args.gas_tolerance_gwei,
        window_hours=args.window_hours,
    )
    print(f"[org_candidates] {len(clusters)} clusters found "
          f"(min_size={args.min_size}, gas_tol={args.gas_tolerance_gwei}, "
          f"window={args.window_hours}h)")
    for c in clusters[:10]:
        print(f"  size={c['size']:>3}  funder={c['funding_source']}  "
              f"gas={c['gas_mean']:.3f}+/-{c['gas_span']:.3f}  chain={c['chain']}")
    if args.dry_run:
        conn.close()
        return
    result = apply_clusters(conn, clusters)
    print(f"[org_candidates] applied: inserted={result['inserted']} "
          f"refreshed={result['refreshed']}")
    conn.close()


if __name__ == "__main__":
    main()
