"""
RIB (Relational Intelligence Benchmark) — Phase 1: Anonymized Dataset Export

Exports the confirmed org_001 organization and its surrounding transaction graph
as an anonymized temporal edge list from the surveillance SQLite database.
"""

import csv
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fixed namespace for deterministic UUID generation
# ---------------------------------------------------------------------------
NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# ---------------------------------------------------------------------------
# org_001 ground truth
# ---------------------------------------------------------------------------
ORG_001: dict[str, dict[str, Any]] = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": {"role": "treasury", "confirmed": True},
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": {"role": "operator", "confirmed": True},
    "0xfd51e33d44b376ef346d24a130a51035db09c1dc": {"role": "operator_2", "confirmed": True},
    "0xc6962004f452be9203591991d15f6b388e09e8d0": {"role": "cashout", "confirmed": True},
    "0x01989c93890aed05a63d179b03424997075b6acf": {"role": "exit_cex", "confirmed": True},
    "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": {"role": "laundry", "confirmed": True},
    "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70": {"role": "lp_companion", "confirmed": True},
    "0x96daa0b8a5499ea9323421ed0cda06b345caab73": {"role": "lp_staging", "confirmed": True},
    "0x360e68faccca8ca495c1b759fd9eee466db9fb32": {"role": "treasury_branch", "confirmed": True},
    "0x3f2cdae910cd13638e38b45881e7a4fc3a9fe320": {"role": "victim", "confirmed": True},
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": {"role": "gas_station", "confirmed": True},
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": {"role": "defi_exit_channel", "confirmed": True},
}

ORG_001_EDGES: list[tuple[str, str, str]] = [
    ("treasury", "operator", "funds"),
    ("treasury", "treasury_branch", "funds"),
    ("treasury", "cashout", "funds"),
    ("gas_station", "operator_2", "provisions_gas"),
    ("treasury_branch", "lp_companion", "funds"),
    ("treasury_branch", "exit_cex", "funds"),
    ("operator", "cashout", "drains_to"),
    ("cashout", "exit_cex", "drains_to"),
    ("cashout", "defi_exit_channel", "obfuscates_through"),
    ("cashout", "lp_staging", "drains_to"),
    ("lp_staging", "cashout", "recycles_to"),
    ("laundry", "exit_cex", "converts_to"),
]


def anonymize(address: str, seed: int = 42) -> str:
    """Generate a deterministic UUID for a given address."""
    return str(uuid.uuid5(NAMESPACE, f"{address.lower()}:{seed}"))


def _ts_to_unix(ts_str: str) -> int:
    """Convert an ISO-8601 or datetime string to a Unix timestamp."""
    if not ts_str:
        return 0
    for fmt in ("%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S.%f+00:00",
                "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(ts_str, fmt).timestamp())
        except ValueError:
            continue
    # Last resort: strip timezone info and try
    try:
        clean = ts_str.split("+")[0].split("Z")[0]
        return int(datetime.fromisoformat(clean).timestamp())
    except Exception:
        return 0


def export_anonymized_dataset(db_path: str, output_dir: str = "rib_dataset", seed: int = 42) -> Path:
    """
    Export the full anonymized RIB dataset from the surveillance database.

    Args:
        db_path: Path to the surveillance SQLite database.
        output_dir: Directory to write the dataset into.
        seed: Seed for deterministic anonymization.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        raise RuntimeError(f"FATAL: Cannot open database at {db_path}: {e}") from e

    address_map: dict[str, str] = {}  # uuid -> original address
    edges: list[dict[str, Any]] = []

    def anon(addr: str) -> str:
        """Anonymize and track an address."""
        uid = anonymize(addr, seed)
        address_map[uid] = addr.lower()
        return uid

    # ------------------------------------------------------------------
    # 1. Transaction events -> contract_call edges
    # ------------------------------------------------------------------
    try:
        cur = conn.execute(
            "SELECT interacting_address, contract_address, timestamp, function_selector, is_reverted "
            "FROM transaction_events"
        )
        for row in cur:
            edges.append({
                "src_id": anon(row["interacting_address"]),
                "dst_id": anon(row["contract_address"]),
                "timestamp_unix": _ts_to_unix(row["timestamp"]),
                "weight_usd": 0.0,
                "edge_type": "contract_call",
            })
        print(f"[rib_export] Transaction events: {len(edges)} edges")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to query transaction_events: {e}") from e

    # ------------------------------------------------------------------
    # 2. Contract deployments -> contract_deploy edges
    # ------------------------------------------------------------------
    deploy_count = 0
    try:
        cur = conn.execute(
            "SELECT deployer_address, contract_address, detection_timestamp FROM contracts"
        )
        for row in cur:
            edges.append({
                "src_id": anon(row["deployer_address"]),
                "dst_id": anon(row["contract_address"]),
                "timestamp_unix": _ts_to_unix(row["detection_timestamp"]),
                "weight_usd": 0.0,
                "edge_type": "contract_deploy",
            })
            deploy_count += 1
        print(f"[rib_export] Contract deployments: {deploy_count} edges")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to query contracts: {e}") from e

    # ------------------------------------------------------------------
    # 3. Deployer funding trails -> eth_transfer edges
    # ------------------------------------------------------------------
    funding_count = 0
    try:
        cur = conn.execute(
            "SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL"
        )
        for row in cur:
            try:
                trail = json.loads(row["funding_trail"])
            except (json.JSONDecodeError, TypeError):
                continue
            funder = trail.get("funder")
            if not funder:
                continue
            ts = trail.get("timestamp", "")
            value_eth = trail.get("value_eth", 0.0)
            edges.append({
                "src_id": anon(funder),
                "dst_id": anon(row["deployer_address"]),
                "timestamp_unix": _ts_to_unix(ts),
                "weight_usd": 0.0,
                "edge_type": "eth_transfer",
            })
            funding_count += 1
        print(f"[rib_export] Funding trail edges: {funding_count} edges")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to query deployers.funding_trail: {e}") from e

    # ------------------------------------------------------------------
    # 4. Pattern matches -> pattern_match edges (if applicable)
    # ------------------------------------------------------------------
    pattern_count = 0
    try:
        cur = conn.execute("SELECT address, pattern_type, evidence FROM pattern_matches")
        for row in cur:
            try:
                ev = json.loads(row["evidence"]) if row["evidence"] else {}
            except (json.JSONDecodeError, TypeError):
                continue
            addr = row["address"]
            # Extract gas_seeds as edges
            for gs in ev.get("gas_seeds", []):
                to_addr = gs.get("to")
                ts = gs.get("ts", "")
                if to_addr:
                    edges.append({
                        "src_id": anon(addr),
                        "dst_id": anon(to_addr),
                        "timestamp_unix": _ts_to_unix(ts),
                        "weight_usd": 0.0,
                        "edge_type": "gas_seed",
                    })
                    pattern_count += 1
            # Extract USDC allocations as edges
            for alloc in ev.get("usdc_allocations", []):
                to_addr = alloc.get("to")
                ts = alloc.get("ts", "")
                usdc = alloc.get("usdc", 0.0)
                if to_addr:
                    edges.append({
                        "src_id": anon(addr),
                        "dst_id": anon(to_addr),
                        "timestamp_unix": _ts_to_unix(ts),
                        "weight_usd": float(usdc),
                        "edge_type": "usdc_transfer",
                    })
                    pattern_count += 1
            # Extract USDC returns as edges
            for ret in ev.get("usdc_returns", []):
                from_addr = ret.get("from")
                ts = ret.get("ts", "")
                usdc = ret.get("usdc", 0.0)
                if from_addr:
                    edges.append({
                        "src_id": anon(from_addr),
                        "dst_id": anon(addr),
                        "timestamp_unix": _ts_to_unix(ts),
                        "weight_usd": float(usdc),
                        "edge_type": "usdc_return",
                    })
                    pattern_count += 1
        print(f"[rib_export] Pattern match edges: {pattern_count} edges")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to query pattern_matches: {e}") from e

    # ------------------------------------------------------------------
    # 5. Write edges.csv
    # ------------------------------------------------------------------
    edges_path = out / "edges.csv"
    try:
        with open(edges_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["src_id", "dst_id", "timestamp_unix", "weight_usd", "edge_type"])
            writer.writeheader()
            writer.writerows(edges)
        print(f"[rib_export] Wrote {len(edges)} edges to {edges_path}")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to write edges.csv: {e}") from e

    # ------------------------------------------------------------------
    # 6. Build and write ground_truth.json
    # ------------------------------------------------------------------
    role_to_addr = {v["role"]: k for k, v in ORG_001.items()}

    gt: dict[str, Any] = {
        "org_001": {
            "nodes": {
                anon(addr): {"role": info["role"], "confirmed": info["confirmed"]}
                for addr, info in ORG_001.items()
            },
            "edges": [
                {
                    "src": anon(role_to_addr[src_role]),
                    "dst": anon(role_to_addr[dst_role]),
                    "relation": rel,
                }
                for src_role, dst_role, rel in ORG_001_EDGES
            ],
        },
        "bot_clusters": {},
    }

    # Bot clusters
    try:
        cur = conn.execute(
            "SELECT selector_cluster, COUNT(*) as members, SUM(total_revert_count) as reverts "
            "FROM bot_candidates WHERE selector_cluster IS NOT NULL "
            "GROUP BY selector_cluster"
        )
        for row in cur:
            gt["bot_clusters"][row["selector_cluster"]] = {
                "member_count": row["members"],
                "total_reverts": row["reverts"],
            }
        print(f"[rib_export] Bot clusters: {len(gt['bot_clusters'])}")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to query bot_candidates: {e}") from e

    gt_path = out / "ground_truth.json"
    try:
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)
        print(f"[rib_export] Wrote ground_truth.json")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to write ground_truth.json: {e}") from e

    # ------------------------------------------------------------------
    # 7. Write metadata.json
    # ------------------------------------------------------------------
    unique_nodes = set()
    for e in edges:
        unique_nodes.add(e["src_id"])
        unique_nodes.add(e["dst_id"])

    edge_type_counts: dict[str, int] = {}
    for e in edges:
        edge_type_counts[e["edge_type"]] = edge_type_counts.get(e["edge_type"], 0) + 1

    ts_values = [e["timestamp_unix"] for e in edges if e["timestamp_unix"] > 0]

    metadata: dict[str, Any] = {
        "dataset": "RIB v0.1 — Relational Intelligence Benchmark",
        "source": "Arbitrum/Base trap contract surveillance system",
        "export_timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "statistics": {
            "total_edges": len(edges),
            "unique_nodes": len(unique_nodes),
            "edge_type_counts": edge_type_counts,
            "org_001_nodes": len(ORG_001),
            "org_001_edges": len(ORG_001_EDGES),
            "bot_clusters": len(gt["bot_clusters"]),
            "timestamp_range": {
                "min_unix": min(ts_values) if ts_values else 0,
                "max_unix": max(ts_values) if ts_values else 0,
                "min_iso": datetime.utcfromtimestamp(min(ts_values)).isoformat() + "Z" if ts_values else "",
                "max_iso": datetime.utcfromtimestamp(max(ts_values)).isoformat() + "Z" if ts_values else "",
            },
        },
    }

    meta_path = out / "metadata.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[rib_export] Wrote metadata.json")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to write metadata.json: {e}") from e

    # ------------------------------------------------------------------
    # 8. Write address_map.json (LOCAL ONLY — never commit)
    # ------------------------------------------------------------------
    map_path = out / "address_map.json"
    try:
        with open(map_path, "w") as f:
            json.dump(address_map, f, indent=2)
        print(f"[rib_export] Wrote address_map.json ({len(address_map)} mappings) — LOCAL ONLY")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to write address_map.json: {e}") from e

    # ------------------------------------------------------------------
    # 9. Write README.md
    # ------------------------------------------------------------------
    readme_path = out / "README.md"
    readme = """# RIB Dataset — Relational Intelligence Benchmark v0.1

## Overview

Anonymized temporal transaction graph extracted from an Arbitrum/Base trap contract
surveillance system. Contains contract calls, deployments, funding trails, and
confirmed organizational structure for one threat actor group (org_001).

All Ethereum addresses have been replaced with deterministic UUIDs. The mapping
file (`address_map.json`) is **not included** in this distribution — it is
generated locally only.

## Schema

### edges.csv

| Column | Type | Description |
|--------|------|-------------|
| `src_id` | UUID | Anonymized source address |
| `dst_id` | UUID | Anonymized destination address |
| `timestamp_unix` | int | Unix timestamp of the event |
| `weight_usd` | float | USD value (0 if unknown) |
| `edge_type` | str | One of: `contract_call`, `contract_deploy`, `eth_transfer`, `gas_seed`, `usdc_transfer`, `usdc_return` |

### ground_truth.json

```json
{
  "org_001": {
    "nodes": { "<uuid>": {"role": "treasury", "confirmed": true}, ... },
    "edges": [ {"src": "<uuid>", "dst": "<uuid>", "relation": "funds"}, ... ]
  },
  "bot_clusters": {
    "cluster_001": {"member_count": 10, "total_reverts": 9503}
  }
}
```

### metadata.json

Dataset statistics including total edges, unique nodes, edge type distribution,
timestamp range, and export parameters.

## Quick Start

```python
import pandas as pd
import json

edges = pd.read_csv("edges.csv")
print(edges.shape)
print(edges["edge_type"].value_counts())

with open("ground_truth.json") as f:
    gt = json.load(f)

org_nodes = set(gt["org_001"]["nodes"].keys())
org_edges = edges[edges["src_id"].isin(org_nodes) | edges["dst_id"].isin(org_nodes)]
print(f"Edges involving org_001: {len(org_edges)}")
```

## License

CC BY-NC-SA 4.0

## Notes

- `address_map.json` is NOT included in public distributions (private/local only)
- All UUIDs are deterministic: same seed + same database = identical output
- Timestamps are Unix epoch seconds
"""
    try:
        with open(readme_path, "w") as f:
            f.write(readme)
        print(f"[rib_export] Wrote README.md")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to write README.md: {e}") from e

    conn.close()
    print(f"\n[rib_export] Export complete -> {out.resolve()}")
    return out


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "surveillance/data/surveillance.db"
    export_anonymized_dataset(db)
