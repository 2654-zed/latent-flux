# RIB Dataset — Relational Intelligence Benchmark v0.1

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
