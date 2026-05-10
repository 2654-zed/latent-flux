"""Audit Layer 3's malicious-flagged addresses against Blockscout's metadata service
(which surfaces Open Labels Initiative tags). Any address we have flagged that ALSO
carries a public institutional tag (Circle, Coinbase, Aave, Uniswap, etc.) is a
candidate mislabel.

Output: reports/blockscout_tag_audit_YYYY-MM-DD.csv with one row per audited address,
including any tags found and a severity assessment.

Methodology context: this script exists because of Correction #20 (bb50 / Circle
deployer mislabeled as Pristine Solo Operator trap class). The detection pipeline does
not consult OLI tags at ingest; the registry-based discount is unbuilt (CLAUDE.md
Priority #2). This audit is a one-shot mitigation pending that work.

Usage:
    python -m scripts.blockscout_tag_audit                # write CSV
    python -m scripts.blockscout_tag_audit --print-hits   # only show hits to stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Iterable

DB_PATH = Path(r"C:\Users\jason\Desktop\ai lang\surveillance\data\surveillance.db")
META_URL = "https://metadata.services.blockscout.com/api/v1/metadata"
CHAIN_ID = "1"  # mainnet — institutional labels live here
BATCH_SIZE = 50
RATE_SLEEP_S = 0.4  # polite pacing between batches

# Sources of "we flagged this as malicious." Each source contributes (address,
# source_label, classification_string).
SOURCE_QUERY = """
SELECT LOWER(address) AS address,
       'watchlist:' || priority AS source,
       COALESCE(entity_name, '') || ' | ' || COALESCE(watch_reason, '') AS classification
FROM watchlist
WHERE active = 1

UNION

SELECT LOWER(address) AS address,
       'entity_classification' AS source,
       category || ' / ' || subtype AS classification
FROM entity_classification
WHERE category IN ('CRIMINAL')
   OR subtype IN (
       'known_attacker','trap_contract','trap_deployer','trap_inventory',
       'infrastructure_parasite','mev_factory','bot_operator','rd_bot',
       'org_laundry','mixer','org_cashout','org_001_shadow_cex_exit',
       'org_001_shadow_lp_staging'
   )

UNION

SELECT LOWER(funder_address) AS address,
       'infra_operator_candidate' AS source,
       'top12_funder | deployers=' || deployer_count || ' contracts=' || contract_count AS classification
FROM infrastructure_operator_candidates
"""


def gather_addresses(conn: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    """Returns {address: [(source, classification), ...]} so we keep multi-source provenance."""
    out: dict[str, list[tuple[str, str]]] = {}
    for addr, source, classification in conn.execute(SOURCE_QUERY):
        out.setdefault(addr, []).append((source, classification))
    return out


def fetch_metadata_batch(addresses: list[str]) -> dict[str, dict]:
    """Batch-query Blockscout metadata service. Returns map of lowercased address -> metadata dict."""
    if not addresses:
        return {}
    params = {
        "addresses": ",".join(addresses),
        "chainId": CHAIN_ID,
        "tagsLimit": "20",
    }
    url = META_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "L3-tag-audit/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"  retry {attempt+1} after error: {e}", file=sys.stderr)
            time.sleep(2)
    raw = payload.get("addresses") or {}
    # The service echoes addresses with EIP-55 mixed case; normalize to lower.
    return {k.lower(): v for k, v in raw.items()}


def severity(tags: list[dict]) -> str:
    """Heuristic severity for any tag that's a likely institutional/legitimate signal."""
    if not tags:
        return "none"
    names = [(t.get("slug") or "") + "|" + (t.get("name") or "") for t in tags]
    blob = " ".join(names).lower()
    if any(k in blob for k in [
        "circle", "coinbase", "binance", "kraken", "okx", "bybit", "gate", "kucoin",
        "uniswap", "aave", "compound", "lido", "maker", "curve", "balancer",
        "1inch", "paraswap", "0x-protocol", "cow-protocol",
        "layerzero", "wormhole", "axelar", "synapse", "stargate", "across",
        "ens", "eas", "safe", "argent", "rabby",
        "centre", "tether", "paypal", "pyusd",
        "infrastructure", "official", "cex", "exchange",
    ]):
        return "HIGH"
    if any(k in blob for k in ["scam", "phishing", "exploit", "drain", "hack", "stolen"]):
        return "self-confirming"  # tag agrees with our flag — not a mislabel
    return "LOW"  # tagged but not a known-institution signal


def run_audit(args: argparse.Namespace) -> int:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    address_map = gather_addresses(conn)
    addresses = sorted(address_map.keys())
    print(f"Auditing {len(addresses)} unique addresses across "
          f"{sum(len(v) for v in address_map.values())} (address, source) pairs", file=sys.stderr)

    # Batch the metadata calls
    results: dict[str, dict] = {}
    for i in range(0, len(addresses), BATCH_SIZE):
        batch = addresses[i : i + BATCH_SIZE]
        print(f"  batch {i // BATCH_SIZE + 1}: addresses {i+1}-{i+len(batch)}", file=sys.stderr)
        batch_results = fetch_metadata_batch(batch)
        results.update(batch_results)
        time.sleep(RATE_SLEEP_S)

    # Build report rows
    out_rows: list[dict] = []
    hits: list[dict] = []
    for addr in addresses:
        meta = results.get(addr) or {}
        tags = meta.get("tags") or []
        sev = severity(tags)
        for source, classification in address_map[addr]:
            row = {
                "address": addr,
                "source": source,
                "current_classification": classification,
                "tag_count": len(tags),
                "severity": sev,
                "tag_names": "; ".join((t.get("name") or "?") for t in tags),
                "tag_slugs": "; ".join((t.get("slug") or "?") for t in tags),
                "main_entities": "; ".join(
                    json.loads(t["meta"]).get("main_entity", "")
                    if isinstance(t.get("meta"), str) else (t.get("meta") or {}).get("main_entity", "")
                    for t in tags if t.get("meta")
                ),
            }
            out_rows.append(row)
            if sev in ("HIGH", "LOW"):
                hits.append(row)

    # Output
    today = date.today().isoformat()
    out_path = Path(__file__).resolve().parents[1] / "reports" / f"blockscout_tag_audit_{today}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n_hi = sum(1 for r in hits if r["severity"] == "HIGH")
    n_lo = sum(1 for r in hits if r["severity"] == "LOW")
    print(f"\nWrote {out_path}", file=sys.stderr)
    print(f"  total rows: {len(out_rows)}", file=sys.stderr)
    print(f"  HIGH-severity mislabel candidates: {n_hi}", file=sys.stderr)
    print(f"  LOW-severity (tagged but unknown class): {n_lo}", file=sys.stderr)

    if args.print_hits or hits:
        print("\n=== AUDIT HITS ===")
        for r in hits:
            print(f"  [{r['severity']}] {r['address']} | {r['source']:30s} | "
                  f"flagged_as: {r['current_classification'][:80]}")
            print(f"      tags: {r['tag_names']}")
            if r["main_entities"].strip("; "):
                print(f"      entities: {r['main_entities']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--print-hits", action="store_true",
                    help="Only print hits to stdout (still writes CSV).")
    args = ap.parse_args()
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
