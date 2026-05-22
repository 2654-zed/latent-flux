"""Phase A — bulk Blockscout enrichment on the confirmed-tier corpus.

Per reports/confirmed_tier_audit_plan.md Phase A:

  Run Tiers 1+2+3 in parallel:
    - Blockscout get_address_info on every confirmed contract + every
      distinct deployer
    - Verified-source-code subset cross-reference
    - Token metadata for ERC-20 subset (holders, market cap)

  Output: confirmed_tier_audit_<date>.csv with one row per contract:
    contract_address, chain, current_tier, current_reason,
    blockscout_verified, token_name, token_symbol, holders_count,
    market_cap_usd, primary_blockscout_tag, deployer_blockscout_tag,
    deployer_mainnet_first_tx, coingecko_url, preliminary_verdict

This script:
  1. Loads the confirmed-tier population from the local DB.
  2. Probes Blockscout v2 REST API for each contract (and deployer).
  3. Caches results in an audit_blockscout_cache table so the script is
     resumable / idempotent.
  4. Applies the audit-plan classifier rules to label each row
     LIKELY_FP / LIKELY_TP / NEEDS_REVIEW.
  5. Writes the CSV.

Resume behavior:
  - Rerunning with the same DB skips contracts already cached.
  - Use --force-refresh to re-probe everything.

Rate limiting: 5 requests/sec across all chains. Blockscout is generally
permissive but we don't want to be a bad citizen.

CLI:
    python scripts/phase_a_blockscout_enrich.py --probe-contracts
    python scripts/phase_a_blockscout_enrich.py --probe-deployers
    python scripts/phase_a_blockscout_enrich.py --build-csv
    python scripts/phase_a_blockscout_enrich.py --all   # all three phases
"""
from __future__ import annotations
import argparse
import csv
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DB_PATH = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
CSV_PATH = OUT_DIR / f"confirmed_tier_audit_{DATE}.csv"

BLOCKSCOUT_BASE = {
    "base":     "https://base.blockscout.com/api/v2",
    "arbitrum": "https://arbitrum.blockscout.com/api/v2",
    "optimism": "https://explorer.optimism.io/api/v2",
}

RATE_LIMIT_SLEEP = 0.25  # 4 req/s
FETCH_TIMEOUT = 30.0  # bumped from 15s — Arbitrum + Optimism are slower
MAX_RETRIES = 2


def ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_blockscout_cache (
            address TEXT NOT NULL,
            chain TEXT NOT NULL,
            kind TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_json TEXT,
            error_status INTEGER,
            PRIMARY KEY (address, chain)
        )
    """)
    conn.commit()


def fetch_address(chain: str, address: str, timeout: float = FETCH_TIMEOUT) -> tuple[dict | None, int | None]:
    """Hit Blockscout's /api/v2/addresses/{address} endpoint.
    Retries up to MAX_RETRIES times on network/timeout errors.
    Returns (json_data, http_status) on success.
    Returns (None, status_code) on HTTP error (404 = not found).
    Returns (None, -1) on persistent network error.
    """
    base = BLOCKSCOUT_BASE.get(chain)
    if not base:
        return (None, -2)
    url = f"{base}/addresses/{address}"
    headers = {"Accept": "application/json",
               "User-Agent": "Mozilla/5.0 (Layer3-Audit/1.0)"}
    for attempt in range(MAX_RETRIES + 1):
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data, resp.status)
        except HTTPError as e:
            return (None, e.code)  # don't retry HTTP errors
        except (URLError, TimeoutError, OSError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * (attempt + 1))
                continue
            sys.stderr.write(f"  net err on {chain}/{address[:14]}: {e}\n")
            return (None, -1)


def probe_set(conn: sqlite3.Connection, addresses: list[tuple[str, str]], kind: str,
              force_refresh: bool = False, sleep_s: float = RATE_LIMIT_SLEEP) -> dict:
    """Probe a list of (address, chain) pairs. Returns counts."""
    ensure_cache_table(conn)
    counts = {"already_cached": 0, "fetched_ok": 0, "fetched_404": 0,
              "fetched_other_err": 0, "fetched_net_err": 0, "total": len(addresses)}
    for i, (addr, chain) in enumerate(addresses):
        if not force_refresh:
            existing = conn.execute(
                "SELECT 1 FROM audit_blockscout_cache WHERE address=? AND chain=?",
                (addr, chain)
            ).fetchone()
            if existing:
                counts["already_cached"] += 1
                continue
        data, status = fetch_address(chain, addr)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        conn.execute(
            "INSERT OR REPLACE INTO audit_blockscout_cache "
            "(address, chain, kind, fetched_at, raw_json, error_status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (addr, chain, kind, now,
             json.dumps(data) if data else None,
             status if status and status != 200 else None)
        )
        if data is not None:
            counts["fetched_ok"] += 1
        elif status == 404:
            counts["fetched_404"] += 1
        elif status == -1:
            counts["fetched_net_err"] += 1
        else:
            counts["fetched_other_err"] += 1
        if (i + 1) % 50 == 0:
            conn.commit()
            sys.stderr.write(f"  [{kind}] {i+1}/{len(addresses)}  ok={counts['fetched_ok']}  "
                             f"404={counts['fetched_404']}  err={counts['fetched_other_err']+counts['fetched_net_err']}  "
                             f"cached={counts['already_cached']}\n")
        time.sleep(sleep_s)
    conn.commit()
    return counts


def load_confirmed_population(conn: sqlite3.Connection) -> list[dict]:
    """Returns list of dicts: contract_address, chain, current_reason,
    detection_method, deployer_address."""
    rows = conn.execute("""
        SELECT contract_address, chain, confidence_reason, detection_method, deployer_address
        FROM contracts
        WHERE confidence_tier='confirmed'
    """).fetchall()
    return [{"contract_address": r[0], "chain": r[1], "current_reason": r[2],
             "detection_method": r[3], "deployer_address": r[4]} for r in rows]


def extract_blockscout_fields(raw_json: str | None) -> dict:
    """Extract the fields we care about from a cached Blockscout address response."""
    if not raw_json:
        return {}
    try:
        d = json.loads(raw_json)
    except Exception:
        return {}
    out = {
        "is_contract": d.get("is_contract"),
        "is_verified": d.get("is_verified"),
        "name": d.get("name") or "",
        "implementation_name": d.get("implementation_name") or "",
    }
    # Public tags
    tags = d.get("public_tags") or []
    if isinstance(tags, list):
        out["public_tags"] = "|".join(
            t.get("display_name", "") if isinstance(t, dict) else str(t) for t in tags
        )
    # Private tags (the OLI-source tags Blockscout serves)
    priv = d.get("private_tags") or []
    if isinstance(priv, list):
        out["private_tags"] = "|".join(
            t.get("display_name", "") if isinstance(t, dict) else str(t) for t in priv
        )
    # Token info
    tok = d.get("token")
    if isinstance(tok, dict):
        out["token_name"] = tok.get("name", "")
        out["token_symbol"] = tok.get("symbol", "")
        out["token_type"] = tok.get("type", "")
        out["holders_count"] = tok.get("holders") or tok.get("holders_count") or ""
        # Some chains return circulating_market_cap, others total_supply
        out["market_cap_usd"] = tok.get("circulating_market_cap") or tok.get("exchange_rate") or ""
        out["coingecko_url"] = tok.get("coingecko_url") if "coingecko_url" in tok else ""
    return out


def classify(row: dict) -> str:
    """Apply the audit-plan classifier rules:
       LIKELY_FP / LIKELY_TP / NEEDS_REVIEW.
    Rules order matters — first match wins."""
    # Strong-evidence retraction signals
    if row.get("holders_count"):
        try:
            if int(str(row["holders_count"]).replace(",", "")) >= 100:
                return "LIKELY_FP"
        except (ValueError, TypeError):
            pass
    if row.get("is_verified") == True and row.get("token_type"):
        # Verified token with type info = likely legitimate token launch
        return "LIKELY_FP"

    # Public Blockscout tags pointing to known entities
    tags = (row.get("public_tags") or "") + "|" + (row.get("private_tags") or "")
    tags_lower = tags.lower()
    for term in ["animoca", "openzeppelin", "uniswap", "coingecko",
                 "compound", "aave", "balancer", "0x protocol",
                 "circle", "binance", "okx", "mexc", "bybit", "relay",
                 "orbiter"]:
        if term in tags_lower:
            return "LIKELY_FP"

    # Deployer institutional signal
    deployer_tags = ((row.get("deployer_public_tags") or "")
                     + "|" + (row.get("deployer_private_tags") or "")).lower()
    for term in ["animoca", "openzeppelin", "uniswap", "circle", "binance",
                 "okx", "mexc", "bybit", "deployer", "official"]:
        if term in deployer_tags:
            return "LIKELY_FP"

    # Strong-evidence retention signals
    reason = (row.get("current_reason") or "").lower()
    # Recidivism + multi-victim is strong TP
    if "behavioral confirmation:" in reason and "vic" in reason and "1 vic" not in reason:
        return "LIKELY_TP"

    # Weak evidence: behavioral-only with single bot + no Blockscout signal
    return "NEEDS_REVIEW"


def build_csv(conn: sqlite3.Connection) -> dict:
    print("Building CSV...")
    pop = load_confirmed_population(conn)
    print(f"  {len(pop)} confirmed contracts to write")

    # Pre-fetch deployer cache
    deployer_cache = {}
    for r in conn.execute(
        "SELECT address || '|' || chain, raw_json FROM audit_blockscout_cache WHERE kind='deployer'"
    ):
        deployer_cache[r[0]] = r[1]

    # Pre-fetch contract cache
    contract_cache = {}
    for r in conn.execute(
        "SELECT address || '|' || chain, raw_json FROM audit_blockscout_cache WHERE kind='contract'"
    ):
        contract_cache[r[0]] = r[1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "contract_address", "chain", "current_reason", "detection_method", "deployer_address",
        "is_contract", "is_verified", "contract_name", "implementation_name",
        "public_tags", "private_tags",
        "token_name", "token_symbol", "token_type", "holders_count", "market_cap_usd", "coingecko_url",
        "deployer_public_tags", "deployer_private_tags",
        "preliminary_verdict",
    ]
    counts = {"LIKELY_FP": 0, "LIKELY_TP": 0, "NEEDS_REVIEW": 0}

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for p in pop:
            key = f"{p['contract_address']}|{p['chain']}"
            c_fields = extract_blockscout_fields(contract_cache.get(key))
            d_fields = extract_blockscout_fields(
                deployer_cache.get(f"{p['deployer_address']}|{p['chain']}") if p['deployer_address'] else None
            )
            row = {
                "contract_address": p["contract_address"],
                "chain": p["chain"],
                "current_reason": (p["current_reason"] or "")[:120],
                "detection_method": p["detection_method"] or "",
                "deployer_address": p["deployer_address"] or "",
                "is_contract": c_fields.get("is_contract"),
                "is_verified": c_fields.get("is_verified"),
                "contract_name": c_fields.get("name", ""),
                "implementation_name": c_fields.get("implementation_name", ""),
                "public_tags": c_fields.get("public_tags", ""),
                "private_tags": c_fields.get("private_tags", ""),
                "token_name": c_fields.get("token_name", ""),
                "token_symbol": c_fields.get("token_symbol", ""),
                "token_type": c_fields.get("token_type", ""),
                "holders_count": c_fields.get("holders_count", ""),
                "market_cap_usd": c_fields.get("market_cap_usd", ""),
                "coingecko_url": c_fields.get("coingecko_url", ""),
                "deployer_public_tags": d_fields.get("public_tags", ""),
                "deployer_private_tags": d_fields.get("private_tags", ""),
            }
            row["preliminary_verdict"] = classify(row)
            counts[row["preliminary_verdict"]] += 1
            w.writerow(row)
    print(f"  Wrote {CSV_PATH}")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--probe-contracts", action="store_true")
    ap.add_argument("--probe-deployers", action="store_true")
    ap.add_argument("--build-csv", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force-refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="probe at most N addresses (0 = all)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_cache_table(conn)

    do_contracts = args.probe_contracts or args.all
    do_deployers = args.probe_deployers or args.all
    do_csv = args.build_csv or args.all

    if do_contracts:
        pop = load_confirmed_population(conn)
        pairs = [(p["contract_address"], p["chain"]) for p in pop]
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"Probing {len(pairs)} contracts...")
        c = probe_set(conn, pairs, "contract", args.force_refresh)
        print(f"  Done. {c}")

    if do_deployers:
        pop = load_confirmed_population(conn)
        # Distinct (deployer, chain) pairs
        seen = set()
        pairs = []
        for p in pop:
            if not p["deployer_address"]:
                continue
            key = (p["deployer_address"], p["chain"])
            if key not in seen:
                seen.add(key)
                pairs.append(key)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"Probing {len(pairs)} distinct deployers...")
        c = probe_set(conn, pairs, "deployer", args.force_refresh)
        print(f"  Done. {c}")

    if do_csv:
        counts = build_csv(conn)
        print()
        print(f"  Verdict counts: {counts}")
        total = sum(counts.values())
        for k, v in counts.items():
            print(f"    {k}: {v} ({100*v/max(total,1):.1f}%)")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
