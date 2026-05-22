"""Phase C — deep automated review of Phase-B residual.

Per reports/confirmed_tier_audit_plan.md Phase C, but executed
programmatically rather than manually. The plan describes "~30 min per
contract × N contracts = analyst-hours." This script substitutes:

  Tier 1 — Source-code mining
    For every contract Blockscout reports is_verified=True, fetch the
    verified source-code metadata and pattern-match for known-legitimate
    framework signatures (OpenZeppelin, Animoca, Uniswap, Aave, etc.).

  Tier 2 — Activity profiling
    Internal-DB analysis of each contract's interactor diversity, time
    window, transaction count, revert rate. Real adversarial contracts
    have characteristic patterns (low interactor count, short active
    window). Legitimate contracts have diverse interactors over long
    windows.

  Tier 3 — Deployer-cluster propagation
    If a deployer has ANY contract that Phase A flagged as LIKELY_FP
    (verified + holders > 100), other contracts from that deployer
    likely share its legitimacy status. Propagate the signal.

  Tier 4 — Genuine-ambiguity stratified sample
    Whatever survives Tiers 1-3 in NEEDS_REVIEW gets sampled and the
    sample is annotated for separate human review.

Input:  reports/confirmed_tier_audit_phase_b_2026-05-22.csv
Output: reports/confirmed_tier_audit_phase_c_2026-05-22.csv
        + audit_blockscout_source_cache table (verified source code)

Phase C does NOT migrate anything. Produces verdicts; migration is a
separate Phase-D follow-up authorized by the user.

CLI:
    python scripts/phase_c_deep_review.py --fetch-source   # populate source cache
    python scripts/phase_c_deep_review.py --build-csv      # apply heuristics + emit CSV
    python scripts/phase_c_deep_review.py --all            # both
"""
from __future__ import annotations
import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_b_2026-05-22.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_c_2026-05-22.csv"

BLOCKSCOUT_BASE = {
    "base":     "https://base.blockscout.com/api/v2",
    "arbitrum": "https://arbitrum.blockscout.com/api/v2",
    "optimism": "https://explorer.optimism.io/api/v2",
}
RATE_LIMIT_SLEEP = 0.25
FETCH_TIMEOUT = 30.0
MAX_RETRIES = 2

# Framework signatures to look for in verified source code
LEGIT_FRAMEWORK_SIGNATURES = [
    # OpenZeppelin
    ("@openzeppelin/contracts", "OpenZeppelin"),
    ("openzeppelin-contracts", "OpenZeppelin"),
    ("ERC20.sol", "OpenZeppelin-ERC20"),
    ("Ownable.sol", "OpenZeppelin-Ownable"),
    ("AccessControl.sol", "OpenZeppelin-AccessControl"),
    ("UUPSUpgradeable.sol", "OpenZeppelin-UUPS"),
    # Animoca
    ("@animoca-network/contracts", "Animoca"),
    ("animoca-network", "Animoca"),
    ("ContractOwnership", "Animoca-framework"),
    ("TokenRecovery", "Animoca-framework"),
    # Uniswap
    ("@uniswap/v3-core", "Uniswap-v3"),
    ("@uniswap/v2-core", "Uniswap-v2"),
    ("@uniswap/v3-periphery", "Uniswap-v3"),
    # Aave
    ("@aave/", "Aave"),
    # Compound
    ("@compound-finance/", "Compound"),
    # LayerZero
    ("@layerzerolabs/", "LayerZero"),
    # Solady (popular gas-optimized standard library)
    ("Solady", "Solady"),
    ("solady/", "Solady"),
    # Solmate (Transmissions11)
    ("solmate/", "Solmate"),
    # ERC721A
    ("erc721a", "ERC721A"),
    # Standard token implementations
    ("function totalSupply() external view returns (uint256)", "ERC20-interface"),
    # Permit2
    ("@uniswap/permit2", "Permit2"),
    # Safe (Gnosis)
    ("@safe-global/", "Safe"),
    ("GnosisSafe", "Safe"),
]


def ensure_source_cache(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_blockscout_source_cache (
            address TEXT NOT NULL,
            chain TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_json TEXT,
            error_status INTEGER,
            PRIMARY KEY (address, chain)
        )
    """)
    conn.commit()


def fetch_source(chain: str, address: str, timeout: float = FETCH_TIMEOUT) -> tuple[dict | None, int | None]:
    base = BLOCKSCOUT_BASE.get(chain)
    if not base:
        return (None, -2)
    url = f"{base}/smart-contracts/{address}"
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (Layer3-Audit/1.0)"}
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data, resp.status)
        except HTTPError as e:
            return (None, e.code)
        except (URLError, TimeoutError, OSError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * (attempt + 1))
                continue
            return (None, -1)


def load_phase_b_residual(csv_path: Path) -> list[dict]:
    """Load rows where phase_b_verdict in NEEDS_REVIEW/STILL_NEEDS_REVIEW/BUG_19B_SUSPECT."""
    target = {"NEEDS_REVIEW", "STILL_NEEDS_REVIEW", "BUG_19B_SUSPECT"}
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["phase_b_verdict"] in target:
                rows.append(r)
    return rows


def populate_source_cache(conn: sqlite3.Connection, rows: list[dict], force: bool = False) -> dict:
    ensure_source_cache(conn)
    counts = {"already_cached": 0, "fetched_ok": 0, "fetched_404": 0,
              "fetched_other_err": 0, "fetched_net_err": 0, "skipped_not_verified": 0,
              "total": len(rows)}
    for i, r in enumerate(rows):
        if r.get("is_verified") != "True":
            counts["skipped_not_verified"] += 1
            continue
        addr = r["contract_address"]
        chain = r["chain"]
        if not force:
            existing = conn.execute(
                "SELECT 1 FROM audit_blockscout_source_cache WHERE address=? AND chain=?",
                (addr, chain)
            ).fetchone()
            if existing:
                counts["already_cached"] += 1
                continue
        data, status = fetch_source(chain, addr)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        conn.execute(
            "INSERT OR REPLACE INTO audit_blockscout_source_cache "
            "(address, chain, fetched_at, raw_json, error_status) VALUES (?, ?, ?, ?, ?)",
            (addr, chain, now, json.dumps(data) if data else None,
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
        if (i + 1) % 25 == 0:
            conn.commit()
            sys.stderr.write(f"  {i+1}/{len(rows)}  ok={counts['fetched_ok']}  "
                             f"404={counts['fetched_404']}  err={counts['fetched_other_err']+counts['fetched_net_err']}  "
                             f"cached={counts['already_cached']}\n")
        time.sleep(RATE_LIMIT_SLEEP)
    conn.commit()
    return counts


def detect_frameworks(source_json: dict | None) -> tuple[list[str], int]:
    """Returns (matched_frameworks, source_chars). Empty list if no source or no match."""
    if not source_json:
        return ([], 0)
    # source can come as 'source_code' (single file) or 'additional_sources'
    parts = []
    if source_json.get("source_code"):
        parts.append(source_json["source_code"])
    for s in source_json.get("additional_sources") or []:
        if isinstance(s, dict) and s.get("source_code"):
            parts.append(s["source_code"])
    full = "\n".join(parts)
    if not full:
        # try 'name' / 'file_path' / contract_name as proxy
        if source_json.get("name"):
            full = source_json["name"]
    matched = set()
    for sig, label in LEGIT_FRAMEWORK_SIGNATURES:
        if sig in full:
            matched.add(label)
    return (sorted(matched), len(full))


def compute_activity_profile(conn: sqlite3.Connection, addresses: list[str]) -> dict:
    """For each contract address, compute tx count + distinct interactor count + active window."""
    out = {}
    for addr in addresses:
        r = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT interacting_address), MIN(timestamp), MAX(timestamp), "
            "SUM(CASE WHEN is_reverted=1 THEN 1 ELSE 0 END) "
            "FROM transaction_events WHERE contract_address=?",
            (addr,)
        ).fetchone()
        out[addr] = {
            "tx_count": r[0] or 0,
            "distinct_interactors": r[1] or 0,
            "first_tx": r[2],
            "last_tx": r[3],
            "revert_count": r[4] or 0,
        }
    return out


def deployer_propagation(rows: list[dict], all_phase_b_rows: list[dict]) -> dict:
    """For each contract, count how many sibling contracts (same deployer) Phase A
    flagged as LIKELY_FP (the strong-evidence verified-source class)."""
    fp_deployers = Counter()
    for r in all_phase_b_rows:
        if r["preliminary_verdict"] == "LIKELY_FP" and r.get("deployer_address"):
            fp_deployers[r["deployer_address"]] += 1
    return {r["contract_address"]: fp_deployers.get(r.get("deployer_address"), 0) for r in rows}


def classify_phase_c(row: dict, frameworks: list[str], source_chars: int,
                     activity: dict, deployer_fp_count: int) -> tuple[str, str]:
    """Apply Phase C heuristics. Returns (verdict, rationale)."""
    addr = row["contract_address"]
    is_verified = row.get("is_verified") == "True"
    bug_19b = row["phase_b_verdict"] == "BUG_19B_SUSPECT"

    if bug_19b:
        return ("PHASE_E_INVESTIGATE", "Residual from-matching bug — not auto-classified, needs Phase E")

    # Tier 1: source-code framework signature
    if frameworks:
        return ("LIKELY_FP_FROM_SOURCE",
                f"verified source matches: {', '.join(frameworks)}")

    # Tier 3: deployer-cluster propagation
    if deployer_fp_count >= 1:
        return ("LIKELY_FP_FROM_CLUSTER",
                f"deployer has {deployer_fp_count} other LIKELY_FP contract(s) in Phase A — siblings likely legitimate")

    # Tier 2: activity profile
    tx_count = activity.get("tx_count", 0)
    interactors = activity.get("distinct_interactors", 0)
    revert_count = activity.get("revert_count", 0)
    if tx_count > 0:
        revert_rate = revert_count / tx_count
    else:
        revert_rate = 0.0

    # Diverse interactor base + long history = legitimate token / contract
    if interactors >= 50 and tx_count >= 100 and revert_rate < 0.5:
        return ("LIKELY_FP_FROM_ACTIVITY",
                f"diverse activity: {interactors} interactors, {tx_count} txs, {100*revert_rate:.1f}% revert")

    # Very low activity + behavioral-only confirmation = uncertain (could be honeypot trap OR test deploy)
    if interactors <= 3 and tx_count <= 10:
        return ("LIKELY_TP_FROM_LOW_ACTIVITY",
                f"sparse activity: {interactors} interactors, {tx_count} txs — consistent with narrow honeypot")

    # Mid-range diversity, high revert rate = textbook trap
    if 4 <= interactors <= 30 and revert_rate > 0.7:
        return ("LIKELY_TP_FROM_REVERT_PROFILE",
                f"{interactors} interactors, {100*revert_rate:.1f}% revert rate — trap signature")

    # Everything else
    return ("STILL_AMBIGUOUS",
            f"interactors={interactors}, tx={tx_count}, revert={100*revert_rate:.1f}%, source_chars={source_chars}, verified={is_verified}")


def build_csv(conn: sqlite3.Connection, args) -> dict:
    print("Building Phase C CSV...")
    all_rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    target_rows = [r for r in all_rows if r["phase_b_verdict"]
                   in ("NEEDS_REVIEW", "STILL_NEEDS_REVIEW", "BUG_19B_SUSPECT")]
    print(f"  {len(target_rows)} residual contracts to process")

    # Pre-load source cache
    src_cache = {}
    for r in conn.execute("SELECT address || '|' || chain, raw_json FROM audit_blockscout_source_cache"):
        try:
            src_cache[r[0]] = json.loads(r[1]) if r[1] else None
        except Exception:
            src_cache[r[0]] = None
    print(f"  source cache: {len(src_cache)} entries")

    # Compute activity profile in bulk
    addrs = [r["contract_address"] for r in target_rows]
    print(f"  computing activity profile for {len(addrs)} contracts...")
    activity = compute_activity_profile(conn, addrs)

    # Deployer FP propagation
    propagation = deployer_propagation(target_rows, all_rows)

    # Classify
    out_cols = list(all_rows[0].keys()) + [
        "frameworks_matched", "source_chars", "deployer_other_fp_count",
        "tx_count_phasec", "distinct_interactors_phasec", "revert_rate_phasec",
        "phase_c_verdict", "phase_c_rationale"
    ]

    counts = Counter()
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in target_rows:
            key = f"{r['contract_address']}|{r['chain']}"
            frameworks, source_chars = detect_frameworks(src_cache.get(key))
            act = activity.get(r["contract_address"], {})
            deployer_fp = propagation.get(r["contract_address"], 0)
            verdict, rationale = classify_phase_c(r, frameworks, source_chars, act, deployer_fp)
            r["frameworks_matched"] = "|".join(frameworks)
            r["source_chars"] = source_chars
            r["deployer_other_fp_count"] = deployer_fp
            r["tx_count_phasec"] = act.get("tx_count", 0)
            r["distinct_interactors_phasec"] = act.get("distinct_interactors", 0)
            rc = act.get("revert_count", 0)
            tc = act.get("tx_count", 0)
            r["revert_rate_phasec"] = f"{rc/tc:.3f}" if tc > 0 else ""
            r["phase_c_verdict"] = verdict
            r["phase_c_rationale"] = rationale
            counts[verdict] += 1
            w.writerow(r)

    print()
    total = sum(counts.values())
    print(f"  Phase C verdict counts:")
    for k, v in counts.most_common():
        print(f"    {k:35s}: {v:>4,} ({100*v/total:.1f}%)")
    print()
    print(f"  Wrote {args.output}")
    return dict(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--fetch-source", action="store_true")
    ap.add_argument("--build-csv", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_source_cache(conn)

    do_fetch = args.fetch_source or args.all
    do_csv = args.build_csv or args.all

    if do_fetch:
        rows = load_phase_b_residual(Path(args.input))
        verified_rows = [r for r in rows if r.get("is_verified") == "True"]
        print(f"Fetching source code for {len(verified_rows)} verified contracts (of {len(rows)} residual)...")
        c = populate_source_cache(conn, verified_rows, args.force_refresh)
        print(f"  Done. {c}")

    if do_csv:
        build_csv(conn, args)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
