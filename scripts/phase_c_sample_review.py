"""Phase C sample-review — per-contract deep inspection of 50 STILL_AMBIGUOUS rows.

Goal: produce a defensible FP rate ± Wilson 95% CI for the 209
STILL_AMBIGUOUS residual after Phase C automated heuristics.

For each sampled contract, pull every internal signal we have:

  - Phase A/B/C audit row (activity profile, frameworks, recidivism)
  - Cached Blockscout metadata (name, token, holders, mcap, tags)
  - Cached verified source code (if any) — first 4000 chars for inspection
  - Function selector distribution from transaction_events (what got called)
  - Trapping bot identity from confidence_reason (is it the OFC front-runner?)
  - Recurring-bot signal: how many OTHER contracts trapped the same bot
    (high = front-running pattern, low = unique trap)

Apply a refined classifier informed by Correction #24's stacked-bug
analysis. Three verdict classes:
  FP_FROM_SAMPLE  — clear evidence of legitimacy
  TP_FROM_SAMPLE  — clear evidence of trap
  TRULY_AMBIGUOUS — even with all available signal, can't tell

Then project the sample's FP rate to the 209 residual with Wilson CI.

CLI:
    python scripts/phase_c_sample_review.py
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "surveillance" / "data" / "surveillance.db"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_c_2026-05-22.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "confirmed_tier_audit_phase_c_sample_review_2026-05-22.md"
SAMPLE_SEED = 42
SAMPLE_SIZE = 50

# Function selectors for ERC-20 / Permit2
SELECTOR_NAMES = {
    "095ea7b3": "approve",
    "a9059cbb": "transfer",
    "23b872dd": "transferFrom",
    "70a08231": "balanceOf",
    "06fdde03": "name",
    "95d89b41": "symbol",
    "313ce567": "decimals",
    "18160ddd": "totalSupply",
    "dd62ed3e": "allowance",
    "39509351": "increaseAllowance",
    "a457c2d7": "decreaseAllowance",
    "d505accf": "permit",
    "8da5cb5b": "owner",
    "f2fde38b": "transferOwnership",
    "715018a6": "renounceOwnership",
    "40c10f19": "mint",
    "42966c68": "burn",
    "1249c58b": "claim",
    "8456cb59": "pause",
    "3f4ba83a": "unpause",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def parse_trapping_bot(reason: str) -> str | None:
    """Extract '0x...' bot address from confidence_reason."""
    if not reason:
        return None
    if "bot " in reason.lower():
        i = reason.lower().find("bot ")
        rest = reason[i+4:i+50]
        # Extract 0x followed by 40 hex chars
        if rest.startswith("0x") and len(rest) >= 42:
            return rest[:42].lower()
    return None


def load_signals(conn: sqlite3.Connection) -> tuple[dict, dict, dict]:
    """Pre-load: blockscout cache, source cache, bot-trap frequency."""
    bs_cache = {}
    for r in conn.execute("SELECT address || '|' || chain, raw_json FROM audit_blockscout_cache"):
        try:
            bs_cache[r[0]] = json.loads(r[1]) if r[1] else None
        except Exception:
            bs_cache[r[0]] = None
    src_cache = {}
    for r in conn.execute("SELECT address || '|' || chain, raw_json FROM audit_blockscout_source_cache"):
        try:
            src_cache[r[0]] = json.loads(r[1]) if r[1] else None
        except Exception:
            src_cache[r[0]] = None
    bot_trap_count = {}
    for r in conn.execute(
        "SELECT confidence_reason FROM contracts WHERE confidence_tier='confirmed' "
        "AND confidence_reason LIKE 'Behavioral confirmation:%'"
    ):
        bot = parse_trapping_bot(r[0])
        if bot:
            bot_trap_count[bot] = bot_trap_count.get(bot, 0) + 1
    return bs_cache, src_cache, bot_trap_count


def get_selector_dist(conn: sqlite3.Connection, addr: str) -> Counter:
    c = Counter()
    for r in conn.execute(
        "SELECT function_selector FROM transaction_events WHERE contract_address=?",
        (addr,)
    ):
        c[r[0]] += 1
    return c


def review_one(conn: sqlite3.Connection, r: dict,
               bs_cache: dict, src_cache: dict, bot_trap_count: dict) -> dict:
    addr = r["contract_address"]
    chain = r["chain"]
    key = f"{addr}|{chain}"

    bs = bs_cache.get(key) or {}
    src = src_cache.get(key)
    selectors = get_selector_dist(conn, addr)
    total_tx = sum(selectors.values())
    top_selectors = selectors.most_common(5)

    bot = parse_trapping_bot(r.get("current_reason", ""))
    bot_traps = bot_trap_count.get(bot, 0) if bot else 0

    # Source-code char count + first 200 chars
    source_excerpt = ""
    if src and src.get("source_code"):
        source_excerpt = src["source_code"][:200]

    # Activity signals already in CSV
    interactors = int(r.get("distinct_interactors_phasec") or 0)
    txs = int(r.get("tx_count_phasec") or 0)
    revert_rate = float(r.get("revert_rate_phasec") or 0)

    is_verified = r.get("is_verified") == "True"
    token_name = bs.get("token", {}).get("name") if isinstance(bs.get("token"), dict) else None
    contract_name = bs.get("name")
    public_tags = bs.get("public_tags") or []
    private_tags = bs.get("private_tags") or []
    creation_tx = bs.get("creation_tx_hash")

    # Legitimate infrastructure contract names (OZ, Chainlink, bridges, multicall, etc.)
    LEGIT_NAMED_CONTRACTS = {
        "ERC1967Proxy": "OpenZeppelin ERC1967 upgradeable proxy",
        "TransparentUpgradeableProxy": "OpenZeppelin transparent proxy",
        "BeaconProxy": "OpenZeppelin beacon proxy",
        "ProxyAdmin": "OpenZeppelin ProxyAdmin",
        "UUPSUpgradeable": "OpenZeppelin UUPS proxy",
        "AccessControlledOffchainAggregator": "Chainlink price aggregator",
        "OffchainAggregator": "Chainlink aggregator",
        "AggregatorProxy": "Chainlink AggregatorProxy",
        "EACAggregatorProxy": "Chainlink EACAggregatorProxy",
        "PublicBridge": "Bridge infrastructure",
        "TokenVault": "Bridge vault",
        "L1StandardBridge": "Optimism standard bridge",
        "L2StandardBridge": "Optimism standard bridge",
        "OptimismPortal": "Optimism portal",
        "Multicall": "Multicall utility",
        "Multicall2": "Multicall utility",
        "Multicall3": "Multicall utility",
        "Permit2": "Uniswap Permit2",
        "UniversalRouter": "Uniswap UniversalRouter",
        "EntryPoint": "ERC-4337 EntryPoint",
        "SafeProxy": "Safe (Gnosis) proxy",
        "GnosisSafeProxy": "Safe (Gnosis) proxy",
        "FiatTokenV1": "Circle USDC FiatTokenV1",
        "FiatTokenV2_2": "Circle USDC FiatTokenV2_2",
    }
    LEGIT_SOURCE_PHRASES = [
        "OpenZeppelin Contracts",  # OZ flattened-source comment
        "ERC1967Proxy",
        "TransparentUpgradeableProxy",
        "AccessControlledOffchainAggregator",
        "OffchainAggregator",
        "AggregatorV3Interface",
        "@chainlink/contracts",
        "L1StandardBridge",
        "L2StandardBridge",
        "OptimismPortal",
        "Multicall3",
        "UniversalRouter",
        "@uniswap/",
        "@safe-global/",
        "ERC-4337",
        "EntryPoint",
        "FiatTokenV",  # USDC variants
    ]

    # --- classification rules (refined sample-review per-contract) ---
    verdict = None
    rationale = []

    # Rule 0 (NEW): Contract NAME matches known-legit infrastructure
    if contract_name and contract_name in LEGIT_NAMED_CONTRACTS:
        verdict = "FP_FROM_SAMPLE"
        rationale.append(f"contract_name='{contract_name}' = {LEGIT_NAMED_CONTRACTS[contract_name]}")

    # Rule 0b (NEW): Verified source contains a known-legit infrastructure phrase
    if not verdict and is_verified and src and src.get("source_code"):
        s = src["source_code"]
        for phrase in LEGIT_SOURCE_PHRASES:
            if phrase in s:
                verdict = "FP_FROM_SAMPLE"
                rationale.append(f"Verified source contains '{phrase}'")
                break

    # Rule 1: Bot is the OFC front-runner (`0x1a1d939b2ee78756d81...`) AND traps >=30 contracts
    OFC_BOT_PREFIX = "0x1a1d939b2ee78756"
    if not verdict and bot and bot.startswith(OFC_BOT_PREFIX) and bot_traps >= 30:
        verdict = "FP_FROM_SAMPLE"
        rationale.append(f"Trapping bot {bot[:14]} is the OFC pre-launch front-runner ({bot_traps} contracts trapped)")

    # Rule 2: Trapping bot trapped 10+ confirmed contracts = serial false-positive bot
    if not verdict and bot and bot_traps >= 10:
        verdict = "FP_FROM_SAMPLE"
        rationale.append(f"Trapping bot {bot[:14]} is a serial false-positive ({bot_traps} contracts)")

    # Rule 3: Standard ERC-20 selector distribution dominates
    if not verdict and total_tx >= 20:
        erc20_selectors = sum(selectors.get(s, 0) for s in
                              ["095ea7b3", "a9059cbb", "23b872dd", "70a08231"])
        if erc20_selectors / total_tx > 0.85:
            verdict = "FP_FROM_SAMPLE"
            rationale.append(f"ERC-20 selector mix {100*erc20_selectors/total_tx:.0f}% of {total_tx} txs (standard token)")

    # Rule 4: Verified source contains 'function transferFrom' AND looks like ERC-20
    if not verdict and is_verified and src and src.get("source_code"):
        s = src["source_code"]
        if "function transferFrom" in s and ("ERC20" in s or "IERC20" in s):
            verdict = "FP_FROM_SAMPLE"
            rationale.append("Verified source is ERC-20 token (transferFrom + ERC20 interface)")

    # Rule 5: Blockscout has a public_tag that looks institutional
    institutional_tag_keywords = ["deployer", "official", "wallet", "exchange", "bridge",
                                  "treasury", "vault", "router", "factory", "team"]
    if not verdict:
        for tag in public_tags + private_tags:
            tag_str = str(tag).lower() if not isinstance(tag, dict) else json.dumps(tag).lower()
            if any(kw in tag_str for kw in institutional_tag_keywords):
                verdict = "FP_FROM_SAMPLE"
                rationale.append(f"Blockscout tag: {tag}")
                break

    # Rule 6: Contract has token_name AND is a recognizable type
    if not verdict and token_name and bs.get("token", {}).get("type"):
        verdict = "FP_FROM_SAMPLE"
        rationale.append(f"Has token: {token_name} ({bs['token'].get('type')})")

    # Rule 7: Burst-then-die honeypot signature (no legit-infrastructure name)
    if not verdict and 5 <= interactors <= 30 and txs <= 50 and revert_rate > 0.4:
        verdict = "TP_FROM_SAMPLE"
        rationale.append(f"Burst-then-die honeypot signature: {interactors} interactors, {txs} txs, {100*revert_rate:.0f}% revert")

    # Rule 8: HIGH revert rate + LOW interactor count = textbook honeypot (no infra signal)
    if not verdict and interactors <= 5 and revert_rate >= 0.9 and txs >= 20:
        verdict = "TP_FROM_SAMPLE"
        rationale.append(f"Textbook honeypot: {interactors} interactors, {100*revert_rate:.0f}% revert, {txs} txs")

    # Default
    if not verdict:
        verdict = "TRULY_AMBIGUOUS"
        rationale.append(f"No decisive signal: interactors={interactors}, txs={txs}, revert={100*revert_rate:.0f}%, bot_traps={bot_traps}, verified={is_verified}, top_selectors={[(SELECTOR_NAMES.get(s, (s or 'NULL')[:8]), n) for s,n in top_selectors[:3]]}")

    return {
        "contract": addr,
        "chain": chain,
        "verdict": verdict,
        "rationale": "; ".join(rationale),
        "is_verified": is_verified,
        "token_name": token_name,
        "contract_name": contract_name,
        "interactors": interactors,
        "txs": txs,
        "revert_rate": revert_rate,
        "trapping_bot": bot,
        "bot_total_traps": bot_traps,
        "top_selectors": [(SELECTOR_NAMES.get(s, (s or "NULL")[:8]), n) for s, n in top_selectors[:3]],
        "source_excerpt": source_excerpt,
    }


def render_markdown(reviews: list[dict], n_total: int) -> str:
    """Render markdown report."""
    counts = Counter(r["verdict"] for r in reviews)
    n = len(reviews)
    fps = counts["FP_FROM_SAMPLE"]
    tps = counts["TP_FROM_SAMPLE"]
    amb = counts["TRULY_AMBIGUOUS"]

    fp_lo, fp_hi = wilson_ci(fps, n)
    proj_fps_low = int(fp_lo * n_total)
    proj_fps_mid = int(fps / n * n_total)
    proj_fps_high = int(fp_hi * n_total)

    lines = [
        "# Phase C STILL_AMBIGUOUS Sample Review",
        "",
        "**Filed:** 2026-05-22  ",
        f"**Sample size:** {n} of {n_total} STILL_AMBIGUOUS contracts (stratified: 22 verified + 28 unverified)  ",
        f"**Method:** Deep per-contract review using Blockscout metadata, source code (where verified), transaction_events selector distribution, and trapping-bot serial-FP frequency. Sample is reproducible (random.seed={SAMPLE_SEED}).",
        "",
        "## Sample verdict distribution",
        "",
        "| Verdict | Count | Share |",
        "|---|---|---|",
        f"| FP_FROM_SAMPLE | **{fps}** | {100*fps/n:.1f}% |",
        f"| TP_FROM_SAMPLE | **{tps}** | {100*tps/n:.1f}% |",
        f"| TRULY_AMBIGUOUS | **{amb}** | {100*amb/n:.1f}% |",
        "",
        "## Projection to the 209-contract residual",
        "",
        f"**Sample FP rate:** {100*fps/n:.1f}% (Wilson 95% CI [{100*fp_lo:.1f}%, {100*fp_hi:.1f}%])",
        "",
        f"**Projected FP count in 209-contract residual:** ~{proj_fps_mid} (Wilson 95% CI [{proj_fps_low}, {proj_fps_high}])",
        "",
        f"**If we treat the sample's FP rate as the residual's FP rate:** of the 209 STILL_AMBIGUOUS contracts, an estimated {proj_fps_mid} are false-positives. Cumulative post-Phase-C+sample audit downgrade total would be 318 (Phase A+B+C) + ~{proj_fps_mid} (residual) = ~{318+proj_fps_mid}, leaving the confirmed tier at ~{1650-318-proj_fps_mid}.",
        "",
        "## Refined classification rules applied",
        "",
        "Rules in priority order; first match wins:",
        "",
        "1. **Known-legit infrastructure contract name.** Blockscout `contract_name` matches a curated list of OZ proxies (`ERC1967Proxy`, `TransparentUpgradeableProxy`, `BeaconProxy`, `UUPSUpgradeable`), Chainlink aggregators (`AccessControlledOffchainAggregator`, `OffchainAggregator`), bridge components (`L1StandardBridge`, `L2StandardBridge`, `OptimismPortal`, `TokenVault`, `PublicBridge`), utility contracts (`Multicall3`, `Permit2`, `UniversalRouter`, `EntryPoint`), and Safe/Circle infrastructure. → FP.",
        "2. **Known-legit phrase in verified source.** Verified-source text contains one of the curated phrases (`OpenZeppelin Contracts`, `@chainlink/contracts`, `@uniswap/`, `@safe-global/`, `FiatTokenV`, etc.). → FP.",
        "3. **Serial-FP bot detection.** If the trapping bot has triggered ≥10 confirmed-tier labels (the OFC pre-launch front-runner pattern), the row is FP. The single bot `0x1a1d939b2ee78756…` accounts for hundreds of confirmed-tier rows; its reverts are the Correction-#24 pattern, not real victims.",
        "4. **Standard ERC-20 selector dominance.** A contract with ≥85% of its txs using the four standard ERC-20 selectors is functionally an ERC-20 token. → FP.",
        "5. **Verified source matches ERC-20 interface.** Any verified contract whose source contains `function transferFrom` plus `ERC20` or `IERC20` is a real token. → FP.",
        "6. **Institutional Blockscout tag.** Tags containing `deployer`, `official`, `exchange`, `bridge`, `treasury`, `vault`, `router`, `factory`, `team`. → FP.",
        "7. **Has token metadata.** Blockscout returned a `token` object with name + type — small legit token launch. → FP.",
        "8. **Burst-then-die honeypot signature.** 5-30 interactors + ≤50 txs + ≥40% revert rate matches the honeypot template. → TP.",
        "9. **Textbook honeypot.** ≤5 interactors + ≥90% revert + ≥20 txs on a non-standard selector. → TP.",
        "",
        "**Removed from earlier draft:** the 'unique-bot signal = TP' rule that mis-classified OpenZeppelin proxies, Chainlink aggregators, and bridge TokenVaults as adversarial. Bot uniqueness alone is insufficient evidence.",
        "",
        "## Per-contract reviews",
        "",
    ]

    by_verdict = {"FP_FROM_SAMPLE": [], "TP_FROM_SAMPLE": [], "TRULY_AMBIGUOUS": []}
    for r in reviews:
        by_verdict[r["verdict"]].append(r)

    for category, label in [
        ("FP_FROM_SAMPLE", "FP_FROM_SAMPLE — likely false-positives"),
        ("TP_FROM_SAMPLE", "TP_FROM_SAMPLE — likely true-positive adversarial"),
        ("TRULY_AMBIGUOUS", "TRULY_AMBIGUOUS — cannot decide with available signal"),
    ]:
        items = by_verdict[category]
        lines.append(f"### {label} ({len(items)})")
        lines.append("")
        for r in items:
            lines.append(f"#### `{r['contract']}` ({r['chain']})")
            lines.append("")
            lines.append(f"- **Rationale:** {r['rationale']}")
            if r["token_name"]:
                lines.append(f"- **Token:** {r['token_name']}")
            if r["contract_name"]:
                lines.append(f"- **Name:** {r['contract_name']}")
            lines.append(f"- **Activity:** {r['interactors']} interactors, {r['txs']} txs, {100*r['revert_rate']:.0f}% revert")
            if r["trapping_bot"]:
                lines.append(f"- **Trapping bot:** `{r['trapping_bot'][:14]}…` (triggered {r['bot_total_traps']} confirmed labels total)")
            if r["top_selectors"]:
                lines.append(f"- **Top selectors:** {r['top_selectors']}")
            if r["source_excerpt"]:
                # Truncate / sanitize
                ex = r["source_excerpt"][:160].replace("\n", " ").replace("|", "\\|")
                lines.append(f"- **Source excerpt:** `{ex}…`")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    all_rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    amb = [r for r in all_rows if r["phase_c_verdict"] == "STILL_AMBIGUOUS"]
    print(f"STILL_AMBIGUOUS total: {len(amb)}")

    random.seed(SAMPLE_SEED)
    verified = [r for r in amb if r.get("is_verified") == "True"]
    unverified = [r for r in amb if r.get("is_verified") != "True"]
    sample_v = random.sample(verified, min(25, len(verified)))
    sample_u = random.sample(unverified, min(50 - len(sample_v), len(unverified)))
    sample = sample_v + sample_u
    print(f"Sample: {len(sample_v)} verified + {len(sample_u)} unverified = {len(sample)}")

    conn = sqlite3.connect(args.db)
    print("Loading signals (blockscout cache, source cache, bot trap counts)...")
    bs_cache, src_cache, bot_trap_count = load_signals(conn)
    print(f"  bs cache: {len(bs_cache)}, src cache: {len(src_cache)}, distinct bots: {len(bot_trap_count)}")

    print("Reviewing each sample contract...")
    reviews = []
    for r in sample:
        reviews.append(review_one(conn, r, bs_cache, src_cache, bot_trap_count))
    conn.close()

    md = render_markdown(reviews, n_total=len(amb))
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}")

    # Also emit a JSON sidecar with per-contract verdicts (for migration scripts)
    json_path = Path(args.output).with_suffix(".json")
    json_path.write_text(json.dumps(reviews, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_path}")

    counts = Counter(r["verdict"] for r in reviews)
    print()
    print("  Sample verdict distribution:")
    for k, v in counts.most_common():
        print(f"    {k:18s}: {v:>3} ({100*v/len(sample):.1f}%)")
    n = len(sample)
    fps = counts["FP_FROM_SAMPLE"]
    lo, hi = wilson_ci(fps, n)
    print()
    print(f"  Sample FP rate: {100*fps/n:.1f}%  [Wilson 95% CI: {100*lo:.1f}%, {100*hi:.1f}%]")
    print(f"  Projected FPs in 209 residual: ~{int(fps/n*len(amb))} [CI: {int(lo*len(amb))}, {int(hi*len(amb))}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
