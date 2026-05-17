"""Prediction verifiability scorer — answers Q-004.

STATUS: SKELETON. The substrate score logic is sketched but the
entity-extraction is rule-based and brittle. Full build replaces the
regex extraction with a structured prediction template that
NEXT_SESSION_PLAN.md authors must use.

For each pre-registered prediction in any memory document, score each
named-entity reference 0-3:
    0 = no entity citation at all
    1 = entity in narrative only (e.g., "the b0b0b690 operator")
    2 = entity present in the queryable corpus
        (deployers, contracts, watchlist, etc.)
    3 = entity with on-chain evidence link
        (tx hash, block number, or specific table-row pointer)

Refuse to load predictions with average score < 2.

Empirical anchor: Phase A 2026-05-15 falsified 3/3 predictions because
they cited entities not in the corpus.

CLI:
    python -m surveillance.sai.prediction_verifiability memory/NEXT_SESSION_PLAN.md
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"
ADDR_PATTERN = re.compile(r"0x[a-fA-F0-9]{40}\b")
SHORT_ADDR_PATTERN = re.compile(r"0x[a-fA-F0-9]{6,40}\b")  # for prefix-style citations


@dataclass
class EntityScore:
    raw: str
    canonical: str | None
    score: int
    rationale: str


def score_entity(conn: sqlite3.Connection, raw: str) -> EntityScore:
    """Score a single named-entity reference.

    Score 3 only achievable if a tx hash / block link is present (not
    detected at the address-level scoring; that's the caller's job).
    """
    raw_clean = raw.strip().lower()
    # If it's a full 40-hex address, check corpus presence
    if ADDR_PATTERN.fullmatch(raw_clean):
        return _score_full_address(conn, raw_clean)
    # Short prefix (e.g., 0xb0b0b6) — degraded match
    if SHORT_ADDR_PATTERN.fullmatch(raw_clean) and len(raw_clean) < 42:
        return _score_prefix(conn, raw_clean)
    # Free text — narrative only, score 1
    return EntityScore(raw, None, 1, "narrative-only reference (no full address)")


def _score_full_address(conn: sqlite3.Connection, addr: str) -> EntityScore:
    """Score a full 0x...40hex address."""
    # Check deployers, contracts, watchlist, oli_labels in order of preference
    for table, col in [
        ("deployers", "deployer_address"),
        ("contracts", "contract_address"),
        ("contracts", "deployer_address"),
        ("watchlist", "address"),
        ("oli_labels", "address"),
        ("approval_watchlist", "drain_caller"),
        ("approval_watchlist", "victim_address"),
    ]:
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = ? LIMIT 1",
                (addr,)
            ).fetchone()[0]
            if n > 0:
                return EntityScore(addr, addr, 2,
                                   f"address present in {table}.{col}")
        except sqlite3.OperationalError:
            continue
    return EntityScore(addr, addr, 1, "full address but not in corpus tables")


def _score_prefix(conn: sqlite3.Connection, prefix: str) -> EntityScore:
    """Score a short prefix reference (e.g., 0xb0b0b6)."""
    # LIKE query against the most-common address columns
    for table, col in [
        ("deployers", "deployer_address"),
        ("contracts", "deployer_address"),
        ("watchlist", "address"),
    ]:
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} LIKE ?",
                (prefix + "%",)
            ).fetchone()[0]
            if n > 0:
                return EntityScore(prefix, None, 2,
                                   f"prefix matches {n} rows in {table}.{col}")
        except sqlite3.OperationalError:
            continue
    return EntityScore(prefix, None, 1,
                       "prefix not matched anywhere in corpus — possible hallucination")


def score_document(path: Path, db_path: Path = DB_PATH) -> dict:
    """Score every address-like entity reference in a markdown document."""
    text = path.read_text(encoding="utf-8")
    full_addrs = set(m.lower() for m in ADDR_PATTERN.findall(text))
    # Prefix mentions: anything matching the short pattern that isn't a full
    prefixes = set(
        m.lower() for m in SHORT_ADDR_PATTERN.findall(text)
        if not ADDR_PATTERN.fullmatch(m)
    )
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    scores: list[EntityScore] = []
    try:
        for a in full_addrs:
            scores.append(score_entity(conn, a))
        for p in prefixes:
            scores.append(score_entity(conn, p))
    finally:
        conn.close()
    avg = sum(s.score for s in scores) / max(len(scores), 1)
    return {
        "path": str(path),
        "n_entities": len(scores),
        "avg_score": round(avg, 2),
        "verdict": "ACCEPT" if avg >= 2.0 else "REJECT_HALLUCINATION_RISK",
        "scores": scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="markdown file to score")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    result = score_document(Path(args.path), Path(args.db))
    print(f"\nVerifiability of {result['path']}:")
    print(f"  Entities: {result['n_entities']}")
    print(f"  Avg score: {result['avg_score']}")
    print(f"  Verdict: {result['verdict']}\n")
    # Show LOW-score entries (the suspicious ones)
    low = [s for s in result["scores"] if s.score < 2]
    if low:
        print(f"  Suspicious entities ({len(low)}):")
        for s in low:
            print(f"    [{s.score}] {s.raw}: {s.rationale}")
    else:
        print("  All entity references are corpus-verified.")
    return 0 if result["verdict"] == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
