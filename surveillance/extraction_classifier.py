"""Extraction-event taxonomy classifier.

Exception-as-rule audit P5 finding: `extraction_events.event_type` is a
free-text TEXT column with no classifier. Categories were hand-assigned at
INSERT time, so a novel event could be silently shoehorned into an existing
bucket without resistance. This module gives the labeling a second opinion
without gating the workflow.

Given a `summary` string (and optionally the `raw_transactions` JSON blob),
`suggest_type` returns a closed-vocabulary label, a signal-count confidence,
and the list of phrases that matched. Callers that accept the suggestion
record it in `extraction_events.event_type_suggestion`; divergence between
documented `event_type` and suggestion is the actionable signal.

Vocabulary is explicit and closed:
  - full_pipeline_cycle
  - infrastructure_parasite
  - oracle_manipulation_lending_exploit
  - oft_adapter_admin_compromise
  - cross_chain_proof_verification_bypass
  - cross_chain_dvn_verification_failure
  - unclassified  (default when no signal matches cleanly)

Adding a new category requires editing _RULES below. That change is
reviewable and leaves git history; the prior free-text regime did not.
"""
import json
import re
from typing import Optional


# Each rule: (event_type, [regex patterns]). Patterns are evaluated against
# the lowercased summary; any match counts 1 toward the type's score.
_RULES: list[tuple[str, list[str]]] = [
    ("full_pipeline_cycle", [
        r"\borg_\d+\b",
        r"\bpipeline\b",
        r"\bcashout\b.*\b(staging|branch|exit)\b",
        r"\bextraction cycle\b",
        r"\blaundry\b",
    ]),
    ("infrastructure_parasite", [
        r"\buniversal router\b",
        r"\binfrastructure parasitism\b",
        r"\bparasit(e|ic|ize)\b",
        r"\b(fake|fraudulent).*\btoken\b",
        r"\b(uniswap|sushi|1inch|aerodrome)\b.*\b(execute|route)\b",
    ]),
    ("oracle_manipulation_lending_exploit", [
        r"\boracle\b.*\b(manipulat|price)\b",
        r"\blending\b",
        r"\bmargin\b",
        r"\bslippage\b",
        r"\b(burrow|aave|compound|morpho|rhea)\b",
        r"\b(collateral|liquidat)\b",
    ]),
    ("oft_adapter_admin_compromise", [
        r"\boft\s*adapter\b",
        r"\badmin\b.*\b(compromis|privileg|takeover)\b",
        r"\b(private key|eoa).*\b(compromis|leak|stolen)\b",
        r"\b(owner|deployer).*\b(compromis|takeover)\b",
    ]),
    ("cross_chain_proof_verification_bypass", [
        r"\bmerkle\s*mountain\s*range\b|\bmmr\b",
        r"\bproof (verification|validation)\s*(bug|bypass|failure)\b",
        r"\bhandler(v\d+)?\b.*\b(post|request)\b",
        r"\bbridge\b.*\b(proof|verification)\b.*\b(bug|bypass)\b",
    ]),
    ("cross_chain_dvn_verification_failure", [
        r"\bdvn\b",
        r"\blayerzero\b",
        r"\bendpoint\s*v?\d*\b",
        r"\b(forged|spoofed).*\b(cross.?chain|message|packet)\b",
        r"\bsrceid\b|\bdsteid\b",
    ]),
]

VOCABULARY = tuple(t for t, _ in _RULES) + ("unclassified",)


def suggest_type(summary: str,
                 raw_transactions: Optional[str] = None
                 ) -> tuple[str, float, dict]:
    """Return (label, confidence_0_1, components).

    confidence is normalized by max-hits-for-the-winning-category. A tie
    at 0 hits returns 'unclassified' with confidence 0. components captures
    every match so callers can see *why* the classifier picked the label.
    """
    text = (summary or "").lower()
    # raw_transactions is JSON — extract any fields that might help
    if raw_transactions:
        try:
            t = json.loads(raw_transactions)
            if isinstance(t, dict):
                text = text + " " + " ".join(str(v).lower()
                                              for v in t.values() if isinstance(v, str))
        except (json.JSONDecodeError, TypeError):
            pass

    hits_per_type: dict[str, list[str]] = {}
    for etype, patterns in _RULES:
        matches = []
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                matches.append(pat)
        if matches:
            hits_per_type[etype] = matches

    if not hits_per_type:
        return ("unclassified", 0.0, {"all_hits": {}})

    # Pick the type with the most hits; tie-break by rule order
    best = max(hits_per_type.items(),
               key=lambda kv: (len(kv[1]),
                               -next(i for i, (t, _) in enumerate(_RULES) if t == kv[0])))
    label = best[0]
    hits = len(best[1])
    # Normalize by max possible hits for that rule
    rule_len = next(len(p) for t, p in _RULES if t == label)
    confidence = round(hits / rule_len, 3) if rule_len else 0.0
    return (label, confidence, {"matched_patterns": best[1], "all_hits": hits_per_type})


def _main():
    """CLI: validate classifier against existing extraction_events rows."""
    import argparse
    import sqlite3
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Validate extraction-type classifier.")
    ap.add_argument("--db", default=str(Path(__file__).parent / "data" / "surveillance.db"))
    ap.add_argument("--apply", action="store_true",
                    help="Backfill event_type_suggestion column for all rows.")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, event_id, event_type, summary, raw_transactions "
                        "FROM extraction_events").fetchall()
    print(f"validating {len(rows)} extraction_events rows")
    print()
    print(f"{'event_id':<20} {'documented':<42} {'suggestion':<42} {'match':<6} conf")
    print("-" * 120)
    agree = 0
    disagreements = []
    for r in rows:
        label, conf, comp = suggest_type(r["summary"], r["raw_transactions"])
        ok = (label == r["event_type"])
        agree += int(ok)
        marker = "OK" if ok else "NO"
        print(f"{r['event_id']:<20} {r['event_type']:<42} {label:<42} {marker:<6} {conf}")
        if not ok:
            disagreements.append((r["event_id"], r["event_type"], label, comp))
    print()
    print(f"agreement: {agree}/{len(rows)} ({100*agree/max(len(rows),1):.0f}%)")
    if disagreements:
        print()
        print("disagreements:")
        for eid, doc, sug, comp in disagreements:
            print(f"  {eid}: documented={doc}  suggested={sug}")
            print(f"    all_hits: {comp.get('all_hits')}")
    if args.apply:
        print()
        print("writing event_type_suggestion...")
        for r in rows:
            label, conf, _ = suggest_type(r["summary"], r["raw_transactions"])
            conn.execute(
                "UPDATE extraction_events SET event_type_suggestion = ?, "
                "event_type_suggestion_confidence = ? WHERE id = ?",
                (label, conf, r["id"]),
            )
        conn.commit()
        print(f"updated {len(rows)} rows")
    conn.close()


if __name__ == "__main__":
    _main()
