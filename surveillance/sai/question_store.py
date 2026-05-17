"""Load, rank, and persist memory/questions.yaml.

The question store is the canonical SAI substrate. Every question in the
system has a structured representation here. New questions enter via the
question_generator (Phase 3); existing questions are re-ranked on every
SAI cycle.

Public API:
    load_questions(path=None) -> list[Question]
    rank(questions) -> list[Question]  # sorted descending by priority_score
    save_questions(questions, path) -> None
    summary(questions) -> str

Ranking formula (matches the SAI plan):
    priority_score = predictive_power * 0.30
                   + actionability    * 0.30
                   + failure_reduction * 0.30
                   + uniqueness       * 0.10

CLI:
    python -m surveillance.sai.question_store           # show ranked table
    python -m surveillance.sai.question_store --top 5   # only top 5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("PyYAML required: pip install pyyaml\n")
    raise

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "memory" / "questions.yaml"

WEIGHTS = {
    "predictive_power": 0.30,
    "actionability": 0.30,
    "failure_reduction": 0.30,
    "uniqueness": 0.10,
}

# A score of 0-5 on each dimension; 5 weights sum to 5.0 max (with the
# 0.30/0.30/0.30/0.10 formula). The cap is the natural ranking ceiling.
MAX_SCORE = 5.0


@dataclass
class Question:
    """One question in the SAI substrate.

    Fields mirror the YAML schema. Use `score()` to compute the
    priority weight; do not pre-compute and store (always re-derive
    from the dimensional scores so a weight-formula change doesn't
    require rewriting the YAML).
    """
    id: str
    category: str
    question: str
    status: str
    origin: str
    predictive_power: int
    actionability: int
    uniqueness: int
    failure_reduction: int
    depends_on: dict[str, list[str]] = field(default_factory=dict)
    produces: dict[str, str] = field(default_factory=dict)
    evolves_from: str | None = None
    instantiates_as: list[str] = field(default_factory=list)
    implementation_target: str | None = None
    empirical_anchor: str | None = None
    note: str | None = None

    def priority_score(self) -> float:
        return (
            self.predictive_power * WEIGHTS["predictive_power"]
            + self.actionability * WEIGHTS["actionability"]
            + self.failure_reduction * WEIGHTS["failure_reduction"]
            + self.uniqueness * WEIGHTS["uniqueness"]
        )

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> "Question":
        score = raw.get("score") or {}
        return cls(
            id=raw["id"],
            category=raw.get("category", "unspecified"),
            question=raw["question"],
            status=raw.get("status", "active"),
            origin=raw.get("origin", "unknown"),
            predictive_power=int(score.get("predictive_power", 0)),
            actionability=int(score.get("actionability", 0)),
            uniqueness=int(score.get("uniqueness", 0)),
            failure_reduction=int(score.get("failure_reduction", 0)),
            depends_on=raw.get("depends_on", {}) or {},
            produces=raw.get("produces", {}) or {},
            evolves_from=raw.get("evolves_from"),
            instantiates_as=raw.get("instantiates_as", []) or [],
            implementation_target=raw.get("implementation_target"),
            empirical_anchor=raw.get("empirical_anchor"),
            note=raw.get("note"),
        )

    def to_yaml(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "depends_on": self.depends_on,
            "produces": self.produces,
            "score": {
                "predictive_power": self.predictive_power,
                "actionability": self.actionability,
                "uniqueness": self.uniqueness,
                "failure_reduction": self.failure_reduction,
            },
            "status": self.status,
            "origin": self.origin,
        }
        if self.evolves_from:
            out["evolves_from"] = self.evolves_from
        if self.instantiates_as:
            out["instantiates_as"] = self.instantiates_as
        if self.implementation_target:
            out["implementation_target"] = self.implementation_target
        if self.empirical_anchor:
            out["empirical_anchor"] = self.empirical_anchor
        if self.note:
            out["note"] = self.note
        return out


def load_questions(path: Path | None = None) -> list[Question]:
    """Load all questions from the YAML store.

    Raises FileNotFoundError if path doesn't exist; raises ValueError if
    a question is missing required fields.
    """
    p = path or DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw_qs = data.get("questions", []) or []
    out: list[Question] = []
    for raw in raw_qs:
        try:
            out.append(Question.from_yaml(raw))
        except KeyError as e:
            raise ValueError(f"question {raw.get('id', '?')} missing field {e}") from e
    return out


def rank(questions: list[Question], status_filter: str | None = "active") -> list[Question]:
    """Return questions sorted desc by priority_score.

    By default, only active questions are returned. Pass status_filter=None
    to include all.
    """
    if status_filter:
        questions = [q for q in questions if q.status == status_filter]
    return sorted(questions, key=lambda q: q.priority_score(), reverse=True)


def save_questions(questions: list[Question], path: Path | None = None) -> None:
    """Persist questions to the YAML store.

    Writes a fixed header comment then dumps the questions. Sort order is
    by id for deterministic diffs.
    """
    p = path or DEFAULT_PATH
    payload = {
        "version": "2026-05-16",
        "schema_version": 1,
        "questions": [q.to_yaml() for q in sorted(questions, key=lambda q: q.id)],
    }
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Layer 3 Question Store (SAI substrate)\n")
        f.write("# AUTO-WRITTEN by surveillance.sai.question_store. Edit via the API.\n\n")
        yaml.safe_dump(payload, f, sort_keys=False, width=80, default_flow_style=False)


def summary(questions: list[Question], top_n: int | None = None) -> str:
    """Format ranked table for stdout.

    Columns: rank | id | priority | category | status | one-line question
    """
    ranked = rank(questions, status_filter=None)
    if top_n:
        ranked = ranked[:top_n]
    lines = [
        f"  {'rank':>4}  {'id':6s}  {'score':>5s}  {'category':12s}  {'status':10s}  question",
        "  " + "-" * 90,
    ]
    for i, q in enumerate(ranked, 1):
        q_text = q.question.replace("\n", " ").strip()
        q_text = " ".join(q_text.split())
        # Truncate to one-line preview
        if len(q_text) > 70:
            q_text = q_text[:67] + "..."
        lines.append(
            f"  {i:>4}  {q.id:6s}  {q.priority_score():>5.2f}  "
            f"{q.category[:12]:12s}  {q.status[:10]:10s}  {q_text}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Show ranked SAI questions")
    ap.add_argument("--top", type=int, default=None, help="show only top N")
    ap.add_argument("--include-all", action="store_true", help="include non-active questions")
    ap.add_argument("--path", type=str, default=None, help="custom YAML path")
    args = ap.parse_args()
    p = Path(args.path) if args.path else None
    qs = load_questions(p)
    if not args.include_all:
        qs = [q for q in qs if q.status == "active"]
    print(f"\nLoaded {len(qs)} questions from {p or DEFAULT_PATH}\n")
    print(summary(qs, top_n=args.top))
    print()
    # Aggregate stats
    by_cat: dict[str, int] = {}
    for q in qs:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    print(f"  by category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"    {cat:18s}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
