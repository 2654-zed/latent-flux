"""Question generator — turns evaluation failures into new questions.

STATUS: SKELETON. The input adapters (parse JOURNAL.md SURPRISE blocks,
parse evaluation_summary fields, parse failed predictions) are not yet
implemented. Full build target: 2-3 sessions.

This module closes the SAI loop: failures → gaps → questions → updated
question store. Without this, the loop is open and the system can't
self-evolve.

The three transformation rules (per SAI Question Evolution):
    decomposition         — vague question → precise sub-questions
    adversarial_inversion — "what is safe?" → "how would this be attacked?"
    temporal_upgrade      — "will this fail?" → "when does this become unstable?"

Each failure event produces 1-3 candidate questions, scored and pending
human/agent review before being committed to questions.yaml.

CLI (when implemented):
    python -m surveillance.sai.question_generator --from-journal
    python -m surveillance.sai.question_generator --from-failed-prediction
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class FailureEvent:
    source: str           # journal entry id, prediction id, weakness id
    failure_type: str     # SURPRISE, INV_VIOLATION, CAPABILITY_DECAY, etc.
    description: str
    context: dict


def generate_questions_from_failure(failure: FailureEvent) -> list[dict]:
    """Generate candidate questions from a failure event.

    TODO: implement the three transformation rules.
    Currently returns a placeholder showing the intended structure.
    """
    return [
        {
            "category": "TBD",
            "question": f"[STUB] What question would have prevented: {failure.description[:80]}?",
            "origin": f"failure_{failure.source}",
            "status": "draft",
            "_skeleton": True,
        }
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-journal", action="store_true")
    args = ap.parse_args()
    print("[SKELETON] question_generator not yet implemented.")
    print()
    print("Intended inputs:")
    print("  - memory/JOURNAL.md SURPRISE blocks (parse Expected/Observed/Implication)")
    print("  - memory/UNKNOWNS.md OPEN entries (each is a latent question)")
    print("  - SAI evaluation_summary FAILED capabilities")
    print()
    print("Intended outputs:")
    print("  - Draft questions appended to questions.yaml with status=draft")
    print("  - Three transformations per failure (decomposition / adversarial / temporal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
