"""Question runner — dispatch active questions to their executable modules.

For each question in the SAI store with `status=active` and an
`implementation_target` path, this module verifies the implementation
exists and (if `--execute` is passed) imports and runs it.

This is the bridge between the question store and the working surveillance
substrate. Without it, the questions are documentation; with it, they are
operations.

CLI:
    python -m surveillance.sai.question_runner            # dry-run: show map
    python -m surveillance.sai.question_runner --execute  # run wired modules
    python -m surveillance.sai.question_runner --id Q-002 # run single question

Current wiring status:
    Q-002 (approval_spike_detector)         WIRED + tested
    Q-001 (role_classifier)                 SKELETON
    Q-004 (prediction_verifiability)        SKELETON
    Q-005 (cross_chain_choreography)        TODO
    Q-008 (capability_liveness)             SKELETON
    everything else                         TODO
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from surveillance.sai.question_store import Question, load_questions, rank

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class WiringStatus:
    question_id: str
    target: str | None
    exists: bool
    importable: bool
    callable_main: bool
    note: str = ""

    def status_label(self) -> str:
        if self.target is None:
            return "NO_TARGET"
        if not self.exists:
            return "MISSING_FILE"
        if not self.importable:
            return "IMPORT_ERROR"
        if not self.callable_main:
            return "NO_MAIN"
        return "WIRED"


def check_wiring(q: Question) -> WiringStatus:
    target = q.implementation_target
    if target is None:
        return WiringStatus(q.id, None, False, False, False, note="no implementation_target")
    path = REPO_ROOT / target
    if not path.exists():
        return WiringStatus(q.id, target, False, False, False, note="file does not exist")
    # Convert to importable module name
    rel = path.relative_to(REPO_ROOT)
    if rel.suffix != ".py":
        return WiringStatus(q.id, target, True, False, False, note="not a .py file")
    module_name = ".".join(rel.with_suffix("").parts)
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None:
            return WiringStatus(q.id, target, True, False, False, note="cannot build import spec")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        has_main = callable(getattr(mod, "main", None))
        return WiringStatus(q.id, target, True, True, has_main,
                            note="" if has_main else "no main() function")
    except Exception as e:
        return WiringStatus(q.id, target, True, False, False, note=f"import error: {e}")


def show_map(questions: list[Question]) -> None:
    print(f"{'id':6s}  {'priority':>8s}  {'status':>12s}  target")
    print("-" * 90)
    for q in rank(questions, status_filter="active"):
        w = check_wiring(q)
        print(f"{q.id:6s}  {q.priority_score():>8.2f}  {w.status_label():>12s}  "
              f"{q.implementation_target or '(none)'}")
        if w.note and w.status_label() != "WIRED":
            print(f"{' ':6s}  {' ':>8s}  {' ':>12s}  note: {w.note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true",
                    help="execute WIRED modules (calls their main())")
    ap.add_argument("--id", type=str, default=None,
                    help="execute only this question id")
    args = ap.parse_args()

    questions = load_questions()
    if args.id:
        questions = [q for q in questions if q.id == args.id]
        if not questions:
            print(f"No question with id={args.id}", file=sys.stderr)
            return 1

    if not args.execute:
        show_map(questions)
        return 0

    failures = 0
    for q in rank(questions, status_filter="active"):
        w = check_wiring(q)
        if w.status_label() != "WIRED":
            continue
        print(f"\n=== Executing {q.id} ({q.implementation_target}) ===")
        try:
            spec = importlib.util.spec_from_file_location(
                "_runner_mod", str(REPO_ROOT / q.implementation_target)
            )
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            rc = mod.main()
            print(f"\n=== {q.id} returned {rc} ===")
            if rc != 0:
                failures += 1
        except Exception as e:
            print(f"=== {q.id} crashed: {e} ===")
            failures += 1
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
