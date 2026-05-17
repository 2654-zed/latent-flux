"""Question generator (Phase 3 of SAI plan) — turn failures into questions.

Closes the SAI self-evolution loop:

    FAILURES (evaluation_log)  →  question_generator  →  DRAFT QUESTIONS  →
        human/agent review  →  promoted to memory/questions.yaml

Until this module exists, new questions surface only via human-driven SAI
cycles (like the 2026-05-15 session that produced the original 18). With
this module, every SURPRISE block in JOURNAL.md and every OPEN entry in
UNKNOWNS.md becomes one or more draft questions, automatically.

Three SAI transformation rules applied per failure:

    decomposition         — "Is this risky?"  →  precise sub-questions
                            (where, when, who, how-much)
    adversarial_inversion — "What is safe?"   →  "How would this be attacked?"
    temporal_upgrade      — "Will this fail?" →  "When does this become
                            unstable?"

Output: candidate questions written to `memory/questions_draft.yaml` (or a
custom path via --out) with status=draft. Human/agent review promotes
selected drafts into the main `memory/questions.yaml`.

CLI:
    python -m surveillance.sai.question_generator
    python -m surveillance.sai.question_generator --from-journal
    python -m surveillance.sai.question_generator --from-unknowns
    python -m surveillance.sai.question_generator --from-sai-alerts
    python -m surveillance.sai.question_generator --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML required: pip install pyyaml\n")
    raise

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JOURNAL = REPO_ROOT / "memory" / "JOURNAL.md"
UNKNOWNS = REPO_ROOT / "memory" / "UNKNOWNS.md"
QUESTIONS = REPO_ROOT / "memory" / "questions.yaml"
DRAFT_OUT = REPO_ROOT / "memory" / "questions_draft.yaml"


@dataclass
class FailureEvent:
    """A parsed failure suitable for question generation."""
    source: str       # "journal_surprise" | "unknown_open" | "sai_alert"
    source_id: str    # journal line ref, UNK-XXX, alert detector
    title: str        # one-line description
    expected: str = ""
    observed: str = ""
    implication: str = ""
    raw_context: str = ""


@dataclass
class DraftQuestion:
    """A draft question pending review."""
    derived_from: str   # FailureEvent.source_id
    transformation: str # "decomposition" | "adversarial_inversion" | "temporal_upgrade"
    category: str
    question: str
    rationale: str
    estimated_score: dict   # {predictive_power, actionability, ...}
    status: str = "draft"
    origin: str = ""

    def to_yaml(self) -> dict:
        return {
            "category": self.category,
            "question": self.question.strip(),
            "depends_on": {"signals": [], "models": []},
            "produces": {"decision_output": self.rationale.strip()},
            "score": self.estimated_score,
            "status": self.status,
            "origin": self.origin,
            "derived_from": self.derived_from,
            "transformation": self.transformation,
        }


# ============================================================
# Parsers
# ============================================================

SURPRISE_BLOCK_RE = re.compile(
    # Matches "SURPRISE: <title>\n- Expected: ...\n- Observed: ...\n- Implication: ..."
    # Tolerates the markdown code-fence + bold variants used in JOURNAL.md.
    r"SURPRISE:\s*([^\n]+)\n"
    r"\s*-\s*Expected:\s*([^\n]+(?:\n(?!\s*-\s*\w+:)[^\n]+)*)\n"
    r"\s*-\s*Observed:\s*([^\n]+(?:\n(?!\s*-\s*\w+:)[^\n]+)*)\n"
    r"\s*-\s*Implication:\s*([^\n]+(?:\n(?!\s*-\s*\w+:)[^\n]+)*)",
    re.MULTILINE,
)


def parse_journal_surprises(path: Path = JOURNAL) -> list[FailureEvent]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for i, m in enumerate(SURPRISE_BLOCK_RE.finditer(text)):
        title = m.group(1).strip()
        expected = m.group(2).strip()
        observed = m.group(3).strip()
        implication = m.group(4).strip()
        # Skip the schema-template SURPRISE block in LOOP.md if it
        # somehow appears in JOURNAL.md (placeholder text).
        if "<one-line description>" in title:
            continue
        out.append(FailureEvent(
            source="journal_surprise",
            source_id=f"journal_surprise_{i+1}",
            title=title,
            expected=expected,
            observed=observed,
            implication=implication,
        ))
    return out


UNK_HEADER_RE = re.compile(r"^### (UNK-\d+) — (.+)$", re.MULTILINE)
UNK_STATUS_RE = re.compile(r"\*\*Status:\*\*\s*([^\n]+)")


def parse_open_unknowns(path: Path = UNKNOWNS) -> list[FailureEvent]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    # Split into sections by UNK header
    parts = []
    last = 0
    for m in UNK_HEADER_RE.finditer(text):
        parts.append((m.start(), m.group(1), m.group(2)))
        last = m.start()
    if not parts:
        return []
    # Get section bodies
    for i, (start, unk_id, title) in enumerate(parts):
        end = parts[i + 1][0] if i + 1 < len(parts) else len(text)
        body = text[start:end]
        status_m = UNK_STATUS_RE.search(body)
        status = (status_m.group(1).strip() if status_m else "OPEN").upper()
        if not ("OPEN" in status or "IN_PROGRESS" in status):
            continue
        # Pull "Why it matters" if present
        wm_m = re.search(r"\*\*Why it matters:\*\*\s*([^\n]+(?:\n(?!\s*-\s*\*\*)[^\n]+)*)", body)
        why = wm_m.group(1).strip() if wm_m else ""
        out.append(FailureEvent(
            source="unknown_open",
            source_id=unk_id,
            title=title.strip(),
            implication=why,
            raw_context=body[:1500],
        ))
    return out


def parse_sai_alerts(db_path: Path | None = None, since: str = "2026-05-15") -> list[FailureEvent]:
    """Recent SAI alerts can themselves be question-source events.

    A STALE OLI verdict, for example, is a failure of the current
    OLI guardrail design. The corresponding question evolution might be
    'how do we make INV-007 robust to addresses that go adversarial post-tag.'
    """
    import sqlite3
    if db_path is None:
        db_path = REPO_ROOT / "surveillance" / "data" / "surveillance.db"
    if not db_path.exists():
        return []
    out = []
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        rows = conn.execute(
            """SELECT detected_at, detector, severity, subject_address, payload
               FROM sai_alerts WHERE detected_at >= ?
               ORDER BY detected_at DESC LIMIT 200""",
            (since,)
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        return []
    for ts, detector, severity, subject, _payload in rows:
        # Treat only high-severity alerts as failure-source candidates
        if severity not in ("T1_IMMINENT", "STALE", "T1_BRIDGE_CORRELATION"):
            continue
        out.append(FailureEvent(
            source="sai_alert",
            source_id=f"{detector}_{ts}_{subject}",
            title=f"{detector} alert: {severity} on {subject}",
            observed=f"{detector} fired {severity} for {subject} at {ts}",
        ))
    return out


# ============================================================
# Transformations
# ============================================================

def transform_decompose(failure: FailureEvent) -> DraftQuestion | None:
    """Generate a decomposition: vague title -> precise sub-question.

    Heuristic: identify the most-vague verb in the title and ask the
    "where/when/who/how-much" version.
    """
    title = failure.title
    if any(v in title.lower() for v in ["risk", "fail", "vulnerab", "issue", "concern"]):
        question = (
            f"For the failure '{title}', what specific addresses, timestamps, "
            f"and signal trajectories would have made the failure visible "
            f"before it manifested? Build a checkable measurement plan."
        )
        rationale = "Decomposition turns a vague-title failure into a measurable precondition set."
    elif failure.source == "unknown_open":
        # Decompose the unknown into substrate-checkable parts
        question = (
            f"What concrete query against surveillance.db answers '{title}' "
            f"with verifiable address-level evidence (score >= 2 on the "
            f"Q-004 verifiability scale)?"
        )
        rationale = "Decomposition of an OPEN unknown into a corpus-verifiable query plan."
    else:
        return None
    return DraftQuestion(
        derived_from=failure.source_id,
        transformation="decomposition",
        category="methodology",
        question=question,
        rationale=rationale,
        estimated_score={
            "predictive_power": 2,
            "actionability": 4,
            "uniqueness": 3,
            "failure_reduction": 4,
        },
        origin=f"question_generator/decompose/{failure.source}",
    )


def transform_adversarial_invert(failure: FailureEvent) -> DraftQuestion | None:
    """Generate the adversarial inversion of a failure.

    "How did this fail?" -> "How would an attacker engineer this failure?"
    """
    title = failure.title
    if failure.implication and "guardrail" in failure.implication.lower():
        attack_target = "guardrail"
    elif "watchlist" in title.lower() or "coverage" in title.lower():
        attack_target = "watchlist coverage gap"
    elif "prediction" in title.lower() or "predict" in failure.observed.lower():
        attack_target = "prediction substrate"
    else:
        attack_target = title.lower()[:50]
    question = (
        f"If an attacker knew that '{attack_target}' could fail in the way "
        f"observed in {failure.source_id}, how would they engineer a "
        f"high-leverage exploitation of that failure mode? What downstream "
        f"systems become vulnerable?"
    )
    rationale = (
        f"Adversarial inversion of {failure.source_id}: convert defender's "
        f"surprise into attacker's playbook."
    )
    return DraftQuestion(
        derived_from=failure.source_id,
        transformation="adversarial_inversion",
        category="adversarial",
        question=question,
        rationale=rationale,
        estimated_score={
            "predictive_power": 3,
            "actionability": 2,
            "uniqueness": 5,
            "failure_reduction": 5,
        },
        origin=f"question_generator/invert/{failure.source}",
    )


def transform_temporal_upgrade(failure: FailureEvent) -> DraftQuestion | None:
    """Generate the temporal upgrade.

    "Will X fail?" -> "When does X become unstable? At what Z-score / time
    window / threshold does the failure mode become detectable?"
    """
    title = failure.title
    if "discharge" in title.lower() or "drain" in title.lower() or "spike" in title.lower():
        question = (
            f"For the event class observed in '{title}', what is the minimum "
            f"lead time T such that the corresponding leading indicator fires "
            f"with confidence > 0.5? Construct the time-to-trigger distribution "
            f"across all corpus instances."
        )
        rationale = "Temporal upgrade: from 'will it fail' to 'how many minutes / hours before the event is it detectable.'"
    elif failure.source == "unknown_open":
        question = (
            f"For '{title}', what is the time window after which leaving the "
            f"question unresolved produces material risk? Set a decay clock "
            f"on the OPEN status."
        )
        rationale = "Temporal upgrade of an open unknown: forces a deadline."
    else:
        return None
    return DraftQuestion(
        derived_from=failure.source_id,
        transformation="temporal_upgrade",
        category="behavior",
        question=question,
        rationale=rationale,
        estimated_score={
            "predictive_power": 4,
            "actionability": 4,
            "uniqueness": 4,
            "failure_reduction": 3,
        },
        origin=f"question_generator/temporal/{failure.source}",
    )


def _detector_id_from_alert(failure: FailureEvent) -> str | None:
    """Extract the detector id (e.g., 'Q-002') from an sai_alert source_id."""
    if failure.source != "sai_alert":
        return None
    # source_id format: f"{detector}_{ts}_{subject}", e.g. "Q-003_2026-05-17T...."
    parts = failure.source_id.split("_", 1)
    return parts[0] if parts and parts[0].startswith("Q-") else None


# ============================================================
# Alert-specific transformations (added 2026-05-17)
# ============================================================
#
# The prose-tuned transformations (decompose / adversarial_invert /
# temporal_upgrade) over-fit JOURNAL.md SURPRISE blocks and OPEN
# UNKnowns: they look for keywords ("risk", "fail", "discharge", "drain",
# "spike") in the title. Machine-generated alert titles like
# "Q-003 alert: STALE on 0x..." don't contain those keywords, so the
# pre-existing transformations produce only adversarial_inversion on
# every sai_alert. Documented as the 2026-05-17 SURPRISE block in
# JOURNAL.md.
#
# These three alert-specific transformations close that gap. Each
# converts a fired-alert into a forward-looking research question that
# would either (a) make the detector fire earlier next time, (b) reveal
# the structural gap exposed by the alert, or (c) make resolution
# actionable per-alert.


def transform_alert_earlier_detection(failure: FailureEvent) -> DraftQuestion | None:
    """For each alert, what new signal would have fired earlier than the
    current detector?"""
    detector = _detector_id_from_alert(failure)
    if detector is None:
        return None
    # Per-detector framing (each has a different "earlier" question)
    detector_specific = {
        "Q-002": "approval-rate Z-score on the bait contract",
        "Q-003": "OLI temporal-validity verdict",
        "Q-005": "cross-chain choreography signal",
        "Q-009": "funding-chain ancestor resolution",
    }.get(detector, "this detector")
    question = (
        f"For alert {failure.source_id}: what additional signal — independent "
        f"of {detector_specific} — would have fired BEFORE this detector did? "
        f"Specifically: at what hour mark of the trailing 72h could a "
        f"complementary detector have surfaced the same subject with confidence "
        f"> 0.5?"
    )
    rationale = (
        f"Earlier-detection upgrade: turns a fired {detector} alert into the "
        f"research question 'what would have caught this even sooner?' — "
        f"each successful detection should produce its own next-generation "
        f"complementary detector candidate."
    )
    return DraftQuestion(
        derived_from=failure.source_id,
        transformation="alert_earlier_detection",
        category="pricing",
        question=question,
        rationale=rationale,
        estimated_score={
            "predictive_power": 4,
            "actionability": 4,
            "uniqueness": 4,
            "failure_reduction": 4,
        },
        origin=f"question_generator/alert_earlier/{detector}",
    )


def transform_alert_structural_gap(failure: FailureEvent) -> DraftQuestion | None:
    """For each alert, what surveillance structural gap does this expose?"""
    detector = _detector_id_from_alert(failure)
    if detector is None:
        return None
    # The structural gap question reframes the alert as a downstream-coverage
    # question instead of a per-subject question.
    gap_specific = {
        "Q-002": (
            "How many other watchlisted bait contracts are in their accumulation "
            "phase with > 1,000 lifetime approvals but no recent Z-spike?"
        ),
        "Q-003": (
            "How many other OLI-tagged addresses share the staleness signal "
            "pattern but have not been alerted because they're missing one "
            "or more of the 7 staleness signals?"
        ),
        "Q-005": (
            "How many other deployers have multi-chain bytecode-family "
            "presence without yet showing the Pattern D temporal-gap signal?"
        ),
        "Q-009": (
            "How many other recent drain_callers fail to resolve to a known "
            "ancestor in the same window, and what funding pattern do they "
            "share that this alert's drainer did not have?"
        ),
    }.get(detector, "what coverage gap is downstream of this alert?")
    question = (
        f"Alert {failure.source_id} fired. {gap_specific} Treat the fired "
        f"alert as evidence of a larger latent cohort that the detector's "
        f"current threshold has not yet surfaced."
    )
    rationale = (
        "Structural-gap framing: a fired alert is one positive in a "
        "distribution. The next question is how the cohort containing the "
        "alert is distributed in the corpus, and where its left tail sits "
        "under our current threshold."
    )
    return DraftQuestion(
        derived_from=failure.source_id,
        transformation="alert_structural_gap",
        category="structural",
        question=question,
        rationale=rationale,
        estimated_score={
            "predictive_power": 3,
            "actionability": 5,
            "uniqueness": 4,
            "failure_reduction": 4,
        },
        origin=f"question_generator/alert_gap/{detector}",
    )


def transform_alert_actionable_resolution(failure: FailureEvent) -> DraftQuestion | None:
    """For each alert, what specific data per alert would make the
    resolution actionable (rather than just a flag)?"""
    detector = _detector_id_from_alert(failure)
    if detector is None:
        return None
    resolution_specific = {
        "Q-002": (
            "estimated time-to-discharge, USD-at-stake estimate, count of "
            "victim addresses approved-but-not-yet-drained"
        ),
        "Q-003": (
            "external attestation method (which OLI submitter can confirm "
            "current control proof), expected institution-side disclosure timeline"
        ),
        "Q-005": (
            "concrete cross-chain coordination event (specific tx hashes "
            "evidencing same-EOA activity on other chains in the window)"
        ),
        "Q-009": (
            "second-hop and third-hop funder identities, cluster membership "
            "of the resolved ancestor"
        ),
    }.get(detector, "structured fields that close the alert")
    question = (
        f"Alert {failure.source_id} produced a flag. What additional "
        f"payload fields ({resolution_specific}) would transform the alert "
        f"from a notification into a complete action ticket?"
    )
    rationale = (
        "Actionable-resolution upgrade: turns 'something is wrong' alerts "
        "into 'here is what to do' alerts. Each detector's payload should "
        "carry enough structured context to drive a downstream decision "
        "without re-querying the corpus."
    )
    return DraftQuestion(
        derived_from=failure.source_id,
        transformation="alert_actionable_resolution",
        category="methodology",
        question=question,
        rationale=rationale,
        estimated_score={
            "predictive_power": 2,
            "actionability": 5,
            "uniqueness": 3,
            "failure_reduction": 3,
        },
        origin=f"question_generator/alert_actionable/{detector}",
    )


def generate_for_failure(failure: FailureEvent) -> list[DraftQuestion]:
    """Apply all transformations. Returns 0-6 candidate questions.

    Three "prose" transformations apply to JOURNAL surprises and UNK
    entries. Three "alert-aware" transformations apply to sai_alert
    failures. The split is necessary because alert titles are
    machine-generated and don't trigger the prose keyword matches.
    """
    out = []
    for fn in (
        transform_decompose,
        transform_adversarial_invert,
        transform_temporal_upgrade,
        # Alert-aware transformations
        transform_alert_earlier_detection,
        transform_alert_structural_gap,
        transform_alert_actionable_resolution,
    ):
        q = fn(failure)
        if q is not None:
            out.append(q)
    return out


# ============================================================
# Output
# ============================================================

def write_drafts(drafts: list[DraftQuestion], path: Path = DRAFT_OUT) -> None:
    """Write draft questions to a YAML file. Overwrites; not append-mode."""
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "surveillance.sai.question_generator",
        "n_drafts": len(drafts),
        "drafts": [
            {"id": f"DRAFT-{i+1:03d}", **d.to_yaml()}
            for i, d in enumerate(drafts)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Layer 3 SAI draft questions (generated)\n")
        f.write("# These are pending human/agent review. Selected drafts get\n")
        f.write("# promoted to memory/questions.yaml with status=active and\n")
        f.write("# a new Q-XXX id.\n\n")
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False, width=80)


def existing_question_ids(path: Path = QUESTIONS) -> set[str]:
    """Existing question ids so we don't generate duplicates."""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {q["id"] for q in (data or {}).get("questions", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-journal", action="store_true",
                    help="parse JOURNAL.md SURPRISE blocks")
    ap.add_argument("--from-unknowns", action="store_true",
                    help="parse UNKNOWNS.md OPEN entries")
    ap.add_argument("--from-sai-alerts", action="store_true",
                    help="parse recent sai_alerts table rows")
    ap.add_argument("--all", action="store_true",
                    help="enable all input sources")
    ap.add_argument("--dry-run", action="store_true",
                    help="print drafts but don't write to file")
    ap.add_argument("--out", default=str(DRAFT_OUT))
    ap.add_argument("--since", default="2026-05-15",
                    help="for --from-sai-alerts; ISO date")
    args = ap.parse_args()

    sources = []
    if args.all or args.from_journal:
        sources.append(("journal", parse_journal_surprises(JOURNAL)))
    if args.all or args.from_unknowns:
        sources.append(("unknowns", parse_open_unknowns(UNKNOWNS)))
    if args.all or args.from_sai_alerts:
        sources.append(("sai_alerts", parse_sai_alerts(since=args.since)))
    if not sources:
        # default: journal + unknowns
        sources.append(("journal", parse_journal_surprises(JOURNAL)))
        sources.append(("unknowns", parse_open_unknowns(UNKNOWNS)))

    failures: list[FailureEvent] = []
    for name, items in sources:
        failures.extend(items)
        print(f"  source '{name}': {len(items)} failure events parsed", file=sys.stderr)

    drafts: list[DraftQuestion] = []
    for f in failures:
        drafts.extend(generate_for_failure(f))
    print(f"\nGenerated {len(drafts)} draft questions from {len(failures)} failures", file=sys.stderr)

    # Quick summary
    print(f"\nTop 10 drafts by estimated score:\n")
    def score(d: DraftQuestion) -> float:
        s = d.estimated_score
        return (s["predictive_power"] * 0.30
                + s["actionability"] * 0.30
                + s["failure_reduction"] * 0.30
                + s["uniqueness"] * 0.10)
    # Sanitize for cp1252 stdout (Windows) — non-ASCII chars from the corpus
    # (Latent Flux symbols, em-dashes, etc.) crash bare print() on Windows.
    def _ascii_safe(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")

    for d in sorted(drafts, key=score, reverse=True)[:10]:
        print(f"  est_score={score(d):.2f}  [{d.transformation:>22s}]  from={d.derived_from}")
        q = d.question.replace("\n", " ").strip()
        if len(q) > 100:
            q = q[:97] + "..."
        print(f"    {_ascii_safe(q)}")
        print()

    if not args.dry_run:
        write_drafts(drafts, Path(args.out))
        print(f"\nWrote {len(drafts)} drafts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
