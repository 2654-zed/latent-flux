"""Drift detector for docs/INDEX.md.

Compares INDEX.md address mentions and case-file references against the
actual contents of `surveillance/data/cases/` and `reports/`. Returns
nonzero exit if drift is detected.

Three classes of drift surfaced:

  1. INDEX.md cites a file path that no longer exists.
  2. A `surveillance/data/cases/` file is not referenced anywhere in INDEX.md.
  3. An address appears in a case file but is not in INDEX.md (informational
     only — many victim/bot addresses correctly aren't enumerated).

Usage:
    python scripts/check_index.py            # report
    python scripts/check_index.py --strict   # also fail on class 3 drift
    python scripts/check_index.py --quiet    # only print drift, no summary

Exit codes:
    0 — no drift in classes 1 and 2 (pre-commit safe)
    1 — drift detected in class 1 (broken file references)
    2 — drift detected in class 2 (case files not in index)
    3 — drift detected in classes 1+2
    Plus +4 in --strict mode if class 3 drift exists.

Designed to be cheap: pure file IO, no DB access, no RPC. Suitable as
pre-commit hook or session-start staleness check.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "docs" / "INDEX.md"
CASES_DIR = REPO_ROOT / "surveillance" / "data" / "cases"
REPORTS_DIR = REPO_ROOT / "reports"

ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
# Match `path/to/file.md` or just `FILE_NAME.md` (with or without backticks)
FILE_REF_RE = re.compile(r"`?([\w/]+\.md)`?")


def load_index() -> str:
    if not INDEX_PATH.exists():
        sys.stderr.write(f"FATAL: {INDEX_PATH} does not exist.\n")
        sys.exit(99)
    return INDEX_PATH.read_text(encoding="utf-8")


def list_md(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {f.name for f in directory.iterdir() if f.is_file() and f.suffix == ".md"}


def addresses_in(text: str) -> set[str]:
    return {m.group(0).lower() for m in ADDR_RE.finditer(text)}


def class1_broken_refs(index_text: str) -> list[str]:
    """File paths cited in INDEX.md that no longer exist in the repo."""
    broken = []
    seen = set()
    for m in FILE_REF_RE.finditer(index_text):
        ref = m.group(1)
        if ref in seen:
            continue
        seen.add(ref)
        # Only check refs that look like real file paths (contain a known directory)
        if not any(seg in ref for seg in ("cases/", "reports/", "docs/", "surveillance/", "scripts/")):
            # Bare filename (e.g., "CASE_FOO.md") — try to resolve under cases/, reports/, docs/, or root
            candidates = [
                CASES_DIR / ref,
                REPORTS_DIR / ref,
                REPO_ROOT / "docs" / ref,
                REPO_ROOT / ref,
            ]
            if not any(c.exists() for c in candidates):
                broken.append(ref)
            continue
        # Path-qualified ref
        path = REPO_ROOT / ref
        if not path.exists():
            broken.append(ref)
    return broken


# Cyclic/ephemeral report prefixes — by convention these are session outputs,
# not entity-level case files. They aren't indexed individually.
EPHEMERAL_PREFIXES = (
    "DAILY_REPORT_",
    "FUND_FLOW_TRACE_",
    "ORG_CYCLES_",
    "INFRA_EVENT_",
    "ENTITY_CLASSIFICATION_",
)


def class2_unindexed_cases(index_text: str) -> list[str]:
    """Case-file names that don't appear in INDEX.md.

    Skips ephemeral cyclic reports (DAILY_REPORT_*, FUND_FLOW_TRACE_*, etc.)
    which are intentionally not indexed per the maintenance protocol.
    """
    case_files = list_md(CASES_DIR)
    unindexed = []
    for f in sorted(case_files):
        if any(f.startswith(p) for p in EPHEMERAL_PREFIXES):
            continue
        # The filename should appear somewhere in the index text
        if f not in index_text:
            unindexed.append(f)
    return unindexed


def class3_unindexed_addresses(index_text: str) -> dict[str, set[str]]:
    """Addresses appearing in case/report files but not in INDEX.md.

    Returns map of address -> set of files where it appears.
    """
    indexed_addrs = addresses_in(index_text)
    drift: dict[str, set[str]] = {}
    for directory in (CASES_DIR, REPORTS_DIR):
        if not directory.is_dir():
            continue
        for f in directory.iterdir():
            if not (f.is_file() and f.suffix == ".md"):
                continue
            try:
                content = f.read_text(encoding="utf-8")
            except Exception:
                continue
            for addr in addresses_in(content):
                if addr not in indexed_addrs:
                    drift.setdefault(addr, set()).add(f.name)
    return drift


def main() -> int:
    ap = argparse.ArgumentParser(description="INDEX.md drift detector")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on class 3 drift (unindexed addresses)")
    ap.add_argument("--quiet", action="store_true",
                    help="Only print drift, suppress summary header")
    args = ap.parse_args()

    index_text = load_index()

    broken = class1_broken_refs(index_text)
    unindexed_cases = class2_unindexed_cases(index_text)
    unindexed_addrs = class3_unindexed_addresses(index_text)

    if not args.quiet:
        print(f"INDEX.md: {INDEX_PATH}")
        print(f"  size: {len(index_text):,} chars")
        print(f"  cases/ files: {len(list_md(CASES_DIR))}")
        print(f"  reports/ files: {len(list_md(REPORTS_DIR))}")
        print()

    exit_code = 0

    # Class 1: broken file refs
    if broken:
        exit_code |= 1
        print(f"CLASS 1 DRIFT — {len(broken)} broken file reference(s) in INDEX.md:")
        for ref in broken:
            print(f"  - {ref}")
        print()
    elif not args.quiet:
        print("CLASS 1: no broken file references.")

    # Class 2: case files not in index
    if unindexed_cases:
        exit_code |= 2
        print(f"CLASS 2 DRIFT — {len(unindexed_cases)} case file(s) not referenced in INDEX.md:")
        for f in unindexed_cases:
            print(f"  - {f}")
        print()
    elif not args.quiet:
        print("CLASS 2: every case file is referenced in INDEX.md.")

    # Class 3: addresses in cases but not in index
    if unindexed_addrs:
        if args.strict:
            exit_code |= 4
        sample = sorted(unindexed_addrs.items())[:10]
        if not args.quiet or args.strict:
            print(f"CLASS 3 INFO — {len(unindexed_addrs)} address(es) appear in case/report files but not in INDEX.md.")
            print(f"  (Many of these are correctly omitted — victim addresses, single-file artifacts, etc.)")
            print(f"  Sample (first 10):")
            for addr, files in sample:
                print(f"    {addr}  in {sorted(files)[0]}")
            print()
    elif not args.quiet:
        print("CLASS 3: every documented address is in INDEX.md (no further sample).")

    if not args.quiet:
        if exit_code == 0:
            print("INDEX.md is clean.")
        else:
            print(f"INDEX.md drift detected (exit code {exit_code}).")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
