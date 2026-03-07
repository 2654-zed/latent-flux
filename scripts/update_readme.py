"""Auto-update dynamic sections of README.md.

Introspects the codebase to regenerate:
  - Primitive count and table
  - Test count
  - Project structure

Sections are delimited by <!-- AUTOGEN:TAG --> ... <!-- AUTOGEN:END_TAG --> markers.
Unmarked content is preserved as-is.

Usage:
    python scripts/update_readme.py          # update in-place
    python scripts/update_readme.py --check  # exit 1 if README would change (for CI)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
TESTS_DIR = os.path.join(ROOT, "tests")
PKG_DIR = os.path.join(ROOT, "flux_manifold")

# ── Primitive registry ────────────────────────────────────────────
# Source of truth: the OPERATORS dict in parser.py maps symbols → canonical names.
# We also define display info here.

PRIMITIVES = [
    ("⟼", "FluxManifold", "core.py", "`s₀, q, f → s*`",
     "Continuous flow: iterate `s ← s + ε·f(s, q)` until `‖s − q‖ < tol`"),
    ("∑_ψ", "Superposition Tensor", "superposition.py", "`{sᵢ, wᵢ}ⁿ → batch`",
     "Weighted parallel exploration of N candidate states"),
    ("∇↓", "Dimensional Squeeze", "dimensional_squeeze.py", "`ℝ^d → ℝ^k` where `k < d`",
     "Lossy projection (PCA or random) to reduce dimensionality early"),
    ("≅", "Drift Equivalence", "drift_equivalence.py", "`(a, b, τ) → bool`",
     "Approximate equality: `‖a − b‖ ≤ τ` (L2 or cosine)"),
    ("↓!", "Commitment Sink", "commitment_sink.py", "`s → s` (irreversible)",
     "Lock state when entropy is low or drift stabilizes. Cannot be undone."),
    ("◉", "Fold-Reference", "fold_reference.py", "`(s, step) → s'`",
     "Mid-flow self-critique: detect NaN, norm blowup, edge crossings; apply correction"),
    ("⇑", "Abstraction Cascade", "abstraction_cascade.py", "`ℝ^d → [ℝ^k₁, ℝ^k₂, …]`",
     "Hierarchical PCA reduction into decreasing-dimensional summaries"),
    ("⧖", "Reservoir State", "reservoir_state.py", "`s → s` (memory)",
     "Continuous memory via ESN dynamics — information echoes across flow steps"),
    ("↺", "Recursive Flow", "recursive_flow.py", "`s₀, q → s*`",
     "Fixed-point iteration with geometric termination (attractor convergence or entropy collapse)"),
    ("⊗", "Attractor Competition", "attractor_competition.py", "`s, {aₖ} → winner`",
     "Geometric pattern matching: state flows under simultaneous attraction, basin determines winner"),
]


def count_tests() -> tuple[int, dict[str, int]]:
    """Run pytest --collect-only to get test counts per file."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS_DIR, "--collect-only", "-q"],
        capture_output=True, text=True, cwd=ROOT,
    )
    output = result.stdout + result.stderr

    # Parse per-file counts from lines like "tests/test_core.py::TestClass::test_name"
    file_counts: dict[str, int] = {}
    for line in output.splitlines():
        if line.startswith("tests/"):
            # Extract file path (first :: delimited segment)
            file_path = line.split("::")[0]
            file_counts[file_path] = file_counts.get(file_path, 0) + 1

    # Parse total from final line like "409 tests collected"
    total = sum(file_counts.values())
    m = re.search(r"(\d+) tests? collected", output)
    if m:
        total = int(m.group(1))

    return total, file_counts


def count_test_files() -> int:
    """Count total test files including proof/ subdirectory."""
    count = 0
    for root, _dirs, files in os.walk(TESTS_DIR):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                count += 1
    return count


def build_primitives_table() -> str:
    """Generate markdown primitives table."""
    n = len(PRIMITIVES)
    lines = [
        f"Latent Flux has exactly **{n} primitives**. Each operates on continuous "
        f"states in ℝ^d. Together they form a complete compute cycle: explore → flow → "
        f"critique → accept → commit → abstract → remember → recurse → compete.",
        "",
        "| Symbol | Name | Signature | Semantics |",
        "|--------|------|-----------|----------|",
    ]
    for sym, name, _file, sig, sem in PRIMITIVES:
        lines.append(f"| **{sym}** | {name} | {sig} | {sem} |")
    return "\n".join(lines)


def build_tests_line() -> str:
    """Generate the test count line."""
    total, _per_file = count_tests()
    n_files = count_test_files()
    return f"{total} tests across {n_files} test files."


def update_section(content: str, tag: str, new_body: str) -> str:
    """Replace content between <!-- AUTOGEN:TAG --> and <!-- AUTOGEN:END_TAG -->."""
    pattern = re.compile(
        rf"(<!-- AUTOGEN:{re.escape(tag)} -->\n).*?(\n<!-- AUTOGEN:END_{re.escape(tag)} -->)",
        re.DOTALL,
    )
    replacement = rf"\g<1>{new_body}\g<2>"
    return pattern.sub(replacement, content)


def main() -> int:
    check_only = "--check" in sys.argv

    with open(README, "r", encoding="utf-8") as f:
        original = f.read()

    updated = original

    # Update primitives table
    updated = update_section(updated, "PRIMITIVES_TABLE", build_primitives_table())

    # Update test count
    updated = update_section(updated, "TESTS", build_tests_line())

    if check_only:
        if updated != original:
            print("README.md is out of date. Run: python scripts/update_readme.py")
            return 1
        print("README.md is up to date.")
        return 0

    if updated != original:
        with open(README, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"README.md updated.")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
