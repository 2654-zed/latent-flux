"""Measurement framework for the Latent Flux Proof Test Suite.

Provides:
- Encoding decision counting from annotated source
- Coherence rubric scoring (automated heuristic)
- Structural information rubric scoring
- Statistical analysis utilities
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass


# ── Encoding Decision Counting ──────────────────────────────────

_ED_PATTERN = re.compile(r'\[ED:(repr|param|scaffold)\]')
_LF_NATIVE = re.compile(r'\[LF:native\]')


def count_encoding_decisions(source: str) -> dict:
    """Count [ED:repr], [ED:param], [ED:scaffold] annotations."""
    matches = _ED_PATTERN.findall(source)
    counts = {'repr': 0, 'param': 0, 'scaffold': 0}
    for m in matches:
        counts[m] += 1
    counts['total'] = sum(counts.values())
    return counts


def count_native_mappings(source: str) -> int:
    """Count [LF:native] annotations."""
    return len(_LF_NATIVE.findall(source))


def encoding_decision_ratio(python_source: str, lf_source: str) -> float:
    """python_decisions / lf_decisions.  >1 = evidence for thesis."""
    py = count_encoding_decisions(python_source)['total']
    lf = count_encoding_decisions(lf_source)['total']
    if lf == 0:
        return float('inf')
    return py / lf


# ── Encoding Decision Model ─────────────────────────────────────
# Instead of counting annotations in static code, model the encoding
# cost per constraint type for each paradigm.

# Python: each constraint type incurs representation + parameter costs.
# Base overhead: learning rate, max_iter, conv_threshold (3 params)
#                + iteration loop, gradient step, convergence check (3 scaffold)
PYTHON_ED_BASE = {'param': 3, 'scaffold': 3}
PYTHON_ED_PER_CONSTRAINT = {
    'proximity':     {'repr': 1, 'param': 1},
    'orthogonality': {'repr': 1, 'param': 1},
    'norm_bound':    {'repr': 1, 'param': 1},
    'subspace':      {'repr': 2, 'param': 1},
}

# LF: composite attractor construction (1 repr), but primitives handle flow.
# Base: 4 native mappings (superpose, flow, reweight, collapse)
# Per-constraint: proximity is free (flow handles it).
# Other types need custom flow function encoding.
LF_ED_BASE = {'repr': 1}
LF_ED_PER_CONSTRAINT = {
    'proximity':     {},                    # native
    'orthogonality': {'repr': 1},           # custom flow fn
    'norm_bound':    {'repr': 1},           # custom flow fn
    'subspace':      {'repr': 1},           # custom flow fn
}
LF_NATIVE_BASE = 4  # ∑_ψ, ⟼, reweight_by_drift, collapse_to_best


def modeled_encoding_ratio(constraint_types: list[str]) -> dict:
    """Compute encoding decision ratio from constraint types (no code needed)."""
    py_total = sum(PYTHON_ED_BASE.values())
    for ct in constraint_types:
        py_total += sum(PYTHON_ED_PER_CONSTRAINT.get(ct, {}).values())

    lf_total = sum(LF_ED_BASE.values())
    for ct in constraint_types:
        lf_total += sum(LF_ED_PER_CONSTRAINT.get(ct, {}).values())

    ratio = py_total / max(lf_total, 1)
    return {
        'python_decisions': py_total,
        'lf_decisions': lf_total,
        'lf_native_mappings': LF_NATIVE_BASE,
        'ratio': ratio,
    }


# ── Coherence Scoring ───────────────────────────────────────────

@dataclass
class CoherenceScore:
    score: int             # 1-5
    rationale: str
    geometric_ops: int
    symbolic_ops: int
    representation_switches: int


_GEO_PATTERNS = [
    r'np\.linalg\.norm', r'np\.dot', r'flow', r'⟼', r'->',
    r'∑_ψ', r'superpos', r'squeeze', r'≅', r'~=', r'↓!', r'commit',
    r'⇑', r'cascade', r'◉', r'fold', r'attractor', r'drift',
    r'collapse', r'reweight', r'entropy', r'flux_flow',
    r'SuperpositionTensor', r'DriftEquivalence', r'normalize_flow',
    r'@',
]
_SYM_PATTERNS = [
    r'\bfor\b', r'\bwhile\b', r'\bif\b', r'\belse\b', r'\belif\b',
    r'\breturn\b', r'\bdef\b', r'\bclass\b', r'\btry\b', r'\bexcept\b',
    r'\.append\b', r'\.items\b', r'\brange\(', r'\blen\(',
]


def score_coherence(source: str) -> CoherenceScore:
    """Score solution coherence on the 1-5 rubric (automated heuristic)."""
    lines = [l.strip() for l in source.strip().split('\n')
             if l.strip() and not l.strip().startswith('#')]

    geo, sym, switches = 0, 0, 0
    last_geo = None
    for line in lines:
        is_geo = any(re.search(p, line) for p in _GEO_PATTERNS)
        is_sym = any(re.search(p, line) for p in _SYM_PATTERNS)
        if is_geo:
            geo += 1
        if is_sym:
            sym += 1
        current = is_geo and not is_sym
        if last_geo is not None and current != last_geo:
            switches += 1
        last_geo = current

    total = max(geo + sym, 1)
    ratio = geo / total
    if ratio >= 0.9 and switches <= 1:
        score = 5
    elif ratio >= 0.7:
        score = 4
    elif ratio >= 0.4:
        score = 3
    elif ratio >= 0.15:
        score = 2
    else:
        score = 1
    return CoherenceScore(score, f"geo={ratio:.2f} sw={switches}", geo, sym, switches)


# ── Structural Information Scoring ──────────────────────────────

@dataclass
class StructuralInfoScore:
    geometry_simplification: int     # 0-3
    binding_constraints: int         # 0-3
    intrinsic_dimensionality: int    # 0-3
    solution_stability: int          # 0-3
    causal_structure: int            # 0-3

    @property
    def total(self) -> int:
        return (self.geometry_simplification + self.binding_constraints +
                self.intrinsic_dimensionality + self.solution_stability +
                self.causal_structure)


def score_structural_info(trace: dict) -> StructuralInfoScore:
    """Score structural information content of an execution trace."""
    # Geometry simplification
    geo = 0
    if 'path' in trace or 'intermediate_states' in trace:
        geo = 2
        if 'compression_ratio' in trace or 'squeeze_ratio' in trace:
            geo = 3
    elif 'converged_state' in trace or 'final_state' in trace:
        geo = 1

    # Binding constraints
    bind = 0
    if 'per_constraint_trajectory' in trace:
        bind = 3
    elif 'constraint_satisfaction' in trace or 'per_constraint_residual' in trace:
        bind = 2
    elif 'final_residuals' in trace:
        bind = 1

    # Intrinsic dimensionality
    dim = 0
    if 'abstraction_levels' in trace:
        dim = 3
    elif 'explained_variance' in trace or 'eigenvalues' in trace:
        dim = 2
    elif 'effective_dim' in trace:
        dim = 1

    # Solution stability
    stab = 0
    if 'drift_trace' in trace and ('entropy' in trace or 'entropy_before' in trace):
        stab = 3
    elif 'drift_trace' in trace:
        stab = 2
    elif 'converged' in trace:
        stab = 1

    # Causal structure
    causal = 0
    if 'trap_type' in trace or 'commit_reason' in trace:
        causal = 3
    elif 'drift_trace' in trace and len(trace.get('drift_trace', [])) > 1:
        causal = 2
    elif 'error' in trace or 'message' in trace:
        causal = 1

    return StructuralInfoScore(geo, bind, dim, stab, causal)


# ── Statistical Utilities ───────────────────────────────────────

@dataclass
class StatResult:
    mean_a: float
    mean_b: float
    difference: float
    t_statistic: float
    p_value: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    n_a: int
    n_b: int
    significant: bool


def compare_groups(a: list[float], b: list[float], alpha: float = 0.05) -> StatResult:
    """Two-sample t-test with effect size."""
    from scipy import stats as sp_stats
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    mean_a, mean_b = float(a_arr.mean()), float(b_arr.mean())
    diff = mean_a - mean_b
    t_stat, p_val = sp_stats.ttest_ind(a_arr, b_arr)
    pooled = np.sqrt((a_arr.var(ddof=1) + b_arr.var(ddof=1)) / 2)
    d_eff = diff / pooled if pooled > 0 else 0.0
    se = np.sqrt(a_arr.var(ddof=1) / len(a_arr) + b_arr.var(ddof=1) / len(b_arr))
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=len(a_arr) + len(b_arr) - 2)
    return StatResult(
        mean_a, mean_b, diff, float(t_stat), float(p_val), float(d_eff),
        diff - t_crit * se, diff + t_crit * se, len(a_arr), len(b_arr),
        float(p_val) < alpha,
    )


def paired_compare(a: list[float], b: list[float], alpha: float = 0.05) -> StatResult:
    """Paired t-test for within-subject comparisons."""
    from scipy import stats as sp_stats
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    diff_arr = a_arr - b_arr
    mean_diff = float(diff_arr.mean())
    t_stat, p_val = sp_stats.ttest_rel(a_arr, b_arr)
    std_diff = float(diff_arr.std(ddof=1)) if len(diff_arr) > 1 else 1.0
    d_eff = mean_diff / std_diff if std_diff > 0 else 0.0
    se = std_diff / np.sqrt(len(diff_arr))
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=len(diff_arr) - 1)
    return StatResult(
        float(a_arr.mean()), float(b_arr.mean()), mean_diff,
        float(t_stat), float(p_val), float(d_eff),
        mean_diff - t_crit * se, mean_diff + t_crit * se,
        len(a_arr), len(b_arr), float(p_val) < alpha,
    )
