"""Problem generators for the Latent Flux Proof Test Suite.

Generates geometry-native problems for comparing translation cost,
expressibility, cognitive load, emergence, and irreducibility
between Python (numpy) and Latent Flux.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Test 1 — Multi-Constraint Soft Optimization
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Constraint:
    """A single soft geometric constraint."""
    ctype: str  # 'proximity', 'orthogonality', 'norm_bound', 'subspace'
    target: np.ndarray | None = None
    radius: float = 0.0
    reference: np.ndarray | None = None
    tolerance: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    basis: np.ndarray | None = None


@dataclass
class MultiConstraintProblem:
    """A multi-constraint soft optimization problem in R^d."""
    problem_id: int
    d: int
    constraints: list[Constraint]
    seed: int

    def composite_target(self) -> np.ndarray:
        targets = [c.target for c in self.constraints
                   if c.ctype == 'proximity' and c.target is not None]
        if targets:
            return np.mean(targets, axis=0).astype(np.float32)
        return np.zeros(self.d, dtype=np.float32)

    def verify(self, v: np.ndarray, scale: float = 2.0) -> dict:
        satisfied = []
        details = []
        for c in self.constraints:
            if c.ctype == 'proximity':
                d = float(np.linalg.norm(v - c.target))
                ok = d <= c.radius * scale
                details.append(f"prox: dist={d:.4f} vs r={c.radius:.4f}*{scale}")
            elif c.ctype == 'orthogonality':
                p = abs(float(np.dot(v, c.reference)))
                ok = p <= c.tolerance * scale
                details.append(f"orth: proj={p:.4f} vs tol={c.tolerance:.4f}*{scale}")
            elif c.ctype == 'norm_bound':
                n = float(np.linalg.norm(v))
                ok = c.lower / scale <= n <= c.upper * scale
                details.append(f"norm: {n:.4f} in [{c.lower/scale:.2f}, {c.upper*scale:.2f}]")
            elif c.ctype == 'subspace':
                proj = c.basis.T @ (c.basis @ v)
                r = float(np.linalg.norm(v - proj))
                ok = r <= c.tolerance * scale
                details.append(f"sub: res={r:.4f} vs tol={c.tolerance:.4f}*{scale}")
            else:
                ok = False
                details.append(f"unknown constraint type: {c.ctype}")
            satisfied.append(ok)
        return {
            'satisfied': satisfied,
            'n_satisfied': sum(satisfied),
            'n_total': len(satisfied),
            'details': details,
        }

    def constraint_type_set(self) -> set[str]:
        return {c.ctype for c in self.constraints}


def _make_constraint(ctype: str, d: int, rng: np.random.Generator) -> Constraint:
    if ctype == 'proximity':
        return Constraint(
            ctype='proximity',
            target=rng.standard_normal(d).astype(np.float32) * 0.5,
            radius=float(rng.uniform(0.3, 0.8)),
        )
    elif ctype == 'orthogonality':
        ref = rng.standard_normal(d).astype(np.float32)
        ref /= np.linalg.norm(ref)
        return Constraint(
            ctype='orthogonality',
            reference=ref,
            tolerance=float(rng.uniform(0.05, 0.2)),
        )
    elif ctype == 'norm_bound':
        return Constraint(
            ctype='norm_bound',
            lower=float(rng.uniform(0.4, 0.8)),
            upper=float(rng.uniform(1.0, 1.8)),
        )
    elif ctype == 'subspace':
        k = max(2, d // 4)
        raw = rng.standard_normal((d, k)).astype(np.float32)
        Q, _ = np.linalg.qr(raw)
        return Constraint(
            ctype='subspace',
            basis=Q[:, :k].T.astype(np.float32),
            tolerance=float(rng.uniform(0.1, 0.4)),
        )
    raise ValueError(f"Unknown constraint type: {ctype}")


def generate_translation_suite(seed: int = 42) -> list[MultiConstraintProblem]:
    """Generate 10 multi-constraint problems of increasing complexity."""
    configs = [
        (4,  2, ['proximity', 'proximity']),
        (8,  2, ['proximity', 'proximity']),
        (12, 3, ['proximity', 'proximity', 'orthogonality']),
        (16, 3, ['proximity', 'orthogonality', 'orthogonality']),
        (20, 4, ['proximity', 'proximity', 'orthogonality', 'norm_bound']),
        (24, 4, ['proximity', 'orthogonality', 'norm_bound', 'norm_bound']),
        (32, 5, ['proximity', 'proximity', 'orthogonality', 'norm_bound', 'subspace']),
        (40, 5, ['proximity', 'orthogonality', 'norm_bound', 'subspace', 'subspace']),
        (48, 6, ['proximity', 'proximity', 'orthogonality', 'orthogonality', 'norm_bound', 'subspace']),
        (64, 6, ['proximity', 'proximity', 'orthogonality', 'norm_bound', 'subspace', 'subspace']),
    ]
    problems = []
    for i, (d, nc, ctypes) in enumerate(configs):
        rng = np.random.default_rng(seed + i)
        constraints = [_make_constraint(ct, d, rng) for ct in ctypes]
        problems.append(MultiConstraintProblem(
            problem_id=i, d=d, constraints=constraints, seed=seed + i,
        ))
    return problems


# ═══════════════════════════════════════════════════════════════════
# Test 2 — Attractor Basin Interference
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AttractorBasinProblem:
    """Multi-basin landscape for testing superposition coupling."""
    problem_id: int
    d: int
    centers: list[np.ndarray]
    strengths: list[float]
    widths: list[float]
    target: np.ndarray          # desired solution point
    n_candidates: int
    seed: int

    def quality(self, v: np.ndarray) -> float:
        """Quality = negative distance to target (higher = better)."""
        return -float(np.linalg.norm(v - self.target))

    def potential(self, x: np.ndarray) -> float:
        """Multi-basin potential: sum of Gaussians."""
        V = 0.0
        for c, s, w in zip(self.centers, self.strengths, self.widths):
            V -= s * np.exp(-np.linalg.norm(x - c) ** 2 / (2 * w ** 2))
        return float(V)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of multi-basin potential."""
        grad = np.zeros_like(x)
        for c, s, w in zip(self.centers, self.strengths, self.widths):
            diff = x - c
            r2 = np.dot(diff, diff)
            grad += s * np.exp(-r2 / (2 * w ** 2)) * diff / (w ** 2)
        return grad


def generate_expressibility_suite(
    n_instances: int = 50, d: int = 16, seed: int = 42
) -> list[AttractorBasinProblem]:
    """Generate attractor basin problems for Test 2."""
    problems = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed + i * 7)
        n_basins = 3
        centers = [rng.standard_normal(d).astype(np.float32) * 1.5
                    for _ in range(n_basins)]
        strengths = [float(rng.uniform(0.5, 2.0)) for _ in range(n_basins)]
        widths = [float(rng.uniform(0.3, 1.0)) for _ in range(n_basins)]
        # Target: weighted centroid biased toward strongest basin
        wt = np.array(strengths)
        wt /= wt.sum()
        target = sum(w * c for w, c in zip(wt, centers)).astype(np.float32)
        problems.append(AttractorBasinProblem(
            problem_id=i, d=d, centers=centers,
            strengths=strengths, widths=widths,
            target=target, n_candidates=20, seed=seed + i * 7,
        ))
    return problems


# ═══════════════════════════════════════════════════════════════════
# Test 3 / Meta — Geometry-Native Problems
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GeometryProblem:
    """A geometry-native problem for solver comparison."""
    problem_id: int
    problem_type: str   # 'interpolation', 'constraint', 'competition', 'projection'
    d: int
    params: dict        # type-specific parameters (numpy arrays inside)
    seed: int

    def quality(self, v: np.ndarray) -> float:
        if self.problem_type == 'interpolation':
            target = (self.params['a'] + self.params['b']) / 2.0
            return -float(np.linalg.norm(v - target))
        elif self.problem_type == 'constraint':
            return -sum(
                max(0, float(np.linalg.norm(v - t)) - r)
                for t, r in zip(self.params['targets'], self.params['radii'])
            )
        elif self.problem_type == 'competition':
            d1 = float(np.linalg.norm(v - self.params['a1']))
            d2 = float(np.linalg.norm(v - self.params['a2']))
            return -abs(d1 - d2)  # best when equidistant
        elif self.problem_type == 'projection':
            basis = self.params['basis']
            proj = basis.T @ (basis @ v)
            return -float(np.linalg.norm(v - proj))
        return 0.0


def _make_geometry_problem(pid: int, ptype: str, d: int, seed: int) -> GeometryProblem:
    rng = np.random.default_rng(seed)
    if ptype == 'interpolation':
        params = {
            'a': rng.standard_normal(d).astype(np.float32),
            'b': rng.standard_normal(d).astype(np.float32),
        }
    elif ptype == 'constraint':
        n_targets = rng.integers(2, 4)
        params = {
            'targets': [rng.standard_normal(d).astype(np.float32) * 0.5
                        for _ in range(n_targets)],
            'radii': [float(rng.uniform(0.3, 0.8)) for _ in range(n_targets)],
        }
    elif ptype == 'competition':
        a1 = rng.standard_normal(d).astype(np.float32)
        params = {'a1': a1, 'a2': -a1}
    elif ptype == 'projection':
        k = max(2, d // 3)
        raw = rng.standard_normal((d, k)).astype(np.float32)
        Q, _ = np.linalg.qr(raw)
        params = {
            'basis': Q[:, :k].T.astype(np.float32),
            'point': rng.standard_normal(d).astype(np.float32),
        }
    else:
        raise ValueError(f"Unknown problem type: {ptype}")
    return GeometryProblem(problem_id=pid, problem_type=ptype, d=d,
                            params=params, seed=seed)


def generate_cognitive_suite(seed: int = 42) -> list[GeometryProblem]:
    """Generate 20 problems for Test 3 (Cognitive Load)."""
    types = ['interpolation', 'constraint', 'competition', 'projection']
    problems = []
    for i in range(20):
        ptype = types[i % 4]
        d = 8 if i < 10 else 16
        problems.append(_make_geometry_problem(i, ptype, d, seed + i * 3))
    return problems


def generate_meta_suite(seed: int = 42) -> list[GeometryProblem]:
    """Generate 50 problems for Meta Test."""
    categories = ['interpolation', 'constraint', 'competition',
                   'projection', 'interpolation']
    problems = []
    for i in range(50):
        ptype = categories[i % 5]
        d = [8, 12, 16, 20, 24][i % 5]
        problems.append(_make_geometry_problem(i, ptype, d, seed + i * 11))
    return problems


# ═══════════════════════════════════════════════════════════════════
# Test 4 — Hidden Structure Problems
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HiddenStructureProblem:
    """Problem with known hidden geometric structure."""
    problem_id: int
    problem_type: str   # 'hierarchy', 'manifold', 'competition'
    data: np.ndarray    # (N, d) point cloud or initial state
    ground_truth: dict  # known structure to be revealed
    d: int
    seed: int


def generate_hierarchical_points(d: int = 32, seed: int = 42) -> HiddenStructureProblem:
    """Points with 3-level hierarchical cluster structure."""
    rng = np.random.default_rng(seed)
    level0 = rng.standard_normal((3, d)).astype(np.float32) * 2.0
    points, labels_l0, labels_l1 = [], [], []
    for i, center in enumerate(level0):
        sub_centers = [center + rng.standard_normal(d).astype(np.float32) * 0.5
                       for _ in range(3)]
        for j, sc in enumerate(sub_centers):
            for _ in range(5):
                points.append(sc + rng.standard_normal(d).astype(np.float32) * 0.1)
                labels_l0.append(i)
                labels_l1.append(i * 3 + j)
    return HiddenStructureProblem(
        problem_id=0, problem_type='hierarchy',
        data=np.array(points, dtype=np.float32),
        ground_truth={
            'n_levels': 3, 'n_top_clusters': 3,
            'n_sub_clusters': 9, 'n_points': 45,
            'labels_level0': labels_l0, 'labels_level1': labels_l1,
        },
        d=d, seed=seed,
    )


def generate_hidden_manifold(
    d: int = 16, intrinsic_d: int = 4, n: int = 100, seed: int = 42
) -> HiddenStructureProblem:
    """Points on a low-dimensional manifold embedded in high-d space."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, intrinsic_d)).astype(np.float32)
    z = rng.standard_normal((n, intrinsic_d)).astype(np.float32)
    X = (z @ A.T + rng.standard_normal((n, d)).astype(np.float32) * 0.05)
    return HiddenStructureProblem(
        problem_id=1, problem_type='manifold',
        data=X.astype(np.float32),
        ground_truth={
            'intrinsic_dim': intrinsic_d,
            'embedding': A,
            'latent_coords': z,
        },
        d=d, seed=seed,
    )


def generate_attractor_competition(d: int = 8, seed: int = 42) -> HiddenStructureProblem:
    """Flow with two competing attractors of equal strength."""
    rng = np.random.default_rng(seed)
    a1 = rng.standard_normal(d).astype(np.float32)
    a1 /= np.linalg.norm(a1)
    a2 = -a1
    s0 = rng.standard_normal(d).astype(np.float32) * 0.1
    return HiddenStructureProblem(
        problem_id=2, problem_type='competition',
        data=s0.reshape(1, -1),
        ground_truth={
            'a1': a1, 'a2': a2,
            'bias': float(np.dot(s0, a1)),
            'expected_winner': 'a1' if np.dot(s0, a1) > 0 else 'a2',
        },
        d=d, seed=seed,
    )


def generate_emergence_suite(seed: int = 42) -> list[HiddenStructureProblem]:
    """Generate all 3 problems for Test 4."""
    return [
        generate_hierarchical_points(d=32, seed=seed),
        generate_hidden_manifold(d=16, intrinsic_d=4, n=100, seed=seed + 1),
        generate_attractor_competition(d=8, seed=seed + 2),
    ]


# ═══════════════════════════════════════════════════════════════════
# Test 5 — Irreducibility (10 problem types)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IrreducibilityProblem:
    """Problem for testing primitive sufficiency."""
    problem_id: int
    name: str
    description: str
    d: int
    params: dict
    seed: int

    def quality(self, v: np.ndarray) -> float:
        """Problem-specific quality metric (higher = better)."""
        if self.name == 'single_attractor':
            return -float(np.linalg.norm(v - self.params['target']))
        elif self.name == 'competing_attractors':
            d1 = np.linalg.norm(v - self.params['a1'])
            d2 = np.linalg.norm(v - self.params['a2'])
            return -float(min(d1, d2))
        elif self.name == 'manifold_projection':
            B = self.params['basis']
            proj = B.T @ (B @ v)
            return -float(np.linalg.norm(v - proj))
        elif self.name == 'hierarchical_cluster':
            centers = self.params['centers']
            return -float(min(np.linalg.norm(v - c) for c in centers))
        elif self.name == 'analogy':
            expected = self.params['a'] - self.params['b'] + self.params['c']
            return -float(np.linalg.norm(v - expected))
        elif self.name == 'soft_constraint':
            return -sum(
                max(0, float(np.linalg.norm(v - t)) - r)
                for t, r in zip(self.params['targets'], self.params['radii'])
            )
        elif self.name == 'geodesic_interp':
            target = self.params['midpoint']
            return -float(np.linalg.norm(v - target))
        elif self.name == 'dim_estimation':
            return float(self.params['intrinsic_dim'])
        elif self.name == 'neighborhood_map':
            refs = self.params['references']
            return -float(np.mean([np.linalg.norm(v - r) for r in refs]))
        elif self.name == 'adversarial_flow':
            return -float(np.linalg.norm(v - self.params['target']))
        return 0.0


def generate_irreducibility_suite(seed: int = 42) -> list[IrreducibilityProblem]:
    """Generate all 10 problems for Test 5."""
    rng = np.random.default_rng(seed)
    problems = []

    # 1. Single attractor convergence
    d = 8
    problems.append(IrreducibilityProblem(
        0, 'single_attractor', 'Flow to a single target in R^8', d,
        {'target': rng.standard_normal(d).astype(np.float32) * 0.5,
         's0': rng.standard_normal(d).astype(np.float32)}, seed))

    # 2. Competing attractor resolution
    d = 8
    a1 = rng.standard_normal(d).astype(np.float32)
    a1 /= np.linalg.norm(a1)
    problems.append(IrreducibilityProblem(
        1, 'competing_attractors', 'Two attractors, find stable point', d,
        {'a1': a1, 'a2': -a1,
         's0': rng.standard_normal(d).astype(np.float32) * 0.3}, seed + 1))

    # 3. Constrained manifold projection
    d = 16
    k = 4
    raw = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(raw)
    problems.append(IrreducibilityProblem(
        2, 'manifold_projection', 'Project onto 4D manifold in R^16', d,
        {'basis': Q[:, :k].T.astype(np.float32),
         'point': rng.standard_normal(d).astype(np.float32)}, seed + 2))

    # 4. Hierarchical clustering
    d = 32
    centers = [rng.standard_normal(d).astype(np.float32) * 2 for _ in range(4)]
    points = []
    for c in centers:
        for _ in range(10):
            points.append(c + rng.standard_normal(d).astype(np.float32) * 0.3)
    problems.append(IrreducibilityProblem(
        3, 'hierarchical_cluster', 'Cluster 40 points into 4 groups', d,
        {'centers': centers, 'points': np.array(points, dtype=np.float32)}, seed + 3))

    # 5. Continuous analogy completion (a:b :: c:?)
    d = 16
    a = rng.standard_normal(d).astype(np.float32)
    b = rng.standard_normal(d).astype(np.float32)
    c = rng.standard_normal(d).astype(np.float32)
    expected = a - b + c
    problems.append(IrreducibilityProblem(
        4, 'analogy', 'a:b :: c:? — find the vector completing the analogy', d,
        {'a': a, 'b': b, 'c': c, 'expected': expected.astype(np.float32)}, seed + 4))

    # 6. Soft constraint satisfaction
    d = 16
    n_targets = 3
    targets = [rng.standard_normal(d).astype(np.float32) * 0.5 for _ in range(n_targets)]
    radii = [float(rng.uniform(0.3, 0.8)) for _ in range(n_targets)]
    problems.append(IrreducibilityProblem(
        5, 'soft_constraint', 'Satisfy 3 soft proximity constraints', d,
        {'targets': targets, 'radii': radii}, seed + 5))

    # 6. State interpolation along geodesic
    d = 8
    p1 = rng.standard_normal(d).astype(np.float32)
    p2 = rng.standard_normal(d).astype(np.float32)
    mid = ((p1 + p2) / 2).astype(np.float32)
    problems.append(IrreducibilityProblem(
        6, 'geodesic_interp', 'Find midpoint between two states', d,
        {'p1': p1, 'p2': p2, 'midpoint': mid}, seed + 6))

    # 8. Dimensionality estimation
    d = 32
    intrinsic = 8
    A = rng.standard_normal((d, intrinsic)).astype(np.float32)
    z = rng.standard_normal((50, intrinsic)).astype(np.float32)
    X = (z @ A.T + rng.standard_normal((50, d)).astype(np.float32) * 0.01)
    problems.append(IrreducibilityProblem(
        7, 'dim_estimation', 'Estimate intrinsic dimension of data cloud', d,
        {'data': X.astype(np.float32), 'intrinsic_dim': intrinsic}, seed + 7))

    # 9. Semantic neighborhood mapping
    d = 16
    refs = [rng.standard_normal(d).astype(np.float32) for _ in range(5)]
    query = rng.standard_normal(d).astype(np.float32)
    problems.append(IrreducibilityProblem(
        8, 'neighborhood_map', 'Find position equidistant to 5 references', d,
        {'references': refs, 'query': query}, seed + 8))

    # 10. Flow under adversarial perturbation
    d = 8
    problems.append(IrreducibilityProblem(
        9, 'adversarial_flow', 'Converge despite stochastic perturbation', d,
        {'target': rng.standard_normal(d).astype(np.float32) * 0.5,
         's0': rng.standard_normal(d).astype(np.float32),
         'noise_scale': 0.1}, seed + 9))

    return problems
