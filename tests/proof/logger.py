"""Solution attempt logger for blind-trial tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SolutionAttempt:
    problem_id: int
    language: str               # 'python' or 'lf'
    problem_type: str
    corrections: list[str] = field(default_factory=list)
    success: bool = False
    subjective_difficulty: int = 3   # 1=trivial, 5=hard
    elapsed_ms: float = 0.0
    error_message: str = ""


class AttemptLogger:
    """Accumulates solution attempts for analysis."""

    def __init__(self) -> None:
        self.attempts: list[SolutionAttempt] = []

    def log(self, attempt: SolutionAttempt) -> None:
        self.attempts.append(attempt)

    def _stats(self, group: list[SolutionAttempt]) -> dict:
        if not group:
            return {'count': 0, 'success_rate': 0.0,
                    'mean_corrections': 0.0, 'mean_difficulty': 0.0}
        corr = [len(a.corrections) for a in group]
        return {
            'count': len(group),
            'success_rate': sum(1 for a in group if a.success) / len(group),
            'mean_corrections': sum(corr) / len(corr),
            'total_corrections': sum(corr),
            'mean_difficulty': sum(a.subjective_difficulty for a in group) / len(group),
            'abandoned': sum(1 for a in group if not a.success),
        }

    def summary(self) -> dict:
        py = [a for a in self.attempts if a.language == 'python']
        lf = [a for a in self.attempts if a.language == 'lf']
        py_stats = self._stats(py)
        lf_stats = self._stats(lf)
        lf_corr = lf_stats.get('mean_corrections', 0.01)
        ratio = py_stats.get('mean_corrections', 0) / max(lf_corr, 0.01)
        return {'python': py_stats, 'lf': lf_stats, 'cognitive_load_ratio': ratio}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [{
            'problem_id': a.problem_id, 'language': a.language,
            'problem_type': a.problem_type, 'n_corrections': len(a.corrections),
            'corrections': a.corrections, 'success': a.success,
            'subjective_difficulty': a.subjective_difficulty,
            'elapsed_ms': a.elapsed_ms, 'error_message': a.error_message,
        } for a in self.attempts]
        with open(path, 'w') as f:
            json.dump({'attempts': data, 'summary': self.summary()}, f, indent=2)
