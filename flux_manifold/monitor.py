"""Monitoring / logging utilities – export trace data to JSON for analysis."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def trace_to_json(trace: dict, label: str = "") -> dict:
    """Convert a flux_flow_traced result to a JSON-serializable record."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "steps": trace["steps"],
        "converged": trace["converged"],
        "drift_trace": trace["drift_trace"],
        "final_drift": trace["drift_trace"][-1] if trace["drift_trace"] else None,
    }


def append_log(record: dict, path: str | Path = "logs/flux_trace.jsonl") -> None:
    """Append a JSON record to a newline-delimited JSON log file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def export_benchmark(results: dict, path: str | Path = "logs/benchmark.json") -> None:
    """Export benchmark results dict to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Make numpy types serializable
    def _default(o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"Not serializable: {type(o)}")

    with open(p, "w") as fh:
        json.dump(results, fh, indent=2, default=_default)
