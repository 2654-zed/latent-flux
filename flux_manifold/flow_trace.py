"""Flow Trace — structured error diagnostics for non-converging pipelines.

When a Latent Flux pipeline fails to converge, the Flow Trace captures
exactly which semantic topology trapped execution, analogous to a Python
stack trace but for continuous optimization flows.

A Flow Trace records:
  - Which operator in the pipeline trapped execution
  - The drift history showing convergence stalls
  - Per-state convergence status for superpositions
  - Topology analysis: plateaus, oscillations, entropy stagnation
  - The full pipeline context (source location, operator chain)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrappedState:
    """A single state that failed to converge."""
    index: int
    final_drift: float
    stall_step: int          # step where drift stopped decreasing meaningfully
    total_steps: int
    trap_type: str           # 'local_minimum', 'saddle_oscillation', 'boundary_stall', 'plateau'


@dataclass
class FlowTraceEntry:
    """One entry in the Flow Trace — one operator's execution record."""
    stage_index: int         # 0-based position in the pipeline
    stage_count: int         # total stages in pipeline
    operator: str            # e.g. 'flow', 'squeeze', 'commit'
    status: str              # 'ok', 'non_convergent', 'error'
    message: str             # human-readable summary
    trapped_states: list[TrappedState] = field(default_factory=list)
    n_converged: int = 0
    n_total: int = 0
    mean_drift: float = 0.0
    target_tol: float = 0.0
    entropy_before: float | None = None
    entropy_after: float | None = None
    steps_used: int = 0
    max_steps: int = 0
    drift_plateau_detected: bool = False


class FlowTrace:
    """Accumulates trace entries across a pipeline evaluation."""

    def __init__(self) -> None:
        self.entries: list[FlowTraceEntry] = []
        self.source_line: str = ""
        self._warnings: list[str] = []

    def add(self, entry: FlowTraceEntry) -> None:
        self.entries.append(entry)
        if entry.status != "ok":
            self._warnings.append(entry.message)

    @property
    def has_failures(self) -> bool:
        return any(e.status != "ok" for e in self.entries)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def format(self) -> str:
        """Format the Flow Trace as a human-readable diagnostic."""
        if not self.has_failures:
            return ""

        lines: list[str] = []
        lines.append("══ FLOW TRACE ═══════════════════════════════════════════")
        if self.source_line:
            lines.append(f"  Pipeline: {self.source_line}")
        lines.append("")

        for entry in self.entries:
            if entry.status == "ok":
                continue

            stage_label = f"Stage {entry.stage_index + 1}/{entry.stage_count}"
            op_display = _OP_DISPLAY.get(entry.operator, entry.operator)
            lines.append(f"  {stage_label}: {op_display}")
            lines.append(f"  {'─' * 53}")

            if entry.status == "non_convergent":
                trapped_count = entry.n_total - entry.n_converged
                lines.append(
                    f"  Status: NON-CONVERGENT "
                    f"({trapped_count}/{entry.n_total} states trapped)"
                )
                lines.append(
                    f"  Steps: {entry.steps_used}/{entry.max_steps} (max reached)"
                )
                lines.append("")

                # Trapped state details (top 5)
                if entry.trapped_states:
                    lines.append("  Trapped States:")
                    for ts in entry.trapped_states[:5]:
                        lines.append(
                            f"    State #{ts.index}:  "
                            f"drift={ts.final_drift:.4f}  "
                            f"stuck at step {ts.stall_step}  "
                            f"({_TRAP_LABELS[ts.trap_type]})"
                        )
                    remaining = len(entry.trapped_states) - 5
                    if remaining > 0:
                        lines.append(f"    ... and {remaining} more")
                    lines.append("")

                # Topology analysis
                lines.append("  Topology Analysis:")
                lines.append(
                    f"    Mean drift: {entry.mean_drift:.4f} "
                    f"(target: {entry.target_tol})"
                )
                if entry.entropy_before is not None and entry.entropy_after is not None:
                    delta = entry.entropy_before - entry.entropy_after
                    label = "negligible" if abs(delta) < 0.1 else f"{delta:+.2f}"
                    lines.append(
                        f"    Entropy: {entry.entropy_before:.2f} bits "
                        f"→ {entry.entropy_after:.2f} bits ({label} reduction)"
                    )
                if entry.drift_plateau_detected:
                    lines.append(
                        "    Drift plateau detected: states stalled for 100+ steps"
                    )
                lines.append(
                    f"  Converged: {entry.n_converged}/{entry.n_total} states "
                    f"({_pct(entry.n_converged, entry.n_total)})"
                )
            elif entry.status == "error":
                lines.append(f"  Status: ERROR")
                lines.append(f"  {entry.message}")

            lines.append("")

        lines.append("═════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ── Trace analysis helpers ────────────────────────────────────────

def analyze_convergence(
    trace_dict: dict,
    tol: float,
) -> FlowTraceEntry:
    """Analyze a flux_flow_traced or flux_flow_traced_batch result.

    Args:
        trace_dict: result from flux_flow_traced / flux_flow_traced_batch
        tol: convergence tolerance used for the flow

    Returns:
        FlowTraceEntry with full diagnostics
    """
    # Batch case
    if "converged_states" in trace_dict:
        return _analyze_batch(trace_dict, tol)
    # Single case
    return _analyze_single(trace_dict, tol)


def _analyze_single(trace: dict, tol: float) -> FlowTraceEntry:
    converged = trace.get("converged", False)
    steps = trace.get("steps", 0)
    drift_trace = trace.get("drift_trace", [])
    max_steps_val = len(drift_trace) if drift_trace else steps

    if converged:
        return FlowTraceEntry(
            stage_index=0, stage_count=1,
            operator="flow", status="ok",
            message="Converged",
            n_converged=1, n_total=1,
            mean_drift=drift_trace[-1] if drift_trace else 0.0,
            target_tol=tol, steps_used=steps, max_steps=max_steps_val,
        )

    final_drift = drift_trace[-1] if drift_trace else float("inf")
    stall_step = _detect_stall(drift_trace)
    trap_type = _classify_trap(drift_trace, stall_step)

    return FlowTraceEntry(
        stage_index=0, stage_count=1,
        operator="flow", status="non_convergent",
        message=f"Flow did not converge: drift={final_drift:.4f} > tol={tol}",
        trapped_states=[TrappedState(
            index=0, final_drift=final_drift,
            stall_step=stall_step, total_steps=steps,
            trap_type=trap_type,
        )],
        n_converged=0, n_total=1,
        mean_drift=final_drift, target_tol=tol,
        steps_used=steps, max_steps=max_steps_val,
        drift_plateau_detected=(trap_type == "plateau"),
    )


def _analyze_batch(trace: dict, tol: float) -> FlowTraceEntry:
    converged_flags = trace.get("converged", np.array([]))
    steps_arr = trace.get("steps", np.array([]))
    total_steps = trace.get("total_steps", 0)
    converged_states = trace.get("converged_states", np.array([]))
    drift_traces = trace.get("drift_traces", None)

    n_total = len(converged_flags)
    n_converged = int(np.sum(converged_flags))

    if n_converged == n_total:
        return FlowTraceEntry(
            stage_index=0, stage_count=1,
            operator="flow", status="ok",
            message="All states converged",
            n_converged=n_converged, n_total=n_total,
            mean_drift=0.0, target_tol=tol,
            steps_used=total_steps, max_steps=total_steps,
        )

    # Analyze trapped states
    trapped: list[TrappedState] = []
    final_drifts: list[float] = []

    for i in range(n_total):
        if converged_flags[i]:
            continue

        if drift_traces is not None and drift_traces.ndim == 2:
            state_drifts = drift_traces[i]
            valid = state_drifts[~np.isnan(state_drifts)]
            final_drift = float(valid[-1]) if len(valid) > 0 else float("inf")
            stall_step = _detect_stall(valid.tolist())
            trap_type = _classify_trap(valid.tolist(), stall_step)
        else:
            final_drift = float("inf")
            stall_step = int(steps_arr[i]) if len(steps_arr) > i else 0
            trap_type = "plateau"

        final_drifts.append(final_drift)
        trapped.append(TrappedState(
            index=i, final_drift=final_drift,
            stall_step=stall_step,
            total_steps=int(steps_arr[i]) if len(steps_arr) > i else total_steps,
            trap_type=trap_type,
        ))

    # Sort by drift (worst first)
    trapped.sort(key=lambda t: -t.final_drift)

    mean_drift = float(np.mean(final_drifts)) if final_drifts else 0.0
    plateau = any(t.trap_type == "plateau" for t in trapped)

    return FlowTraceEntry(
        stage_index=0, stage_count=1,
        operator="flow", status="non_convergent",
        message=(
            f"Flow non-convergent: {len(trapped)}/{n_total} states trapped, "
            f"mean drift={mean_drift:.4f}"
        ),
        trapped_states=trapped,
        n_converged=n_converged, n_total=n_total,
        mean_drift=mean_drift, target_tol=tol,
        steps_used=total_steps, max_steps=total_steps,
        drift_plateau_detected=plateau,
    )


def _detect_stall(drift_trace: list[float], window: int = 50) -> int:
    """Find the step where drift stopped decreasing meaningfully."""
    if len(drift_trace) < window:
        return 0
    for i in range(window, len(drift_trace)):
        recent = drift_trace[i - window:i]
        if len(recent) < 2:
            continue
        improvement = recent[0] - recent[-1]
        if improvement < 1e-6:
            return i - window
    return len(drift_trace)


def _classify_trap(drift_trace: list[float], stall_step: int) -> str:
    """Classify what kind of topology trapped the state."""
    if len(drift_trace) < 10:
        return "plateau"

    tail = drift_trace[-min(50, len(drift_trace)):]

    # Check for oscillation: sign changes in drift delta
    if len(tail) > 5:
        deltas = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        sign_changes = sum(
            1 for i in range(len(deltas) - 1)
            if deltas[i] * deltas[i + 1] < 0
        )
        if sign_changes > len(deltas) * 0.4:
            return "saddle_oscillation"

    # Check for plateau: drift barely changes
    if len(tail) > 2:
        spread = max(tail) - min(tail)
        if spread < 0.01:
            return "local_minimum"

    # Check for very slow decrease (boundary stall)
    if stall_step > 0 and stall_step < len(drift_trace) * 0.8:
        return "boundary_stall"

    return "plateau"


# ── Display constants ─────────────────────────────────────────────

_OP_DISPLAY = {
    "flow": "⟼ (FluxManifold flow)",
    "squeeze": "∇↓ (DimensionalSqueeze)",
    "equiv": "≅ (DriftEquivalence)",
    "commit": "↓! (CommitmentSink)",
    "cascade": "⇑ (AbstractionCascade)",
    "fold": "◉ (FoldReference)",
    "superpose": "∑_ψ (SuperpositionTensor)",
}

_TRAP_LABELS = {
    "local_minimum": "local minimum trap",
    "saddle_oscillation": "saddle point oscillation",
    "boundary_stall": "boundary stall",
    "plateau": "drift plateau",
}


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{100 * n / total:.0f}%"
