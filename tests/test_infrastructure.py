"""Tests for the three infrastructure features:

1. Flow Traces — structured error diagnostics for non-convergence
2. I/O — load_json, load_csv, to_manifold built-in functions
3. Import System — native module linker with relative resolution
"""

import os
import json
import numpy as np
import pytest

from flux_manifold.parser import (
    run, run_file, EvalContext, parse_program, evaluate_program,
    _json_to_array,
)
from flux_manifold.flow_trace import (
    FlowTrace, FlowTraceEntry, TrappedState,
    analyze_convergence, _detect_stall, _classify_trap,
)
from flux_manifold.superposition import SuperpositionTensor

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ══════════════════════════════════════════════════════════════
# 1. FLOW TRACES
# ══════════════════════════════════════════════════════════════

class TestFlowTraceAnalysis:
    """Test flow trace analysis on synthetic trace dicts."""

    def test_converged_single_returns_ok(self):
        trace = {
            "converged_state": np.zeros(3),
            "steps": 10,
            "converged": True,
            "drift_trace": [1.0, 0.5, 0.2, 0.05, 0.001],
            "path": [],
        }
        entry = analyze_convergence(trace, tol=0.01)
        assert entry.status == "ok"
        assert entry.n_converged == 1
        assert entry.n_total == 1

    def test_non_converged_single_returns_failure(self):
        trace = {
            "converged_state": np.ones(3),
            "steps": 500,
            "converged": False,
            "drift_trace": [1.0] * 500,
            "path": [],
        }
        entry = analyze_convergence(trace, tol=0.001)
        assert entry.status == "non_convergent"
        assert entry.n_converged == 0
        assert entry.n_total == 1
        assert len(entry.trapped_states) == 1
        assert entry.trapped_states[0].trap_type in (
            "local_minimum", "saddle_oscillation", "boundary_stall", "plateau"
        )

    def test_batch_partial_convergence(self):
        trace = {
            "converged_states": np.zeros((5, 3)),
            "steps": np.array([100, 500, 50, 500, 200]),
            "converged": np.array([True, False, True, False, True]),
            "total_steps": 500,
            "drift_traces": np.full((5, 500), np.nan),
        }
        # Fill some drift traces
        for i in range(5):
            trace["drift_traces"][i, :int(trace["steps"][i])] = np.linspace(1.0, 0.5 if trace["converged"][i] else 0.8, int(trace["steps"][i]))

        entry = analyze_convergence(trace, tol=0.001)
        assert entry.status == "non_convergent"
        assert entry.n_converged == 3
        assert entry.n_total == 5
        assert len(entry.trapped_states) == 2

    def test_batch_all_converged(self):
        trace = {
            "converged_states": np.zeros((3, 4)),
            "steps": np.array([50, 30, 80]),
            "converged": np.array([True, True, True]),
            "total_steps": 100,
        }
        entry = analyze_convergence(trace, tol=0.01)
        assert entry.status == "ok"
        assert entry.n_converged == 3


class TestFlowTraceFormat:
    """Test flow trace formatting output."""

    def test_empty_trace_no_output(self):
        ft = FlowTrace()
        assert ft.format() == ""
        assert not ft.has_failures

    def test_failure_produces_readable_output(self):
        ft = FlowTrace()
        entry = FlowTraceEntry(
            stage_index=1, stage_count=4,
            operator="flow", status="non_convergent",
            message="Flow non-convergent: 3/5 states trapped",
            trapped_states=[
                TrappedState(index=2, final_drift=0.45, stall_step=300, total_steps=500, trap_type="local_minimum"),
                TrappedState(index=4, final_drift=0.81, stall_step=50, total_steps=500, trap_type="saddle_oscillation"),
            ],
            n_converged=2, n_total=5,
            mean_drift=0.63, target_tol=0.001,
            entropy_before=3.2, entropy_after=3.1,
            steps_used=500, max_steps=500,
            drift_plateau_detected=False,
        )
        ft.add(entry)
        output = ft.format()
        assert "FLOW TRACE" in output
        assert "NON-CONVERGENT" in output
        assert "3/5 states trapped" in output
        assert "State #2" in output
        assert "local minimum trap" in output
        assert "saddle point oscillation" in output
        assert "Entropy:" in output
        assert "40%" in output  # converged percentage

    def test_multiple_entries(self):
        ft = FlowTrace()
        ft.add(FlowTraceEntry(
            stage_index=0, stage_count=3, operator="flow",
            status="non_convergent", message="first issue",
            n_total=5, n_converged=3,
        ))
        ft.add(FlowTraceEntry(
            stage_index=2, stage_count=3, operator="commit",
            status="error", message="commitment failed",
        ))
        assert ft.has_failures
        output = ft.format()
        assert "Stage 1/3" in output
        assert "Stage 3/3" in output


class TestDetectStall:
    def test_constant_drift_stalls_early(self):
        drift = [1.0] * 200
        stall = _detect_stall(drift, window=50)
        assert stall <= 50

    def test_decreasing_drift_no_stall(self):
        drift = list(np.linspace(1.0, 0.0, 100))
        stall = _detect_stall(drift, window=50)
        assert stall >= 50


class TestClassifyTrap:
    def test_oscillation_detected(self):
        # Create oscillating drift
        drift = [1.0 + 0.1 * ((-1) ** i) for i in range(100)]
        trap = _classify_trap(drift, stall_step=0)
        assert trap == "saddle_oscillation"

    def test_local_minimum_detected(self):
        # Flat at a nonzero value
        drift = [0.5] * 100
        trap = _classify_trap(drift, stall_step=0)
        assert trap == "local_minimum"


class TestFlowTraceIntegration:
    """Test flow traces emitted during actual pipeline evaluation."""

    def test_non_convergent_flow_emits_trace(self):
        messages = []
        ctx = EvalContext(
            max_steps=5, tol=1e-10,  # impossibly tight => non-convergence
            on_message=lambda m: messages.append(m),
        )
        # Flow that won't converge in 5 steps
        run("[0, 0] -> [100, 100]", ctx=ctx)
        # Should have emitted a flow trace
        assert any("FLOW TRACE" in m for m in messages)
        assert ctx.flow_trace.has_failures

    def test_converged_flow_no_trace(self):
        messages = []
        ctx = EvalContext(
            max_steps=500, tol=1.0,  # very loose => instant convergence
            on_message=lambda m: messages.append(m),
        )
        run("[0.5, 0.5] -> [0, 0]", ctx=ctx)
        assert not ctx.flow_trace.has_failures

    def test_superposition_flow_trace_has_entropy(self):
        messages = []
        ctx = EvalContext(
            max_steps=3, tol=1e-10,
            on_message=lambda m: messages.append(m),
        )
        run("∑_ψ [1, 0; 0, 1; 0.5, 0.5] ⟼ [0, 0]", ctx=ctx)
        if ctx.flow_trace.has_failures:
            entry = ctx.flow_trace.entries[0]
            assert entry.entropy_before is not None
            assert entry.entropy_after is not None


# ══════════════════════════════════════════════════════════════
# 2. I/O AND EXTERNAL STATE LOADING
# ══════════════════════════════════════════════════════════════

class TestJsonToArray:
    """Test the _json_to_array helper."""

    def test_list_of_numbers(self):
        result = _json_to_array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == np.float32

    def test_list_of_lists(self):
        result = _json_to_array([[1, 2], [3, 4]])
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_nested_dict_with_data_key(self):
        result = _json_to_array({"data": [[1, 2], [3, 4]]})
        assert result.shape == (2, 2)

    def test_list_of_dicts(self):
        result = _json_to_array([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}])
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_dict_with_numeric_values(self):
        result = _json_to_array({"a": 1.0, "b": 2.0, "c": 3.0})
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_scalar(self):
        result = _json_to_array(42)
        np.testing.assert_array_equal(result, [42.0])

    def test_empty_list(self):
        result = _json_to_array([])
        assert len(result) == 0


class TestLoadJsonBuiltin:
    """Test load_json() from Latent Flux expressions."""

    def test_load_json_list_of_lists(self):
        path = os.path.join(FIXTURES_DIR, "points.json")
        ctx = EvalContext()
        ctx._import_dirs.append(FIXTURES_DIR)
        result = run(f'load_json("{path}")', ctx=ctx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 3)
        np.testing.assert_allclose(result[0], [0.1, 0.2, 0.3], atol=1e-6)

    def test_load_json_nested_data_key(self):
        path = os.path.join(FIXTURES_DIR, "nested.json")
        ctx = EvalContext()
        result = run(f'load_json("{path}")', ctx=ctx)
        assert result.shape == (3, 2)

    def test_load_json_records(self):
        path = os.path.join(FIXTURES_DIR, "records.json")
        ctx = EvalContext()
        result = run(f'load_json("{path}")', ctx=ctx)
        assert result.shape == (3, 2)

    def test_load_json_to_manifold_pipeline(self):
        path = os.path.join(FIXTURES_DIR, "points.json")
        ctx = EvalContext()
        result = run(f'to_manifold(load_json("{path}"))', ctx=ctx)
        assert isinstance(result, SuperpositionTensor)
        assert result.n == 5
        assert result.d == 3


class TestLoadCsvBuiltin:
    """Test load_csv() from Latent Flux expressions."""

    def test_load_csv_with_header(self):
        path = os.path.join(FIXTURES_DIR, "data.csv")
        ctx = EvalContext()
        result = run(f'load_csv("{path}")', ctx=ctx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)  # 4 data rows, 3 columns
        np.testing.assert_allclose(result[0], [0.1, 0.2, 0.3], atol=1e-6)

    def test_load_csv_no_header(self):
        path = os.path.join(FIXTURES_DIR, "noheader.csv")
        ctx = EvalContext()
        result = run(f'load_csv("{path}", 0)', ctx=ctx)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0], atol=1e-6)

    def test_csv_to_manifold(self):
        path = os.path.join(FIXTURES_DIR, "data.csv")
        ctx = EvalContext()
        result = run(f'to_manifold(load_csv("{path}"))', ctx=ctx)
        assert isinstance(result, SuperpositionTensor)
        assert result.n == 4
        assert result.d == 3


class TestToManifold:
    """Test to_manifold() built-in."""

    def test_1d_to_superposition(self):
        ctx = EvalContext()
        result = run("to_manifold([1, 2, 3])", ctx=ctx)
        assert isinstance(result, SuperpositionTensor)
        assert result.n == 1
        assert result.d == 3

    def test_2d_to_superposition(self):
        ctx = EvalContext()
        result = run("to_manifold([1, 2; 3, 4; 5, 6])", ctx=ctx)
        assert isinstance(result, SuperpositionTensor)
        assert result.n == 3
        assert result.d == 2


# ══════════════════════════════════════════════════════════════
# 3. IMPORT SYSTEM
# ══════════════════════════════════════════════════════════════

class TestImportSystem:
    """Test native module linker with relative import resolution."""

    def test_import_from_same_directory(self):
        """importer.lf imports helper.lf from the same directory."""
        importer_path = os.path.join(FIXTURES_DIR, "importer.lf")
        ctx = EvalContext()
        result = run_file(importer_path, ctx=ctx)
        # helper.lf defines helper_origin and helper_target
        assert "helper_origin" in ctx.variables
        assert "helper_target" in ctx.variables
        # result should be the committed state
        assert isinstance(result, np.ndarray)

    def test_nested_import_with_relative_path(self):
        """sub/nested_importer.lf imports ../helper.lf."""
        nested_path = os.path.join(FIXTURES_DIR, "sub", "nested_importer.lf")
        ctx = EvalContext()
        result = run_file(nested_path, ctx=ctx)
        assert "helper_origin" in ctx.variables
        assert "nested_result" in ctx.variables

    def test_stdlib_import_from_anywhere(self):
        """Import stdlib/geometry from a non-stdlib directory."""
        ctx = EvalContext()
        source = 'import "stdlib/geometry"'
        from flux_manifold.parser import parse_program, evaluate_program
        program = parse_program(source)
        evaluate_program(program, ctx)
        # geometry.lf defines origin_2d, etc.
        assert "origin_2d" in ctx.variables

    def test_duplicate_import_is_noop(self):
        """Importing the same module twice doesn't re-execute."""
        ctx = EvalContext()
        source = '''import "stdlib/geometry"
import "stdlib/geometry"'''
        program = parse_program(source)
        evaluate_program(program, ctx)
        # Should succeed without error
        assert "origin_2d" in ctx.variables

    def test_import_nonexistent_raises(self):
        """Importing a nonexistent module raises FileNotFoundError."""
        ctx = EvalContext()
        source = 'import "nonexistent_module_xyz"'
        program = parse_program(source)
        with pytest.raises(FileNotFoundError):
            evaluate_program(program, ctx)

    def test_import_exposes_bindings(self):
        """Variables defined in imported module are accessible."""
        importer_path = os.path.join(FIXTURES_DIR, "importer.lf")
        ctx = EvalContext()
        run_file(importer_path, ctx=ctx)
        origin = ctx.get("helper_origin")
        np.testing.assert_array_equal(origin, [0, 0, 0])

    def test_import_dirs_stack_managed(self):
        """Import dirs stack is properly pushed and popped."""
        ctx = EvalContext()
        assert len(ctx._import_dirs) == 0
        importer_path = os.path.join(FIXTURES_DIR, "importer.lf")
        run_file(importer_path, ctx=ctx)
        # After run_file completes, stack should be empty again
        assert len(ctx._import_dirs) == 0
