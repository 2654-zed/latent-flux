"""Tests for the Latent Flux expression parser and evaluator."""

import numpy as np
import pytest
from flux_manifold.parser import (
    tokenize, parse, evaluate, run, EvalContext,
    LFVector, LFNumber, LFSymbol, LFFuncCall, LFPipeline, LFOp,
)
from flux_manifold.superposition import SuperpositionTensor


# ── Tokenizer ──────────────────────────────────────────────────────

class TestTokenizer:
    def test_unicode_operators(self):
        tokens = tokenize("∑_ψ ⟼ ∇↓ ≅ ↓! ⇑ ◉")
        ops = [t[1] for t in tokens]
        assert ops == ["superpose", "flow", "squeeze", "equiv", "commit", "cascade", "fold"]

    def test_ascii_aliases(self):
        tokens = tokenize("sum_psi -> squeeze ~= commit cascade fold")
        ops = [t[1] for t in tokens]
        assert ops == ["superpose", "flow", "squeeze", "equiv", "commit", "cascade", "fold"]

    def test_word_operators(self):
        tokens = tokenize("superpose flow squeeze equiv commit cascade fold")
        ops = [t[1] for t in tokens]
        assert ops == ["superpose", "flow", "squeeze", "equiv", "commit", "cascade", "fold"]

    def test_vector_tokens(self):
        tokens = tokenize("[1, 2, 3]")
        types = [t[0] for t in tokens]
        assert "LBRACKET" in types
        assert "RBRACKET" in types
        assert types.count("NUMBER") == 3

    def test_pipe_token(self):
        tokens = tokenize("x | y")
        assert ("PIPE", "|") in tokens

    def test_number_negative(self):
        tokens = tokenize("-3.14")
        assert ("NUMBER", "-3.14") in tokens

    def test_func_call_tokens(self):
        tokens = tokenize("random(10, 32)")
        assert ("IDENT", "random") in tokens
        assert ("LPAREN", "(") in tokens
        assert ("RPAREN", ")") in tokens

    def test_invalid_char(self):
        with pytest.raises(SyntaxError):
            tokenize("@#$")


# ── Parser ─────────────────────────────────────────────────────────

class TestParser:
    def test_simple_vector(self):
        ast = parse("[1, 2, 3]")
        assert isinstance(ast, LFPipeline)
        assert len(ast.stages) == 1
        assert isinstance(ast.stages[0], LFVector)
        assert ast.stages[0].values == [[1.0, 2.0, 3.0]]

    def test_matrix_vector(self):
        ast = parse("[1, 0; 0, 1]")
        vec = ast.stages[0]
        assert isinstance(vec, LFVector)
        assert len(vec.values) == 2
        assert vec.values[0] == [1.0, 0.0]
        assert vec.values[1] == [0.0, 1.0]

    def test_number(self):
        ast = parse("42")
        assert isinstance(ast.stages[0], LFNumber)
        assert ast.stages[0].value == 42.0

    def test_symbol(self):
        ast = parse("state")
        assert isinstance(ast.stages[0], LFSymbol)
        assert ast.stages[0].name == "state"

    def test_func_call(self):
        ast = parse("random(5, 8)")
        assert isinstance(ast.stages[0], LFFuncCall)
        assert ast.stages[0].name == "random"
        assert len(ast.stages[0].args) == 2

    def test_pipeline_with_pipe(self):
        ast = parse("[1, 2] | flow [0, 0] | commit")
        assert isinstance(ast, LFPipeline)
        assert len(ast.stages) >= 2

    def test_pipeline_unicode(self):
        ast = parse("∑_ψ [1, 0; 0, 1] ⟼ [0, 0]")
        assert isinstance(ast, LFPipeline)
        # Should have superpose op, then flow op
        assert any(isinstance(s, LFOp) and s.symbol == "superpose" for s in ast.stages)
        assert any(isinstance(s, LFOp) and s.symbol == "flow" for s in ast.stages)

    def test_empty_vector_raises(self):
        with pytest.raises(SyntaxError):
            parse("[]")


# ── Evaluator ──────────────────────────────────────────────────────

class TestEvaluator:
    def test_vector_literal(self):
        result = run("[1, 2, 3]")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1, 2, 3], atol=1e-5)

    def test_number_literal(self):
        result = run("42")
        assert result == 42.0

    def test_func_zeros(self):
        result = run("zeros(4)")
        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        np.testing.assert_allclose(result, 0)

    def test_func_ones(self):
        result = run("ones(3)")
        np.testing.assert_allclose(result, [1, 1, 1])

    def test_func_randn(self):
        result = run("randn(8)")
        assert len(result) == 8

    def test_func_random_superposition(self):
        result = run("random(5, 8)")
        assert isinstance(result, SuperpositionTensor)
        assert result.n == 5
        assert result.d == 8

    def test_variable_assignment(self):
        ctx = EvalContext()
        ctx.set("x", np.array([1.0, 2.0], dtype=np.float32))
        result = run("x", ctx=ctx)
        np.testing.assert_allclose(result, [1, 2])

    def test_flow_single_vector(self):
        """⟼: single vector flows toward attractor."""
        result = run("[5, 5] ⟼ [0, 0]")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0, 0], atol=0.2)

    def test_flow_pipe_syntax(self):
        """Pipe syntax for flow."""
        result = run("[3, 3] | flow [0, 0]")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0, 0], atol=0.2)

    def test_superpose(self):
        """∑_ψ creates a SuperpositionTensor."""
        result = run("∑_ψ [1, 0; 0, 1; 0.5, 0.5]")
        assert isinstance(result, SuperpositionTensor)
        assert result.n == 3

    def test_superpose_then_flow(self):
        """∑_ψ + ⟼: superposition states flow toward attractor."""
        result = run("∑_ψ [1, 0; 0, 1] ⟼ [0, 0]")
        assert isinstance(result, SuperpositionTensor)
        # After flow, states should be near [0, 0]
        for s in result.states:
            assert np.linalg.norm(s) < 0.3

    def test_squeeze_single(self):
        """∇↓ on a single vector truncates."""
        result = run("[1, 2, 3, 4, 5] | squeeze 3")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_commit(self):
        """↓! commits and is irreversible."""
        result = run("[1, 2, 3] | commit")
        np.testing.assert_allclose(result, [1, 2, 3])

    def test_cascade(self):
        """⇑ returns list of abstraction levels."""
        result = run("[1, 2, 3, 4] | cascade 2")
        assert isinstance(result, list)

    def test_fold(self):
        """◉ runs fold-reference check."""
        result = run("[1, 2, 3] | fold")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1, 2, 3])

    def test_equiv(self):
        """≅ returns equivalence info."""
        ctx = EvalContext()
        # Need to flow first so there's a trace
        result = run("[1, 1] ⟼ [0, 0] | ≅ 0.5", ctx=ctx)
        assert isinstance(result, dict)
        assert "quality" in result

    def test_full_pipeline(self):
        """Full pipeline: ∑_ψ → ⟼ → ◉ → ↓!"""
        result = run("∑_ψ [1, 0; 0, 1] ⟼ [0, 0] | ◉ | ↓!")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0, 0], atol=0.3)

    def test_ascii_full_pipeline(self):
        """Full pipeline with ASCII operators."""
        result = run("sum_psi [2, 0; 0, 2] -> [0, 0] | fold | commit")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0, 0], atol=0.3)

    def test_func_linspace(self):
        result = run("linspace(0, 1, 5)")
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        np.testing.assert_allclose(result[0], 0, atol=1e-5)
        np.testing.assert_allclose(result[-1], 1, atol=1e-5)

    def test_context_epsilon(self):
        """EvalContext epsilon affects flow."""
        ctx = EvalContext(epsilon=0.5)
        result = run("[10, 10] ⟼ [0, 0]", ctx=ctx)
        np.testing.assert_allclose(result, [0, 0], atol=1.0)

    def test_undefined_variable_error(self):
        with pytest.raises(NameError):
            run("undefined_var")

    def test_missing_flow_target_error(self):
        with pytest.raises((ValueError, TypeError)):
            run("[1, 2] | flow")

    def test_commit_emits_message(self):
        """↓! should call on_message callback."""
        messages = []
        ctx = EvalContext(on_message=lambda msg: messages.append(msg))
        run("[1, 2, 3] | commit", ctx=ctx)
        assert len(messages) == 1
        assert "Committed" in messages[0]

    def test_commit_superposition_emits_entropy(self):
        """↓! on superposition emits entropy message."""
        messages = []
        ctx = EvalContext(on_message=lambda msg: messages.append(msg))
        run("∑_ψ [1, 0; 0, 1] ⟼ [0, 0] | ↓!", ctx=ctx)
        assert any("Entropy" in m for m in messages)

    def test_nearest_neighbor_function(self):
        """nearest_neighbor(cities) returns state vector."""
        ctx = EvalContext()
        cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        ctx.set("cities", cities)
        result = run("nearest_neighbor(cities)", ctx=ctx)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4


# ── REPL commands ──────────────────────────────────────────────────

class TestREPLCommands:
    def test_handle_help(self):
        from flux_manifold.repl import _handle_command
        result = _handle_command(":help", EvalContext())
        assert "Operators" in result

    def test_handle_vars_empty(self):
        from flux_manifold.repl import _handle_command
        result = _handle_command(":vars", EvalContext())
        assert "no variables" in result

    def test_handle_vars_with_data(self):
        from flux_manifold.repl import _handle_command
        ctx = EvalContext()
        ctx.set("x", np.array([1.0]))
        result = _handle_command(":vars", ctx)
        assert "x" in result

    def test_handle_reset(self):
        from flux_manifold.repl import _handle_command
        ctx = EvalContext()
        ctx.set("x", 42)
        _handle_command(":reset", ctx)
        assert len(ctx.variables) == 0

    def test_handle_set_epsilon(self):
        from flux_manifold.repl import _handle_command
        ctx = EvalContext()
        _handle_command(":set epsilon 0.5", ctx)
        assert ctx.epsilon == 0.5

    def test_handle_set_tol(self):
        from flux_manifold.repl import _handle_command
        ctx = EvalContext()
        _handle_command(":set tol 0.01", ctx)
        assert ctx.tol == 0.01

    def test_handle_quit(self):
        from flux_manifold.repl import _handle_command
        result = _handle_command(":quit", EvalContext())
        assert result is None

    def test_handle_unknown(self):
        from flux_manifold.repl import _handle_command
        result = _handle_command(":blah", EvalContext())
        assert "Unknown" in result

    def test_format_array(self):
        from flux_manifold.repl import _format_result
        out = _format_result(np.array([1.0, 2.0]))
        assert "1.0000" in out

    def test_format_superposition(self):
        from flux_manifold.repl import _format_result
        sp = SuperpositionTensor(np.array([[1, 0], [0, 1]], dtype=np.float32))
        out = _format_result(sp)
        assert "SuperpositionTensor" in out

    def test_format_dict(self):
        from flux_manifold.repl import _format_result
        out = _format_result({"quality": 0.95, "equivalent": True})
        assert "quality" in out

    def test_format_none(self):
        from flux_manifold.repl import _format_result
        out = _format_result(None)
        assert "none" in out
