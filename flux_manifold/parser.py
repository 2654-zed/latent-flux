"""Latent Flux Expression Parser – parse and evaluate ∑_ψ ⟼ ∇↓ ≅ ↓! ⇑ ◉ syntax.

Grammar (simplified):
    expr     ::= pipeline
    pipeline ::= atom (OP atom)*
    atom     ::= SYMBOL | NUMBER | VECTOR | '(' pipeline ')' | func_call
    OP       ::= '⟼' | '∇↓' | '≅' | '↓!' | '⇑' | '◉' | '∑_ψ' | '|'

Syntax examples:
    ∑_ψ [0.1, 0.2; 0.3, 0.4] ⟼ [0, 0]
    state ⟼ [1,1] | ∇↓ 8 | ≅ 0.05 | ↓! | ⇑ 3
    ∑_ψ random(10, 32) ⟼ zeros(32) | ◉ | ↓!

The parser builds an AST of operations, then the evaluator executes
them left-to-right as a pipeline.
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from flux_manifold.core import flux_flow_traced
from flux_manifold.flows import normalize_flow, sin_flow, damped_flow, adaptive_flow
from flux_manifold.superposition import SuperpositionTensor
from flux_manifold.drift_equivalence import DriftEquivalence
from flux_manifold.commitment_sink import CommitmentSink
from flux_manifold.abstraction_cascade import AbstractionCascade
from flux_manifold.fold_reference import FoldReference, no_nan_critique, norm_bound_critique
from flux_manifold.dimensional_squeeze import DimensionalSqueeze


# ── AST Nodes ──────────────────────────────────────────────────────

@dataclass
class LFVector:
    """Literal vector: [1.0, 2.0, 3.0]"""
    values: list[list[float]]  # rows (1 row = 1-D vector, N rows = superposition)

@dataclass
class LFNumber:
    """Literal number."""
    value: float

@dataclass
class LFSymbol:
    """Named variable reference."""
    name: str

@dataclass
class LFFuncCall:
    """Built-in function call: random(10, 32), zeros(8), etc."""
    name: str
    args: list[Any]

@dataclass
class LFPipeline:
    """Chain of operations: source | op1 | op2 ..."""
    stages: list[Any]  # List of (operation_name, operand_or_None)

@dataclass
class LFOp:
    """A pipeline operator with optional argument."""
    symbol: str  # '⟼', '∇↓', '≅', '↓!', '⇑', '◉', '∑_ψ'
    arg: Any = None  # Operand (vector, number, None)


# ── Tokenizer ─────────────────────────────────────────────────────

# Unicode operators and ASCII aliases
OPERATORS = {
    "⟼": "flow", "->": "flow", "flow": "flow",
    "∇↓": "squeeze", "squeeze": "squeeze",
    "≅": "equiv", "~=": "equiv", "equiv": "equiv",
    "↓!": "commit", "commit": "commit",
    "⇑": "cascade", "cascade": "cascade",
    "◉": "fold", "fold": "fold",
    "∑_ψ": "superpose", "sum_psi": "superpose", "superpose": "superpose",
}

TOKEN_PATTERN = re.compile(
    r"""
    (\∑_ψ|⟼|∇↓|≅|↓!|⇑|◉)          # Unicode operators
    |(\->|~=|sum_psi)                 # ASCII aliases
    |(flow|squeeze|equiv|commit|cascade|fold|superpose)  # Word operators
    |(\|)                              # Pipe
    |(\[)                              # Open bracket
    |(\])                              # Close bracket
    |(\()                              # Open paren
    |(\))                              # Close paren
    |(;)                               # Row separator in matrices
    |(,)                               # Comma
    |(-?[0-9]+\.?[0-9]*)              # Number
    |([a-zA-Z_][a-zA-Z_0-9]*)        # Identifier
    |(\s+)                             # Whitespace (skip)
    """,
    re.VERBOSE,
)


def tokenize(source: str) -> list[tuple[str, str]]:
    """Tokenize a Latent Flux expression. Returns [(type, value), ...]."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(source):
        m = TOKEN_PATTERN.match(source, pos)
        if not m:
            raise SyntaxError(f"Unexpected character at position {pos}: {source[pos:]!r}")
        val = m.group()
        pos = m.end()

        if val.strip() == "":
            continue  # skip whitespace

        # Classify
        if val in OPERATORS:
            tokens.append(("OP", OPERATORS[val]))
        elif val == "|":
            tokens.append(("PIPE", "|"))
        elif val == "[":
            tokens.append(("LBRACKET", "["))
        elif val == "]":
            tokens.append(("RBRACKET", "]"))
        elif val == "(":
            tokens.append(("LPAREN", "("))
        elif val == ")":
            tokens.append(("RPAREN", ")"))
        elif val == ";":
            tokens.append(("SEMI", ";"))
        elif val == ",":
            tokens.append(("COMMA", ","))
        elif re.match(r"^-?[0-9]", val):
            tokens.append(("NUMBER", val))
        else:
            tokens.append(("IDENT", val))

    return tokens


# ── Parser ─────────────────────────────────────────────────────────

class Parser:
    """Recursive descent parser for Latent Flux expressions."""

    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")
        if expected_type and tok[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok[0]} ({tok[1]!r})")
        self.pos += 1
        return tok

    def parse(self) -> LFPipeline:
        """Parse full expression as a pipeline."""
        stages = [self.parse_stage()]

        while self.peek() is not None:
            tok = self.peek()
            if tok[0] == "PIPE":
                self.consume("PIPE")
                stages.append(self.parse_stage())
            elif tok[0] == "OP":
                stages.append(self.parse_stage())
            else:
                break

        return LFPipeline(stages=stages)

    def parse_stage(self) -> Any:
        """Parse a single pipeline stage: operator+arg or atom."""
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Expected stage, got end of expression")

        if tok[0] == "OP":
            return self.parse_op()
        else:
            return self.parse_atom()

    def parse_op(self) -> LFOp:
        """Parse an operator with optional argument."""
        tok = self.consume("OP")
        op_name = tok[1]

        # Check if next token is an argument (number, vector, or func call)
        arg = None
        nxt = self.peek()
        if nxt is not None and nxt[0] in ("NUMBER", "LBRACKET", "IDENT"):
            if nxt[0] != "IDENT" or nxt[1] not in OPERATORS:
                # Don't consume the next operator as an argument
                if nxt[0] == "IDENT" and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == "LPAREN":
                    arg = self.parse_func_call()
                elif nxt[0] != "IDENT" or nxt[1] not in ("flow", "squeeze", "equiv", "commit", "cascade", "fold", "superpose"):
                    arg = self.parse_atom()

        return LFOp(symbol=op_name, arg=arg)

    def parse_atom(self) -> Any:
        """Parse a literal value, variable, or function call."""
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Expected atom, got end of expression")

        if tok[0] == "NUMBER":
            self.consume()
            return LFNumber(value=float(tok[1]))

        if tok[0] == "LBRACKET":
            return self.parse_vector()

        if tok[0] == "IDENT":
            # Check for function call
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == "LPAREN":
                return self.parse_func_call()
            self.consume()
            return LFSymbol(name=tok[1])

        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse()
            self.consume("RPAREN")
            return expr

        raise SyntaxError(f"Unexpected token: {tok}")

    def parse_vector(self) -> LFVector:
        """Parse [1, 2, 3] or [1, 2; 3, 4] (matrix/superposition)."""
        self.consume("LBRACKET")
        rows: list[list[float]] = []
        current_row: list[float] = []

        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed bracket")
            if tok[0] == "RBRACKET":
                self.consume()
                if current_row:
                    rows.append(current_row)
                break
            elif tok[0] == "NUMBER":
                self.consume()
                current_row.append(float(tok[1]))
            elif tok[0] == "COMMA":
                self.consume()
            elif tok[0] == "SEMI":
                self.consume()
                rows.append(current_row)
                current_row = []
            else:
                raise SyntaxError(f"Unexpected in vector: {tok}")

        if not rows:
            raise SyntaxError("Empty vector")
        return LFVector(values=rows)

    def parse_func_call(self) -> LFFuncCall:
        """Parse func(arg1, arg2, ...)."""
        name = self.consume("IDENT")[1]
        self.consume("LPAREN")
        args: list[Any] = []
        while self.peek() and self.peek()[0] != "RPAREN":
            if self.peek()[0] == "COMMA":
                self.consume()
                continue
            args.append(self.parse_atom())
        self.consume("RPAREN")
        return LFFuncCall(name=name, args=args)


def parse(source: str) -> LFPipeline:
    """Parse a Latent Flux expression string into an AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    ast = parser.parse()
    return ast


# ── Evaluator ──────────────────────────────────────────────────────

FLOW_FNS = {
    "normalize": normalize_flow,
    "sin": sin_flow,
    "damped": damped_flow,
    "adaptive": adaptive_flow,
}


class EvalContext:
    """Evaluation context holding variables and configuration."""

    def __init__(self, seed: int = 42, epsilon: float = 0.1, tol: float = 1e-3,
                 max_steps: int = 500, flow_name: str = "normalize"):
        self.variables: dict[str, Any] = {}
        self.seed = seed
        self.epsilon = epsilon
        self.tol = tol
        self.max_steps = max_steps
        self.flow_fn = FLOW_FNS.get(flow_name, normalize_flow)
        self.last_trace: dict | None = None
        self.last_superposition: SuperpositionTensor | None = None
        self.commitment = CommitmentSink()

    def set(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get(self, name: str) -> Any:
        if name not in self.variables:
            raise NameError(f"Undefined variable: {name!r}")
        return self.variables[name]


def evaluate(ast: LFPipeline, ctx: EvalContext | None = None) -> Any:
    """Evaluate a parsed Latent Flux pipeline."""
    ctx = ctx or EvalContext()
    result: Any = None

    for stage in ast.stages:
        result = _eval_stage(stage, result, ctx)

    return result


def _eval_stage(node: Any, current: Any, ctx: EvalContext) -> Any:
    """Evaluate a single pipeline stage."""
    if isinstance(node, LFOp):
        return _eval_op(node, current, ctx)
    elif isinstance(node, LFVector):
        return _eval_vector(node)
    elif isinstance(node, LFNumber):
        return node.value
    elif isinstance(node, LFSymbol):
        return ctx.get(node.name)
    elif isinstance(node, LFFuncCall):
        return _eval_func(node, ctx)
    elif isinstance(node, LFPipeline):
        return evaluate(node, ctx)
    else:
        raise TypeError(f"Unknown AST node: {type(node)}")


def _eval_vector(node: LFVector) -> np.ndarray:
    """Evaluate a vector literal."""
    if len(node.values) == 1:
        return np.array(node.values[0], dtype=np.float32)
    return np.array(node.values, dtype=np.float32)


def _eval_func(node: LFFuncCall, ctx: EvalContext) -> Any:
    """Evaluate built-in functions."""
    name = node.name
    args = [_eval_stage(a, None, ctx) if not isinstance(a, (int, float)) else a for a in node.args]
    # Extract raw numbers from LFNumber
    raw_args = []
    for a in node.args:
        if isinstance(a, LFNumber):
            raw_args.append(a.value)
        elif isinstance(a, (int, float)):
            raw_args.append(a)
        else:
            raw_args.append(_eval_stage(a, None, ctx))

    if name == "random":
        n, d = int(raw_args[0]), int(raw_args[1])
        return SuperpositionTensor.from_random(n, d, seed=ctx.seed)
    elif name == "zeros":
        d = int(raw_args[0])
        return np.zeros(d, dtype=np.float32)
    elif name == "ones":
        d = int(raw_args[0])
        return np.ones(d, dtype=np.float32)
    elif name == "randn":
        d = int(raw_args[0])
        return np.random.default_rng(ctx.seed).standard_normal(d).astype(np.float32)
    elif name == "linspace":
        start, stop, n = float(raw_args[0]), float(raw_args[1]), int(raw_args[2])
        return np.linspace(start, stop, n, dtype=np.float32)
    elif name == "set":
        # set(name, value) — store in context
        var_name = node.args[0].name if isinstance(node.args[0], LFSymbol) else str(raw_args[0])
        ctx.set(var_name, raw_args[1] if len(raw_args) > 1 else None)
        return raw_args[1] if len(raw_args) > 1 else None
    else:
        raise NameError(f"Unknown function: {name!r}")


def _eval_op(op: LFOp, current: Any, ctx: EvalContext) -> Any:
    """Evaluate a pipeline operator."""
    sym = op.symbol

    if sym == "superpose":
        # ∑_ψ: create or wrap as superposition
        arg = _eval_stage(op.arg, current, ctx) if op.arg is not None else current
        if isinstance(arg, SuperpositionTensor):
            ctx.last_superposition = arg
            return arg
        if isinstance(arg, np.ndarray):
            if arg.ndim == 1:
                arg = arg.reshape(1, -1)
            sp = SuperpositionTensor(arg)
            ctx.last_superposition = sp
            return sp
        raise TypeError(f"∑_ψ expects array or SuperpositionTensor, got {type(arg)}")

    elif sym == "flow":
        # ⟼: flow toward attractor
        q = _eval_stage(op.arg, current, ctx) if op.arg is not None else None
        if q is None:
            raise ValueError("⟼ requires a target attractor")
        q = np.asarray(q, dtype=np.float32)

        if isinstance(current, SuperpositionTensor):
            traces = current.flow_all(q, ctx.flow_fn,
                                       epsilon=ctx.epsilon, tol=ctx.tol,
                                       max_steps=ctx.max_steps)
            ctx.last_trace = traces[0] if traces else None
            current.reweight_by_drift(q)
            ctx.last_superposition = current
            return current
        elif isinstance(current, np.ndarray):
            trace = flux_flow_traced(current, q, ctx.flow_fn,
                                      epsilon=ctx.epsilon, tol=ctx.tol,
                                      max_steps=ctx.max_steps)
            ctx.last_trace = trace
            return trace["converged_state"]
        else:
            raise TypeError(f"⟼ expects array or SuperpositionTensor, got {type(current)}")

    elif sym == "squeeze":
        # ∇↓: dimensional squeeze
        target_dim = int(_eval_stage(op.arg, current, ctx)) if op.arg is not None else 8
        if isinstance(current, SuperpositionTensor):
            ds = DimensionalSqueeze(target_dim=target_dim)
            ds.fit(current.states, seed=ctx.seed)
            current.states = ds.squeeze(current.states)
            current.d = target_dim
            return current
        elif isinstance(current, np.ndarray):
            if current.ndim == 1:
                # Single vector — truncate
                return current[:target_dim].copy()
            ds = DimensionalSqueeze(target_dim=target_dim)
            ds.fit(current, seed=ctx.seed)
            return ds.squeeze(current)
        raise TypeError(f"∇↓ expects array or SuperpositionTensor, got {type(current)}")

    elif sym == "equiv":
        # ≅: drift equivalence check
        tolerance = float(_eval_stage(op.arg, current, ctx)) if op.arg is not None else 0.05
        eq = DriftEquivalence(tolerance=tolerance)
        if isinstance(current, SuperpositionTensor) and ctx.last_trace:
            q_approx = current.mean_state()
            quality = eq.quality(current.collapse_to_best(q_approx), q_approx)
            return {"state": current, "quality": quality, "equivalent": quality > 0}
        elif isinstance(current, np.ndarray) and ctx.last_trace:
            q = ctx.last_trace.get("converged_state", current)
            quality = eq.quality(current, q)
            return {"state": current, "quality": quality, "equivalent": quality > 0}
        return {"state": current, "quality": None, "equivalent": None}

    elif sym == "commit":
        # ↓!: commitment sink
        if ctx.commitment.committed:
            return ctx.commitment.committed_state
        if isinstance(current, SuperpositionTensor):
            best = current.collapse_to_best(current.mean_state())
            return ctx.commitment.commit(best, reason="repl_commit")
        elif isinstance(current, dict) and "state" in current:
            inner = current["state"]
            if isinstance(inner, SuperpositionTensor):
                best = inner.collapse_to_best(inner.mean_state())
                return ctx.commitment.commit(best, reason="repl_commit")
            elif isinstance(inner, np.ndarray):
                return ctx.commitment.commit(inner.copy(), reason="repl_commit")
        elif isinstance(current, np.ndarray):
            return ctx.commitment.commit(current.copy(), reason="repl_commit")
        raise TypeError(f"↓! expects array or SuperpositionTensor, got {type(current)}")

    elif sym == "cascade":
        # ⇑: abstraction cascade
        levels = int(_eval_stage(op.arg, current, ctx)) if op.arg is not None else 3
        ac = AbstractionCascade(levels=levels)
        if isinstance(current, SuperpositionTensor):
            return ac.cascade(current.states)
        elif isinstance(current, np.ndarray):
            if current.ndim == 1:
                return ac.cascade_single(current)
            return ac.cascade(current)
        raise TypeError(f"⇑ expects array or SuperpositionTensor, got {type(current)}")

    elif sym == "fold":
        # ◉: fold-reference (self-critique)
        fr = FoldReference(critique_fn=no_nan_critique, interval=1)
        if isinstance(current, SuperpositionTensor):
            for i in range(current.n):
                corrected, _ = fr.check(current.states[i], step=i)
                current.states[i] = corrected
            return current
        elif isinstance(current, np.ndarray):
            corrected, _ = fr.check(current, step=0)
            return corrected
        raise TypeError(f"◉ expects array or SuperpositionTensor, got {type(current)}")

    else:
        raise ValueError(f"Unknown operator: {sym!r}")


# ── One-liner evaluate ─────────────────────────────────────────────

def run(source: str, ctx: EvalContext | None = None, **kwargs) -> Any:
    """Parse and evaluate a Latent Flux expression in one call.

    Args:
        source: Latent Flux expression string.
        ctx: Optional evaluation context. Created fresh if None.
        **kwargs: Passed to EvalContext constructor if ctx is None.

    Returns:
        The result of the pipeline evaluation.
    """
    ctx = ctx or EvalContext(**kwargs)
    ast = parse(source)
    return evaluate(ast, ctx)
