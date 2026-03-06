"""Interactive REPL for Latent Flux expressions.

Launch: python -m flux_manifold.repl
"""

from __future__ import annotations

import sys
import traceback
import numpy as np

from flux_manifold.parser import parse, evaluate, run, EvalContext


BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║               Latent Flux REPL  v0.2                      ║
  ║  Operators: ∑_ψ  ⟼  ∇↓  ≅  ↓!  ⇑  ◉                   ║
  ║  ASCII:     sum_psi -> squeeze ~= commit cascade fold     ║
  ║  Pipe:      expr | op1 | op2                              ║
  ║  Functions: random(n,d) zeros(d) ones(d) randn(d)         ║
  ║  Commands:  :help :vars :reset :set <key> <val> :quit     ║
  ╚═══════════════════════════════════════════════════════════╝
"""

HELP = """
  Latent Flux Expression Syntax
  ─────────────────────────────
  Operators (Unicode or ASCII):
    ∑_ψ / sum_psi / superpose   Create superposition from states
    ⟼  / ->  / flow              Flow states toward attractor
    ∇↓ / squeeze                 Dimensional squeeze
    ≅  / ~=  / equiv             Drift equivalence check
    ↓! / commit                  Commitment sink (irreversible)
    ⇑  / cascade                 Abstraction cascade
    ◉  / fold                    Fold-reference self-critique

  Pipe operator:
    expr | op arg | op arg       Chain operations left-to-right

  Examples:
    ∑_ψ random(5, 8) ⟼ zeros(8) | ↓!
    [1, 2, 3] ⟼ [0, 0, 0]
    sum_psi [1,0; 0,1; 0.5,0.5] -> [1, 1] | fold | commit
    random(10, 32) | superpose | squeeze 8 | -> zeros(8) | cascade 3

  Variables:
    let x = <expr>               Assign result to variable
    x                            Use variable in expression

  REPL commands:
    :help                        Show this help
    :vars                        Show defined variables
    :reset                       Reset context (clear vars + commitment)
    :set epsilon <val>           Set flow epsilon
    :set tol <val>               Set convergence tolerance
    :set maxsteps <val>          Set max flow steps
    :set flow <name>             Set flow function (normalize/sin/damped/adaptive)
    :set seed <val>              Set random seed
    :quit / :q                   Exit REPL
"""


def _format_result(result) -> str:
    """Pretty-print a pipeline result."""
    if result is None:
        return "  (none)"

    from flux_manifold.superposition import SuperpositionTensor

    if isinstance(result, SuperpositionTensor):
        lines = [f"  SuperpositionTensor(n={result.n}, d={result.d})"]
        lines.append(f"    entropy = {result.entropy():.4f}")
        lines.append(f"    mean    = [{', '.join(f'{v:.4f}' for v in result.mean_state()[:8])}{'...' if result.d > 8 else ''}]")
        if result.n <= 6:
            lines.append(f"    weights = [{', '.join(f'{w:.3f}' for w in result.weights)}]")
        else:
            lines.append(f"    weights = [{', '.join(f'{w:.3f}' for w in result.weights[:4])} ... {', '.join(f'{w:.3f}' for w in result.weights[-2:])}]")
        return "\n".join(lines)

    if isinstance(result, np.ndarray):
        if result.size <= 16:
            return f"  [{', '.join(f'{v:.4f}' for v in result.flat)}]"
        return f"  array(shape={result.shape}, mean={result.mean():.4f}, std={result.std():.4f})"

    if isinstance(result, dict):
        lines = ["  {"]
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                if v.size <= 8:
                    vs = f"[{', '.join(f'{x:.4f}' for x in v.flat)}]"
                else:
                    vs = f"array(shape={v.shape})"
            elif isinstance(v, float):
                vs = f"{v:.6f}"
            else:
                vs = str(v)
            lines.append(f"    {k}: {vs}")
        lines.append("  }")
        return "\n".join(lines)

    if isinstance(result, list):
        lines = [f"  Cascade ({len(result)} levels):"]
        for i, level in enumerate(result):
            if isinstance(level, np.ndarray):
                lines.append(f"    L{i}: dim={level.shape[-1] if level.ndim > 0 else 0}")
            else:
                lines.append(f"    L{i}: {type(level).__name__}")
        return "\n".join(lines)

    return f"  {result}"


def _handle_command(line: str, ctx: EvalContext) -> str | None:
    """Handle :command lines. Returns output string or None to quit."""
    parts = line.split()
    cmd = parts[0].lower()

    if cmd in (":quit", ":q", ":exit"):
        return None

    if cmd == ":help":
        return HELP

    if cmd == ":vars":
        if not ctx.variables:
            return "  (no variables defined)"
        lines = []
        for k, v in ctx.variables.items():
            if isinstance(v, np.ndarray):
                lines.append(f"  {k} = array(shape={v.shape})")
            else:
                lines.append(f"  {k} = {type(v).__name__}")
        return "\n".join(lines)

    if cmd == ":reset":
        ctx.variables.clear()
        ctx.commitment = __import__("flux_manifold.commitment_sink", fromlist=["CommitmentSink"]).CommitmentSink()
        ctx.last_trace = None
        ctx.last_superposition = None
        return "  Context reset."

    if cmd == ":set":
        if len(parts) < 3:
            return "  Usage: :set <key> <value>"
        key, val = parts[1].lower(), parts[2]
        if key == "epsilon":
            ctx.epsilon = float(val)
            return f"  epsilon = {ctx.epsilon}"
        elif key == "tol":
            ctx.tol = float(val)
            return f"  tol = {ctx.tol}"
        elif key in ("maxsteps", "max_steps"):
            ctx.max_steps = int(val)
            return f"  max_steps = {ctx.max_steps}"
        elif key == "flow":
            from flux_manifold.parser import FLOW_FNS
            if val not in FLOW_FNS:
                return f"  Unknown flow: {val}. Choose from: {', '.join(FLOW_FNS)}"
            ctx.flow_fn = FLOW_FNS[val]
            return f"  flow = {val}"
        elif key == "seed":
            ctx.seed = int(val)
            return f"  seed = {ctx.seed}"
        else:
            return f"  Unknown setting: {key}"

    return f"  Unknown command: {cmd}"


def repl(ctx: EvalContext | None = None) -> None:
    """Run the interactive Latent Flux REPL."""
    ctx = ctx or EvalContext()
    print(BANNER)

    while True:
        try:
            line = input("flux> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not line:
            continue

        # REPL commands
        if line.startswith(":"):
            result = _handle_command(line, ctx)
            if result is None:
                print("  Goodbye.")
                break
            print(result)
            continue

        # Variable assignment: let x = expr
        var_name = None
        expr_str = line
        if line.startswith("let ") and "=" in line:
            eq_pos = line.index("=")
            var_name = line[4:eq_pos].strip()
            expr_str = line[eq_pos + 1:].strip()

        # Parse and evaluate
        try:
            ast = parse(expr_str)
            result = evaluate(ast, ctx)
            if var_name:
                ctx.set(var_name, result)
                print(f"  {var_name} =")
            print(_format_result(result))
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Entry point for python -m flux_manifold.repl"""
    repl()


if __name__ == "__main__":
    main()
