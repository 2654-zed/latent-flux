"""CLI entry point: python -m flux_manifold run <file.lf>"""
import sys
import time
import numpy as np

from flux_manifold.parser import run_file, EvalContext


def _format_value(val: object) -> str:
    """Format a result value for display."""
    if isinstance(val, np.ndarray):
        if val.size <= 10:
            return f"ndarray{val.shape}: {val}"
        return f"ndarray{val.shape}: [{val.ravel()[0]:.4f} ... {val.ravel()[-1]:.4f}]"
    if isinstance(val, list):
        return f"list[{len(val)}]"
    if hasattr(val, "states") and hasattr(val, "n") and hasattr(val, "d"):
        return f"SuperpositionTensor(n={val.n}, d={val.d})"
    return repr(val)


def cmd_run(filepath: str, verbose: bool = False) -> None:
    """Execute a .lf file."""
    messages: list[str] = []

    def on_message(msg: str) -> None:
        messages.append(msg)
        if verbose:
            print(f"  {msg}")

    ctx = EvalContext(on_message=on_message)
    t0 = time.perf_counter()
    result = run_file(filepath, ctx=ctx)
    elapsed = time.perf_counter() - t0

    # Print final bindings
    if ctx.variables:
        print("── Bindings ──")
        for name, val in ctx.variables.items():
            print(f"  {name} = {_format_value(val)}")

    # Print result
    print("── Result ──")
    print(f"  {_format_value(result)}")

    # Print messages
    if messages and not verbose:
        print("── Messages ──")
        for m in messages:
            print(f"  {m}")

    # Print Flow Trace if any non-convergence occurred
    if ctx.flow_trace.has_failures and not verbose:
        # Already emitted via messages in verbose mode
        pass  # trace was already emitted via on_message callback

    print(f"── Elapsed: {elapsed:.3f}s ──")


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m flux_manifold <command> [options]")
        print()
        print("Commands:")
        print("  run <file.lf>     Execute a Latent Flux program")
        print()
        print("Options:")
        print("  -v, --verbose     Print runtime messages as they occur")
        sys.exit(0)

    cmd = args[0]
    if cmd == "run":
        if len(args) < 2:
            print("Error: run requires a .lf file path", file=sys.stderr)
            sys.exit(1)
        verbose = "-v" in args or "--verbose" in args
        filepath = [a for a in args[1:] if not a.startswith("-")][0]
        cmd_run(filepath, verbose=verbose)
    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print("Run 'python -m flux_manifold --help' for usage.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
