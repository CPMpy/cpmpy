"""
Run a CPMpy script but, instead of actually solving, dump every model that
would be solved to a `.pickle` file.

Usage:
    python dump_model.py [--out=<dir>] path/to/script.py [args...]

The monkey-patched `cpmpy.Model.solve` / `Model.solveAll` write the model to
disk via `Model.to_file` instead of invoking a solver. The output filename is
always derived from the input script's name (e.g. `script.py` ->
`script.pickle`); `--out=<dir>` only chooses the directory it is written into
(default: the current directory). If the script solves more than one model,
subsequent dumps get a numeric suffix (`script_1.pickle`, ...).

Since `.solve()` returns a fake "no solution" result, scripts often raise or
exit right after (e.g. ``if not m.solve(): raise ...``). Such a failure *after*
at least one model has been dumped is treated as success, so the dumped
pickle(s) are kept.
"""

import os
import sys
import runpy

import cpmpy as cp

# Base name for the output pickle(s), set in main() from the input script name.
_output_base = None
# Directory to write the pickle(s) into (None -> current directory).
_output_dir = None
# Counter so multiple solve() calls in the same script don't overwrite each other.
_dump_count = 0


def _dump_model(self):
    global _dump_count
    if _dump_count == 0:
        fname = _output_base + ".pickle"
    else:
        fname = "{}_{}.pickle".format(_output_base, _dump_count)
    _dump_count += 1
    if _output_dir is not None:
        fname = os.path.join(_output_dir, fname)

    self.to_file(fname)
    print("[dump_model] wrote model to {}".format(fname), file=sys.stderr)


def _dump_solve(self, *args, **kwargs):
    _dump_model(self)
    # solve() is documented to return a bool; pretend no solution was found.
    return False


def _dump_solveAll(self, *args, **kwargs):
    _dump_model(self)
    # solveAll() returns the number of solutions found; pretend there were none.
    return 0


if __name__ == "__main__":
    # Our own flags (--out=) come before the script path; everything from the
    # script path onwards belongs to the script being run.
    args = sys.argv[1:]
    while args and args[0].startswith("--out="):
        _output_dir = args[0].split("=", 1)[1]
        args = args[1:]

    if not args:
        print("Usage: python dump_model.py [--out=<dir>] path/to/script.py [args...]",
              file=sys.stderr)
        sys.exit(1)

    script_path = args[0]
    _output_base = os.path.splitext(os.path.basename(script_path))[0]
    if _output_dir is not None:
        os.makedirs(_output_dir, exist_ok=True)

    # Install the patches.
    cp.Model.solve = _dump_solve
    cp.Model.solveAll = _dump_solveAll

    # Make the executed script see its own argv (drop our own program name/flags).
    sys.argv = args

    # Run the target script as if it were the main module. A crash after we have
    # already dumped a model is expected (the script reacted to the fake "no
    # solution" result) and is not treated as a failure.
    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception:
        if _dump_count == 0:
            raise
        import traceback
        print("[dump_model] script failed after dumping {} model(s); keeping them"
              .format(_dump_count), file=sys.stderr)
        traceback.print_exc()

    if _dump_count == 0:
        # The script ran fine but never went through Model.solve / Model.solveAll
        # (e.g. it solves via a raw solver-interface object), so nothing was
        # captured. Warn rather than producing a phantom (missing) output.
        print("[dump_model] warning: no model dumped for {} (it never called "
              "Model.solve/solveAll)".format(script_path), file=sys.stderr)
