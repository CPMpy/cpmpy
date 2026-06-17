"""
Load a pickled CPMpy model, solve it with a given solver and parameters, and
write a JSON record of the run to a predefined output directory.

Usage:
    python run_model.py path/to/model.pickle <solver_name> [--ablate=...] [--out=<path>] [key=value ...]

Each `key=value` is passed as a solver kwarg. Values are parsed as JSON when
possible (so `num_search_workers=4`, `cp_model_probing_level=0`, `foo=true`,
`bar=1.5` get their natural types), otherwise kept as a string. The special
keys `time_limit` and `memory_limit` are handled separately: `time_limit` is
forwarded to `Model.solve(time_limit=...)`, and `memory_limit` (megabytes) caps
the solver subprocess via ``resource.RLIMIT_AS`` so the parent can always
finish and write a JSON record even when the limit is exceeded.

`--memory-limit=<MB>` is an alternative way to set the memory cap.

`--out=<path>` writes the JSON record to that exact file (parent dirs created
as needed); without it, a timestamped file is written into the default output
directory.

`--ablate=` optionally disables one transformation optimization for the run by
monkey-patching the solver class's `transform` method with an ablated pipeline
(defined below). Valid values:
    no-ilpfriendly        -> use the generic `decompose_in_tree` instead of the
                             ILP-friendly `decompose_linear`
    no-detect-categorical -> drop the `linearize_reified_variables` step that
                             detects/encodes categorical variables
Only the solvers with such a pipeline (gurobi, pysat, rc2, scip) are supported.

The output JSON contains the model that was run, the solver and its
parameters, the runtime, the solver status, and (for optimization problems)
the objective value.
"""

import os
import sys
import json
import time
import signal
import functools
import subprocess
import tempfile

import cpmpy as cp
from cpmpy.transformations.flatten_model import toplevel_list, flatten_constraint
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.linearize import decompose_linear, linearize_constraint, \
    linearize_reified_variables, only_positive_bv, only_positive_coefficients
from cpmpy.transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.normalize import simplify_boolean
from cpmpy.transformations.int2bool import int2bool

_HERE = os.path.dirname(os.path.abspath(__file__))

# Predefined output directory for the run records.
OUTPUT_DIR = os.path.join(_HERE, "run_results")

# Valid --ablate choices.
ABLATE_NO_ILPFRIENDLY = "no-ilpfriendly"
ABLATE_NO_CATEGORICAL = "no-detect-categorical"
ABLATE_CHOICES = (ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL)


# The ablated transform pipelines. These were originally written as solver
# subclasses in journal_experiments/ablation.py; they are inlined here as plain
# (self, cpm_expr) methods so this script is self-contained. The two ablations
# differ in only two places, so each solver has a single function that branches
# on `ablate`:
#   - no-ilpfriendly:        decompose with the generic `decompose_in_tree`
#   - no-detect-categorical: decompose with `decompose_linear` but skip the
#                            `linearize_reified_variables` (categorical) step

def gurobi_transform(self, cpm_expr, ablate):
    # expressions have to be linearized to fit in MIP model, see transformations/linearize
    decompose = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})  # linearize and decompose expect safe exprs
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']), csemap=self._csemap)  # constraints that support reification
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # supports >, <, !=
    if ablate == ABLATE_NO_ILPFRIENDLY:
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
    # gurobi does not round towards zero, so no 'div' in supported set: https://github.com/CPMpy/cpmpy/pull/593#issuecomment-2786707188
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "->", "sub", "min", "max", "mul", "abs", "pow"}), csemap=self._csemap)  # the core of the MIP-linearization
    cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv
    return cpm_cons


def pysat_transform(self, cpm_expr, ablate):
    decompose = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = simplify_boolean(cpm_cons)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
    if ablate == ABLATE_NO_ILPFRIENDLY:
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap, ivarmap=self.ivarmap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "->", "and", "or"}), csemap=self._csemap)  # the core of the MIP-linearization
    cpm_cons = int2bool(cpm_cons, self.ivarmap, encoding=self.encoding, csemap=self._csemap)
    cpm_cons = only_positive_coefficients(cpm_cons)
    return cpm_cons


def scip_transform(self, cpm_expr, ablate):
    decompose = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons, supported=self.supported_global_constraints, supported_reified=self.supported_reified_global_constraints, csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    if ablate == ABLATE_NO_ILPFRIENDLY:
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "abs", "->"}) | self.supported_global_constraints, csemap=self._csemap)
    cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
    return cpm_cons


# Map a (base) solver name to its ablated transform. RC2 is PySAT-based and
# reuses the PySAT pipeline.
SOLVER_TRANSFORM = {
    "gurobi": gurobi_transform,
    "pysat": pysat_transform,
    "rc2": pysat_transform,
    "scip": scip_transform,
}


def patch_transform(solver_name, ablate):
    """Replace the `transform` of the solver class for `solver_name` with the
    ablated pipeline `ablate`. Returns a short description of what was patched."""
    base = solver_name.split(":", 1)[0].lower()
    if base not in SOLVER_TRANSFORM:
        raise ValueError("No ablation pipeline for solver '{}' (have: {})".format(
            solver_name, ", ".join(sorted(SOLVER_TRANSFORM))))

    # patch the actual solver class CPMpy will instantiate for this name. The
    # transform stays a function defined in this module, so its helper imports
    # resolve here; `ablate` is bound via partialmethod so the method keeps the
    # expected (self, cpm_expr) signature.
    transform = SOLVER_TRANSFORM[base]
    solver_cls = cp.SolverLookup.lookup(solver_name)
    solver_cls.transform = functools.partialmethod(transform, ablate=ablate)
    return "{}({})".format(transform.__name__, ablate)


def do_solve(model_path, solver_name, ablate, time_limit, solver_kwargs):
    """Load a pickled model, optionally patch the transform, solve, return a record."""
    ablate_variant = None
    if ablate is not None:
        ablate_variant = patch_transform(solver_name, ablate)
        print("[run_model] patched {} transform with {}".format(solver_name, ablate_variant),
              file=sys.stderr)

    model = cp.Model.from_file(model_path)

    t0 = time.time()
    solver = cp.SolverLookup.get(solver_name, model)
    transformation_time = time.time() - t0
    solver.solve(time_limit=time_limit, **solver_kwargs)

    status = solver.status()
    return {
        "model": os.path.basename(model_path),
        "model_path": os.path.abspath(model_path),
        "solver": solver_name,
        "solver_kwargs": solver_kwargs,
        "time_limit": time_limit,
        "ablate": ablate,
        "ablate_variant": ablate_variant,
        "runtime": status.runtime,
        "transformation_time": transformation_time,
        "status": status.exitstatus.name,
        "objective_value": model.objective_value() if model.has_objective() else None,
        "error": None,
    }


def apply_memory_limit(bytes_limit):
    """Apply a virtual-memory cap in the current process (Unix only)."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))


def run_worker(config_path):
    """Child entry point: solve under an already-applied memory limit, write result JSON."""
    with open(config_path, "r") as f:
        config = json.load(f)

    payload = {"ok": False, "reason": "error", "error": "unknown", "record": None}
    try:
        record = do_solve(
            config["model_path"],
            config["solver_name"],
            config["ablate"],
            config["time_limit"],
            config["solver_kwargs"],
        )
        record["memory_limit_mb"] = config.get("memory_limit_mb")
        payload = {"ok": True, "record": record}
    except MemoryError as exc:
        payload = {"ok": False, "reason": "memory_limit", "error": str(exc), "record": None}
    except Exception as exc:
        payload = {"ok": False, "reason": "error", "error": "{}: {}".format(type(exc).__name__, exc), "record": None}

    with open(config["result_path"], "w") as f:
        json.dump(payload, f)


def solve_with_memory_limit(model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb):
    """Run do_solve in a child process with RLIMIT_AS so the parent can report OOM."""
    with tempfile.TemporaryDirectory(prefix="run_model_") as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        result_path = os.path.join(tmpdir, "result.json")
        config = {
            "model_path": model_path,
            "solver_name": solver_name,
            "ablate": ablate,
            "time_limit": time_limit,
            "solver_kwargs": solver_kwargs,
            "memory_limit_mb": memory_limit_mb,
            "result_path": result_path,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        preexec = functools.partial(apply_memory_limit, memory_limit_mb * 1024 * 1024)
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", config_path],
            preexec_fn=preexec,
        )
        proc.wait()

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                payload = json.load(f)
            if payload.get("ok"):
                return payload["record"]
            if payload.get("reason") == "memory_limit":
                return _memory_limit_record(
                    model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb,
                    payload.get("error", "MemoryError"),
                )

        # Child was killed (often SIGKILL from the kernel OOM killer) or crashed
        # without writing a result file.
        if proc.returncode is not None and proc.returncode < 0 and -proc.returncode == signal.SIGKILL:
            return _memory_limit_record(
                model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb,
                "process killed (signal {})".format(signal.SIGKILL),
            )
        return _error_record(
            model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb,
            "solver subprocess exited with code {}".format(proc.returncode),
        )


def _base_record(model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb):
    return {
        "model": os.path.basename(model_path),
        "model_path": os.path.abspath(model_path),
        "solver": solver_name,
        "solver_kwargs": solver_kwargs,
        "time_limit": time_limit,
        "memory_limit_mb": memory_limit_mb,
        "ablate": ablate,
        "ablate_variant": None,
        "runtime": None,
        "transformation_time": None,
        "status": "ERROR",
        "objective_value": None,
        "error": None,
    }


def _memory_limit_record(model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb, detail):
    record = _base_record(model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb)
    record["status"] = "MEMORY_LIMIT"
    record["error"] = "Memory limit of {} MB exceeded: {}".format(memory_limit_mb, detail)
    return record


def _error_record(model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb, detail):
    record = _base_record(model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb)
    record["error"] = detail
    return record


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
        sys.exit(0)

    # Pull out the optional --ablate / --out / --memory-limit flags; everything else is positional.
    ablate = None
    out_path = None
    memory_limit_mb = None
    positional = []
    for arg in sys.argv[1:]:
        if arg.startswith("--ablate="):
            ablate = arg.split("=", 1)[1]
        elif arg.startswith("--out="):
            out_path = arg.split("=", 1)[1]
        elif arg.startswith("--memory-limit="):
            memory_limit_mb = int(arg.split("=", 1)[1])
        else:
            positional.append(arg)

    if len(positional) < 2:
        print("Usage: python run_model.py path/to/model.pickle <solver_name> [--ablate=...] [key=value ...]",
              file=sys.stderr)
        sys.exit(1)
    if ablate is not None and ablate not in ABLATE_CHOICES:
        print("Invalid --ablate value '{}' (choose from: {})".format(
            ablate, ", ".join(ABLATE_CHOICES)), file=sys.stderr)
        sys.exit(1)

    model_path = positional[0]
    solver_name = positional[1]

    # Parse the remaining "key=value" arguments into solver kwargs.
    solver_kwargs = {}
    for arg in positional[2:]:
        if "=" not in arg:
            print("Ignoring malformed kwarg (expected key=value): {}".format(arg), file=sys.stderr)
            continue
        key, raw_value = arg.split("=", 1)
        try:
            value = json.loads(raw_value)  # int/float/bool/null/list when possible
        except json.JSONDecodeError:
            value = raw_value  # leave as plain string
        solver_kwargs[key] = value

    # time_limit / memory_limit are dedicated arguments, not solver-specific kwargs.
    time_limit = solver_kwargs.pop("time_limit", None)
    if memory_limit_mb is None and "memory_limit" in solver_kwargs:
        memory_limit_mb = int(solver_kwargs.pop("memory_limit"))

    if memory_limit_mb is not None:
        record = solve_with_memory_limit(
            model_path, solver_name, ablate, time_limit, solver_kwargs, memory_limit_mb,
        )
    else:
        record = do_solve(model_path, solver_name, ablate, time_limit, solver_kwargs)
        record["memory_limit_mb"] = None

    if out_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        out_name = "{}_{}_{}.json".format(model_name, solver_name, int(time.time() * 1000))
        out_path = os.path.join(OUTPUT_DIR, out_name)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print("[run_model] wrote run record to {}".format(out_path), file=sys.stderr)
    print(json.dumps(record, indent=2))
