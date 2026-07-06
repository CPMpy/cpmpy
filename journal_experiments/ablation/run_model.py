#!/usr/bin/env python
"""
Load a pickled CPMpy model, solve it with a given solver and parameters, and
write a JSON record of the run to a predefined output directory.

Usage:
    python run_model.py path/to/model.pickle <solver_name> [--ablate=...] [--out=<path>] [--stop-after-transform] [key=value ...]

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
directory. When solver kwargs are passed on the command line, they are also
encoded in the output filename as ``key1-value1_key2-value2`` (inserted before
the ablation segment in ``model__solver__ablation.json`` paths).

`--ablate=` optionally disables one transformation optimization for the run by
monkey-patching the solver class's `transform` method with an ablated pipeline
(defined below). Valid values:
    no-ilpfriendly        -> use the generic `decompose_in_tree` instead of the
                             ILP-friendly `decompose_linear` (ILP solvers)
    no-detect-categorical -> drop the `linearize_reified_variables` step that
                             detects/encodes categorical variables (ILP solvers)
    no-cp-friendly        -> use `decompose_linear` instead of the generic
                             `decompose_in_tree` (CP solvers: choco, ortools)
    no-positive-decompositions -> skip positive-only decompositions (use
                             `decompose` / linear decomp instead of
                             `decompose_positive` / linear-positive)
Only the solvers with a matching pipeline are supported.

`--stop-after-transform` runs the transform pipeline only (no solve) and
still writes a JSON record. The record always includes final model-size metrics
(``n_constraints``, ``n_integer``, ``n_boolean``) together with the usual solve
fields (which are null when ``--stop-after-transform`` is set).

The output JSON contains the model that was run, the solver and its
parameters, the runtime, the solver status, the CPMpy git commit (when
available), and (for optimization problems) the objective value.
"""

import argparse
import os
import resource
import sys
import json
import time
import signal
import functools
import subprocess
import tempfile
import gc

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from cpmpy.transformations.flatten_model import toplevel_list, flatten_constraint
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.linearize import decompose_linear, get_linear_decompositions, \
    linearize_constraint, linearize_reified_variables, only_positive_bv, \
    only_positive_coefficients, canonical_comparison
from cpmpy.transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.normalize import simplify_boolean
from cpmpy.transformations.int2bool import int2bool

_HERE = os.path.dirname(os.path.abspath(__file__))

# Predefined output directory for the run records.
OUTPUT_DIR = os.path.join(_HERE, "run_results")

DEFAULT_FAILURE_ERROR = "run_model.py exited nonzero (killed: OOM/segfault)"

# Valid --ablate choices.
ABLATE_NO_ILPFRIENDLY = "no-ilpfriendly"
ABLATE_NO_CATEGORICAL = "no-detect-categorical"
ABLATION_NO_CP_FRIENDLY = "no-cp-friendly"
ABLATE_NO_POSITIVE = "no-positive-decompositions"
ABLATE_CHOICES = (
    ABLATE_NO_ILPFRIENDLY,
    ABLATE_NO_CATEGORICAL,
    ABLATION_NO_CP_FRIENDLY,
    ABLATE_NO_POSITIVE,
)


class FallbackToNormalDecomp:
    """Dict stand-in: intercept positive decomps, fall back to normal ones."""

    def __init__(self, decompose_custom=None):
        self._decompose_custom = decompose_custom or {}

    def __contains__(self, name):
        return True

    def __getitem__(self, name):
        if name in self._decompose_custom:
            return self._decompose_custom[name]
        return lambda expr: expr.decompose()


def decompose_no_positive(lst_of_expr, *, supported=None, supported_reified=None,
                          csemap=None, decompose_custom=None):
    dcp = FallbackToNormalDecomp(decompose_custom)
    return decompose_in_tree(
        list(lst_of_expr),
        supported=supported,
        supported_reified=supported_reified,
        csemap=csemap,
        decompose_custom=decompose_custom,
        decompose_custom_positive=dcp,
    )


def decompose_linear_no_positive(lst_of_expr, supported=None, supported_reified=None,
                                 csemap=None):
    dc = get_linear_decompositions()
    return decompose_no_positive(
        lst_of_expr,
        supported=supported,
        supported_reified=supported_reified,
        csemap=csemap,
        decompose_custom=dc,
    )


def decompose_in_tree_no_positive(lst_of_expr, supported=None, supported_reified=None,
                                  csemap=None):
    return decompose_no_positive(
        lst_of_expr,
        supported=supported,
        supported_reified=supported_reified,
        csemap=csemap,
    )


def pick_decompose(ablate, base_decompose):
    """Apply no-positive ablation on top of the solver's default decompose step."""
    if ablate == ABLATE_NO_POSITIVE:
        if base_decompose is decompose_linear:
            return decompose_linear_no_positive
        return decompose_in_tree_no_positive
    return base_decompose


# The ablated transform pipelines. These were originally written as solver
# subclasses in journal_experiments/ablation.py; they are inlined here as plain
# (self, cpm_expr) methods so this script is self-contained. Each solver has a
# single function that branches on `ablate`:
#   - no-ilpfriendly:        decompose with the generic `decompose_in_tree`
#   - no-detect-categorical: decompose with `decompose_linear` but skip the
#                            `linearize_reified_variables` (categorical) step
#   - no-cp-friendly:        decompose with `decompose_linear` instead of
#                            `decompose_in_tree` (CP solvers only)

def gurobi_transform(self, cpm_expr, ablate):
    # expressions have to be linearized to fit in MIP model, see transformations/linearize
    base = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})  # linearize and decompose expect safe exprs
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']), csemap=self._csemap)  # constraints that support reification
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # supports >, <, !=
    if ablate == ABLATE_NO_CATEGORICAL:
        pass
    else: # don't detect categorical variables
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
    # gurobi does not round towards zero, so no 'div' in supported set: https://github.com/CPMpy/cpmpy/pull/593#issuecomment-2786707188
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "->", "sub", "min", "max", "mul", "abs", "pow"}), csemap=self._csemap)  # the core of the MIP-linearization
    cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv

    return cpm_cons


def pysat_transform(self, cpm_expr, ablate):
    base = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = simplify_boolean(cpm_cons)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
    if ablate == ABLATE_NO_CATEGORICAL:
        pass
    else: # don't detect categorical variables
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap, ivarmap=self.ivarmap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "->", "and", "or"}), csemap=self._csemap)  # the core of the MIP-linearization
    cpm_cons = int2bool(cpm_cons, self.ivarmap, encoding=self.encoding, csemap=self._csemap)
    cpm_cons = only_positive_coefficients(cpm_cons)
    return cpm_cons


def scip_transform(self, cpm_expr, ablate):
    base = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons, supported=self.supported_global_constraints, supported_reified=self.supported_reified_global_constraints, csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    if ablate == ABLATE_NO_CATEGORICAL:
        pass
    else: # don't detect categorical variables
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "abs", "->"}) | self.supported_global_constraints, csemap=self._csemap)
    cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
    return cpm_cons


def highs_transform(self, cpm_expr, ablate):
    base = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons, supported=self.supported_global_constraints, supported_reified=self.supported_reified_global_constraints, csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset({"sum", "wsum", "sub"}), csemap=self._csemap)
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset({"sum", "wsum", "sub"}), csemap=self._csemap)
    if ablate == ABLATE_NO_CATEGORICAL:
        pass
    else: # don't detect categorical variables
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum"}), csemap=self._csemap)
    cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
    return cpm_cons


def exact_transform(self, cpm_expr, ablate):
    base = decompose_in_tree if ablate == ABLATE_NO_ILPFRIENDLY else decompose_linear
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element", "nd_element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    if ablate == ABLATE_NO_CATEGORICAL:
        pass
    else: # don't detect categorical variables
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "->", "mul"}), csemap=self._csemap)
    cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
    return cpm_cons


def pindakaas_transform(self, cpm_expr, ablate):
    base = decompose_linear if ablate == ABLATION_NO_CP_FRIENDLY else decompose_in_tree
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(
        cpm_cons,
        supported=self.supported_global_constraints,
        supported_reified=self.supported_reified_global_constraints,
        csemap=self._csemap,
    )
    cpm_cons = simplify_boolean(cpm_cons)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
    if ablate == ABLATE_NO_CATEGORICAL:
        pass
    else: # don't detect categorical variables
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap, ivarmap=self.ivarmap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "->", "and", "or"}), csemap=self._csemap)
    cpm_cons = int2bool(cpm_cons, self.ivarmap, encoding=self.encoding, csemap=self._csemap)
    return cpm_cons

def choco_transform(self, cpm_expr, ablate):
    base = decompose_linear if ablate == ABLATION_NO_CP_FRIENDLY else decompose_in_tree
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons)
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
    cpm_cons = canonical_comparison(cpm_cons)
    cpm_cons = reify_rewrite(cpm_cons,
                             supported=self.supported_global_constraints | {"sum", "wsum"},
                             csemap=self._csemap)
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)
    return cpm_cons


def ortools_transform(self, cpm_expr, ablate):
    base = decompose_linear if ablate == ABLATION_NO_CP_FRIENDLY else decompose_in_tree
    decompose = pick_decompose(ablate, base)
    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset({"div", "mod"}))
    cpm_cons = push_down_negation(cpm_cons)
    cpm_cons = decompose(cpm_cons,
                         supported=self.supported_global_constraints,
                         supported_reified=self.supported_reified_global_constraints,
                         csemap=self._csemap)
    cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
    cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
    cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)
    cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
    cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
    return cpm_cons


# Map a (base) solver name to its ablated transform. RC2 is PySAT-based and
# reuses the PySAT pipeline.
SOLVER_TRANSFORM = {
    "gurobi": gurobi_transform,
    "pysat": pysat_transform,
    "rc2": pysat_transform,
    "scip": scip_transform,
    "highs": highs_transform,
    "exact": exact_transform,
    "choco": choco_transform,
    "ortools": ortools_transform,
    "pindakaas": pindakaas_transform,
}

# Which ablations each solver supports.
SOLVER_ABLATIONS = {
    "gurobi": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
    "pysat": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
    "rc2": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
    "scip": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
    "highs": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
    "exact": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
    "choco": frozenset({ABLATION_NO_CP_FRIENDLY, ABLATE_NO_POSITIVE}),
    "ortools": frozenset({ABLATION_NO_CP_FRIENDLY, ABLATE_NO_POSITIVE}),
    "pindakaas": frozenset({ABLATE_NO_ILPFRIENDLY, ABLATE_NO_CATEGORICAL, ABLATE_NO_POSITIVE}),
}


def cpmpy_git_info():
    """Return (commit hash, commit message) for the CPMpy checkout, or (None, None)."""
    pkg_dir = os.path.dirname(os.path.abspath(cp.__file__))
    try:
        root = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        out = subprocess.run(
            ["git", "-C", root, "log", "-1", "--format=%H%n%s"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        commit, _, message = out.partition("\n")
        return commit.strip(), message.rstrip("\n")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None, None


def format_kwarg_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def solver_kwargs_filename_segment(solver_kwargs):
    """Encode solver kwargs as ``key1-value1_key2-value2`` for use in filenames."""
    if not solver_kwargs:
        return ""
    return "_".join(
        "{}-{}".format(key, format_kwarg_value(solver_kwargs[key]))
        for key in sorted(solver_kwargs)
    )


def out_path_with_solver_kwargs(out_path, solver_kwargs):
    """Insert a solver-kwargs segment into a ``model__solver__ablation.json`` path."""
    segment = solver_kwargs_filename_segment(solver_kwargs)
    if not segment or segment in out_path:
        return out_path
    root, ext = os.path.splitext(out_path)
    base, sep, tail = root.rpartition("__")
    if not sep:
        return "{}__{}{}".format(root, segment, ext)
    return "{}__{}__{}{}".format(base, segment, tail, ext)


def patch_transform(solver_name, ablate):
    """Replace the `transform` of the solver class for `solver_name` with the
    ablated pipeline `ablate`. Returns a short description of what was patched."""
    base = solver_name.split(":", 1)[0].lower()
    if base not in SOLVER_TRANSFORM:
        raise ValueError("No ablation pipeline for solver '{}' (have: {})".format(
            solver_name, ", ".join(sorted(SOLVER_TRANSFORM))))
    if ablate not in SOLVER_ABLATIONS[base]:
        raise ValueError("Ablation '{}' not supported for solver '{}' (choose from: {})".format(
            ablate, solver_name, ", ".join(sorted(SOLVER_ABLATIONS[base]))))

    # patch the actual solver class CPMpy will instantiate for this name. The
    # transform stays a function defined in this module, so its helper imports
    # resolve here; `ablate` is bound via partialmethod so the method keeps the
    # expected (self, cpm_expr) signature.
    transform = SOLVER_TRANSFORM[base]
    solver_cls = cp.SolverLookup.lookup(solver_name)
    solver_cls.transform = functools.partialmethod(transform, ablate=ablate)
    return solver_cls


def count_transform_stats(solver, cpm_expr):
    """Run ``solver.transform`` and return final model-size metrics."""
    cpm_cons = solver.transform(cpm_expr)
    vars = get_variables(cpm_cons)
    n_bool = sum(1 for v in vars if v.is_bool())
    n_int = len(vars) - n_bool
    return {
        "n_constraints": len(cpm_cons),
        "n_integer": n_int,
        "n_boolean": n_bool,
    }

def do_solve(model_path, solver_name, ablate, time_limit, memory_limit, solver_kwargs, stop_after_transform=False):
    """Load a pickled model, optionally patch the transform, solve, return a record."""
    model = cp.Model.from_file(model_path)
    sname = solver_name
    if model.has_objective() and sname == "pysat":
        sname = "rc2" # PySAT cannot handle objectives, switch to RC2 Max-SAT solver

    if ablate is not None:
        solver_cls = patch_transform(sname, ablate)
        print("Patched solver {} for ablation {}".format(sname, ablate))

    record = new_run_record(
        model_path=model_path,
        solver_name=solver_name,
        ablate=ablate,
        time_limit=time_limit,
        memory_limit=memory_limit,
        solver_kwargs=solver_kwargs,
        stop_after_transform=stop_after_transform,
    )

    try:
        t0 = time.time()
        solver = cp.SolverLookup.get(sname)
        transform_stats = count_transform_stats(solver, model.constraints)
        transformation_time = time.time() - t0

        record["transformation_time"] = transformation_time
        record.update(transform_stats)
    
    except Exception as e:
        # clean up memory, its probably a memory error...
        gc.collect()
        print("[run_model] error during init: {}".format(e), file=sys.stderr)
        record["status"] = "error"
        record["error"] = str(e)
        return record

    gc.collect() # force garbage collection to free up memory
    if stop_after_transform:
        return record

    try:
        # do the solve
        # re-initialize the solver... no easy way to count transform and also do the solve...
        # could split it up, but lets keep this for now...
        if sname == "exact": # requires setting args during init
            solver = cp.SolverLookup.get(sname, model, **solver_kwargs)
            solver_kwargs = dict()
        else:
            solver = cp.SolverLookup.get(sname, model, **solver_kwargs)
            
        solver.solve(time_limit=time_limit, **solver_kwargs)

        status = solver.status()
        record["runtime"] = status.runtime
        record["status"] = status.exitstatus.name
        record["objective_value"] = model.objective_value() if model.has_objective() else None

        if status.exitstatus.name == "FEASIBLE" or status.exitstatus.name == "OPTIMAL":
            # check that all constraints are satisfied
            for c in toplevel_list(model.constraints):
                if c.value() is False:
                    record['error'] = "Solution check failed: constraint {} is not satisfied".format(c)

        return record
    except Exception as e:
        print("[run_model] error during solving: {}".format(e), file=sys.stderr)
        record['status'] = "error"
        record['error'] = str(e)
        return record


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Load a pickled CPMpy model and solve it with a given solver.")
    parser.add_argument("--model-path", help="path to model.pickle")
    parser.add_argument("--solver-name", help="solver name (e.g. ortools, gurobi)")
    parser.add_argument("--ablate", choices=ABLATE_CHOICES, default=None,
                        help="disable one transformation optimization")
    parser.add_argument("--out", default=None, help="write JSON record to this path")
    parser.add_argument("--memory-limit", type=int, default=None,
                        dest="memory_limit_gb", help="memory cap in gigabytes (RLIMIT_AS)")
    parser.add_argument("--time-limit", type=int, required=True,
                        dest="time_limit", help="time limit in seconds")
    parser.add_argument("--stop-after-transform", action="store_true",
                        help="run transform pipeline only, skip solve")
    parser.add_argument("solver_kwargs", nargs="*",
                        help="solver kwargs as key=value (values parsed as JSON when possible)")
    return parser


def parse_explicit_solver_kwargs(solver_kwargs_args):
    explicit_solver_kwargs = {}
    for arg in solver_kwargs_args:
        if "=" not in arg:
            print("Ignoring malformed kwarg (expected key=value): {}".format(arg), file=sys.stderr)
            continue
        key, raw_value = arg.split("=", 1)
        try:
            value = json.loads(raw_value)  # int/float/bool/null/list when possible
        except json.JSONDecodeError:
            value = raw_value  # leave as plain string
        explicit_solver_kwargs[key] = value
    return explicit_solver_kwargs


def prepare_solver_kwargs(solver_name, explicit_solver_kwargs):
    solver_kwargs = dict(explicit_solver_kwargs)
    if solver_name == "gurobi":
        solver_kwargs["Threads"] = 1
    if solver_name == "ortools":
        solver_kwargs["num_workers"] = 1
    return solver_kwargs


def resolve_out_path(model_path, solver_name, ablate, out_path, explicit_solver_kwargs):
    if out_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        ablate_name = ablate or "baseline"
        kwargs_segment = solver_kwargs_filename_segment(explicit_solver_kwargs)
        if kwargs_segment:
            out_name = "{}__{}__{}__{}.json".format(
                model_name, solver_name, kwargs_segment, ablate_name)
        else:
            out_name = "{}__{}__{}.json".format(model_name, solver_name, ablate_name)
        return os.path.join(OUTPUT_DIR, out_name)

    out_path = out_path_with_solver_kwargs(out_path, explicit_solver_kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    return out_path


def new_run_record(model_path, solver_name, ablate, time_limit, memory_limit, solver_kwargs,
                   stop_after_transform):
    cpmpy_commit, cpmpy_commit_message = cpmpy_git_info()
    return {
        "model": os.path.basename(model_path),
        "model_path": os.path.abspath(model_path),
        "solver": solver_name,
        "solver_kwargs": solver_kwargs,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
        "ablate": ablate,
        "stop_after_transform": stop_after_transform,
        "cpmpy_commit": cpmpy_commit,
        "cpmpy_commit_message": cpmpy_commit_message,
        "transformation_time": None,
        "runtime": None,
        "status": None,
        "objective_value": None,
        "error": None,
    }


def make_failure_record(model_path, solver_name, ablate, time_limit, memory_limit, solver_kwargs,
                        stop_after_transform, error=DEFAULT_FAILURE_ERROR):
    record = new_run_record(
        model_path=model_path,
        solver_name=solver_name,
        ablate=ablate,
        time_limit=time_limit,
        memory_limit=memory_limit,
        solver_kwargs=solver_kwargs,
        stop_after_transform=stop_after_transform,
    )
    record["status"] = "error"
    record["error"] = error
    return record


def collect_run_settings(args):
    explicit_solver_kwargs = parse_explicit_solver_kwargs(args.solver_kwargs)
    solver_kwargs = prepare_solver_kwargs(args.solver_name, explicit_solver_kwargs)
    out_path = resolve_out_path(
        args.model_path, args.solver_name, args.ablate, args.out, explicit_solver_kwargs)
    return {
        "model_path": args.model_path,
        "solver_name": args.solver_name,
        "ablate": args.ablate,
        "time_limit": args.time_limit,
        "memory_limit": args.memory_limit_gb,
        "solver_kwargs": solver_kwargs,
        "stop_after_transform": args.stop_after_transform,
        "out_path": out_path,
    }


def write_record(out_path, record):
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


def emit_record(out_path, record, tag):
    write_record(out_path, record)
    print("[{}] wrote run record to {}".format(tag, out_path), file=sys.stderr)
    print(json.dumps(record, indent=2))


def apply_memory_limit(memory_limit_gb):
    if memory_limit_gb is not None:
        print("Setting memory limit to {} GB".format(memory_limit_gb), file=sys.stderr)
        limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))


def run_from_args(args):
    settings = collect_run_settings(args)
    apply_memory_limit(args.memory_limit_gb)
    out_path = settings.pop("out_path")
    record = do_solve(**settings)
    emit_record(out_path, record, "run_model")


def write_failure_from_args(args, error=None):
    if error is None:
        error = getattr(args, "error", DEFAULT_FAILURE_ERROR)
    settings = collect_run_settings(args)
    out_path = settings.pop("out_path")
    record = make_failure_record(error=error, **settings)
    emit_record(out_path, record, "write_dummy_file")


if __name__ == "__main__":
    run_from_args(build_arg_parser().parse_args())
