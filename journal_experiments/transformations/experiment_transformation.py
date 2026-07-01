#!/usr/bin/env python
"""
Run the PySAT transformation stack on one pickled CPMpy model, stage by stage,
and write a JSON record of per-stage model-size metrics.

Usage:
    python experiment_transformation.py --model-path path/to/model.pickle --out=result.json
    python experiment_transformation.py --model-path path/to/model.pickle --memory-limit=8

Each stage records ``n_constraints``, ``n_integer``, ``n_boolean``, and
``runtime`` (seconds). ``--memory-limit`` caps the process via
``resource.RLIMIT_AS`` (gigabytes), same as ``run_model.py``.

Without ``--out``, a timestamped JSON is written under ``transformation_results/``.
"""

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time

import cpmpy as cp
from cpmpy import Model
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from cpmpy.transformations.normalize import toplevel_list, simplify_boolean
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.linearize import (
    decompose_linear,
    linearize_constraint,
    linearize_reified_variables,
    only_positive_coefficients,
)
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_implies, only_bv_reifies
from cpmpy.transformations.int2bool import int2bool

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "transformation_results")


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


def count_variables(cpm_cons):
    n_integer = 0
    n_boolean = 0
    for v in get_variables(cpm_cons):
        if isinstance(v, _BoolVarImpl):
            n_boolean += 1
        elif isinstance(v, _IntVarImpl):
            n_integer += 1
    return n_integer, n_boolean


def record_stage(stages, stage, cpm_cons, runtime):
    n_integer, n_boolean = count_variables(cpm_cons)
    stages[stage] = {
        "n_constraints": len(cpm_cons),
        "n_integer": n_integer,
        "n_boolean": n_boolean,
        "runtime": runtime,
    }


def run_transformation(model):
    """Apply the PySAT transformation stack stage by stage; return stage metrics."""
    solver = CPM_pysat()
    stages = {}

    cpm_cons = list(model.constraints)
    record_stage(stages, "input", cpm_cons, 0.0)

    t0 = time.perf_counter()
    cpm_cons = toplevel_list(cpm_cons)
    record_stage(stages, "toplevel_list", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
    record_stage(stages, "no_partial_functions", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = push_down_negation(cpm_cons)
    record_stage(stages, "push_down_negation", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = decompose_linear(
        cpm_cons,
        supported=solver.supported_global_constraints,
        supported_reified=solver.supported_reified_global_constraints,
        csemap=solver._csemap,
    )
    record_stage(stages, "decompose_linear", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = simplify_boolean(cpm_cons)
    record_stage(stages, "simplify_boolean", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = flatten_constraint(cpm_cons, csemap=solver._csemap)
    record_stage(stages, "flatten_constraint", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = linearize_reified_variables(
        cpm_cons, min_values=2, csemap=solver._csemap, ivarmap=solver.ivarmap
    )
    record_stage(stages, "linearize_reified_variables", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = only_bv_reifies(cpm_cons, csemap=solver._csemap)
    record_stage(stages, "only_bv_reifies", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = only_implies(cpm_cons, csemap=solver._csemap)
    record_stage(stages, "only_implies", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = linearize_constraint(
        cpm_cons,
        supported=frozenset({"sum", "wsum", "->", "and", "or"}),
        csemap=solver._csemap,
    )
    record_stage(stages, "linearize_constraint", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = int2bool(cpm_cons, solver.ivarmap, encoding=solver.encoding, csemap=solver._csemap)
    record_stage(stages, "int2bool", cpm_cons, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpm_cons = only_positive_coefficients(cpm_cons)
    record_stage(stages, "only_positive_coefficients", cpm_cons, time.perf_counter() - t0)

    return stages


def do_experiment(model_path):
    """Load one model, run the transformation pipeline, return a JSON-serializable record."""
    cpmpy_commit, cpmpy_commit_message = cpmpy_git_info()
    record = {
        "model": os.path.basename(model_path),
        "model_path": os.path.abspath(model_path),
        "cpmpy_commit": cpmpy_commit,
        "cpmpy_commit_message": cpmpy_commit_message,
        "stages": None,
        "total_runtime": None,
        "error": None,
    }

    try:
        model = Model.from_file(model_path)
        t0 = time.perf_counter()
        stages = run_transformation(model)
        record["stages"] = stages
        record["total_runtime"] = time.perf_counter() - t0
    except Exception as e:
        gc.collect()
        print("[experiment_transformation] error: {}".format(e), file=sys.stderr)
        record["error"] = "{}: {}".format(type(e).__name__, e)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the PySAT transformation stack on one pickled model.")
    parser.add_argument("--model-path", required=True, help="path to model.pickle")
    parser.add_argument("--out", default=None, help="write JSON record to this path")
    parser.add_argument("--memory-limit", type=int, default=None,
                        dest="memory_limit_gb", help="memory cap in gigabytes (RLIMIT_AS)")
    args = parser.parse_args()

    if args.memory_limit_gb is not None:
        print("Setting memory limit to {} GB".format(args.memory_limit_gb), file=sys.stderr)
        limit_bytes = args.memory_limit_gb * 1024 * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))
        except (ValueError, OSError) as e:
            print("Warning: could not set memory limit: {}".format(e), file=sys.stderr)

    record = do_experiment(args.model_path)

    if args.out is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        out_name = "{}_{}.json".format(model_name, int(time.time() * 1000))
        out_path = os.path.join(OUTPUT_DIR, out_name)
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print("[experiment_transformation] wrote record to {}".format(out_path), file=sys.stderr)
    print(json.dumps(record, indent=2))
