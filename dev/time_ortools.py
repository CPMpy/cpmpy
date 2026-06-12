#!/usr/bin/env python3
# ruff: noqa: E402
"""
Time OR-Tools transformation and posting steps on XCSP3 instances.

Iterates over XCSP3 instances (using cpmpy.tools.xcsp3), runs the same
transformation pipeline as CPM_ortools.transform() step by step, and records
the time for each phase. It also times `post_constraints`, which measures only
constraint posting to the OR-Tools model using already transformed constraints.

Output: a single-row CSV with columns date_finished, git_tag, then one column
per step (aggregated total seconds), step order = order of first appearance in records.
Also writes time_ortools.records.csv in the same directory: one row per instance, columns instance
then one per step (same order as in output), values are times for that instance; updated after every instance.

Usage:
  python dev/time_ortools.py [--year YEAR] [--track TRACK] [-o OUTPUT] [--limit LIMIT]
                             [--offset OFFSET] [--stop-after STEPNAME]
                             [--instances-per-problem N] [-j WORKERS]
                             [--download] [--data PATH] [-v|--verbose]
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pathlib
import sys
import traceback
import time

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.time_transformations import _calc_stats, _write_records, _write_output, step_order_from_records

from cpmpy.tools.xcsp3 import XCSP3Dataset, _parse_xcsp3, _load_xcsp3, decompress_lzma
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_bv_reifies, only_implies, reify_rewrite
from cpmpy.transformations.comparison import only_numexpr_equality


def _process_instance(args):
    """Process a single instance; returns (records, stats) or (None, error_msg) on failure.
    Must be a top-level function for ProcessPoolExecutor pickling.
    stats: dict with 'runtime', 'n_constraints', 'n_variables'."""
    path, metadata, track, stop_after = args
    instance_id = f"{track}/{metadata['name']}"
    start_time = time.perf_counter()

    if str(path).endswith(".lzma"):
        path = decompress_lzma(path)

    try:
        records = []
        step = "parse_xcsp3"
        t0 = time.perf_counter()
        parser = _parse_xcsp3(path)  # type: ignore
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, {"runtime": time.perf_counter() - start_time, "n_constraints": None, "n_variables": None}
        step = "create_model"
        t0 = time.perf_counter()
        model = _load_xcsp3(parser)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, model.constraints)
    except Exception as e:
        return (None, str(e))  # (None, error_msg) for load failure

    solver = CPM_ortools(cpm_model=None)
    cpm_expr = model.constraints

    try:
        step = "toplevel_list"
        t0 = time.perf_counter()
        cpm_expr = toplevel_list(cpm_expr)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "no_partial_functions"
        t0 = time.perf_counter()
        cpm_expr = no_partial_functions(cpm_expr, safen_toplevel=frozenset({"div", "mod"}))
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "decompose_in_tree"
        t0 = time.perf_counter()
        cpm_expr = decompose_in_tree(
            cpm_expr,
            supported=solver.supported_global_constraints,
            supported_reified=solver.supported_reified_global_constraints,
            csemap=solver._csemap,
        )
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "push_down_negation"
        t0 = time.perf_counter()
        cpm_expr = push_down_negation(cpm_expr)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "flatten_constraint"
        t0 = time.perf_counter()
        cpm_expr = flatten_constraint(cpm_expr, csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "reify_rewrite"
        t0 = time.perf_counter()
        cpm_expr = reify_rewrite(cpm_expr, supported=frozenset({"sum", "wsum"}), csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "only_numexpr_equality"
        t0 = time.perf_counter()
        cpm_expr = only_numexpr_equality(
            cpm_expr,
            supported=frozenset({"sum", "wsum", "sub"}),
            csemap=solver._csemap,
        )
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "only_bv_reifies"
        t0 = time.perf_counter()
        cpm_expr = only_bv_reifies(cpm_expr, csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "only_implies"
        t0 = time.perf_counter()
        cpm_expr = only_implies(cpm_expr, csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        # Posting-only timing: transformed constraints are already computed.
        step = "post_constraints"
        t0 = time.perf_counter()
        for con in cpm_expr:
            solver._post_constraint(con)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        # Calculate final stats
        return records, _calc_stats(start_time, cpm_expr)
    except Exception:
        # Capture full traceback for transform failures
        tb_str = traceback.format_exc()
        return (None, tb_str)  # (None, traceback_string) for transform failure


def time_ortools(dataset, output, limit, offset=0, stop_after=None, instances_per_problem=1, workers=1, verbose=False):
    """Run transformation timing over the dataset; aggregate and append results to output CSV."""
    track = getattr(dataset, "track", "default")
    output_path = pathlib.Path(output)
    records_path = output_path.parent / "time_ortools.records.csv"

    records = []  # list of (instance, step, time)
    records_cols = ["instance", "step", "time"]
    n_failed = 0

    instances = list(dataset)
    prefix_count = defaultdict(int)
    filtered = []
    for path, meta in instances:
        prefix = meta["name"].split("-")[0]
        if prefix_count[prefix] < instances_per_problem:
            prefix_count[prefix] += 1
            filtered.append((path, meta))
    filtered = filtered[offset : offset + limit] if limit is not None else filtered[offset:]

    # Read existing output now so we can append our cumulative row (preserves on interrupt)
    existing_output = None
    if output_path.exists():
        try:
            existing_output = pd.read_csv(output_path)
        except Exception:
            existing_output = None

    task_args = [(path, metadata, track, stop_after) for path, metadata in filtered]

    def _format_stats(stats):
        """Format stats for printing."""
        runtime_str = f"{stats['runtime']:.2f}s" if stats['runtime'] is not None else "N/A"
        constraints_str = f"{stats['n_constraints']:,}" if stats['n_constraints'] is not None else "N/A"
        variables_str = f"{stats['n_variables']:,}" if stats['n_variables'] is not None else "N/A"
        return f"runtime={runtime_str}, constraints={constraints_str}, variables={variables_str}"

    def _handle_completed_instance(metadata, result):
        """Shared post-processing for sequential and parallel execution."""
        nonlocal n_failed
        if isinstance(result, tuple) and result[0] is None:
            n_failed += 1
            if result[1]:
                if "\n" in result[1]:
                    print(f"Transform failed {track}/{metadata['name']}:", file=sys.stderr)
                    print(result[1], file=sys.stderr)
                else:
                    print(f"Load failed {track}/{metadata['name']}: {result[1]}", file=sys.stderr)
            else:
                print(f"Transform failed {track}/{metadata['name']}", file=sys.stderr)
            return

        records_list, stats = result
        records.extend(records_list)
        _write_records(records, records_path, records_cols)
        _write_output(records, output_path, records_cols, existing_output)
        if verbose:
            print(f"Completed transforming {track}/{metadata['name']} ({_format_stats(stats)})")
        else:
            print(f"Completed transforming {track}/{metadata['name']}")

    if workers <= 1:
        # Sequential: no pool, process one by one, write after each
        for path, metadata in filtered:
            result = _process_instance((path, metadata, track, stop_after))
            _handle_completed_instance(metadata, result)
    else:
        # Parallel: each worker runs in its own process (no shared state)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_meta = {pool.submit(_process_instance, args): args for args in task_args}
            for future in as_completed(future_to_meta):
                _, metadata, _, _ = future_to_meta[future]
                try:
                    result = future.result()
                except Exception as e:
                    n_failed += 1
                    print(f"Worker failed {track}/{metadata['name']}: {e}", file=sys.stderr)
                    continue
                _handle_completed_instance(metadata, result)

    # Final write (redundant if loop ran, but ensures we have output if no instances completed)
    if records:
        _write_output(records, output_path, records_cols, existing_output)
    step_order = step_order_from_records(records)
    if step_order:
        final_step = step_order[-1]
        n_ok = len({inst for inst, step, _ in records if step == final_step})
    else:
        n_ok = 0
    print(f"Wrote 1 row ({n_ok} instances) to {output_path}")
    if n_failed:
        print(f"({n_failed} instances failed load or transform)", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time OR-Tools transformations and posting on XCSP3 instances (no solving).")
    parser.add_argument("--year", type=int, default=2024, help="Year of the dataset (default 2024)")
    parser.add_argument("--track", type=str, default="COP", help="XCSP3 track (e.g. COP, MiniCOP)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("time_ortools.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of instances to process (default None)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many instances before starting")
    parser.add_argument("--stop-after", type=str, default=None, metavar="STEPNAME",
                        help="Stop after this step (e.g. flatten_constraint or post_constraints)")
    parser.add_argument("--instances-per-problem", type=int, default=1, metavar="N",
                        help="Max instances per problem type (prefix before first '-'); default 1")
    parser.add_argument("-j", "--workers", type=int, default=1, metavar="N",
                        help="Number of parallel workers (default 1). Each worker runs in a separate process.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-instance runtime/constraints/variables stats.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if it doesn't exist")
    parser.add_argument("--data", type=pathlib.Path, default=None, help="Path to the dataset. If combined with --download, the dataset will be downloaded to this path. If not provided, the dataset will be downloaded to / looked for in the current working directory.")
    args = parser.parse_args()

    root = args.data.resolve() if args.data else pathlib.Path(".").resolve()
    year = args.year
    track = args.track

    if not CPM_ortools.supported():
        print("ortools is not available; install cpmpy[ortools].", file=sys.stderr)
        sys.exit(1)

    try:
        dataset = XCSP3Dataset(root=str(root), year=year, track=track, download=args.download)
    except ValueError as e:
        print(f"Dataset not found for track {track}: {e}", file=sys.stderr)
        print("Use option --download to download the dataset", file=sys.stderr)
        sys.exit(1)

    time_ortools(
        dataset,
        args.output,
        args.limit,
        args.offset,
        args.stop_after,
        args.instances_per_problem,
        args.workers,
        args.verbose,
    )
