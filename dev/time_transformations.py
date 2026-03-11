#!/usr/bin/env python3
"""
Time each transformation used in the pysat solver on XCSP3 instances.

Iterates over XCSP3 instances (using cpmpy.tools.xcsp3), runs the same
transformation pipeline as CPM_pysat.transform() step by step, and records
the time for each transformation. Does not solve the models.

Output: a single-row CSV with columns date_finished, git_tag, then one column
per step (aggregated total seconds), step order = order of first appearance in records.
Also writes time_transformations.records.csv in the same directory: one row per instance, columns instance
then one per step (same order as in output), values are times for that instance; updated after every instance.

Usage:
  python dev/time_transformations.py [--year YEAR] [--track TRACK] [-o OUTPUT] [--limit LIMIT]
                                     [--offset OFFSET] [--stop-after STEPNAME] 
                                     [--instances-per-problem N] [-j WORKERS] 
                                     [--download] [--data PATH] [-v|--verbose]
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pathlib
import subprocess
import sys
import traceback
import time
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cpmpy as cp
from cpmpy.tools.xcsp3 import XCSP3Dataset, _parse_xcsp3, _load_xcsp3, decompress_lzma
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.normalize import toplevel_list, simplify_boolean
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_bv_reifies, only_implies
from cpmpy.transformations.linearize import linearize_constraint, only_positive_coefficients, decompose_linear
from cpmpy.transformations.int2bool import int2bool
from cpmpy.transformations.get_variables import get_variables


def _calc_stats(start_time, cpm_expr):
    """Calculate stats from current state."""
    runtime = time.perf_counter() - start_time
    n_constraints = len(cpm_expr) if isinstance(cpm_expr, (list, tuple)) else 1
    n_variables = len(get_variables(cpm_expr))
    return {"runtime": runtime, "n_constraints": n_constraints, "n_variables": n_variables}


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
        parser = _parse_xcsp3(path)
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

    solver = CPM_pysat(cpm_model=None)
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
        cpm_expr = no_partial_functions(cpm_expr, safen_toplevel={"div", "mod", "element"})
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "decompose_linear"
        t0 = time.perf_counter()
        cpm_expr = decompose_linear(
            cpm_expr,
            supported=solver.supported_global_constraints,
            supported_reified=solver.supported_reified_global_constraints,
            csemap=solver._csemap,
        )
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "flatten_constraint"
        t0 = time.perf_counter()
        cpm_expr = flatten_constraint(cpm_expr, csemap=solver._csemap)
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

        step = "linearize_constraint"
        t0 = time.perf_counter()
        cpm_expr = linearize_constraint(
            cpm_expr,
            supported=frozenset({"sum", "wsum", "->", "and", "or"}),
            csemap=solver._csemap,
        )
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        step = "int2bool"
        t0 = time.perf_counter()
        cpm_expr = int2bool(cpm_expr, solver.ivarmap, encoding=solver.encoding)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records

        step = "only_positive_coefficients"
        t0 = time.perf_counter()
        cpm_expr = only_positive_coefficients(cpm_expr)
        records.append((instance_id, step, time.perf_counter() - t0))
        if stop_after and step == stop_after:
            return records, _calc_stats(start_time, cpm_expr)

        # Calculate final stats
        return records, _calc_stats(start_time, cpm_expr)
    except Exception:
        # Capture full traceback for transform failures
        tb_str = traceback.format_exc()
        return (None, tb_str)  # (None, traceback_string) for transform failure


def _write_records(records, records_path, records_cols):
    """Write records to CSV; called from main process only to avoid concurrent writes."""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    step_order = step_order_from_records(records)
    df_flat = pd.DataFrame(records, columns=records_cols)
    df_rec = df_flat.pivot_table(index="instance", columns="step", values="time").reset_index()
    step_cols = [s for s in step_order if s in df_rec.columns]
    df_rec["total"] = df_rec[step_cols].sum(axis=1)
    rec_cols = ["instance", "total"] + step_cols
    df_rec[["total"] + step_cols] = df_rec[["total"] + step_cols].round(2)
    df_rec[rec_cols].to_csv(records_path, index=False)


def _write_output(records, output_path, records_cols, existing_df=None):
    """Write aggregated output CSV. Writes [existing_df, cumulative_row] so results are
    preserved on interrupt. Call after each instance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    step_order = step_order_from_records(records)
    df_raw = pd.DataFrame(records, columns=records_cols)
    agg = df_raw.groupby("step", as_index=False)["time"].sum()
    step_totals = agg.set_index("step")["time"]
    date_finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_tag = get_git_tag(REPO_ROOT)
    row = {"date_finished": date_finished, "git_tag": git_tag}
    row["total"] = round(step_totals.sum(), 2)
    for step in step_order:
        row[step] = round(step_totals.get(step, 0.0), 2)
    cols = ["date_finished", "git_tag", "total"] + step_order
    new_row_df = pd.DataFrame([row])[cols]
    if existing_df is not None and len(existing_df) > 0 and list(existing_df.columns) == cols:
        df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        df = new_row_df
    df = df.round(2)
    df.to_csv(output_path, index=False)


def time_transformations(dataset, output, limit, offset=0, stop_after=None, instances_per_problem=1, workers=1, verbose=False):
    """Run transformation timing over the dataset; aggregate and append results to output CSV."""
    track = getattr(dataset, "track", "default")
    output_path = pathlib.Path(output)
    records_path = output_path.parent / "time_transformations.records.csv"

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
    df_raw = pd.DataFrame(records, columns=records_cols)
    n_ok = df_raw[df_raw["step"] == step_order[-1]]["instance"].nunique() if step_order else 0
    print(f"Wrote 1 row ({n_ok} instances) to {output_path}")
    if n_failed:
        print(f"({n_failed} instances failed load or transform)", file=sys.stderr)


def get_git_tag(repo_root: pathlib.Path) -> str:
    """Return short git describe; append '+' if working tree has uncommitted changes."""
    try:
        r = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        tag = "unknown" if r.returncode != 0 else r.stdout.strip()
        if tag.endswith("-dirty"):
            tag = tag[:-5] + "+"
        return tag
    except Exception:
        return "unknown"


def step_order_from_records(records):
    """Return ordered list of unique step names (order of first appearance)."""
    step_order = []
    seen = set()
    for _, step, _ in records:
        if step not in seen:
            seen.add(step)
            step_order.append(step)
    return step_order


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time pysat transformations on XCSP3 instances (no solving).")
    parser.add_argument("--year", type=int, default=2024, help="Year of the dataset (default 2024)")
    parser.add_argument("--track", type=str, default="COP", help="XCSP3 track (e.g. COP, MiniCOP)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("time_transformations.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of instances to process (default None)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many instances before starting")
    parser.add_argument("--stop-after", type=str, default=None, metavar="STEPNAME",
                        help="Stop transformation pipeline after this step (e.g. flatten_constraint)")
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

    if not CPM_pysat.supported():
        print("pysat is not available; install cpmpy[pysat] or python-sat.", file=sys.stderr)
        sys.exit(1)

    try:
        dataset = XCSP3Dataset(root=str(root), year=year, track=track, download=args.download)
    except ValueError as e:
        print(f"Dataset not found for track {track}: {e}", file=sys.stderr)
        print("Use option --download to download the dataset", file=sys.stderr)
        sys.exit(1)

    time_transformations(
        dataset,
        args.output,
        args.limit,
        args.offset,
        args.stop_after,
        args.instances_per_problem,
        args.workers,
        args.verbose,
    )
