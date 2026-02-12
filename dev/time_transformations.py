#!/usr/bin/env python3
"""
Time each transformation used in the pysat solver on XCSP3 instances.

Iterates over XCSP3 instances (using cpmpy.tools.xcsp3), runs the same
transformation pipeline as CPM_pysat.transform() step by step, and records
the time for each transformation. Does not solve the models.

Output: a single-row CSV with columns date_finished, git_tag, then one column
per step (aggregated total seconds), step order = order of first appearance in records.
Also writes time_transformations.records.csv in the same directory: one row per instance, columns instance
then one per step (same order as in output), values are times for that instance; updated after every instance. Usage:
  python dev/time_pysat_transformations_xcsp3.py [--root PATH] [--year YEAR] [--track TRACK] [--output CSV]
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from queue import Empty

import pandas as pd
from tqdm import tqdm

from multiprocessing import Manager

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cpmpy as cp
from cpmpy.tools.xcsp3 import XCSP3Dataset, _parse_xcsp3, _load_xcsp3, decompress_lzma
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.normalize import toplevel_list, simplify_boolean
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_bv_reifies, only_implies
from cpmpy.transformations.linearize import linearize_constraint, only_positive_coefficients
from cpmpy.transformations.int2bool import int2bool


def _count_constraints(expr):
    """Return number of constraints in expression/list; None if unknown."""
    if expr is None:
        return None
    if isinstance(expr, (list, tuple)):
        return len(expr)
    return 1


def _progress_put(queue, instance_id, step, n_constraints=None):
    """Send progress update to queue; non-blocking to avoid slowing workers."""
    if queue is not None:
        try:
            queue.put_nowait((instance_id, step, n_constraints))
        except Exception:
            pass


def _process_instance(args):
    """Process a single instance; returns list of (instance_id, step, time) or None on failure.
    Must be a top-level function for ProcessPoolExecutor pickling.
    args: (path, metadata, track, stop_after) or (path, metadata, track, stop_after, queue)."""
    if len(args) == 5:
        path, metadata, track, stop_after, queue = args
    else:
        path, metadata, track, stop_after = args
        queue = None

    instance_id = f"{track}/{metadata['name']}"

    if str(path).endswith(".lzma"):
        path = decompress_lzma(path)

    _progress_put(queue, instance_id, "parse_xcsp3", None)  # show we started
    try:
        records = []
        step = "parse_xcsp3"
        t0 = time.perf_counter()
        parser = _parse_xcsp3(path)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, None)  # no constraint count after parse
        if stop_after and step == stop_after:
            return records
        step = "create_model"
        t0 = time.perf_counter()
        model = _load_xcsp3(parser)
        records.append((instance_id, step, time.perf_counter() - t0))
        original_count = _count_constraints(model.constraints)
        _progress_put(queue, instance_id, step, original_count)
        if stop_after and step == stop_after:
            return records
    except Exception as e:
        return (None, str(e))  # (None, error_msg) for load failure

    solver = CPM_pysat(cpm_model=None)
    cpm_expr = model.constraints
    
    # Show original model constraint count before transformations begin
    _progress_put(queue, instance_id, "original_model", original_count)

    try:
        step = "toplevel_list"
        t0 = time.perf_counter()
        cpm_expr = toplevel_list(cpm_expr)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "no_partial_functions"
        t0 = time.perf_counter()
        cpm_expr = no_partial_functions(cpm_expr, safen_toplevel={"div", "mod", "element"})
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "decompose_in_tree"
        t0 = time.perf_counter()
        cpm_expr = decompose_in_tree(
            cpm_expr,
            supported=solver.supported_global_constraints | {"alldifferent"},
            supported_reified=solver.supported_reified_global_constraints,
            csemap=solver._csemap,
        )
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "flatten_constraint"
        t0 = time.perf_counter()
        cpm_expr = flatten_constraint(cpm_expr, csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "only_bv_reifies"
        t0 = time.perf_counter()
        cpm_expr = only_bv_reifies(cpm_expr, csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "only_implies"
        t0 = time.perf_counter()
        cpm_expr = only_implies(cpm_expr, csemap=solver._csemap)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "linearize_constraint"
        t0 = time.perf_counter()
        cpm_expr = linearize_constraint(
            cpm_expr,
            supported=frozenset({"sum", "wsum", "->", "and", "or"}),
            csemap=solver._csemap,
        )
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "int2bool"
        t0 = time.perf_counter()
        cpm_expr = int2bool(cpm_expr, solver.ivarmap, encoding=solver.encoding)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        step = "only_positive_coefficients"
        t0 = time.perf_counter()
        cpm_expr = only_positive_coefficients(cpm_expr)
        records.append((instance_id, step, time.perf_counter() - t0))
        _progress_put(queue, instance_id, step, _count_constraints(cpm_expr))
        if stop_after and step == stop_after:
            return records

        return records
    except Exception:
        return (None, None)  # (None, None) for transform failure


def _format_short_name(name, max_len=35):
    """Truncate instance name for display."""
    return name if len(name) <= max_len else name[: max_len - 3] + "…"


def _format_elapsed(seconds):
    """Format elapsed seconds as human-readable string (e.g. 2.3s, 1m 23s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m}m {s}s" if s else f"{m}m"


def _refresh_progress_display(main_pbar, worker_bars, active_instances, n_workers):
    """Update worker bars to show active instances, current step, constraint count, and elapsed time."""
    now = time.time()
    sorted_active = sorted(active_instances.items(), key=lambda x: x[0])[:n_workers]
    # Fixed column widths to prevent jumping
    INSTANCE_WIDTH = 35
    STEP_WIDTH = 25
    CONSTRAINTS_WIDTH = 18
    ELAPSED_WIDTH = 8
    
    for i, pbar in enumerate(worker_bars):
        if i < len(sorted_active):
            inst_id, (step, start_time, n_constraints) = sorted_active[i]
            short = _format_short_name(inst_id.split("/")[-1] if "/" in inst_id else inst_id, INSTANCE_WIDTH)
            elapsed = _format_elapsed(now - start_time)
            
            # Format with fixed widths
            instance_col = f"{short:<{INSTANCE_WIDTH}}"
            step_col = f"{step:<{STEP_WIDTH}}"
            # Constraint count: right-aligned number (15 chars) + " constraints" (11 chars) = 26 chars total
            CONSTRAINTS_FULL_WIDTH = 27
            if n_constraints is not None:
                constraints_col = f"{n_constraints:>15,} constraints"
            else:
                constraints_col = " " * CONSTRAINTS_FULL_WIDTH
            elapsed_col = f"{elapsed:>{ELAPSED_WIDTH}}"
            
            pbar.set_description_str(
                f"  ▸ {instance_col} │ {step_col} │ {constraints_col} │ ({elapsed_col})",
                refresh=True
            )
            pbar.n = 0
            pbar.total = 1
            pbar.refresh()
        else:
            pbar.set_description_str("  ○ idle", refresh=True)
            pbar.refresh()


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


def time_transformations(dataset, output, limit, offset=0, stop_after=None, instances_per_problem=1, workers=1):
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

    total = len(filtered)
    n_workers = max(1, workers)
    use_progress_bars = total > 0

    # Read existing output now so we can append our cumulative row (preserves on interrupt)
    existing_output = None
    if output_path.exists():
        try:
            existing_output = pd.read_csv(output_path)
        except Exception:
            existing_output = None

    # Build task args; add progress queue when we want multi-bar display
    with Manager() as manager:
        progress_queue = manager.Queue(maxsize=512) if use_progress_bars else None
        task_args = [
            (path, metadata, track, stop_after, progress_queue) if progress_queue else (path, metadata, track, stop_after)
            for path, metadata in filtered
        ]

        if workers <= 1:
            executor_cls = ThreadPoolExecutor
            max_workers = 1
        else:
            executor_cls = ProcessPoolExecutor
            max_workers = workers

        active_instances = {}  # instance_id -> current step
        worker_bars = []

        with executor_cls(max_workers=max_workers) as pool:
            future_to_args = {pool.submit(_process_instance, args): args for args in task_args}
            pending = set(future_to_args)

            # Main progress bar (overall)
            main_pbar = tqdm(
                total=total,
                desc=track,
                unit="instance",
                position=0,
                leave=True,
                ncols=100,
            )

            # Worker status bars (UV-style: one per worker slot below main bar)
            if use_progress_bars and progress_queue is not None:
                for i in range(n_workers):
                    wb = tqdm(
                        total=1,
                        initial=0,
                        desc="  ○ idle",
                        position=1 + i,
                        leave=False,
                        bar_format="{desc}",
                        ncols=100,
                    )
                    worker_bars.append(wb)

            try:
                while pending:
                    # Wait for any future to complete (short timeout to poll queue).
                    # When nothing is pending (e.g. total==0), we're done.
                    done, _ = wait(pending, timeout=0.08)

                    # Drain progress queue
                    if progress_queue is not None:
                        now = time.time()
                        while True:
                            try:
                                msg = progress_queue.get_nowait()
                                instance_id = msg[0]
                                step = msg[1]
                                n_constraints = msg[2] if len(msg) >= 3 else None
                                if instance_id not in active_instances:
                                    active_instances[instance_id] = (step, now, n_constraints)
                                else:
                                    _, start_time, _ = active_instances[instance_id]
                                    active_instances[instance_id] = (step, start_time, n_constraints)
                            except Empty:
                                break

                    # Refresh worker bars
                    if worker_bars:
                        _refresh_progress_display(
                            main_pbar, worker_bars, active_instances, n_workers
                        )

                    # Process completed futures
                    for future in done:
                        pending.discard(future)
                        args = future_to_args[future]
                        _, metadata, _, _ = args[:4]
                        instance_id = f"{track}/{metadata['name']}"

                        try:
                            result = future.result()
                        except Exception as e:
                            n_failed += 1
                            print(f"Worker failed {instance_id}: {e}", file=sys.stderr)
                            main_pbar.update(1)
                            active_instances.pop(instance_id, None)
                            continue

                        if isinstance(result, tuple) and result[0] is None:
                            n_failed += 1
                            if result[1]:
                                print(f"Load failed {instance_id}: {result[1]}", file=sys.stderr)
                            else:
                                print(f"Transform failed {instance_id}", file=sys.stderr)
                            main_pbar.update(1)
                            active_instances.pop(instance_id, None)
                            continue

                        records.extend(result)
                        active_instances.pop(instance_id, None)
                        _write_records(records, records_path, records_cols)
                        _write_output(records, output_path, records_cols, existing_output)
                        main_pbar.update(1)

            finally:
                main_pbar.close()
                for wb in worker_bars:
                    wb.close()

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


class _StopAfter(BaseException):
    """Raised to exit the transformation step sequence after a given step (--stop-after)."""
    pass


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
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--track", type=str, default="COP", help="XCSP3 track (e.g. COP, MiniCOP)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("time_transformations.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many instances before starting")
    parser.add_argument("--stop-after", type=str, default=None, metavar="STEPNAME",
                        help="Stop transformation pipeline after this step (e.g. flatten_constraint)")
    parser.add_argument("--instances-per-problem", type=int, default=1, metavar="N",
                        help="Max instances per problem type (prefix before first '-'); default 1")
    parser.add_argument("-j", "--workers", type=int, default=1, metavar="N",
                        help="Number of parallel workers (default 1). Each worker runs in a separate process.")
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

    time_transformations(dataset, args.output, args.limit, args.offset, args.stop_after, args.instances_per_problem, args.workers)
