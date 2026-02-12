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
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

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


def time_transformations(dataset, output, limit, offset=0, stop_after=None, instances_per_problem=1):
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

    with tqdm(filtered, desc=track, unit="instance") as pbar:
        for path, metadata in pbar:
            pbar.set_postfix(instance=metadata["name"])
            instance_id = f"{track}/{metadata['name']}"

            if str(path).endswith(".lzma"):
                path = decompress_lzma(path)

            try:
                step = "parse_xcsp3"
                t0 = time.perf_counter()
                parser = _parse_xcsp3(path)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()
                step = "create_model"
                t0 = time.perf_counter()
                model = _load_xcsp3(parser)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()
            except _StopAfter:
                pass
            except Exception as e:
                print(f"Load failed {instance_id}: {e}", file=sys.stderr)
                n_failed += 1
                continue

            solver = CPM_pysat(cpm_model=None)  # needed for csemap/ivarmap, don't give it the constraints!
            cpm_expr = model.constraints

            try:
                step = "toplevel_list"
                t0 = time.perf_counter()
                cpm_expr = toplevel_list(cpm_expr)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "no_partial_functions"
                t0 = time.perf_counter()
                cpm_expr = no_partial_functions(cpm_expr, safen_toplevel={"div", "mod", "element"})
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "decompose_in_tree"
                t0 = time.perf_counter()
                cpm_expr = decompose_in_tree(
                    cpm_expr,
                    supported=solver.supported_global_constraints | {"alldifferent"},
                    supported_reified=solver.supported_reified_global_constraints,
                    csemap=solver._csemap,
                )
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "flatten_constraint"
                t0 = time.perf_counter()
                cpm_expr = flatten_constraint(cpm_expr, csemap=solver._csemap)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "only_bv_reifies"
                t0 = time.perf_counter()
                cpm_expr = only_bv_reifies(cpm_expr, csemap=solver._csemap)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "only_implies"
                t0 = time.perf_counter()
                cpm_expr = only_implies(cpm_expr, csemap=solver._csemap)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "linearize_constraint"
                t0 = time.perf_counter()
                cpm_expr = linearize_constraint(
                    cpm_expr,
                    supported=frozenset({"sum", "wsum", "->", "and", "or"}),
                    csemap=solver._csemap,
                )
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "int2bool"
                t0 = time.perf_counter()
                cpm_expr = int2bool(cpm_expr, solver.ivarmap, encoding=solver.encoding)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()

                step = "only_positive_coefficients"
                t0 = time.perf_counter()
                cpm_expr = only_positive_coefficients(cpm_expr)
                records.append((instance_id, step, time.perf_counter() - t0))
                if stop_after and step == stop_after:
                    raise _StopAfter()
            except _StopAfter:
                pass
            except Exception:
                print(f"Transform failed {instance_id}", file=sys.stderr)
                n_failed += 1
                instance_failed = True

            records_path.parent.mkdir(parents=True, exist_ok=True)
            step_order = step_order_from_records(records)
            df_flat = pd.DataFrame(records, columns=records_cols)
            df_rec = df_flat.pivot_table(index="instance", columns="step", values="time").reset_index()
            step_cols = [s for s in step_order if s in df_rec.columns]
            df_rec["total"] = df_rec[step_cols].sum(axis=1)
            rec_cols = ["instance", "total"] + step_cols
            df_rec[["total"] + step_cols] = df_rec[["total"] + step_cols].round(2)
            df_rec[rec_cols].to_csv(records_path, index=False)

    # Aggregate records to total time per step.
    df_raw = pd.DataFrame(records, columns=records_cols)
    agg = df_raw.groupby("step", as_index=False)["time"].sum()
    step_totals = agg.set_index("step")["time"]

    # Build one output row with date, git tag, total, and step totals in order of first appearance.
    date_finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_tag = get_git_tag(REPO_ROOT)
    step_order = step_order_from_records(records)
    row = {"date_finished": date_finished, "git_tag": git_tag}
    row["total"] = round(step_totals.sum(), 2)
    for step in step_order:
        row[step] = round(step_totals.get(step, 0.0), 2)
    cols = ["date_finished", "git_tag", "total"] + step_order
    new_row_df = pd.DataFrame([row])[cols]

    # Append to existing CSV if columns match, else overwrite; then write.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if list(existing.columns) == cols:
            df = pd.concat([existing, new_row_df], ignore_index=True)
        else:
            df = new_row_df
    else:
        df = new_row_df
    df = df.round(2)
    df.to_csv(output_path, index=False)
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
    default_root = pathlib.Path(cp.__file__).resolve().parent / "tools" / "xcsp3"
    parser.add_argument("--root", type=pathlib.Path, default=default_root)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--track", type=str, default="COP", help="XCSP3 track (e.g. COP, MiniCOP)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("time_transformations.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many instances before starting")
    parser.add_argument("--stop-after", type=str, default=None, metavar="STEPNAME",
                        help="Stop transformation pipeline after this step (e.g. flatten_constraint)")
    parser.add_argument("--instances-per-problem", type=int, default=1, metavar="N",
                        help="Max instances per problem type (prefix before first '-'); default 1")
    parser.add_argument("--download", action="store_true", help="Download the dataset if it doesn't exist")
    args = parser.parse_args()

    root = args.root.resolve()
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

    time_transformations(dataset, args.output, args.limit, args.offset, args.stop_after, args.instances_per_problem)
