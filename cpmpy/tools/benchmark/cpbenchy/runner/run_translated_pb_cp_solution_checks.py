#!/usr/bin/env python3
"""
Solve translated OPB instances and check PB solutions against source CP models.

This runner is local to cpbenchy. It scans a translated dataset tree produced
with VeriPB annotations, solves the OPB files with OR-Tools, and enables the
PBCPSolutionCheckerObserver to decode the PB assignment back to CP variables.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from cpmpy.tools.benchmark.cpbenchy.runner.run_benchmark import run_batch


SUPPORTED_DATASETS = frozenset({"psplib", "jsplib", "nurserostering"})


def _sidecar_path(instance_path: Path) -> Optional[Path]:
    candidates = [Path(str(instance_path) + ".meta.json")]
    if instance_path.suffix == ".xz":
        candidates.append(Path(str(instance_path).removesuffix(".xz") + ".meta.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _dataset_name(instance_path: Path) -> Optional[str]:
    sidecar_path = _sidecar_path(instance_path)
    if sidecar_path is None:
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    dataset = sidecar.get("dataset") if isinstance(sidecar.get("dataset"), dict) else {}
    name = dataset.get("name")
    return name if isinstance(name, str) else None


def iter_translated_instances(
    translated_root: Path,
    datasets: Optional[set[str]] = None,
) -> Iterable[Path]:
    selected = datasets or set(SUPPORTED_DATASETS)
    instances = {
        *translated_root.rglob("*.opb"),
        *translated_root.rglob("*.opb.xz"),
    }
    for instance_path in sorted(instances):
        dataset_name = _dataset_name(instance_path)
        if dataset_name in selected:
            yield instance_path


def _observer_spec(source_root: Path, max_violations_to_print: int) -> str:
    return (
        "PBCPSolutionCheckerObserver("
        f"source_root={str(source_root)!r},"
        f"max_violations_to_print={max_violations_to_print}"
        ")"
    )


def _group_by_parent(instances: list[Path], translated_root: Path) -> dict[Path, list[str]]:
    groups: dict[Path, list[str]] = defaultdict(list)
    for instance in instances:
        try:
            relative_parent = instance.parent.relative_to(translated_root)
        except ValueError:
            relative_parent = Path(".")
        groups[relative_parent].append(str(instance))
    return groups


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--translated-root",
        type=Path,
        default=Path("transform_PB_comp_dataset"),
        help="Root containing translated OPB files (default: transform_PB_comp_dataset)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data"),
        help="Root containing original CP dataset files (default: data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_outp"),
        help="Output directory for cpbenchy result files (default: test_outp)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset to include. Can be specified multiple times. Defaults to all supported datasets.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N translated instances")
    parser.add_argument("--solver", default="ortools", help="Solver to use (default: ortools)")
    parser.add_argument("--time_limit", type=float, default=None, help="Time limit in seconds")
    parser.add_argument("--mem_limit", type=int, default=None, help="Memory limit per run in MiB")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument(
        "--cores_per_worker",
        type=str,
        default="1",
        help="Cores per worker: count or explicit core list (default: 1)",
    )
    parser.add_argument(
        "--resource-manager",
        choices=("runexec", "python"),
        default="runexec",
        help="Resource manager to use (default: runexec)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--intermediate", action="store_true", help="Print intermediate solutions")
    parser.add_argument("--verbose", action="store_true", help="Verbose cpbenchy output")
    parser.add_argument("--no-pin-cores", action="store_true", help="Disable runexec CPU pinning")
    parser.add_argument(
        "--max-violations-to-print",
        type=int,
        default=5,
        help="Maximum CP-level violations to print per instance (default: 5)",
    )
    args = parser.parse_args(argv)

    translated_root = args.translated_root.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output.resolve()
    datasets = set(args.dataset) if args.dataset else None

    instances = list(iter_translated_instances(translated_root, datasets))
    if args.limit is not None:
        instances = instances[: args.limit]
    if not instances:
        print("No supported translated OPB instances found", file=sys.stderr)
        return 1

    observer_specs = [_observer_spec(source_root, args.max_violations_to_print)]
    grouped_instances = _group_by_parent(instances, translated_root)

    print(f"Running {len(instances)} translated OPB instance(s)")
    print(f"Translated root: {translated_root}")
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")

    for relative_parent, group in sorted(grouped_instances.items(), key=lambda item: str(item[0])):
        output_dir = output_root / relative_parent
        if args.verbose:
            print(f"Running {len(group)} instance(s) from {relative_parent} -> {output_dir}")
        run_batch(
            instances=group,
            runner_path="opb",
            solver=args.solver,
            time_limit=args.time_limit,
            mem_limit=args.mem_limit,
            seed=args.seed,
            workers=min(args.workers, len(group)),
            cores_per_worker=args.cores_per_worker,
            intermediate=args.intermediate,
            verbose=args.verbose,
            output_dir=str(output_dir),
            observer_specs=observer_specs,
            solution_checker=False,
            resource_manager=args.resource_manager,
            no_pin_cores=args.no_pin_cores,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
