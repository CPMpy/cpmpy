"""
Deprecated batch runner for XCSP3 instances.

This module is a compatibility shim. New code should use::

    python -m cpmpy.tools.benchmark.cpbenchy.runner.run_benchmark \\
        --dataset cpmpy.tools.datasets.xcsp3.XCSP3Dataset \\
        --dataset-year YEAR --dataset-track TRACK --dataset-download \\
        --runner xcsp3 --solver SOLVER --workers N \\
        --output ./results [--xcsp3-checker-jar checker.jar]

Note: the output format has changed.  cpbenchy writes per-instance
``.txt`` and ``.metadata.json`` files under ``--output``, not the
single CSV produced by the old benchmark.py.  ``analyze.py`` continues
to work with existing CSV files.
"""

import argparse
import warnings
from typing import Optional


def xcsp3_benchmark(
    year: int,
    track: str,
    solver: str,
    workers: int = 1,
    time_limit: int = 300,
    mem_limit: Optional[int] = 4096,
    cores: int = 1,
    output_dir: str = "results",
    verbose: bool = False,
    intermediate: bool = False,
    checker_path: Optional[str] = None,
):
    """
    Deprecated. Delegates to cpbenchy's run_batch.

    Output format: per-instance .txt / .metadata.json files under output_dir
    (no longer a single CSV).
    """
    warnings.warn(
        "xcsp3_benchmark() is deprecated; use run_benchmark.py with "
        "--dataset cpmpy.tools.datasets.xcsp3.XCSP3Dataset --runner xcsp3 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    warnings.warn(
        "Output format has changed: cpbenchy writes per-instance .txt/.metadata.json "
        "files, not the legacy single CSV. analyze.py still works with existing CSVs.",
        DeprecationWarning,
        stacklevel=2,
    )

    from cpmpy.tools.datasets.xcsp3 import XCSP3Dataset
    from cpmpy.tools.benchmark.cpbenchy.runner.run_benchmark import run_batch

    dataset = XCSP3Dataset(year=year, track=track, download=True)
    instances = [str(path) for path, _ in dataset]

    observer_specs = None
    if checker_path is not None:
        observer_specs = [
            f"cpmpy.tools.benchmark.cpbenchy.observer.xcsp3_jar_checker"
            f".XCSP3JarCheckerObserver(jar_path='{checker_path}')"
        ]

    run_batch(
        instances=instances,
        runner_path="xcsp3",
        solver=solver,
        time_limit=time_limit,
        mem_limit=mem_limit,
        seed=None,
        workers=workers,
        cores_per_worker=str(cores),
        total_memory=None,
        memory_per_worker=None,
        ignore_memory_check=False,
        intermediate=intermediate,
        verbose=verbose,
        output_dir=output_dir,
        observer_specs=observer_specs,
        solution_checker=False,
        resource_manager="runexec",
        no_pin_cores=False,
    )
    # No longer returns a CSV path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "[DEPRECATED] Benchmark solvers on XCSP3 instances. "
            "Use run_benchmark.py with --dataset ... --runner xcsp3 instead."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="Competition year (e.g., 2023)")
    parser.add_argument("--track", type=str, required=True, help="Track type (e.g., COP, CSP, MiniCOP)")
    parser.add_argument("--solver", type=str, required=True, help="Solver name (e.g., ortools, exact, choco)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--time-limit", type=int, default=300, dest="time_limit", help="Time limit in seconds per instance")
    parser.add_argument("--mem-limit", type=int, default=8192, dest="mem_limit", help="Memory limit in MiB per instance")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to assign to a single instance")
    parser.add_argument("--output-dir", type=str, default="results", dest="output_dir", help="Output directory for result files")
    parser.add_argument("--verbose", action="store_true", help="Show solver output")
    parser.add_argument("--intermediate", action="store_true", help="Report on intermediate solutions")
    parser.add_argument("--checker-path", type=str, default=None, dest="checker_path",
                        help="Path to the XCSP3 solution checker JAR file")

    args = parser.parse_args()

    if not args.verbose:
        warnings.filterwarnings("ignore")

    xcsp3_benchmark(**vars(args))
