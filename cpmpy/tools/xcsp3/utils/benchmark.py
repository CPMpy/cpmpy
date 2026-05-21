"""
Deprecated runner script. Use run_benchmark.py with --dataset ... --runner xcsp3 instead.
"""

import subprocess
import logging
import sys
import warnings

warnings.warn(
    "cpmpy.tools.xcsp3.utils.benchmark is deprecated; "
    "use run_benchmark.py with --dataset cpmpy.tools.datasets.xcsp3.XCSP3Dataset --runner xcsp3 instead.",
    DeprecationWarning,
    stacklevel=2,
)

TIME_LIMIT = 60
NR_WORKERS = 20
SOLVERS = ["ortools", "exact", "choco", "z3", "minizinc", "gurobi", "exact", "cpo"]

TARGETS = {
    "2024": ["COP", "CSP", "MiniCSP", "MiniCOP"]
}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def run_with_solver(solver: str, year:str, track:str, time_limit: int = 60, workers: int=20):
    logging.info(f"Running solver: {solver}")
    command = [
        sys.executable, "-m",
        "cpmpy.tools.benchmark.cpbenchy.runner.run_benchmark",
        "--dataset", "cpmpy.tools.datasets.xcsp3.XCSP3Dataset",
        "--dataset-year", year,
        "--dataset-track", track,
        "--dataset-download",
        "--runner", "xcsp3",
        "--solver", solver,
        "--time_limit", str(time_limit),
        "--workers", str(workers),
        "--output", f"results/{year}/{track}"
    ]
    try:
        print(" ".join(command))
        subprocess.run(command)#, check=True)
        logging.info(f"Solver {solver} finished successfully.\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"Solver {solver} failed with error code {e.returncode}.\n")


for year, tracks in TARGETS.items():
    for track in tracks:
        for solver in SOLVERS:
            run_with_solver(solver, year=year, track=track, time_limit=TIME_LIMIT, workers=NR_WORKERS)