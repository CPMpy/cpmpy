"""
Run the benchmark for the targets (year and track) across all solvers.
"""

import subprocess
import logging
import sys

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
        sys.executable,
        "cpmpy/tools/xcsp3/xcsp3_benchmark.py",
        "--year", year,
        "--track", track,
        "--solver", solver,
        "--time-limit", str(time_limit),
        "--workers", str(workers),
        "--output-dir", f"results/{year}/{track}"
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