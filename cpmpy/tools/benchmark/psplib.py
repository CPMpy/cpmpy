"""
PSPLIB as a CPMpy benchmark

This module provides a benchmarking framework for running CPMpy on PSPLIB 
instances.

Command-line Interface
----------------------
This script can be run directly to benchmark solvers on PSPLIB datasets.

Usage:
    python psplib.py --year 2024 --variant rcpsp --family j30

Arguments:
    --variant       Problem variant (e.g., rcpsp).
    --family        Problem family (e.g., j30, j120, ...)
    --solver        Solver name (e.g., ortools, exact, choco, ...).
    --workers       Number of parallel workers to use.
    --time-limit    Time limit in seconds per instance.
    --mem-limit     Memory limit in MB per instance.
    --cores         Number of cores to assign to a single instance.
    --output-dir    Output directory for CSV files.
    --verbose       Show solver output if set.
    --intermediate  Report intermediate solutions if supported.

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    PSPLIBBenchmark

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    solution_psplib
"""

import warnings
import argparse
from enum import Enum
from pathlib import Path
from datetime import datetime

# CPMpy
from cpmpy.tools.benchmark.runner import benchmark_runner
from cpmpy.tools.benchmark._base import Benchmark, ExitStatus
from cpmpy.tools.rcpsp import read_rcpsp
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus


def solution_psplib(model):
    """
    Convert a CPMpy model solution into the solution string format.

    Arguments:
        model (cp.solvers.SolverInterface): The solver-specific model for which to print its solution

    Returns:
        str: formatted solution string.
    """
    variables = {var.name: var.value() for var in model.user_vars if var.name[:2] not in ["IV", "BV", "B#"]} # dirty workaround for all missed aux vars in user vars TODO fix with Ignace
    return str(variables)

class PSPLIBBenchmark(Benchmark):

    """
    PSPLIB as a CPMpy benchmark.
    """

    def __init__(self):
        self.sol_time = None
        super().__init__(reader=read_rcpsp) # TODO: reader should depend on problem variant
    
    def print_comment(self, comment:str):
        print('c' + chr(32) + comment.rstrip('\n'), end="\r\n", flush=True)

    def print_status(self, status: ExitStatus) -> None:
        print('s' + chr(32) + status.value, end="\n", flush=True)

    def print_value(self, value: str) -> None:
        print('v' + chr(32) + value, end="\n", flush=True)

    def print_objective(self, objective: int) -> None:
        print('o' + chr(32) + str(objective), end="\n", flush=True)
    
    def print_intermediate(self, objective:int):
        self.print_objective(objective)

    def print_result(self, s):
        if s.status().exitstatus == CPMStatus.OPTIMAL:
            self.print_objective(s.objective_value())
            self.print_value(solution_psplib(s))
            self.print_status(ExitStatus.optimal)
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            self.print_objective(s.objective_value())
            self.print_value(solution_psplib(s))
            self.print_status(ExitStatus.sat)
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            self.print_status(ExitStatus.unsat)
        else:
            self.print_comment("Solver did not find any solution within the time/memory limit")
            self.print_status(ExitStatus.unknown)

    def handle_memory_error(self, mem_limit):
        super().handle_memory_error(mem_limit)
        self.print_status(ExitStatus.unknown)

    def handle_not_implemented(self, e):
        super().handle_not_implemented(e)
        self.print_status(ExitStatus.unsupported)

    def handle_exception(self, e):
        super().handle_exception(e)
        self.print_status(ExitStatus.unknown)

    
    def handle_sigterm(self):
        """
        Handles a SIGTERM. Gives us 1 second to finish the current job before we get killed.
        """
        # Report that we haven't found a solution in time
        self.print_status(ExitStatus.unknown)
        self.print_comment("SIGTERM raised.")
        return 0
        
    def handle_rlimit_cpu(self):
        """
        Handles a SIGXCPU.
        """
        # Report that we haven't found a solution in time
        self.print_status(ExitStatus.unknown)
        self.print_comment("SIGXCPU raised.")
        return 0

    def parse_output_line(self, line, result):
        if line.startswith('s '):
            result['status'] = line[2:].strip()
        elif line.startswith('v '):
            # only record first line, contains 'type' and 'cost'
            solution = line.split("\n")[0][2:].strip()
            if solution not in result:
                result['solution'] = solution
            else:
                result['solution'] = result['solution'] + ' ' + str(solution)
        elif line.startswith('c Solution'):
            parts = line.split(', time = ')
            # Get solution time from comment for intermediate solution -> used for annotating 'o ...' lines
            self.sol_time = float(parts[-1].replace('s', '').rstrip())
        elif line.startswith('o '):
            obj = int(line[2:].strip())
            if result['intermediate'] is None:
                result['intermediate'] = []
            if self.sol_time is not None:
                result['intermediate'] += [(self.sol_time, obj)]
            result['objective_value'] = obj
            obj = None
        elif line.startswith('c took '):
            # Parse timing information
            parts = line.split(' seconds to ')
            if len(parts) == 2:
                time_val = float(parts[0].replace('c took ', ''))
                action = parts[1].strip()
                if action.startswith('parse'):
                    result['time_parse'] = time_val
                elif action.startswith('convert'):
                    result['time_model'] = time_val
                elif action.startswith('post'):
                    result['time_post'] = time_val
                elif action.startswith('solve'):
                    result['time_solve'] = time_val

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark solvers on PSPLIB instances')
    parser.add_argument('--variant', type=str, required=True, help='Problem variant (e.g., rcpsp)')
    parser.add_argument('--family', type=str, required=True, help='Problem family (e.g., j30, j120, ...)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds per instance')
    parser.add_argument('--mem-limit', type=int, default=8192, help='Memory limit in MB per instance')
    parser.add_argument('--cores', type=int, default=1, help='Number of cores to assign tp a single instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    parser.add_argument('--intermediate', action='store_true', help='Report on intermediate solutions')
    # parser.add_argument('--checker-path', type=str, default=None,
    #                 help='Path to the XCSP3 solution checker JAR file')
    args = parser.parse_args()

    if not args.verbose:
        warnings.filterwarnings("ignore")
    
    # Load benchmark instances (as a dataset)
    from cpmpy.tools.dataset.problem.psplib import PSPLibDataset
    dataset = PSPLibDataset(variant=args.variant, family=args.family, download=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current timestamp in a filename-safe format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output file path with timestamp
    output_file = str(output_dir / "psplib" / f"psplib_{args.variant}_{args.family}_{args.solver}_{timestamp}.csv")

    # Run the benchmark
    instance_runner = PSPLIBBenchmark()
    output_file = benchmark_runner(dataset=dataset, instance_runner=instance_runner, output_file=output_file, **vars(args))
    print(f"Results added to {output_file}")
