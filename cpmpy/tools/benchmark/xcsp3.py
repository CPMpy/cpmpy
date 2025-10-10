"""
XCSP3 competition as a CPMpy benchmark

This module provides a benchmarking framework for running CPMpy on XCSP3 
competition instances. It extends the generic `Benchmark` base class with
XCSP3-specific logging and result reporting.

Command-line Interface
----------------------
This script can be run directly to benchmark solvers on XCSP3 datasets.

Usage:
    python xcsp3.py --year 2024 --track CSP --solver ortools

Arguments:
    --year          Competition year (e.g., 2024).
    --track         Track type (e.g., CSP, COP).
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

    XCSP3ExitStatus
    XCSP3Benchmark

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    solution_xcsp3
"""

import warnings
import argparse
from enum import Enum
from pathlib import Path
from datetime import datetime

# CPMpy
from cpmpy.tools.benchmark.runner import benchmark_runner
from cpmpy.tools.benchmark._base import Benchmark
from cpmpy.tools.xcsp3 import read_xcsp3
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus

# PyCSP3
from xml.etree.ElementTree import ParseError
import xml.etree.cElementTree as ET


class XCSP3ExitStatus(Enum):
    unsupported:str = "UNSUPPORTED" # instance contains an unsupported feature (e.g. a unsupported global constraint)
    sat:str = "SATISFIABLE" # CSP : found a solution | COP : found a solution but couldn't prove optimality
    optimal:str = "OPTIMUM" + chr(32) + "FOUND" # optimal COP solution found
    unsat:str = "UNSATISFIABLE" # instance is unsatisfiable
    unknown:str = "UNKNOWN" # any other case

def solution_xcsp3(model, useless_style="*", boolean_style="int"):
    """
        Formats a solution according to the XCSP3 specification.

        Arguments:
            model: CPMpy model for which to format its solution (should be solved first)
            useless_style: How to process unused decision variables (with value `None`). 
                           If "*", variable is included in reporting with value "*". 
                           If "drop", variable is excluded from reporting.
            boolean_style: Print style for boolean constants.
                           "int" results in 0/1, "bool" results in False/True.

        Returns:
            XML-formatted model solution according to XCSP3 specification.
    """

    # CSP
    if not model.has_objective():
        root = ET.Element("instantiation", type="solution")
    # COP
    else:
        root = ET.Element("instantiation", type="optimum", cost=str(int(model.objective_value())))

    # How useless variables should be handled
    #    (variables which have value `None` in the solution)
    variables = {var.name: var for var in model.user_vars if var.name[:2] not in ["IV", "BV", "B#"]} # dirty workaround for all missed aux vars in user vars TODO fix with Ignace
    if useless_style == "*":
        variables = {k:(v.value() if v.value() is not None else "*") for k,v in variables.items()}
    elif useless_style == "drop":
        variables = {k:v.value() for k,v in variables.items() if v.value() is not None}

    # Convert booleans
    if boolean_style == "bool":
        pass
    elif boolean_style == "int":
        variables = {k:(v if (not isinstance(v, bool)) else (1 if v else 0)) for k,v in variables.items()}

    # Build XCSP3 XML tree
    ET.SubElement(root, "list").text=" " + " ".join([str(v) for v in variables.keys()]) + " "
    ET.SubElement(root, "values").text=" " + " ".join([str(v) for v in variables.values()]) + " "
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    res = ET.tostring(root).decode("utf-8")

    return str(res)


class XCSP3Benchmark(Benchmark):
    """
    The XCSP3 competition as a CPMpy benchmark.
    """

    def __init__(self):
        super().__init__(reader=read_xcsp3, exit_status=XCSP3ExitStatus)
    
    def print_comment(self, comment:str):
        print('c' + chr(32) + comment.rstrip('\n'), end="\r\n", flush=True)

    def print_status(self, status: XCSP3ExitStatus) -> None:
        print('s' + chr(32) + status.value, end="\n", flush=True)

    def print_value(self, value: str) -> None:
        value = value[:-2].replace("\n", "\nv" + chr(32)) + value[-2:]
        print('v' + chr(32) + value, end="\n", flush=True)

    def print_objective(self, objective: int) -> None:
        print('o' + chr(32) + str(objective), end="\n", flush=True)

    def print_intermediate(self, objective:int):
        self.print_objective(objective)

    def print_result(self, s):
        if s.status().exitstatus == CPMStatus.OPTIMAL:
            self.print_value(solution_xcsp3(s))
            self.print_status(XCSP3ExitStatus.optimal)
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            self.print_value(solution_xcsp3(s))
            self.print_status(XCSP3ExitStatus.sat)
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            self.print_status(XCSP3ExitStatus.unsat)
        else:
            self.print_comment("Solver did not find any solution within the time/memory limit")
            self.print_status(XCSP3ExitStatus.unknown)

    def handle_memory_error(self, mem_limit):
        super().handle_memory_error(mem_limit)
        self.print_status(XCSP3ExitStatus.unknown)

    def handle_not_implemented(self, e):
        super().handle_not_implemented(e)
        self.print_status(XCSP3ExitStatus.unsupported)

    def handle_exception(self, e):
        if isinstance(e, ParseError):
            if "out of memory" in e.msg:
                self.print_comment(f"MemoryError raised by parser.")
                self.print_status(XCSP3ExitStatus.unknown)
            else:
                self.print_comment(f"An {type(e)} got raised by the parser: {e}")
                self.print_status(XCSP3ExitStatus.unknown)
        else:
            super().handle_exception(e)
            self.print_status(XCSP3ExitStatus.unknown)

    def parse_output_line(self, line, result):
        if line.startswith('s '):
            result['status'] = line[2:].strip()
        elif line.startswith('v ') and result['solution'] is None:
            # only record first line, contains 'type' and 'cost'
            solution = line.split("\n")[0][2:].strip()
            result['solution'] = solution
            complete_solution = line
            if "cost" in solution:
                result['objective_value'] = solution.split('cost="')[-1][:-2]
        elif line.startswith('o '):
            obj = int(line[2:].strip())
            if result['intermediate'] is None:
                result['intermediate'] = []
            result['intermediate'] += [(sol_time, obj)]
            result['objective_value'] = obj
            obj = None
        elif line.startswith('c Solution'):
            parts = line.split(', time = ')
            # Get solution time from comment for intermediate solution -> used for annotating 'o ...' lines
            sol_time = float(parts[-1].replace('s', '').rstrip())
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

    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, required=True, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, required=True, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds per instance')
    parser.add_argument('--mem-limit', type=int, default=8192, help='Memory limit in MB per instance')
    parser.add_argument('--cores', type=int, default=1, help='Number of cores to assign tp a single instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    parser.add_argument('--intermediate', action='store_true', help='Report on intermediate solutions')
    parser.add_argument('--checker-path', type=str, default=None,
                    help='Path to the XCSP3 solution checker JAR file')
    args = parser.parse_args()

    if not args.verbose:
        warnings.filterwarnings("ignore")
    
    # Load benchmark instances (as a dataset)
    from cpmpy.tools.dataset.model.xcsp3 import XCSP3Dataset
    dataset = XCSP3Dataset(year=args.year, track=args.track, download=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current timestamp in a filename-safe format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output file path with timestamp
    output_file = str(output_dir / "xcsp3" / f"xcsp3_{args.year}_{args.track}_{args.solver}_{timestamp}.csv")

    # Run the benchmark
    instance_runner = XCSP3Benchmark()
    output_file = benchmark_runner(dataset=dataset, instance_runner=instance_runner, output_file=output_file, **vars(args))
    print(f"Results added to {output_file}")

       
