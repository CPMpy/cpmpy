"""
    CLI script for the XCSP3 competition.
"""
import argparse
import signal
import time
import sys, os
import random
import resource
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pathlib
from dataclasses import dataclass

# sys.path.insert(1, os.path.join(pathlib.Path(__file__).parent.resolve(), "..", ".."))

# CPMpy
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus

# PyCSP3
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3

# Utils
from solution import solution_xml
from callbacks import CallbacksCPMPy

# Configuration
SUPPORTED_SOLVERS = ["choco", "ortools", "exact", "z3", "minizinc"]
SUPPORTED_SUBSOLVERS = {
    "minizinc": ["gecode", "chuffed", "scip"]
}
DEFAULT_SOLVER = "ortools"
TIME_BUFFER = 1 # seconds
# TODO : see if good value
MEMORY_BUFFER_SOFT = 20 # MiB
MEMORY_BUFFER_HARD = 2 # MiB


def sigterm_handler(_signo, _stack_frame):
    """
        Handles a SIGTERM. Gives us 1 second to finish the current job before we get killed.
    """
    # Report that we haven't found a solution in time
    print_status(ExitStatus.unknown)
    print_comment("SIGTERM raised.")
    print(flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------- #
#                            XCSP3 Output formatting                           #
# ---------------------------------------------------------------------------- #

def status_line_start() -> str:
    return 's' + chr(32)

def value_line_start() -> str:
    return 'v' + chr(32)

def objective_line_start() -> str:
    return 'o' + chr(32)

def comment_line_start() -> str:
    return 'c' + chr(32)

class ExitStatus(Enum):
    unsupported:str = "UNSUPPORTED" # instance contains a unsupported feature (e.g. a unsupported global constraint)
    sat:str = "SATISFIABLE" # CSP : found a solution | COP : found a solution but couldn't prove optimality
    optimal:str = "OPTIMUM" + chr(32) + "FOUND" # optimal COP solution found
    unsat:str = "UNSATISFIABLE" # instance is unsatisfiable
    unknown:str = "UNKNOWN" # any other case

def print_status(status: ExitStatus) -> None:
    print(status_line_start() + status.value, end="\n", flush=True)

def print_value(value: str) -> None:
    value = value[:-2].replace("\n", "\nv" + chr(32)) + value[-2:]
    print(value_line_start() + value, end="\n", flush=True)

def print_objective(objective: int) -> None:
    print(objective_line_start() + str(objective), end="\n", flush=True)

def print_comment(comment: str) -> None:
    print(comment_line_start() + comment.rstrip('\n'), end="\r\n", flush=True)

# ---------------------------------------------------------------------------- #
#                          CLI argument type checkers                          #
# ---------------------------------------------------------------------------- #

def dir_path(path):
    if os.path.isfile(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def is_supported_solver(solver:Optional[str]):
    if (solver is not None) and (solver not in SUPPORTED_SOLVERS):
        return False    
    else:
        return True
    
def is_supported_subsolver(solver, subsolver:Optional[str]):
    if (subsolver is not None) and (subsolver not in SUPPORTED_SUBSOLVERS[solver]):
        return False
    else:
        return True

def supported_solver(solver:Optional[str]):
    if is_supported_solver(solver):
        argparse.ArgumentTypeError(f"solver:{solver} is not a supported solver. Options are: {str(SUPPORTED_SOLVERS)}")
    else:
        return solver
    

# ---------------------------------------------------------------------------- #
#                         Executable & Solver arguments                        #
# ---------------------------------------------------------------------------- #
@dataclass
class Args:
    benchname:str
    seed:int=None
    time_limit:int=None
    mem_limit:int=None
    cores:int=1
    tmpdir:os.PathLike=None
    dir:os.PathLike=None
    solver:str=DEFAULT_SOLVER
    subsolver:str=None
    time_buffer:int=TIME_BUFFER
    intermediate:bool=False


    def __post_init__(self):
        if self.dir is not None:
            self.dir = dir_path(self.dir)
        if not is_supported_solver(self.solver):
            raise(ValueError(f"solver:{self.solver} is not a supported solver. Options are: {str(SUPPORTED_SOLVERS)}"))
        if not is_supported_subsolver(self.solver, self.subsolver):
            raise(ValueError(f"subsolver:{self.subsolver} is not a supported subsolver for solver {self.solver}. Options are: {str(SUPPORTED_SUBSOLVERS[self.solver])}"))

    @staticmethod
    def from_cli(args):
        return Args(
            benchname = args.benchname,
            seed = args.seed,
            time_limit = args.time_limit if args.time_limit is not None else args.time_out,
            mem_limit = args.mem_limit,
            cores = args.cores if args.cores is not None else 1,
            tmpdir = args.tmpdir,
            dir = args.dir,
            solver = args.solver if args.solver is not None else DEFAULT_SOLVER,
            subsolver = args.subsolver if args.subsolver is not None else None,
            time_buffer = args.time_buffer if args.time_buffer is not None else TIME_BUFFER,
            intermediate = args.intermediate if args.intermediate is not None else False,
        )

    def __str__(self):
        return f"Args(benchname='{self.benchname}', seed={self.seed}, time_limit={self.time_limit}[s], mem_limit={self.mem_limit}[MiB], cores={self.cores}, tmpdir='{self.tmpdir}', dir='{self.dir})"

def choco_arguments(args: Args): 
    return {}

def ortools_arguments(args: Args):

    from ortools.sat.python import cp_model as ort
    class OrtSolutionCallback(ort.CpSolverSolutionCallback):
        """
            For intermediate objective printing.
        """

        def __init__(self):
            super().__init__()
            self.__start_time = time.time()
            self.__solution_count = 1

        def on_solution_callback(self):
            """Called on each new solution."""
            
            current_time = time.time()
            obj = int(self.ObjectiveValue())
            print_comment('Solution %i, time = %0.2fs' % 
                        (self.__solution_count, current_time - self.__start_time))
            print_objective(obj)
            self.__solution_count += 1
        

        def solution_count(self):
            """Returns the number of solutions found."""
            return self.__solution_count
        
    # https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto
    res = {
        "num_search_workers": args.cores,
    }
    if args.seed is not None: 
        res |= { "random_seed": args.seed }
    if args.intermediate:
        res |= { "solution_callback": OrtSolutionCallback() }
    return res
        
def exact_arguments(args: Args):
    return {}

def z3_arguments(args: Args):
    return {}

def minizinc_arguments(args: Args):
    return {}

def solver_arguments(args: Args):
    if args.solver == "ortools": return ortools_arguments(args)
    elif args.solver == "exact": return exact_arguments(args)
    elif args.solver == "choco": return choco_arguments(args)
    elif args.solver == "z3": return z3_arguments(args)
    elif args.solver == "minizinc": return minizinc_arguments(args)
    else: raise()

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

def main():

    # ------------------------------ Argument parsing ------------------------------ #

    parser = argparse.ArgumentParser(
                        prog='CPMpy-XCSP-Executable',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    
    # File containing the XCSP3 instance to solve
    # - not clear if we only need to support one of these
    parser.add_argument("benchname", type=dir_path) # BENCHNAME: Name of the file with path and extension
    # parser.add_argument("benchnamenoext") # BENCHNAMENOEXT: Name of the file with path, but without extension
    # parser.add_argument("benchnamenopath") # BENCHNAMENOPATH: Name of the file without path, but with extension
    # parser.add_argument("namenppathnoext") # NAMENOPATHNOEXT: Name of the file without path and without extension
    # RANDOMSEED: Seed between 0 and 4294967295
    parser.add_argument("-s", "--seed", required=False, type=int)
    # Total CPU time in seconds (before it gets killed)
    parser.add_argument("-l", "--time-limit", required=False, type=int) # TIMELIMIT
    parser.add_argument("-o", "--time-out", required=False, type=int) # TIMEOUT
    # MEMLIMIT: Total amount of memory in MiB (mebibyte = 1024 * 1024 bytes)
    parser.add_argument("-m", "--mem-limit", required=False, type=int)
    # NBCORE: Number of processing units (can by any of the following: a processor / a processor core / logical processor (hyper-threading))
    parser.add_argument("-c", "--cores", required=False, type=int)
    # TMPDIR: Only location where temporary read/write is allowed
    parser.add_argument("-t","--tmpdir", required=False, type=dir_path)
    # DIR: Name of the directory where the solver files will be stored (where this script is located?)
    parser.add_argument("-d", "--dir", required=False, type=dir_path)
    # The underlying solver which should be used
    parser.add_argument("--solver", required=False, type=supported_solver)
    # The underlying subsolver which should be used
    parser.add_argument("--subsolver", required=False, type=str)
    # How much time before SIGTERM should we halt solver (for the final post-processing steps and solution printing)
    parser.add_argument("--time_buffer", required=False, type=int)
    # If intermediate solutions should be printed (if the solver supports it)
    parser.add_argument("--intermediate", action=argparse.BooleanOptionalAction)

    # Process cli arguments
    args = Args.from_cli(parser.parse_args())
    print_comment(str(args))
    
    run(args)

def run(args: Args):

    # --------------------------- Global Configuration --------------------------- #

    if args.seed is not None:
        random.seed(args.seed)
    if args.mem_limit is not None:
        # TODO : validate if it works
        soft = (args.mem_limit - MEMORY_BUFFER_SOFT) * 1024 * 1024 # MiB to bytes
        hard = (args.mem_limit - MEMORY_BUFFER_HARD) * 1024 * 1024 # MiB to bytes
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    sys.argv = ["-nocompile"] # Stop pyxcsp3 from complaining on exit

    # ------------------------------ Parse instance ------------------------------ #

    start = time.time()
    parser = ParserXCSP3(args.benchname)
    print_comment(f"took {(time.time() - start):.4f} seconds to parse XCSP3 model")

    # -------------------------- Configure XCSP3 parser callbacks -------------------------- #
    start = time.time()
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)

    try:
        callbacker.load_instance()
    except NotImplementedError as e:
        print_status(ExitStatus.unsupported)
        print_comment(str(e))
        exit(1)
    except Exception as e:
        print_status(ExitStatus.unknown)
        print_comment(str(e))
        exit(1)
    print_comment(f"took {(time.time() - start):.4f} seconds to convert to CPMpy model")
    
    # ------------------------------ Solve instance ------------------------------ #

    # CPMpy model
    model = callbacks.cpm_model

    # Transfer model to solver
    start = time.time()
    s = cp.SolverLookup.get(args.solver + ((":" + args.subsolver) if args.subsolver is not None else ""), model)
    transfer_time = time.time() - start
    print_comment(f"took {transfer_time:.4f} seconds to transfer model to {args.solver}")

    # Solve model
    start = time.time()
    sat = s.solve(
        time_limit = args.time_limit - transfer_time - args.time_buffer if args.time_limit is not None else None,
        **solver_arguments(args)
    ) 
    print_comment(f"took {(time.time() - start):.4f} seconds to solve")

    # ------------------------------- Print result ------------------------------- #

    if s.status().exitstatus == CPMStatus.OPTIMAL:
        print_status(ExitStatus.optimal)
        print_value(solution_xml(callbacks.cpm_variables, s))
    elif s.status().exitstatus == CPMStatus.FEASIBLE:
        print_status(ExitStatus.sat)
        print_value(solution_xml(callbacks.cpm_variables, s))
    elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
        print_status(ExitStatus.unsat)
    elif s.status().exitstatus == CPMStatus.UNKNOWN:
        print_status(ExitStatus.unknown)
    else:
        print_status(ExitStatus.unknown)
      
    
if __name__ == "__main__":
    # Configure signal handles
    # signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Main program
    main()
