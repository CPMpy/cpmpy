"""
    CLI script for the XCSP3 competition.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
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
import psutil
from io import StringIO 
from gurobipy import GurobiError
import json


# sys.path.insert(1, os.path.join(pathlib.Path(__file__).parent.resolve(), "..", ".."))

# CPMpy
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus

# PyCSP3
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3

# Utils
import os, pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve()))
from solution import solution_xml
from callbacks import CallbacksCPMPy
from xcsp3.perf_timer import PerfContext, TimerContext

# Configuration
SUPPORTED_SOLVERS = ["choco", "ortools", "exact", "z3", "minizinc", "gurobi"]
SUPPORTED_SUBSOLVERS = {
    "minizinc": ["gecode", "chuffed"]
}
DEFAULT_SOLVER = "ortools"
TIME_BUFFER = 5 # seconds
# TODO : see if good value
MEMORY_BUFFER_SOFT = 2 # MB
MEMORY_BUFFER_HARD = 2 # MMB
MEMORY_BUFFER_SOLVER = 20 # MB

def mib_as_bytes(mib: int) -> int:
    return mib * 1024 * 1024

def mb_as_bytes(mb: int) -> int:
    return mb * 1000 * 1000

def bytes_as_mb(bytes: int) -> int:
    return bytes // (1000 * 1000)

def bytes_as_gb(bytes: int) -> int:
    return bytes // (1000 * 1000 * 1000)

def bytes_as_mb_float(bytes: int) -> float:
    return bytes / (1000 * 1000)

def bytes_as_gb_float(bytes: int) -> float:
    return bytes / (1000 * 1000 * 1000)

def current_memory_usage() -> int: # returns bytes
    # not really sure which memory measurement to use: https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management/21049737#21049737
    return psutil.Process(os.getpid()).memory_info().rss

def remaining_memory(limit:int) -> int: # bytes
    return limit # - current_memory_usage()

def get_subsolver(args:Args, model:cp.Model):
    # Update subsolver in args
    if args.solver == "z3":
        if model.objective_ is not None:
            args.subsolver = "opt"
        else:
            args.subsolver = "sat"
    elif args.subsolver == None:
        if args.solver in SUPPORTED_SUBSOLVERS:
            args.subsolver = SUPPORTED_SUBSOLVERS[args.solver][0]
        else:
            pass

    return args.subsolver

original_stdout = sys.stdout

def sigterm_handler(_signo, _stack_frame):
    """
        Handles a SIGTERM. Gives us 1 second to finish the current job before we get killed.
    """
    # Report that we haven't found a solution in time
    sys.stdout = original_stdout
    print_status(ExitStatus.unknown)
    print_comment("SIGTERM raised.")
    print(flush=True)
    sys.exit(0)

def memory_error_handler(args: Args):
    sys.stdout = original_stdout
    print_status(ExitStatus.unknown)
    print_comment(f"MemoryError raised. Reached limit of {bytes_as_mb_float(args.mem_limit)} MB / {bytes_as_gb_float(args.mem_limit)} GB")
    print(flush=True)
    sys.exit(0)

def error_handler(e: Exception):
    sys.stdout = original_stdout
    print_status(ExitStatus.unknown)
    print_comment(f"An error got raised: {e}")
    print(flush=True)
    sys.exit(0)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

class Callback:

    def __init__(self, model:cp.Model, callback_function):
        self.__start_time = time.time()
        self.__solution_count = 1
        self.model = model
        self.callback_function = callback_function

    def callback(self, *args, **kwargs):
        current_time = time.time()
        try:
            obj = self.callback_function(*args, **kwargs)
            # obj = self.model.objective_value()
            print_comment('Solution %i, time = %0.2fs' % 
                        (self.__solution_count, current_time - self.__start_time))
            print_objective(obj)
            self.__solution_count += 1
        except AttributeError:
            pass

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
    if (subsolver is not None) and (subsolver not in SUPPORTED_SUBSOLVERS.get(solver, [None])):
        return False
    else:
        return True

def supported_solver(solver:Optional[str]):
    if not is_supported_solver(solver):
        argparse.ArgumentTypeError(f"solver:{solver} is not a supported solver. Options are: {str(SUPPORTED_SOLVERS)}")
    else:
        return solver
    

# ---------------------------------------------------------------------------- #
#                         Executable & Solver arguments                        #
# ---------------------------------------------------------------------------- #

@dataclass
class Args:
    benchpath:str
    benchdir:str=None
    benchname:str=None
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
    sat:bool=False
    opt:bool=False
    solve:bool=True
    profiler:bool=False

    def __post_init__(self):
        if self.dir is not None:
            self.dir = dir_path(self.dir)
        if not is_supported_solver(self.solver):
            raise(ValueError(f"solver:{self.solver} is not a supported solver. Options are: {str(SUPPORTED_SOLVERS)}"))
        # if self.subsolver is not None:
        #     if not is_supported_subsolver(self.solver, self.subsolver):
        #         raise(ValueError(f"subsolver:{self.subsolver} is not a supported subsolver for solver {self.solver}. Options are: {str(SUPPORTED_SUBSOLVERS[self.solver])}"))
        self.benchdir = os.path.join(*(str(self.benchpath).split(os.path.sep)[:-1]))
        self.benchname = str(self.benchpath).split(os.path.sep)[-1].split(".")[0]

    @staticmethod
    def from_cli(args):
        return Args(
            benchpath = args.benchname,
            seed = args.seed,
            time_limit = args.time_limit if args.time_limit is not None else args.time_out,
            mem_limit = mib_as_bytes(args.mem_limit) if args.mem_limit is not None else None, # MiB to bytes
            cores = args.cores if args.cores is not None else 1,
            tmpdir = args.tmpdir,
            dir = args.dir,
            solver = args.solver if args.solver is not None else DEFAULT_SOLVER,
            subsolver = args.subsolver if args.subsolver is not None else None,
            time_buffer = args.time_buffer if args.time_buffer is not None else TIME_BUFFER,
            intermediate = args.intermediate if args.intermediate is not None else False,
            solve = not args.only_transform,
            profiler = args.profiler,
        )
    
    @property
    def parallel(self) -> bool:
        return self.cores > 1

    def __str__(self):
        return f"Args(benchname='{self.benchname}', seed={self.seed}, time_limit={self.time_limit}[s], mem_limit={self.mem_limit}[MiB], cores={self.cores}, tmpdir='{self.tmpdir}', dir='{self.dir}, solver='{self.solver}')"

def choco_arguments(args: Args, model:cp.Model): 
    # Documentation: https://github.com/chocoteam/pychoco/blob/master/pychoco/solver.py
    return {}

def ortools_arguments(args: Args, model:cp.Model):

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
    if args.intermediate and args.opt:
        # res |= { "solution_callback": Callback(model) }
        res |= { "solution_callback": OrtSolutionCallback() }
    return res
        
def exact_arguments(args: Args, model:cp.Model):
    # Documentation: https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp?ref_type=heads
    res = {
        "seed": args.seed,
    }
    return {k:v for (k,v) in res.items() if v is not None}

def z3_arguments(args: Args, model:cp.Model):
    # Documentation: https://microsoft.github.io/z3guide/programming/Parameters/
    # -> is outdated, just let it crash and z3 will report the available options

    # Global parameters
    res = {
        
    }
    
    # Sat parameters
    if args.sat:
        res |= {
            "random_seed": args.seed,
            "threads": args.cores, # TODO what with hyperthreadding, when more threads than cores
            "max_memory": bytes_as_mb(remaining_memory(args.mem_limit)) if args.mem_limit is not None else None, # hard upper limit, given in MB
        }
    # Opt parameters
    if args.opt:
        res |= {          
            # opt does not seem to support setting max memory
            # opt does also not allow setting the random seed
        }

    return {k:v for (k,v) in res.items() if v is not None}


def minizinc_arguments(args: Args, model:cp.Model):

    # Documentation: https://minizinc-python.readthedocs.io/en/latest/api.html#minizinc.instance.Instance.solve

    res = {
        "processes": args.cores,
        "random_seed": args.seed,
    } | subsolver_arguments(args, model)

    return {k:v for (k,v) in res.items() if v is not None}

def gecode_arguments(args: Args, model:cp.Model):
    # Documentation: https://www.minizinc.org/doc-2.4.3/en/lib-gecode.html
    return {}

def chuffed_arguments(args:Args, model:cp.Model):
    # Documentation: 
    # - https://www.minizinc.org/doc-2.5.5/en/lib-chuffed.html
    # - https://github.com/chuffed/chuffed/blob/develop/chuffed/core/options.h
    return {}

def gurobi_arguments(args: Args, model:cp.Model):
    # Documentation: https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
    res = {
        "MemLimit": bytes_as_gb(remaining_memory(args.mem_limit)) if args.mem_limit is not None else None,
        "Seed": args.seed,
        "Threads": args.cores,
    }
    if args.intermediate and args.opt:
        res |= { "solution_callback": Callback(model, lambda x,_: int(x.getObjective().getValue())).callback }
    return {k:v for (k,v) in res.items() if v is not None}

def solver_arguments(args: Args, model:cp.Model):

    if model.objective_ is not None:
        args.opt = True
    else:
        args.sat = True

    if args.solver == "ortools": return ortools_arguments(args, model)
    elif args.solver == "exact": return exact_arguments(args, model)
    elif args.solver == "choco": return choco_arguments(args, model)
    elif args.solver == "z3": return z3_arguments(args, model)
    elif args.solver == "minizinc": return minizinc_arguments(args, model)
    elif args.solver == "gurobi": return gurobi_arguments(args, model)
    else: raise()

def subsolver_arguments(args: Args, model:cp.Model):
    if args.subsolver == "gecode": return gecode_arguments(args, model)
    elif args.subsolver == "chuffed": return choco_arguments(args, model)
    else: return {}

@contextmanager
def prepend_print():
    # Save the original stdout
    original_stdout = sys.stdout
    
    class PrependStream:
        def __init__(self, stream):
            self.stream = stream
        
        def write(self, message):
            # Prepend 'c' to each message before writing it
            if message.strip():  # Avoid prepending 'c' to empty messages (like newlines)
                self.stream.write('c ' + message)
            else:
                self.stream.write(message)
        
        def flush(self):
            self.stream.flush()
    
    # Override stdout with our custom stream
    sys.stdout = PrependStream(original_stdout)
    
    try:
        yield
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

def main():

    start = time.time()

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
    # Disable solving, only do transformation
    parser.add_argument("--only-transform", action=argparse.BooleanOptionalAction)
    # Enable profiling measurements
    parser.add_argument("--profiler", action=argparse.BooleanOptionalAction)

    # Process cli arguments 
    args = Args.from_cli(parser.parse_args())
    print_comment(str(args))
    
    from xml.etree.ElementTree import ParseError
    try:
        run(args)
    except MemoryError as e:
        memory_error_handler(args)
    except ParseError as e:
        if "out of memory" in e.msg:
            memory_error_handler(args)
        else:
            raise e
    except Exception as e:
        raise e

def run(args: Args):
    if args.profiler:
        perf_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "perf_stats", args.solver)
        if args.subsolver is not None:
            perf_dir = os.path.join(perf_dir, args.subsolver)
        Path(perf_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(perf_dir, args.benchname)
    else:
        path = None

    with PerfContext(path=path):
        with TimerContext("total") as tc:
            run_helper(args)
    print_comment(f"Total time taken: {tc.time}")
    

def run_helper(args:Args):
    import sys, os

    # --------------------------- Global Configuration --------------------------- #

    if args.seed is not None:
        random.seed(args.seed)
    if args.mem_limit is not None:
        # TODO : validate if it works
        soft = max(args.mem_limit - mb_as_bytes(MEMORY_BUFFER_SOFT), mb_as_bytes(MEMORY_BUFFER_SOFT))
        hard = max(args.mem_limit - mb_as_bytes(MEMORY_BUFFER_HARD), mb_as_bytes(MEMORY_BUFFER_HARD))
        print_comment(f"Setting memory limit: {soft}-{hard}")
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard)) # limit memory in number of bytes

    sys.argv = ["-nocompile"] # Stop pyxcsp3 from complaining on exit

    # ------------------------------ Parse instance ------------------------------ #

    parse_start = time.time()
    parser = ParserXCSP3(args.benchpath)
    print_comment(f"took {(time.time() - parse_start):.4f} seconds to parse XCSP3 model [{args.benchname}]")

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


    print_comment(f"took {(time.time() - start):.4f} seconds to convert to CPMpy model")
    
    # ------------------------------ Solve instance ------------------------------ #

    # CPMpy model
    model = callbacks.cpm_model

    # Subsolver
    subsolver = get_subsolver(args, model)

    # Transfer model to solver
    with prepend_print():# as output: #TODO immediately print
        with TimerContext("transform") as tc:
            if args.solver == "exact": # Exact2 takes its options at creation time
                s = cp.SolverLookup.get(args.solver + ((":" + subsolver) if subsolver is not None else ""), model, **solver_arguments(args, model))
            else:
                s = cp.SolverLookup.get(args.solver + ((":" + subsolver) if subsolver is not None else ""), model)
    # for o in output:
    #     print_comment(o)
    print_comment(f"took {tc.time:.4f} seconds to transfer model to {args.solver}")

    # Solve model
    time_limit = args.time_limit - (time.time() - parse_start) - args.time_buffer if args.time_limit is not None else None
    print_comment(f"{time_limit}s left to solve")

    # If not time left
    if time_limit is not None and time_limit <= 0:
        # Not enough time to start a solve call (we're already over the limit)
        # We should never get in this situation, as a SIGTERM will be raised by the competition runner
        #   if we get over time during the transformation
        print_comment("Giving up, not enough time to start solving.")
        print_status(ExitStatus.unknown)
        return 
    
    if args.solve:
        try:
            with TimerContext("solve") as tc:
                if args.solver == "exact": # Exact2 takes its options at creation time
                    sat = s.solve(
                        time_limit=time_limit,
                    ) 
                else:
                    sat = s.solve(
                        time_limit=time_limit,
                        **solver_arguments(args, model)
                    ) 
            print_comment(f"took {(tc.time):.4f} seconds to solve")
        except MemoryError:
            print_comment("Ran out of memory when trying to solve.")
        except GurobiError as e:
            print_comment("Error from Gurobi: " + str(e))


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
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGABRT, sigterm_handler)

    # Main program
    try:
        main()
    except Exception as e:
        error_handler(e)
