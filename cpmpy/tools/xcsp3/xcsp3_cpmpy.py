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
# import psutil
from io import StringIO 
import json


# sys.path.insert(1, os.path.join(pathlib.Path(__file__).parent.resolve(), "..", ".."))

# CPMpy
import cpmpy as cp
from cpmpy.solvers.gurobi import CPM_gurobi
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.xcsp3 import _parse_xcsp3, _load_xcsp3

# PyCSP3
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from xml.etree.ElementTree import ParseError

# Utils
import os, pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve()))
from xcsp3_solution import solution_xml
from parser_callbacks import CallbacksCPMPy
# from xcsp3.perf_timer import PerfContext, TimerContext

import xcsp3_natives

# Configuration
SUPPORTED_SOLVERS = ["choco", "ortools", "exact", "z3", "minizinc", "gurobi"]
SUPPORTED_SUBSOLVERS = {
    "minizinc": ["gecode", "chuffed"]
}
DEFAULT_SOLVER = "ortools"
TIME_BUFFER = 5 # seconds
# TODO : see if good value
MEMORY_BUFFER_SOFT = 2 # MiB
MEMORY_BUFFER_HARD = 0 # MiB
MEMORY_BUFFER_SOLVER = 20 # MB

original_stdout = sys.stdout

def sigterm_handler(_signo, _stack_frame):
    """
        Handles a SIGTERM. Gives us 1 second to finish the current job before we get killed.
    """
    # Report that we haven't found a solution in time
    #sys.stdout = original_stdout
    print_status(ExitStatus.unknown)
    print_comment("SIGTERM raised.")
    print(flush=True)
    raise SystemExit("SIGTERM received")

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

# def current_memory_usage() -> int: # returns bytes
#     # not really sure which memory measurement to use: https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management/21049737#21049737
#     return psutil.Process(os.getpid()).memory_info().rss

def remaining_memory(limit:int) -> int: # bytes
    return limit # - current_memory_usage()

def get_subsolver(solver: str, model: cp.Model, subsolver: Optional[str] = None) -> Optional[str]:
    # Update subsolver
    if solver == "z3":
        if model.objective_ is not None:
            return "opt"
        else:
            return "sat"
    elif subsolver is None:
        if solver in SUPPORTED_SUBSOLVERS:
            return SUPPORTED_SUBSOLVERS[solver][0]
    return subsolver


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

    def __init__(self, model:cp.Model):
        self.__start_time = time.time()
        self.__solution_count = 0
        self.model = model

    def callback(self, *args, **kwargs):
        current_time = time.time()
        model, state = args

        # Callback codes: https://www.gurobi.com/documentation/current/refman/cb_codes.html#sec:CallbackCodes
        
        from gurobipy import GurobiError, GRB
        # if state == GRB.Callback.MESSAGE: # verbose logging
        #     print_comment("log message: " + str(model.cbGet(GRB.Callback.MSG_STRING)))
        if state == GRB.Callback.MIP: # callback from the MIP solver
            if model.cbGet(GRB.Callback.MIP_SOLCNT) > self.__solution_count: # do we have a new solution?

                obj = int(model.cbGet(GRB.Callback.MIP_OBJBST))
                print_comment('Solution %i, time = %0.2fs' % 
                            (self.__solution_count, current_time - self.__start_time))
                print_objective(obj)
                self.__solution_count = model.cbGet(GRB.Callback.MIP_SOLCNT)


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

def ortools_arguments(model: cp.Model,
                      cores: Optional[int] = None,
                      seed: Optional[int] = None,
                      intermediate: bool = False,
                      **kwargs):
    # https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto
    res = dict()
    if cores is not None:
        res |= { "num_search_workers": cores }
    if seed is not None: 
        res |= { "random_seed": seed }

    if intermediate and model.has_objective():
        # Define custom ORT solution callback, then register it
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
            
        # Register the callback
        res |= { "solution_callback": OrtSolutionCallback() }

    return res

def exact_arguments(seed: Optional[int] = None, **kwargs):
    # Documentation: https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp?ref_type=heads
    res = dict()
    if seed is not None: 
        res |= { "seed": seed }

    return res

def choco_arguments(): 
    # Documentation: https://github.com/chocoteam/pychoco/blob/master/pychoco/solver.py
    return {}

def z3_arguments(model: cp.Model,
                 cores: int = 1,
                 seed: Optional[int] = None,
                 mem_limit: Optional[int] = None,
                 **kwargs):
    # Documentation: https://microsoft.github.io/z3guide/programming/Parameters/
    # -> is outdated, just let it crash and z3 will report the available options

    res = dict()
    
    if model.has_objective():
        # Opt does not seem to support setting random seed or max memory
        pass
    else:
        # Sat parameters
        if cores is not None:
            res |= { "threads": cores }  # TODO what with hyperthreadding, when more threads than cores
        if seed is not None: 
            res |= { "random_seed": seed }
        if mem_limit is not None:
            res |= { "max_memory": bytes_as_mb(mem_limit) }

    return res

def minizinc_arguments(solver: str,
                       cores: Optional[int] = None,
                       seed: Optional[int] = None,
                       **kwargs):
    # Documentation: https://minizinc-python.readthedocs.io/en/latest/api.html#minizinc.instance.Instance.solve
    res = dict()
    if cores is not None:
        res |= { "processes": cores }
    if seed is not None: 
        res |= { "random_seed": seed }

    #if solver.endswith("gecode"):
        # Documentation: https://www.minizinc.org/doc-2.4.3/en/lib-gecode.html
    #elif solver.endswith("chuffed"):
        # Documentation: 
        # - https://www.minizinc.org/doc-2.5.5/en/lib-chuffed.html
        # - https://github.com/chuffed/chuffed/blob/develop/chuffed/core/options.h
    
    return res

def gurobi_arguments(model: cp.Model,
                     cores: Optional[int] = None,
                     seed: Optional[int] = None,
                     mem_limit: Optional[int] = None,
                     intermediate: bool = False,
                     **kwargs):
    # Documentation: https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
    res = dict()
    if cores is not None:
        res |= { "Threads": cores }
    if seed is not None:
        res |= { "Seed": seed }
    if mem_limit is not None:
        res |= { "MemLimit": bytes_as_gb(remaining_memory(mem_limit)) }

    if intermediate and model.has_objective():
        res |= { "solution_callback": Callback(model).callback }

    return res

def solver_arguments(solver: str, 
                     model: cp.Model, 
                     seed: Optional[int] = None,
                     intermediate: bool = False,
                     cores: int = 1,
                     mem_limit: Optional[int] = None,
                     **kwargs):
    opt = model.objective_ is not None
    sat = not opt

    if solver == "ortools":
        return ortools_arguments(model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
    elif solver == "exact":
        return exact_arguments(seed=seed, **kwargs)
    elif solver == "choco":
        return choco_arguments()
    elif solver == "z3":
        return z3_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, **kwargs)
    elif solver.startswith("minizinc"):  # also can have a subsolver
        return minizinc_arguments(solver, cores=cores, seed=seed, **kwargs)
    elif solver == "gurobi":
        return gurobi_arguments(solver, model, cores=cores, seed=seed, mem_limit=mem_limit, intermediate=intermediate, opt=opt, **kwargs)
    else:
        print_comment(f"setting parameters of {solver} is not (yet) supported")
        return dict()

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

 
def xcsp3_cpmpy(benchname: str,
                seed: Optional[int] = None,
                time_limit: Optional[int] = None,
                mem_limit: Optional[int] = None,  # MiB: 1024 * 1024 bytes
                cores: int = 1,
                solver: str = None,
                time_buffer: int = 0,
                intermediate: bool = False,
                **kwargs,
):
    try:

        # --------------------------- Global Configuration --------------------------- #

        if seed is not None:
            random.seed(seed)
        if mem_limit is not None:
            soft = max(mib_as_bytes(mem_limit) - mib_as_bytes(MEMORY_BUFFER_SOFT), mib_as_bytes(MEMORY_BUFFER_SOFT))
            hard = max(mib_as_bytes(mem_limit) - mib_as_bytes(MEMORY_BUFFER_HARD), mib_as_bytes(MEMORY_BUFFER_HARD))
            print_comment(f"Setting memory limit: {soft} -- {hard}")
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard)) # limit memory in number of bytes

        sys.argv = ["-nocompile"] # Stop pyxcsp3 from complaining on exit

        time_start = time.time()

        # ------------------------------ Parse instance ------------------------------ #

        time_parse = time.time()
        parser = _parse_xcsp3(benchname)
        time_parse = time.time() - time_parse
        print_comment(f"took {time_parse:.4f} seconds to parse XCSP3 model [{benchname}]")

        if time_limit and time_limit < (time.time() - time_start):
            print_comment("Time's up!")
            print_status(ExitStatus.unknown)
            return 

        # ---------------- Convert XCSP3 to CPMpy model with callbacks --------------- #

        time_callback = time.time()
        try:
            model = _load_xcsp3(parser)
        except NotImplementedError as e:
            print_status(ExitStatus.unsupported)
            print_comment(str(e))
            return
            #exit(1)
        time_callback = time.time() - time_callback
        print_comment(f"took {time_callback:.4f} seconds to convert to CPMpy model")
        
        if time_limit and time_limit < (time.time() - time_start):
            print_comment("Time's up!")
            print_status(ExitStatus.unknown)
            return 

        # ------------------------ Post CPMpy model to solver ------------------------ #

        solver_args = solver_arguments(solver, model=model, seed=seed,
                                       intermediate=intermediate,
                                       cores=cores, mem_limit=mem_limit,
                                       **kwargs)
        # time_limit is generic for all, done later

        # Additional XCSP3-specific native transformations
        added_natives = {
            "ortools": {
                "no_overlap2d": xcsp3_natives.ort_nooverlap2d,
                "subcircuit": xcsp3_natives.ort_subcircuit,
                "subcircuitwithstart": xcsp3_natives.ort_subcircuitwithstart,
            },
            "choco": {
                "subcircuit": xcsp3_natives.choco_subcircuit,
            },
            "minizinc": {
                "subcircuit": xcsp3_natives.minizinc_subcircuit,
            },
        }

        # Post model to solver
        time_post = time.time()
        with prepend_print():  # catch prints and prepend 'c' to each line (still needed?)
            if solver == "exact": # Exact2 takes its options at creation time
                s = cp.SolverLookup.get(solver, model, **solver_args, added_natives=added_natives.get(solver, {}))
                solver_args = dict()  # no more solver args needed
            else:
                s = cp.SolverLookup.get(solver, model, added_natives=added_natives.get(solver, {}))
        time_post = time.time() - time_post
        print_comment(f"took {time_post:.4f} seconds to post model to {solver}")

        if time_limit and time_limit < (time.time() - time_start):
            print_comment("Time's up!")
            print_status(ExitStatus.unknown)
            return 


        # ------------------------------- Solve model ------------------------------- #
        
        if time_limit:
            # give solver only the remaining time
            time_limit = time_limit - (time.time() - time_start)
            print_comment(f"{time_limit}s left to solve")
        
        time_solve = time.time()
        is_sat = s.solve(time_limit=time_limit, **solver_args)
        time_solve = time.time() - time_solve
        print_comment(f"took {time_solve:.4f} seconds to solve")

        # ------------------------------- Print result ------------------------------- #

        if s.status().exitstatus == CPMStatus.OPTIMAL:
            # TODO: simplify and let print_status take a CPMStatus?
            print_status(ExitStatus.optimal)
            print_value(solution_xml(s))
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            print_status(ExitStatus.sat)
            print_value(solution_xml(s))
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            print_status(ExitStatus.unsat)
        elif s.status().exitstatus == CPMStatus.UNKNOWN:
            print_status(ExitStatus.unknown)
        else:
            print_status(ExitStatus.unknown)
        
    except MemoryError as e:
        # TODO: simplify all error throwing/handling with a single outer catch here?
        # everything inside would simply raise and nothing would be catched?
        print_comment(f"MemoryError raised. Reached limit of {mem_limit} MiB")
        print_status(ExitStatus.unknown)
    except ParseError as e:
        if "out of memory" in e.msg:
            print_comment(f"MemoryError raised by parser. Reached limit of {mem_limit} MiB")
            print_status(ExitStatus.unknown)
        else:
            raise e
    except Exception as e:
        print_comment(f"An {type(e)} got raised: {e}")
        import traceback
        print_comment("Stack trace:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                print_comment(line)
        print_status(ExitStatus.unknown)


if __name__ == "__main__":
    # Configure signal handles
    # signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGABRT, sigterm_handler)

    # ------------------------------ Argument parsing ------------------------------ #
    parser = argparse.ArgumentParser("CPMpy XCSP3 executable")
    
    ## XCSP3 required arguments:
    # BENCHNAME: Name of the XCSP3 XML file with path and extension
    parser.add_argument("benchname", type=dir_path) 
    # RANDOMSEED: Seed between 0 and 4294967295
    parser.add_argument("-s", "--seed", required=False, type=int)
    # TIMELIMIT: Total CPU time in seconds (before it gets killed)
    parser.add_argument("-l", "--time-limit", required=False, type=int) # TIMELIMIT
    # MEMLIMIT: Total amount of memory in MiB (mebibyte = 1024 * 1024 bytes)
    parser.add_argument("-m", "--mem-limit", required=False, type=int)
    # TMPDIR: Only location where temporary read/write is allowed
    parser.add_argument("-t","--tmpdir", required=False, type=dir_path)
    # NBCORE: Number of processing units (can by any of the following: a processor / a processor core / logical processor (hyper-threading))
    parser.add_argument("-c", "--cores", required=False, type=int)
    # DIR: not needed, e.g. we just import files

    ## CPMpy optional arguments:
    # The underlying solver which should be used (can also be "solver:subsolver")
    parser.add_argument("--solver", required=True, type=str)
    # How much time before SIGTERM should we halt solver (for the final post-processing steps and solution printing)
    parser.add_argument("--time-buffer", required=False, type=int)
    # If intermediate solutions should be printed (if the solver supports it)
    parser.add_argument("--intermediate", action=argparse.BooleanOptionalAction)

    # Process cli arguments 
    args = parser.parse_args()
    print_comment(f"Arguments: {args}")

    xcsp3_cpmpy(**vars(args))