
from __future__ import annotations

import argparse
import lzma
import warnings
import psutil
import signal
import time
import sys, os
import random


if sys.platform != "win32":
    import resource
    
import pathlib
from pathlib import Path
from enum import Enum
from typing import Optional
from io import StringIO 

from contextlib import contextmanager

# CPMpy
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.opb import read_opb
from cpmpy.solvers.ortools import CPM_ortools

# Utils
import os, pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve()))


SUPPORTED_SOLVERS = [name for name,_ in cp.SolverLookup().base_solvers()]

MEMORY_BUFFER_SOFT = 2 # MiB
MEMORY_BUFFER_HARD = 0 # MiB
MEMORY_BUFFER_SOLVER = 20 # MB

original_stdout = sys.stdout


def sigterm_handler(_signo, _stack_frame):
    """
        Handles a SIGTERM. Gives us 1 second to finish the current job before we get killed.
    """
    # Report that we haven't found a solution in time
    print_status(ExitStatus.unknown)
    print_comment("SIGTERM raised.")
    sys.exit(0)
    
def rlimit_cpu_handler(_signo, _stack_frame):
    """
        Handles a SIGXCPU.
    """
    # Report that we haven't found a solution in time
    print_status(ExitStatus.unknown)
    print_comment("SIGXCPU raised.")
    print(flush=True)
    sys.exit(0)

def init_signal_handlers():
    """
    Configure signal handlers
    """
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGABRT, sigterm_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGXCPU, rlimit_cpu_handler)
    else:
        warnings.warn("Windows does not support setting SIGXCPU signal")

def set_memory_limit(mem_limit, verbose:bool=False):
    """
    Set memory limit (Virtual Memory Size). 
    """
    if mem_limit is not None:
        soft = max(mib_as_bytes(mem_limit) - mib_as_bytes(MEMORY_BUFFER_SOFT), mib_as_bytes(MEMORY_BUFFER_SOFT))
        hard = max(mib_as_bytes(mem_limit) - mib_as_bytes(MEMORY_BUFFER_HARD), mib_as_bytes(MEMORY_BUFFER_HARD))
        if verbose:
            print_comment(f"Setting memory limit: {soft} -- {hard}")
        if sys.platform != "win32":
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard)) # limit memory in number of bytes
        else:
            warnings.warn("Memory limits using `resource` are not supported on Windows. Skipping hard limit.")

def set_time_limit(time_limit, verbose:bool=False):
    """
    Set time limit (CPU time in seconds).
    """
    if time_limit is not None:
        if sys.platform != "win32":
            soft = time_limit
            hard = resource.RLIM_INFINITY
            if verbose:
                print_comment(f"Setting time limit: {soft} -- {hard}")
                resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
            else:
                resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        else:
            warnings.warn("CPU time limits using `resource` are not supported on Windows. Skipping hard limit.")

def wall_time(p: psutil.Process):
    return time.time() - p.create_time()

def mib_as_bytes(mib: int) -> int:
    return mib * 1024 * 1024

def bytes_as_mb(bytes: int) -> int:
    return bytes // (1000 * 1000)

def bytes_as_gb(bytes: int) -> int:
    return bytes // (1000 * 1000 * 1000)



# def current_memory_usage() -> int: # returns bytes
#     # not really sure which memory measurement to use: https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management/21049737#21049737
#     return psutil.Process(os.getpid()).memory_info().rss

def remaining_memory(limit:int) -> int: # bytes
    return limit # - current_memory_usage()


# ---------------------------------------------------------------------------- #
#                             OPB Output formatting                            #
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
    unsupported:str = "UNSUPPORTED" # instance contains an unsupported feature (e.g. a unsupported global constraint)
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


def solution_opb(model, useless_style="*", boolean_style="int"):
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


    variables = [var for var in model.user_vars if var.name[:2] not in ["IV", "BV", "B#"]] # dirty workaround for all missed aux vars in user vars
    return " ".join([var.name.replace("[","").replace("]","") if var.value() else "-"+var.name.replace("[","").replace("]","") for var in variables])


# ---------------------------------------------------------------------------- #
#                          CLI argument type checkers                          #
# ---------------------------------------------------------------------------- #

def dir_path(path):
    if os.path.isfile(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
    
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

    # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
    res |= {
        "interleave_search": True,
        "use_rins_lns": False,
    }
    if not model.has_objective():
        res |= { "num_violation_ls": 1 }

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

    def internal_options(solver: CPM_ortools):
        # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
        solver.ort_solver.parameters.subsolvers.extend(["default_lp", "max_lp", "quick_restart"])
        if not model.has_objective():
            solver.ort_solver.parameters.subsolvers.append("core_or_no_lp")
        if len(solver.ort_model.proto.search_strategy) != 0:
            solver.ort_solver.parameters.subsolvers.append("fixed")

    return res, internal_options

def exact_arguments(seed: Optional[int] = None, **kwargs):
    # Documentation: https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp?ref_type=heads
    res = dict()
    if seed is not None: 
        res |= { "seed": seed }

    return res, None

def choco_arguments(): 
    # Documentation: https://github.com/chocoteam/pychoco/blob/master/pychoco/solver.py
    return {}, None

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

    return res, None

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
    
    return res, None

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

        class GurobiSolutionCallback:
            def __init__(self, model:cp.Model):
                self.__start_time = time.time()
                self.__solution_count = 0
                self.model = model

            def callback(self, *args, **kwargs):
                current_time = time.time()
                model, state = args

                # Callback codes: https://www.gurobi.com/documentation/current/refman/cb_codes.html#sec:CallbackCodes
                
                from gurobipy import GRB
                # if state == GRB.Callback.MESSAGE: # verbose logging
                #     print_comment("log message: " + str(model.cbGet(GRB.Callback.MSG_STRING)))
                if state == GRB.Callback.MIP: # callback from the MIP solver
                    if model.cbGet(GRB.Callback.MIP_SOLCNT) > self.__solution_count: # do we have a new solution?

                        obj = int(model.cbGet(GRB.Callback.MIP_OBJBST))
                        print_comment('Solution %i, time = %0.2fs' % 
                                    (self.__solution_count, current_time - self.__start_time))
                        print_objective(obj)
                        self.__solution_count = model.cbGet(GRB.Callback.MIP_SOLCNT)

        res |= { "solution_callback": GurobiSolutionCallback(model).callback }

    return res, None

def cpo_arguments(model: cp.Model,
                  cores: Optional[int] = None,
                  seed: Optional[int] = None,
                  intermediate: bool = False,
                  **kwargs):
    # Documentation: https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.parameters.py.html#docplex.cp.parameters.CpoParameters
    res = dict()
    if cores is not None:
        res |= { "Workers": cores }
    if seed is not None:
        res |= { "RandomSeed": seed }

    if intermediate and model.has_objective():
        from docplex.cp.solver.solver_listener import CpoSolverListener

        class CpoSolutionCallback(CpoSolverListener):

            def __init__(self):
                super().__init__()
                self.__start_time = time.time()
                self.__solution_count = 1

            def result_found(self, solver, sres):
                current_time = time.time()
                obj = sres.get_objective_value()
                if obj is not None:
                    print_comment('Solution %i, time = %0.2fs' % 
                                (self.__solution_count, current_time - self.__start_time))
                    print_objective(obj)
                    self.__solution_count += 1

            def solution_count(self):
                """Returns the number of solutions found."""
                return self.__solution_count

        # Register the callback
        res |= { "solution_callback": CpoSolutionCallback }

    return res, None


def solver_arguments(solver: str, 
                     model: cp.Model, 
                     seed: Optional[int] = None,
                     intermediate: bool = False,
                     cores: int = 1,
                     mem_limit: Optional[int] = None,
                     **kwargs):
    opt = model.has_objective()
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
        return gurobi_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, intermediate=intermediate, opt=opt, **kwargs)
    elif solver == "cpo":
        return cpo_arguments(model=model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
    else:
        print_comment(f"setting parameters of {solver} is not (yet) supported")
        return dict()

def opb_cpmpy(
        benchname: str,
        seed: Optional[int] = None,
        time_limit: Optional[int] = None,
        mem_limit: Optional[int] = None,  # MiB: 1024 * 1024 bytes
        cores: int = 1,
        solver: str = None,
        time_buffer: int = 0,
        intermediate: bool = False,
        verbose: bool = False,
        **kwargs,
):
    if not verbose:
        warnings.filterwarnings("ignore")

    try:

        # --------------------------- Global Configuration --------------------------- #

        # Get the current process
        p = psutil.Process()

        # pychoco currently does not support setting the mem_limit
        if solver == "choco" and mem_limit is not None:
            warnings.warn("'mem_limit' is currently not supported with choco, issues with GraalVM")
            mem_limit = None

        # Set random seed (if provided)
        if seed is not None:
            random.seed(seed)

        # Set memory limit (if provided)
        if mem_limit is not None:
            set_memory_limit(mem_limit, verbose=verbose)

        # Set time limit (if provided)
        if time_limit is not None:
            set_time_limit(int(time_limit - wall_time(p) + time.process_time()), verbose=verbose) # set remaining process time != wall time
   
        # ------------------------------ Parse instance ------------------------------ #

        time_parse = time.time()
        model = read_opb(benchname)
        time_parse = time.time() - time_parse
        if verbose: print_comment(f"took {time_parse:.4f} seconds to parse OPB model [{benchname}]")

        if time_limit and time_limit < wall_time(p):
            raise TimeoutError("Time's up after parse")
        
        # ------------------------ Post CPMpy model to solver ------------------------ #

        solver_args, internal_options = solver_arguments(solver, model=model, seed=seed,
                                       intermediate=intermediate,
                                       cores=cores, mem_limit=mib_as_bytes(mem_limit) if mem_limit is not None else None,
                                       **kwargs)

        # Post model to solver
        time_post = time.time()

        if solver == "exact": # Exact2 takes its options at creation time
            s = cp.SolverLookup.get(solver, model, **solver_args)
            solver_args = dict()  # no more solver args needed
        else:
            s = cp.SolverLookup.get(solver, model)
        time_post = time.time() - time_post
        if verbose: print_comment(f"took {time_post:.4f} seconds to post model to {solver}")

        if time_limit and time_limit < wall_time(p):
            raise TimeoutError("Time's up after post")
        
        # ------------------------------- Solve model ------------------------------- #
        
        if time_limit:
            # give solver only the remaining time
            time_limit = time_limit - wall_time(p) - time_buffer
            # disable signal-based time limit and let the solver handle it (solvers don't play well with difference between cpu and wall time)
            set_time_limit(None)
            
            if verbose: print_comment(f"{time_limit}s left to solve")
        
        time_solve = time.time()
        try:
            if internal_options is not None:
                internal_options(s) # Set more internal solver options (need access to native solver object)
            is_sat = s.solve(time_limit=time_limit, **solver_args)
        except RuntimeError as e:
            if "Program interrupted by user." in str(e): # Special handling for Exact
                raise TimeoutError("Exact interrupted due to timeout")
            else:
                raise e

        time_solve = time.time() - time_solve
        if verbose: print_comment(f"took {time_solve:.4f} seconds to solve")

        # ------------------------------- Print result ------------------------------- #

        if s.status().exitstatus == CPMStatus.OPTIMAL:
            print_value(solution_opb(s))
            print_status(ExitStatus.optimal)
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            print_value(solution_opb(s))
            print_status(ExitStatus.sat)
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            print_status(ExitStatus.unsat)
        else:
            print_comment("Solver did not find any solution within the time/memory limit")
            print_status(ExitStatus.unknown)
        
        # ------------------------------------- - ------------------------------------ #

        
    except MemoryError as e:
        print_comment(f"MemoryError raised. Reached limit of {mem_limit} MiB")
        print_status(ExitStatus.unknown)
        raise e
    except NotImplementedError as e:
        print_comment(str(e))
        print_status(ExitStatus.unsupported)
        raise e
    except Exception as e:
        print_comment(f"An {type(e)} got raised: {e}")
        import traceback
        print_comment("Stack trace:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                print_comment(line)
        print_status(ExitStatus.unknown)
        raise e
    

if __name__ == "__main__":
    
    # ------------------------------ Argument parsing ------------------------------ #

    parser = argparse.ArgumentParser("CPMpy OPB PB24 executable")
    
    ## XCSP3 required arguments:
    # BENCHNAME: Name of the XCSP3 XML file with path and extension
    parser.add_argument("benchname", type=dir_path, help="OPB file to parse, with full path and extension.") 
    # RANDOMSEED: Seed between 0 and 4294967295
    parser.add_argument("-s", "--seed", required=False, type=int, default=None, help="Random seed (integer between 0 and 4294967295).")
    # TIMELIMIT: Total CPU time in seconds (before it gets killed)
    parser.add_argument("-l", "--time-limit", required=False, type=int, default=None, help="Time limit in seconds.") # TIMELIMIT
    # MEMLIMIT: Total amount of memory in MiB (mebibyte = 1024 * 1024 bytes)
    parser.add_argument("-m", "--mem-limit", required=False, type=int, default=None, help="Memory limit in MiB (1 MiB = 1024 x 1024 bytes). Gets passed on to the solver (if supported) and is also enforced on the Python/CPMpy side. Measured as Virtual Memory Size")
    # TMPDIR: Only location where temporary read/write is allowed
    parser.add_argument("-t","--tmpdir", required=False, type=dir_path, help="Directory for temporary read/write operations.")
    # NBCORE: Number of processing units (can by any of the following: a processor / a processor core / logical processor (hyper-threading))
    parser.add_argument("-c", "--cores", required=False, type=int, default=None, help="Number of processing units to use (e.g., cores or logical processors).")
    # DIR: not needed, e.g. we just import files

    ## CPMpy optional arguments:
    # The underlying solver which should be used (can also be "solver:subsolver")
    parser.add_argument("--solver", required=False, type=str, default="ortools", help="Underlying CPMpy solver to use (can be 'solver:subsolver').")
    # How much time before SIGTERM should we halt solver (for the final post-processing steps and solution printing)
    parser.add_argument("--time-buffer", required=False, type=int, default=0, help="Time buffer (in seconds) to reserve before SIGTERM for cleanup/postprocessing.") # buffer to give subsolver time to return and do some post-processing
    # If intermediate solutions should be printed (if the solver supports it)
    parser.add_argument("--intermediate", action=argparse.BooleanOptionalAction, help="Whether to print intermediate solutions (if supported by the solver).")

    ## Executable optional arguments
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, help="Enable verbose output for debugging or detailed logs.")
    

    # Process cli arguments 
    args = parser.parse_args()
    if args.verbose:
        print_comment(f"Arguments: {args}")

    try:
        # Configure signal handles
        init_signal_handlers()

        # if str(args.benchname).endswith(".lzma"):
        #     # Decompress the XZ file
        #     with lzma.open(args.benchname, 'rt', encoding='utf-8') as f:
        #         xml_file = StringIO(f.read()) # read to memory-mapped file
        #         args.benchname = xml_file

        opb_cpmpy(**vars(args))
    except Exception as e:
        print_comment(f"{type(e).__name__} -- {e}")