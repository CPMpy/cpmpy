"""
Benchmark framework for CPMpy models.

This module provides the `Benchmark` base class, designed to run constraint programming 
benchmarks in a structured fashion. It allows reading instances, posting them to different 
back-end solvers, and handling solver execution with limits on time and memory. 
It also provides hooks for customizing logging, intermediate solution printing, and 
error handling. Although this base class can be used on its own (example below),
users will most likely want to have a look at one of its subclasses for running a specific
benchmark dataset, e.g. xcsp3, opb, mse, ...


Usage Example
-------------
>>> from myparser import read_instance    # your custom model parser (or one included in CPMpy)
>>> bm = Benchmark(reader=read_instance)
>>> bm.run(
...     instance="example.extension",     # your benchmark instance (e.g. coming from a CPMpy model dataset)
...     solver="ortools",
...     time_limit=30,
...     mem_limit=1024,
...     verbose=True
... )
Status: OPTIMAL
Objective: 42
Solution: ...

"""


from abc import ABC

import os
import signal
import sys
import time
import math
import random
from xml.parsers.expat import model
import psutil
import warnings
from enum import Enum
from typing import Optional

from pathlib import Path
from collections import defaultdict # Added for efficient mapping

# --- Necessary Imports for Conversion (assumed to be available in the runner's context) ---
import cpmpy as cp
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.variables import _BoolVarImpl
from cpmpy.tools.dimacs_ import write_gcnf
from cpmpy.tools.explain.mus import make_assump_model
from cpmpy.solvers.pysat import CPM_pysat
from pysat.formula import CNF

import cpmpy as cp
from cpmpy.tools.benchmark import _mib_as_bytes, _wall_time, set_memory_limit, set_time_limit, _bytes_as_mb, _bytes_as_gb, disable_memory_limit
from cpmpy.tools.explain import mus, quickxplain, pb_mus, cp_mus
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.int2bool import int2bool
from cpmpy.transformations.linearize import linearize_constraint, only_positive_coefficients, only_positive_coefficients
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list
from cpmpy.transformations.reification import only_bv_reifies, only_implies
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.to_cnf import to_cnf

# # --- Configuration (Define these paths outside the function, e.g., in self or module globals) ---
path = "/home/orestis_ubuntu/work/"
path = "/cw/dtailocal/orestis/"
OUTPUT_CNF_DIR = f"{path}benchmarks/2025/XCSP_CNF/" 
OUTPUT_GCNF_DIR = f"{path}benchmarks/2025/XCSP_GCNF/" 
Path(OUTPUT_CNF_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_GCNF_DIR).mkdir(parents=True, exist_ok=True)

class ExitStatus(Enum):
    unsupported:str = "unsupported" # instance contains an unsupported feature (e.g. a unsupported global constraint)
    sat:str = "sat" # CSP : found a solution | COP : found a solution but couldn't prove optimality
    optimal:str = "optimal" # optimal COP solution found
    unsat:str = "unsat" # instance is unsatisfiable
    unknown:str = "unknown" # any other case

class Benchmark(ABC):
    """
    Abstract base class for running CPMpy benchmarks.

    The `Benchmark` class provides a standardized framework for reading instances,
    posting models to solvers, and managing solver runs with resource limits.
    It is designed to be extended or customized for specific benchmarking needs.    
    """

    def __init__(self, reader:callable, exit_status:Enum=ExitStatus):
        """
        Arguments:
            reader (callable): A parser from a model format to a CPMPy model.
        """
        self.reader = reader
        self.exit_status = exit_status
        
    def read_instance(self, instance, open) -> cp.Model:
        """
        Parse a model instance to a CPMpy model.

        Arguments:
            instance (str or os.PathLike): The model instance to parse into a CPMpy model.
        """
        return self.reader(instance, open=open)
    
    """
    Callback methods which can be overwritten to make a custom benchmark run.
    """

    def print_comment(self, comment:str):
        print(comment)

    def print_intermediate(self, objective:int):
        self.print_comment("Intermediate solution:", objective)

    def print_result(self, s):
        self.print_comment(s.status())
        
    def print_mus(self, mus_res, model):
        self.print_comment(mus_res)

    def handle_memory_error(self, mem_limit):
        self.print_comment(f"MemoryError raised. Reached limit of {mem_limit} MiB")
    
    def handle_not_implemented(self, e):
        self.print_comment(str(e))

    def handle_exception(self, e):
        self.print_comment(f"An {type(e)} got raised: {e}")
        import traceback
        self.print_comment("Stack trace:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.print_comment(line)

    def handle_sigterm(self):
        pass
        
    def handle_rlimit_cpu(self):
        pass

    """
    Solver arguments (can also be tweaked for a specific benchmark).
    """

    def ortools_arguments(
            self,
            model: cp.Model,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            intermediate: bool = False,
            **kwargs
        ):
        # https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto
        res = dict()

        # return { "num_search_workers": cores }, None
        # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
        res |= {
            # "interleave_search": True,
            # "use_rins_lns": False,
        }
        # if not model.has_objective():
        #     res |= { "num_violation_ls": 1 }

        if cores is not None:
            res |= { "num_search_workers": cores }
            
        return res, None
        if seed is not None: 
            res |= { "random_seed": seed }

        if intermediate and model.has_objective():
            # Define custom ORT solution callback, then register it
            _self = self
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
                    _self.print_comment('Solution %i, time = %0.4fs' % 
                                (self.__solution_count, current_time - self.__start_time))
                    _self.print_intermediate(obj)
                    self.__solution_count += 1
                

                def solution_count(self):
                    """Returns the number of solutions found."""
                    return self.__solution_count
                
            # Register the callback
            res |= { "solution_callback": OrtSolutionCallback() }

        def internal_options(solver: "CPM_ortools"):
            # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
            solver.ort_solver.parameters.subsolvers.extend(["default_lp", "max_lp", "quick_restart"])
            if not model.has_objective():
                solver.ort_solver.parameters.subsolvers.append("core_or_no_lp")
            if len(solver.ort_model.proto.search_strategy) != 0:
                solver.ort_solver.parameters.subsolvers.append("fixed")

        return res, internal_options
    
    def exact_arguments(
            self,
            seed: Optional[int] = None, 
            **kwargs
        ):
        # Documentation: https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp?ref_type=heads
        res = dict()
        if seed is not None: 
            res |= { "seed": seed }

        return res, None

    def choco_arguments(): 
        # Documentation: https://github.com/chocoteam/pychoco/blob/master/pychoco/solver.py
        return {}, None
    
    def pindakaas_arguments(
        self, 
        **kwargs
    ): 
        # Documentation: https://github.com/pindakaashq/pindakaas/blob/develop/crates/pyndakaas/python/pindakaas/solver.py

        res = dict()
        return res, None
    
    def pysat_arguments(
            self,
            **kwargs
    ):
        res = dict()
        return res, None
    
    def pumpkin_arguments(
            self,
            **kwargs
    ):
        res = dict()
        return res, None

    def z3_arguments(
            self,
            model: cp.Model,
            cores: int = 1,
            seed: Optional[int] = None,
            mem_limit: Optional[int] = None,
            **kwargs
        ):
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
                res |= { "max_memory": _bytes_as_mb(mem_limit) }

        return res, None

    def minizinc_arguments(
            self,
            solver: str,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            **kwargs
        ):
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

    def gurobi_arguments(
            self,
            model: cp.Model,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            mem_limit: Optional[int] = None,
            intermediate: bool = False,
            **kwargs
        ):
        # Documentation: https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
        res = dict()
        if cores is not None:
            res |= { "Threads": cores }
        if seed is not None:
            res |= { "Seed": seed }
        if mem_limit is not None:
            res |= { "MemLimit": _bytes_as_gb(mem_limit) }

        if intermediate and model.has_objective():

            _self = self

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
                            _self.print_comment('Solution %i, time = %0.4fs' % 
                                        (self.__solution_count, current_time - self.__start_time))
                            _self.print_intermediate(obj)
                            self.__solution_count = model.cbGet(GRB.Callback.MIP_SOLCNT)

            res |= { "solution_callback": GurobiSolutionCallback(model).callback }

        return res, None

    def cpo_arguments(
            self,
            model: cp.Model,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            intermediate: bool = False,
            **kwargs
        ):
        # Documentation: https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.parameters.py.html#docplex.cp.parameters.CpoParameters
        res = dict()
        if cores is not None:
            res |= { "Workers": cores }
        if seed is not None:
            res |= { "RandomSeed": seed }

        if intermediate and model.has_objective():
            from docplex.cp.solver.solver_listener import CpoSolverListener
            _self = self
            class CpoSolutionCallback(CpoSolverListener):

                def __init__(self):
                    super().__init__()
                    self.__start_time = time.time()
                    self.__solution_count = 1

                def result_found(self, solver, sres):
                    current_time = time.time()
                    obj = sres.get_objective_value()
                    if obj is not None:
                        _self.print_comment('Solution %i, time = %0.4fs' % 
                                    (self.__solution_count, current_time - self.__start_time))
                        _self.print_intermediate(obj)
                        self.__solution_count += 1

                def solution_count(self):
                    """Returns the number of solutions found."""
                    return self.__solution_count

            # Register the callback
            res |= { "solution_callback": CpoSolutionCallback }

        return res, None

    def cplex_arguments(
        self,
        cores: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        res = dict()
        if cores is not None:
            res |= {"threads": cores}
        if seed is not None:
            res |= {"randomseed": seed}

        return res, None
    
    def hexaly_arguments(
        self,
        model: cp.Model,
        cores: Optional[int] = None,
        seed: Optional[int] = None,
        intermediate: bool = False,
        **kwargs
    ):
        res = dict()
        #res |= {"nb_threads": cores}
        #res |= {"seed": seed}


        if intermediate and model.has_objective():
            # Define custom Hexaly solution callback, then register it

            _self = self
            class HexSolutionCallback:
    
                def __init__(self):
                    self.__start_time = time.time()
                    self.__solution_count = 0
          

                def on_solution_callback(self, optimizer, cb_type):
                    """Called on each new solution."""
                    # check if solution with different objective (or if verbose)
                    current_time = time.time()
                    obj = optimizer.model.objectives[0]
                    _self.print_comment('Solution %i, time = %0.4fs' % 
                                (self.__solution_count, current_time - self.__start_time))
                    _self.print_intermediate(obj)
                    self.__solution_count += 1

                def solution_count(self):
                    return self.__solution_count
                
            # Register the callback
            res |= { "solution_callback": HexSolutionCallback().on_solution_callback }


        # def internal_options(solver: "CPM_hexaly"):
        #     # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
        #     #solver.native_model.get_param().set_seed(seed)
        #     #solver.native_model.get_param().set_nr_threads(cores)

        #     _self = self
        #     class CallbackExample:
        #         def __init__(self):
        #             self.last_best_value = 0
        #             self.last_best_running_time = 0
        #             self.__solution_count = 0
        #             self.__start_time = time.time()

        #         def my_callback(self, optimizer, cb_type):
        #             stats = optimizer.statistics
        #             obj = optimizer.model.objectives[0]
        #             current_time = time.time()
        #             #obj = int(self.ObjectiveValue())
        #             #obj = optimizer.get_objective_bound(0).value
        #             if obj.value > self.last_best_value:
        #                 self.last_best_running_time = stats.running_time
        #                 self.last_best_value = obj.value
        #                 self.__solution_count += 1
                  
        #                 _self.print_comment('Solution %i, time = %0.4fs' % 
        #                         (self.__solution_count, current_time - self.__start_time))
        #                 _self.print_intermediate(obj.value)

            # optimizer = solver.native_model
            # cb = CallbackExample()
            # from hexaly.optimizer import HxCallbackType
            # optimizer.add_callback(HxCallbackType.TIME_TICKED, cb.my_callback)

        return res, None

    """
    Methods which can, bit most likely shouldn't, be overwritten.
    """
    
    def set_memory_limit(self, mem_limit):
        set_memory_limit(mem_limit)

    def set_time_limit(self, time_limit):
        p = psutil.Process()
        if time_limit is not None:
            set_time_limit(int(time_limit - _wall_time(p) + time.process_time()))
        else:
            set_time_limit(None)

    def sigterm_handler(self, _signo, _stack_frame):
        exit_code = self.handle_sigterm()
        print(flush=True)
        os._exit(exit_code)
        
    def rlimit_cpu_handler(self, _signo, _stack_frame):
        exit_code = self.handle_rlimit_cpu()
        print(flush=True)
        os._exit(exit_code)

    def init_signal_handlers(self):
        """
        Configure signal handlers
        """
        signal.signal(signal.SIGINT, self.sigterm_handler)
        signal.signal(signal.SIGTERM, self.sigterm_handler)
        signal.signal(signal.SIGINT, self.sigterm_handler)
        signal.signal(signal.SIGABRT, self.sigterm_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGXCPU, self.rlimit_cpu_handler)
        else:
            warnings.warn("Windows does not support setting SIGXCPU signal")

    def post_model(self, model, solver, solver_args=None):
        """
        Post the model to the selected backend solver.
        """
        if solver == "exact": # Exact2 takes its options at creation time
            s = cp.SolverLookup.get(solver, model, **solver_args)
            solver_args = dict()  # no more solver args needed
        else:
            s = cp.SolverLookup.get(solver, model)
        return s

    
    """
    Internal workings
    """

    def solver_arguments(
            self,
            solver: str, 
            model: cp.Model, 
            seed: Optional[int] = None,
            intermediate: bool = False,
            cores: int = 1,
            mem_limit: Optional[int] = None,
            **kwargs
        ):
        opt = model.has_objective()
        sat = not opt

        if solver == "ortools":
            return self.ortools_arguments(model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "exact":
            return self.exact_arguments(seed=seed, **kwargs)
        elif solver == "choco":
            return self.choco_arguments()
        elif solver == "z3":
            return self.z3_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, **kwargs)
        elif solver.startswith("minizinc"):  # also can have a subsolver
            return self.minizinc_arguments(solver, cores=cores, seed=seed, **kwargs)
        elif solver == "gurobi":
            return self.gurobi_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, intermediate=intermediate, opt=opt, **kwargs)
        elif solver == "cpo":
            return self.cpo_arguments(model=model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "pindakaas":
            return self.pindakaas_arguments(model=model, mem_limit=mem_limit, **kwargs)
        elif solver == "pumpkin":
            return self.pumpkin_arguments(model=model, mem_limit=mem_limit, **kwargs)
        elif solver == "pysat":
            return self.pysat_arguments(model=model, mem_limit=mem_limit, **kwargs)
        elif solver == "hexaly":
            return self.hexaly_arguments(model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "cplex":
            return self.cplex_arguments(cores=cores, **kwargs) 
        else:
            self.print_comment(f"setting parameters of {solver} is not (yet) supported")
            return dict(), None

    def run(
        self,
        instance:str,                           # path to the instance to run
        instance_name:str,                               # name of the instance to run
        open:Optional[callable] = None,         # how to 'open' the instance file
        seed: Optional[int] = None,             # random seed
        time_limit: Optional[int] = None,       # time limit for this single instance
        mem_limit: Optional[int] = None,        # MiB: 1024 * 1024 bytes
        cores: int = 1,                         
        solver: str = None,                     # which backend solver to use
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

            self.init_signal_handlers()

            # Set memory limit (if provided)
            if mem_limit is not None:
                self.set_memory_limit(mem_limit)

            # Set time limit (if provided)
            if time_limit is not None:
                self.set_time_limit(time_limit) # set remaining process time != wall time
    
            # ------------------------------ Parse instance ------------------------------ #

            time_parse = time.time()
            model = self.read_instance(instance, open=open)
            
            # model.objective_ = None
            time_parse = time.time() - time_parse
            
            # model, _, assumps = make_assump_model(model.constraints)
            
            # print(f"model: {model}", flush=True)
            if verbose: self.print_comment(f"took {time_parse:.4f} seconds to parse model")

            if time_limit and time_limit < _wall_time(p):
                raise TimeoutError("Time's up after parse")
            
            # ------------------------ Post CPMpy model to solver ------------------------ #

            solver_args, internal_options = self.solver_arguments(solver, model=model, seed=seed,
                                        intermediate=intermediate,
                                        cores=cores, mem_limit=_mib_as_bytes(mem_limit) if mem_limit is not None else None,
                                        **kwargs)
            

            # Post model to solver
            time_post = time.time()
            s = self.post_model(model, solver, solver_args)
            time_post = time.time() - time_post
            if verbose: self.print_comment(f"took {time_post:.4f} seconds to post model to {solver}")
            
            if time_limit and time_limit < _wall_time(p):
                raise TimeoutError("Time's up after post")
            
            # ------------------------------- Solve model ------------------------------- #
            
            if time_limit:
                # give solver only the remaining time
                time_limit = time_limit - _wall_time(p) - time_buffer
                # disable signal-based time limit and let the solver handle it (solvers don't play well with difference between cpu and wall time)
                self.set_time_limit(None)
                
                if verbose: self.print_comment(f"{time_limit}s left to solve")
            
            time_solve = time.time()
            try:
                if internal_options is not None:
                    internal_options(s) # Set more internal solver options (need access to native solver object)
                # is_sat = s.solve(time_limit=time_limit, assumptions=assumps, **solver_args)
                print(f"Solving with solver {solver}...", flush=True)
                if solver == "pysat:Cadical195":
                    is_sat = s.solve(**solver_args)
                else:
                    is_sat = s.solve(time_limit=time_limit, **solver_args)
            except RuntimeError as e:
                if "Program interrupted by user." in str(e): # Special handling for Exact
                    raise TimeoutError("Exact interrupted due to timeout")
                else:
                    raise e

            time_solve = time.time() - time_solve
            if verbose: self.print_comment(f"took {time_solve:.4f} seconds to solve")

            # ------------------------------- Print result ------------------------------- #

            self.print_result(s)

            from cpmpy.solvers.solver_interface import ExitStatus
            basename = os.path.basename(instance_name)
            path = "/home/orestis_ubuntu/work/"
            # path = "/cw/dtailocal/orestis/"
            
            if model.has_objective() and s.status().exitstatus == ExitStatus.OPTIMAL:
                for p in [0.25, 0.5, 0.75, 1]:
                    alt_model = cp.Model(model.constraints)
                    if model.objective_is_min:
                        alt_model += model.objective_ < int(p * model.objective_value())
                        alt_model.to_file(f"{path}/benchmarks/2025/ALL-XCSP-UNSAT/{basename}_{p}.pkl")
                    else:
                        alt_model += model.objective_ > int((2-p) * model.objective_value())
                        alt_model.to_file(f"{path}/benchmarks/2025/ALL-XCSP-UNSAT/{basename}_{2-p}.pkl")
                
                print(f"Saved OPTIMAL instance to {path}/benchmarks/2025/ALL-XCSP-UNSAT/{basename}_*.pkl", flush=True)
            elif s.status().exitstatus == ExitStatus.UNSATISFIABLE:
                model.to_file(f"{path}/benchmarks/2025/ALL-XCSP-UNSAT/{basename}.pkl")
                
                print(f"Saved UNSAT instance to {path}/benchmarks/2025/ALL-XCSP-UNSAT/{basename}.pkl", flush=True)
            # ------------------------------------- - ------------------------------------ #

            
        except MemoryError as e:
            disable_memory_limit()
            self.handle_memory_error(mem_limit)
            raise e
        except NotImplementedError as e:
            self.handle_not_implemented(e)
            raise e
        except TimeoutError as e:
            self.handle_exception(e) # TODO add callback for timeout?
            raise e
        except Exception as e:
            self.handle_exception(e)
            raise e
       
        
    def run_mus(
        self,
        instance:str,                           # path to the instance to run
        open:Optional[callable] = None,         # how to 'open' the instance file
        seed: Optional[int] = None,             # random seed
        time_limit: Optional[int] = None,       # time limit for this single instance
        mem_limit: Optional[int] = None,        # MiB: 1024 * 1024 bytes
        cores: int = 1,                         
        solver: str = None,                     # which backend solver to use
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

            self.init_signal_handlers()

            # Set memory limit (if provided)
            if mem_limit is not None:
                self.set_memory_limit(mem_limit)

            # Set time limit (if provided)
            if time_limit is not None:
                self.set_time_limit(time_limit) # set remaining process time != wall time
    
            # ------------------------------ Parse instance ------------------------------ #

            # print(f"Now solving: {instance}")
            
            time_parse = time.time()
            model = self.read_instance(instance, open=open)
            
            
            
            time_parse = time.time() - time_parse
            
            # print(f"model: {model}", flush=True)
            if verbose: self.print_comment(f"took {time_parse:.4f} seconds to parse model")

            if time_limit and time_limit < _wall_time(p):
                raise TimeoutError("Time's up after parse")
            
            # ------------------------ Post CPMpy model to solver ------------------------ #

            solver_args, internal_options = self.solver_arguments(solver, model=model, seed=seed,
                                        intermediate=intermediate,
                                        cores=cores, mem_limit=_mib_as_bytes(mem_limit) if mem_limit is not None else None,
                                        **kwargs)
            

            # # Post model to solver
            # time_post = time.time()
            # s = self.post_model(model, solver, solver_args)
            # time_post = time.time() - time_post
            # if verbose: self.print_comment(f"took {time_post:.4f} seconds to post model to {solver}")
            
            if time_limit and time_limit < _wall_time(p):
                raise TimeoutError("Time's up after post")
            
            # ------------------------------- Solve model ------------------------------- #
            
            if time_limit:
                # give solver only the remaining time
                time_limit = time_limit - _wall_time(p) - time_buffer
                # disable signal-based time limit and let the solver handle it (solvers don't play well with difference between cpu and wall time)
                self.set_time_limit(None)
                
                if verbose: self.print_comment(f"{time_limit}s left to solve")
            
            time_solve = time.time()
            try:
                mus_res, nb_rf, nb_mr, nb_symm, sat, unsat, total_solve_time = cp_mus(model.constraints, solver=solver, time_limit=time_limit, model_rotation=False, redundancy_removal=False, assumption_removal=False, use_symmetries=False, block=True, eager=False, **solver_args)
            except RuntimeError as e:
                if "Program interrupted by user." in str(e): # Special handling for Exact
                    raise TimeoutError("Exact interrupted due to timeout")
                else:
                    raise e

            time_solve = time.time() - time_solve
            if verbose: self.print_comment(f"took {time_solve:.4f} seconds to solve")

            # ------------------------------- Print result ------------------------------- #

            # self.print_result()
            self.print_comment(f"Number of constraints in MUS: {len(mus_res)}")
            self.print_mus(mus_res, model)
            self.print_comment(f"Number of refinement removals: {nb_rf}")
            self.print_comment(f"Number of model rotations: {nb_mr}")
            self.print_comment(f"Number of symmetric constraints: {nb_symm}")
            self.print_comment(f"Number of calls with answer SAT: {sat}")
            self.print_comment(f"Number of calls with answer UNSAT: {unsat}")
            self.print_comment(f"Total solve time inside MUS (s): {total_solve_time:.4f}")
            

            # ------------------------------------- - ------------------------------------ #

            
        except MemoryError as e:
            disable_memory_limit()
            self.handle_memory_error(mem_limit)
            raise e
        except NotImplementedError as e:
            self.handle_not_implemented(e)
            raise e
        except TimeoutError as e:
            self.handle_exception(e) # TODO add callback for timeout?
            raise e
        except Exception as e:
            self.handle_exception(e)
            raise e
        
    def run_conv_cp(
        self,
        instance:str,
        #open:Optional[callable] = None,
        seed: Optional[int] = None,
        time_limit: Optional[int] = None,
        mem_limit: Optional[int] = None, # MiB: 1024 * 1024 bytes
        cores: int = 1,
        solver: str = None,
        time_buffer: int = 0,
        intermediate: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        
        if not verbose:
            warnings.filterwarnings("ignore")
            
        instance_path = Path(kwargs.get('instance_name'))
        file_name = instance_path.name
        instance_name = file_name.replace(".opb.xz", "")
        cnf_output_path = Path(OUTPUT_CNF_DIR) / f"{instance_name}.cnf"
        gcnf_output_path = Path(OUTPUT_GCNF_DIR) / f"{instance_name}.gcnf"
            
        try:

            # --------------------------- Global Configuration --------------------------- #

            # Get the current process
            p = psutil.Process()

            self.init_signal_handlers()

            # Set memory limit (if provided)
            if mem_limit is not None:
                # mem_limit is in MiB, convert to bytes for resource module
                mem_limit_bytes = mem_limit * 1024 * 1024
                self.set_memory_limit(mem_limit_bytes) # Assuming self.set_memory_limit handles resource.setrlimit

            # Set time limit (if provided)
            if time_limit is not None:
                self.set_time_limit(time_limit) # set remaining process time != wall time

            # ------------------------------ Parse instance ------------------------------ #

            time_parse = time.time()
            # NOTE: self.read_instance needs to handle the decompression of .opb.xz 
            # and return the cpmpy Model object.
            model = self.read_instance(instance, open=open) 
            time_parse = time.time() - time_parse
            
            if verbose: self.print_comment(f"took {time_parse:.4f} seconds to parse model")

            if time_limit and time_limit < _wall_time(p):
                raise TimeoutError("Time's up after parse")
            
            # ------------------------ Post CPMpy model to solver ------------------------ #
            time_solve = time.time() # Start total conversion timer

            # Setup for conversion: This part prepares the model components needed for CNF generation
            model, _, assumps = make_assump_model(model.constraints, name="sel")
            pysat = CPM_pysat(model)
            constrs = model.constraints[0]
            
            time_pb_to_cnf = time.time()
            cpm_cons = toplevel_list(constrs)
            cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
            cpm_cons = decompose_in_tree(cpm_cons, supported=frozenset({"alldifferent"}), csemap=pysat._csemap)
            cpm_cons = simplify_boolean(cpm_cons)
            cpm_cons = flatten_constraint(cpm_cons, csemap=pysat._csemap)
            cpm_cons = only_bv_reifies(cpm_cons, csemap=pysat._csemap)
            cpm_cons = only_implies(cpm_cons, csemap=pysat._csemap)
            cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum","wsum", "->", "and", "or"}), csemap=pysat._csemap)
            cpm_cons = int2bool(cpm_cons, pysat.ivarmap, encoding=pysat.encoding)
            cpm_cons = only_positive_coefficients(cpm_cons)
            formula = []
            # Get the integer literals for the assumption variables used for reification
            assumps_literals = set(map(pysat.solver_var, assumps)) 
            # dmap = dict(zip(assumps, pysat.solver_var(assumps)))
            
            # ------------------------------- Solve model / Convert ------------------------------- #
            
            # Adapt time limit setting for the conversion phase
            if time_limit:
                time_limit_remaining = time_limit - _wall_time(p) - time_buffer
                self.set_time_limit(None) # Disable signal-based time limit for the main block
                if verbose: self.print_comment(f"{time_limit_remaining}s left for conversion")
            

            # --- 1. PB to CNF Conversion ---
            if verbose: self.print_comment(f"Starting PB-to-CNF conversion...")

            
            for constraint in cpm_cons:
                if constraint.name == 'or':
                    formula.append(pysat.solver_vars(constraint.args))
                elif constraint.name == '->':
                    a0, a1 = constraint.args
                    if isinstance(a1, _BoolVarImpl):
                        formula.append([pysat.solver_var(~a0), pysat.solver_var(a1)])
                    elif isinstance(a1, Operator) and a1.name == 'or':
                        formula.append([pysat.solver_var(~a0)] + pysat.solver_vars(a1.args))
                    elif isinstance(a1, Comparison) and a1.args[0].name == "sum":
                        cnf = pysat._pysat_cardinality(a1, reified=True)
                        formula.extend([[pysat.solver_var(~a0)] + c for c in cnf])
                    elif isinstance(a1, Comparison) and a1.args[0].name == "wsum":
                        formula.extend(pysat._pysat_pseudoboolean(a1, conditional=a0))
                elif isinstance(constraint, Comparison):
                    if constraint.args[0].name == "sum":
                        formula.extend(pysat._pysat_cardinality(constraint))
                    elif constraint.args[0].name == "wsum":
                        formula.extend(pysat._pysat_pseudoboolean(constraint))

            # Save CNF
            cnf_obj = CNF(from_clauses=formula)
            cnf_obj.to_file(str(cnf_output_path))
            
            # --- NEW PRINT FOR PB-to-CNF TIME ---
            time_pb_to_cnf = time.time() - time_pb_to_cnf
            self.print_comment(f"took {time_pb_to_cnf:.4f} seconds to convert PB to CNF")
            # ------------------------------------

            # --- 2. CNF to GCNF Conversion (Optimized) ---
            if verbose: self.print_comment(f"Starting CNF-to-GCNF conversion...")

            time_cnf_to_gcnf = time.time()

            assumps_abs = sorted([abs(a) for a in assumps_literals])
            sel_vars = []
            
            # OPTIMIZATION: Build a mapping table instead of using O(N) list search per literal
            max_var = cnf_obj.nv # Get max variable index from the CNF object
            
            # Build the remapping table for non-selection variables
            var_mapping = {}
            shift = 0
            
            for i in range(1, max_var + 1):
                if i in assumps_abs:
                    shift += 1
                    # Selection variable is removed from the regular variable space
                else:
                    # Regular variable: new index is original index - accumulated shift
                    var_mapping[i] = i - shift
            
            # Process clauses and apply mapping
            soft_lines = []  # Group clauses
            hard_lines = []  # Non-group clauses
            real_count = 0
            vars_set = set()
            
            for line_ints in formula: # formula is the in-memory list of clauses
                
                real_group = False
                current_line_sel_var = None

                # Check for selection variable
                for l in line_ints:
                    if -l in assumps_literals:
                        current_line_sel_var = -l
                        real_group = True
                        break
                        
                new_line = []
                if real_group:
                    real_count += 1
                    if current_line_sel_var not in sel_vars:
                        sel_vars.append(current_line_sel_var)
                    
                    group_index = sel_vars.index(current_line_sel_var) + 1
                    new_line.append(f"{{{group_index}}}")
                else:
                    new_line.append("{0}")

                # Apply optimized mapping to literals
                for l in line_ints:
                    if l == 0:
                        continue
                    
                    if -l in assumps_literals:
                        # Skip the selection literal itself in the clause body
                        continue
                    else:
                        abs_l = abs(l)
                        new_abs_l = abs_l
                        
                        if l not in assumps_literals:
                        
                            vars_set.add(new_abs_l)
                        
                        if l > 0:
                            new_line.append(f"{new_abs_l}")
                        else:
                            new_line.append(f"-{new_abs_l}")

                new_line.append("0")
                
                # Group clauses are written first in GCNF format
                if real_group:
                    soft_lines.append(" ".join(new_line))
                else:
                    hard_lines.append(" ".join(new_line))
                    
            # Construct GCNF file content
            nb_vars = len(vars_set) + len(assumps_literals)
            nb_clauses = len(soft_lines) + len(hard_lines)
            nb_groups = len(sel_vars)
            header = f"p gcnf {nb_vars} {nb_clauses} {nb_groups}"
            dimacs_gcnf = "\n".join([header] + soft_lines + hard_lines) + "\n"
            
            with open(gcnf_output_path, "w") as f:
                f.write(dimacs_gcnf)

            # ------------------- End of Conversion Logic -------------------
            
            # --- NEW PRINT FOR CNF-to-GCNF TIME ---
            time_cnf_to_gcnf = time.time() - time_cnf_to_gcnf
            self.print_comment(f"took {time_cnf_to_gcnf:.4f} seconds to convert CNF to GCNF")
            # ------------------------------------

            time_solve = time.time() - time_solve # Total conversion time
            self.print_comment(f"took {time_solve:.4f} seconds to convert")
            
            # ... (return value structure would go here, e.g., a dictionary of results) ...
            return {
                'status': 'CONVERTED',
                'time_total': time_parse + time_solve,
                'time_convert': time_solve,
                'cnf_path': str(cnf_output_path),
                'gcnf_path': str(gcnf_output_path),
                'nb_vars_gcnf': nb_vars,
                'nb_clauses_gcnf': nb_clauses
            }
            # ------------------------------------- - ------------------------------------ #

        except MemoryError as e:
            # NOTE: disable_memory_limit() should be called from the parent process 
            # or globally managed if using signals, but for this context:
            # disable_memory_limit() # Placeholder
            self.handle_memory_error(mem_limit)
            raise e
        except NotImplementedError as e:
            self.handle_not_implemented(e)
            raise e
        except TimeoutError as e:
            self.handle_exception(e) # TODO add callback for timeout?
            raise e
        except Exception as e:
            self.handle_exception(e)
            raise e
        
    def run_conv(
        self,
        instance:str,
        #open:Optional[callable] = None,
        seed: Optional[int] = None,
        time_limit: Optional[int] = None,
        mem_limit: Optional[int] = None, # MiB: 1024 * 1024 bytes
        cores: int = 1,
        solver: str = None,
        time_buffer: int = 0,
        intermediate: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        
        if not verbose:
            warnings.filterwarnings("ignore")
            
        instance_path = Path(kwargs.get('instance_name'))
        file_name = instance_path.name
        instance_name = file_name.replace(".opb.xz", "")
        cnf_output_path = Path(OUTPUT_CNF_DIR) / f"{instance_name}.cnf"
        gcnf_output_path = Path(OUTPUT_GCNF_DIR) / f"{instance_name}.gcnf"
            
        try:

            # --------------------------- Global Configuration --------------------------- #

            # Get the current process
            p = psutil.Process()

            self.init_signal_handlers()

            # Set memory limit (if provided)
            if mem_limit is not None:
                # mem_limit is in MiB, convert to bytes for resource module
                mem_limit_bytes = mem_limit * 1024 * 1024
                self.set_memory_limit(mem_limit_bytes) # Assuming self.set_memory_limit handles resource.setrlimit

            # Set time limit (if provided)
            if time_limit is not None:
                self.set_time_limit(time_limit) # set remaining process time != wall time

            # ------------------------------ Parse instance ------------------------------ #

            time_parse = time.time()
            # NOTE: self.read_instance needs to handle the decompression of .opb.xz 
            # and return the cpmpy Model object.
            model = self.read_instance(instance, open=open) 
            
            
            # ------------------------ Post CPMpy model to solver ------------------------ #

            # Setup for conversion: This part prepares the model components needed for CNF generation
            time_parse = time.time() - time_parse
            # print(model)
            
            if verbose: self.print_comment(f"took {time_parse:.4f} seconds to parse model")

            if time_limit and time_limit < _wall_time(p):
                raise TimeoutError("Time's up after parse")
            
            start_cp_to_cnf = time.time()
            
            
            
            gcnf = write_gcnf(model.constraints, fname=gcnf_output_path, normalize=True)
            
            # first line looks like p gcnf <num_vars> <num_clauses> <num_groups>
            first_line = gcnf.splitlines()[0]
            _, _, nb_vars_str, nb_clauses_str, _ = first_line.split()
            nb_vars = int(nb_vars_str)
            nb_clauses = int(nb_clauses_str)
            
            # assert not pysat.solve(assumptions=list(assumps)), "The CNF model is SAT"
            
            # ------------------------------- Solve model / Convert ------------------------------- #
            
            # Adapt time limit setting for the conversion phase
            if time_limit:
                time_limit_remaining = time_limit - _wall_time(p) - time_buffer
                self.set_time_limit(None) # Disable signal-based time limit for the main block
                if verbose: self.print_comment(f"{time_limit_remaining}s left for conversion")

            
            # --- NEW PRINT FOR PB-to-CNF TIME ---
            time_cp_to_cnf = time.time() - start_cp_to_cnf
            self.print_comment(f"took {time_cp_to_cnf:.4f} seconds to convert")
            
            # ... (return value structure would go here, e.g., a dictionary of results) ...
            return {
                'status': 'CONVERTED',
                'time_total': time_parse + time_cp_to_cnf,
                'time_convert': time_cp_to_cnf,
                'cnf_path': str(cnf_output_path),
                'gcnf_path': str(gcnf_output_path),
                'nb_vars_gcnf': nb_vars,
                'nb_clauses_gcnf': nb_clauses
            }
            # ------------------------------------- - ------------------------------------ #

        except MemoryError as e:
            # NOTE: disable_memory_limit() should be called from the parent process 
            # or globally managed if using signals, but for this context:
            # disable_memory_limit() # Placeholder
            self.handle_memory_error(mem_limit)
            raise e
        except NotImplementedError as e:
            self.handle_not_implemented(e)
            raise e
        except TimeoutError as e:
            self.handle_exception(e) # TODO add callback for timeout?
            raise e
        except Exception as e:
            self.handle_exception(e)
            raise e
        
        
