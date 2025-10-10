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
import random
import psutil
import warnings
from enum import Enum
from typing import Optional

import cpmpy as cp
from cpmpy.tools.benchmark import _mib_as_bytes, _wall_time, set_memory_limit, set_time_limit, _bytes_as_mb, _bytes_as_gb, disable_memory_limit

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

    def __init__(self, reader:callable, exit_status:Enum):
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
                    _self.print_comment('Solution %i, time = %0.2fs' % 
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
                            _self.print_comment('Solution %i, time = %0.2fs' % 
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
                        _self.print_comment('Solution %i, time = %0.2fs' % 
                                    (self.__solution_count, current_time - self.__start_time))
                        _self.print_intermediate(obj)
                        self.__solution_count += 1

                def solution_count(self):
                    """Returns the number of solutions found."""
                    return self.__solution_count

            # Register the callback
            res |= { "solution_callback": CpoSolutionCallback }

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

    def post_model(self, model, solver, solver_args):
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
        else:
            self.print_comment(f"setting parameters of {solver} is not (yet) supported")
            return dict()

    def run(
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

            time_parse = time.time()
            model = self.read_instance(instance, open=open)
            time_parse = time.time() - time_parse
            if verbose: self.print_comment(f"took {time_parse:.4f} seconds to parse model [{instance}]")

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
        
    
    