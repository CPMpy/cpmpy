from abc import ABC, abstractmethod

import psutil
from cpmpy.model import Model
import logging
import signal
import argparse
import sys
import warnings
import os
import time
from pathlib import Path
from typing import Optional
from functools import partial
import contextlib
import cpmpy as cp
from cpmpy.solvers import solver_interface
from cpmpy.tools.benchmark import set_time_limit, set_memory_limit

from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.benchmark.opb import solution_opb
from cpmpy.tools.benchmark import _mib_as_bytes, _wall_time, set_memory_limit, set_time_limit, _bytes_as_mb, _bytes_as_gb, disable_memory_limit


class Runner:

    def __init__(self, reader: callable):
        self.observers = []
        self.solver_args = {}
        self.reader = reader

    def register_observer(self, observer):
        self.observers.append(observer)

    def read_instance(self, instance: str):
        return self.reader(instance)

    def post_model(self, model: cp.Model, solver:str):
        return cp.SolverLookup.get(solver, model)

    def run(self, instance: str, solver: Optional[str] = None, time_limit: Optional[int] = None, mem_limit: Optional[int] = None, seed: Optional[int] = None, intermediate: bool = False, cores: int = 1, **kwargs):
        self.solver = solver
        self.time_limit = time_limit
        self.mem_limit = mem_limit
        self.seed = seed
        self.intermediate = intermediate
        self.cores = cores
        self.kwargs = kwargs
        self.time_buffer = 1
        self.verbose = True

        with self.observer_context():
            self.observe_init()

            with self.print_forwarding_context():
                self.model = self.read_instance(instance)
            

            self.observe_pre_transform()
            with self.print_forwarding_context():
                self.s = self.post_model(self.model, solver)
            self.observe_post_transform()

            self.solver_args = self.participate_solver_args()

            if self.time_limit:
                # Get the current process
                p = psutil.Process()
                
                # give solver only the remaining time
                time_limit = self.time_limit - _wall_time(p) - self.time_buffer
                if self.verbose: self.print_comment(f"{time_limit}s left to solve")
            
            else:
                time_limit = None

            if time_limit is not None:
                if time_limit < 0:
                    raise TimeoutError(f"Time limit of {self.time_limit} seconds reached")
                    

            self.observe_pre_solve()
            with self.print_forwarding_context():
                self.is_sat = self.s.solve(time_limit = time_limit, **self.solver_args)
            self.observe_post_solve()
            
            # Check if solver timed out (UNKNOWN status with time limit set)
            if time_limit is not None and self.s.status().exitstatus == CPMStatus.UNKNOWN:
                # Check if we're near the time limit (within 2 seconds)
                p = psutil.Process()
                elapsed = _wall_time(p)
                if elapsed >= self.time_limit - 2:
                    self.print_comment(f"Timeout: Solver reached time limit of {self.time_limit} seconds (elapsed: {elapsed:.2f}s)")

            self.observe_end()

            #print(self.is_sat)
            return self.is_sat

    def print_comment(self, comment: str):
        for observer in self.observers:
            observer.print_comment(comment)

    @contextlib.contextmanager
    def print_forwarding_context(self):
        """Context manager that forwards all print statements and warnings to observers."""
        class PrintForwarder:
            def __init__(self, runner):
                self.runner = runner
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr
                self.buffer = []
            
            def write(self, text):
                # Buffer the output
                self.buffer.append(text)
                # Also write to original stdout to preserve normal behavior
                self.original_stdout.write(text)
            
            def flush(self):
                self.original_stdout.flush()
            
            def forward_to_observers(self):
                # Forward buffered output to observers line by line
                if self.buffer:
                    full_text = ''.join(self.buffer)
                    for line in full_text.splitlines(keepends=True):
                        if line.strip():  # Only forward non-empty lines
                            self.runner.print_comment(line.rstrip())
        
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            """Custom warning handler that forwards warnings to observers."""
            # Format the warning message
            warning_msg = f"{category.__name__}: {str(message).rstrip()}"
            # Forward to observers
            self.print_comment(warning_msg)
            # Also call the original warning handler to preserve normal behavior
            original_showwarning(message, category, filename, lineno, file, line)
        
        forwarder = PrintForwarder(self)
        original_showwarning = warnings.showwarning
        
        try:
            # Redirect stdout and stderr
            sys.stdout = forwarder
            sys.stderr = forwarder
            # Redirect warnings
            warnings.showwarning = warning_handler
            yield
        finally:
            # Restore stdout and stderr
            sys.stdout = forwarder.original_stdout
            sys.stderr = forwarder.original_stderr
            # Restore warnings
            warnings.showwarning = original_showwarning
            # Forward any remaining buffered output
            forwarder.forward_to_observers()

    def observer_context(self):
        return ObserverContext(observers=self.observers, runner=self)

    def observe_init(self):
        for observer in self.observers:
            observer.observe_init(runner=self)

    def observe_pre_transform(self):
        for observer in self.observers:
            observer.observe_pre_transform(runner=self)

    def observe_post_transform(self):
        for observer in self.observers:
            observer.observe_post_transform(runner=self)


    def observe_pre_solve(self):
        for observer in self.observers:
            observer.observe_pre_solve(runner=self)

    def observe_post_solve(self):
        for observer in self.observers:
            observer.observe_post_solve(runner=self)

    def observe_end(self):
        for observer in self.observers:
            observer.observe_end(runner=self)

    def participate_solver_args(self):
        solver_args = {}
        for observer in self.observers:
            observer.participate_solver_args(runner=self, solver_args=solver_args)
        return solver_args

class ObserverContext:
    def __init__(self, observers: list, runner: Runner):
        self.observers = observers or []
        self.runner = runner
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        # Enter all context managers from observers
        if self.observers:
            for observer in self.observers:
                cm = observer.get_context_manager(runner=self.runner)
                if cm is not None:
                    self.exit_stack.enter_context(cm)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # First, exit all context managers (in reverse order)
        # This happens automatically when we exit the ExitStack
        exit_result = None
        if self.exit_stack:
            exit_result = self.exit_stack.__exit__(exc_type, exc_value, traceback)
        
        if exc_type is not None and self.observers:
            # An exception occurred, notify all observers
            # Let ResourceLimitObserver handle it and decide if exception should be suppressed
            suppress_exception = True
            for observer in self.observers:
                try:
                    # Pass exception to observer, let it handle it
                    result = observer.observe_exception(runner=self.runner, exc_type=exc_type, exc_value=exc_value, traceback=traceback)
                    # If observer returns True, it wants to suppress the exception
                    if result is True:
                        suppress_exception = True
                except Exception:
                    # Don't let observer exceptions mask the original exception
                    pass
            
            # If any observer wants to suppress, suppress the exception
            if suppress_exception:
                return True
        
        # Always call observe_exit on all observers
        if self.observers:
            for observer in self.observers:
                try:
                    observer.observe_exit(runner=self.runner)
                except Exception:
                    # Don't let observer exceptions interfere with cleanup
                    pass
        
        # Return the exit result from ExitStack (False to propagate, True to suppress)
        return exit_result if exit_result is not None else False

class Observer(ABC):

    def observe_init(self, runner: Runner):
        pass

    def observe_pre_transform(self, runner: Runner):
        pass

    def observe_post_transform(self, runner: Runner):
        pass

    def observe_pre_solve(self, runner: Runner):
        pass

    def observe_post_solve(self, runner: Runner):
        pass

    def participate_solver_args(self, runner: Runner, solver_args: dict):
        return solver_args

    def observe_exception(self, runner: Runner, exc_type, exc_value, traceback):
        """
        Called when an exception occurs in the context.
        
        Returns:
            True if the exception should be suppressed, False/None to propagate it.
        """
        pass

    def observe_exit(self, runner: Runner):
        pass

    def observe_end(self, runner: Runner):
        pass

    def print_comment(self, comment: str):
        pass

    def observe_intermediate(self, runner: Runner, objective: int):
        pass

    def get_context_manager(self, runner: Runner):
        """
        Return a context manager that will be entered when the ObserverContext is entered.
        Return None if this observer doesn't provide a context manager.
        """
        return None

class HandlerObserver(Observer):

    def __init__(self):
        self.runner = None

    def observe_init(self, runner: Runner):
        self.runner = runner
        signal.signal(signal.SIGINT, self._sigterm_handler)
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)
        signal.signal(signal.SIGABRT, self._sigterm_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGXCPU, self._rlimit_cpu_handler)
        else:
            warnings.warn("Windows does not support setting SIGXCPU signal")

    def _sigterm_handler(self, _signo, _stack_frame):
        exit_code = self.handle_sigterm()
        print(flush=True)
        os._exit(exit_code)
        
    def _rlimit_cpu_handler(self, _signo, _stack_frame):
        # Raise TimeoutError - ObserverContext will handle notifying observers
        # Don't notify here to avoid duplicates
        raise TimeoutError("CPU time limit reached (SIGXCPU)")

    def handle_sigterm(self):
        return 0

    def handle_rlimit_cpu(self):
        return 0

class LoggerObserver(Observer):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Add a StreamHandler to output to stdout if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def observe_init(self, runner: Runner):
        self.logger.info("Initializing runner")

    def observe_pre_transform(self, runner: Runner):
        self.logger.info("Pre-transforming")

    def observe_post_transform(self, runner: Runner):
        self.logger.info("Post-transforming")

    def observe_pre_solve(self, runner: Runner):
        self.logger.info("Pre-solving")

    def observe_post_solve(self, runner: Runner):
        self.logger.info("Post-solving")

    def print_comment(self, comment: str):
        self.logger.info(comment)

class CompetitionPrintingObserver(Observer):

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def print_comment(self, comment: str):
        print('c' + chr(32) + comment.rstrip('\n'), end="\r\n", flush=True)

    def observe_post_solve(self, runner: Runner):
        self.print_result(runner.s)

    def observe_intermediate(self, objective: int):
        self.print_intermediate(objective)

    def print_status(self, status: str):
        print('s' + chr(32) + status, end="\n", flush=True)

    def print_value(self, value: str):
        print('v' + chr(32) + value, end="\n", flush=True)

    def print_objective(self, objective: int):
        print('o' + chr(32) + str(objective), end="\n", flush=True)

    def print_intermediate(self, objective: int):
        self.print_objective(objective)

    def print_result(self, s):
        if s.status().exitstatus == CPMStatus.OPTIMAL:
            self.print_objective(s.objective_value())
            self.print_value(solution_opb(s))
            self.print_status("OPTIMAL" + chr(32) + "FOUND")
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            self.print_objective(s.objective_value())
            self.print_value(solution_opb(s))
            self.print_status("SATISFIABLE")
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            self.print_status("UNSATISFIABLE")
        else:
            self.print_comment("Solver did not find any solution within the time/memory limit")
            self.print_status("UNKNOWN")

class ResourceLimitObserver(Observer):
    def __init__(self, time_limit: Optional[int] = None, mem_limit: Optional[int] = None):
        self.time_limit = time_limit
        self.mem_limit = mem_limit

    def observe_init(self, runner: Runner):
        if self.time_limit is not None: 
            set_time_limit(self.time_limit)
        if self.mem_limit is not None:
            set_memory_limit(self.mem_limit)
    
    def _handle_memory_error(self, runner: Runner, mem_limit: int):
        runner.print_comment(f"MemoryError raised. Reached limit of {mem_limit} MiB")

    def _handle_timeout(self, runner: Runner, time_limit: int):
        if time_limit is not None:
            runner.print_comment(f"TimeoutError raised. Reached limit of {time_limit} seconds")
        else:
            runner.print_comment(f"TimeoutError raised. CPU time limit reached")

    def observe_exception(self, runner: Runner, exc_type, exc_value, traceback):
        """
        Handle exceptions related to resource limits.
        Returns True to suppress the exception after handling.
        """
        if exc_type is MemoryError:
            # Only handle if we have a memory limit set
            if self.mem_limit is not None:
                self._handle_memory_error(runner=runner, mem_limit=self.mem_limit)
                return True  # Suppress the exception after handling
        elif exc_type is TimeoutError:
            # Only handle if we have a time limit set
            if self.time_limit is not None:
                self._handle_timeout(runner=runner, time_limit=self.time_limit)
                return True  # Suppress the exception after handling
        return False  # Don't suppress other exceptions
  

class SolverArgsObserver(Observer):

    def __init__(self):
        self.time_limit = None
        self.mem_limit = None
        self.seed = None
        self.intermediate = False
        self.cores = 1
        self.mem_limit = None
        self.kwargs = dict()

    def observe_init(self, runner: Runner):
        self.time_limit = runner.time_limit
        self.mem_limit = runner.mem_limit
        self.seed = runner.seed
        self.intermediate = runner.intermediate
        self.cores = runner.cores
        self.mem_limit = runner.mem_limit
        self.kwargs = runner.kwargs

    def _ortools_arguments(
            self,
            runner: Runner,
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
                    _self.print_comment('Solution %i, time = %0.4fs' % 
                                (self.__solution_count, current_time - self.__start_time))
                    _self.observe_intermediate(runner=runner, objective=obj)
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
    
    def _exact_arguments(
            self,
            seed: Optional[int] = None, 
            **kwargs
        ):
        # Documentation: https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp?ref_type=heads
        res = dict()
        if seed is not None: 
            res |= { "seed": seed }

        return res, None

    def _choco_arguments(self): 
        # Documentation: https://github.com/chocoteam/pychoco/blob/master/pychoco/solver.py
        return {}, None

    def _z3_arguments(
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

    def _minizinc_arguments(
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

    def _gurobi_arguments(
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

    def _cpo_arguments(
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

    def _cplex_arguments(
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
    
    def _hexaly_arguments(
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

    def _solver_arguments(
            self,
            runner: Runner,
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
            return self._ortools_arguments(runner, model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "exact":
            return self._exact_arguments(seed=seed, **kwargs)
        elif solver == "choco":
            return self._choco_arguments()
        elif solver == "z3":
            return self._z3_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, **kwargs)
        elif solver.startswith("minizinc"):  # also can have a subsolver
            return self._minizinc_arguments(solver, cores=cores, seed=seed, **kwargs)
        elif solver == "gurobi":
            return self._gurobi_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, intermediate=intermediate, opt=opt, **kwargs)
        elif solver == "cpo":
            return self._cpo_arguments(model=model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "hexaly":
            return self._hexaly_arguments(model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "cplex":
            return self._cplex_arguments(cores=cores, **kwargs) 
        else:
            runner.print_comment(f"setting parameters of {solver} is not (yet) supported")
            return dict(), None

    def participate_solver_args(self, runner: Runner, solver_args: dict):
        args, internal_options = self._solver_arguments(runner, runner.solver, model=runner.model, seed=self.seed,
                                        intermediate=self.intermediate,
                                        cores=self.cores, mem_limit=_mib_as_bytes(self.mem_limit) if self.mem_limit is not None else None,
                                        **self.kwargs)
    
        if internal_options is not None:
            internal_options(runner.s)
        solver_args |= args
        runner.print_comment(f"Solver arguments: {args}")

class ProfilingObserver(Observer):

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_transform_time = None
        self.end_transform_time = None

    def observe_init(self, runner: Runner):
        self.start_time = time.time()

    def observe_pre_transform(self, runner: Runner):
        self.start_transform_time = time.time()

    def observe_post_transform(self, runner: Runner):
        self.end_transform_time = time.time()
        runner.print_comment(f"Time taken to transform: {self.end_transform_time - self.start_transform_time} seconds")

    def observe_post_solve(self, runner: Runner):
        runner.print_comment(f"Time taken to solve: {runner.s.status().runtime} seconds")

    def observe_end(self, runner: Runner):
        runner.print_comment(f"Total time taken: {time.time() - self.start_time} seconds")

class SolutionCheckerObserver(Observer):

    def observe_end(self, runner: Runner):
        runner.print_comment(f"Run solution checker here...")

class WriteToFileObserver(Observer):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_context_manager(self, runner: Runner):
        """Return a context manager that redirects stdout to a file."""
        @contextlib.contextmanager
        def redirect_to_file():
            with open(self.file_path, 'w') as f:
                with contextlib.redirect_stdout(f):
                    yield
        return redirect_to_file() 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--solver", type=str, default="ortools")
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--mem_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--intermediate", action="store_true", default=False)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    # parser.add_argument("--kwargs", type=str, default="")

    args = parser.parse_args()


    if args.output_file is None:
        args.output_file = f"results/{args.solver}_{args.instance}.txt"
    else:
        args.output_file = f"results/{args.output_file}"

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)


    from cpmpy.tools.rcpsp import read_rcpsp
    from cpmpy.tools.dataset.problem.psplib import PSPLibDataset
    dataset = PSPLibDataset(root="./data", download=True)

    runner = Runner(reader=partial(read_rcpsp, open=dataset.open))
    # runner.register_observer(LoggerObserver())
    runner.register_observer(CompetitionPrintingObserver())
    runner.register_observer(ProfilingObserver())
    # runner.register_observer(ResourceLimitObserver(time_limit=args.time_limit, mem_limit=args.mem_limit))
    runner.register_observer(HandlerObserver())
    runner.register_observer(SolverArgsObserver())
    runner.register_observer(SolutionCheckerObserver())
    runner.register_observer(WriteToFileObserver(file_path=args.output_file))
    print(vars(args))
    runner.run(**vars(args))

if __name__ == "__main__":
    main()

    # from cpmpy.tools.dataset.model.xcsp3 import XCSP3Dataset
    # from cpmpy.tools.xcsp3 import read_xcsp3

    # from cpmpy.tools.dataset.model.opb import OPBDataset
    # from cpmpy.tools.opb import read_opb    

    # from cpmpy.tools.dataset.problem.jsplib import JSPLibDataset
    # from cpmpy.tools.jsplib import read_jsplib

    # from cpmpy.tools.dataset.problem.psplib import PSPLibDataset
    # from cpmpy.tools.rcpsp import read_rcpsp

    # # dataset = XCSP3Dataset(root="./data", year=2025, track="CSP25", download=True)
    # dataset = OPBDataset(root="./data", year=2024, track="DEC-LIN", download=True)
    # dataset = JSPLibDataset(root="./data", download=True)
    # dataset = PSPLibDataset(root="./data", download=True)

    # for instance, metadata in dataset:
    #     print(instance, metadata)
    #     runner = Runner(reader=partial(read_rcpsp, open=dataset.open))
    #     #runner.register_observer(LoggerObserver())
    #     runner.register_observer(CompetitionPrintingObserver())
    #     runner.register_observer(ProfilingObserver())
    #     #runner.register_observer(ResourceLimitObserver(time_limit=10, mem_limit=1024))
    #     runner.register_observer(HandlerObserver())
    #     runner.register_observer(SolverArgsObserver())
    #     runner.register_observer(SolutionCheckerObserver())
    #     runner.run(instance, solver="ortools")

    #     break