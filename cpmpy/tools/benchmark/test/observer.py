from abc import ABC

import logging
import signal
import sys
import warnings
import os
import time
from typing import Optional
import contextlib
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.benchmark.opb import solution_opb
from cpmpy.tools.benchmark import set_memory_limit, set_time_limit, _bytes_as_mb, _bytes_as_gb, _mib_as_bytes

from .runner import Runner


class Observer(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

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
            True if the exception should be suppressed, False/None to propagate.
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


# ---------------------------------------------------------------------------- #
#                       Collection of pre-made observers:                      #
# ---------------------------------------------------------------------------- #


class HandlerObserver(Observer):

    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
        # Use a unique logger name for this observer instance
        self.logger = logging.getLogger(f"{__name__}.LoggerObserver")
        # Set level to INFO to ensure messages are logged
        self.logger.setLevel(logging.INFO)
        # Disable propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
        # Store reference to original stdout to always print there, even if redirected
        self.original_stdout = sys.__stdout__
        # Always add a new handler to ensure it writes to original stdout
        # Remove existing handlers first to avoid duplicates
        self.logger.handlers.clear()
        handler = logging.StreamHandler(self.original_stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Force the logger to be effective at INFO level
        self.logger.disabled = False

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
        # Use info level to log comments
        self.logger.info(comment)
        # Also ensure it's flushed immediately
        for handler in self.logger.handlers:
            handler.flush()


class CompetitionPrintingObserver(Observer):

    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
    
    def print_comment(self, comment: str):
        # Comment is already formatted by Runner.print_comment() before being passed to observers
        # So just print it as-is
        print(comment.rstrip('\n'), end="\r\n", flush=True)

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
    def __init__(self, time_limit: Optional[int] = None, mem_limit: Optional[int] = None, **kwargs):
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

    def __init__(self, **kwargs):
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


class RuntimeObserver(Observer):

    def __init__(self, **kwargs):
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
    def __init__(self, output_file: str, overwrite: bool = True, **kwargs):
        self.file_path = output_file
        self.file_handle = None
        self.context_active = False
        self.overwrite = overwrite
        self.file_opened = False  # Track if file has been opened in write mode

    def get_context_manager(self, runner: Runner):
        """Return a context manager that redirects stdout to a file."""
        @contextlib.contextmanager
        def redirect_to_file():
            # If overwrite and file hasn't been opened yet, open in write mode
            # Otherwise, append to preserve existing content
            mode = 'w' if (self.overwrite and not self.file_opened) else 'a'
            with open(self.file_path, mode) as f:
                self.file_handle = f
                self.context_active = True
                self.file_opened = True
                with contextlib.redirect_stdout(f):
                    yield
                self.context_active = False
                self.file_handle = None
        return redirect_to_file()
    
    def print_comment(self, comment: str, runner: 'Runner' = None):
        """Write comments to the file using the print_comment hook (in addition to stdout)."""
        # Comment is already formatted by Runner.print_comment() before being passed to observers
        formatted_comment = comment.rstrip('\n\r')
        
        if self.context_active and self.file_handle is not None:
            # Context is active, write directly to the file handle
            self.file_handle.write(formatted_comment + '\r\n')
            self.file_handle.flush()
        else:
            # Context not active yet or has exited
            # If overwrite and file hasn't been opened, open in write mode to truncate
            # Otherwise, append to preserve existing content
            if self.overwrite and not self.file_opened:
                mode = 'w'
                self.file_opened = True
            else:
                mode = 'a'
            with open(self.file_path, mode) as f:
                f.write(formatted_comment + '\r\n')
    
    def observe_init(self, runner: Runner):
        """Store reference to runner so we can access instance_runner."""
        self._runner = runner 