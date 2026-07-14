from __future__ import annotations

import psutil
import sys
import warnings
import time
from typing import Optional
import contextlib
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.benchmark import _wall_time


class ObserverContext:
    """
    Context manager with registerable observers.
    Upon entering the context, all context managers from the observers are entered.
    """
    def __init__(self, observers: list, runner: Runner):
        """
        Arguments:
            observers: List of observers to register
            runner: Runner instance
        """
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
            # Let observers handle it and decide if exception should be suppressed
            suppress_exception = False
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

class Runner:
    """
    Generic experiment runner for loading, transforming and solving CP model, with registerable observers.

    Forms the core of any experiment-specific runner. Executes a very simple flow:
    1) Read the instance
    2) Post the model
    3) Solve the model
    4) Report the result
    (some stages can be optional)

    If your experiment does not match this execution flow, this runner is not the right tool for you.

    The runner utilises the observer pattern to provide hookable callbacks at many points in the execution flow.
    So this is really the core module to build your own experiment runner on top of.
    """

    def __init__(self, reader: callable):
        """
        Arguments:
            reader: Reader function to read the instance
        """
        self.observers = []
        self.solver_args = {}
        self.reader = reader

    def register_observer(self, observer):
        """
        Register an observer.
        """
        self.observers.append(observer)

    def read_instance(self, instance: str):
        return self.reader(instance)

    def post_model(self, model: cp.Model, solver:str):
        return cp.SolverLookup.get(solver, model)

    def run(self, instance: str, solver: Optional[str] = None, time_limit: Optional[int] = None, mem_limit: Optional[int] = None, seed: Optional[int] = None, intermediate: bool = False, cores: int = 1, **kwargs):
        """
        Run the runner.
        
        Arguments:
            instance: Instance file path
            solver: Solver to use
            time_limit: Time limit in seconds
            mem_limit: Memory limit in bytes
            seed: Random seed
            intermediate: Whether to print intermediate solutions
            cores: Number of cores to use
            **kwargs: Additional arguments

        Returns:
            True if the instance is satisfiable, False otherwise

        #
        # -------------------- Runner Execution Flow (with hooks) --------------------
        #
        #  0. [Hook]   with self.observer_context():
        #              - enter each observer.get_context_manager(...)
        #  1. [Hook]   observe_init()
        #  2.          self.model = self.read_instance(instance)
        #  3. [Hook]   observe_pre_transform()
        #  4.          self.s = self.post_model(self.model, solver)
        #  5. [Hook]   observe_post_transform()
        #  6.          self.solver_args = self.collect_solver_args()
        #  7. [Hook]   observe_pre_solve()
        #  8.          self.s.solve(..., **self.solver_args)
        #     [Hook]   observe_intermediate(objective)     # (may be called multiple times)
        #  9.          collect run metadata (runner_metadata, stage_timings)
        # 10. [Hook]   observe_post_solve()
        # 11. [Hook]   observe_end()
        # 12. [Hook]   exit self.observer_context():
        #              - on exception: observe_exception(...)
        #              - always: observe_exit(...)
        #
        #  At each point marked [Hook], all registered observers are notified via
        #  the corresponding method. This allows for monitoring, logging, statistics,
        #  backend posting, or output file writing at every stage of execution.
        #
        # --------------------------------------------------------------------------- #
        """

        # Set up runner attributes
        self.solver = solver
        self.time_limit = time_limit
        self.mem_limit = mem_limit
        self.seed = seed
        self.intermediate = intermediate
        self.cores = cores
        self.kwargs = kwargs
        self.time_buffer = 1 # Time buffer to account for solver overhead
        self.verbose = bool(kwargs.get("verbose", False))


        # 0) [Hook] Enter observer context managers (`get_context_manager(...)` for each observer).
        with self.observer_context():
            t0_total = time.perf_counter()

            # 1) [Hook] observe_init()
            self.observe_init()
            
            # 2) Read/parse instance.
            t0_parse = time.perf_counter()
            with self.print_forwarding_context():
                self.model = self.read_instance(instance)
            t1_parse = time.perf_counter()

            # 3) [Hook] observe_pre_transform()
            self.observe_pre_transform()
            t0_post = time.perf_counter()

            # 4) Transform/post model to solver.
            with self.print_forwarding_context():
                self.s = self.post_model(self.model, solver)
            t1_post = time.perf_counter()
            
            # 5) [Hook] observe_post_transform()
            self.observe_post_transform()

            # 6) Collect solver args from observers.
            self.solver_args = self.collect_solver_args()            

            if self.time_limit:
                # Get the current process
                p = psutil.Process()
                
                # Give solver only the remaining time
                time_limit = self.time_limit - _wall_time(p) - self.time_buffer
                if self.verbose:
                    self.print_comment(f"{time_limit}s left to solve")
            
            else:
                time_limit = None

            time_limit_expired_before_solve = time_limit is not None and time_limit < 0
                    
            # 7) [Hook] observe_pre_solve()
            self.observe_pre_solve()

            # 8) Solve model ([Hook] observe_intermediate(objective) may occur multiple times).
            t0_solve = time.perf_counter()
            if time_limit_expired_before_solve:
                self.is_sat = None
                self.print_comment(f"Timeout: Time limit of {self.time_limit} seconds reached before solve")
            else:
                with self.print_forwarding_context():
                    self.is_sat = self.s.solve(time_limit = time_limit, **self.solver_args)
            t1_solve = time.perf_counter()
            
            # Check if solver timed out (UNKNOWN status with time limit set)
            termination_reason = "walltime" if time_limit_expired_before_solve else None
            if not time_limit_expired_before_solve and time_limit is not None and self.s.status().exitstatus == CPMStatus.UNKNOWN:
                # Check if we're near the time limit (within 2 seconds)
                p = psutil.Process()
                elapsed = _wall_time(p)
                if elapsed >= self.time_limit - 2:
                    termination_reason = "walltime"
                    self.print_comment(f"Timeout: Solver reached time limit of {self.time_limit} seconds (elapsed: {elapsed:.2f}s)")

            # 9) Collect run metadata (runner_metadata, stage_timings).
            # Expose generic runner metadata so observers/backends can report exit details
            try:
                solver_status = self.s.status().exitstatus if self.s is not None else None
                status_name = solver_status.name if hasattr(solver_status, "name") else str(solver_status)
            except Exception:
                status_name = None

            p = psutil.Process()
            elapsed = _wall_time(p)
            t0_result = time.perf_counter()
            try:
                # Access solver status/objective to force result retrieval cost now.
                _ = self.s.status() if self.s is not None else None
                if self.is_sat:
                    _ = self.s.objective_value()
            except Exception:
                pass
            t1_result = time.perf_counter()

            stage_timings = {
                "parse_instance": max(t1_parse - t0_parse, 0.0),
                # In the Python runner, post_model() performs model transformation + posting.
                "transform_model": max(t1_post - t0_post, 0.0),
                "post_model": max(t1_post - t0_post, 0.0),
                "solve_model": max(t1_solve - t0_solve, 0.0),
                "retrieve_result": max(t1_result - t0_result, 0.0),
                "total": max(time.perf_counter() - t0_total, 0.0),
            }

            self.termination_reason = termination_reason
            self.runner_metadata = {
                "solver": self.solver,
                "termination_reason": termination_reason,
                "exit_status": status_name,
                "exit": {
                    "raw": None,
                    "code": None,
                    "signal": None,
                    "status_name": status_name,
                },
                "limits": {
                    "time_limit": self.time_limit,
                    "memory_limit_mb": self.mem_limit,
                    "cores": self.cores,
                },
                "stats": {
                    "walltime": elapsed,
                    "cputime": None,
                    "memory": None,
                },
                "stage_timings": stage_timings,
            }

            # 10) [Hook] observe_post_solve()
            self.observe_post_solve()

            # 11) [Hook] observe_end()
            self.observe_end()

            # 12) [Hook] Exit observer context on block exit:
            #     - on exception: observe_exception(...)
            #     - always: observe_exit(...)
            return self.is_sat

    def print_comment(self, comment: str):
        # Pass comment to all observers
        for observer in self.observers:
            observer.print_comment(comment, runner=self)

    def print_raw(self, text: str):
        # Pass raw text only to observers.
        # Output targets (stdout/file/etc.) are handled by concrete observers.
        for observer in self.observers:
            observer.print_raw(text)

    def _forward_captured_print_line(self, text: str):
        """Forward solver-library stdout captured by print_forwarding_context."""
        from ..observer.write_to_stdout import write_raw_to_stdout

        for observer in self.observers:
            if getattr(observer, "bypass_print_forwarding", False):
                continue
            observer.print_raw(text)
        if self.verbose:
            write_raw_to_stdout(text)

    @contextlib.contextmanager
    def print_forwarding_context(self):
        """Context manager that forwards all print statements and warnings to observers."""
        class PrintForwarder:
            def __init__(self, runner):
                self.runner = runner
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr
                self._pending_line = ""
                self._forwarding = False

            def _forward_line(self, line: str):
                line = line.rstrip()
                if not line.strip():
                    return
                if line.startswith("c" + chr(32)):
                    formatted = line
                else:
                    formatted = "c" + chr(32) + line
                self.runner._forward_captured_print_line(formatted)

            def write(self, text):
                if self._forwarding:
                    self.original_stdout.write(text)
                    return
                if not text:
                    return
                text = self._pending_line + text
                self._pending_line = ""

                lines = text.split("\n")
                if text.endswith("\n"):
                    complete_lines = lines[:-1]
                else:
                    complete_lines = lines[:-1]
                    self._pending_line = lines[-1]

                self._forwarding = True
                try:
                    for line in complete_lines:
                        self._forward_line(line)
                finally:
                    self._forwarding = False

            def flush(self):
                if self._pending_line.strip():
                    self._forwarding = True
                    try:
                        self._forward_line(self._pending_line)
                    finally:
                        self._forwarding = False
                    self._pending_line = ""
                if self.runner.verbose:
                    self.original_stdout.flush()

            def forward_to_observers(self):
                if self._pending_line.strip():
                    self._forwarding = True
                    try:
                        self._forward_line(self._pending_line)
                    finally:
                        self._forwarding = False
                    self._pending_line = ""
        
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            """Custom warning handler that forwards warnings to observers."""
            # Format the warning message
            warning_msg = f"{category.__name__}: {str(message).rstrip()}"
            # Forward to observers
            self.print_comment(warning_msg)
            # Mirror warnings to console only in verbose mode
            if self.verbose:
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


    # ---------------------------------------------------------------------------- #
    #                            Observer callback hooks                           #
    # ---------------------------------------------------------------------------- #

    def observer_context(self):
        return ObserverContext(observers=self.observers, runner=self)

    def observe_init(self):
        for observer in self.observers:
            observer.observe_init(runner=self)

    def observe_pre_transform(self):
        for observer in self.observers:
            try:
                observer.observe_pre_transform(runner=self)
            except Exception as e:
                import logging
                logging.warning(f"Observer {observer.__class__.__name__}.observe_pre_transform failed: {e}")

    def observe_post_transform(self):
        for observer in self.observers:
            try:
                observer.observe_post_transform(runner=self)
            except Exception as e:
                import logging
                logging.warning(f"Observer {observer.__class__.__name__}.observe_post_transform failed: {e}")

    def observe_pre_solve(self):
        for observer in self.observers:
            try:
                observer.observe_pre_solve(runner=self)
            except Exception as e:
                import logging
                logging.warning(f"Observer {observer.__class__.__name__}.observe_pre_solve failed: {e}")

    def observe_post_solve(self):
        for observer in self.observers:
            try:
                observer.observe_post_solve(runner=self)
            except Exception as e:
                import logging
                logging.warning(f"Observer {observer.__class__.__name__}.observe_post_solve failed: {e}")

    def observe_intermediate(self, objective):
        """Notify all observers of an intermediate solution objective value."""
        for observer in self.observers:
            observer.observe_intermediate(runner=self, objective=objective)

    def observe_end(self):
        for observer in self.observers:
            try:
                observer.observe_end(runner=self)
            except Exception as e:
                import logging
                logging.warning(f"Observer {observer.__class__.__name__}.observe_end failed: {e}")

    def collect_solver_args(self):
        solver_args = {}
        for observer in self.observers:
            observer.participate_solver_args(runner=self, solver_args=solver_args)
        return solver_args
