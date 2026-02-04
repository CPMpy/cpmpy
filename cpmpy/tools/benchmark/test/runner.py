from __future__ import annotations

import psutil
import sys
import warnings
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
    Generic runner with registerable observers.
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
        """

        self.solver = solver
        self.time_limit = time_limit
        self.mem_limit = mem_limit
        self.seed = seed
        self.intermediate = intermediate
        self.cores = cores
        self.kwargs = kwargs
        self.time_buffer = 1
        self.verbose = True


        with self.observer_context(): # Enter all context managers from the observers
            self.observe_init()

            with self.print_forwarding_context():
                self.model = self.read_instance(instance)

            self.observe_pre_transform()
            with self.print_forwarding_context():
                self.s = self.post_model(self.model, solver)
            self.observe_post_transform()

            self.solver_args = self.collect_solver_args()            

            if self.time_limit:
                # Get the current process
                p = psutil.Process()
                
                # Give solver only the remaining time
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

            return self.is_sat

    def print_comment(self, comment: str):
        # Format the comment using instance_runner if available, before passing to observers
        formatted_comment = comment
        if hasattr(self, 'instance_runner') and self.instance_runner is not None:
            # Capture the formatted output from instance_runner.print_comment
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.instance_runner.print_comment(comment)
                formatted_comment = sys.stdout.getvalue().rstrip('\n\r')
            finally:
                sys.stdout = old_stdout
        
        # Pass formatted comment to all observers
        for observer in self.observers:
            # Pass runner to print_comment if observer accepts it
            if hasattr(observer.print_comment, '__code__'):
                import inspect
                sig = inspect.signature(observer.print_comment)
                if 'runner' in sig.parameters:
                    observer.print_comment(formatted_comment, runner=self)
                else:
                    observer.print_comment(formatted_comment)
            else:
                observer.print_comment(formatted_comment)

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

    def collect_solver_args(self):
        solver_args = {}
        for observer in self.observers:
            observer.participate_solver_args(runner=self, solver_args=solver_args)
        return solver_args
