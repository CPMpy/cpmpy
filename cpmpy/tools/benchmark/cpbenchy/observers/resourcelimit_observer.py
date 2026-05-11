from typing import Optional

from cpmpy.tools.benchmark import set_memory_limit, set_time_limit

from ..runner import Runner
from .base import Observer


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
            runner.print_comment("TimeoutError raised. CPU time limit reached")

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