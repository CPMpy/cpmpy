import time

from ..runner.runner import Runner
from .base import Observer


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
        if runner.s is None:
            # Solver object not available (e.g., in parent process with RunExecResourceManager)
            # Try to get runtime from runner attributes or use elapsed time
            if hasattr(runner, "runtime") and runner.runtime is not None:
                runner.print_comment(f"Time taken to solve: {runner.runtime} seconds")
            elif self.start_time:
                elapsed = time.time() - self.start_time
                runner.print_comment(f"Time taken to solve: {elapsed} seconds")
        else:
            runner.print_comment(f"Time taken to solve: {runner.s.status().runtime} seconds")

    def observe_end(self, runner: Runner):
        runner.print_comment(f"Total time taken: {time.time() - self.start_time} seconds")
