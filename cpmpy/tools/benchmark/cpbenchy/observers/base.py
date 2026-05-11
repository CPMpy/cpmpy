from abc import ABC

from ..runner import Runner


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

    def observe_intermediate(self, runner: Runner = None, objective: int = None, elapsed_seconds=None, **kwargs):
        pass

    def get_context_manager(self, runner: Runner):
        """
        Return a context manager that will be entered when the ObserverContext is entered.
        Return None if this observer doesn't provide a context manager.
        """
        return None