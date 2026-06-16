"""DIMACS-style competition output printer (s, v, o, c lines). Reusable across formats via solution_printer."""
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus

from ..runner.runner import Runner
from .base import Observer


class DIMACSPrintingObserver(Observer):
    """
    DIMACS-style competition output printer (s, v, o, c lines).
    Pass solution_printer in the constructor to format solutions for your competition format.
    """

    def __init__(
        self,
        solution_printer,
        verbose: bool = False,
        **kwargs,
    ):
        self.solution_printer = solution_printer
        self.verbose = verbose

    def print_comment(self, comment: str, runner: Runner = None):
        adapter = getattr(runner, "instance_runner", None) if runner else None
        adapter_printer = getattr(adapter, "print_comment", None) if adapter else None
        formatted = "c" + chr(32) + comment.rstrip("\n")
        if callable(adapter_printer) and getattr(runner, "verbose", False):
            adapter_printer(comment)
        else:
            runner.print_raw(formatted)

    def observe_post_solve(self, runner: Runner):
        self.print_result(runner.s, runner)

    def observe_intermediate(self, runner: Runner, objective: int):
        self.print_intermediate(objective, runner)

    def print_status(self, status: str, runner: Runner):
        runner.print_raw("s" + chr(32) + status)

    def print_value(self, value: str, runner: Runner):
        prefix = "v" + chr(32)
        runner.print_raw(prefix + str(value).rstrip("\r\n").replace("\n", "\n" + prefix))

    def print_objective(self, objective: int, runner: Runner):
        runner.print_raw("o" + chr(32) + str(objective))

    def print_intermediate(self, objective: int, runner: Runner):
        self.print_objective(objective, runner)

    def _final_objective(self, s, runner: Runner) -> int:
        model = getattr(runner, "model", None) if runner else None
        if model is not None and model.has_objective():
            try:
                return int(model.objective_.value())
            except Exception:
                pass
        return s.objective_value()

    def _has_objective(self, s, runner: Runner = None) -> bool:
        model = getattr(runner, "model", None) if runner else None
        if model is not None:
            return model.has_objective()
        return s.has_objective()

    def print_result(self, s, runner: Runner = None):
        if s is None:
            return
        # Pass include_aux_vars from adapter when present (for solution checkers that need IV/BV)
        adapter = getattr(runner, "instance_runner", None) if runner else None
        include_aux = getattr(adapter, "include_aux_vars", False) if adapter else False
        value = self.solution_printer(s, include_aux_vars=include_aux) if include_aux else self.solution_printer(s)
        if s.status().exitstatus == CPMStatus.OPTIMAL:
            if self._has_objective(s, runner):
                self.print_objective(self._final_objective(s, runner), runner)
            self.print_value(value, runner)
            self.print_status("OPTIMUM" + chr(32) + "FOUND", runner)
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            if self._has_objective(s, runner):
                self.print_objective(self._final_objective(s, runner), runner)
            self.print_value(value, runner)
            self.print_status("SATISFIABLE", runner)
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            self.print_status("UNSATISFIABLE", runner)
        else:
            self.print_comment("Solver did not find any solution within the time/memory limit", runner)
            self.print_status("UNKNOWN", runner)
