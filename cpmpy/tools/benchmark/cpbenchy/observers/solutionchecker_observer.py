import sys

from cpmpy.tools.solution_checker import check_solution
from cpmpy.transformations.get_variables import get_variables_model

from ..runner import Runner
from .base import Observer


class SolutionCheckerObserver(Observer):

    def observe_end(self, runner: Runner):
        

        runner.check_result = None
        if not getattr(runner, "is_sat", False):
            return

        var_map = {
            v.name: v.value()
            for v in get_variables_model(runner.model)
            if v.value() is not None
        }
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 50_000))
        try:
            result = check_solution(runner.model, runner.s.status().exitstatus, var_map)
        finally:
            sys.setrecursionlimit(old_limit)
        runner.check_result = result

        if result.skipped:
            runner.print_comment("checker: skipped")
        elif result.valid:
            obj_str = f", objective={result.objective_value}" if result.objective_value is not None else ""
            runner.print_comment(f"checker: VALID{obj_str}")
        else:
            runner.print_comment("checker: INVALID")
            for v in result.violations[:5]:
                runner.print_comment(f"  violation: {v}")