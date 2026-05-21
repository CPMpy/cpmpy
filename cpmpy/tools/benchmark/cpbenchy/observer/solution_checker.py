from ..runner.runner import Runner
from .base import Observer


class SolutionCheckerObserver(Observer):
    def __init__(self, max_violations_to_print: int = 5, **kwargs):
        self.max_violations_to_print = max_violations_to_print

    def observe_end(self, runner: Runner):
        # In runexec parent-process replay mode, model/solver objects may be absent.
        if not hasattr(runner, "model") or runner.model is None:
            return
        if not hasattr(runner, "s") or runner.s is None:
            return

        import importlib

        check_solution = importlib.import_module(
            "cpmpy.tools.solution_checker"
        ).check_solution
        get_variables_model = importlib.import_module(
            "cpmpy.transformations.get_variables"
        ).get_variables_model

        try:
            status_obj = runner.s.status()
            if status_obj is None or status_obj.exitstatus is None:
                return
            exit_status = status_obj.exitstatus
        except Exception:
            return

        # Build var_map from current model variable values.
        var_map = {}
        for var in get_variables_model(runner.model):
            val = var.value()
            if val is not None:
                var_map[var.name] = val

        declared_objective = getattr(runner, "objective_value", None)
        result = check_solution(
            runner.model,
            exit_status,
            var_map=var_map,
            expected_objective=declared_objective,
        )

        checker_payload = {
            "valid": result.valid,
            "skipped": result.skipped,
            "summary": result.summary(),
            "objective_value": result.objective_value,
            "warnings": list(result.warnings),
            "violations": [
                {
                    "stage": v.stage,
                    "kind": v.kind,
                    "message": v.message,
                    "context": str(v.context) if v.context is not None else None,
                }
                for v in result.violations
            ],
            "violation_count": len(result.violations),
        }
        setattr(runner, "solution_checker", checker_payload)
        if hasattr(runner, "runner_metadata") and isinstance(runner.runner_metadata, dict):
            runner.runner_metadata["solution_checker"] = checker_payload

        runner.print_comment(f"Solution checker: {result.summary()}")

        for warning in result.warnings:
            runner.print_comment(f"Solution checker warning: {warning}")

        if not result.valid and not result.skipped:
            for violation in result.violations[: self.max_violations_to_print]:
                runner.print_comment(f"Solution checker violation: {violation}")
            remaining = len(result.violations) - self.max_violations_to_print
            if remaining > 0:
                runner.print_comment(
                    f"Solution checker: {remaining} additional violations omitted"
                )
