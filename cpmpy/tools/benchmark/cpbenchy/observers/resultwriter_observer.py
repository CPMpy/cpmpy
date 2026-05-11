from ..runner import Runner
from .base import Observer


class ResultWriterObserver(Observer):
    """Writes a JSON result file per instance alongside the text output file."""

    def __init__(self, output_file: str = None, **kwargs):
        self._json_path = (output_file + ".json") if output_file else None
        self._start_time = None
        self._parse_end_time = None
        self._solve_start_time = None
        self._solve_end_time = None
        self._intermediates = []  # list of {"elapsed_seconds": t, "objective": v}

    def observe_init(self, runner: Runner):
        import time
        self._start_time = time.time()
        self._intermediates = []

    def observe_post_transform(self, runner: Runner):
        import time
        self._parse_end_time = time.time()

    def observe_pre_solve(self, runner: Runner):
        import time
        self._solve_start_time = time.time()

    def observe_post_solve(self, runner: Runner):
        import time
        self._solve_end_time = time.time()

    def observe_intermediate(self, runner: Runner, objective, elapsed_seconds=None):
        self._intermediates.append({"elapsed_seconds": elapsed_seconds, "objective": objective})

    def observe_end(self, runner: Runner):
        import time
        import json
        import pathlib
        if self._json_path is None:
            return
        now = time.time()
        total = (now - self._start_time) if self._start_time else None
        parse_time = (self._parse_end_time - self._start_time) if self._parse_end_time and self._start_time else None
        solve_time = (self._solve_end_time - self._solve_start_time) if self._solve_end_time and self._solve_start_time else None

        s = getattr(runner, "s", None)
        status = s.status() if s else None
        exitstatus = status.exitstatus.name if status else "UNKNOWN"
        obj = None
        try:
            if s and runner.model.has_objective() and getattr(runner, "is_sat", False):
                obj = s.objective_value()
        except Exception:
            pass
        solver_runtime = getattr(status, "runtime", None) if status else None

        check = None
        cr = getattr(runner, "check_result", None)
        if cr is not None:
            check = {
                "status": "skipped" if cr.skipped else ("valid" if cr.valid else "invalid"),
                "valid": None if cr.skipped else cr.valid,
                "violations": [
                    {"stage": v.stage, "kind": v.kind, "message": v.message}
                    for v in cr.violations
                ],
                "objective_value": cr.objective_value,
            }

        result = {
            "instance_path": getattr(runner, "_instance_path", None),
            "solver": runner.solver,
            "exitstatus": exitstatus,
            "objective_value": obj,
            "time_total_seconds": total,
            "time_parse_seconds": parse_time,
            "time_solve_seconds": solve_time,
            "solver_runtime_seconds": solver_runtime,
            "intermediate_solutions": self._intermediates if self._intermediates else None,
            "check": check,
        }
        pathlib.Path(self._json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._json_path, "w") as f:
            json.dump(result, f, indent=2) 