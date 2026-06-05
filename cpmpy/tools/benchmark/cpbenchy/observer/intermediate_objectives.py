import time

from ..runner.runner import Runner
from .base import Observer


class IntermediateObjectivesObserver(Observer):
    """Collect intermediate objective values on the runner."""

    def observe_init(self, runner: Runner):
        runner.intermediate_objectives = []
        runner.intermediate_solutions = []
        self._run_started_at = time.time()
        self._solve_started_at = None

    def observe_pre_solve(self, runner: Runner):
        self._solve_started_at = time.time()

    def observe_intermediate(self, runner: Runner, objective: int):
        if not hasattr(runner, "intermediate_objectives") or runner.intermediate_objectives is None:
            runner.intermediate_objectives = []
        if not hasattr(runner, "intermediate_solutions") or runner.intermediate_solutions is None:
            runner.intermediate_solutions = []
        now_ts = time.time()
        elapsed = (now_ts - self._run_started_at) if hasattr(self, "_run_started_at") and self._run_started_at is not None else None
        solve_elapsed = (now_ts - self._solve_started_at) if hasattr(self, "_solve_started_at") and self._solve_started_at is not None else None
        runner.intermediate_objectives.append(objective)
        runner.intermediate_solutions.append({
            "objective": objective,
            "elapsed": elapsed,
            "solve_elapsed": solve_elapsed,
            "ts": now_ts,
        })

    def observe_post_solve(self, runner: Runner):
        if hasattr(runner, "runner_metadata") and isinstance(runner.runner_metadata, dict):
            if hasattr(runner, "intermediate_objectives"):
                runner.runner_metadata["intermediate_objectives"] = list(runner.intermediate_objectives)
            if hasattr(runner, "intermediate_solutions"):
                runner.runner_metadata["intermediate_solutions"] = list(runner.intermediate_solutions)