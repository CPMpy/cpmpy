import json
import os

from ..runner.runner import Runner
from .base import Observer


class MetadataSidecarObserver(Observer):
    """
    Write a metadata sidecar file next to the run output file.
    """

    def __init__(self, output_file: str, overwrite: bool = True, **kwargs):
        self.output_file = output_file
        self.overwrite = overwrite
        self.sidecar_file = f"{output_file}.metadata.json"
        os.makedirs(os.path.dirname(os.path.abspath(self.sidecar_file)), exist_ok=True)

    def _to_json_safe(self, value):
        from datetime import datetime
        from decimal import Decimal

        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {k: self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _build_payload(self, runner: Runner):
        runner_metadata = getattr(runner, "runner_metadata", None)
        exit_status = None
        if isinstance(runner_metadata, dict):
            exit_status = runner_metadata.get("exit_status") or (
                runner_metadata.get("exit") or {}
            ).get("status_name")
        payload = {
            "instance": getattr(runner, "instance_path", None) or getattr(runner, "_instance_path", None),
            "solver": getattr(runner, "solver", None),
            "is_sat": getattr(runner, "is_sat", None),
            "objective_value": getattr(runner, "objective_value", None),
            "intermediate_objectives": getattr(runner, "intermediate_objectives", None),
            "intermediate_solutions": getattr(runner, "intermediate_solutions", None),
            "termination_reason": getattr(runner, "termination_reason", None),
            "exit_status": exit_status,
            "solution_checker": getattr(runner, "solution_checker", None),
            "runner_metadata": runner_metadata,
        }
        return self._to_json_safe(payload)

    def observe_end(self, runner: Runner):
        payload = self._build_payload(runner)
        mode = "w" if self.overwrite else "a"
        with open(self.sidecar_file, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n")
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
