import json
import os
import time
import uuid

from ..runner.runner import Runner
from .base import Observer


class HookPipeRelayObserver(Observer):
    """
    Prototype JSONL IPC relay observer for subprocess-safe hook propagation.
    """

    def __init__(self, ipc_events_file: str, **kwargs):
        self.ipc_events_file = ipc_events_file
        self._start_time = None
        self._solve_start_time = None
        self._ipc_handle = None
        self._run_id = uuid.uuid4().hex
        self._seq = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.ipc_events_file)), exist_ok=True)

    def _emit(self, event_type: str, **payload):
        self._seq += 1
        event = {
            "event": event_type,
            "run_id": self._run_id,
            "seq": self._seq,
            "ts": time.time(),
            **payload,
        }
        try:
            if self._ipc_handle is not None:
                self._ipc_handle.write(json.dumps(event, ensure_ascii=True) + "\n")
                self._ipc_handle.flush()
            else:
                with open(self.ipc_events_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception:
            # IPC relay is best-effort and must not interfere with solving.
            pass

    def observe_init(self, runner: Runner):
        self._start_time = time.time()
        try:
            self._ipc_handle = open(self.ipc_events_file, "w", encoding="utf-8", buffering=1)
        except Exception:
            self._ipc_handle = None
        self._emit("observe_init")

    def observe_pre_transform(self, runner: Runner):
        self._emit("observe_pre_transform")

    def observe_post_transform(self, runner: Runner):
        self._emit("observe_post_transform")

    def observe_pre_solve(self, runner: Runner):
        self._solve_start_time = time.time()
        self._emit("observe_pre_solve")

    def observe_intermediate(self, runner: Runner, objective: int):
        now_ts = time.time()
        elapsed = (now_ts - self._start_time) if self._start_time is not None else None
        solve_elapsed = (now_ts - self._solve_start_time) if self._solve_start_time is not None else None
        self._emit(
            "observe_intermediate",
            objective=objective,
            elapsed=elapsed,
            solve_elapsed=solve_elapsed,
        )

    def print_raw(self, text: str):
        self._emit("print_raw", text=text)

    def observe_post_solve(self, runner: Runner):
        status_name = None
        objective = None
        stage_timings = None
        try:
            if runner.s is not None:
                status = runner.s.status().exitstatus
                status_name = status.name if hasattr(status, "name") else str(status)
                if runner.is_sat:
                    objective = runner.s.objective_value()
        except Exception:
            pass
        try:
            md = getattr(runner, "runner_metadata", None)
            if isinstance(md, dict):
                stage_timings = md.get("stage_timings")
        except Exception:
            pass
        self._emit(
            "observe_post_solve",
            is_sat=getattr(runner, "is_sat", None),
            exit_status=status_name,
            objective=objective,
            stage_timings=stage_timings,
        )

    def observe_exception(self, runner: Runner, exc_type, exc_value, traceback):
        self._emit(
            "observe_exception",
            exc_type=getattr(exc_type, "__name__", str(exc_type)),
            exc_value=str(exc_value),
        )

    def observe_end(self, runner: Runner):
        elapsed = (time.time() - self._start_time) if self._start_time is not None else None
        self._emit(
            "observe_end",
            elapsed=elapsed,
            solution_checker=getattr(runner, "solution_checker", None),
        )

    def observe_exit(self, runner: Runner):
        self._emit("observe_exit")
        if self._ipc_handle is not None:
            try:
                self._ipc_handle.close()
            except Exception:
                pass
            self._ipc_handle = None
