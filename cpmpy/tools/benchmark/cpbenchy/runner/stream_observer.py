from abc import ABC, abstractmethod
import sys
from typing import Optional, Callable


class StreamObserver(ABC):
    """Parent-process observer interface for relayed child hook events."""

    @abstractmethod
    def on_event(self, event: dict) -> None:
        pass


class HookStreamDispatcher:
    """Central dispatcher for relayed hook stream events."""

    def __init__(self):
        self._observers: list[StreamObserver] = []

    def register_observer(self, observer: StreamObserver) -> None:
        self._observers.append(observer)

    def dispatch_event(self, event: dict) -> None:
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception:
                # Stream observers are best-effort and should not break run execution.
                pass


class RunStateStreamObserver(StreamObserver):
    """
    Maintains aggregated run state from relayed child hook events.
    Does not depend on or receive a runner object.
    """

    def __init__(self):
        self.is_sat = None
        self.exit_status = None
        self.objective_value = None
        self.intermediate_objectives: list[float] = []
        self.intermediate_solutions: list[dict] = []
        self.stage_timings: dict = {}
        self.solution_checker: dict = {}

    def on_event(self, event: dict) -> None:
        event_type = event.get("event")
        if event_type == "observe_intermediate":
            objective = event.get("objective")
            if isinstance(objective, (int, float)):
                objective = float(objective)
                self.objective_value = objective
                self.intermediate_objectives.append(objective)
                self.intermediate_solutions.append({
                    "objective": objective,
                    "elapsed": event.get("elapsed") if isinstance(event.get("elapsed"), (int, float)) else None,
                    "solve_elapsed": event.get("solve_elapsed") if isinstance(event.get("solve_elapsed"), (int, float)) else None,
                    "ts": event.get("ts") if isinstance(event.get("ts"), (int, float)) else None,
                })
        elif event_type == "observe_post_solve":
            if event.get("is_sat") is not None:
                self.is_sat = event.get("is_sat")
            if event.get("exit_status") is not None:
                self.exit_status = event.get("exit_status")
            if event.get("objective") is not None:
                self.objective_value = event.get("objective")
            timings = event.get("stage_timings")
            if isinstance(timings, dict):
                self.stage_timings.update(timings)
        elif event_type == "observe_end":
            checker = event.get("solution_checker")
            if isinstance(checker, dict):
                self.solution_checker = checker


class RawStreamObserver(StreamObserver):
    """Forwards relayed print_raw events to parent sinks."""

    def __init__(
        self,
        raw_sink: Optional[Callable[[str], None]] = None,
        echo_master_stdout: bool = False,
    ):
        self.raw_sink = raw_sink
        self.echo_master_stdout = echo_master_stdout

    def on_event(self, event: dict) -> None:
        if event.get("event") != "print_raw":
            return
        text = event.get("text")
        if not isinstance(text, str):
            return

        if callable(self.raw_sink):
            self.raw_sink(text)

        if self.echo_master_stdout:
            # Bypass temporary stdout redirections in parent forwarding contexts.
            sys.__stdout__.write(text + "\r\n")
            sys.__stdout__.flush()
