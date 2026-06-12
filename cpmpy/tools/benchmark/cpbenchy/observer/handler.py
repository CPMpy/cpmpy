import os
import signal
import sys
import warnings

from ..runner.runner import Runner
from .base import Observer


class HandlerObserver(Observer):

    def __init__(self, **kwargs):
        self.runner = None

    def observe_init(self, runner: Runner):
        self.runner = runner
        signal.signal(signal.SIGINT, self._sigterm_handler)
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)
        signal.signal(signal.SIGABRT, self._sigterm_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGXCPU, self._rlimit_cpu_handler)
        else:
            warnings.warn("Windows does not support setting SIGXCPU signal")

    def _sigterm_handler(self, _signo, _stack_frame):
        exit_code = self.handle_sigterm()
        print(flush=True)
        os._exit(exit_code)

    def _rlimit_cpu_handler(self, _signo, _stack_frame):
        # Raise TimeoutError - ObserverContext will handle notifying observers
        # Don't notify here to avoid duplicates
        raise TimeoutError("CPU time limit reached (SIGXCPU)")

    def handle_sigterm(self):
        return 0

    def handle_rlimit_cpu(self):
        return 0
