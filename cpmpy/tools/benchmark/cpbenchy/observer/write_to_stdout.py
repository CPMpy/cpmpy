from .base import Observer
import sys


def write_raw_to_stdout(text: str) -> None:
    """Write directly to the process stdout, bypassing print_forwarding_context."""
    sys.__stdout__.write(text + "\r\n")
    sys.__stdout__.flush()


class WriteToStdoutObserver(Observer):
    """Observer sink for raw output channel to stdout."""

    bypass_print_forwarding = True

    def print_raw(self, text: str):
        write_raw_to_stdout(text)