from .base import Observer


class WriteToStdoutObserver(Observer):
    """Observer sink for raw output channel to stdout."""

    def print_raw(self, text: str):
        print(text, end="\r\n", flush=True)