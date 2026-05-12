import contextlib
import io
import os

from ..runner.runner import Runner
from .base import Observer


class WriteToFileObserver(Observer):
    def __init__(self, output_file: str, overwrite: bool = True, **kwargs):
        self.file_path = output_file
        self.file_handle = None
        self.context_active = False
        self.overwrite = overwrite
        self.file_opened = False  # Track if file has been opened in write mode
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)

    def get_context_manager(self, runner: Runner):
        """Return a context manager that redirects stdout to a file."""

        @contextlib.contextmanager
        def redirect_to_file():
            # If overwrite and file hasn't been opened yet, open in write mode
            # Otherwise, append to preserve existing content
            mode = "w" if (self.overwrite and not self.file_opened) else "a"
            with open(self.file_path, mode) as f:
                self.file_handle = f
                self.context_active = True
                self.file_opened = True
                with contextlib.redirect_stdout(f):
                    yield
                self.context_active = False
                self.file_handle = None
        return redirect_to_file()

    def print_comment(self, comment: str, runner: "Runner" = None):
        # When not verbose, DIMACSPrintingObserver emits via runner.print_raw, so we receive
        # and write in print_raw. Writing here would duplicate.
        # When verbose, DIMACSPrintingObserver only prints to stdout (adapter_printer), so we
        # must write here to get comments into the file.
        if self.context_active and self.file_handle is not None:
            return
        if not getattr(runner, "verbose", False):
            return
        formatted_comment = self._format_with_adapter(comment, runner).rstrip("\n\r")
        self.print_raw(formatted_comment)

    def print_raw(self, text: str):
        if self.context_active and self.file_handle is not None:
            # Write directly to active redirected file handle for raw channel output.
            self.file_handle.write(text + "\r\n")
            self.file_handle.flush()
            return
        if self.overwrite and not self.file_opened:
            mode = "w"
            self.file_opened = True
        else:
            mode = "a"
        with open(self.file_path, mode) as f:
            f.write(text + "\r\n")

    def _format_with_adapter(self, comment: str, runner: "Runner" = None) -> str:
        """
        Format comment like the active adapter would print it (e.g. XCSP3 'c ' prefix).
        Falls back to raw comment if adapter formatting is unavailable.
        """
        if runner is None:
            return comment

        adapter = getattr(runner, "instance_runner", None)
        adapter_printer = getattr(adapter, "print_comment", None)
        if not callable(adapter_printer):
            return comment

        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                adapter_printer(comment)
            rendered = buf.getvalue()
            if rendered:
                return rendered.rstrip("\r\n")
        except Exception:
            pass
        return comment

    def observe_init(self, runner: Runner):
        """Store reference to runner so we can access instance_runner."""
        self._runner = runner
