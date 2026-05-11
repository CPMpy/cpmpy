import contextlib

from ..runner import Runner
from .base import Observer


class WriteToFileObserver(Observer):
    def __init__(self, output_file: str, overwrite: bool = True, **kwargs):
        self.file_path = output_file
        self.file_handle = None
        self.context_active = False
        self.overwrite = overwrite
        self.file_opened = False  # Track if file has been opened in write mode

    def get_context_manager(self, runner: Runner):
        """Return a context manager that redirects stdout to a file."""
        @contextlib.contextmanager
        def redirect_to_file():
            # If overwrite and file hasn't been opened yet, open in write mode
            # Otherwise, append to preserve existing content
            mode = 'w' if (self.overwrite and not self.file_opened) else 'a'
            with open(self.file_path, mode) as f:
                self.file_handle = f
                self.context_active = True
                self.file_opened = True
                with contextlib.redirect_stdout(f):
                    yield
                self.context_active = False
                self.file_handle = None
        return redirect_to_file()
    
    def print_comment(self, comment: str, runner: 'Runner' = None):
        """Write comments to the file using the print_comment hook (in addition to stdout)."""
        # Comment is already formatted by Runner.print_comment() before being passed to observers
        formatted_comment = comment.rstrip('\n\r')

        if self.context_active and self.file_handle is not None:
            # Context is active: stdout is already redirected to this file by get_context_manager(),
            # so any print() calls from other observers already land in the file.
            # Writing here again would duplicate the content — skip.
            return
        else:
            # Context not active yet or has exited
            # If overwrite and file hasn't been opened, open in write mode to truncate
            # Otherwise, append to preserve existing content
            if self.overwrite and not self.file_opened:
                mode = 'w'
                self.file_opened = True
            else:
                mode = 'a'
            with open(self.file_path, mode) as f:
                f.write(formatted_comment + '\r\n')
    
    def observe_init(self, runner: Runner):
        """Store reference to runner so we can access instance_runner."""
        self._runner = runner