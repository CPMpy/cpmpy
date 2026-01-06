import os
import sys
import argparse
import signal
import importlib
import importlib.util
import contextlib
import warnings
import logging
from pathlib import Path

from cpmpy.tools.benchmark.test.instance_runner import InstanceRunner
from cpmpy.tools.benchmark.test.xcsp3_instance_runner import XCSP3InstanceRunner
from cpmpy.tools.benchmark.test.runner import ResourceLimitObserver


class ResourceManager:
    pass

class RunExecResourceManager:
    
    @contextlib.contextmanager
    def _print_forwarding_context(self, runner: InstanceRunner):
        """Context manager that forwards all print statements, warnings, and logging to runner.print_comment."""
        class PrintForwarder:
            def __init__(self, runner, is_stderr=False):
                self.runner = runner
                self.original_stream = sys.stderr if is_stderr else sys.stdout
                self.is_stderr = is_stderr
                self.buffer = []
                # Track if we're in a logging handler to avoid duplicates
                self._in_logging_handler = False
            
            def _is_from_benchexec(self):
                """Check if the current call stack includes benchexec code."""
                import inspect
                frame = None
                try:
                    # Skip the current frame (write) and the caller frame
                    frame = inspect.currentframe()
                    if frame and frame.f_back:
                        frame = frame.f_back.f_back  # Skip write and its immediate caller
                        while frame:
                            module_name = frame.f_globals.get('__name__', '')
                            if 'benchexec' in module_name:
                                return True
                            frame = frame.f_back
                    return False
                except Exception:
                    # If inspection fails, err on the side of forwarding
                    return False
                finally:
                    # Explicitly delete frame reference to avoid reference cycles
                    if frame is not None:
                        del frame
            
            def write(self, text):
                # Skip forwarding if output is coming from benchexec/RunExecutor
                if self._is_from_benchexec():
                    # Just write to original stream, don't forward
                    self.original_stream.write(text)
                    return
                
                # Skip forwarding if this is stderr and looks like a logging message
                # (logging handler will forward it instead)
                if self.is_stderr and text.strip():
                    # Check if this looks like a logging message (starts with log level)
                    first_line = text.split('\n')[0].strip()
                    if first_line.startswith(('WARNING:', 'ERROR:', 'CRITICAL:', 'INFO:', 'DEBUG:')):
                        # This is a logging message, don't forward (logging handler will handle it)
                        self.original_stream.write(text)
                        return
                
                # Forward immediately line by line for real-time forwarding
                if text:
                    # Split by newlines and forward each complete line
                    lines = text.split('\n')
                    # If text doesn't end with newline, the last part is incomplete
                    if text.endswith('\n'):
                        # All lines are complete
                        for line in lines[:-1]:  # Last element is empty string
                            if line.strip():
                                self.runner.print_comment(line.rstrip())
                    else:
                        # Forward complete lines, buffer incomplete line
                        for line in lines[:-1]:
                            if line.strip():
                                self.runner.print_comment(line.rstrip())
                        # Buffer the incomplete line
                        self.buffer.append(lines[-1])
                # Also write to original stream to preserve normal behavior
                self.original_stream.write(text)
            
            def flush(self):
                self.original_stream.flush()
            
            def forward_to_runner(self):
                # Forward any remaining buffered output
                if self.buffer:
                    full_text = ''.join(self.buffer)
                    if full_text.strip():
                        self.runner.print_comment(full_text.rstrip())
                    self.buffer = []
        
        class LoggingHandler(logging.Handler):
            """Custom logging handler that forwards log messages to runner."""
            def __init__(self, runner):
                super().__init__()
                self.runner = runner
                # Use a simple format similar to default logging format
                self.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
                # Prevent propagation to avoid duplicate messages in stderr
                self.propagate = False
            
            def emit(self, record):
                try:
                    # Format the log message
                    log_msg = self.format(record)
                    # Forward to runner
                    self.runner.print_comment(log_msg)
                except Exception:
                    # Ignore errors in logging handler to avoid recursion
                    pass
        
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            """Custom warning handler that forwards warnings to runner."""
            # Format the warning message
            warning_msg = f"{category.__name__}: {str(message).rstrip()}"
            # Forward to runner
            runner.print_comment(warning_msg)
            # Also call the original warning handler to preserve normal behavior
            original_showwarning(message, category, filename, lineno, file, line)
        
        stdout_forwarder = PrintForwarder(runner, is_stderr=False)
        stderr_forwarder = PrintForwarder(runner, is_stderr=True)
        logging_handler = LoggingHandler(runner)
        logging_handler.setLevel(logging.WARNING)  # Only capture WARNING and above
        
        # Get root logger and benchexec logger
        root_logger = logging.getLogger()
        benchexec_logger = logging.getLogger('benchexec')
        original_root_handlers = root_logger.handlers[:]
        original_root_level = root_logger.level
        original_root_propagate = root_logger.propagate
        original_benchexec_handlers = benchexec_logger.handlers[:]
        original_benchexec_level = benchexec_logger.level
        original_benchexec_propagate = benchexec_logger.propagate
        
        # Find and temporarily remove stderr handlers to prevent duplicates
        # (logging handlers write to stderr, which our stderr forwarder would also capture)
        # Store original stderr reference before redirecting
        original_stderr = sys.stderr
        stderr_handlers_to_remove = []
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == original_stderr:
                stderr_handlers_to_remove.append(handler)
        for handler in stderr_handlers_to_remove:
            root_logger.removeHandler(handler)
        
        benchexec_stderr_handlers_to_remove = []
        for handler in benchexec_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == original_stderr:
                benchexec_stderr_handlers_to_remove.append(handler)
        for handler in benchexec_stderr_handlers_to_remove:
            benchexec_logger.removeHandler(handler)
        
        original_showwarning = warnings.showwarning
        
        try:
            # Redirect stdout and stderr
            sys.stdout = stdout_forwarder
            sys.stderr = stderr_forwarder
            # Redirect warnings
            warnings.showwarning = warning_handler
            # Temporarily disable propagation to prevent duplicate messages
            root_logger.propagate = False
            benchexec_logger.propagate = False
            # Disable lastResort handler (Python 3.2+) to prevent fallback to stderr
            if hasattr(logging, 'lastResort'):
                original_last_resort = logging.lastResort
                logging.lastResort = None
            else:
                original_last_resort = None
            # Add logging handler to both root and benchexec loggers
            root_logger.addHandler(logging_handler)
            root_logger.setLevel(logging.WARNING)  # Ensure we capture warnings
            benchexec_logger.addHandler(logging_handler)
            benchexec_logger.setLevel(logging.WARNING)
            yield
        finally:
            # Restore lastResort handler if we disabled it
            if original_last_resort is not None:
                logging.lastResort = original_last_resort
            # Restore stdout and stderr
            sys.stdout = stdout_forwarder.original_stream
            sys.stderr = stderr_forwarder.original_stream
            # Restore warnings
            warnings.showwarning = original_showwarning
            # Remove our logging handler
            root_logger.removeHandler(logging_handler)
            benchexec_logger.removeHandler(logging_handler)
            # Restore original handlers (including stderr handlers)
            root_logger.handlers = original_root_handlers
            root_logger.setLevel(original_root_level)
            root_logger.propagate = original_root_propagate
            benchexec_logger.handlers = original_benchexec_handlers
            benchexec_logger.setLevel(original_benchexec_level)
            benchexec_logger.propagate = original_benchexec_propagate
            # Forward any remaining buffered output
            stdout_forwarder.forward_to_runner()
            stderr_forwarder.forward_to_runner()
    
    def run(self, instance: str, runner: InstanceRunner, time_limit: float, memory_limit: int, cores: list[int]):

        runner.print_comment(f"Running instance {instance} with time limit {time_limit} and memory limit {memory_limit} and cores {cores}")
        runner.print_comment(f"Running with manager {self.__class__.__name__}")

        from benchexec.runexecutor import RunExecutor

        # Use a temporary file to capture subprocess output, then forward it
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Capture warnings from benchexec itself (current process) and subprocess output
            # Set up forwarding context BEFORE creating executor to catch all warnings
            with self._print_forwarding_context(runner):
                executor = RunExecutor(
                    use_namespaces=False,
                )

                def signal_handler_kill(signum, frame):
                    executor.stop()

                signal.signal(signal.SIGTERM, signal_handler_kill)
                signal.signal(signal.SIGQUIT, signal_handler_kill)
                signal.signal(signal.SIGINT, signal_handler_kill)

                cmd = runner.cmd(instance)
                if time_limit is not None:
                    cmd.append("--time_limit")
                    cmd.append(str(time_limit))

                cmd += [
                    "--seed", "1234567890", 
                    "--intermediate", 
                    #"--cores", str(len(cores))  # Pass number of cores to the solver
                ]

                result = executor.execute_run(
                        args=cmd,
                        output_filename=tmp_filename,  # Capture subprocess output to temp file
                        # stdin=stdin,
                        # hardtimelimit=options.timelimit,
                        # softtimelimit=options.softtimelimit,
                        walltimelimit=time_limit,
                        cores=cores,
                        memlimit=memory_limit,
                        # memory_nodes=options.memoryNodes,
                        # cgroupValues=cgroup_values,
                        # workingDir=options.dir,
                        # maxLogfileSize=options.maxOutputSize,
                        # files_count_limit=options.filesCountLimit,
                        # files_size_limit=options.filesSizeLimit,
                        write_header=False,
                        # **container_output_options,
                    )
            
            # Read the output file and forward subprocess output to runner
            # Filter out RunExecutor-specific messages that get mixed into subprocess output
            def _is_runexec_message(line):
                """Check if a line is a RunExecutor-specific message that should be filtered."""
                line_lower = line.lower().strip()
                # Filter specific RunExecutor warning patterns (very specific to avoid false positives)
                runexec_patterns = [
                    'warning: no variables in this model (and so, no generated file)',
                    'warning: no variables in this model',
                ]
                return any(pattern in line_lower for pattern in runexec_patterns)
            
            try:
                with open(tmp_filename, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line_stripped = line.strip()
                        # Skip empty lines and RunExecutor messages
                        if line_stripped and not _is_runexec_message(line_stripped):
                            # Subprocess output is already formatted by the runner's observers,
                            # so print it directly without wrapping in print_comment to avoid double-prefixing
                            print(line_stripped, flush=True)
            except FileNotFoundError:
                # Output file might not exist if process was killed before writing
                pass
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_filename)
            except Exception:
                pass

        runner.print_comment(f"RunExec result: {result}")

        if "terminationreason" in result:
            reason = result["terminationreason"]
            if reason == "memory":
                runner.print_comment("Memory limit exceeded")
            elif reason == "walltime":
                runner.print_comment("Wall time limit exceeded")

class PythonResourceManager:
    
    def run(self, instance: str, runner: InstanceRunner, time_limit: int, memory_limit: int, cores: list[int]):
        # Programmatically add ResourceLimitObserver if limits are provided
        if time_limit is not None or memory_limit is not None:
            # Add a resource observer with limits
            resource_observer = ResourceLimitObserver(
                time_limit=time_limit if time_limit is not None else None,
                mem_limit=memory_limit if memory_limit is not None else None
            )
            runner.register_observer(resource_observer)
        
        # Run the instance using the runner's run method
        runner.run(instance=instance, time_limit=time_limit, mem_limit=memory_limit, cores=len(cores) if cores else None)
        



def run_instance(instance: str, instance_runner: InstanceRunner, time_limit: int, memory_limit: int, cores: list[int], resource_manager: ResourceManager):


    """
    Run a single instance with assigned cores.
    
    Args:
        instance: Instance file path
        time_limit: Time limit in seconds
        memory_limit: Memory limit in MB
        cores: List of core IDs to assign to this run (e.g., [0, 1] for cores 0 and 1)
    """


    resource_manager.run(instance, instance_runner, time_limit, memory_limit, cores)
    
    
    # Convert cores list to comma-separated string for runexec
    #cores_str = ",".join(map(str, cores))
    
    # cmd_runexec = [
    #     "runexec", 
    #     "--walltimelimit", f"{time_limit}s", 
    #     "--memlimit", f"{memory_limit}MB",
    #     "--no-container",
    #     "--cores", cores_str,
    #     "--"
    # ]

    


def load_instance_runner(runner_path: str) -> InstanceRunner:
    """
    Load an instance runner class from a module path.
    
    Args:
        runner_path: Path to the instance runner class, e.g., 
                     "cpmpy.tools.benchmark.test.xcsp3_instance_runner.XCSP3InstanceRunner"
                     or a file path like "/path/to/module.py:ClassName"
    
    Returns:
        InstanceRunner instance
    """
    if ":" in runner_path:
        # Format: /path/to/module.py:ClassName
        file_path, class_name = runner_path.rsplit(":", 1)
        file_path = Path(file_path).resolve()
        
        # Add parent directory to sys.path if needed
        parent_dir = str(file_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import the module
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the class
        runner_class = getattr(module, class_name)
    elif "." in runner_path:
        # Format: module.path.ClassName
        module_path, class_name = runner_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        runner_class = getattr(module, class_name)
    else:
        # Default to xcsp3 if just a name
        if runner_path == "xcsp3":
            return XCSP3InstanceRunner()
        else:
            raise ValueError(f"Invalid runner path format: {runner_path}. Use 'module.path.ClassName' or '/path/to/file.py:ClassName'")
    
    if not issubclass(runner_class, InstanceRunner):
        raise ValueError(f"{runner_class} is not a subclass of InstanceRunner")
    
    return runner_class()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--time_limit", type=float, required=False, default=None)
    parser.add_argument("--memory_limit", type=int, required=False, default=None)
    parser.add_argument("--cores", type=list[int], required=False, default=None)
    parser.add_argument("--runner", type=str, required=False, default="xcsp3",
                        help="Path to instance runner class. Can be:\n"
                             "- 'xcsp3' (default)\n"
                             "- Module path: 'cpmpy.tools.benchmark.test.xcsp3_instance_runner.XCSP3InstanceRunner'\n"
                             "- File path: '/path/to/module.py:ClassName'")
    parser.add_argument("--resource_manager", type=str, required=False, default="runexec")
    args = parser.parse_args()

    if args.resource_manager == "runexec":
        resource_manager = RunExecResourceManager()
    elif args.resource_manager == "python":
        resource_manager = PythonResourceManager()
    else:
        raise ValueError(f"Invalid resource manager: {args.resource_manager}")

    # Load the instance runner
    if args.runner == "xcsp3":
        instance_runner = XCSP3InstanceRunner()
    else:
        instance_runner = load_instance_runner(args.runner)

    resource_manager.run(args.instance, instance_runner, args.time_limit, args.memory_limit, args.cores)

if __name__ == "__main__":
    main()