from abc import ABC, abstractmethod
import os
import sys
import argparse
import select
import importlib
import importlib.util
import contextlib
import warnings
import logging
import secrets
import subprocess
import json
import queue
import multiprocessing as mp
import traceback
import time
from pathlib import Path
from typing import Optional, List

from cpmpy.tools.benchmark import _mib_as_bytes
from ..adapter._base import InstanceAdapter
from ..adapter.xcsp3 import XCSP3Adapter
from ..adapter.opb import OPBAdapter
from ..adapter.nurserostering import NurseRosteringAdapter
from ..adapter.jsplib import JSPLibAdapter
from ..adapter.psplib import PSPLibAdapter
from ..adapter.mse import MSEAdapter
from ..observer import ResourceLimitObserver
from ..runner.stream_observer import StreamObserver, HookStreamDispatcher, RunStateStreamObserver, RawStreamObserver

FORCE_EXPLICIT_SYSTEMD_SCOPE = os.environ.get(
    "CPLAB_FORCE_SYSTEMD_SCOPE",
    "false",
).lower() in ("1", "true", "yes", "on")

ALLOW_NAMESPACE_FALLBACK = os.environ.get(
    "CPLAB_ALLOW_NAMESPACE_FALLBACK",
    "true",
).lower() in ("1", "true", "yes", "on")


def _to_json_safe(value):
    from datetime import datetime
    from decimal import Decimal

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    # benchexec ProcessExitCode-like object
    if hasattr(value, "raw") and hasattr(value, "value") and hasattr(value, "signal"):
        return {
            "raw": getattr(value, "raw", None),
            "code": getattr(value, "value", None),
            "signal": getattr(value, "signal", None),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_exit_info(exit_info):
    """
    Normalize BenchExec exit metadata to dict format.
    BenchExec can provide `exitcode` either as dict-like object or list
    `[raw, code, signal]`.
    """
    if isinstance(exit_info, dict):
        # CLI fallback parses exitcode as {"value": N} — map to "code"
        code = exit_info.get("code")
        if code is None and "value" in exit_info:
            code = exit_info["value"]
        return {
            "raw": exit_info.get("raw"),
            "code": code,
            "signal": exit_info.get("signal"),
        }
    if isinstance(exit_info, (list, tuple)):
        raw = exit_info[0] if len(exit_info) > 0 else None
        code = exit_info[1] if len(exit_info) > 1 else None
        signal = exit_info[2] if len(exit_info) > 2 else None
        return {"raw": raw, "code": code, "signal": signal}
    return None


def _tail_text_file(path: str, max_lines: int = 80) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines:
            return ""
        return "".join(lines[-max_lines:]).strip()
    except Exception:
        return ""


def _parse_runexec_output(stdout: str) -> dict:
    """
    Parse the key=value pairs that ``runexec`` prints to stdout.

    Returns a dict whose values mirror what ``RunExecutor.execute_run()``
    returns (floats for times/memory, nested dict for exitcode, etc.).
    """
    result = {}
    for line in stdout.strip().splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()

        # Time values end with "s"
        if key in ("walltime", "cputime") or key.startswith("cputime-"):
            try:
                result[key] = float(raw_value.rstrip("s"))
            except ValueError:
                result[key] = raw_value
        # Memory / IO values end with "B"
        elif key in ("memory", "blkio-read", "blkio-write"):
            try:
                result[key] = int(raw_value.rstrip("B"))
            except ValueError:
                result[key] = raw_value
        elif key == "returnvalue":
            try:
                result.setdefault("exitcode", {})["value"] = int(raw_value)
            except ValueError:
                result.setdefault("exitcode", {})["value"] = raw_value
        elif key == "exitsignal":
            try:
                result.setdefault("exitcode", {})["signal"] = int(raw_value)
            except ValueError:
                result.setdefault("exitcode", {})["signal"] = raw_value
        elif key == "terminationreason":
            result[key] = raw_value
        elif key == "starttime":
            result[key] = raw_value
        else:
            # Pressure values end with "s", energy with "J", etc.
            try:
                result[key] = float(raw_value.rstrip("sJB"))
            except ValueError:
                result[key] = raw_value
    return result


def _runexec_worker(cmd, tmp_filename, time_limit, cores, memory_limit, pin_cores, result_queue):
    """
    Execute *cmd* via BenchExec.

    Strategy
    --------
    1. Try the in-process ``RunExecutor`` Python API (fast, no extra process).
    2. If that fails with the well-known cgroup ``SystemExit``, fall back to
       invoking the ``runexec`` **CLI** wrapped inside
       ``systemd-run --user --scope --slice=benchexec -p Delegate=yes``
       — which is exactly the invocation that works from your shell.

    cores : list of core IDs for pinning (ignored if pin_cores=False)
    pin_cores : if True, pass cores to RunExecutor / runexec for CPU pinning
    """
    def _emit(payload: dict):
        result_queue.put(payload)
        close_fn = getattr(result_queue, "close", None)
        join_fn = getattr(result_queue, "join_thread", None)
        if callable(close_fn):
            close_fn()
        if callable(join_fn):
            join_fn()

    def _is_cgroup_failure(exc: BaseException) -> bool:
        if not isinstance(exc, SystemExit):
            return False
        msg = str(getattr(exc, "code", exc) or "")
        return "BenchExec was not able to use cgroups" in msg

    # ------------------------------------------------------------------
    # Strategy 1: in-process RunExecutor (preferred)
    # ------------------------------------------------------------------
    def _try_python_api():
        scope_ok = _ensure_systemd_scope()
        if not scope_ok:
            logging.warning(
                "Could not create delegated systemd scope in runexec worker; "
                "RunExecutor may fail to initialize cgroups."
            )
        from benchexec.runexecutor import RunExecutor
        executor = RunExecutor(use_namespaces=False)
        runexec_cores = cores if pin_cores else None
        result_local = executor.execute_run(
            args=cmd,
            output_filename=tmp_filename,
            walltimelimit=time_limit,
            cores=runexec_cores,
            memlimit=_mib_as_bytes(memory_limit),
            write_header=False,
        )
        payload = _to_json_safe(dict(result_local))
        if isinstance(payload, dict):
            payload["_runexec_mode"] = "python_api"
        return payload

    # ------------------------------------------------------------------
    # Strategy 2: shell out to ``systemd-run … runexec …``
    # ------------------------------------------------------------------
    def _try_cli_fallback():
        logging.warning(
            "Falling back to runexec CLI wrapped in systemd-run."
        )
        runexec_cmd = ["runexec"]
        if time_limit is not None:
            runexec_cmd += ["--walltimelimit", str(int(time_limit))]
        if memory_limit is not None:
            runexec_cmd += ["--memlimit", str(int(_mib_as_bytes(memory_limit)))]
        if pin_cores and cores:
            runexec_cmd += ["--cores", ",".join(str(c) for c in cores)]
        runexec_cmd += ["--no-container"]
        runexec_cmd += ["--output", tmp_filename]
        # Forward environment variables through runexec using env command
        # Use /tmp as HOME to avoid NFS flock issues (e.g. pysat's portalocker)
        env_forwards = {"HOME": "/tmp"}
        for env_key in ("GRB_LICENSE_FILE", "HX_LICENSE_PATH", "HX_LICENSE_CONTENT"):
            val = os.environ.get(env_key)
            if val:
                env_forwards[env_key] = val
        runexec_cmd += ["--"]
        if env_forwards:
            runexec_cmd += ["env"] + [f"{k}={v}" for k, v in env_forwards.items()]
        runexec_cmd += cmd

        # Wrap with systemd-run for cgroup delegation
        scope_prefix = ["systemd-run"]
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            scope_prefix.append("--user")
        scope_prefix += ["--scope", "--slice=benchexec", "-p", "Delegate=yes"]
        # Forward key environment variables (systemd-run may not inherit them)
        for env_key in ("GRB_LICENSE_FILE", "HX_LICENSE_PATH", "HX_LICENSE_CONTENT", "PATH", "HOME", "VIRTUAL_ENV"):
            val = os.environ.get(env_key)
            if val:
                scope_prefix += [f"--setenv={env_key}={val}"]
        full_cmd = scope_prefix + runexec_cmd

        logging.info("runexec CLI fallback command: %s", full_cmd)
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=int((time_limit or 3600) + 120),
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError(
                f"runexec CLI failed (rc={proc.returncode}):\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        payload = _parse_runexec_output(proc.stdout)
        payload = _to_json_safe(payload)
        if isinstance(payload, dict):
            payload["_runexec_mode"] = "cli_fallback"
        return payload

    # ------------------------------------------------------------------
    try:
        try:
            result_payload = _try_python_api()
        except SystemExit as e:
            if not (ALLOW_NAMESPACE_FALLBACK and _is_cgroup_failure(e)):
                raise
            result_payload = _try_cli_fallback()
        _emit({"ok": True, "result": result_payload})
    except BaseException as e:
        _emit({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def _ensure_explicit_systemd_scope(cmd: list[str]) -> list[str]:
    """
    Wrap command in an explicit fresh systemd scope with cgroup delegation.
    This is the most reliable mode for BenchExec on Ubuntu 24.04+.
    """
    if not cmd:
        return cmd
    first = os.path.basename(str(cmd[0]))
    if first == "systemd-run":
        return cmd
    scope_prefix = ["systemd-run"]
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        scope_prefix.append("--user")
    scope_prefix.extend(["--scope", "--slice=benchexec", "-p", "Delegate=yes"])
    return scope_prefix + cmd


def _ensure_systemd_scope():
    """
    Ensure the current process is in its own systemd scope with cgroup delegation.
    
    This is required for BenchExec's RunExecutor to work properly with cgroups v2.
    When running under a parent systemd scope (e.g., via systemd-run), child processes
    need their own scope to enable cgroup subtree delegation.
    
    Uses busctl to call systemd's D-Bus API directly, avoiding the need for pystemd.
    
    Returns True if successful or already in a suitable scope, False otherwise.
    """
    # Check if we're already in our own benchexec scope (to avoid re-creating)
    try:
        with open("/proc/self/cgroup", "r") as f:
            cgroup_info = f.read()
            if "benchexec_worker_" in cgroup_info:
                logging.debug("Already in a benchexec worker scope")
                return True
    except Exception:
        pass
    
    # Create a new transient scope for this process.
    random_suffix = secrets.token_urlsafe(8)
    scope_name = f"benchexec_worker_{random_suffix}.scope"

    def _try_start_scope(use_user_bus: bool, slice_name: str) -> tuple[bool, str]:
        cmd = ["busctl"]
        if use_user_bus:
            cmd.append("--user")
        cmd.extend([
            "call",
            "org.freedesktop.systemd1",
            "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager",
            "StartTransientUnit",
            "ssa(sv)a(sa(sv))",
            scope_name,
            "fail",
            "3",
            "PIDs", "au", "1", str(os.getpid()),
            "Delegate", "b", "true",
            "Slice", "s", slice_name,
            "0",
        ])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        err = (result.stderr or result.stdout or "").strip()
        return False, err

    errors = []
    try:
        ok, err = _try_start_scope(use_user_bus=True, slice_name="benchexec.slice")
        if ok:
            logging.debug("Created systemd scope via user bus: %s", scope_name)
            time.sleep(0.1)
            return True
        if err:
            errors.append(f"user-bus: {err}")
    except FileNotFoundError:
        logging.warning("busctl not found, cannot create systemd scope")
        return False
    except subprocess.TimeoutExpired:
        errors.append("user-bus: timeout creating systemd scope")
    except Exception as e:
        errors.append(f"user-bus: {e}")

    # Fallback: try system bus (works for privileged/system services).
    try:
        ok, err = _try_start_scope(use_user_bus=False, slice_name="system.slice")
        if ok:
            logging.debug("Created systemd scope via system bus: %s", scope_name)
            time.sleep(0.1)
            return True
        if err:
            errors.append(f"system-bus: {err}")
    except subprocess.TimeoutExpired:
        errors.append("system-bus: timeout creating systemd scope")
    except Exception as e:
        errors.append(f"system-bus: {e}")

    if errors:
        logging.warning("Failed to create systemd scope (%s)", " | ".join(errors))
    return False


class ResourceManager(ABC):
    """
    Abstract base class for resource managers.

    Manages the allocation of resources (time, memory, cores) to a single instance run.
    Sets limits on the resources and handles callbacks when these limits are exceeded.
    """

    def _append_manager_metadata(self, instance_runner: InstanceAdapter):
        base_runner = getattr(instance_runner, "runner", None)
        if base_runner is None:
            raise ValueError("Runner has no runner attribute")
        metadata = getattr(base_runner, "runner_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            base_runner.runner_metadata = metadata
        metadata["resource_manager"] = self.__class__.__name__

    def run(
            self,
            instance: str,
            runner: InstanceAdapter,
            time_limit: float,
            memory_limit: int,
            cores: list[int],
            solver: str,
            seed: int,
            intermediate: bool,
            verbose: bool,
            output_file: str,
            setup_command: Optional[List[str]] = None,
            solver_params: Optional[dict] = None,
        ) -> Optional[dict]:

        result = self._run(
            instance=instance,
            runner=runner,
            time_limit=time_limit,
            memory_limit=memory_limit,
            cores=cores,
            solver=solver,
            seed=seed,
            intermediate=intermediate,
            verbose=verbose,
            output_file=output_file,
            setup_command=setup_command,
            solver_params=solver_params or {},
        )
        # Shared post-processing: resource-manager metadata enrichment.
        self._append_manager_metadata(runner)
        return result

    @abstractmethod
    def _run(
            self,
            instance: str,
            runner: InstanceAdapter,
            time_limit: float,
            memory_limit: int,
            cores: list[int],
            solver: str,
            seed: int,
            intermediate: bool,
            verbose: bool,
            output_file: str,
            setup_command: Optional[List[str]] = None,
            solver_params: Optional[dict] = None,
        ) -> Optional[dict]:
        pass


class LocalResourceManager(ResourceManager):
    """
    Base for managers that run locally in-process and do not require stream relays.
    """


class StreamingResourceManager(ResourceManager):
    """
    Base for managers that execute out-of-process and consume a stream of hook events.
    """

    def __init__(self):
        self._stream_observers: list[StreamObserver] = []

    def register_stream_observer(self, observer: StreamObserver):
        self._stream_observers.append(observer)

    def get_stream_observers(self) -> list[StreamObserver]:
        return self._stream_observers

    def _prepare_stream_channel(
            self,
            dir_prefix: str = "cpbenchy-ipc-",
            fifo_name: str = "events.fifo",
        ) -> dict:
        """
        Create a reusable stream channel descriptor (FIFO + workspace dir).
        """
        import tempfile

        stream_dir = tempfile.mkdtemp(prefix=dir_prefix)
        fifo_path = os.path.join(stream_dir, fifo_name)
        os.mkfifo(fifo_path)
        return {"dir": stream_dir, "fifo": fifo_path}

    def _cleanup_stream_channel(self, channel: Optional[dict]) -> None:
        if not channel:
            return
        fifo_path = channel.get("fifo")
        stream_dir = channel.get("dir")
        if fifo_path:
            try:
                os.unlink(fifo_path)
            except Exception:
                pass
        if stream_dir:
            try:
                os.rmdir(stream_dir)
            except Exception:
                pass

    def _dispatch_stream_buffer(self, ipc_buffer: str, stream_dispatcher: HookStreamDispatcher) -> str:
        while "\n" in ipc_buffer:
            line, ipc_buffer = ipc_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                stream_dispatcher.dispatch_event(json.loads(line))
            except Exception:
                continue
        return ipc_buffer

    def _run_worker_with_stream(
            self,
            fifo_path: str,
            stream_dispatcher: HookStreamDispatcher,
            worker_target,
            worker_args: tuple,
            poll_timeout: float = 0.2,
            queue_timeout: float = 2.0,
        ) -> dict:
        """
        Run worker process and consume live FIFO events until completion.
        Returns the worker's JSON payload dict.
        """
        ipc_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        ipc_guard_fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        ipc_buffer = ""
        result_queue = mp.Queue()
        worker = mp.Process(target=worker_target, args=(*worker_args, result_queue), daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                ready, _, _ = select.select([ipc_fd], [], [], poll_timeout)
                if ipc_fd in ready:
                    chunk = os.read(ipc_fd, 8192)
                    if chunk:
                        ipc_buffer += chunk.decode("utf-8", errors="replace")
                        ipc_buffer = self._dispatch_stream_buffer(ipc_buffer, stream_dispatcher)

            worker.join()
            while True:
                ready, _, _ = select.select([ipc_fd], [], [], 0)
                if ipc_fd not in ready:
                    break
                chunk = os.read(ipc_fd, 8192)
                if not chunk:
                    break
                ipc_buffer += chunk.decode("utf-8", errors="replace")
            self._dispatch_stream_buffer(ipc_buffer, stream_dispatcher)
        finally:
            os.close(ipc_guard_fd)
            os.close(ipc_fd)

        try:
            # Queue feeder threads can lag slightly after process exit.
            # Poll briefly before failing with a missing-payload error.
            wait_budget = max(float(queue_timeout), 10.0)
            deadline = time.monotonic() + wait_budget
            worker_payload = None
            while time.monotonic() < deadline:
                try:
                    worker_payload = result_queue.get_nowait()
                    break
                except queue.Empty:
                    time.sleep(0.05)
            if worker_payload is None:
                signal_info = ""
                if isinstance(worker.exitcode, int) and worker.exitcode < 0:
                    signal_info = f", signal={-worker.exitcode}"
                raise RuntimeError(
                    "RunExec worker did not return a result payload "
                    f"(exitcode={worker.exitcode}{signal_info})"
                )
        except queue.Empty:
            signal_info = ""
            if isinstance(worker.exitcode, int) and worker.exitcode < 0:
                signal_info = f", signal={-worker.exitcode}"
            raise RuntimeError(
                "RunExec worker did not return a result payload "
                f"(exitcode={worker.exitcode}{signal_info})"
            )
        if not worker_payload.get("ok", False):
            tb = worker_payload.get("traceback")
            if isinstance(tb, str) and tb.strip():
                raise RuntimeError(f"{worker_payload.get('error', 'RunExec worker failed')}\n{tb}")
            raise RuntimeError(worker_payload.get("error", "RunExec worker failed"))
        return worker_payload.get("result", {})

    def _build_stream_dispatch(self) -> tuple[HookStreamDispatcher, RunStateStreamObserver]:
        """
        Create dispatcher + default state observer and register all extra stream observers.
        """
        stream_dispatcher = HookStreamDispatcher()
        stream_state = RunStateStreamObserver()
        stream_dispatcher.register_observer(stream_state)
        for stream_observer in self.get_stream_observers():
            stream_dispatcher.register_observer(stream_observer)
        return stream_dispatcher, stream_state


class RunExecResourceManager(StreamingResourceManager):
    """
    Resource manager that uses benchexec's RunExecutor for resource control (build on cgroups and kernel namespaces).
    Requires `benchexec` to be installed.

    pin_cores: if True (default), RunExecutor pins the process to the assigned cores. If False, no CPU pinning;
               only the number of cores is passed to the solver via --cores.
    """

    def __init__(self, pin_cores: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pin_cores = pin_cores

    @contextlib.contextmanager
    def _print_forwarding_context(self, runner: InstanceAdapter):
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
    
    def _run(self,
            instance: str,
            runner: InstanceAdapter,
            time_limit: float,
            memory_limit: int,
            cores: list[int],
            solver: str,
            seed: int,
            intermediate: bool,
            verbose: bool,
            output_file: str,
            setup_command: Optional[List[str]] = None,
            solver_params: Optional[dict] = None,
        ) -> Optional[dict]:
        """
        Run a single instance with assigned resources.

        Arguments:
            instance: Instance file path
            runner: Instance runner
            time_limit: Time limit in seconds
            memory_limit: Memory limit in MB
            cores: List of core IDs to assign to this run (e.g., [0, 1] for cores 0 and 1)

        runexec creates a new process and namespace for the instance run. So the benchmark needs to be run in a
        separate process for runexec to be able to control the resources.
        """

        # Automatically add WriteToFileObserver and MetadataSidecarObserver if output_file is provided
        if output_file is not None:
            from functools import partial
            from ..observer import WriteToFileObserver, MetadataSidecarObserver, IntermediateObjectivesObserver
            runner.register_observer(partial(WriteToFileObserver, output_file=output_file, overwrite=True))
            runner.register_observer(IntermediateObjectivesObserver())
            runner.register_observer(partial(MetadataSidecarObserver, output_file=output_file))

        _runner = runner.get_runner(instance, solver, output_file, overwrite=True)
        # Expose the effective runner instance similarly to PythonResourceManager.run().
        runner.runner = _runner

        # Set up Runner attributes needed by observers before calling hooks
        _runner.instance = instance
        _runner.instance_path = instance
        _runner._instance_path = instance  # Also set _instance_path for observers that check this
        _runner.solver = solver
        _runner.time_limit = time_limit
        _runner.mem_limit = memory_limit
        _runner.seed = seed
        _runner.cores = len(cores) if cores else None
        _runner.intermediate = intermediate
        _runner.verbose = verbose
        _runner.kwargs = {'solver_params': solver_params} if solver_params else {}
        _runner.solver_args = {}  # Initialize solver_args (will be populated by collect_solver_args if needed)
        
        # Use runner's print_comment to go through the callback system (observers)
        _runner.print_comment(f"Running instance {instance} with time limit {time_limit} and memory limit {memory_limit} and cores {cores}")
        _runner.print_comment(f"Running with manager {self.__class__.__name__}")

        # Ensure we're in our own systemd scope for cgroup delegation (required for cgroups v2)
        _ensure_systemd_scope()

        # Use temporary files to capture subprocess output and live IPC events.
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as tmp_file:
            tmp_filename = tmp_file.name
        stream_channel = self._prepare_stream_channel()
        ipc_events_filename = stream_channel["fifo"]
        
        # Call observer hooks in parent process around subprocess execution
        # Use observer_context to ensure proper lifecycle management
        try:
            with _runner.observer_context():
                # Call observe_init before subprocess starts
                try:
                    _runner.observe_init()
                except Exception as e:
                    # Log but don't fail if observe_init fails (e.g., network error in BackendObserver)
                    import logging
                    logging.warning(f"observe_init failed: {e}")
                
                # Optionally read instance and call transform hooks (may be expensive)
                # For now, we skip these to avoid duplicating work done in subprocess
                # If observers need model info, they can parse it from subprocess output
                
                # Track if subprocess completes successfully (initialize before try block)
                subprocess_completed = False
                
                try:
                    # Build parent-side state from live IPC relay events.
                    _runner.status = None  # Default to None (UNKNOWN)
                    _runner.model = None  # Model not available in parent process
                    _runner.s = None  # Solver object not available in parent process
                    _runner.is_sat = None  # Will be set based on relayed status
                    stream_dispatcher, stream_state = self._build_stream_dispatch()
                    stream_dispatcher.register_observer(
                        RawStreamObserver(
                            raw_sink=_runner.print_raw,
                            echo_master_stdout=bool(verbose),
                        )
                    )

                    # Capture warnings from benchexec itself (current process) and subprocess output
                    # Set up forwarding context before executing runexec worker process.
                    with self._print_forwarding_context(runner):
                        cmd = runner.cmd(instance, solver=solver, output_file=output_file)
                        if time_limit is not None:
                            cmd.append("--time_limit")
                            cmd.append(str(time_limit))

                        if _runner.seed is not None:
                            cmd += ["--seed", str(_runner.seed)]
                        if _runner.intermediate:
                            cmd += ["--intermediate"]
                        # Always pass core count to solver so it can use the right number of threads
                        if cores:
                            cmd += ["--cores", str(len(cores))]
                        subprocess_observers = [
                            *getattr(runner, "subprocess_observers", []),
                            f"cpmpy.tools.benchmark.cpbenchy.observer.HookPipeRelayObserver(ipc_events_file={ipc_events_filename!r})",
                        ]
                        cmd += ["--observers", *subprocess_observers]

                        # Prepend setup_command if provided - this wraps the entire command invocation
                        # e.g., systemd-run --user --scope --slice=benchexec -p Delegate=yes python script.py --args
                        if setup_command:
                            cmd = list(setup_command) + cmd
                        # Optional explicit delegated scope around each runexec invocation.
                        # This can break some environments (e.g. missing user bus in nested contexts),
                        # so keep it opt-in.
                        if FORCE_EXPLICIT_SYSTEMD_SCOPE:
                            cmd = _ensure_explicit_systemd_scope(cmd)

                        # Call observe_pre_solve just before starting subprocess
                        try:
                            _runner.observe_pre_solve()
                        except Exception as e:
                            import logging
                            logging.warning(f"observe_pre_solve failed: {e}")

                        result = self._run_worker_with_stream(
                            fifo_path=ipc_events_filename,
                            stream_dispatcher=stream_dispatcher,
                            worker_target=_runexec_worker,
                            worker_args=(cmd, tmp_filename, time_limit, cores, memory_limit, self.pin_cores),
                        )

                    import logging
                    logging.info(f"Subprocess execution returned for instance {instance}, setting subprocess_completed=True")
                    subprocess_completed = True

                    try:
                        with open(tmp_filename, 'r', encoding='utf-8', errors='replace') as f:
                            for line in f:
                                line_stripped = line.strip()
                                if line_stripped and verbose:
                                    # Replay child output verbatim in verbose mode without
                                    # assuming competition-specific line prefixes.
                                    _runner.print_raw(line_stripped)
                    except FileNotFoundError:
                        # Output file might not exist if process was killed before writing
                        logging.warning(f"Output file not found for instance {instance}, but subprocess completed")
                    except Exception as e:
                        # Log but don't fail - subprocess completed, we just couldn't parse output
                        logging.warning(f"Error parsing output file for instance {instance}: {e}")

                    _runner.is_sat = stream_state.is_sat
                    _runner.status = stream_state.is_sat
                    objective_value = stream_state.objective_value
                    intermediate_objectives = list(stream_state.intermediate_objectives)
                    intermediate_solutions = list(stream_state.intermediate_solutions)
                    stage_timings = dict(stream_state.stage_timings)
                    solution_checker = dict(stream_state.solution_checker)
                    
                    # Check termination reason from benchexec result
                    termination_reason = None
                    if "terminationreason" in result:
                        reason = result["terminationreason"]
                        termination_reason = reason
                        if reason == "memory":
                            _runner.print_comment("Memory limit exceeded")
                            # Status remains None for memory errors (observers can check)
                        elif reason == "walltime":
                            _runner.print_comment("Wall time limit exceeded")
                            # Status remains None for timeouts (observers can check)
                    _runner.termination_reason = termination_reason

                    # Attach generic, JSON-safe runner metadata for observers/backends.
                    raw_result = result if isinstance(result, dict) else {}
                    exit_info = _normalize_exit_info(raw_result.get("exitcode"))
                    exit_code = exit_info.get("code") if isinstance(exit_info, dict) else None
                    if termination_reason is None and isinstance(exit_code, int) and exit_code != 0:
                        termination_reason = f"exitcode:{exit_code}"
                        _runner.termination_reason = termination_reason
                    if raw_result.get("walltime") is not None:
                        stage_timings.setdefault("total", raw_result.get("walltime"))
                    if (
                        "retrieve_result" not in stage_timings
                        and isinstance(stage_timings.get("total"), (int, float))
                        and isinstance(stage_timings.get("solve_model"), (int, float))
                    ):
                        transform_time = float(stage_timings.get("transform_model", 0.0) or 0.0)
                        stage_timings["retrieve_result"] = max(
                            float(stage_timings["total"]) - float(stage_timings["solve_model"]) - transform_time,
                            0.0,
                        )
                    _runner.exitcode = exit_code
                    _runner.runner_metadata = {
                        "solver": solver,
                        "termination_reason": termination_reason,
                        "exit_status": stream_state.exit_status,
                        "exit": exit_info,
                        "command": [str(x) for x in cmd],
                        "limits": {
                            "time_limit": time_limit,
                            "memory_limit_mb": memory_limit,
                            "cores": list(cores) if cores else [],
                        },
                        "stats": {
                            "walltime": raw_result.get("walltime"),
                            "cputime": raw_result.get("cputime"),
                            "memory": raw_result.get("memory"),
                        },
                        "stage_timings": stage_timings,
                        "raw_result": raw_result,
                    }
                    # Always capture output_tail for debugging; especially important for errors
                    tail = _tail_text_file(tmp_filename)
                    if tail:
                        _runner.runner_metadata["output_tail"] = tail
                    if intermediate_objectives:
                        _runner.intermediate_objectives = intermediate_objectives
                        _runner.runner_metadata["intermediate_objectives"] = intermediate_objectives
                    if intermediate_solutions:
                        _runner.intermediate_solutions = intermediate_solutions
                        _runner.runner_metadata["intermediate_solutions"] = intermediate_solutions
                    if solution_checker:
                        _runner.solution_checker = solution_checker
                        _runner.runner_metadata["solution_checker"] = solution_checker
                    
                    # Store objective value if found (observers can access it)
                    if objective_value is not None:
                        _runner.objective_value = objective_value
                    
                except Exception as e:
                    # Handle exceptions during subprocess execution
                    import logging
                    import traceback
                    logging.warning(f"Exception during subprocess execution for instance {instance}: {e}")
                    logging.warning(f"Traceback: {traceback.format_exc()}")
                    subprocess_completed = False
                    exc_type, exc_value, exc_traceback = type(e), e, e.__traceback__
                    for observer in _runner.observers:
                        try:
                            if hasattr(observer, 'observe_exception'):
                                observer.observe_exception(_runner, exc_type, exc_value, exc_traceback)
                        except Exception:
                            # Don't let observer exceptions mask the original exception
                            logging.warning(f"observe_exception failed for {observer}: {e}")
                    raise
                finally:
                    # Call post_solve hook if subprocess completed successfully
                    # This ensures it's called even if there were exceptions during output parsing
                    try:
                        import logging
                        if subprocess_completed:
                            logging.info(f"Calling observe_post_solve for instance {instance} (subprocess_completed=True)")
                            _runner.observe_post_solve()
                            logging.info(f"observe_post_solve completed for instance {instance}")
                        else:
                            logging.info(f"Skipping observe_post_solve for instance {instance} (subprocess_completed=False)")
                    except Exception as e:
                        import logging
                        import traceback
                        logging.warning(f"observe_post_solve failed: {e}")
                        logging.warning(f"Traceback: {traceback.format_exc()}")
                    
                    # Always call observe_end
                    try:
                        _runner.observe_end()
                    except Exception as e:
                        import logging
                        logging.warning(f"observe_end failed: {e}")
        finally:
            # Clean up temp artifacts
            try:
                os.unlink(tmp_filename)
            except Exception:
                pass
            self._cleanup_stream_channel(stream_channel)

        _runner.print_comment(f"RunExec result: {result}")
        if hasattr(_runner, "runner_metadata"):
            md = _runner.runner_metadata
            exit_md = md.get("exit") if isinstance(md.get("exit"), dict) else {}
            _runner.print_comment(
                "Runner exit metadata: "
                f"reason={md.get('termination_reason') or 'none'} "
                f"code={exit_md.get('code')} signal={exit_md.get('signal')} "
                f"walltime={md.get('stats', {}).get('walltime')}"
            )

        if "terminationreason" in result:
            reason = result["terminationreason"]
            if reason == "memory":
                _runner.print_comment("Memory limit exceeded")
            elif reason == "walltime":
                _runner.print_comment("Wall time limit exceeded")
        return result

class PythonResourceManager(LocalResourceManager):
    """
    Resource manager that uses Python's resource module for resource control.
    """

    def _run(self,
            instance: str,
            runner: InstanceAdapter,
            time_limit: int,
            memory_limit: int,
            cores: list[int],
            solver: str,
            seed: int,
            intermediate: bool,
            verbose: bool,
            output_file: str,
            setup_command: Optional[List[str]] = None,
            solver_params: Optional[dict] = None,
        ) -> Optional[dict]:
        """
        Run a single instance with assigned resources.

        Arguments:
            instance: Instance file path
            runner: Instance runner
            time_limit: Time limit in seconds
            memory_limit: Memory limit in MB
            cores: List of core IDs to assign to this run (e.g., [0, 1] for cores 0 and 1)

        The python native approach to setting resource limits does not require spawning a separate process for the instance run.
        As a downside, it offers less control over the resources and is less robust.
        """
        # Automatically add WriteToFileObserver and MetadataSidecarObserver if output_file is provided
        if output_file is not None:
            from functools import partial
            from ..observer import WriteToFileObserver, MetadataSidecarObserver, IntermediateObjectivesObserver
            runner.register_observer(partial(WriteToFileObserver, output_file=output_file, overwrite=True))
            runner.register_observer(IntermediateObjectivesObserver())
            runner.register_observer(partial(MetadataSidecarObserver, output_file=output_file))
        
        # Programmatically add ResourceLimitObserver if limits are provided
        if time_limit is not None or memory_limit is not None:
            # Add a resource observer with limits
            resource_observer = ResourceLimitObserver(
                time_limit=time_limit if time_limit is not None else None,
                mem_limit=memory_limit if memory_limit is not None else None
            )
            runner.register_observer(resource_observer)
        
        # Run the instance using the runner's run method.
        runner.run(
            instance=instance,
            solver=solver,
            seed=seed,
            intermediate=intermediate,
            verbose=verbose,
            output_file=output_file,
            time_limit=time_limit,
            mem_limit=memory_limit,
            cores=len(cores) if cores else None,
            **({"solver_params": solver_params} if solver_params else {}),
        )
        return None
        



def run_instance(instance: str, instance_runner: InstanceAdapter, time_limit: int, memory_limit: int, cores: list[int], resource_manager: ResourceManager, solver: str, seed: int, intermediate: bool, verbose: bool, output_file: str, setup_command=None, solver_params=None):
    """
    Run a single instance with assigned cores.

    Arguments:
        instance: Instance file path
        instance_runner: Instance runner
        time_limit: Time limit in seconds
        memory_limit: Memory limit in MB
        cores: List of core IDs to assign to this run (e.g., [0, 1] for cores 0 and 1)
        setup_command: Optional command to prefix before running (list of strings)
        solver_params: Optional dict of extra solver parameters (e.g. {"IntegralityFocus": 1})
    """
    resource_manager.run(instance, instance_runner, time_limit, memory_limit, cores, solver, seed, intermediate, verbose, output_file, setup_command, solver_params=solver_params or {})
    
    
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

    


def load_instance_runner(runner_path: str, **kwargs) -> InstanceAdapter:
    """
    Load an instance runner class from a module path.

    Arguments:
        runner_path: Path to the instance runner class, e.g.,
                     "cpmpy.tools.benchmark.runner.xcsp3_instance_runner.XCSP3InstanceAdapter"
                     or a file path like "/path/to/module.py:ClassName", or a known name
                     ("xcsp3", "opb", "nurserostering", "jsplib", "psplib", "mse").
        **kwargs: Optional attributes to set on the adapter after creation (e.g.
                  include_aux_vars=True to write auxiliary variable values to solution output).

    Returns:
        InstanceAdapter instance
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
        # Known runner names
        if runner_path == "xcsp3":
            adapter = XCSP3Adapter()
        elif runner_path == "opb":
            adapter = OPBAdapter()
        elif runner_path == "nurserostering":
            adapter = NurseRosteringAdapter()
        elif runner_path == "jsplib":
            adapter = JSPLibAdapter()
        elif runner_path == "psplib":
            adapter = PSPLibAdapter()
        elif runner_path == "mse":
            adapter = MSEAdapter()
        else:
            raise ValueError(
                f"Unknown runner: {runner_path}. Use 'xcsp3', 'opb', 'nurserostering', "
                "'jsplib', 'psplib', 'mse', or full module path."
            )
        for key, value in kwargs.items():
            setattr(adapter, key, value)
        return adapter

    if not issubclass(runner_class, InstanceAdapter):
        raise ValueError(f"{runner_class} is not a subclass of InstanceAdapter")

    adapter = runner_class()
    for key, value in kwargs.items():
        setattr(adapter, key, value)
    return adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--time_limit", type=float, required=False, default=None)
    parser.add_argument("--memory_limit", type=int, required=False, default=None)
    parser.add_argument("--cores", type=list[int], required=False, default=None)
    parser.add_argument("--runner", type=str, required=False, default="xcsp3",
                        help="Path to instance runner class. Can be:\n"
                             "- 'xcsp3' (default)\n"
                             "- Module path: 'cpmpy.tools.benchmark.runner.xcsp3_instance_runner.XCSP3InstanceAdapter'\n"
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
        instance_runner = XCSP3Adapter()
    else:
        instance_runner = load_instance_runner(args.runner)

    resource_manager.run(args.instance, instance_runner, args.time_limit, args.memory_limit, args.cores)

if __name__ == "__main__":
    main()