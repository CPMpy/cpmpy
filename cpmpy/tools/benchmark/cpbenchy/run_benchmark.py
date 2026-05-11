#!/usr/bin/env python3
"""
Generic CLI for running benchmarks with any InstanceRunner.

This script provides a flexible command-line interface for running benchmarks
with configurable runners, observers, and run settings.

Usage Examples:
    # Run a single instance
    python run_benchmark.py instance.xml --runner xcsp3 --solver ortools

    # Run a single instance with output file
    python run_benchmark.py instance.xml --runner xcsp3 --solver ortools --output /path/to/output.txt

    # Run with custom observers
    python run_benchmark.py instance.xml --runner xcsp3 --observers CompetitionPrintingObserver RuntimeObserver

    # Run with observer constructor arguments
    python run_benchmark.py instance.xml --runner xcsp3 --observers "WriteToFileObserver(output_file=\"/path/to/file.txt\", overwrite=False)"

    # Run a batch of instances in parallel with output directory
    python run_benchmark.py --batch instances.txt --runner xcsp3 --workers 4 --output ./results

    # Run a dataset with output directory
    python run_benchmark.py --dataset cpmpy.tools.dataset.model.xcsp3.XCSP3Dataset --dataset-year 2024 --dataset-track COP --dataset-download --runner xcsp3 --output ./results

    # Run a dataset with custom root directory
    python run_benchmark.py --dataset cpmpy.tools.dataset.model.xcsp3.XCSP3Dataset --dataset-year 2024 --dataset-track CSP --dataset-root ./data --runner xcsp3 --workers 4 --output ./results

    # Load a custom runner
    python run_benchmark.py instance.xml --runner cpmpy.tools.benchmark.test.xcsp3_instance_runner.XCSP3InstanceRunner
"""

import argparse
import importlib
import inspect
import sys
import ast
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from .instance_runner import InstanceRunner
from .manager import load_instance_runner, run_instance, RunExecResourceManager
from .instance_runner import create_output_file
from .observers import (
    Observer,
    CompetitionPrintingObserver,
    HandlerObserver,
    LoggerObserver,
    # ResourceLimitObserver,
    SolverArgsObserver,
    RuntimeObserver,
    ResultWriterObserver,
    ResourceLimitObserver,
    SolutionCheckerObserver,
    WriteToFileObserver,
)


# Map of observer names to classes
# Note: WriteToFileObserver is not included here as it requires a file_path argument
# Use format "WriteToFileObserver:/path/to/file.txt" if needed, or omit it
# (output files are automatically created in results/ directory via output_file parameter)
OBSERVER_CLASSES = {
    "CompetitionPrintingObserver": CompetitionPrintingObserver,
    "HandlerObserver": HandlerObserver,
    "LoggerObserver": LoggerObserver,
    "ResourceLimitObserver": ResourceLimitObserver,
    "ResultWriterObserver": ResultWriterObserver,
    "SolverArgsObserver": SolverArgsObserver,
    "RuntimeObserver": RuntimeObserver,
    "SolutionCheckerObserver": SolutionCheckerObserver,
}

# Aliases for shorter names
OBSERVER_ALIASES = {
    "WriteToFile": "WriteToFileObserver",
    "Competition": "CompetitionPrintingObserver",
    "Handler": "HandlerObserver",
    "Logger": "LoggerObserver",
    "ResourceLimit": "ResourceLimitObserver",
    "ResultWriter": "ResultWriterObserver",
    "SolverArgs": "SolverArgsObserver",
    "Runtime": "RuntimeObserver",
    "SolutionChecker": "SolutionCheckerObserver",
}


def parse_observer_with_args(observer_spec: str) -> tuple[str, Dict[str, Any]]:
    """
    Parse an observer specification that may include constructor arguments.
    
    Supports formats:
    - "ObserverClass" -> ("ObserverClass", {})
    - "module.path.ObserverClass" -> ("module.path.ObserverClass", {})
    - "ObserverClass(arg1=val1,arg2=val2)" -> ("ObserverClass", {"arg1": val1, "arg2": val2})
    - "module.path.ObserverClass(arg1=val1,arg2=val2)" -> ("module.path.ObserverClass", {"arg1": val1, "arg2": val2})
    
    Arguments:
        observer_spec: Observer specification string
    
    Returns:
        Tuple of (observer_path, kwargs_dict)
    """
    # Check if there are constructor arguments
    # Match pattern: classname(...) where ... can contain nested parentheses
    # We need to find the last opening parenthesis that matches a closing one
    paren_pos = observer_spec.rfind('(')
    if paren_pos != -1 and observer_spec.endswith(')'):
        observer_path = observer_spec[:paren_pos]
        args_str = observer_spec[paren_pos + 1:-1]  # Remove the parentheses
        
        # Parse the arguments string into a dict
        kwargs = {}
        if args_str.strip():
            # Use ast.literal_eval to safely parse the arguments
            # Wrap in braces to make it a dict literal
            try:
                parsed = ast.literal_eval(f"{{{args_str}}}")
                if isinstance(parsed, dict):
                    kwargs = parsed
                else:
                    raise ValueError(f"Invalid argument format: {args_str}. Expected key=value pairs")
            except (ValueError, SyntaxError):
                # If that fails, try manual parsing for key=value pairs
                # This handles cases where values might have commas or special characters
                for pair in args_str.split(','):
                    pair = pair.strip()
                    if '=' in pair:
                        # Find the first = sign (key=value)
                        eq_pos = pair.find('=')
                        key = pair[:eq_pos].strip()
                        value = pair[eq_pos + 1:].strip()
                        # Try to parse the value
                        try:
                            # Try as literal (bool, int, float, None, string)
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # If that fails, treat as string (remove quotes if present)
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                parsed_value = value[1:-1]
                            else:
                                parsed_value = value
                        kwargs[key] = parsed_value
                    else:
                        raise ValueError(f"Invalid argument format: {pair}. Expected 'key=value'")
        
        return observer_path, kwargs
    else:
        return observer_spec, {}


def load_observer(observer_name: str) -> Observer:
    """
    Load an observer by name or module path, optionally with constructor arguments.

    Arguments:
        observer_name: Either a simple name (e.g., "CompetitionPrintingObserver")
                      or a full module path (e.g., "cpmpy.tools.benchmark.test.observer.CompetitionPrintingObserver")
                      or a file path (e.g., "/path/to/file.py:ClassName" or "path/to/file.py::ClassName")
                      or with arguments (e.g., "WriteToFileObserver(file_path='/path/to/file.txt')")
                      For WriteToFileObserver, use format "WriteToFileObserver:file_path" or provide file_path separately

    Returns:
        Observer instance
    """
    import importlib.util
    from pathlib import Path

    # Parse observer name and arguments
    observer_path, kwargs = parse_observer_with_args(observer_name)

    # Resolve aliases at the top level (e.g., "WriteToFile" -> "WriteToFileObserver")
    if observer_path in OBSERVER_ALIASES:
        observer_path = OBSERVER_ALIASES[observer_path]

    # Check for file path format: /path/to/file.py:ClassName or path/to/file.py::ClassName
    # Also handle module.path.to.file.py::ClassName (convert to module path)
    if "::" in observer_path or ("::" not in observer_path and ".py:" in observer_path):
        # Split on :: or : (but not :/ for absolute paths on Windows)
        if "::" in observer_path:
            file_part, class_name = observer_path.rsplit("::", 1)
        else:
            file_part, class_name = observer_path.rsplit(":", 1)

        # Resolve alias for class name (e.g., "WriteToFile" -> "WriteToFileObserver")
        if class_name in OBSERVER_ALIASES:
            class_name = OBSERVER_ALIASES[class_name]

        # Convert to module path format if it looks like module.path.file.py
        if ".py" in file_part and not file_part.startswith("/") and not file_part.startswith("."):
            # Format: cpmpy.tools.benchmark.test.observer.py -> cpmpy.tools.benchmark.test.observer
            module_path = file_part.replace(".py", "")
            try:
                module = importlib.import_module(module_path)
                observer_class = getattr(module, class_name)
                if not issubclass(observer_class, Observer):
                    raise ValueError(f"{observer_class} is not a subclass of Observer")

                # Handle WriteToFileObserver special case
                if class_name == "WriteToFileObserver":
                    if "file_path" in kwargs and "output_file" not in kwargs:
                        kwargs["output_file"] = kwargs.pop("file_path")
                    if "output_file" not in kwargs:
                        # Default output file
                        kwargs["output_file"] = "results/output.txt"

                return observer_class(**kwargs)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load observer '{observer_path}': {e}")

        # Handle actual file paths
        file_path = Path(file_part).resolve()
        if file_path.exists():
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
            observer_class = getattr(module, class_name)
            if not issubclass(observer_class, Observer):
                raise ValueError(f"{observer_class} is not a subclass of Observer")

            # Handle WriteToFileObserver special case
            if class_name == "WriteToFileObserver":
                if "file_path" in kwargs and "output_file" not in kwargs:
                    kwargs["output_file"] = kwargs.pop("file_path")
                if "output_file" not in kwargs:
                    # Default output file
                    kwargs["output_file"] = "results/output.txt"

            return observer_class(**kwargs)

    # Special handling for WriteToFileObserver
    if observer_path.startswith("WriteToFileObserver") or observer_path.endswith("WriteToFileObserver"):
        if ":" in observer_path and "::" not in observer_path:
            # Format: WriteToFileObserver:/path/to/file.txt (legacy format)
            _, file_path = observer_path.split(":", 1)
            kwargs["output_file"] = file_path
            return WriteToFileObserver(**kwargs)
        # Support both output_file and file_path for backward compatibility
        if "file_path" in kwargs and "output_file" not in kwargs:
            kwargs["output_file"] = kwargs.pop("file_path")
        if "output_file" not in kwargs:
            # Default output file
            kwargs["output_file"] = "results/output.txt"
        return WriteToFileObserver(**kwargs)

    # Check if it's a known observer name
    if observer_path in OBSERVER_CLASSES:
        observer_class = OBSERVER_CLASSES[observer_path]
        return observer_class(**kwargs)

    # Try to load from module path
    if "." in observer_path:
        module_path, class_name = observer_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            observer_class = getattr(module, class_name)
            if not issubclass(observer_class, Observer):
                raise ValueError(f"{observer_class} is not a subclass of Observer")
            # Check if it's WriteToFileObserver loaded via module path
            if class_name == "WriteToFileObserver":
                # Support both output_file and file_path for backward compatibility
                if "file_path" in kwargs and "output_file" not in kwargs:
                    kwargs["output_file"] = kwargs.pop("file_path")
                if "output_file" not in kwargs:
                    # Default output file
                    kwargs["output_file"] = "results/output.txt"
            return observer_class(**kwargs)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load observer '{observer_path}': {e}")

    raise ValueError(f"Unknown observer: {observer_path}. Available: {', '.join(OBSERVER_CLASSES.keys())}")


def load_observers(observer_names: Optional[List[str]]) -> List[Observer]:
    """
    Load multiple observers from a list of names.
    
    Arguments:
        observer_names: List of observer names or module paths
    
    Returns:
        List of Observer instances
    """
    if not observer_names:
        return []
    
    observers = []
    for name in observer_names:
        observers.append(load_observer(name))
    
    return observers


def benchmark_output_file(output_dir: Optional[str], solver: str, instance: str) -> Optional[str]:
    """Return the deterministic text output path for a batch/dataset instance."""
    if output_dir is None:
        return None
    instance_name = Path(instance).stem
    return create_output_file(None, output_dir, solver, instance_name)


def benchmark_json_file(output_dir: Optional[str], solver: str, instance: str) -> Optional[str]:
    """Return the JSON result path expected beside the text output file."""
    output_file = benchmark_output_file(output_dir, solver, instance)
    return f"{output_file}.json" if output_file is not None else None


def filter_existing_results(instances: List[str], output_dir: Optional[str], solver: str, force: bool = False) -> tuple[List[str], int]:
    """Skip instances with an existing JSON result unless force is enabled."""
    if force or output_dir is None:
        return instances, 0

    pending = []
    skipped = 0
    for instance in instances:
        json_file = benchmark_json_file(output_dir, solver, instance)
        if json_file is not None and Path(json_file).exists():
            skipped += 1
        else:
            pending.append(instance)
    return pending, skipped


def _progress_line(done: int, total: int, width: int = 36) -> str:
    if total <= 0:
        return "Progress: 0/0"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"Progress [{bar}] {done}/{total} ({ratio * 100:5.1f}%)"


def _count_completed_results(expected_json_files: List[str], start_ns: int) -> int:
    done = 0
    for json_file in expected_json_files:
        try:
            path = Path(json_file)
            if path.exists() and path.stat().st_mtime_ns >= start_ns:
                done += 1
        except OSError:
            pass
    return done


def _flatten_json(prefix: str, value: Any, row: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_json(nested_key, nested_value, row)
    elif isinstance(value, (list, tuple)):
        row[prefix] = json.dumps(value)
    else:
        row[prefix] = value


def write_summary_csv(output_dir: str, summary_file: Optional[str] = None) -> str:
    """Write a generic CSV summary by flattening all JSON results under output_dir."""
    output_path = Path(output_dir)
    if summary_file is None:
        summary_path = output_path / "summary.csv"
    else:
        summary_path = Path(summary_file)

    rows = []
    fieldnames = set()
    for json_file in sorted(output_path.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue
        row = {"result_file": str(json_file)}
        _flatten_json("", data, row)
        rows.append(row)
        fieldnames.update(row.keys())

    preferred = [
        "result_file",
        "instance_path",
        "solver",
        "exitstatus",
        "objective_value",
        "time_total_seconds",
        "time_parse_seconds",
        "time_solve_seconds",
        "solver_runtime_seconds",
    ]
    ordered_fields = [name for name in preferred if name in fieldnames]
    ordered_fields.extend(sorted(fieldnames - set(ordered_fields)))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(summary_path)


def build_dataset_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Build dataset constructor kwargs from generic CLI arguments."""
    dataset_kwargs = {"root": args.dataset_root}
    if args.dataset_download:
        dataset_kwargs["download"] = True
    if args.dataset_year is not None:
        dataset_kwargs["year"] = args.dataset_year
    if args.dataset_track:
        dataset_kwargs["track"] = args.dataset_track
    if args.dataset_variant:
        dataset_kwargs["variant"] = args.dataset_variant
    if args.dataset_family:
        dataset_kwargs["family"] = args.dataset_family
    if args.dataset_option:
        for key, value in args.dataset_option:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
            dataset_kwargs[key] = value
    return dataset_kwargs


def resolve_instances(args: argparse.Namespace) -> List[str]:
    """Resolve benchmark instances from exactly one CLI input source."""
    if args.instance is not None:
        return [args.instance]
    if args.batch is not None:
        return parse_instance_list(args.batch)

    dataset = load_dataset(args.dataset, build_dataset_kwargs(args))
    instances = []
    for instance, metadata in dataset:
        instances.append(instance)
    return instances


def print_dry_run(
    instances: List[str],
    runner_path: str,
    solver: str,
    workers: Optional[int],
    cores_per_worker: int,
    memory_per_worker: Optional[int],
    mem_limit: Optional[int],
    output_dir: Optional[str],
) -> None:
    print("Dry run")
    print(f"Runner: {runner_path}")
    print(f"Solver: {solver}")
    print(f"Instances: {len(instances)}")
    print(f"Workers: {workers}")
    print(f"Cores per worker: {cores_per_worker}")
    print(f"Memory per worker: {memory_per_worker} MiB" if memory_per_worker is not None else "Memory per worker: none")
    print(f"Memory limit: {mem_limit} MiB" if mem_limit is not None else "Memory limit: none")
    if output_dir is not None:
        print(f"Output directory: {output_dir}")
    for instance in instances[:20]:
        output_file = benchmark_output_file(output_dir, solver, instance)
        suffix = f" -> {output_file}" if output_file is not None else ""
        print(f"  {instance}{suffix}")
    if len(instances) > 20:
        print(f"  ... {len(instances) - 20} more")


def run_single_instance(
    instance: str,
    runner: InstanceRunner,
    solver: str = "ortools",
    time_limit: Optional[float] = None,
    mem_limit: Optional[int] = None,
    seed: Optional[int] = None,
    cores: Optional[int] = None,
    intermediate: bool = False,
    verbose: bool = False,
    output_file: Optional[str] = None,
    additional_observers: Optional[List[Observer]] = None,
):
    """
    Run a single instance with the given runner and settings.
    """
    # Automatically add WriteToFileObserver if output_file is provided
    if output_file is not None:
        # Ensure the output file path is absolute or properly constructed
        from functools import partial
        output_file = create_output_file(output_file, None, solver, instance)
        runner.register_observer(partial(WriteToFileObserver, output_file=output_file, overwrite=True))
    
    # Register additional observers
    if additional_observers:
        for observer in additional_observers:
            runner.register_observer(observer)
    
    # Run the instance
    runner.run(
        instance=instance,
        solver=solver,
        time_limit=time_limit,
        mem_limit=mem_limit,
        seed=seed,
        cores=cores,
        intermediate=intermediate,
        verbose=verbose,
        output_file=output_file,
    )


def _print_log_tail(log_path: str, max_lines: int = 40) -> None:
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        lines = []
    tail = "\n".join(lines[-max_lines:]) if lines else "(no captured output)"
    print(f"Worker log tail ({log_path}):", file=sys.stderr)
    print(tail, file=sys.stderr)


def worker_function(worker_id, cores, job_queue, time_limit, memory_limit, runner_path, solver, seed, intermediate, verbose, output_dir, quiet: bool = False):
    """Worker function for parallel execution."""
    import contextlib
    import tempfile

    resource_manager = RunExecResourceManager()
    
    while True:
        try:
            instance, metadata = job_queue.get_nowait()
        except Exception:
            break
        
        # Create a fresh instance_runner for each instance to avoid observer accumulation
        instance_runner = load_instance_runner(runner_path)
        
        # Construct output_file path for this instance
        output_file = benchmark_output_file(output_dir, solver, instance)

        if quiet:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=f".worker-{worker_id}.log", delete=False) as log_file:
                log_path = log_file.name
                try:
                    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                        run_instance(
                            instance,
                            instance_runner,
                            time_limit,
                            memory_limit,
                            cores,
                            resource_manager,
                            solver,
                            seed,
                            intermediate,
                            verbose,
                            output_file,
                        )
                except Exception:
                    _print_log_tail(log_path)
                    raise
        else:
            run_instance(
                instance,
                instance_runner,
                time_limit,
                memory_limit,
                cores,
                resource_manager,
                solver,
                seed,
                intermediate,
                verbose,
                output_file,
            )
        job_queue.task_done()


def compute_workers_and_memory(
    workers: Optional[int],
    total_memory: Optional[int],
    memory_per_worker: Optional[int],
    ignore_check: bool = False,
) -> tuple[int, Optional[int]]:
    """
    Compute workers and memory_per_worker from the given parameters.
    
    Derives whichever value is missing:
    - If total_memory and memory_per_worker are set, derive workers
    - If total_memory and workers are set, derive memory_per_worker
    - If memory_per_worker and workers are set, derive total_memory (but return memory_per_worker)
    
    If total_memory is not provided, it will be automatically measured from the system.
    
    If all are set, checks feasibility: total_memory == workers * memory_per_worker
    
    Arguments:
        workers: Number of workers (None to derive)
        total_memory: Total memory in MiB (None to derive or measure)
        memory_per_worker: Memory per worker in MiB (None to derive)
        ignore_check: If True, ignore feasibility check and just warn
    
    Returns:
        Tuple of (workers, memory_per_worker)
    """
    import psutil
    
    # If total_memory is not provided, measure it from the system
    if total_memory is None:
        # Get total virtual memory in bytes and convert to MiB
        total_memory = psutil.virtual_memory().total // (1024 * 1024)
    
    # Count how many values are set (now total_memory is always set)
    set_count = sum(1 for x in [workers, memory_per_worker] if x is not None)
    
    if set_count == 0:
        # Defaults: 1 worker, no memory limit per worker
        return 1, None
    
    if set_count == 1:
        # Only one value set - derive the other from total_memory
        if workers is not None:
            # Derive memory_per_worker from total_memory and workers
            if total_memory % workers != 0:
                raise ValueError(
                    f"Measured total-memory ({total_memory} MiB) is not evenly divisible by "
                    f"workers ({workers})"
                )
            memory_per_worker = total_memory // workers
            if memory_per_worker < 1:
                raise ValueError(
                    f"Derived memory-per-worker ({memory_per_worker} MiB) must be at least 1. "
                    f"Check your workers value relative to available memory ({total_memory} MiB)."
                )
            return workers, memory_per_worker
        else:  # memory_per_worker is not None
            # Derive workers from total_memory and memory_per_worker
            if total_memory % memory_per_worker != 0:
                raise ValueError(
                    f"Measured total-memory ({total_memory} MiB) is not evenly divisible by "
                    f"memory-per-worker ({memory_per_worker} MiB)"
                )
            workers = total_memory // memory_per_worker
            if workers < 1:
                raise ValueError(
                    f"Derived workers ({workers}) must be at least 1. "
                    f"Check your memory-per-worker value relative to available memory ({total_memory} MiB)."
                )
            return workers, memory_per_worker
    
    if set_count == 2:
        # Both workers and memory_per_worker are provided - use them as-is
        # Derive total_memory for validation only (don't override user input)
        expected_total = workers * memory_per_worker
        if total_memory < expected_total:
            # Warn if measured total is less than what's needed
            message = (
                f"Memory configuration: workers ({workers}) × memory-per-worker ({memory_per_worker} MiB) = "
                f"{expected_total} MiB, but measured total-memory is {total_memory} MiB. "
                f"Using specified memory-per-worker ({memory_per_worker} MiB) anyway."
            )
            print(f"WARNING: {message}", file=sys.stderr)
        # Use the user-provided values as-is - manual input always takes precedence
        return workers, memory_per_worker
    
    else:  # set_count == 3, all values are set
        # Check feasibility
        expected_total = workers * memory_per_worker
        if total_memory != expected_total:
            message = (
                f"Memory configuration is not feasible: "
                f"workers ({workers}) × memory-per-worker ({memory_per_worker} MiB) = "
                f"{expected_total} MiB, but total-memory is {total_memory} MiB"
            )
            if ignore_check:
                print(f"WARNING: {message}. Continuing anyway...", file=sys.stderr)
            else:
                raise ValueError(message + ". Use --ignore-memory-check to override.")
    
    return workers, memory_per_worker


def run_batch(
    instances: List[str],
    runner_path: str,
    solver: str = "ortools",
    time_limit: Optional[float] = None,
    mem_limit: Optional[int] = None,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    cores_per_worker: int = 1,
    total_memory: Optional[int] = None,
    memory_per_worker: Optional[int] = None,
    ignore_memory_check: bool = False,
    intermediate: bool = False,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    quiet_progress: bool = False,
    show_progress: bool = True,
):
    """
    Run a batch of instances in parallel.
    """
    import psutil
    
    # Store original user inputs to preserve manual overrides
    original_memory_per_worker = memory_per_worker
    original_workers = workers
    
    # Compute workers and memory_per_worker from the given parameters
    computed_workers, computed_memory_per_worker = compute_workers_and_memory(
        workers, total_memory, memory_per_worker, ignore_memory_check
    )
    
    # Use computed workers (unless both were provided, then use original)
    if original_workers is not None and original_memory_per_worker is not None:
        # Both were provided - use original values (compute_workers_and_memory already returns them)
        workers = computed_workers
        memory_per_worker = computed_memory_per_worker
    else:
        # Use computed values
        workers = computed_workers
        if original_memory_per_worker is not None:
            # User explicitly provided memory_per_worker - use it
            memory_per_worker = original_memory_per_worker
        else:
            memory_per_worker = computed_memory_per_worker
    
    # Use memory_per_worker as mem_limit if not explicitly set
    # But if user explicitly set mem_limit, that takes precedence
    if mem_limit is None and memory_per_worker is not None:
        mem_limit = memory_per_worker
    
    total_cores = psutil.cpu_count(logical=False)
    
    if workers * cores_per_worker > total_cores:
        raise ValueError(
            f"Not enough cores: {workers} workers × {cores_per_worker} cores = "
            f"{workers * cores_per_worker} cores needed, but only {total_cores} available"
        )
    
    # Assign cores to each worker
    worker_cores = []
    for i in range(workers):
        start_core = i * cores_per_worker
        end_core = start_core + cores_per_worker
        cores = list(range(start_core, end_core))
        worker_cores.append(cores)
    
    if verbose:
        print(f"Total cores: {total_cores}, Workers: {workers}, Cores per worker: {cores_per_worker}")
        for i, cores in enumerate(worker_cores):
            print(f"Worker {i}: cores {cores}")
    
    # Create a queue of all jobs
    with Manager() as manager:
        job_queue = manager.Queue()
        for instance in instances:
            job_queue.put((instance, {}))
        
        expected_json_files = [
            json_file for json_file in (
                benchmark_json_file(output_dir, solver, instance)
                for instance in instances
            )
            if json_file is not None
        ]
        start_ns = time.time_ns()

        # Submit workers to the executor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    worker_function,
                    worker_id,
                    cores,
                    job_queue,
                    time_limit,
                    mem_limit,
                    runner_path,
                    solver,
                    seed,
                    intermediate,
                    verbose,
                    output_dir,
                    quiet_progress,
                )
                for worker_id, cores in enumerate(worker_cores)
            ]
            # Wait for all workers to finish
            while any(not future.done() for future in futures):
                if show_progress and expected_json_files:
                    done = _count_completed_results(expected_json_files, start_ns)
                    print("\r" + _progress_line(done, len(expected_json_files)), end="", flush=True)
                time.sleep(1.0)
            if show_progress and expected_json_files:
                done = _count_completed_results(expected_json_files, start_ns)
                print("\r" + _progress_line(done, len(expected_json_files)), flush=True)
            for future in futures:
                future.result()


def parse_instance_list(file_path: str) -> List[str]:
    """Parse a file containing instance paths (one per line)."""
    with open(file_path, 'r') as f:
        instances = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return instances


def load_dataset(dataset_path, dataset_kwargs: dict):
    """
    Load a dataset class and instantiate it with the given kwargs.
    
    Arguments:
        dataset_path: Dataset class/callable, or path to the dataset class, e.g.,
                     "cpmpy.tools.dataset.model.xcsp3.XCSP3Dataset"
                     or a file path like "/path/to/dataset.py:ClassName"
        dataset_kwargs: Dictionary of keyword arguments to pass to the dataset constructor
    
    Returns:
        Dataset instance
    """
    import importlib.util
    from pathlib import Path

    if isinstance(dataset_path, type) or (callable(dataset_path) and not isinstance(dataset_path, str)):
        dataset_class = dataset_path
    elif ":" in dataset_path:
        # Format: /path/to/dataset.py:ClassName
        file_path, class_name = dataset_path.rsplit(":", 1)
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
        dataset_class = getattr(module, class_name)
    elif "." in dataset_path:
        # Format: module.path.ClassName
        module_path, class_name = dataset_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
    else:
        raise ValueError(f"Invalid dataset path format: {dataset_path}. Use 'module.path.ClassName' or '/path/to/file.py:ClassName'")
    
    # Instantiate the dataset with the provided kwargs
    return dataset_class(**dataset_kwargs)


class BenchmarkCLI:
    """Configurable generic benchmark CLI for domain-specific entrypoints."""

    def __init__(
        self,
        default_runner: Any = "xcsp3",
        default_dataset: Any = None,
        description: str = "Generic CLI for running benchmarks with any InstanceRunner",
        epilog: Optional[str] = __doc__,
        show_runner: bool = True,
        show_dataset: bool = True,
        show_observers: bool = True,
        show_dataset_options: bool = True,
        show_dataset_extra_options: bool = True,
    ):
        self.default_runner = default_runner
        self.default_dataset = default_dataset
        self.description = description
        self.epilog = epilog
        self.show_runner = show_runner
        self.show_dataset = show_dataset
        self.show_observers = show_observers
        self.show_dataset_options = show_dataset_options
        self.show_dataset_extra_options = show_dataset_extra_options

    @staticmethod
    def _normalize_runner_spec(runner_spec: Any) -> str:
        """Accept runner as spec string or InstanceRunner class."""
        if isinstance(runner_spec, str):
            return runner_spec
        if inspect.isclass(runner_spec) and issubclass(runner_spec, InstanceRunner):
            return f"{Path(inspect.getfile(runner_spec)).resolve()}:{runner_spec.__name__}"
        raise TypeError(
            f"Unsupported runner spec type {type(runner_spec).__name__}; "
            "expected runner spec string or InstanceRunner subclass."
        )

    def argparser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.epilog,
        )

        parser.add_argument("instance", nargs="?", type=str, help="Path to a single instance file to run")
        parser.add_argument("--batch", type=str, metavar="FILE", help="Path to a file containing instance paths (one per line)")
        parser.add_argument(
            "--dataset",
            type=str,
            default=self.default_dataset,
            metavar="DATASET_CLASS",
            help=(
                "Dataset class to use. Can be a full module path or a file path like '/path/to/dataset.py:ClassName'"
                if self.show_dataset
                else argparse.SUPPRESS
            ),
        )
        parser.add_argument(
            "--runner",
            type=str,
            default=self._normalize_runner_spec(self.default_runner),
            help=(
                "InstanceRunner to use. Can be a simple name, module path, or '/path/to/runner.py:ClassName'"
                if self.show_runner
                else argparse.SUPPRESS
            ),
        )
        parser.add_argument(
            "--observers",
            type=str,
            nargs="+",
            default=None,
            metavar="OBSERVER",
            help=(
                "Additional observers to register. Available: " + ", ".join(OBSERVER_CLASSES.keys())
                if self.show_observers
                else argparse.SUPPRESS
            ),
        )

        parser.add_argument("--solver", type=str, default="ortools", help="Solver to use (default: ortools)")
        parser.add_argument("--time_limit", type=float, default=None, help="Time limit in seconds")
        parser.add_argument("--mem_limit", type=int, default=None, help="Memory limit in MiB")
        parser.add_argument("--seed", type=int, default=None, help="Random seed")
        parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use (for single instance)")
        parser.add_argument("--intermediate", action="store_true", help="Print intermediate solutions")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output path: file for a single instance, directory for batch/dataset runs",
        )
        parser.add_argument("--force", action="store_true", help="Rerun batch/dataset instances with existing JSON results")
        parser.add_argument("--dry-run", action="store_true", help="Print the planned run without executing instances")
        parser.add_argument(
            "--summary-csv",
            nargs="?",
            const=True,
            default=False,
            metavar="PATH",
            help="Write a generic CSV summary from JSON results. With no PATH, writes summary.csv in the output directory.",
        )
        parser.add_argument("--quiet-progress", action="store_true", help="Hide worker output and show artifact-count progress")
        parser.add_argument("--no-progress", action="store_true", help="Disable batch/dataset progress display")

        parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers for batch/dataset runs")
        parser.add_argument("--cores_per_worker", type=int, default=1, help="Number of cores per worker (default: 1)")
        parser.add_argument("--total-memory", type=int, default=None, metavar="MiB", help="Total memory available in MiB")
        parser.add_argument("--memory-per-worker", type=int, default=None, metavar="MiB", help="Memory per worker in MiB")
        parser.add_argument("--ignore-memory-check", action="store_true", help="Warn instead of failing on infeasible memory settings")

        parser.add_argument(
            "--dataset-root",
            type=str,
            default="./data",
            help="Root directory for dataset (default: './data')" if self.show_dataset_options else argparse.SUPPRESS,
        )
        parser.add_argument(
            "--dataset-year",
            type=int,
            default=None,
            help="Dataset year" if self.show_dataset_options else argparse.SUPPRESS,
        )
        parser.add_argument(
            "--dataset-track",
            type=str,
            default=None,
            help="Dataset track" if self.show_dataset_options else argparse.SUPPRESS,
        )
        parser.add_argument(
            "--dataset-download",
            action="store_true",
            help="Download dataset if not available locally" if self.show_dataset_options else argparse.SUPPRESS,
        )
        parser.add_argument(
            "--dataset-variant",
            type=str,
            default=None,
            help="Dataset variant" if self.show_dataset_extra_options else argparse.SUPPRESS,
        )
        parser.add_argument(
            "--dataset-family",
            type=str,
            default=None,
            help="Dataset family" if self.show_dataset_extra_options else argparse.SUPPRESS,
        )
        parser.add_argument(
            "--dataset-option",
            type=str,
            nargs=2,
            action="append",
            metavar=("KEY", "VALUE"),
            help=(
                "Additional dataset options as key-value pairs"
                if self.show_dataset_extra_options
                else argparse.SUPPRESS
            ),
        )
        return parser

    def run(self, argv: Optional[List[str]] = None):
        parser = self.argparser()
        args = parser.parse_args(argv)

        provided = sum([
            args.instance is not None,
            args.batch is not None,
            args.dataset is not None and args.dataset != self.default_dataset,
        ])
        if provided > 1:
            parser.error("Only one of 'instance', '--batch', or '--dataset' can be provided")
        if args.instance is None and args.batch is None and args.dataset is None:
            parser.error("One of 'instance', '--batch', or '--dataset' must be provided")

        try:
            if args.runner == "xcsp3":
                from .run_xcsp3_instance import XCSP3InstanceRunner
                runner = XCSP3InstanceRunner()
            else:
                runner = load_instance_runner(args.runner)
        except Exception as e:
            print(f"Error loading runner '{args.runner}': {e}", file=sys.stderr)
            sys.exit(1)

        additional_observers = None
        if args.observers:
            try:
                additional_observers = load_observers(args.observers)
            except Exception as e:
                print(f"Error loading observers: {e}", file=sys.stderr)
                sys.exit(1)

        try:
            self._run_args(args, runner, additional_observers)
        except Exception as e:
            print(f"Error running benchmark: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _run_args(self, args: argparse.Namespace, runner: InstanceRunner, additional_observers: Optional[List[Observer]] = None):
        print("Preparing benchmark...", flush=True)
        if args.instance is not None:
            print(f"Input source: single instance {args.instance}", flush=True)
        elif args.batch is not None:
            print(f"Input source: batch list {args.batch}", flush=True)
        else:
            dataset_name = getattr(args.dataset, "__name__", str(args.dataset))
            print(f"Input source: dataset {dataset_name}", flush=True)

        instances = resolve_instances(args)
        if not instances:
            print("No instances found", file=sys.stderr)
            sys.exit(1)
        print(f"Resolved {len(instances)} instance(s)", flush=True)

        # A domain-specific CLI may carry a default dataset on args even when
        # the user supplied a positional instance. Treat that as a single run.
        is_batch_run = args.batch is not None or (args.instance is None and args.dataset is not None)

        workers = args.workers
        memory_per_worker = args.memory_per_worker
        effective_mem_limit = args.mem_limit
        if is_batch_run:
            workers, memory_per_worker = compute_workers_and_memory(
                args.workers, args.total_memory, args.memory_per_worker, args.ignore_memory_check
            )
            if effective_mem_limit is None:
                effective_mem_limit = memory_per_worker

        if is_batch_run:
            before = len(instances)
            instances, skipped = filter_existing_results(
                instances=instances,
                output_dir=args.output,
                solver=args.solver,
                force=args.force,
            )
            if skipped:
                print(f"Skipping {skipped} already-completed instance(s); {len(instances)} queued")
            if not instances:
                print("Nothing to run.")
                if args.summary_csv and args.output:
                    summary_path = write_summary_csv(
                        args.output,
                        None if args.summary_csv is True else args.summary_csv,
                    )
                    print(f"Summary written to {summary_path}")
                return
            if args.verbose:
                import psutil
                actual_total = args.total_memory if args.total_memory is not None else psutil.virtual_memory().total // (1024 * 1024)
                source = "user-specified" if args.total_memory else "measured from system"
                print(f"Total memory: {actual_total} MiB ({source})")
                if memory_per_worker:
                    print(f"Memory per worker: {memory_per_worker} MiB")
                print(f"Running {len(instances)} of {before} instance(s) with {workers} worker(s)")
            else:
                print(f"Queued {len(instances)} instance(s) on {workers} worker(s)", flush=True)

        if args.dry_run:
            print_dry_run(
                instances=instances,
                runner_path=args.runner,
                solver=args.solver,
                workers=workers if is_batch_run else 1,
                cores_per_worker=args.cores_per_worker if is_batch_run else (args.cores or 1),
                memory_per_worker=memory_per_worker,
                mem_limit=effective_mem_limit,
                output_dir=args.output if is_batch_run else None,
            )
            return

        if is_batch_run:
            print("Starting batch run...", flush=True)
            run_batch(
                instances=instances,
                runner_path=args.runner,
                solver=args.solver,
                time_limit=args.time_limit,
                mem_limit=effective_mem_limit,
                seed=args.seed,
                workers=workers,
                cores_per_worker=args.cores_per_worker,
                total_memory=args.total_memory,
                memory_per_worker=memory_per_worker,
                ignore_memory_check=args.ignore_memory_check,
                intermediate=args.intermediate,
                verbose=args.verbose,
                output_dir=args.output,
                quiet_progress=args.quiet_progress,
                show_progress=not args.no_progress,
            )
            if args.summary_csv:
                if not args.output:
                    raise ValueError("--summary-csv requires --output for batch/dataset runs")
                summary_path = write_summary_csv(
                    args.output,
                    None if args.summary_csv is True else args.summary_csv,
                )
                print(f"Summary written to {summary_path}")
        else:
            run_single_instance(
                instance=instances[0],
                runner=runner,
                solver=args.solver,
                time_limit=args.time_limit,
                mem_limit=args.mem_limit,
                seed=args.seed,
                cores=args.cores,
                intermediate=args.intermediate,
                verbose=args.verbose,
                output_file=args.output,
                additional_observers=additional_observers,
            )


def main():
    BenchmarkCLI().run()


if __name__ == "__main__":
    main()
