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
import sys
import ast
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from cpmpy.tools.benchmark.test.instance_runner import InstanceRunner
from cpmpy.tools.benchmark.test.manager import load_instance_runner, run_instance, RunExecResourceManager
from cpmpy.tools.benchmark.test.observer import (
    Observer,
    CompetitionPrintingObserver,
    HandlerObserver,
    LoggerObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
    RuntimeObserver,
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
        from cpmpy.tools.benchmark.test.instance_runner import create_output_file
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


def worker_function(worker_id, cores, job_queue, time_limit, memory_limit, runner_path, solver, seed, intermediate, verbose, output_dir):
    """Worker function for parallel execution."""
    resource_manager = RunExecResourceManager()
    
    while True:
        try:
            instance, metadata = job_queue.get_nowait()
        except Exception:
            break
        
        # Create a fresh instance_runner for each instance to avoid observer accumulation
        instance_runner = load_instance_runner(runner_path)
        
        # Construct output_file path for this instance
        output_file = None
        if output_dir is not None:
            from cpmpy.tools.benchmark.test.instance_runner import create_output_file
            # Extract instance name for filename
            import os
            instance_name = os.path.splitext(os.path.basename(instance))[0]
            output_file = create_output_file(None, output_dir, solver, instance_name)
            # Note: WriteToFileObserver will be automatically added by the resource manager
        
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
                )
                for worker_id, cores in enumerate(worker_cores)
            ]
            # Wait for all workers to finish
            for future in futures:
                future.result()


def parse_instance_list(file_path: str) -> List[str]:
    """Parse a file containing instance paths (one per line)."""
    with open(file_path, 'r') as f:
        instances = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return instances


def load_dataset(dataset_path: str, dataset_kwargs: dict):
    """
    Load a dataset class and instantiate it with the given kwargs.
    
    Arguments:
        dataset_path: Path to the dataset class, e.g., 
                     "cpmpy.tools.dataset.model.xcsp3.XCSP3Dataset"
                     or a file path like "/path/to/dataset.py:ClassName"
        dataset_kwargs: Dictionary of keyword arguments to pass to the dataset constructor
    
    Returns:
        Dataset instance
    """
    import importlib.util
    from pathlib import Path
    
    if ":" in dataset_path:
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


def main():
    parser = argparse.ArgumentParser(
        description="Generic CLI for running benchmarks with any InstanceRunner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Instance input - use optional positional and check manually to avoid argparse issues
    parser.add_argument(
        "instance",
        nargs="?",
        type=str,
        help="Path to a single instance file to run"
    )
    parser.add_argument(
        "--batch",
        type=str,
        metavar="FILE",
        help="Path to a file containing instance paths (one per line) for batch processing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        metavar="DATASET_CLASS",
        help="Dataset class to use. Can be a full module path "
             "(e.g., 'cpmpy.tools.dataset.model.xcsp3.XCSP3Dataset') "
             "or a file path (e.g., '/path/to/dataset.py:ClassName')"
    )
    
    # Runner configuration
    parser.add_argument(
        "--runner",
        type=str,
        default="xcsp3",
        help="InstanceRunner to use. Can be a simple name (e.g., 'xcsp3') or a full module path "
             "(e.g., 'cpmpy.tools.benchmark.test.xcsp3_instance_runner.XCSP3InstanceRunner') "
             "or a file path (e.g., '/path/to/runner.py:ClassName')"
    )
    
    # Observer configuration
    parser.add_argument(
        "--observers",
        type=str,
        nargs="+",
        default=None,
        metavar="OBSERVER",
        help="Additional observers to register. Can specify multiple. "
             "Available: " + ", ".join(OBSERVER_CLASSES.keys()) + ". "
             "Or use full module path like 'cpmpy.tools.benchmark.test.observer.CompetitionPrintingObserver'. "
             "To pass constructor arguments, use format 'ObserverClass(arg1=val1,arg2=val2)'. "
             "Example: 'WriteToFileObserver(file_path=\"/path/to/file.txt\", overwrite=False)'. "
             "Note: WriteToFileObserver is automatically added to write outputs to results/ directory. "
             "To use a custom file path, use format 'WriteToFileObserver(file_path=\"/path/to/file.txt\")'"
    )
    
    # Solver settings
    parser.add_argument(
        "--solver",
        type=str,
        default="ortools",
        help="Solver to use (default: ortools)"
    )
    
    # Run settings
    parser.add_argument(
        "--time_limit",
        type=float,
        default=None,
        help="Time limit in seconds"
    )
    parser.add_argument(
        "--mem_limit",
        type=int,
        default=None,
        help="Memory limit in MiB"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use (for single instance)"
    )
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Print intermediate solutions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path: for single instance, this is the output file path; "
             "for batch/dataset, this is the directory where output files will be placed"
    )
    
    # Batch processing settings
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for batch processing. If not set, will be derived from "
             "total-memory and memory-per-worker if those are set."
    )
    parser.add_argument(
        "--cores_per_worker",
        type=int,
        default=1,
        help="Number of cores per worker for batch processing (default: 1)"
    )
    parser.add_argument(
        "--total-memory",
        type=int,
        default=None,
        metavar="MiB",
        help="Total memory available in MiB. If set along with memory-per-worker, will derive "
             "number of workers. If set along with workers, will derive memory-per-worker."
    )
    parser.add_argument(
        "--memory-per-worker",
        type=int,
        default=None,
        metavar="MiB",
        help="Memory per worker in MiB. If set along with total-memory, will derive "
             "number of workers. If set along with workers, will derive total-memory."
    )
    parser.add_argument(
        "--ignore-memory-check",
        action="store_true",
        help="Ignore feasibility check when all memory/worker parameters are set. "
             "Will print a warning if configuration is not feasible but still allow the run to start."
    )
    
    # Dataset configuration options
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./data",
        help="Root directory for dataset (default: './data')"
    )
    parser.add_argument(
        "--dataset-year",
        type=int,
        default=None,
        help="Year for dataset (e.g., 2024 for XCSP3Dataset)"
    )
    parser.add_argument(
        "--dataset-track",
        type=str,
        default=None,
        help="Track for dataset (e.g., 'COP', 'CSP' for XCSP3Dataset)"
    )
    parser.add_argument(
        "--dataset-download",
        action="store_true",
        help="Download dataset if not available locally"
    )
    parser.add_argument(
        "--dataset-variant",
        type=str,
        default=None,
        help="Variant for dataset (e.g., for PSPLibDataset)"
    )
    parser.add_argument(
        "--dataset-family",
        type=str,
        default=None,
        help="Family for dataset (e.g., for PSPLibDataset)"
    )
    parser.add_argument(
        "--dataset-option",
        type=str,
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        help="Additional dataset options as key-value pairs. Can be specified multiple times. "
             "Example: --dataset-option transform my_transform --dataset-option target_transform my_target"
    )
    
    args = parser.parse_args()
    
    # Check that exactly one of instance, --batch, or --dataset is provided
    provided = sum([args.instance is not None, args.batch is not None, args.dataset is not None])
    if provided == 0:
        parser.error("One of 'instance', '--batch', or '--dataset' must be provided")
    elif provided > 1:
        parser.error("Only one of 'instance', '--batch', or '--dataset' can be provided")
    
    # Load the runner
    try:
        if args.runner == "xcsp3":
            # Special case for xcsp3
            from cpmpy.tools.benchmark.test.run_xcsp3_instance import XCSP3InstanceRunner
            runner = XCSP3InstanceRunner()
        else:
            runner = load_instance_runner(args.runner)
    except Exception as e:
        print(f"Error loading runner '{args.runner}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load observers
    additional_observers = None
    if args.observers:
        try:
            additional_observers = load_observers(args.observers)
        except Exception as e:
            print(f"Error loading observers: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run single instance, batch, or dataset
    if args.dataset:
        # Dataset processing
        try:
            # Build dataset kwargs from arguments
            dataset_kwargs = {}
            
            # Common parameters
            # Always set root (default is "./data")
            dataset_kwargs["root"] = args.dataset_root
            if args.dataset_download:
                dataset_kwargs["download"] = True
            
            # Year/track parameters (for XCSP3Dataset, OPBDataset, etc.)
            if args.dataset_year is not None:
                dataset_kwargs["year"] = args.dataset_year
            if args.dataset_track:
                dataset_kwargs["track"] = args.dataset_track
            
            # Variant/family parameters (for PSPLibDataset, etc.)
            if args.dataset_variant:
                dataset_kwargs["variant"] = args.dataset_variant
            if args.dataset_family:
                dataset_kwargs["family"] = args.dataset_family
            
            # Additional options from --dataset-option
            if args.dataset_option:
                for key, value in args.dataset_option:
                    # Try to convert value to appropriate type
                    try:
                        # Try int first
                        value = int(value)
                    except ValueError:
                        try:
                            # Try float
                            value = float(value)
                        except ValueError:
                            # Try bool
                            if value.lower() in ("true", "false"):
                                value = value.lower() == "true"
                            # Otherwise keep as string
                    dataset_kwargs[key] = value
            
            # Load and instantiate the dataset
            dataset = load_dataset(args.dataset, dataset_kwargs)
            
            # Get instances from dataset
            instances = []
            for instance, metadata in dataset:
                instances.append(instance)
            
            if not instances:
                print("No instances found in dataset", file=sys.stderr)
                sys.exit(1)
            
            # Compute workers and memory configuration
            workers, memory_per_worker = compute_workers_and_memory(
                args.workers, args.total_memory, args.memory_per_worker, args.ignore_memory_check
            )
            
            if args.verbose:
                # Get the actual total memory used (may have been measured)
                import psutil
                actual_total = args.total_memory if args.total_memory is not None else psutil.virtual_memory().total // (1024 * 1024)
                if args.total_memory:
                    print(f"Total memory: {args.total_memory} MiB (user-specified)")
                else:
                    print(f"Total memory: {actual_total} MiB (measured from system)")
                if memory_per_worker:
                    print(f"Memory per worker: {memory_per_worker} MiB")
                print(f"Running {len(instances)} instances from dataset with {workers} workers")
            
            run_batch(
                instances=instances,
                runner_path=args.runner,
                solver=args.solver,
                time_limit=args.time_limit,
                mem_limit=args.mem_limit if args.mem_limit is not None else memory_per_worker,
                seed=args.seed,
                workers=workers,
                cores_per_worker=args.cores_per_worker,
                total_memory=args.total_memory,
                memory_per_worker=memory_per_worker,
                ignore_memory_check=args.ignore_memory_check,
                intermediate=args.intermediate,
                verbose=args.verbose,
                output_dir=args.output,
            )
        except Exception as e:
            print(f"Error running dataset: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.batch:
        # Batch processing
        try:
            instances = parse_instance_list(args.batch)
            if not instances:
                print(f"No instances found in {args.batch}", file=sys.stderr)
                sys.exit(1)
            
            # Compute workers and memory configuration
            workers, memory_per_worker = compute_workers_and_memory(
                args.workers, args.total_memory, args.memory_per_worker, args.ignore_memory_check
            )
            
            if args.verbose:
                # Get the actual total memory used (may have been measured)
                import psutil
                actual_total = args.total_memory if args.total_memory is not None else psutil.virtual_memory().total // (1024 * 1024)
                if args.total_memory:
                    print(f"Total memory: {args.total_memory} MiB (user-specified)")
                else:
                    print(f"Total memory: {actual_total} MiB (measured from system)")
                if memory_per_worker:
                    print(f"Memory per worker: {memory_per_worker} MiB")
                print(f"Running {len(instances)} instances with {workers} workers")
            
            run_batch(
                instances=instances,
                runner_path=args.runner,
                solver=args.solver,
                time_limit=args.time_limit,
                mem_limit=args.mem_limit if args.mem_limit is not None else memory_per_worker,
                seed=args.seed,
                workers=workers,
                cores_per_worker=args.cores_per_worker,
                total_memory=args.total_memory,
                memory_per_worker=memory_per_worker,
                ignore_memory_check=args.ignore_memory_check,
                intermediate=args.intermediate,
                verbose=args.verbose,
                output_dir=args.output,
            )
        except Exception as e:
            print(f"Error running batch: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Single instance
        if not args.instance:
            parser.error("Either provide an instance path or use --batch")
        
        try:
            run_single_instance(
                instance=args.instance,
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
        except Exception as e:
            print(f"Error running instance: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

