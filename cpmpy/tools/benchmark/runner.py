"""
Benchmark Runner for CPMpy Instances

This module provides tools to execute benchmark instances in parallel while
safely capturing solver output, enforcing time and memory limits, and
writing structured results to a CSV file. The included functions should not
be used directly, but rather through one of the available benchmarks.

Key Features
------------
- Supports running multiple instances in parallel using threads.
- Executes each instance in a separate subprocess for isolation.
- Forwards stdout to both console and parent process, preserving output.
- Handles timeouts and SIGTERM/SIGKILL signals gracefully.
- Writes results to a CSV file.
- Optional reporting of intermediate solutions and solution checking.
"""

import csv
from io import StringIO
import os
import signal
import time
import sys
import warnings
import traceback
import multiprocessing
from tqdm import tqdm
from typing import Optional, Tuple
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor

from cpmpy.tools.xcsp3.xcsp3_cpmpy import xcsp3_cpmpy, init_signal_handlers, ExitStatus

class Tee:
    """
    A stream-like object that duplicates writes to multiple underlying streams.
    """
    def __init__(self, *streams):
        """
        Arguments:
            *streams: Any number of file-like objects that implement a write() method,
                      such as sys.stdout, sys.stderr, or StringIO.
        """
        self.streams = streams

    def write(self, data):
        """
        Write data to all underlying streams.

        Args:
            data (str): The string to write.
        """
        for s in self.streams:
            s.write(data)

    def flush(self):
        """
        Flush all underlying streams to ensure all data is written out.
        """
        for s in self.streams:
            s.flush()

class PipeWriter:
    """
    Stdout wrapper for a multiprocessing pipe.
    """
    def __init__(self, conn):
        self.conn = conn
    def write(self, data):
        if data:  # avoid empty writes
            try:
                self.conn.send(data)
            except:
                pass
    def flush(self):
        pass  # no buffering


def wrapper(instance_runner, conn, kwargs, verbose):
    """
    Wraps a call to a benchmark as to correctly 
    forward stdout to the multiprocessing pipe (conn).
    Also sends a last status report though the pipe.

    Status report can be missing when process has been terminated by a SIGTERM.
    """
    
    original_stdout = sys.stdout
    pipe_writer = PipeWriter(conn)

    if not verbose:
        warnings.filterwarnings("ignore")
        sys.stdout = pipe_writer # only forward to pipe
    else:
        sys.stdout = Tee(original_stdout, pipe_writer) # forward to pipe and console

    try:
        init_signal_handlers() # configure OS signal handlers
        instance_runner.run(**kwargs)
        conn.send({"status": "ok"})
    except Exception as e: # capture exceptions and report in state
        tb_str = traceback.format_exc()
        conn.send({"status": "error", "exception": e, "traceback": tb_str})
    finally:
        sys.stdout = original_stdout
        conn.close()

# exec_args = (instance_runner, filename, metadata, open, solver, time_limit, mem_limit, output_file, verbose) 
def execute_instance(args: Tuple[callable, str, dict, callable, str, int, int, int, str, bool, bool, str]) -> None:
    """
    Solve a single benchmark instance and write results to file immediately.
    
    Args is a list of:
        filename: Path to the instance file
        metadata: Dictionary containing instance metadata (year, track, name)
        solver: Name of the solver to use
        time_limit: Time limit in seconds
        mem_limit: Memory limit in MB
        output_file: Path to the output CSV file
        verbose: Whether to show solver output
    """
    
    instance_runner, filename, metadata, open, solver, time_limit, mem_limit, cores, output_file, verbose, intermediate, checker_path = args

    # Fieldnames for the CSV file
    fieldnames = ['instance'] + list(metadata.keys()) + \
                 ['solver',
                  'time_total', 'time_parse', 'time_model', 'time_post', 'time_solve',
                  'status', 'objective_value', 'solution', 'intermediate', 'checker_result']
    result = dict.fromkeys(fieldnames)  # init all fields to None
    for k in metadata.keys():
        result[k] = metadata[k]
    result['solver'] = solver

    # Decompress before timers start
    with open(filename) as f:   # <- dataset-specific 'open' callable
        filename = StringIO(f.read()) # read to memory-mapped file

    # Start total timing
    total_start = time.time()
    
    # Call xcsp3 in separate process
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = multiprocessing.Pipe() # communication pipe between processes
    process = ctx.Process(target=wrapper, args=(
                                                    instance_runner,
                                                    child_conn, 
                                                      {
                                                          "instance": filename, 
                                                          "solver": solver, 
                                                          "time_limit": time_limit, 
                                                          "mem_limit": mem_limit, 
                                                          "intermediate": intermediate, 
                                                          "force_mem_limit": True,
                                                          "time_buffer": 1,
                                                          "cores": cores,
                                                        }, 
                                                    verbose))
    process.start()
    process.join(timeout=time_limit)

    # Replicate competition convention on how jobs get terminated
    if process.is_alive():
        # Send sigterm to let process know it reached its time limit
        os.kill(process.pid, signal.SIGTERM)
        # 1 second grace period
        process.join(timeout=1)
        # Kill if still alive
        if process.is_alive():
            os.kill(process.pid, signal.SIGKILL)
            process.join()

    result['time_total'] = time.time() - total_start
          
    # Default status if nothing returned by subprocess
    # -> process exited prematurely due to sigterm
    status = {"status": "error", "exception": "sigterm"}

    # Parse the output to get status, solution and timings
    while parent_conn.poll(timeout=1):
        line = parent_conn.recv()

        # Received a print statement from the subprocess
        if isinstance(line, str):
            instance_runner.parse_output_line(line, result)
        
        # Received a new status from the subprocess
        elif isinstance(line, dict):
            status = line

        else:
            raise()

    # Parse the exit status
    if status["status"] == "error":
        # Ignore timeouts
        if "TimeoutError" in repr(status["exception"]):
            pass
        # All other exceptions, put in solution field
        elif result['solution'] is None:
            result['status'] = ExitStatus.unknown.value
            result["solution"] = status["exception"]    

    # if checker_path is not None and complete_solution is not None: TODO: generalise 'checkers' for benchmarks
    #     checker_output, checker_time = run_solution_checker(
    #         JAR=checker_path,
    #         instance_location=file_path,
    #         out_file="'" + complete_solution.replace("\n\r", " ").replace("\n", " ").replace("v   ", "").replace("v ", "")+ "'",
    #         verbose=verbose,
    #         cpm_time=result.get('time_solve', 0)  # or total solve time you have
    #     )

    #     if checker_output is not None:
    #         result['checker_result'] = checker_output
    #     else:
    #         result['checker_result'] = None

    # Use a lock file to prevent concurrent writes
    lock_file = f"{output_file}.lock"
    lock = FileLock(lock_file)
    try:
        with lock:
            # Pre-check if file exists to determine if we need to write header
            write_header = not os.path.exists(output_file)

            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
    finally:
        # Optional: cleanup if the lock file somehow persists
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass  # avoid crashing on cleanup



def benchmark_runner(
        dataset, instance_runner,
        output_file: str,
        solver: str, workers: int = 1, 
        time_limit: int = 300, mem_limit: Optional[int] = 4096, cores: int=1,
        verbose: bool = False, intermediate: bool = False,
        checker_path: Optional[str] = None,
        **kwargs
    ) -> str:
    """
    Run a benchmark over all instances in a dataset using multiple threads.

    Arguments:
        dataset (_Dataset):             Dataset object containing instances to benchmark.
        instance_runner (Benchmark):    Benchmark runner that implements the run() method.
        output_file (str):              Path to the CSV file where results will be stored.
        solver (str):                   Name of the solver to use.
        workers (int):                  Number of parallel processes to run instances (default=1).
        time_limit (int):               Time limit in seconds for each instance (default=300).
        mem_limit (int, optional):      Memory limit in MB per instance (default=4096).
        cores (int):                    Number of CPU cores assigned per instance (default=1).
        verbose (bool):                 Whether to show solver output in stdout (default=False).
        intermediate (bool):            Whether to report intermediate solutions if supported (default=False).
        checker_path (str, optional):   Path to a solution checker for validating instance solutions.
        **kwargs:                       Additional arguments passed to `execute_instance`.

    Returns:
        str: Path to the CSV file where benchmark results were written.
    """

    # Process instances in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and track their futures
        futures = [executor.submit(execute_instance,  # below: args
                                   (instance_runner, filename, metadata, dataset.open, solver, time_limit, mem_limit, cores, output_file, verbose, intermediate, checker_path))
                   for filename, metadata in dataset]
        # Process results as they complete
        for i, future in enumerate(tqdm(futures, total=len(futures), desc=f"Running {solver}")):
            try:
                _ = future.result(timeout=time_limit + 60)  # for cleanliness sake, result is empty
            except TimeoutError:
                pass
            except Exception as e:
                print(f"Job {i}: {dataset[i][1]['name']}, ProcessPoolExecutor caught: {e}")
                if verbose: traceback.print_exc()
    
    return output_file
