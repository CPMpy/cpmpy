"""
Benchmark solvers on XCSP3 Instances by replicating the XCSP3 competition.

A command-line tool for benchmarking constraint solvers on XCSP3 competition instances.
Supports parallel execution, time/memory limits, and solver configuration.

Required Arguments
------------------
--year : int
    The competition year (e.g., 2023).

--track : str
    The competition track (e.g., "CSP", "COP", "MiniCOP").

--solver : str
    The name of the solver to benchmark (e.g., "ortools", "exact", "choco").

Optional Arguments
------------------
--workers : int, default=4
    Number of parallel workers to use.

--time-limit : int, default=300
    Time limit in seconds per instance.

--mem-limit : int, default=8192
    Memory limit in megabytes per instance.

--output-dir : str, default='results'
    Directory where result CSV files will be saved.

--verbose
    If set, display full xcsp3 output during execution.

--intermediate
    If set, report intermediate solutions (if supported by the solver).
"""

import csv
import os
import signal
import subprocess
import time
import lzma
import sys
import argparse
import warnings
import traceback
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
from io import StringIO
from datetime import datetime
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor

from cpmpy.tools.xcsp3.dataset import XCSP3Dataset
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


def xcsp3_wrapper(conn, kwargs, verbose):
    """
    Wraps a call to xcsp3_cpmpy as to correctly 
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
        xcsp3_cpmpy(**kwargs)
        conn.send({"status": "ok"})
    except Exception as e: # capture exceptions and report in state
        tb_str = traceback.format_exc()
        conn.send({"status": "error", "exception": e, "traceback": tb_str})
    finally:
        sys.stdout = original_stdout
        conn.close()

# exec_args = (filename, metadata, solver, time_limit, mem_limit, output_file, verbose) 
def execute_instance(args: Tuple[str, dict, str, int, int, int, str, bool, bool, str]) -> None:
    """
    Solve a single XCSP3 instance and write results to file immediately.
    
    Args is a list of:
        filename: Path to the XCSP3 instance file
        metadata: Dictionary containing instance metadata (year, track, name)
        solver: Name of the solver to use
        time_limit: Time limit in seconds
        mem_limit: Memory limit in MB
        output_file: Path to the output CSV file
        verbose: Whether to show solver output
    """
    
    filename, metadata, solver, time_limit, mem_limit, cores, output_file, verbose, intermediate, checker_path = args

    # Fieldnames for the CSV file
    fieldnames = ['year', 'track', 'instance', 'solver',
                  'time_total', 'time_parse', 'time_model', 'time_post', 'time_solve',
                  'status', 'objective_value', 'solution', 'intermediate', 'checker_result']
    result = dict.fromkeys(fieldnames)  # init all fields to None
    result['year'] = metadata['year']
    result['track'] = metadata['track']
    result['instance'] = metadata['name'] 
    result['solver'] = solver

    # Decompress before timers start
    file_path = filename
    if str(filename).endswith(".lzma"):
        # Decompress the XZ file
        with lzma.open(filename, 'rt', encoding='utf-8') as f:
            xml_file = StringIO(f.read()) # read to memory-mapped file
            filename = xml_file
            
    # Start total timing
    total_start = time.time()
    
    # Call xcsp3 in separate process
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = multiprocessing.Pipe() # communication pipe between processes
    process = ctx.Process(target=xcsp3_wrapper, args=(
                                                    child_conn, 
                                                      {
                                                          "benchname": filename, 
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
       
    sol_time = None # For annotation intermediate solutions (when they were received)
    
    # Default status if nothing returned by subprocess
    # -> process exited prematurely due to sigterm
    status = {"status": "error", "exception": "sigterm"}

    # Parse the output to get status, solution and timings
    complete_solution = None
    while parent_conn.poll(timeout=1):
        line = parent_conn.recv()

        # Received a print statement from the subprocess
        if isinstance(line, str):
            if line.startswith('s '):
                result['status'] = line[2:].strip()
            elif line.startswith('v ') and result['solution'] is None:
                # only record first line, contains 'type' and 'cost'
                solution = line.split("\n")[0][2:].strip()
                result['solution'] = str(solution)
                complete_solution = line
                if "cost" in solution:
                    result['objective_value'] = solution.split('cost="')[-1][:-2]
            elif line.startswith('o '):
                obj = int(line[2:].strip())
                if result['intermediate'] is None:
                    result['intermediate'] = []
                result['intermediate'] += [(sol_time, obj)]
                result['objective_value'] = obj
                obj = None
            elif line.startswith('c Solution'):
                parts = line.split(', time = ')
                # Get solution time from comment for intermediate solution -> used for annotating 'o ...' lines
                sol_time = float(parts[-1].replace('s', '').rstrip())
            elif line.startswith('c took '):
                # Parse timing information
                parts = line.split(' seconds to ')
                if len(parts) == 2:
                    time_val = float(parts[0].replace('c took ', ''))
                    action = parts[1].strip()
                    if action.startswith('parse'):
                        result['time_parse'] = time_val
                    elif action.startswith('convert'):
                        result['time_model'] = time_val
                    elif action.startswith('post'):
                        result['time_post'] = time_val
                    elif action.startswith('solve'):
                        result['time_solve'] = time_val
        
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

    if checker_path is not None and complete_solution is not None:
        checker_output, checker_time = run_solution_checker(
            JAR=checker_path,
            instance_location=file_path,
            out_file="'" + complete_solution.replace("\n\r", " ").replace("\n", " ").replace("v   ", "").replace("v ", "")+ "'",
            verbose=verbose,
            cpm_time=result.get('time_solve', 0)  # or total solve time you have
        )

        if checker_output is not None:
            result['checker_result'] = checker_output
        else:
            result['checker_result'] = None

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


def run_solution_checker(JAR, instance_location, out_file, verbose, cpm_time):

    start = time.time()
    command = ["java", "-jar", JAR, "'" + str(instance_location) + "'" + " " + str(out_file)]
    command = " ".join(command)
    test_res_str = subprocess.run(command, capture_output=True, text=True, shell=True)
    checker_time = time.time() - start

    if verbose:
        for line in test_res_str.stdout.split("\n"):
            print("c " + line)
        print(f"c cpmpy time: {cpm_time}")
        print(f"c validation time: {checker_time}")
        print(f"c elapsed time: {cpm_time + checker_time}")
    
    return test_res_str.stdout.split("\n")[-2], checker_time


def xcsp3_benchmark(year: int, track: str, solver: str, workers: int = 1, 
                   time_limit: int = 300, mem_limit: Optional[int] = 4096, cores: int=1,
                   output_dir: str = 'results',
                   verbose: bool = False, intermediate: bool = False,
                   checker_path: Optional[str] = None) -> str:
    """
    Benchmark a solver on XCSP3 instances.
    
    Args:
        year (int): Competition year (e.g., 2023)
        track (str): Track type (e.g., COP, CSP, MiniCOP)
        solver (str): Solver name (e.g., ortools, exact, choco, ...)
        workers (int): Number of parallel workers
        time_limit (int): Time limit in seconds per instance
        mem_limit (int): Memory limit in MB per instance
        output_dir (str): Output directory for CSV files
        verbose (bool): Whether to show solver output
        
    Returns:
        str: Path to the output CSV file
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current timestamp in a filename-safe format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output file path with timestamp
    output_file = str(output_dir / f"xcsp3_{year}_{track}_{solver}_{timestamp}.csv")
    
    # Initialize dataset
    dataset = XCSP3Dataset(year=year, track=track, download=True)

    # Process instances in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and track their futures
        futures = [executor.submit(execute_instance,  # below: args
                                   (filename, metadata, solver, time_limit, mem_limit, cores, output_file, verbose, intermediate, checker_path))
                   for filename, metadata in dataset]
        # Process results as they complete
        for i,future in enumerate(tqdm(futures, total=len(futures), desc=f"Running {solver}")):
            try:
                _ = future.result(timeout=time_limit+60)  # for cleanliness sake, result is empty
            except TimeoutError:
                pass
            except Exception as e:
                print(f"Job {i}: {dataset[i][1]['name']}, ProcessPoolExecutor caught: {e}")

        raise()
    
    return output_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, required=True, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, required=True, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds per instance')
    parser.add_argument('--mem-limit', type=int, default=8192, help='Memory limit in MB per instance')
    parser.add_argument('--cores', type=int, default=1, help='Number of cores to assign tp a single instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    parser.add_argument('--intermediate', action='store_true', help='Report on intermediate solutions')
    parser.add_argument('--checker-path', type=str, default=None,
                    help='Path to the XCSP3 solution checker JAR file')
    
    
    args = parser.parse_args()

    if not args.verbose:
        warnings.filterwarnings("ignore")
    
    output_file = xcsp3_benchmark(**vars(args))
    print(f"Results added to {output_file}")
