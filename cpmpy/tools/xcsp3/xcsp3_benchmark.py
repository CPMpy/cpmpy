import argparse
import csv
import multiprocessing
import os
import io
import time
import lzma
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import StringIO
import sys
from datetime import datetime
import warnings
from tqdm import tqdm
import concurrent.futures
import traceback
from filelock import FileLock

import cpmpy
from cpmpy.tools.xcsp3.xcsp3_dataset import XCSP3Dataset
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from cpmpy.tools.xcsp3.parser_callbacks import CallbacksCPMPy
from cpmpy.tools.xcsp3.xcsp3_solution import solution_xml
from cpmpy.tools.xcsp3.xcsp3_cpmpy import xcsp3_cpmpy, ExitStatus

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

# exec_args = (filename, metadata, solver, time_limit, mem_limit, output_file, verbose) 
def execute_instance(args: Tuple[str, dict, str, int, int, str, bool]) -> None:
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
    warnings.filterwarnings("ignore")
    
    filename, metadata, solver, time_limit, mem_limit, output_file, verbose = args

    # Fieldnames for the CSV file
    fieldnames = ['year', 'track', 'instance', 'solver',
                  'time_total', 'time_parse', 'time_model', 'time_post', 'time_solve',
                  'status', 'objective_value', 'solution']
    result = dict.fromkeys(fieldnames)  # init all fields to None
    result['year'] = metadata['year']
    result['track'] = metadata['track']
    result['instance'] = metadata['name'] 
    result['solver'] = solver
            
    # Start total timing
    total_start = time.time()

    try:
        # Decompress the XZ file
        with lzma.open(filename, 'rt', encoding='utf-8') as f:
            xml_file = io.StringIO(f.read()) # read to memory-mapped file
                
        # Capture stdout for output extranction
        captured_output = StringIO()
        original_stdout = sys.stdout
        if not verbose:
            # prevent xcsp3_cpmpy from printing if not verbose
            sys.stdout = captured_output
        else:
            # print to original stdout and captured_output
            sys.stdout = Tee(original_stdout, captured_output)
        
        try:
            # Call xcsp3_cpmpy with the solver and limits
            xcsp3_cpmpy(xml_file, solver=solver, time_limit=time_limit, mem_limit=mem_limit, cores=1)
            xml_file.close()  # Explicitly close the StringIO object
                            
            # Get the output and restore stdout
            output = captured_output.getvalue()
            sys.stdout = original_stdout
            
            # Parse the output to get status, solution and timings
            for line in output.split('\n'):
                if line.startswith('s '):
                    result['status'] = line[2:].strip()
                elif line.startswith('v '):
                    result['solution'] = line[2:].strip()
                elif line.startswith('o '):
                    result['objective_value'] = int(line[2:].strip())
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
            
        except Exception as e:
            raise e
                
        finally:
            # Restore stdout in case of exception
            if not verbose:
                sys.stdout = original_stdout
            captured_output.close()  # Close the captured output StringIO
        
    except Exception as e:
        result['status'] = ExitStatus.unknown
        result['solution'] = str(e)  # abuse solution field for error message

    result['time_total'] = time.time() - total_start

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


def run_with_timeout(func, args, timeout):
    def wrapper(queue):
        try:
            result = func(args)
            queue.put((True, result))
        except Exception as e:
            queue.put((False, traceback.format_exc()))

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(queue,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("Function timed out")
    
    success, result = queue.get()
    if not success:
        raise RuntimeError(f"Function raised exception:\n{result}")
    return result
    
def submit_wrapped(filename, metadata, solver, time_limit, mem_limit, output_file, verbose):
    return run_with_timeout(execute_instance, 
                            (filename, metadata, solver, time_limit, mem_limit, output_file, verbose),
                            timeout=time_limit*2) # <- change limit as needed, now a very gracious doubling of the time

def xcsp3_benchmark(year: int, track: str, solver: str, workers: int = 1, 
                   time_limit: int = 300, mem_limit: Optional[int] = 4096, output_dir: str = 'results',
                   verbose: bool = False) -> str:
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
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and track their futures
        futures = [executor.submit(submit_wrapped,  # below: args
                                   filename, metadata, solver, time_limit, mem_limit, output_file, verbose)
                   for filename, metadata in dataset]
        # Process results as they complete
        for i,future in enumerate(tqdm(futures, total=len(futures), desc=f"Running {solver}")):
            try:
                _ = future.result(timeout=10)  # for cleanliness sake, result is empty
            except TimeoutError:
                print(f"Timeout on job {i}: {dataset[i][1]['name']}")  # print the metadata
            except Exception as e:
                print(f"Job {i}: {dataset[i][1]['name']}, ProcessPoolExecutor caught: {e}")
    
    return output_file

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, required=True, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, required=True, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds per instance')
    parser.add_argument('--mem-limit', type=int, default=4096, help='Memory limit in MB per instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    
    args = parser.parse_args()
    
    output_file = xcsp3_benchmark(**vars(args))
    print(f"Results added to {output_file}")
