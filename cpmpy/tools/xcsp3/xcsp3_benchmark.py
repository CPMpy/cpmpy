import argparse
import csv
import multiprocessing
import os
import io
import time
import lzma
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import StringIO
import sys
from datetime import datetime
import warnings
from tqdm import tqdm
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

def solve_instance(args: Tuple[str, Dict[str, Any], str, int, int, str, bool]) -> Dict[str, Any]:
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
    original_stdout = sys.stdout
    
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
        # Capture output in a StringIO
        output_buffer = StringIO()
        if verbose:
            sys.stdout = Tee(original_stdout, output_buffer)
        else:
            sys.stdout = output_buffer

        # Run the solver
        xcsp3_cpmpy(benchname=filename, solver=solver, time_limit=time_limit, mem_limit=mem_limit)
        
        # Parse the output
        output = output_buffer.getvalue()
        for line in output.split('\n'):
            if line.startswith('s '):
                result['status'] = line[2:].strip()
            elif line.startswith('v ') and result['solution'] is None:
                result['solution'] = line[2:].strip()
            elif line.startswith('o '):
                result['objective_value'] = int(line[2:].strip())
            elif line.startswith('c took '):
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
        if "TimeoutError" not in repr(e):
            result['status'] = ExitStatus.unknown.value
            result["solution"] = str(e)
    finally:
        sys.stdout = original_stdout
        result['time_total'] = time.time() - total_start

    # Write results to file with locking
    lock_file = f"{output_file}.lock"
    lock = FileLock(lock_file)
    try:
        with lock:
            write_header = not os.path.exists(output_file)
            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass

    return result

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

    # Prepare arguments for each instance
    args_list = [(filename, metadata, solver, time_limit, mem_limit, output_file, verbose)
                 for filename, metadata in dataset]

    # Use multiprocessing Pool to process instances in parallel
    with multiprocessing.Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(solve_instance, args_list), 
                 total=len(args_list), 
                 desc=f"Running {solver}"))
    
    return output_file

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, required=True, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, required=True, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds per instance')
    parser.add_argument('--mem-limit', type=int, default=8192, help='Memory limit in MB per instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    
    args = parser.parse_args()
    
    output_file = xcsp3_benchmark(**vars(args))
    print(f"Results added to {output_file}")
