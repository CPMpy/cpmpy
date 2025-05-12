import argparse
import csv
import os
import io
import time
import lzma
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Tuple
from io import StringIO
import sys
from datetime import datetime
from tqdm import tqdm
import concurrent.futures

import cpmpy
from cpmpy.tools.xcsp3.xcsp3_dataset import XCSP3Dataset
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from cpmpy.tools.xcsp3.parser_callbacks import CallbacksCPMPy
from cpmpy.tools.xcsp3.xcsp3_solution import solution_xml
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.xcsp3.xcsp3_cpmpy import xcsp3_cpmpy, ExitStatus

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
    filename, metadata, solver, time_limit, mem_limit, output_file, verbose = args

    # Fieldnames for the CSV file
    fieldnames = ['year', 'track', 'instance', 'solver',
                  'time_total', 'time_parse', 'time_model', 'time_post', 'time_solve',
                  'is_sat', 'objective_value', 'solution']
    result = dict.fromkeys(fieldnames)  # init all fields to None
    result['year'] = metadata['year']
    result['track'] = metadata['track']
    result['instance'] = metadata['name'] 
    result['solver'] = solver
            
    # Start total timing
    total_start = time.time()

    try:
        # Decompress the XZ file
        # TODO: should be tmp file or in mem or?
        with lzma.open(filename, 'rt', encoding='utf-8') as f:
            xml_file = io.StringIO(f.read()) # read to memory-mapped file
                
        # Capture stdout to prevent xcsp3_cpmpy from printing if not verbose
        captured_output = StringIO()
        original_stdout = sys.stdout
        if not verbose:
            sys.stdout = captured_output
        
        try:
            # Call xcsp3_cpmpy with the solver and limits
            xcsp3_cpmpy(xml_file, solver=solver, time_limit=time_limit, mem_limit=mem_limit, cores=1)
            
            # Get the output and restore stdout if not verbose
            if not verbose:
                output = captured_output.getvalue()
                sys.stdout = original_stdout
                
                # Parse the output to get status, solution and timings
                status = None
                
                for line in output.split('\n'):
                    if line.startswith('s '):
                        status = line[2:].strip()
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
                
                # Map status to is_sat
                # TODO: rename field to 'status' and use a str name of the status
                if status == ExitStatus.sat.value or status == ExitStatus.optimal.value:
                    result['is_sat'] = True
                elif status == ExitStatus.unsat.value:
                    result['is_sat'] = False
                else:
                    result['is_sat'] = None
                
        finally:
            result['time_total'] = time.time() - total_start
            # Restore stdout in case of exception
            if not verbose:
                sys.stdout = original_stdout
        
    except Exception as e:
        result['is_sat'] = False
        result['solution'] = str(e)  # abuse solution field for error message
        result['time_total'] = time.time() - total_start
    
    # Use a lock file to prevent concurrent writes
    lock_file = f"{output_file}.lock"
    while os.path.exists(lock_file):
        time.sleep(0.1)
    
    try:
        # Create lock file
        open(lock_file, 'w').close()
            
        # Pre-check if file exists to determine if we need to write header
        write_header = not os.path.exists(output_file)
        
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)
    finally:
        # Remove lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)

def xcsp3_benchmark(year: int, track: str, solver: str, workers: int = 1, 
                   time_limit: int = 300, mem_limit: int = 4096, output_dir: str = 'results',
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
        futures = [executor.submit(execute_instance,  # below: args
                                   (filename, metadata, solver, time_limit, mem_limit, output_file, verbose))
                   for filename, metadata in dataset]
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {solver}"):
            try:
                _ = future.result()  # for cleanliness sake, result is empty
            except Exception as e:
                print(f"ProcessPoolExecutor caught: {e}")
    
    return output_file

if __name__ == "__main__":
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
