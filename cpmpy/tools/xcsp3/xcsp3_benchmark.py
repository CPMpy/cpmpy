import argparse
import csv
import os
import time
import lzma
from concurrent.futures import ProcessPoolExecutor
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

# exec_args = (filename, metadata, solver, timeout, output_file, verbose) 
def execute_instance(args: Tuple[str, dict, str, int, str, bool]) -> None:
    """
    Solve a single XCSP3 instance and write results to file immediately.
    
    Args is a list of:
        filename: Path to the XCSP3 instance file
        metadata: Dictionary containing instance metadata (year, track, name)
        solver: Name of the solver to use
        timeout: Timeout in seconds
        output_file: Path to the output CSV file
        verbose: Whether to show solver output
    """
    filename, metadata, solver, timeout, output_file, verbose = args

    # Fieldnames for the CSV file
    fieldnames = ['year', 'track', 'instance', 'solver',
                  'time_total', 'time_parse', 'time_model', 'time_post', 'time_solve',
                  'is_sat', 'objective_value', 'solution']
    result = dict.fromkeys(fieldnames)  # init all fields to None
    result['year'] = metadata['year']
    result['track'] = metadata['track']
    result['instance'] = metadata['name'] 
    result['solver'] = solver

    try:
        # Decompress the XZ file
        with lzma.open(filename, 'rb') as f:
            xml_content = f.read().decode('utf-8')
        
        # Create a temporary file with the decompressed content
        temp_file = filename + '.xml'
        with open(temp_file, 'w') as f:
            f.write(xml_content)
            
        # Start total timing
        total_start = time.time()
        
        # Capture stdout to prevent xcsp3_cpmpy from printing if not verbose
        captured_output = StringIO()
        original_stdout = sys.stdout
        if not verbose:
            sys.stdout = captured_output
        
        try:
            # Call xcsp3_cpmpy with the solver and timeout
            xcsp3_cpmpy(temp_file, solver=solver, time_limit=timeout, cores=1)
            
            # Get the output and restore stdout if not verbose
            if not verbose:
                output = captured_output.getvalue()
                sys.stdout = original_stdout
                
                # Parse the output to get status, solution and timings
                status = None
                solution = None
                objective = None
                
                for line in output.split('\n'):
                    if line.startswith('s '):
                        status = line[2:].strip()
                    elif line.startswith('v '):
                        solution = line[2:].strip()
                    elif line.startswith('o '):
                        objective = int(line[2:].strip())
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
                if status == ExitStatus.sat.value or status == ExitStatus.optimal.value:
                    result['is_sat'] = True
                elif status == ExitStatus.unsat.value:
                    result['is_sat'] = False
                else:
                    result['is_sat'] = None
                    
                # Set solution and objective if found
                if solution:
                    result['solution'] = solution
                if objective is not None:
                    result['objective_value'] = objective
                
        finally:
            result['time_total'] = time.time() - total_start
            # Restore stdout in case of exception
            if not verbose:
                sys.stdout = original_stdout
            
        # Clean up temporary file
        os.remove(temp_file)
        
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

def xcsp3_benchmark(year: int, track: str, solver: str, threads: int = 1, 
                   timeout: int = 300, output_dir: str = 'results',
                   verbose: bool = False) -> str:
    """
    Benchmark a solver on XCSP3 instances.
    
    Args:
        year (int): Competition year (e.g., 2023)
        track (str): Track type (e.g., COP, CSP, MiniCOP)
        solver (str): Solver name (e.g., ortools, exact, choco, ...)
        threads (int): Number of parallel threads
        timeout (int): Timeout in seconds per instance
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
    with ProcessPoolExecutor(max_workers=threads) as executor:
        # Submit all tasks and track their futures
        futures = [executor.submit(execute_instance,  # below: args
                                   (filename, metadata, solver, timeout, output_file, verbose))
                   for filename, metadata in dataset]
        # Process results as they complete
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Running {solver}"):
            pass
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, required=True, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, required=True, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds per instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    
    args = parser.parse_args()
    
    output_file = xcsp3_benchmark(**vars(args))
    print(f"Results added to {output_file}")
