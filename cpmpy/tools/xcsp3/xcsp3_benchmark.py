import argparse
import csv
import os
import time
import lzma
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, Tuple

import cpmpy
from cpmpy.tools.xcsp3.xcsp3_dataset import XCSP3Dataset
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from cpmpy.tools.xcsp3.parser_callbacks import CallbacksCPMPy
from cpmpy.tools.xcsp3.xcsp3_solution import solution_xml
from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus

def solve_instance(filename: str, metadata: dict, solver: str, timeout: int, output_file: str) -> None:
    """
    Solve a single XCSP3 instance and write results to file immediately.
    
    Args:
        filename: Path to the XCSP3 instance file
        metadata: Dictionary containing instance metadata (year, track, name)
        solver: Name of the solver to use
        timeout: Timeout in seconds
        output_file: Path to the output CSV file
    """
    # Fieldnames for the CSV file
    fieldnames = ['year', 'track', 'instance', 'solver',
                  'time_total', 'time_parse', 'time_model', 'time_solve',
                  'is_sat', 'objective_value', 'solution']
    result = dict.fromkeys(fieldnames)  # init all fields to None
    result['year'] = metadata['year']
    result['track'] = metadata['track']
    result['instance'] = metadata['name'] 
    result['solver'] = solver
    print(f"Running {solver} on {filename}...", flush=True)

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
    
        # Parse the XCSP3 model
        t_parse = time.time()
        parser = ParserXCSP3(temp_file)
        callbacks = CallbacksCPMPy()
        callbacks.force_exit = True
        callbacker = CallbackerXCSP3(parser, callbacks)
        callbacker.load_instance()
        
        # Clean up temporary file
        os.remove(temp_file)
        
        result['time_parse'] = time.time() - t_parse

        # Create the solver and post the constraints
        t_model = time.time()
        model = callbacks.cpm_model
        s = cpmpy.SolverLookup.get(solver, model)
        result['time_model'] = time.time() - t_model
        
        # Solve the model
        t_solve = time.time()
        if solver != "ortools":
            result['is_sat'] = s.solve(time_limit=timeout)
        else:
            result['is_sat'] = s.solve(time_limit=timeout, num_search_workers=1)
        result['time_solve'] = time.time() - t_solve
        
        if result['is_sat']:
            # Get objective value if available
            if model.objective_ is not None:
                result['objective_value'] = model.objective_value()

            # Get solution if found
            result['solution'] = solution_xml(callbacks.cpm_variables, s).replace('\n', '  ')
        
        result['time_total'] = time.time() - total_start
        
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

def _solve_instance_wrapper(args):
    """Wrapper function to unpack arguments for solve_instance."""
    return solve_instance(**args)

def xcsp3_benchmark(year: int, track: str, solver: str, threads: int = 1, 
                   timeout: int = 300, output_dir: str = 'results') -> str:
    """
    Benchmark a solver on XCSP3 instances.
    
    Args:
        year (int): Competition year (e.g., 2023)
        track (str): Track type (e.g., COP, CSP, MiniCOP)
        solver (str): Solver name (e.g., ortools, exact, choco, ...)
        threads (int): Number of parallel threads
        timeout (int): Timeout in seconds per instance
        output_dir (str): Output directory for CSV files
        
    Returns:
        str: Path to the output CSV file
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file path
    output_file = str(output_dir / f"xcsp3_{year}_{track}_{solver}.csv")
    
    # Initialize dataset
    dataset = XCSP3Dataset(year=year, track=track, download=True)
    
    # Process instances in parallel
    with ProcessPoolExecutor(max_workers=threads) as executor:
        # Prepare arguments for parallel processing
        solve_args = [{'filename': filename, 'metadata': metadata, 'solver': solver, 
                      'timeout': timeout, 'output_file': output_file} 
                     for filename, metadata in dataset]
        # Pass each dictionary as a single argument to solve_instance and collect results
        list(executor.map(_solve_instance_wrapper, solve_args))
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark solvers on XCSP3 instances')
    parser.add_argument('--year', type=int, required=True, help='Competition year (e.g., 2023)')
    parser.add_argument('--track', type=str, required=True, help='Track type (e.g., COP, CSP, MiniCOP)')
    parser.add_argument('--solver', type=str, required=True, help='Solver name (e.g., ortools, exact, choco, ...)')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds per instance')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    output_file = xcsp3_benchmark(**vars(args))
    print(f"Results added to {output_file}")
