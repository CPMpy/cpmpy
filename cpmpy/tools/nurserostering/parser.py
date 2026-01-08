"""
Parser for the Nurse Rostering format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_nurserostering
"""


import os
import sys
import argparse
import tempfile
import cpmpy as cp
from typing import Union

from cpmpy.tools.dataset.problem.nurserostering import (
    parse_scheduling_period,
    nurserostering_model
)


_std_open = open
def read_nurserostering(instance: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Parser for Nurse Rostering format. Reads in an instance and returns its matching CPMpy model.

    Arguments: 
        instance (str or os.PathLike):
            - A file path to a Nurse Rostering file
            - OR a string containing the Nurse Rostering content directly
        open (callable):
            If instance is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the Nurse Rostering instance.
    """
    # If instance is a path to a file that exists -> use it directly
    if isinstance(instance, (str, os.PathLike)) and os.path.exists(instance):
        fname = instance
    # If instance is a string containing file content -> write to temp file
    else:
        # Create a temporary file and write the content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write(instance)
            fname = tmp.name

    try:
        # Use the existing parser from the dataset (expects a file path)
        data = parse_scheduling_period(fname)
        
        # Create the CPMpy model using the existing model builder
        model, _ = nurserostering_model(**data)
        
        return model
    finally:
        # Clean up temporary file if we created one
        if isinstance(instance, str) and not os.path.exists(instance) and os.path.exists(fname):
            os.unlink(fname)


def main():
    parser = argparse.ArgumentParser(description="Parse and solve a Nurse Rostering model using CPMpy")
    parser.add_argument("model", help="Path to a Nurse Rostering file (or raw content string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw Nurse Rostering string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_nurserostering(args.model)
        else:
            model = read_nurserostering(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        sys.exit(1)

    # Solve the model
    try:
        if args.solver:
            result = model.solve(solver=args.solver, time_limit=args.time_limit)
        else:
            result = model.solve(time_limit=args.time_limit)
    except Exception as e:
        sys.stderr.write(f"Error solving model: {e}\n")
        sys.exit(1)

    # Print results
    print("Status:", model.status())
    if result is not None:
        if model.has_objective():
            print("Objective:", model.objective_value())
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()

