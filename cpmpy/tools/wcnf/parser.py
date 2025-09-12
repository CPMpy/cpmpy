"""
Parser for the WCNF format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_wcnf
"""


import os
import sys
import lzma
import argparse
import cpmpy as cp
from io import StringIO
from typing import Union


def _get_var(i, vars_dict):
    """
    Returns CPMpy boolean decision variable matching to index `i` if exists, else creates a new decision variable.

    Arguments:
        i: index
        vars_dict (dict): dictionary to keep track of previously generated decision variables
    """
    if i not in vars_dict:
        vars_dict[i] = cp.boolvar(name=f"x{i}") # <- be carefull that name doesn't clash with generated variables during transformations / user variables
    return vars_dict[i]

_std_open = open
def read_wcnf(wcnf: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Parser for WCNF format. Reads in an instance and returns its matching CPMpy model.

    Arguments: 
        wcnf (str or os.PathLike):
            - A file path to an WCNF file (optionally LZMA-compressed with `.xz`)
            - OR a string containing the WCNF content directly
        open: (callable):
            If wcnf is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the WCNF instance.
    """
    # If wcnf is a path to a file -> open file
    if isinstance(wcnf, (str, os.PathLike)) and os.path.exists(wcnf):
        if open is not None:
            f = open(wcnf)
        else:
            f = _std_open(wcnf, "rt")
    # If wcnf is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(wcnf)

    model = cp.Model()
    vars = {}
    soft_terms = []

    for raw in f:
        line = raw.strip()

        # Empty line or a comment -> skip
        if not line or line.startswith("c"):
            continue

        # Hard clause
        if line[0] == "h":
            literals = map(int, line[1:].split())
            clause = [_get_var(i, vars) if i > 0 else ~_get_var(-i, vars)
                      for i in literals if i != 0]
            model.add(cp.any(clause))

        # Soft clause (weight first)
        else:
            parts = line.split()
            weight = int(parts[0])
            literals = map(int, parts[1:])
            clause = [_get_var(i, vars) if i > 0 else ~_get_var(-i, vars)
                    for i in literals if i != 0]
            soft_terms.append(weight * cp.any(clause))

    # Objective = sum of soft clause terms
    if soft_terms:
        model.maximize(sum(soft_terms))

    return model

def main():
    parser = argparse.ArgumentParser(description="Parse and solve a WCNF model using CPMpy")
    parser.add_argument("model", help="Path to a WCNF file (or raw WCNF string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw WCNF string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_wcnf(args.model)
        else:
            model = read_wcnf(os.path.expanduser(args.model))
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