"""
Parser for the XCSP3 format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_xcsp3

========================
List of helper functions
========================

.. autosummary::
    :nosignatures:

    _parse_xcsp3
    _load_xcsp3
"""

import os
import sys
import argparse
from io import StringIO

import cpmpy as cp

# Special case for optional cpmpy dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycsp3.parser.xparser import ParserXCSP3

def _parse_xcsp3(path: os.PathLike) -> "ParserXCSP3":
    """
    Parses an XCSP3 instance file (.xml) and returns a `ParserXCSP3` instance.
    
    Arguments:
        path: location of the XCSP3 instance to read (expects a .xml file).
    
    Returns:
        A parser object.
    """
    try:
        from pycsp3.parser.xparser import ParserXCSP3
    except ImportError as e:
        raise ImportError("The 'pycsp3' package is required to parse XCSP3 files. "
                          "Please install it with `pip install pycsp3`.") from e
    
    parser = ParserXCSP3(path)
    return parser

def _load_xcsp3(parser: "ParserXCSP3") -> cp.Model:
    """
    Takes in a `ParserXCSP3` instance and loads its captured model as a CPMpy model.

    Arguments:
        parser (ParserXCSP3): A parser object to load from.

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    from .parser_callbacks import CallbacksCPMPy
    from pycsp3.parser.xparser import CallbackerXCSP3
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)
    callbacker.load_instance()
    model = callbacks.cpm_model
   
    return model

_std_open = open
def read_xcsp3(xcsp3: os.PathLike, open=open) -> cp.Model:
    """
    Reads in an XCSP3 instance (.xml or .xml.lzma) and returns its matching CPMpy model.

    Arguments:
        xcsp3 (str or os.PathLike):
            - A file path to an WCNF file (optionally LZMA-compressed with `.lzma`)
            - OR a string containing the WCNF content directly
        open: (callable):
            If wcnf is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    # If wcnf is a path to a file -> open file
    if isinstance(xcsp3, (str, os.PathLike)) and os.path.exists(xcsp3):
        if open is not None:
            f = open(xcsp3)
        else:
            f = _std_open(xcsp3, "rt")
    # If wcnf is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(xcsp3)

    # Parse and create CPMpy model
    parser = _parse_xcsp3(f)
    model = _load_xcsp3(parser)
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
            model = read_xcsp3(args.model)
        else:
            model = read_xcsp3(os.path.expanduser(args.model))
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
    