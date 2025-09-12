#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
OPB parser.

Currently only the restricted OPB PB24 format is supported (without WBO).


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_opb
"""


import os
import re
import sys
import lzma
import argparse
import cpmpy as cp
from io import StringIO
from typing import Union
from functools import reduce
from operator import mul

# Regular expressions
HEADER_RE = re.compile(r'(.*)\s*#variable=\s*(\d+)\s*#constraint=\s*(\d+).*')
TERM_RE = re.compile(r"([+-]?\d+)((?:\s+~?x\d+)+)")
OBJ_TERM_RE = re.compile(r'^min:')
IND_TERM_RE = re.compile(r'([>=|<=|=]+)\s+([+-]?\d+)')
IND_TERM_RE = re.compile(r'(>=|<=|=)\s*([+-]?\d+)')


def _parse_term(line, vars):
    """
    Parse a line containing OPB terms into a CPMpy expression.

    Supports:
        - Linear terms (e.g., +2 x1)
        - Non-linear terms (e.g., -1 x1 x14)
        - Negated variables using '~' (e.g., ~x5)

    Arguments:
        line (str):                 A string containing one or more terms.
        vars (list[cp.boolvar]):    List or array of CPMpy Boolean variables.

    Returns:
        cp.Expression: A CPMpy expression representing the sum of all parsed terms.

    Example:
        >>> _parse_term("2 x2 x3 +3 x4 ~x5", vars)
        sum([2, 3] * [(IV2*IV3), (IV4*~IV5)])
    """

    terms = []
    for w, vars_str in TERM_RE.findall(line):
        factors = []

        for v in vars_str.split():
            if v.startswith("~x"):
                idx = int(v[2:]) # remove "~x"
                factors.append(~vars[idx])
            else:
                idx = int(v[1:]) # remove "x"
                factors.append(vars[idx])
        
        term = int(w) * reduce(mul, factors, 1) # create weighted term
        terms.append(term)

    return cp.sum(terms)

def _parse_constraint(line, vars):
    """
    Parse a single OPB constraint line into a CPMpy comparison expression.

    Arguments:
        line (str):                 A string representing a single OPB constraint.
        vars (list[cp.boolvar]):    List or array of CPMpy Boolean variables. Will be index to get the variables for the constraint.

    Returns:
        cp.expressions.core.Comparison: A CPMpy comparison expression representing
                                        the constraint.

    Example:
        >>> _parse_constraint("-1 x1 x14 -1 x1 ~x17 >= -1", vars)
        sum([-1, -1] * [(IV1*IV14), (IV1*~IV17)]) >= -1
    """

    op, ind_term = IND_TERM_RE.search(line).groups()
    lhs = _parse_term(line, vars)

    rhs = int(ind_term) if ind_term.lstrip("+-").isdigit() else vars[int(ind_term)]

    return cp.expressions.core.Comparison(
        name="==" if op == "=" else ">=",
        left=lhs,
        right=rhs
    )

def read_opb(opb: Union[str, os.PathLike]) -> cp.Model:
    """
    Parser for OPB (Pseudo-Boolean) format. Reads in an instance and returns its matching CPMpy model.

    Based on PyPBLib's example parser: https://hardlog.udl.cat/static/doc/pypblib/html/library/index.html#example-from-opb-to-cnf-file

    Supports:
        - Linear and non-linear terms (e.g., -1 x1 x14 +2 x2)
        - Negated variables using '~' (e.g., ~x5)
        - Minimisation objective
        - Comparison operators in constraints: '=', '>='

    Arguments:
        opb (str or os.PathLike): 
            - A file path to an OPB file (optionally LZMA-compressed with `.xz`)
            - OR a string containing the OPB content directly

    Returns:
        cp.Model: The CPMpy model of the OPB instance.

    Example:
        >>> opb_text = '''
        ... * #variable= 5 #constraint= 2 #equal= 1 intsize= 64 #product= 5 sizeproduct= 13
        ... min: 2 x2 x3 +3 x4 ~x5 +2 ~x1 x2 +3 ~x1 x2 x3 ~x4 ~x5 ;
        ... 2 x2 x3 -1 x1 ~x3 = 5 ;
        ... '''
        >>> model = read_opb(opb_text)
        >>> print(model)
        Model(...)
    
    Notes:
        - Comment lines starting with '*' are ignored.
        - Only "min:" objectives are supported; "max:" is not recognized.
    """

    
    # If opb is a path to a file -> open file
    if isinstance(opb, (str, os.PathLike)) and os.path.exists(opb):
        f_open = lzma.open if str(opb).endswith(".xz") else open
        f = f_open(opb, 'rt')
    # If opb is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(opb)

    # Look for header on first line
    line = f.readline()
    header = HEADER_RE.match(line)
    if not header: # If not found on first line, look on second (happens when passing multi line string)
        _line = f.readline()
        header = HEADER_RE.match(_line)
        if not header:
            raise ValueError(f"Missing or incorrect header: \n0: {line}1: {_line}2: ...")
    nr_vars = int(header.group(2)) + 1

    # Generator without comment lines
    reader = (l for l in map(str.strip, f) if l and l[0] != '*')

    # CPMpy objects
    vars = cp.boolvar(shape=nr_vars, name="x")
    model = cp.Model()
    
    # Special case for first line -> might contain objective function
    first_line = next(reader)
    if OBJ_TERM_RE.match(first_line):
        obj_expr = _parse_term(first_line, vars)
        model.minimize(obj_expr)
    else: # no objective found, parse as a constraint instead
        model.add(_parse_constraint(first_line, vars))

    # Start parsing line by line
    for line in reader:
        model.add(_parse_constraint(line, vars))

    return model


def main():
    parser = argparse.ArgumentParser(description="Parse and solve an OPB model using CPMpy")
    parser.add_argument("model", help="Path to an OPB file (or raw OPB string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw OPB string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_opb(args.model)
        else:
            model = read_opb(os.path.expanduser(args.model))
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
