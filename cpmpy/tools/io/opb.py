#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## opb.py
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
import argparse
from io import StringIO
from typing import Union, Optional, Callable
from functools import reduce
from operator import mul

import cpmpy as cp
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.to_opb import to_opb, to_opb_objective
from cpmpy.expressions.variables import NegBoolView, allow_reserved_var_names
from cpmpy.expressions.core import Operator, Comparison


# Regular expressions
HEADER_RE = re.compile(r'(.*)\s*#variable=\s*(\d+)\s*#constraint=\s*(\d+).*')
TERM_RE = re.compile(r"([+-])\s*((?:(?:\d+\s+)?~?[^\s;]+(?:\s+~?[^\s;]+)*))(?=\s+[+-]|$)")
TERM_RE = re.compile(r"([+-]?)\s*((?:(?:\d+\s+)?~?[^\s;=]+(?:\s+~?[^\s;=]+)*?))(?=\s*[+-]|\s*=|\s*;|$)")
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
    text = line.strip()

    # Keep only the expression fragment for objective/constraint lines.
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace(";", " ").strip()
    if not text:
        raise ValueError(f"Could not parse OPB term from empty line: {line}")

    # OPB allows first term without an explicit sign. Normalize by prepending '+'
    # and then parse term-by-term using TERM_RE.
    if text and text[0] not in "+-":
        text = "+ " + text

    parsed_terms = TERM_RE.findall(text) # contains each part of the sum we're building
    if not parsed_terms:
        raise ValueError(f"Could not parse any OPB terms from line: {line}")

    for sign_tok, body in parsed_terms:
        parts = body.split()
        if not parts:
            raise ValueError(f"Missing variable token in OPB term: {line}")

        coeff = 1
        # Support unsigned/signed integer coefficients. Unsigned is interpreted as '+'.
        if parts[0].lstrip("+-").isdigit():
            coeff = int(parts[0])
            parts = parts[1:]
        if not parts:
            raise ValueError(f"Missing variable token in OPB term: {line}")

        factors = []
        for tok in parts:
            neg = tok.startswith("~")
            var_name = tok[1:] if neg else tok
            if var_name not in vars:
                # OPB annotations may contain reserved prefixes (IV*/BV*); accept them while parsing.
                with allow_reserved_var_names():
                    vars[var_name] = cp.boolvar(name=var_name)
            factors.append(~vars[var_name] if neg else vars[var_name])
        
        weight = -coeff if sign_tok == "-" else coeff
        term = weight * reduce(mul, factors, 1) # create weighted term
        terms.append(term)

    if len(terms) == 0:
        raise ValueError(f"Could not parse any OPB terms from line: {line}")
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

    match = IND_TERM_RE.search(line)
    if match is None:
        raise ValueError(f"Could not parse OPB comparator/rhs from line: {line}")
    op, ind_term = match.groups()
    lhs = _parse_term(line[:match.start()], vars)

    rhs = int(ind_term) if ind_term.lstrip("+-").isdigit() else vars[ind_term]

    return cp.expressions.core.Comparison(
        name="==" if op == "=" else ">=",
        left=lhs,
        right=rhs
    )

_std_open = open
def load_opb(opb: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Loader for OPB (Pseudo-Boolean) format. Loads an instance and returns its matching CPMpy model.

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
        open: (callable):
            If wcnf is the path to a file, a callable to "open" that file (default=python standard library's 'open').

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
        if open is not None:
            f = open(opb)
        else:
            f = _std_open(opb, "rt")
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
    nr_vars_declared = int(header.group(2))

    # Generator without comment lines
    reader = (line_text for line_text in map(str.strip, f) if line_text and line_text[0] != '*')

    # CPMpy objects
    vars = {}
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
        cons = _parse_constraint(line, vars)
        assert isinstance(cons, Comparison), cons
        lhs, rhs = cons.args
        assert isinstance(lhs, Operator) and lhs.name == "wsum" or lhs.name == "sum", lhs
        assert isinstance(rhs, int), rhs
        model.add(cons)

    if len(vars) > nr_vars_declared:
        import warnings
        warnings.warn(
            f"Header declares {nr_vars_declared} variables but found {len(vars)} unique variables while parsing",
            stacklevel=2,
        )

    return model

def write_opb(model, fname=None, encoding="auto", header=None, open=None, annotate: Optional[Callable] = None):
    """
    Export a CPMpy model to the OPB (Pseudo-Boolean) format.

    This function transforms the given CPMpy model into OPB format, which is a standard textual
    format for representing Pseudo-Boolean optimization problems. The OPB file will contain
    a header specifying the number of variables and constraints, the objective (optional), and the 
    list of constraints using integer-weighted Boolean variables.

    Arguments:
        model (cp.Model): The CPMpy model to export.
        fname (str, optional): The file name to write the OPB output to. If None, the OPB string is returned.
        encoding (str, optional): The encoding used for `int2bool`. Options: ("auto", "direct", "order", "binary").
        header (str, optional): Optional header text to add as OPB comments. If provided, each line
            will be prefixed with "* ".
        open (callable, optional): Callable to open the file for writing (default: builtin ``open``).
            Called as ``open(fname, "w")``. This mirrors the ``open=`` argument
            in loaders and allows custom compression or I/O (e.g.
            ``lambda p, mode='w': lzma.open(p, 'wt')``).
        annotate (callable, optional): ``annotate(bool_var, ivarmap) -> str`` for each OPB identifier.
            If omitted, uses names derived from the integer encoding map:
            ``source>=threshold``, ``source=value``, or ``source[bit=i]`` (or ``var.name`` for plain Booleans).

    Returns:
        str or None: The OPB string if `fname` is None, otherwise nothing (writes to file).

    Format:
        * #variable= <n_vars> #constraint= <n_constraints>
        min/max: <objective>;
        <constraint_1>;
        <constraint_2>;
        ...

    Note:
        Solvers that only accept ``x<int>`` can pass a custom ``annotate`` callback.

    Example:
        >>> from cpmpy import *
        >>> x = boolvar(shape=3)
        >>> m = Model(x[0] + x[1] + x[2] >= 2)
        >>> print(write_opb(m))
    """

    csemap, ivarmap = CSEMap(), dict()
    opb_cons = to_opb(model.constraints, csemap, ivarmap, encoding)

    # Transform objective, if present
    if model.objective_ is not None:
        opb_obj, const, extra_cons = to_opb_objective(model.objective_, csemap, ivarmap, encoding)
        opb_cons += to_opb(extra_cons, csemap, ivarmap, encoding)
    else:
        opb_obj = None

    # Form header and variable mapping
    # Use all variables occurring in constraints and the objective
    all_vars = get_variables(opb_cons + ([opb_obj] if opb_obj is not None else []))
    out = [
        f"* #variable= {len(all_vars)} #constraint= {len(opb_cons)}",
    ]
    if header:
        header_lines = ["* " + line for line in str(header).splitlines()]
        out.extend(header_lines)

    # Map variables to OPB variable names
    if annotate is None:
        varmap = {v: f"x{i+1}" for i, v in enumerate(all_vars)} 
    else:
        varmap = {v: ann for v, ann in zip(all_vars, annotate(all_vars, ivarmap))}
    
    # Write objective, if present
    if model.objective_ is not None:
        objective_str = _wsum_to_str(opb_obj, varmap)
        out.append(f"{'min' if model.objective_is_min else 'max'}: {objective_str};")

    # Write constraints
    for cons in opb_cons:
        assert isinstance(cons, Comparison), f"Expected a comparison, but got {cons}"
        lhs, rhs = cons.args
        constraint_str = f"{_wsum_to_str(lhs, varmap)} {cons.name} {rhs};"
        out.append(constraint_str)

    # Output to file or string
    contents = "\n".join(out)
    if fname is None:
        return contents
    opener = open if open is not None else _std_open
    with opener(fname, "w") as f:
        f.write(contents)


def _wsum_to_str(cpm_expr, varmap):
    """
    Convert a weighted sum CPMpy expression to a string in OPB format.

    args:
        cpm_expr (Operator): wsum CPMpy expression
        varmap (dict): dictionary mapping CPMpy variables to OPB variable names
    """
    assert isinstance(cpm_expr, Operator) and cpm_expr.name == "wsum", f"Expected a wsum, but got {cpm_expr}"
    weights, args = cpm_expr.args

    out = []
    for w, var in zip(weights, args):
        if w == 0:
            continue
        lit = varmap[var] if not isinstance(var, NegBoolView) else "~" + str(varmap[var._bv])
        # OPB requires no space between sign and digits (e.g., "-2", "+4").
        out.append(f"{w:+d} {lit}")
    
    str_out = " ".join(out)
    return str_out


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
            model = load_opb(args.model)
        else:
            model = load_opb(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        if isinstance(e, AssertionError):
            raise e
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
