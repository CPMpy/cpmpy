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
from typing import Union
from functools import reduce
from operator import mul


import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list,simplify_boolean
from cpmpy.transformations.safening import no_partial_functions, safen_objective
from cpmpy.transformations.decompose_global import decompose_in_tree, decompose_objective
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_objective
from cpmpy.transformations.reification import only_implies, only_bv_reifies
from cpmpy.transformations.linearize import linearize_constraint, only_positive_bv_wsum
from cpmpy.transformations.int2bool import int2bool, _encode_lin_expr
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.variables import _IntVarImpl, NegBoolView, _BoolVarImpl
from cpmpy.expressions.core import Operator, Comparison
from cpmpy import __version__


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
                idx = int(v[2:]) - 1 # remove "~x" and opb is 1-based indexing
                factors.append(~vars[idx])
            else:
                idx = int(v[1:]) - 1 # remove "x" and opb is 1-based indexing
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

_std_open = open
def read_opb(opb: Union[str, os.PathLike], open=open) -> cp.Model:
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
    nr_vars = int(header.group(2))

    # Generator without comment lines
    reader = (l for l in map(str.strip, f) if l and l[0] != '*')

    # CPMpy objects
    vars = cp.boolvar(shape=nr_vars, name="x")
    if nr_vars == 1:
        vars = cp.cpm_array([vars]) # ensure vars is indexable even for single variable case
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

def write_opb(model, fname=None, encoding="auto"):
    """
    Export a CPMpy model to the OPB (Pseudo-Boolean) format.

    This function transforms the given CPMpy model into OPB format, which is a standard textual
    format for representing Pseudo-Boolean optimization problems. The OPB file will contain
    a header specifying the number of variables and constraints, the objective (optional), and the 
    list of constraints using integer-weighted Boolean variables.

    Args:
        model (cp.Model): The CPMpy model to export.
        fname (str, optional): The file name to write the OPB output to. If None, the OPB string is returned.
        encoding (str, optional): The encoding used for `int2bool`. Options: ("auto", "direct", "order", "binary").

    Returns:
        str or None: The OPB string if `fname` is None, otherwise nothing (writes to file).

    Format:
        * #variable= <n_vars> #constraint= <n_constraints>
        * OPB file generated by CPMpy version <version>
        min/max: <objective>;
        <constraint_1>;
        <constraint_2>;
        ...

    Note:
        Some solvers only support variable names of the form x<int>. The OPB writer will remap
        all CPMpy variables to such a format internally.

    Example:
        >>> from cpmpy import *
        >>> x = boolvar(shape=3)
        >>> m = Model(x[0] + x[1] + x[2] >= 2)
        >>> print(write_opb(m))
    """

    csemap, ivarmap = dict(), dict()
    opb_cons = _transform(model.constraints, csemap, ivarmap, encoding)

    if model.objective_ is not None:
        opb_obj, const, extra_cons = _transform_objective(model.objective_, csemap, ivarmap, encoding)
        opb_cons += extra_cons
    else:
        opb_obj = None

    # Form header and variable mapping
    # Use all variables occurring in constraints and the objective
    all_vars = get_variables(opb_cons + ([opb_obj] if opb_obj is not None else []))
    out = [
        f"* #variable= {len(all_vars)} #constraint= {len(opb_cons)}",
        f"* OPB file generated by CPMpy version {__version__}",
    ]
    # Remap variables to 'x1', 'x2', ..., the standard OPB way
    varmap = {v: f"x{i+1}" for i, v in enumerate(all_vars)}
    
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
    else:
        with open(fname, "w") as f:
            f.write(contents)

def _normalized_comparison(lst_of_expr):
    """
    Convert a list of linear CPMpy expressions into OPB-compatible pseudo-Boolean constraints.

    Transforms a list of Boolean-linear CPMpy expressions (as output by `linearize_constraint`) into a list
    of OPB-normalized constraints, expressed as comparisons between weighted Boolean sums
    (using "wsum") and integer constants. Handles Boolean vars, reifications, implications,
    and ensures all equalities are decomposed into two inequalities.
    
    Args:
        lst_of_expr (list): List of CPMpy Boolean-linear expressions.

    Returns:
        list: List of normalized CPMpy `Comparison` objects representing pseudo-Boolean constraints.
    """
    newlist = []
    for cpm_expr in lst_of_expr:
        if isinstance(cpm_expr, cp.BoolVal) and cpm_expr.value() is False:
            raise NotImplementedError(f"Cannot transform {cpm_expr} to OPB constraint")
        
        # single Boolean variable
        if isinstance(cpm_expr, _BoolVarImpl):
            cpm_expr = Operator("sum", [cpm_expr]) >= 1

        # implication
        if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            bv, subexpr = cpm_expr.args
            assert isinstance(subexpr, _BoolVarImpl), "Only bv -> bv should reach here, but got {subexpr}"
            cpm_expr = Operator("wsum", [[-1, 1], [bv, subexpr]]) >= 0
            newlist.append(cpm_expr)
            continue
        
        # Comparison, can be single Boolean variable or (weighted) sum of Boolean variables
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            if isinstance(lhs, _BoolVarImpl):
                lhs = Operator("sum", [lhs])
            if lhs.name == "sum":
                lhs = Operator("wsum", [[1]*len(lhs.args), lhs.args])

            assert isinstance(lhs, Operator) and lhs.name == "wsum", f"Expected a wsum, but got {lhs}"

            # convert comparisons into >= constraints
            if cpm_expr.name == "==":
                newlist += _normalized_comparison([lhs <= rhs])
                newlist += _normalized_comparison([lhs >= rhs])
            elif cpm_expr.name == ">=":
                newlist.append(lhs >= rhs)
            elif cpm_expr.name == "<=":
                new_weights = [-w for w in lhs.args[0]]
                newlist.append(Operator("wsum", [new_weights, lhs.args[1]]) >= -rhs)
            else:
                raise ValueError(f"Unknown comparison {cpm_expr.name}")
        else:
            raise NotImplementedError(f"Expected a comparison, but got {cpm_expr}")

    return newlist

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
        var = varmap[var] if not isinstance(var, NegBoolView) else f"~{varmap[var._bv]}"
        if w < 0:
            out.append(f"- {w} {var}")
        elif w > 0:
            out.append(f"+ {w} {var}")
        else:
            pass # zero weight, ignore
    
    str_out = " ".join(out)
    return str_out

def _transform(cpm_expr, csemap, ivarmap, encoding="auto"):
    """
        Transform a list of CPMpy expressions into a list of Pseudo-Boolean constraints.
    """

    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
    cpm_cons = decompose_in_tree(cpm_cons,
        supported={"alldifferent"},  # alldiff has a specialized MIP decomp in linearize
        csemap=csemap
    )
    cpm_cons = simplify_boolean(cpm_cons)
    cpm_cons = flatten_constraint(cpm_cons, csemap=csemap)  # flat normal form
    cpm_cons = only_bv_reifies(cpm_cons, csemap=csemap)
    cpm_cons = only_implies(cpm_cons, csemap=csemap)
    cpm_cons = linearize_constraint(
        cpm_cons, supported=frozenset({"sum", "wsum"}), csemap=csemap
    )
    cpm_cons = int2bool(cpm_cons, ivarmap, encoding=encoding)

    return _normalized_comparison(cpm_cons)

def _transform_objective(expr, csemap, ivarmap, encoding="auto"):
    """
    Transform a CPMpy objective expression into a weighted sum expression
    """

    # transform objective
    obj, safe_cons = safen_objective(expr)
    obj, decomp_cons = decompose_objective(obj, supported={"alldifferent"},
                                            csemap=csemap)
    obj, flat_cons = flatten_objective(obj, csemap=csemap)
    obj = only_positive_bv_wsum(obj)  # remove negboolviews

    weights, xs, const = [], [], 0
    # we assume obj is a var, a sum or a wsum (over int and bool vars)
    if isinstance(obj, _IntVarImpl) or isinstance(obj, NegBoolView):  # includes _BoolVarImpl
        weights = [1]
        xs = [obj]
    elif obj.name == "sum":
        xs = obj.args
        weights = [1] * len(xs)
    elif obj.name == "wsum":
        weights, xs = obj.args
    else:
        raise NotImplementedError(f"OPB: Non supported objective {obj} (yet?)")

    terms, cons, k = _encode_lin_expr(ivarmap, xs, weights, encoding)

    # remove terms with coefficient 0 (`only_positive_coefficients_` may return them and RC2 does not accept them)
    terms = [(w, x) for w,x in terms if w != 0]  

    obj = Operator("wsum", [[w for w,x in terms], [x for w,x in terms]])
    return obj, const, safe_cons + decomp_cons + flat_cons


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
