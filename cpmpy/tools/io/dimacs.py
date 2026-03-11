#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py
##
"""
    This file implements helper functions for exporting CPMpy models from and to DIMACS format.
    DIMACS is a textual format to represent CNF problems.
    The header of the file should be formatted as ``p cnf <n_vars> <n_constraints>``.
    If the number of variables and constraints are not given, it is inferred by the parser.

    Each remaining line of the file is formatted as a list of integers.
    An integer represents a Boolean variable and a negative Boolean variable is represented using a `'-'` sign.
"""

import os

import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl
from cpmpy.expressions.core import Operator

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.safening import safen_objective
from cpmpy.transformations.flatten_model import flatten_objective
from cpmpy.transformations.linearize import decompose_linear_objective, only_positive_coefficients_
from cpmpy.transformations.int2bool import _encode_lin_expr

from typing import Optional, Callable, Union
import builtins


def _transform_objective(expr, encoding="auto"):
    """
        Transform objective into weighted Boolean literals plus helper constraints.

        Returns:
            (weights, xs, const, extra_cons)
    """
    csemap, ivarmap = dict(), dict()
    obj, safe_cons = safen_objective(expr)
    obj, decomp_cons = decompose_linear_objective(
        obj,
        supported=frozenset(),
        supported_reified=frozenset(),
        csemap=csemap,
    )
    obj, flat_cons = flatten_objective(obj, csemap=csemap)

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
        raise NotImplementedError(f"DIMACS: Non supported objective {obj} (yet?)")

    terms, enc_cons, k = _encode_lin_expr(ivarmap, xs, weights, encoding, csemap=csemap)
    const += k

    extra_cons = safe_cons + decomp_cons + flat_cons + enc_cons

    # remove terms with coefficient 0 (`only_positive_coefficients_` may return them and RC2 does not accept them)
    terms = [(w, x) for w, x in terms if w != 0]
    if len(terms) == 0:
        return [], [], const, extra_cons

    ws, xs = zip(*terms)  # unzip
    new_weights, new_xs, k = only_positive_coefficients_(ws, xs)
    const += k

    return list(new_weights), list(new_xs), const, extra_cons


def write_dimacs(model, fname=None, encoding="auto", p_header:bool=False, header:Optional[str]="DIMACS file written by CPMpy", open: Optional[Callable]=None):
    """
        Writes CPMpy model to DIMACS format
        Uses the "to_cnf" transformation from CPMpy

        .. todo::
            TODO: implement pseudoboolean constraints in to_cnf

        :param model: a CPMpy model
        :param fname: optional, file name to write the DIMACS output to
        :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
        :param p_header: whether to include the ``p ...`` problem header line (default: ``False``)
        :param open: optional callable to open the file for writing (default: builtin ``open``).
            Called as ``open(fname, "w")``. This mirrors the ``open=`` argument
            in loaders and allows custom compression or I/O (e.g.
            ``lambda p, mode='w': lzma.open(p, 'wt')``).
    """

    if model.has_objective():
        hard_prefix = "h "
    else:
        hard_prefix = ""

    constraints = toplevel_list(model.constraints)
    objective_lits = []
    objective_weights = []
    if model.has_objective():
        objective_weights, objective_lits, _, extra_cons = _transform_objective(model.objective_, encoding=encoding)
        constraints += extra_cons
    constraints = to_cnf(constraints, encoding=encoding)

    vars = get_variables(constraints + objective_lits)
    mapping = {v : i+1 for i, v in enumerate(vars)}
    out = ""

    
    for cons in constraints:

        if isinstance(cons, _BoolVarImpl):
            cons = Operator("or", [cons])

        if not (isinstance(cons, Operator) and cons.name == "or"):
            raise NotImplementedError(f"Unsupported constraint {cons}")

        # write clause to cnf format
        ints = []
        for v in cons.args:
            if isinstance(v, NegBoolView):
                ints.append(str(-mapping[v._bv]))
            elif isinstance(v, _BoolVarImpl):
                ints.append(str(mapping[v]))
            else:
                raise ValueError(f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}")

        out += hard_prefix + " ".join(ints + ["0"]) + "\n"

    if model.has_objective():
        max_weight = max(objective_weights)
        for w, x in zip(objective_weights, objective_lits):
            if isinstance(x, NegBoolView):
                lit = -mapping[x._bv]
            elif isinstance(x, _BoolVarImpl):
                lit = mapping[x]
            else:
                raise ValueError(f"Expected Boolean literal in objective, but got {x} of type {type(x)}")
            transformed_weight = max_weight - w if model.objective_is_min else w
            out += f"{transformed_weight} {lit} 0\n"

        if p_header:
            out = f"p wcnf {len(vars)} {len(constraints)} {max(objective_weights)}\n" + out
    else:
        if p_header:
            out = f"p cnf {len(vars)} {len(constraints)}\n" + out

    if header is not None:
        header_lines = ["c " + line for line in header.splitlines()]
        out = "\n".join(header_lines) + "\n" + out

    if fname is not None:
        opener = open if open is not None else builtins.open
        with opener(fname, "w") as f:
            f.write(out)

    return out


def load_dimacs(dimacs: Union[str, os.PathLike], open=None):
    """
        Load a CPMpy model from a DIMACS formatted file strictly following the specification:
        https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html

        .. note::
            The p-line has to denote the correct number of variables and clauses

        :param dimacs:
            - A file path to a DIMACS/WCNF file
            - OR a string containing DIMACS/WCNF content directly
        :param open: optional callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.cnf.xz``.
    """
    if open is None:
        open = builtins.open

    if isinstance(dimacs, (str, os.PathLike)) and os.path.exists(dimacs):
        with open(dimacs, "r") as f:
            lines = f.readlines()
    else:
        lines = str(dimacs).splitlines()

    # Auto-detect weighted instances:
    # - explicit `p wcnf ...` header
    # - any hard-clause line starting with `h`
    # - no header but all non-comment clause lines look weighted (weight literals... 0)
    is_weighted = False
    weighted_compatible = True
    saw_clause_line = False
    for raw in lines:
        line = raw.strip()
        if line == "" or line.startswith("c"):
            continue
        if line.startswith("p"):
            params = line.split()
            assert len(params) >= 4, f"Expected p-header to be formed `p <typ> ...` but got {line}"
            _, typ, *_ = params
            if typ == "wcnf":
                is_weighted = True
            elif typ != "cnf":
                raise ValueError(f"Expected `cnf` or `wcnf` as file format, but got {typ} which is not supported.")
            break
        if line.startswith("h"):
            is_weighted = True
            break
        saw_clause_line = True
        try:
            ints = [int(tok) for tok in line.split()]
        except ValueError:
            weighted_compatible = False
            continue
        if len(ints) < 2 or ints[-1] != 0 or ints[0] < 0:
            weighted_compatible = False

    if not is_weighted and saw_clause_line and weighted_compatible:
        is_weighted = True

    if is_weighted:
        from cpmpy.tools.io.wcnf import load_wcnf
        return load_wcnf(dimacs, open=open)

    # CNF parse (strict with p-line counts when present, inferred otherwise)
    m = cp.Model()
    clause = []
    clauses = []
    nr_vars_declared = None
    nr_cls_declared = None
    max_var = 0

    for raw in lines:
        line = raw.strip()
        if line == "" or line.startswith("c"):
            continue  # skip empty and comment lines
        if line.startswith("p"):
            params = line.split()
            assert len(params) == 4, f"Expected p-header to be formed `p cnf nr_vars nr_cls` but got {line}"
            _, typ, nr_vars, nr_cls = params
            if typ != "cnf":
                raise ValueError(f"Expected `cnf` (i.e. DIMACS) as file format, but got {typ} which is not supported.")
            nr_vars_declared = int(nr_vars)
            nr_cls_declared = int(nr_cls)
            continue

        for token in line.split():
            i = int(token)
            if i == 0:
                clauses.append(clause)
                clause = []
            else:
                max_var = max(max_var, abs(i))
                clause.append(i)

    assert len(clause) == 0, "Expected last clause to be terminated by 0, but it was not"

    nr_vars = nr_vars_declared if nr_vars_declared is not None else max_var
    if nr_vars_declared is not None:
        assert max_var <= nr_vars_declared, f"Expected at most {nr_vars_declared} variables (from p-line) but found literal index {max_var}"

    bvs = cp.boolvar(shape=nr_vars) if nr_vars > 0 else []
    for cl in clauses:
        lits = []
        for i in cl:
            bv = bvs[abs(i)-1]
            lits.append(bv if i > 0 else ~bv)
        m += cp.any(lits)

    if nr_cls_declared is not None:
        assert len(m.constraints) == nr_cls_declared, f"Number of clauses was declared in p-line as {nr_cls_declared}, but was {len(m.constraints)}"

    return m

# Backward compatibility alias
read_dimacs = load_dimacs
