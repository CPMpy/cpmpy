#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py
##
"""
This file implements helper functions for exporting CPMpy models from and to DIMACS format.
DIMACS is a textual format to represent CNF problems.

The header of the file can optionally be formatted as ``p cnf <n_vars> <n_constraints>``.
If the number of variables and constraints are not given, it is inferred by the parser.

Each remaining line of the file is formatted as a list of integers.
An integer represents a Boolean variable and a negative Boolean variable is represented using a `'-'` sign.
"""

import os
import builtins
from typing import Optional, Callable, Union

import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView, NDVarArray
from cpmpy.expressions.core import Operator

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf, to_cnf_objective
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.int2bool import IntVarEnc


def write_dimacs(
        model: cp.Model, 
        fname:Optional[str]=None, 
        encoding:str="auto", 
        p_header:bool=False, header:Optional[str]="DIMACS file written by CPMpy", 
        open: Optional[Callable]=None, 
        annotate: Optional[Callable]=None
    ):
    """
    Writes CPMpy model to DIMACS format
    Uses the "to_cnf" transformation from CPMpy

    .. todo::
        TODO: implement pseudoboolean constraints in to_cnf

    Arguments:
        model (cp.Model): a CPMpy model
        fname (str, optional): optional, file name to write the DIMACS output to
        encoding (str): the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary") (default: "auto")
        p_header (bool): whether to include the ``p ...`` problem header line (default: ``False``)
            open (Callable, optional): optional callable to open the file for writing (default: builtin ``open``).
            Called as ``open(fname, "w")``. This mirrors the ``open=`` argument
            in loaders and allows custom compression or I/O (e.g.
            ``lambda p, mode='w': lzma.open(p, 'wt')``).
        annotate (Callable, optional): variable annotation strategy. Controls how DIMACS literal IDs are
            mapped back to original CPMpy variables. Options:
            - None (default): no annotation, output identical to previous behaviour
            - "dimacs_comments": Sugar-style 'c <id> <name>' comment lines (self-contained)
            - "json_sidecar": comments + a .map.json sidecar file (BumbleBee pattern)
            - "none": explicit no-op (same as None)
            - VariableAnnotator instance: fully custom strategy
    """

    if model.has_objective():
        hard_prefix = "h "
    else:
        hard_prefix = ""

    # Shared maps so both objective and constraint transformations populate
    # the same ivarmap, enabling annotation of all integer variable encodings.
    ivarmap: dict[str, IntVarEnc] = dict()
    csemap = CSEMap()

    constraints = toplevel_list(model.constraints)
    objective_lits = []
    objective_weights = []
    if model.has_objective():
        objective_weights, objective_lits, _, extra_cons = to_cnf_objective(
            model.objective_, encoding=encoding, csemap=csemap, ivarmap=ivarmap
        )
        constraints += extra_cons
    constraints = to_cnf(constraints, csemap=csemap, ivarmap=ivarmap, encoding=encoding)

    vars = get_variables(constraints + objective_lits)
    mapping = {v : i+1 for i, v in enumerate(vars)}
    out = ""

    # Write constraints to DIMACS format
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

    # Write objective to DIMACS format
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

    # Write annotations to DIMACS string
    if annotate is not None:
        if callable(annotate):
            comments = annotate(vars, ivarmap)
            if comments:
                comment_block = "\n".join(f"c {i+1} " + c for i,c in enumerate(comments)) + "\n"
                out = comment_block + out
        else:
            raise ValueError(f"Expected a Callable annotate, but got {type(annotate)}")

    # Optional p-header
    if p_header:
        if model.has_objective():
            out = f"p wcnf {len(vars)} {len(constraints)} {max(objective_weights)}\n" + out
        else:
            out = f"p cnf {len(vars)} {len(constraints)}\n" + out

    # Optional header
    if header is not None:
        header_lines = ["c " + line for line in header.splitlines()]
        out = "\n".join(header_lines) + "\n" + out

    # Write to file
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
        The (optional) p-line has to denote the correct number of variables and clauses

    Arguments:
        dimacs:
            - A file path to a DIMACS/WCNF file
            - OR a string containing DIMACS/WCNF content directly
        open: optional callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.cnf.xz``.
    """
    if open is None:
        open = builtins.open

    # Read from file or string
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

    # If weighted, delegate to WCNF loader
    if is_weighted:
        from cpmpy.tools.io.wcnf import load_wcnf
        return load_wcnf(dimacs, open=open)

    # CNF parse (strict with p-line counts when present, inferred otherwise)
    m = cp.Model()
    clause: list[int] = []
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

    if nr_vars > 0:
        bvs: NDVarArray = cp.boolvar(shape=(nr_vars,))
        for cl in clauses:
            lits = []
            for i in cl:
                bv = bvs[abs(i)-1]
                lits.append(bv if i > 0 else ~bv)
            m += cp.any(lits)

    if nr_cls_declared is not None:
        assert len(m.constraints) == nr_cls_declared, f"Number of clauses was declared in p-line as {nr_cls_declared}, but was {len(m.constraints)}"

    return m
