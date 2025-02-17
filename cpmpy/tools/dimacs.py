#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py
##
"""
    This file implements helper functions for exporting CPMpy models from and to DIMACS format.
    DIMACS is a textual format to represent CNF problems.
    The header of the file should be formatted as `p cnf <n_vars> <n_constraints>`
    If the number of variables and constraints are not given, it is inferred by the parser.

    Each remaining line of the file is formatted as a list of integers.
    An integer represents a Boolean variable and a negative Boolean variable is represented using a `-` sign.
"""

import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables

import re


def write_dimacs(model, fname=None):
    """
        Writes CPMpy model to DIMACS format
        Uses the "to_cnf" transformation from CPMpy

        # TODO: implement pseudoboolean constraints in to_cnf
        :param model: a CPMpy model
        :fname: optional, file name to write the DIMACS output to
    """

    constraints = toplevel_list(model.constraints)
    constraints = to_cnf(constraints)

    vars = get_variables(constraints)
    mapping = {v : i+1 for i, v in enumerate(vars)}

    out = f"p cnf {len(vars)} {len(constraints)}\n"
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

        out += " ".join(ints + ["0"]) + "\n"

    if fname is not None:
        with open(fname, "w") as f:
            f.write(out)

    return out


def read_dimacs(fname):
    """
        Read a CPMpy model from a DIMACS formatted file striclty following the specification: https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html
        Note: the p-line has to denote the correct number of variables and clauses
        :param: fname: the name of the DIMACS file
        :param: sep: optional, separator used in the DIMACS file, will try to infer if None
    """

    m = cp.Model()

    with open(fname, "r") as f:
        clause = []
        had_p_line = False
        for line in f.readlines():
            if line == "" or line.startswith("c"):
                continue  # skip empty and comment lines
            elif line.startswith("p"):
                try:
                    _,typ,nr_vars,nr_cls = line.strip().split(" ")
                except ValueError:
                    raise cp.exceptions.CPMpyException(f"Invalid DIMACS file p-header: {line}")
                if typ != "cnf":
                    raise ValueError("Expected `cnf` (i.e. DIMACS) as file format, but got {typ} which is not supported.")
                nr_vars = int(nr_vars)
                if nr_vars>0:
                    bvs = cp.boolvar(shape=nr_vars)
                nr_cls = int(nr_cls)
                had_p_line = True
            else:
                assert had_p_line, "Expected p-line before clauses"
                for token in line.strip().split():
                    i = int(token.strip())
                    if i == 0:
                        m+=cp.any(clause)
                        clause = []
                    else:
                        try:
                            bv = bvs[abs(i)-1]
                        except IndexError:
                            raise AssertionError("Expected <={nr_vars} variables (from p-line) but found literal {i} in clause {line}")

                        clause.append(bv if i > 0 else ~bv)

        assert len(m.constraints) == nr_cls, f"Number of clauses was declared in p-line as {nr_cls}, but was {len(m.constraints)}"
        assert not clause, f"Expected last clause to be terminated by 0, but was not {clause}"

    return m





