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

import itertools

import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView, Operator

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf, to_gcnf
from cpmpy.transformations.get_variables import get_variables

def write_gcnf(soft, hard=None, name=None, fname=None, encoding="auto"):
    """
        Writes CPMpy model to GDIMACS format.
        Uses the "to_gcnf" transformation from CPMpy.
        For GDIMACS, it follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint, which is a requirement of the format. MUSSER2 is known to give incorrect results if groups are not disjoint.

        :param soft: list of soft constraints
        :param hard: list of hard constraints
        :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    """
    _, soft, hard, assumptions = to_gcnf(soft, hard, name=name, encoding=encoding)
    return write_dimacs_(hard, groups=zip(assumptions, soft), fname=fname)

def write_dimacs(model, fname=None, encoding="auto"):
    """
        Writes CPMpy model to DIMACS format.
        :param model: a CPMpy model
        :param fname: optional, file name to write the DIMACS output to
        :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    """

    constraints = toplevel_list(model.constraints)
    constraints = to_cnf(constraints, encoding=encoding)
    return write_dimacs_(constraints, fname=fname)

def write_dimacs_(constraints, groups=None, fname=None):
    vars = get_variables(constraints)
    mapping = {v : i+1 for i, v in enumerate(vars)}

    # Check explicitly for None, since groups=[] is a GCNF with only hard constraints (the {0} group), while groups=None should be a CNF
    is_gcnf = groups is not None

    out = ""
    if is_gcnf:
        typ = "gcnf"
        # exhaust iterator to get number of groups up-front
        groups = list(groups)
        out += f' {len(groups)}'
    else:
        typ = "cnf"
        groups = []

    out = f"p {typ} {len(vars)} {len(constraints) + len(groups)}{f' {len(groups)}' if is_gcnf else ''}"

    out += '\n'

    for group, cons in itertools.chain(
            zip(itertools.repeat(0), constraints),
            ((i, constraint) for i, (_, constraint) in enumerate(groups, start=1))
        ):

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

        if is_gcnf:
            out += f"{{{group}}} "
        out += " ".join(ints + ["0"]) + "\n"

    if fname is not None:
        with open(fname, "w") as f:
            f.write(out)

    return out


def read_dimacs(fname):
    """
        Read a CPMpy model from a DIMACS formatted file strictly following the specification:
        https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html
        
        .. note::
            The p-line has to denote the correct number of variables and clauses
        
        :param fname: the name of the DIMACS file
        :param sep: optional, separator used in the DIMACS file, will try to infer if None
    """

    m = cp.Model()

    with open(fname, "r") as f:
        clause = []
        nr_vars = None
        for line in f.readlines():
            if line == "" or line.startswith("c"):
                continue  # skip empty and comment lines
            elif line.startswith("p"):
                params = line.strip().split(" ")
                assert len(params) == 4, f"Expected p-header to be formed `p cnf nr_vars nr_cls` but got {line}"
                _,typ,nr_vars,nr_cls = params
                if typ != "cnf":
                    raise ValueError("Expected `cnf` (i.e. DIMACS) as file format, but got {typ} which is not supported.")
                nr_vars = int(nr_vars)
                if nr_vars>0:
                    bvs = cp.boolvar(shape=nr_vars)
                nr_cls = int(nr_cls)
            else:
                assert nr_vars is not None, "Expected p-line before first clause"
                for token in line.strip().split():
                    i = int(token.strip())
                    if i == 0:
                        m += cp.any(clause)
                        clause = []
                    else:
                        var=abs(i)-1
                        assert var < nr_vars, "Expected at most {nr_vars} variables (from p-line) but found literal {i} in clause {line}"
                        bv = bvs[var]

                        clause.append(bv if i > 0 else ~bv)

        assert nr_vars is not None, "Expected file to contain p-line, but did not"
        assert len(clause) == 0, f"Expected last clause to be terminated by 0, but it was not"
        assert len(m.constraints) == nr_cls, f"Number of clauses was declared in p-line as {nr_cls}, but was {len(m.constraints)}"

    return m





