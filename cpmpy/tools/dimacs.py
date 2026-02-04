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
from cpmpy.transformations.to_cnf import to_cnf, to_gcnf, _to_clauses
from cpmpy.transformations.get_variables import get_variables

def write_gcnf(soft, hard=None, name=None, fname=None, encoding="auto", normalize=False):
    """
        Writes CPMpy model to GDIMACS format.
        Uses the "to_gcnf" transformation from CPMpy.
        For GDIMACS, it follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint, which is a requirement of the format. MUSSER2 is known to give incorrect results if groups are not disjoint.

        :param soft: list of soft constraints
        :param hard: list of hard constraints
        :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    """
    _, soft, hard, assumptions = to_gcnf(soft, hard, name=name, encoding=encoding, normalize=normalize)
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

    # Check explicitly for None, since groups=[] is a GCNF with only hard constraints (the {0} group), while groups=None should be a CNF
    is_gcnf = groups is not None

    groups = list(groups) if groups is not None else []

    vars = get_variables([constraints] + [con for _, con in groups])
    mapping = {v : i+1 for i, v in enumerate(vars)}

    out = ""
    n_clauses = 0
    for group, cons in itertools.chain(
            zip(itertools.repeat(0), constraints),
            ((i, constraint) for i, (_, constraint) in enumerate(groups, start=1))
        ):


        clauses = _to_clauses(cons)
        n_clauses += len(clauses)
        for clause in clauses:
            # write clause to cnf format
            ints = []
            for v in clause:
                if v is True:
                    continue
                elif isinstance(v, NegBoolView):
                    ints.append(str(-mapping[v._bv]))
                elif isinstance(v, _BoolVarImpl):
                    ints.append(str(mapping[v]))
                else:
                    raise ValueError(f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}")

            if is_gcnf:
                out += f"{{{group}}} "
            out += " ".join(ints + ["0"]) + "\n"

    out = f"p {'g' if is_gcnf else ''}cnf {len(vars)} {n_clauses}{f' {len(groups)}' if is_gcnf else ''}\n" + out

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
        reader = DimacsReader()
        for line in f.readlines():
            reader.read_line(line)

        m = reader.to_model()

        assert reader.nr_vars is not None, "Expected file to contain p-line, but did not"
        assert len(reader.clause) == 0, f"Expected last clause to be terminated by 0, but it was not"
        assert len(m.constraints) == reader.nr_cls, f"Number of clauses was declared in p-line as {reader.nr_cls}, but was {len(m.constraints)}"

    return m

class DimacsReader:

    def __init__(self):
        self.clause = []
        self.clauses = []
        self.nr_vars = None
        self.nr_cls = None
        self.bvs = []

    def read_p_line(self, typ, nr_vars, nr_cls):
        # assert len(params) == 4, f"Expected p-header to be formed `p cnf nr_vars nr_cls` but got {line}"
        if typ != "cnf":
            raise ValueError("Expected `cnf` (i.e. DIMACS) as file format, but got {typ} which is not supported.")
        self.nr_vars = int(nr_vars)
        if self.nr_vars>0:
            self.bvs = cp.boolvar(shape=self.nr_vars)
        self.nr_cls = int(nr_cls)

    def read_line(self, line):
        if line == "" or line.startswith("c"):
            pass  # skip empty and comment lines
        elif line.startswith("p"):
            self.read_p_line(*line.strip().split(" ")[1:])
        else:
            # assert nr_vars is not None, "Expected p-line before first clause"
            for token in line.strip().split():
                i = int(token.strip())
                if i == 0:
                    self.clauses.append(self.clause)
                    self.clause=[]
                else:
                    var=abs(i)-1
                    if self.nr_vars is not None:
                        assert var < self.nr_vars, "Expected at most {self.nr_vars} variables (from p-line) but found literal {i} in clause {line}"
                        bv = self.bvs[var]
                    else:
                        bv = cp.boolvar()
                        self.bvs.append(bv)
                    self.clause.append(bv if i > 0 else ~bv)


    def to_model(self):
        return cp.Model([cp.any(clause) for clause in self.clauses])


