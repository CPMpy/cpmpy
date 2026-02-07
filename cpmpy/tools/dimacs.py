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
from cpmpy.tools.explain.marco import make_assump_model

def write_gdimacs(soft, hard=None, name=None, fname=None, encoding="auto", disjoint=False, canonical=False):
    """
        Writes CPMpy model to GDIMACS format.
        Uses the "to_gcnf" transformation from CPMpy.
        For GDIMACS, it follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint, which is a requirement of the format. MUSSER2 is known to give incorrect results if groups are not disjoint.

        :param soft: list of soft constraints
        :param hard: list of hard constraints
        :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    """
    _, soft, hard, assumptions = to_gcnf(soft, hard, name=name, encoding=encoding, disjoint=disjoint)
    return write_dimacs_(hard, groups=zip(assumptions, soft), fname=fname, canonical=canonical)

def write_dimacs(model, fname=None, encoding="auto", canonical=False):
    """
        Writes CPMpy model to DIMACS format.
        :param model: a CPMpy model
        :param fname: optional, file name to write the DIMACS output to
        :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    """

    constraints = toplevel_list(model.constraints)
    constraints = to_cnf(constraints, encoding=encoding)
    return write_dimacs_(constraints, fname=fname, canonical=canonical)

def write_dimacs_(constraints, groups=None, fname=None, canonical=False):

    # Check explicitly for None, since groups=[] is a GCNF with only hard constraints (the {0} group), while groups=None should be a CNF
    is_gcnf = groups is not None

    groups = list(groups) if groups is not None else []

    vars = get_variables([constraints] + [con for _, con in groups])

    if canonical:
        vars.sort(key=lambda x: x.name)

    mapping = { v : i for i, v in enumerate(vars, start=1) }

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
            lits = []
            for v in clause:
                if v is True:
                    lits = None
                    break
                elif v is False:
                    lits = []
                    break
                elif isinstance(v, NegBoolView):
                    lits.append(-mapping[v._bv])
                elif isinstance(v, _BoolVarImpl):
                    lits.append(mapping[v])
                else:
                    raise ValueError(f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}")

            if lits is None:
                continue

            if canonical:
                lits.sort(key=abs)
            
            lits.append(0)

            if is_gcnf:
                out += f"{{{group}}} "

            out += " ".join(str(lit) for lit in lits) + "\n"

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

    return DimacsReader().read(fname)

def read_gdimacs(fname):
    return GDimacsReader().read(fname)

class DimacsReader:

    def __init__(self):
        self.typ = None
        self.clause = None
        self.clauses = []
        self.n_vars = None
        self.n_cls = None
        self.bvs = None

    def read(self, fname):
        with open(fname, "r") as f:
            for line in f.readlines():
                self.read_tokens(line.strip().split(" "))
        assert len(self.clauses) == self.n_clauses
        assert self.clause is None, "Untermined final clause"
        return self.to_model()


    def read_tokens(self, tokens):
        match tokens:
            case [] | ["c", *_]:
                pass  # skip empty/comment lines
            case ["p", "cnf", *params]:
                self.n_vars, self.n_clauses = [int(p) for p in params]
                self.bvs = cp.boolvar(shape=(self.n_vars,))
            case clause:
                assert self.n_vars is not None
                self.read_clause(clause)

    def to_model(self):
        return cp.Model([cp.any(clause) for clause in self.clauses])

    def read_clause(self, tokens):
        if self.clause is None:
            self.clause = []  # distinguish between empty clause and new clause not yet started
        for lit in tokens:
            lit = int(lit.strip())
            if lit == 0:
                self.clauses.append(self.clause)
                self.clause = None
            else:
                var = abs(lit) - 1
                assert var < self.n_vars, "Expected at most {self.n_vars} variables (from p-line) but found literal {i} in clause {line}"
                bv = self.bvs[var]
                self.clause.append(bv if lit > 0 else ~bv)

class GDimacsReader(DimacsReader):

    def read(self, fname):
        # TODO how to dedup?
        with open(fname, "r") as f:
            for line in f.readlines():
                self.read_tokens(line.strip().split(" "))
        assert len(self.clauses) == self.n_clauses
        assert self.clause is None, "Untermined final clause"
        return self.to_model()



    def __init__(self):
        super().__init__()
        self.groups = None
        self.n_groups = None

    def read_tokens(self, tokens):
        match tokens:
            case ["p", "gcnf", *params]:
                self.n_vars, self.n_clauses, self.n_groups = [int(p) for p in params]
                self.groups = self.n_clauses * []
                self.bvs = cp.boolvar(shape=(self.n_vars,))
                self.assumptions = cp.boolvar(shape=(self.n_groups,))
            case [group, *clause] if group.startswith("{"):
                self.groups.append(int(tokens[0][1:-1]))  # e.g. {1} 1 -2 3 0
                self.read_clause(clause)
            case tokens:
                super().read_tokens(tokens)

    def to_model(self):
        soft = []
        hard = []
        for k, clauses in itertools.groupby(enumerate(self.clauses), key=lambda clause: self.groups[clause[0]]):
            cnf = cp.all(cp.any(clause) for i, clause in clauses)
            (hard if k == 0 else soft).append(cnf)

        model, soft, assumptions = make_assump_model(soft, hard=hard)
        return model, soft, hard, assumptions 


