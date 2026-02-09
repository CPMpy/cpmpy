#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf, to_gcnf, _to_clauses
from cpmpy.transformations.get_variables import get_variables
from cpmpy.tools.explain.marco import make_assump_model


def write_gdimacs(soft, hard=None, name=None, fname=None, encoding="auto", disjoint=True, canonical=False):
    """
    Writes CPMpy constraints to GDIMACS (Grouped DIMACS) format for MUS extraction.

    Uses the "to_gcnf" transformation to convert soft and hard constraints into grouped CNF.
    The GDIMACS format follows the specification at:
    https://satisfiability.org/competition/2011/rules.pdf

    Each soft constraint is assigned to a separate group. Hard constraints are placed in group {0}.
    This format is used by MUS (Minimal Unsatisfiable Subset) solvers to find minimal explanations
    for infeasibility in SAT instances.

    :param soft: list of CPMpy constraints that can be violated (soft constraints)
    :param hard: list of CPMpy constraints that must be satisfied (hard constraints), optional
    :param name: prefix for assumption variable names, optional
    :param fname: file path to write the GDIMACS output
    :param encoding: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    :param disjoint: if True, ensures groups are disjoint by introducing auxiliary variables.
                    Required by some MUS solvers (e.g., MUSSER2) for correctness.
                    We have seen an overhead of ~25% when enabled.
    :param canonical: if True, outputs variables in sorted order and literals within clauses
                     sorted by variable (positive before negative for same variable)

    :return: GDIMACS formatted string
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
    """Helper function: constraints are assumped to be CNF (i.e. a list of conjunctions of hard clauses), groups are a list of tuples of (assumption variable, soft clauses)"""

    # Check explicitly for None, since groups=[] is a GCNF with only hard constraints (the {0} group), while groups=None should be a CNF
    is_gcnf = groups is not None

    groups = list(groups) if groups is not None else []

    vars = get_variables([constraints] + [con for _, con in groups])

    if canonical:
        vars.sort(key=lambda x: x.name)

    mapping = {v: i for i, v in enumerate(vars, start=1)}

    out = ""
    n_clauses = 0
    for group, cons in itertools.chain(
        zip(itertools.repeat(0), constraints),
        ((i, constraint) for i, (_, constraint) in enumerate(groups, start=1)),
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
                    raise ValueError(
                        f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}"
                    )

            if lits is None:
                continue

            if canonical:
                lits.sort(key=lambda x: (abs(x), x))

            lits.append(0)

            if is_gcnf:
                out += f"{{{group}}} "

            out += " ".join(str(lit) for lit in lits) + "\n"

    out = (
        f"p {'g' if is_gcnf else ''}cnf {len(vars)} {n_clauses}{f' {len(groups)}' if is_gcnf else ''}\n" + out
    )

    if fname is not None:
        with open(fname, "w") as f:
            f.write(out)

    return out


def read_dimacs(fname, name=None):
    """
    Read a CPMpy model from a DIMACS formatted file strictly following the specification:
    https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html

    .. note::
        The p-line has to denote the correct number of variables and clauses

    :param fname: the name of the DIMACS file
    :param sep: optional, separator used in the DIMACS file, will try to infer if None
    """

    return DimacsReader(name=name).read(fname)


def read_gdimacs(fname):
    """
    Read a CPMpy model from a GDIMACS (Grouped DIMACS) formatted file.

    GDIMACS extends DIMACS CNF format by grouping clauses, typically used for MUS extraction.
    Format specification: https://satisfiability.org/competition/2011/rules.pdf

    :param fname: path to the GDIMACS file

    :return: tuple (model, soft, hard, assumptions) where:
             - model: CPMpy Model with all constraints
             - soft: list of soft constraint groups
             - hard: list of hard constraints (from group 0)
             - assumptions: assumption variables for each soft constraint group
    """
    return GDimacsReader().read(fname)


class DimacsReader:
    def __init__(self, name=None):
        self.clauses = None
        self.clause_idx = 0
        self.bvs = None
        self.name = name

    def n_vars(self):
        return len(self.bvs)

    def n_clauses(self):
        return len(self.clauses)

    def read(self, fname):
        with open(fname, "r") as f:
            for line in f.readlines():
                self.read_tokens(line.strip().split(" "))
        assert self.clause_idx == self.n_clauses(), "Number of clauses did not match the p-line"
        return self.to_model()

    def initialize(self, n_vars, n_clauses):
        # note: do not use [[]] * n_clauses, it will have n_clauses references to the same list
        self.clauses = [[] for _ in range(n_clauses)]
        self.bvs = cp.boolvar(shape=(n_vars,), name=None if self.name is None else self.name)

    def read_tokens(self, tokens):
        match tokens:
            case [] | ["c", *_]:
                pass  # skip empty/comment lines
            case ["p", "cnf", *params]:
                n_vars, n_clauses = [int(p) for p in params]
                self.initialize(n_vars, n_clauses)
            case clause:
                assert self.clauses is not None
                self.read_clause(clause)

    def to_model(self):
        return cp.Model([cp.any(clause) for clause in self.clauses])

    def read_clause(self, tokens):
        for lit in tokens:
            lit = int(lit.strip())
            if lit == 0:
                self.clause_idx += 1
            else:
                assert self.clause_idx < self.n_clauses(), "Too many clauses"

                var = abs(lit) - 1
                assert var < self.n_vars(), (
                    f"Expected at most {self.n_vars()} variables (from p-line) but found literal {lit} in clause {' '.join(tokens)}"
                )
                bv = self.bvs[var]
                self.clauses[self.clause_idx].append(bv if lit > 0 else ~bv)


class GDimacsReader(DimacsReader):
    def __init__(self):
        super().__init__()
        self.groups = None

    def n_groups(self):
        return len(self.groups)

    def read_tokens(self, tokens):
        match tokens:
            case ["p", "gcnf", *params]:
                n_vars, n_clauses, n_groups = [int(p) for p in params]
                self.initialize(n_vars, n_clauses)
                self.groups = [None] * self.n_clauses()
                self.assumptions = cp.boolvar(shape=(self.n_groups(),))
            case [group, *clause] if group.startswith("{"):
                group_num = int(tokens[0][1:-1])  # e.g. {1} 1 -2 3 0
                assert group_num >= 0, f"Group number must be non-negative, but got {group_num}"
                self.groups[self.clause_idx] = group_num
                self.read_clause(clause)

    def to_model(self):
        soft = []
        hard = []
        for k, clauses in itertools.groupby(
            enumerate(self.clauses), key=lambda clause: self.groups[clause[0]]
        ):
            cnf = cp.all(cp.any(clause) for i, clause in clauses)
            (hard if k == 0 else soft).append(cnf)

        model, soft, assumptions = make_assump_model(soft, hard=hard)
        return model, soft, hard, assumptions
