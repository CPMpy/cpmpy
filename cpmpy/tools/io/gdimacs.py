#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## gdimacs.py
##
"""
Helper functions for loading CPMpy constraints from and writing to GDIMACS (Grouped DIMACS) format.

GDIMACS is a textual format to represent grouped CNF (GCNF) problems. 
However, there is no guarantee that the groups are disjoint.
More can be read about it here: 

- https://satisfiability.org/competition/2011/rules.pdf


Each clause is prefixed by a group index in curly braces; group ``{0}`` contains the hard clauses,
the other groups each represent one soft constraint. This format is used by MUS
(Minimal Unsatisfiable Subset) solvers to find minimal explanations for infeasibility.

E.g. the hard clause ``(a or b)``, the soft constraint ``(~a)`` and ``(~b)`` in group 1
(a group with two clauses) and the soft clause ``(a or ~b)`` in group 2 are represented as:

.. code-block:: text

    p gcnf 2 4 2
    {0} 1 2 0
    {1} -1 0
    {1} -2 0
    {2} 1 -2 0

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_gdimacs
    write_gdimacs
"""

import os
import re
import builtins
import itertools
from functools import partial
from typing import Callable, Optional, TextIO, Union

import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.transformations.to_gcnf import to_gcnf, _to_clauses
from cpmpy.transformations.get_variables import get_variables
from cpmpy.tools.explain.utils import make_assump_model
from cpmpy.tools.io.utils import _handle_loader_input


def write_gdimacs(
        soft,
        hard=None,
        path: Optional[Union[str, os.PathLike]] = None,
        encoding: str = "auto",
        disjoint: bool = True,
        canonical: bool = False,
        open: Callable = partial(builtins.open, mode="w"),
    ) -> str:
    """
    Writes CPMpy constraints to GDIMACS (Grouped DIMACS) format for MUS extraction.

    Uses the :func:`~cpmpy.transformations.to_gcnf.to_gcnf` transformation to convert
    soft and hard constraints into grouped CNF.

    Each soft constraint is assigned to a separate group. Hard constraints are placed in group ``{0}``.

    Arguments:
        soft: list of CPMpy constraints that can be violated (soft constraints)
        hard: list of CPMpy constraints that must be satisfied (hard constraints), optional
        path (str or os.PathLike, optional): file path to write the GDIMACS output to.
            If None, the GDIMACS string is only returned.
        encoding (str): the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
        disjoint (bool): if True, ensures groups are disjoint by introducing auxiliary variables.
            Required by some MUS solvers (e.g., MUSSER2) for correctness.
            We have seen an overhead of ~25% when enabled.
        canonical (bool): if True, outputs variables in sorted order and literals within clauses
            sorted by variable (positive before negative for same variable)
        open (Callable): callable to open the file for writing (default: builtin ``open`` in write mode).

    Returns:
        GDIMACS formatted string
    """
    _, soft, hard, assumptions = to_gcnf(soft, hard, encoding=encoding, disjoint=disjoint)
    return _write_clauses(hard, groups=zip(assumptions, soft), path=path, canonical=canonical, open=open)


def _write_clauses(constraints, groups=None, path=None, canonical=False, open=partial(builtins.open, mode="w")) -> str:
    """
    Helper function: constraints are assumed to be CNF (i.e. a list of conjunctions of hard clauses),
    groups are a list of tuples of (assumption variable, soft clauses).

    Check explicitly for ``groups=None``: ``groups=[]`` is a GCNF with only hard constraints
    (the ``{0}`` group), while ``groups=None`` is a plain CNF.
    """
    is_gcnf = groups is not None

    groups = list(groups) if groups is not None else []

    vars = get_variables([constraints] + [con for _, con in groups])

    if canonical:
        # natural sort: auto-named variables ("BV8" < "BV10") would break a plain
        # string sort across digit boundaries, making the output counter-dependent
        vars.sort(key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x.name)])

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

    if path is not None:
        with open(path) as f:
            f.write(out)

    return out


def load_gdimacs(
        gdimacs: Union[str, os.PathLike, TextIO],
        open: Callable = builtins.open,
        var_name: Optional[str] = None,
        assumption_name: Optional[str] = None,
    ):
    """
    Load CPMpy constraints from a GDIMACS (Grouped DIMACS) formatted file
    (strictly following https://satisfiability.org/competition/2011/rules.pdf,
    except that groups are allowed to have disjoint sets of clauses).

    Arguments:
        gdimacs (str or os.PathLike or TextIO):
            - A file path to a GDIMACS file, or
            - A string containing GDIMACS content directly, or
            - A TextIO object already open for reading
        open (Callable): callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.gcnf.xz``.
        var_name (str, optional): prefix for variable names
        assumption_name (str, optional): prefix for assumption variable names

    Returns:
        tuple (model, soft, hard, assumptions) where:
            - model: CPMpy Model with all constraints
            - soft: list of soft constraint groups
            - hard: list of hard constraints (from group 0)
            - assumptions: assumption variables for each soft constraint group
    """
    with _handle_loader_input(gdimacs, open=open) as f:
        return _GDimacsReader(var_name=var_name, assumption_name=assumption_name).read(f)


class _DimacsReader:
    """Line-based reader for DIMACS CNF content, base class for :class:`_GDimacsReader`."""

    def __init__(self, var_name=None):
        self.clauses = None
        self.clause_idx = 0
        self.bvs = None
        self.var_name = var_name

    def n_vars(self):
        return len(self.bvs)

    def n_clauses(self):
        return len(self.clauses)

    def read(self, f: TextIO):
        for line in f.readlines():
            self.read_tokens(line.strip().split(" "))
        assert self.clause_idx == self.n_clauses(), "Number of clauses did not match the p-line"
        return self.to_model()

    def initialize(self, n_vars, n_clauses):
        # note: do not use [[]] * n_clauses, it will have n_clauses references to the same list
        self.clauses = [[] for _ in range(n_clauses)]
        self.bvs = cp.boolvar(shape=(n_vars,), name=self.var_name)

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


class _GDimacsReader(_DimacsReader):
    """Line-based reader for GDIMACS (grouped CNF) content."""

    def __init__(self, var_name=None, assumption_name=None):
        super().__init__(var_name=var_name)
        self.groups = None
        self.assumption_name = assumption_name

    def read_tokens(self, tokens):
        match tokens:
            case [] | ["c", *_]:
                pass  # skip empty/comment lines
            case ["p", "gcnf", *params]:
                n_vars, n_clauses, n_groups = [int(p) for p in params]
                self.initialize(n_vars, n_clauses)
                self.groups = [None] * self.n_clauses()
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

        model, soft, assumptions = make_assump_model(soft, hard=hard, name=self.assumption_name)
        return model, soft, hard, assumptions
