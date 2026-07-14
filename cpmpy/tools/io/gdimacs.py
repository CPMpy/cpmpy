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

from cpmpy.expressions.core import Expression
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.transformations.to_gcnf import to_gcnf, _to_clauses
from cpmpy.transformations.get_variables import get_variables
from cpmpy.tools.explain.utils import make_assump_model
from cpmpy.tools.io.utils import _handle_loader_input


def write_gdimacs(
        soft: list[cp.Expression],
        hard: Optional[list[cp.Expression]] = None,
        assumptions: Optional[list[cp.BoolVar]] = None,
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

    Each soft constraint is assigned to a separate group (after transformation to CNF, 
    the resulting clauses are grouped by the soft constraint that they belong to).
    Hard constraints are placed in group ``{0}``.

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

    constraints = hard
    groups = list(zip(assumptions, soft)) if assumptions is not None else None

    is_gcnf = groups is not None
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
            skip_clause = False
            lits: list[int] = []
            for v in clause:
                if v is True:
                    skip_clause = True
                    break
                elif v is False:
                    break
                elif isinstance(v, NegBoolView):
                    lits.append(-mapping[v._bv])
                elif isinstance(v, _BoolVarImpl):
                    lits.append(mapping[v])
                else:
                    raise ValueError(
                        f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}"
                    )

            if skip_clause:
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

        nr_vars = None
        nr_cls = None
        clauses: list[list[int]] = []  # parsed clauses, as lists of literal ints
        cls_groups: list[int] = []  # group index of each parsed clause

        for raw in f:
            line = raw.strip()
            if line == "" or line.startswith("c"):
                continue  # skip empty and comment lines
            if line.startswith("p"):
                params = line.split()
                assert len(params) == 5, f"Expected p-header to be formed `p gcnf nr_vars nr_cls nr_groups` but got {line}"
                _, typ, nr_vars_text, nr_cls_text, _ = params
                if typ != "gcnf":
                    raise ValueError(f"Expected `gcnf` (i.e. GDIMACS) as file format, but got {typ} which is not supported.")
                nr_vars = int(nr_vars_text)
                nr_cls = int(nr_cls_text)
                continue

            assert nr_vars is not None and nr_cls is not None, "Expected p-line before first clause"
            group_text, *tokens = line.split()  # e.g. {1} 1 -2 3 0
            assert group_text.startswith("{") and group_text.endswith("}"), \
                f"Expected clause to be prefixed with its group, e.g. `{{1}} 1 -2 3 0`, but got {line}"
            group = int(group_text[1:-1])
            assert group >= 0, f"Group number must be non-negative, but got {group}"

            clause: list[int] = []
            for token in tokens:
                i = int(token)
                if i == 0:  # end of clause
                    assert len(clauses) < nr_cls, "Too many clauses"
                    clauses.append(clause)
                    cls_groups.append(group)
                    clause = []
                else:
                    assert abs(i) <= nr_vars, f"Expected at most {nr_vars} variables (from p-line) but found literal {i} in clause {line}"
                    clause.append(i)
            assert len(clause) == 0, "Expected clause to be terminated by 0"

        assert nr_vars is not None and nr_cls is not None, "Expected p-line in file"
        assert len(clauses) == nr_cls, "Number of clauses did not match the p-line"

        bvs = cp.boolvar(shape=(nr_vars,), name=var_name)

        # each consecutive run of clauses with the same group index forms one constraint;
        # group 0 is hard, every other group becomes one soft constraint
        soft: list[Expression] = []
        hard: list[Expression] = []
        for group, members in itertools.groupby(zip(cls_groups, clauses), key=lambda gc: gc[0]):
            cnf = cp.all(cp.any([bvs[abs(i)-1] if i > 0 else ~bvs[abs(i)-1] for i in cl]) for _, cl in members)
            (hard if group == 0 else soft).append(cnf)

        model, soft, assumptions = make_assump_model(soft, hard=hard, name=assumption_name)
        return model, soft, hard, assumptions
