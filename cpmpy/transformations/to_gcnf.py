"""
Transform constraints to **Grouped Conjunctive Normal Form** (GCNF), where each
(soft) constraint becomes a group of clauses guarded by an assumption variable,
for use by MUS solvers. Optional hard constraints go in group 0.
"""

from typing import Optional, Sequence, cast

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl
from cpmpy.expressions.core import Expression, NestedBoolExprLike, Operator
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.int2bool import IntVarEnc
from cpmpy.transformations.to_cnf import to_cnf


def to_gcnf(
        constraints: Sequence[NestedBoolExprLike],
        hard: Optional[Sequence[NestedBoolExprLike]] = None,
        name: Optional[str] = None,
        csemap: Optional[CSEMap] = None,
        ivarmap: Optional[dict[str, IntVarEnc]] = None,
        encoding: str = "auto",
        disjoint: bool = True
    ) -> tuple[cp.Model, list[Expression], list[Expression], list[_BoolVarImpl]]:
    """
    Similar to `make_assump_model`, but the returned model is in (grouped) CNF.

    Follows https://satisfiability.org/competition/2011/rules.pdf. 
    To guarentee that the groups are disjoint, set `disjoint` to True.

    Arguments:
        constraints (Sequence[NestedBoolExprLike]): CPMpy constraints treated as soft (one GCNF group each)
        hard (Sequence[NestedBoolExprLike], optional): constraints that must be satisfied (group 0)
        name (str, optional): name of the model
        csemap (CSEMap, optional): CSE map
        ivarmap (dict[str, IntVarEnc], optional): IVAR map
        encoding (str): encoding of the model
        disjoint (bool): whether to make the groups disjoint

    Returns:
        tuple[cp.Model, list[Expression], list[Expression], list[_BoolVarImpl]]:
        model, soft constraints, hard constraints, and assumption variables
    """
    # deferred import: cpmpy.tools imports this module at package-import time
    from cpmpy.tools.explain.utils import make_assump_model

    model, soft_, assump = make_assump_model(
        constraints, hard=[] if hard is None else hard, name=name
    )

    cnf = to_cnf(model.constraints, encoding=encoding, csemap=csemap, ivarmap=ivarmap)

    groups: dict[_BoolVarImpl | bool, list[frozenset[_BoolVarImpl] | list[_BoolVarImpl]]] = {
        True: [],  # hard clauses
        **{a: [] for a in assump},  # assumption mapped to its soft clauses
    }

    # create a set for efficiency
    negative_assumptions = frozenset(~a for a in assump)

    for cpm_expr in cnf:
        for clause in _to_clauses(cpm_expr):
            # assumption var is often the first literal, but this is not guaranteed
            assumption = next((lit for lit in clause if lit in negative_assumptions), None)
            if assumption is None:
                groups[True].append(clause)
            else:
                # soft clause without assumption
                assump_var = cast(_BoolVarImpl, ~assumption)
                groups[assump_var].append(clause - {assumption})

    if disjoint:
        cl_db: set[frozenset[_BoolVarImpl]] = set()
        # to make groups disjoint..
        for group_clauses in groups.values():
            for i, clause_entry in enumerate(group_clauses):
                if not isinstance(clause_entry, frozenset):
                    continue
                if clause_entry in cl_db:
                    # replace duplicate clause `C` by new variable `f` and add `f -> C` to hard clauses
                    f = cp.boolvar()
                    group_clauses[i] = [f]
                    new_clause = frozenset([cast(_BoolVarImpl, ~f)]) | clause_entry
                    groups[True].append(new_clause)
                else:
                    cl_db.add(clause_entry)

    model = cp.Model(cnf)
    soft_out: list[Expression] = [cp.all(cp.any(c) for c in groups[a]) for a in assump]
    hard_out: list[Expression] = (
        [cp.all(cp.any(c) for c in groups[True])] if groups[True] else []
    )

    return (model, soft_out, hard_out, assump)


def _to_clauses(cons: Expression) -> list[frozenset[_BoolVarImpl]]:
    """
    Takes some CPMpy constraints in CNF + half-reifications and returns clauses as list of sets of literals

    Arguments:
        cons (Expression): CPMpy constraint

    Returns:
        list[frozenset[_BoolVarImpl]]: list of clauses
    """
    if isinstance(cons, _BoolVarImpl):
        return [frozenset([cons])]
    elif isinstance(cons, Operator):
        if cons.name == "or":
            return [frozenset(cons.args)]
        elif cons.name == "and":
            return [c_ for c in cons.args for c_ in _to_clauses(c)]
        elif cons.name == "->":
            return [frozenset({~cons.args[0]} | c) for c in _to_clauses(cons.args[1])]
        else:
            raise NotImplementedError(f"Unsupported Op {cons.name}")
    elif cons is True:
        return []
    elif cons is False:
        return [frozenset()]
    else:
        raise NotImplementedError(f"Unsupported constraint {cons}")
