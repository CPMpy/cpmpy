"""
Transform soft and hard constraints to **Grouped Conjunctive Normal Form** (GCNF), where each soft
constraint becomes a group of clauses guarded by an assumption variable, for use by MUS solvers.
"""

import cpmpy as cp
from ..expressions.variables import _BoolVarImpl
from ..expressions.core import Operator
from .to_cnf import to_cnf


def to_gcnf(soft, hard=None, name=None, csemap=None, ivarmap=None, encoding="auto", disjoint=True):
    """
    Similar to `make_assump_model`, but the returned model is in (grouped) CNF. Separately the soft clauses, hard clauses, and assumption variables. Follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint.
    """
    # deferred import: cpmpy.tools imports this module at package-import time
    from cpmpy.tools.explain.utils import make_assump_model

    model, soft_, assump = make_assump_model(soft, hard=[] if hard is None else hard, name=name)

    cnf = to_cnf(model.constraints, encoding=encoding, csemap=csemap, ivarmap=ivarmap)

    groups = {
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
                groups[~assumption].append(clause - {assumption})

    if disjoint:
        cl_db = set()
        # to make groups disjoint..
        for clauses in groups.values():
            for i, clause in enumerate(clauses):
                if clause in cl_db:
                    # replace duplicate clause `C` by new variable `f` and add `f -> C` to hard clauses
                    f = cp.boolvar()
                    clauses[i] = [f]
                    new_clause = frozenset([~f]) | clause
                    groups[True].append(new_clause)
                else:
                    cl_db.add(clause)

    model = cp.Model(cnf)
    soft = [cp.all(cp.any(c) for c in groups[a]) for a in assump]
    hard = [cp.all(cp.any(c) for c in groups[True])] if groups[True] else []

    return (model, soft, hard, assump)


def _to_clauses(cons):
    """Takes some CPMpy constraints in CNF + half-reifications and returns clauses as list of sets of literals"""
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
