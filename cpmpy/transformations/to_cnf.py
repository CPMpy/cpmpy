"""
Transform constraints to **Conjunctive Normal Form** (i.e. an `and` of `or`s of literals, i.e. Boolean variables or their negation, e.g. from `x xor y` to `(x or ~y) and (~x or y)`) using a back-end encoding library and its transformation pipeline.
"""

import time

from tqdm import tqdm
import cpmpy as cp
from ..expressions.variables import _BoolVarImpl
from ..expressions.core import Operator
from ..solvers.pindakaas import CPM_pindakaas
from cpmpy.tools.explain.marco import make_assump_model
from cpmpy.expressions.utils import all_pairs


def to_cnf(constraints, csemap=None, ivarmap=None, encoding="auto"):
    """
    Converts all constraints into **Conjunctive Normal Form**

    Arguments:
        constraints:    list[Expression] or Operator
        csemap:         `dict()` used for CSE
        ivarmap:        `dict()` used to map integer variables to their encoding (usefull for finding the values of the now-encoded integer variables)
        encoding:       the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    Returns:
        Equivalent CPMpy constraints in CNF, and the updated `ivarmap`
    """
    if not CPM_pindakaas.supported():
        raise ImportError(
            f"Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use the `to_cnf` transformation"
        )

    import pindakaas as pdk

    slv = CPM_pindakaas()
    slv.encoding = encoding

    if ivarmap is not None:
        slv.ivarmap = ivarmap
    slv._csemap = csemap

    # the encoded constraints (i.e. `PB`s) will be added to this `pdk.CNF` object
    slv.pdk_solver = pdk.CNF()

    # add, transform, and encode constraints into CNF/clauses
    slv += constraints

    # now we read the pdk.CNF back to cpmpy constraints by mapping from `pdk.Lit` to CPMpy lit
    cpmpy_vars = {str(slv.solver_var(x).var()): x for x in slv._int2bool_user_vars()}

    # if a user variable `x` does not occur in any clause, it should be added as `x | ~x`
    free_vars = set(cpmpy_vars.values())

    def to_cpmpy_clause(clause):
        """Lazily convert `pdk.CNF` to CPMpy."""
        for lit in clause:
            x = str(lit.var())
            if x not in cpmpy_vars:
                cpmpy_vars[x] = cp.boolvar()
            elif cpmpy_vars[x] in free_vars:  # cpmpy_vars[x] is only in free_vars if it existed before
                free_vars.remove(cpmpy_vars[x])
            yield ~cpmpy_vars[x] if lit.is_negated() else cpmpy_vars[x]

    clauses = []
    clauses += (cp.any(to_cpmpy_clause(clause)) for clause in slv.pdk_solver.clauses())
    clauses += ((x | ~x) for x in free_vars)  # add free variables so they are "known" by the CNF

    return clauses


def to_gcnf(soft, hard=None, name=None, csemap=None, ivarmap=None, encoding="auto", disjoint=False):
    """
    Similar to `make_assump_model`, but the returned model is in (grouped) CNF. Separately the soft clauses, hard clauses, and assumption variables. Follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint.
    """

    model, soft_, assump = make_assump_model(soft, hard=hard, name=name)

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
            return [frozenset(~cons.args[0], *c) for c in _to_clauses(cons.args[1])]
        else:
            raise NotImplementedError(f"Unsupported Op {cons.name}")
    elif cons is True:
        return []
    elif cons is False:
        return [frozenset()]
    else:
        raise NotImplementedError(f"Unsupported constraint {cons}")
