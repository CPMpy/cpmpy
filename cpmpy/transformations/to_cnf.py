"""
Transform constraints to **Conjunctive Normal Form** (i.e. an `and` of `or`s of literals, i.e. Boolean variables or their negation, e.g. from `x xor y` to `(x or ~y) and (~x or y)`) using a back-end encoding library and its transformation pipeline.
"""

import itertools
import cpmpy as cp
import pindakaas as pdk
from ..solvers.pindakaas import CPM_pindakaas


def to_cnf(constraints, csemap=None, ivarmap=None):
    """
    Converts all constraints into **Conjunctive Normal Form**

    Arguments:
        constraints:    list[Expression] or Operator
        csemap:         `dict()` used for CSE
        ivarmap:        `dict()` used to map integer variables to their encoding (usefull for finding the values of the now-encoded integer variables)
    Returns:
        Equivalent CPMpy constraints in CNF, and the updated `ivarmap`
    """
    slv = CPM_pindakaas()
    slv.pdk_solver = pdk.CNF()
    if ivarmap is not None:
        slv.ivarmap = ivarmap
    slv._csemap = csemap
    slv += constraints
    return to_cpmpy_cnf(slv), slv.ivarmap


def to_cpmpy_cnf(slv):
    # from pdk var to cpmpy var
    cpmpy_vars = {str(slv.solver_var(x).var()): x for x in slv._int2bool_user_vars()}

    def to_cpmpy_clause(clause):
        for lit in clause:
            x = str(lit.var())
            if x not in cpmpy_vars:
                cpmpy_vars[x] = cp.boolvar()
            y = cpmpy_vars[x]
            if lit.is_negated():
                yield ~y
            else:
                yield y

    return list(
        itertools.chain(
            (x | ~x for x in cpmpy_vars.values()),  # ensure all vars are "known" in the CNF
            (cp.any(to_cpmpy_clause(clause)) for clause in slv.pdk_solver.clauses()),
        )
    )
