"""
Transform constraints to **Conjunctive Normal Form** (i.e. an `and` of `or`s of literals, i.e. Boolean variables or their negation, e.g. from `x xor y` to `(x or ~y) and (~x or y)`) using a back-end encoding library and its transformation pipeline.
"""

import itertools
import cpmpy as cp
import pindakaas as pdk
from ..solvers.pindakaas import CPM_pindakaas
from ..transformations.get_variables import get_variables


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
    if ivarmap is not None:
        slv.ivarmap = ivarmap
    slv._csemap = csemap

    # the encoded constraints (i.e. `PB`s) will be added to this `pdk.CNF` object
    slv.pdk_solver = pdk.CNF()

    # however, we bypass `pindakaas` for simple clauses
    clauses = []
    slv._add_clause = lambda cpm_expr: clauses.append(cp.any(cpm_expr))

    # add, transform, and encode constraints into CNF/clauses
    slv += constraints

    # now we read the pdk.CNF back to cpmpy constraints by mapping from `pdk.Lit` to CPMpy lit
    cpmpy_vars = {str(slv.solver_var(x).var()): x for x in slv._int2bool_user_vars()}

    # if a user variable `x` does not occur in any clause, they should be added as `x | ~x`
    free_vars = set(cpmpy_vars.values()) - set(get_variables(clauses))

    def to_cpmpy_clause(clause):
        """Lazily convert `pdk.CNF` to CPMpy."""
        for lit in clause:
            x = str(lit.var())
            if x not in cpmpy_vars:
                cpmpy_vars[x] = cp.boolvar()
            y = cpmpy_vars[x]
            try:
                free_vars.remove(y)
            except KeyError:
                pass
            if lit.is_negated():
                yield ~y
            else:
                yield y

    clauses += (cp.any(to_cpmpy_clause(clause)) for clause in slv.pdk_solver.clauses())
    clauses += ((x | ~x) for x in free_vars)  # add free variables so they are "known" by the CNF

    return clauses, slv.ivarmap
