"""
Transform constraints to **Conjunctive Normal Form** (i.e. an `and` of `or`s of literals, i.e. Boolean variables or their negation, e.g. from `x xor y` to `(x or ~y) and (~x or y)`) using a back-end encoding library and its transformation pipeline.
"""

import cpmpy as cp
from ..expressions.variables import _BoolVarImpl
from ..expressions.core import Operator
from ..solvers.pindakaas import CPM_pindakaas
from ..transformations.get_variables import get_variables
from cpmpy.tools.explain.marco import make_assump_model


def to_gcnf(soft, hard, name=None, csemap=None, ivarmap=None, encoding="auto"):
    """
    Or `make_assump_cnf`; returns an assumption CNF model, and separately the soft clauses, hard clauses, and assumption variables. Follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint.
    """

    model, soft_, assump = make_assump_model(soft, hard=hard, name=name)
    model_ = to_cnf(model.constraints, encoding=encoding, csemap=csemap, ivarmap=ivarmap)

    hard_clauses = []
    soft_clauses = []

    def add_gcnf_clause(lits):
        # ASSUMPTION: first literal should be the (negated) assumption variable
        if ~lits[0] in assump:
            soft_clauses.append(cp.any(lits[1:]))
        else:
            hard_clauses.append(cp.any(lits))

    for c in model_:
        if isinstance(c, _BoolVarImpl):
            add_gcnf_clause([c])
        elif isinstance(c, cp.expressions.variables.NDVarArray):
            for ci in c:
                add_gcnf_clause(ci.args)
        else:
            add_gcnf_clause(c.args)
    return cp.Model(model_), soft_clauses, hard_clauses, assump


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
        raise ImportError(f"Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use the `to_cnf` transformation")

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
