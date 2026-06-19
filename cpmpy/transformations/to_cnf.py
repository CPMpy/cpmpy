"""
Transform constraints to **Conjunctive Normal Form** (i.e. an `and` of `or`s of literals, i.e. Boolean variables or their negation, e.g. from `x xor y` to `(x or ~y) and (~x or y)`) using a back-end encoding library and its transformation pipeline.
"""

import cpmpy as cp
from ..solvers.pindakaas import CPM_pindakaas

from cpmpy.expressions.variables import NegBoolView, _IntVarImpl

from cpmpy.transformations.safening import safen_objective
from cpmpy.transformations.flatten_model import flatten_objective
from cpmpy.transformations.linearize import decompose_linear_objective, only_positive_coefficients_
from cpmpy.transformations.int2bool import _encode_lin_expr
from cpmpy.transformations.cse import CSEMap


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
    if csemap is not None:
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

def to_cnf_objective(expr, encoding="auto", csemap=None, ivarmap=None):
    """
        Transform objective into weighted Boolean literals plus helper constraints.

        :param csemap: optional shared CSE cache (populated in-place)
        :param ivarmap: optional shared integer variable encoding dict (populated in-place)

        Returns:
            (weights, xs, const, extra_cons)
    """
    if csemap is None:
        csemap = CSEMap()
    if ivarmap is None:
        ivarmap = dict()
    obj, safe_cons = safen_objective(expr)
    obj, decomp_cons = decompose_linear_objective(
        obj,
        supported=frozenset(),
        supported_reified=frozenset(),
        csemap=csemap,
    )
    obj, flat_cons = flatten_objective(obj, csemap=csemap)

    weights, xs, const = [], [], 0
    # we assume obj is a var, a sum or a wsum (over int and bool vars)
    if isinstance(obj, _IntVarImpl) or isinstance(obj, NegBoolView):  # includes _BoolVarImpl
        weights = [1]
        xs = [obj]
    elif obj.name == "sum":
        xs = obj.args
        weights = [1] * len(xs)
    elif obj.name == "wsum":
        weights, xs = obj.args
    else:
        raise NotImplementedError(f"DIMACS: Non supported objective {obj} (yet?)")

    terms, enc_cons, k = _encode_lin_expr(ivarmap, xs, weights, encoding, csemap=csemap)
    const += k

    extra_cons = safe_cons + decomp_cons + flat_cons + enc_cons

    # remove terms with coefficient 0 (`only_positive_coefficients_` may return them and RC2 does not accept them)
    terms = [(w, x) for w, x in terms if w != 0]
    if len(terms) == 0:
        return [], [], const, extra_cons

    ws, xs = zip(*terms)  # unzip
    new_weights, new_xs, k = only_positive_coefficients_(ws, xs)
    const += k

    return list(new_weights), list(new_xs), const, extra_cons
