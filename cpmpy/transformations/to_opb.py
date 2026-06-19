"""
Transform CPMpy expressions to OPB-compatible pseudo-Boolean constraints.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    transform
    transform_objective
"""

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list,simplify_boolean
from cpmpy.transformations.safening import no_partial_functions, safen_objective
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_objective
from cpmpy.transformations.reification import only_implies, only_bv_reifies
from cpmpy.transformations.linearize import (
    decompose_linear,
    decompose_linear_objective,
    linearize_constraint,
    only_positive_bv_wsum,
)
from cpmpy.transformations.int2bool import int2bool, _encode_int_var, _decide_encoding
from cpmpy.expressions.variables import _IntVarImpl, NegBoolView, _BoolVarImpl
from cpmpy.expressions.core import Operator, Comparison
from cpmpy.expressions.utils import is_num


def _normalized_comparison(lst_of_expr):
    """
    Convert a list of linear CPMpy expressions into OPB-compatible pseudo-Boolean constraints.

    Transforms a list of Boolean-linear CPMpy expressions (as output by `linearize_constraint`) into a list
    of OPB-normalized constraints, expressed as comparisons between weighted Boolean sums
    (using "wsum") and integer constants. Handles Boolean vars, reifications, implications,
    and ensures all equalities are decomposed into two inequalities.
    
    Args:
        lst_of_expr (list): List of CPMpy Boolean-linear expressions.

    Returns:
        list: List of normalized CPMpy `Comparison` objects representing pseudo-Boolean constraints.
    """
    newlist = []
    for cpm_expr in lst_of_expr:
        if isinstance(cpm_expr, cp.BoolVal):
            if cpm_expr.value() is False:
                raise NotImplementedError(f"Cannot transform {cpm_expr} to OPB constraint")
            continue  # trivially True, skip

        # single Boolean variable
        if isinstance(cpm_expr, _BoolVarImpl):
            cpm_expr = Operator("sum", [cpm_expr]) >= 1

        # implication
        if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            bv, subexpr = cpm_expr.args
            assert isinstance(subexpr, _BoolVarImpl), "Only bv -> bv should reach here, but got {subexpr}"
            cpm_expr = Operator("wsum", [[-1, 1], [bv, subexpr]]) >= 0
            newlist.append(cpm_expr)
            continue
        
        # Comparison, can be single Boolean variable or (weighted) sum of Boolean variables
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            if isinstance(lhs, (_BoolVarImpl, _IntVarImpl)):
                lhs = Operator("sum", [lhs])
            if lhs.name == "sum":
                lhs = Operator("wsum", [[1]*len(lhs.args), lhs.args])

            assert isinstance(lhs, Operator) and lhs.name == "wsum", f"Expected a wsum, but got {lhs}"

            # convert comparisons into >= constraints
            if cpm_expr.name == "==":
                newlist += _normalized_comparison([lhs <= rhs])
                newlist += _normalized_comparison([lhs >= rhs])
            elif cpm_expr.name == ">=":
                newlist.append(lhs >= rhs)
            elif cpm_expr.name == "<=":
                new_weights = [-w for w in lhs.args[0]]
                newlist.append(Operator("wsum", [new_weights, lhs.args[1]]) >= -rhs)
            else:
                raise ValueError(f"Unknown comparison {cpm_expr.name}")
        else:
            raise NotImplementedError(f"Expected a comparison, but got {cpm_expr}")

    return newlist


def _encode_lin_expr(ivarmap, xs, weights, encoding="auto"):
    """
    Encode a linear expression (weights * xs) to PB terms and domain constraints.

    Returns:
        (terms, constraints, k)
    """
    terms = []
    constraints = []
    k = 0

    for w, x in zip(weights, xs):
        if is_num(x):
            k += w * x
        elif isinstance(x, _BoolVarImpl):
            terms.append((w, x))
        else:
            enc, cons = _encode_int_var(ivarmap, x, _decide_encoding(x, None, encoding))
            constraints += cons
            new_terms, k_i = enc.encode_term(w)
            terms += new_terms
            k += k_i

    return terms, constraints, k


def transform(cpm_expr, csemap, ivarmap, encoding="auto"):
    """
        Transform a list of CPMpy expressions into a list of Pseudo-Boolean constraints.
    """

    cpm_cons = toplevel_list(cpm_expr)
    cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
    # Use linear-specific decompositions (e.g. AllDifferent.decompose_linear)
    # before linearization, consistent with MIP backends.
    cpm_cons = decompose_linear(
        cpm_cons,
        supported=frozenset(),
        supported_reified=frozenset(),
        csemap=csemap,
    )
    cpm_cons = simplify_boolean(cpm_cons)
    cpm_cons = flatten_constraint(cpm_cons, csemap=csemap)  # flat normal form
    cpm_cons = only_bv_reifies(cpm_cons, csemap=csemap)
    cpm_cons = only_implies(cpm_cons, csemap=csemap)
    cpm_cons = linearize_constraint(
        cpm_cons, supported=frozenset({"sum", "wsum"}), csemap=csemap
    )
    cpm_cons = int2bool(cpm_cons, ivarmap, encoding=encoding)

    return _normalized_comparison(cpm_cons)


def transform_objective(expr, csemap, ivarmap, encoding="auto"):
    """
    Transform a CPMpy objective expression into a weighted sum expression
    """

    # transform objective
    obj, safe_cons = safen_objective(expr)
    obj, decomp_cons = decompose_linear_objective(
        obj,
        supported=frozenset(),
        supported_reified=frozenset(),
        csemap=csemap,
    )
    obj, flat_cons = flatten_objective(obj, csemap=csemap)
    obj = only_positive_bv_wsum(obj)  # remove negboolviews

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
        raise NotImplementedError(f"OPB: Non supported objective {obj} (yet?)")

    terms, cons, k = _encode_lin_expr(ivarmap, xs, weights, encoding)

    # remove terms with coefficient 0 (`only_positive_coefficients_` may return them and RC2 does not accept them)
    terms = [(w, x) for w,x in terms if w != 0]  

    obj = Operator("wsum", [[w for w,x in terms], [x for w,x in terms]])
    return obj, const, safe_cons + decomp_cons + flat_cons