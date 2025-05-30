"""
  Transforms non-equality comparisons into equality comparisons as needed.
  
  Let <op> be one of `==` or `!=`, `<`, `<=`, `>`, `>=`. Numeric expressions in **Flat Normal Form** are of the kind:

    - `NumExpr <op> IV`
    - `BoolVar == NumExpr <op> IV`
    - `BoolVar -> NumExpr <op> IV`
    - `NumExpr <op> IV -> BoolVar`

  The `NumExpr` can be a sum, wsum or global function with a non-bool return type.
    
  This file implements:
    - :func:`only_numexpr_equality()`:    transforms `NumExpr <op> IV` (also reified) to `(NumExpr == A) & (A <op> IV)` if not supported
"""

import copy
from .flatten_model import get_or_make_var
from ..expressions.core import Comparison, Operator
from ..expressions.utils import is_boolexpr
from ..expressions.variables import _NumVarImpl, _BoolVarImpl

def only_numexpr_equality(constraints, supported=frozenset(), csemap=None):
    """
        Transforms ``NumExpr <op> IV`` to ``(NumExpr == A) & (A <op> IV)`` if not supported.
        Also for the reified uses of `NumExpr`

        :param supported:  a (frozen)set of expression names that supports all comparisons in the solver
    """

    # shallow copy (could support inplace too this way...)
    newcons = copy.copy(constraints)

    for i,cpm_expr in enumerate(newcons):

        if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            if not isinstance(cond, _BoolVarImpl): # expr -> bv
                res = only_numexpr_equality([cond], supported, csemap=csemap)
                if len(res) > 1:
                    newcons[i] = res[1].implies(subexpr)
                    newcons.insert(i, res[0])

            elif not isinstance(subexpr, _BoolVarImpl):  # bv -> expr
                res = only_numexpr_equality([subexpr], supported, csemap=csemap)
                if len(res) > 1:
                    newcons[i] = cond.implies(res[1])
                    newcons.insert(i, res[0])
            else: #bv -> bv
                pass


        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            if cpm_expr.name == "==" and is_boolexpr(lhs) and is_boolexpr(rhs): # reification, check recursively

                if not isinstance(lhs, _BoolVarImpl):  # expr == bv
                    res = only_numexpr_equality([lhs], supported, csemap=csemap)
                    if len(res) > 1:
                        newcons[i] = res[1] == rhs
                        newcons.insert(i, res[0])

                elif not isinstance(rhs, _BoolVarImpl):  # bv == expr
                    res = only_numexpr_equality([rhs], supported, csemap=csemap)
                    if len(res) > 1:
                        newcons[i] = lhs == res[1]
                        newcons.insert(i, res[0])
                else:  # bv == bv
                    pass

            elif cpm_expr.name != "==":
                # LHS <op> IV    with <op> one of !=,<,<=,>,>=
                lhs = cpm_expr.args[0]
                if not isinstance(lhs, _NumVarImpl) and lhs.name not in supported:
                    # LHS is unsupported for LHS <op> IV, rewrite to `(LHS == A) & (A <op> IV)`
                    (lhsvar, lhscons) = get_or_make_var(lhs, csemap=csemap)
                    # replace comparison by A <op> IV
                    newcons[i] = Comparison(cpm_expr.name, lhsvar, cpm_expr.args[1])
                    # add lhscon(s), which will be [(LHS == A)]
                    if len(lhscons) == 1:
                        newcons.insert(i, lhscons[0])
                    else:
                        assert (len(lhscons) == 0), "only_numexpr_eq: lhs surprisingly non-flat"
                        # can be 0 because of CSE

    return newcons
