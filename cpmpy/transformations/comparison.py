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

    newlist = []
    for cpm_expr in constraints:

        if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            if not isinstance(cond, _BoolVarImpl): # expr -> bv
                idx = 0
            elif not isinstance(subexpr, _BoolVarImpl): # bv -> expr
                idx = 1
            else: # bv -> bv
                newlist.append(cpm_expr)
                continue

            new_arg, new_cons = _rewrite_comparison(cpm_expr.args[idx], supported=supported,csemap=csemap)
            if new_arg is not cpm_expr.args[idx]: # changed
                cpm_expr = copy.copy(cpm_expr) # shallow copy
                cpm_expr.args[idx] = new_arg                
                cpm_expr.update_args(cpm_expr.args) # XXX redundant? we know it's flat so no subexprs anyway
            
            newlist += [cpm_expr] + new_cons

            
        elif isinstance(cpm_expr, Comparison):

            lhs, rhs = cpm_expr.args
            if cpm_expr.name == "==" and is_boolexpr(lhs) and is_boolexpr(rhs): # reification
                if not isinstance(lhs, _BoolVarImpl):  # expr == bv
                    idx = 0
                elif not isinstance(rhs, _BoolVarImpl):  # bv == expr
                    idx = 1
                else: # bv == bv
                    newlist.append(cpm_expr)
                    continue

                # identical to the above, but keep for readability?
                new_arg, new_cons = _rewrite_comparison(cpm_expr.args[idx], supported=supported,csemap=csemap)
                if new_arg is not cpm_expr.args[idx]: # changed
                    cpm_expr = copy.copy(cpm_expr) # shallow copy
                    cpm_expr.args[idx] = new_arg
                    cpm_expr.update_args(cpm_expr.args) # XXX redundant? we know it's flat so no subexprs anyway

                newlist += [cpm_expr] + new_cons

            elif cpm_expr.name != "==": # numerical comparison
                new_expr, new_cons = _rewrite_comparison(cpm_expr, supported=supported,csemap=csemap)
                newlist += [new_expr] + new_cons
            
            else:
                newlist.append(cpm_expr) # equality constraint, keep

        else:
            # default, keep original
            newlist.append(cpm_expr)
                
    return newlist


def _rewrite_comparison(cpm_expr, supported=frozenset(), csemap=None):
    """
    Rewrite a comparison to an equality comparison, and a defining constraint.

    E.g., max(x,y,z) < p is rewritten to:
        max(x,y,z) == iv & iv < p

    :param cpm_expr: the comparison to rewrite
    :param csemap: the cse map to use
    :return: the rewritten comparison and the defining constraint
    """
    if not isinstance(cpm_expr, Comparison):
        return cpm_expr, []

    lhs, rhs = cpm_expr.args # flat, so expression will be on left hand side
    if cpm_expr.name != "==" and not isinstance(lhs, _NumVarImpl) and lhs.name not in supported:
        # lhs is unsupported, rewrite to `(LHS == A) & (A <op> RHS)`
        cpm_expr = copy.copy(cpm_expr)
        new_lhs, new_cons = get_or_make_var(lhs, csemap=csemap)
        cpm_expr.args[0] = new_lhs
        cpm_expr.update_args(cpm_expr.args) # XXX redundant? we know it's flat so no subexprs anyway
        return cpm_expr, new_cons
    
    return cpm_expr, []


