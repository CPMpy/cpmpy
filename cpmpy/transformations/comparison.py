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
from typing import Optional
from cpmpy.transformations.cse import CSEMap
from .flatten_model import get_or_make_var
from ..expressions.core import Comparison, Operator, Expression
from ..expressions.utils import is_boolexpr
from ..expressions.variables import _NumVarImpl, _BoolVarImpl

def only_numexpr_equality(constraints: list[Expression], supported=frozenset(), csemap: Optional[CSEMap]=None) -> list[Expression]:
    """
    Transforms unsupported non-equality comparisons of numeric expressions into an equality
    plus a comparison over an auxiliary variable.

    In **Flat Normal Form**, numeric comparisons have the form ``NumExpr <op> IV``
    (or a reification thereof). Solvers that only support ``NumExpr == IV`` need the
    remaining operators rewritten. For example, ``max(x, y, z) < p`` becomes
    ``[max(x, y, z) == iv, iv < p]``.

    Also applied to reified comparisons:

    - ``BoolVar -> (NumExpr <op> IV)`` :: ``BoolVar -> (NumExpr == A) & (A <op> IV)``
    - ``(NumExpr <op> IV) -> BoolVar`` :: ``(NumExpr == A) & (A <op> IV) -> BoolVar``
    - ``BoolVar == (NumExpr <op> IV)`` :: ``BoolVar == (NumExpr == A) & (A <op> IV)``

    Accepts a list of CPMpy expressions as input and returns a (new) list of CPMpy expressions.
    Input is expected to be free of partial functions and in Flat Normal Form
    (after :func:`~cpmpy.transformations.safening.no_partial_functions` and :func:`~cpmpy.transformations.flatten_model.flatten_constraint`).
    Output will also be in Flat Normal Form.

    Arguments:
        constraints (list[Expression]): list of CPMpy expressions
        supported (set[str]): names of numeric expressions that support all comparison operators
        csemap (Optional[CSEMap]): csemap
    """

    newlist: list[Expression] = []
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
                args = list(cpm_expr.args)
                args[idx] = new_arg                
                cpm_expr.update_args(args, has_subexpr=cpm_expr.has_subexpr())
            
            newlist.append(cpm_expr)
            newlist.extend(new_cons)

            
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

                # identical to the above, but kept for readability
                new_arg, new_cons = _rewrite_comparison(cpm_expr.args[idx], supported=supported,csemap=csemap)
                if new_arg is not cpm_expr.args[idx]: # changed
                    cpm_expr = copy.copy(cpm_expr) # shallow copy
                    args = list(cpm_expr.args)
                    args[idx] = new_arg
                    cpm_expr.update_args(args, has_subexpr=cpm_expr.has_subexpr())

                newlist.append(cpm_expr)
                newlist.extend(new_cons)

            elif cpm_expr.name != "==": # numerical comparison
                new_expr, new_cons = _rewrite_comparison(cpm_expr, supported=supported,csemap=csemap)
                newlist.append(new_expr)
                newlist.extend(new_cons)
            
            else: # equality constraint, keep, continue
                newlist.append(cpm_expr) 
        
        else: # default, keep original
            newlist.append(cpm_expr)
                
    return newlist


def _rewrite_comparison(cpm_expr: Expression, supported=frozenset(), csemap: Optional[CSEMap]=None) -> tuple[Expression, list[Expression]]:
    """
    Rewrite a non-equality comparison of an unsupported numeric expression into an
    equality plus a comparison over an auxiliary variable.

    For example, ``max(x, y, z) < p`` is rewritten to ``iv < p`` together with the
    defining constraint ``max(x, y, z) == iv``.

    INTERNAL function, not guaranteed to remain backward compatible.

    Arguments:
        cpm_expr (Expression): the comparison to rewrite
        supported (set[str]): names of numeric expressions that support all comparison operators
        csemap (Optional[CSEMap]): csemap

    Returns:
        tuple[Expression, list[Expression]]
        new_expr (Expression): the rewritten comparison (``A <op> IV``), or input cpm_expr if unchanged.
        new_cons (list[Expression]): the defining constraint(s)
    """
    if not isinstance(cpm_expr, Comparison):
        return cpm_expr, []

    lhs, rhs = cpm_expr.args # flat, so expression will be on left hand side
    if cpm_expr.name != "==" and not isinstance(lhs, _NumVarImpl) and lhs.name not in supported:
        # lhs is unsupported, rewrite to `(LHS == A) & (A <op> RHS)`
        new_expr = copy.copy(cpm_expr)
        new_lhs, new_cons = get_or_make_var(lhs, csemap=csemap)
        args = list(cpm_expr.args)
        args[0] = new_lhs
        new_expr.update_args(args, cpm_expr.has_subexpr())
        return new_expr, new_cons
    
    return cpm_expr, []


