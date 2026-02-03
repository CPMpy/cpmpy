"""
Functions to decompose global constraints and global functions not supported by the solver.

This transformation is necessary for all non-CP solvers, and also used to decompose 
global constraints and global functions not implemented in a CP-solver.

While a solver may natively support a global constraint, it may not support it natively in a reified context.
In this case, we will als also decompose the global constraint.

For numerical global functions, we will only decompose them if they are not supported in non-reified context.
Even if the solver does not explicitely support them in a subexpression, 
we can rewrite them using func:`cpmpy.transformations.reification.reify_rewrite` to a non-reified version when the function is total.
E.g., bv <-> max(a,b,c) >= 4 can be rewritten as [bv <-> IV0 >= 4, IV0 == max(a,b,c)]

Unsupported global constraints and global functions are decomposed in-place and the resulting set of constraints
is wrapped in a conjunction.
E.g., x + ~AllDifferent(a,b,c) >= 2 is decomposed into x + ~((a) != (b) & (a) != (c) & (b) != (c)) >= 2
This allows to post the decomposed expression tree to the solver if it supports it (e.g., SMT-solvers, MiniZinc, CPO)
"""

import copy
from typing import List, Set, Optional, Dict, Tuple, Union, Sequence
import numpy as np

from .normalize import toplevel_list
from ..expressions.core import Expression
from ..expressions.variables import NDVarArray
from ..expressions.utils import is_any_list
from ..expressions.python_builtins import all as cpm_all


def decompose_in_tree(lst_of_expr: Sequence[Expression], supported: Set[str] = set(), supported_reified: Set[str] = set(), _toplevel=None, nested=False, csemap: Optional[Dict[Expression, Expression]] = None) -> List[Expression]:
    """
    Decomposes global constraint or global function not supported by the solver.

    Accepts a list of CPMpy expressions as input and returns a (new) list of CPMpy expressions.

    :param lst_of_expr: list of CPMpy expressions that may contain global constraints or global functions.
    :param supported: a set of names of supported global constraints and global functions (will not be decomposed).
    :param supported_reified: a set of names of supported reified global constraints (those with Boolean return type only).
    :param _toplevel: DEPRECATED
    :param nested: DEPRECATED
    :param csemap: a dictionary of 'expr: expr' mappings, for Common Subexpression Elimination

    Supported numerical global functions remain in the expression tree as is. They can be rewritten using
    :func:`cpmpy.transformations.reification.reify_rewrite`
    E.g. ``bv -> NumExpr <comp> Var/Const`` will then be rewritten as  ``[bv -> IV0 <comp> Var/Const, NumExpr == IV0]``.
    """
    assert _toplevel is None, "decompose_in_tree: argument '_toplevel' is deprecated, do not use/modify it"
    assert nested is False, "decompose_in_tree: argument 'nested' is deprecated, do not use/modify it"

    changed, newlst_of_expr, todo_toplevel = _decompose_in_tree(lst_of_expr, supported=supported, supported_reified=supported_reified, is_toplevel=True, csemap=csemap)
    if not changed:
        return lst_of_expr

    # new toplevel constraints may need to be decomposed too
    while len(todo_toplevel):
        changed, decomp, next_toplevel = _decompose_in_tree(todo_toplevel, supported=supported, supported_reified=supported_reified, is_toplevel=True, csemap=csemap)
        if not changed:
            newlst_of_expr.extend(todo_toplevel)
            break

        # changed, loop again
        newlst_of_expr.extend(decomp)
        todo_toplevel = next_toplevel # decompositions may have introduced nested lists or ands

    return newlst_of_expr


def decompose_objective(expr: Expression, supported: Set[str] = set(), supported_reified: Set[str] = set(), csemap: Optional[Dict[Expression, Expression]] = None) -> Tuple[Expression, List[Expression]]:
    """
    Decompose any global constraint or global function not supported by the solver
    in the objective function expression (numeric or global).

    Accepts a single objective expression and returns the decomposed expression plus
    a list of auxiliary constraints to add as model constraints.

    :param expr: objective expression (e.g. ``min(x)``, ``sum(arr)``).
    :param supported: a set of names of supported global constraints and global functions (will not be decomposed).
    :param supported_reified: a set of names of supported reified global constraints (those with Boolean return type only).
    :param csemap: a dictionary of 'expr: expr' mappings, for Common Subexpression Elimination

    :returns: ``(decomp_expr, toplevel)`` where ``decomp_expr`` is the decomposed
        objective and ``toplevel`` is the list of auxiliary constraints.

    .. warning::
        The returned ``toplevel`` list may itself contain global constraints or
        functions. When adding these to the solver, the solver should still
        decompose them.
    """
    assert isinstance(expr, Expression), "decompose_objective: expected a single expression as objective but got {expr}"

    changed, newexpr, todo_toplevel = _decompose_in_tree((expr,), supported=supported, supported_reified=supported_reified, is_toplevel=False, csemap=csemap)
    if not changed:
        return expr, []

    assert len(newexpr) == 1, "decompose_objective: expected a single expression as decomposed objective but got {newexpr}"
    return newexpr[0], todo_toplevel


def _decompose_in_tree(lst_of_expr: Union[Sequence[Expression], NDVarArray], supported: Set[str], supported_reified: Set[str], is_toplevel: bool, csemap: Optional[Dict[Expression, Expression]]) -> Tuple[bool, List[Expression], List[Expression]]:
    """
    Decompose any global constraint or global function not supported by the solver, recursive internal version.

    INTERNAL function, not guaranteed to remain backward compatible.

    :param lst_of_expr: list, tuple, :class:`~cpmpy.expressions.variables.NDVarArray`,
        or other sequence of expressions that may use global constraints or global functions.
    :param supported: a set of names of supported global constraints and global functions (will not be decomposed).
    :param supported_reified: a set of names of supported reified global constraints (those with Boolean return type only).
    :param is_toplevel: whether ``lst_of_expr`` is the toplevel list of constraints.
        If False, ``lst_of_expr`` is an argument to another expression and its global constraints must support reification.
    :param csemap: a dictionary of 'expr: expr' mappings, for Common Subexpression Elimination

    :returns: ``(changed, newexpr, toplevel)`` where:
        - ``changed`` is True if a decomposition was done (or a recursive call changed something).
        - ``newexpr`` is the decomposed sequence (same length as ``lst_of_expr``).
        - ``toplevel`` is the list of auxiliary constraints to post at top level.
    """
    changed = False
    newlist: List[Expression] = []
    toplevel: List[Expression] = []
    has_csemap = (csemap is not None)

    for expr in lst_of_expr:
        if is_any_list(expr):
            assert not is_toplevel, "Lists in lists is only allowed for arguments (e.g. of global constrainst)." \
                                    "Make sure to run func:`cpmpy.transformations.normalize.toplevel_list` first."

            if isinstance(expr, NDVarArray) and not expr.has_subexpr():
                pass  # no subexpressions, nothing to do
            elif isinstance(expr, np.ndarray) and expr.dtype != object:
                pass  # only constants, nothing to do
            else:
                rec_changed, rec_expr, rec_toplevel = _decompose_in_tree(expr, supported=supported, supported_reified=supported_reified, is_toplevel=False, csemap=csemap)
                if rec_changed:
                    expr = rec_expr
                    toplevel.extend(rec_toplevel)
                    changed = True
            newlist.append(expr)
            continue

        # if an expression, decompose its arguments first
        if isinstance(expr, Expression) and expr.has_subexpr():
            rec_changed, newargs, rec_toplevel = _decompose_in_tree(expr.args, supported=supported, supported_reified=supported_reified, is_toplevel=False, csemap=csemap)
            if rec_changed:
                expr = copy.copy(expr)
                expr.update_args(newargs)
                toplevel.extend(rec_toplevel)
                changed = True

        if hasattr(expr, "decompose"):  # it is a global function or global constraint
            is_supported = expr.name in supported
            if not is_toplevel and expr.is_bool():
                # argument to another expression, only possible if supported reified
                is_supported = expr.name in supported_reified

            if is_supported is False:
                if has_csemap and expr in csemap:
                    # we might have already decomposed it previously
                    newexpr = csemap[expr]
                else:
                    newexpr, define = expr.decompose()
                    toplevel.extend(define)

                    # decomposed constraints may introduce new globals
                    if isinstance(newexpr, list):  # globals return a list instead of a single expression (TODO: change?)
                        rec_changed, rec_newexpr, rec_toplevel = _decompose_in_tree(newexpr, supported=supported, supported_reified=supported_reified, is_toplevel=is_toplevel, csemap=csemap)
                        if rec_changed:
                            newexpr = rec_newexpr
                            toplevel.extend(rec_toplevel)
                        newexpr = cpm_all(newexpr)  # make the list a single expression
                    else:
                        rec_changed, rec_lst_newexpr, rec_toplevel = _decompose_in_tree((newexpr,), supported=supported, supported_reified=supported_reified, is_toplevel=is_toplevel, csemap=csemap)
                        if rec_changed:
                            newexpr = rec_lst_newexpr[0]
                            toplevel.extend(rec_toplevel)

                    if has_csemap:
                        csemap[expr] = newexpr

                newlist.append(newexpr)
                changed = True
                continue

        # constants, variables, other expressions are left as is
        newlist.append(expr)

    assert is_toplevel or len(newlist) == len(lst_of_expr), f"Nested decomposition should not change the number of expressions\n{lst_of_expr}\n{newlist}"
    return (changed, newlist, toplevel)
