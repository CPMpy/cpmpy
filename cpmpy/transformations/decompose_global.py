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
from typing import List, AbstractSet, Optional, Dict, Tuple, Any, Callable, cast
import numpy as np

from ..expressions.core import Expression, ListLike, BoolVal, Operator
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import NDVarArray, cpm_array


def decompose_in_tree(lst_of_expr: list[Expression],
                      supported: Optional[AbstractSet[str]] = None,
                      supported_reified: Optional[AbstractSet[str]] = None,
                      _toplevel=None, nested=False,
                      csemap: Optional[Dict[Expression, Expression]] = None,
                      decompose_custom: Optional[Dict[str, Callable]] = None) -> List[Expression]:
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

    if supported is None:
        supported = frozenset[str]()
    if supported_reified is None:
        supported_reified = frozenset[str]()

    newlist: List[Expression] = []
    todolist: List[Expression] = []  # these still need to be decomposed
    for expr in lst_of_expr:
        if isinstance(expr, GlobalConstraint) and expr.name not in supported:
            # toplevel/positive global constraint, decompose
            if decompose_custom is not None and expr.name in decompose_custom:
                exprs, toplevel_exprs = cast(Tuple[List[Expression], List[Expression]], decompose_custom[expr.name](expr))
            else:
                exprs, toplevel_exprs = expr.decompose()
            # both might contain globals too
            todolist.extend(exprs)
            if len(toplevel_exprs) > 0:
                todolist.extend(toplevel_exprs)
        elif isinstance(expr, (bool, np.bool_)):
            # TODO: violates type!!!
            newlist.append(BoolVal(expr))
        elif expr.has_subexpr():
            # decompose its arguments
            changed, newargs, rec_toplevel = _decompose_in_tree_args(expr.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
            if changed:
                expr = copy.copy(expr)
                expr.update_args(newargs)
                if len(rec_toplevel) > 0:
                    todolist.extend(rec_toplevel)
            newlist.append(expr)
        else:
            newlist.append(expr)

    # recurse on any newly generated toplevel expressions
    if len(todolist) == 0:
        return newlist
    return newlist + decompose_in_tree(todolist, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)


def decompose_objective(expr: Expression,
                        supported: Optional[AbstractSet[str]] = None,
                        supported_reified: Optional[AbstractSet[str]] = None,
                        csemap: Optional[Dict[Expression, Expression]] = None,
                        decompose_custom: Optional[Dict[str, Callable]]=None) -> Tuple[Expression, List[Expression]]:
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
    if supported is None:
        supported = frozenset[str]()
    if supported_reified is None:
        supported_reified = frozenset[str]()

    changed, newexprs, todo_toplevel = _decompose_in_tree_args((expr,), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
    if changed:
        assert len(newexprs) == 1, "decompose_objective: expected a single expression as decomposed objective but got {newexprs}"
        return newexprs[0], todo_toplevel
    else:
        return expr, []

def _decompose_in_tree_args(args: ListLike[Any],
                            supported: AbstractSet[str],
                            supported_reified: AbstractSet[str],
                            csemap: Optional[Dict[Expression, Expression]]=None,
                            decompose_custom:Optional[Dict[str, Callable]]=None) -> Tuple[bool, List[Any], List[Expression]]:
    """
    TODO: OUTDATED DOC!!
    Decompose any global constraint or global function not supported by the solver, recursive internal version.

    INTERNAL function, not guaranteed to remain backward compatible.

    :param lst_of_expr: list, tuple, :class:`~cpmpy.expressions.variables.NDVarArray`,
        or other sequence of expressions that may use global constraints or global functions.
    :param supported: a set of names of supported global constraints and global functions (will not be decomposed).
    :param supported_reified: a set of names of supported reified global constraints (those with Boolean return type only).
    :param csemap: a dictionary of 'expr: expr' mappings, for Common Subexpression Elimination

    :returns: ``(changed, newexpr, toplevel)`` where:
        - ``changed`` is True if a decomposition was done (or a recursive call changed something).
        - ``newexpr`` is the decomposed sequence (same length as ``lst_of_expr``).
        - ``toplevel`` is the list of auxiliary constraints to post at top level.
    """
    changed = False
    toplevel: List[Expression] = []
    newargs: List[Any] = []
    for arg in args:
        if isinstance(arg, Expression):
            orig_for_csemap: Optional[Expression] = None  # if set, will store the new arg in the csemap

            # a nested expression (its inside an args)
            if isinstance(arg, GlobalConstraint) and arg.name not in supported_reified:
                changed = True
                if (csemap is not None) and arg in csemap:  # have we decomposed it before?
                    newargs.append(csemap[arg])
                    continue
                else:
                    # nested global constraint, decompose
                    if decompose_custom is not None and arg.name in decompose_custom:
                        exprs, toplevel_exprs = cast(Tuple[List[Expression], List[Expression]], decompose_custom[arg.name](arg))
                    else:
                        exprs, toplevel_exprs = arg.decompose()

                    # replace arg by conjunction of decompose
                    orig_for_csemap = arg
                    arg = Operator("and", exprs)  # don't use cpm_all as for len=1 it shortcuts
                    if len(toplevel_exprs) > 0:
                        toplevel.extend(toplevel_exprs)
            elif isinstance(arg, GlobalFunction) and arg.name not in supported:
                changed = True
                if (csemap is not None) and arg in csemap:  # have we decomposed it before?
                    newargs.append(csemap[arg])
                    continue
                else:
                    # nested global function, decompose
                    orig_for_csemap = arg
                    # this is a bit awkward, but the decompose can return a new GlobFunc to decompose...
                    while isinstance(arg, GlobalFunction) and arg.name not in supported:
                        if decompose_custom is not None and arg.name in decompose_custom:
                            newarg, toplevel_exprs = cast(Tuple[Expression, List[Expression]], decompose_custom[arg.name](arg))
                        else:
                            newarg, toplevel_exprs = arg.decompose()
                        arg = newarg
                        if len(toplevel_exprs) > 0:
                            toplevel.extend(toplevel_exprs)

                        # TODO: violates type!!!
                        # apparently in #630 we decided that decompose may return an int (e.g. for element)...
                        # we should change that (Element constructor requires variable index; the []/__get__ override can still take anything)
                        if isinstance(arg, int):
                            # no need to recurse further, stop here
                            newargs.append(arg)
                            break
                    if isinstance(arg, int):
                        continue
            
            # if it has subexprs, decompose its arguments too
            if arg.has_subexpr():
                rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    changed = True
                    arg = copy.copy(arg)
                    arg.update_args(rec_newargs)
                    if len(rec_toplevel) > 0:
                        toplevel.extend(rec_toplevel)

            # very special case: "and" with single argument (from simple glob.decomp)
            if arg.name == "and" and len(arg.args) == 1:
                arg = cast(Expression, arg.args[0])

            if orig_for_csemap is not None and csemap is not None:
                csemap[orig_for_csemap] = arg
            newargs.append(arg)
        
        elif isinstance(arg, np.ndarray) and arg.dtype == object:
            if isinstance(arg, NDVarArray):
                if arg.has_subexpr():
                    rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg.flatten(), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                    if rec_changed:
                        changed = True
                        newargs.append(cpm_array(rec_newargs).reshape(arg.shape))
                    else:
                        newargs.append(arg)
                else:
                    newargs.append(arg)
            else:  # regular np.array
                if arg.dtype == object:
                    rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg.flatten(), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                    if rec_changed:
                        changed = True
                        newargs.append(np.array(rec_newargs).reshape(arg.shape))
                    else:
                        newargs.append(arg)
                else:
                    newargs.append(arg)
        else:
            # constants, variables, other expressions are left as is
            newargs.append(arg)

    return (changed, newargs, toplevel)
