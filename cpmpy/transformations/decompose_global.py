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
from typing import AbstractSet, Optional, Dict, Any, Callable, cast
import numpy as np

from .cse import CSEMap
from ..expressions.core import Expression, BoolVal, Operator
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import NDVarArray, cpm_array


def decompose_in_tree(lst_of_expr: list[Expression],
                      supported: Optional[AbstractSet[str]] = None,
                      supported_reified: Optional[AbstractSet[str]] = None,
                      _toplevel=None, nested=False,
                      csemap: Optional[CSEMap] = None,
                      decompose_custom: Optional[Dict[str, Callable]] = None) -> list[Expression]:
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

    todolist: list[Expression] = []  # these still need to be decomposed
    newlist: list[Expression] = []
    changed = False
    for expr in lst_of_expr:
        if isinstance(expr, (bool, np.bool_)):
            # TODO: violates type!!! from `.decompose()` functions that are not cleaned yet
            changed = True
            newlist.append(BoolVal(expr))
            continue

        # decompose arguments if needed
        if expr.has_subexpr():
            args_changed, args_new, args_toplevel = _decompose_in_tree_args(expr.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
            if args_changed:
                changed = True
                expr = copy.copy(expr)
                expr.update_args(args_new)
                if len(args_toplevel) > 0:
                    todolist.extend(args_toplevel)

        if isinstance(expr, GlobalConstraint) and expr.name not in supported:
            changed = True
            # toplevel/positive global constraint, decompose
            if decompose_custom is not None and expr.name in decompose_custom:
                exprs, toplevel_exprs = cast(tuple[list[Expression], list[Expression]], decompose_custom[expr.name](expr))
            else:
                exprs, toplevel_exprs = expr.decompose()
            # we merge the list toplevel rather than create an 'and', also means we can not store it in csemap
            # both might contain globals and need to be checked
            todolist.extend(exprs)
            if len(toplevel_exprs) > 0:
                todolist.extend(toplevel_exprs)
        else:
            newlist.append(expr)

    # recurse on any newly generated toplevel expressions
    if len(todolist) > 0:
        return newlist + decompose_in_tree(todolist, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
    elif changed:
        return newlist
    else:  # not changed
        return lst_of_expr


def decompose_objective(expr: Expression,
                        supported: Optional[AbstractSet[str]] = None,
                        supported_reified: Optional[AbstractSet[str]] = None,
                        csemap: Optional[CSEMap] = None,
                        decompose_custom: Optional[Dict[str, Callable]]=None) -> tuple[Expression, list[Expression]]:
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

def _decompose_in_tree_args(args: list[Any]|tuple[Any, ...],
                            supported: AbstractSet[str],
                            supported_reified: AbstractSet[str],
                            csemap: Optional[CSEMap]=None,
                            decompose_custom:Optional[Dict[str, Callable]]=None) -> tuple[bool, list[Any]|tuple[Any], list[Expression]]:
    """
    Well-typed recursive helper function to decompose unsupported global constraints
    and global functions in the arguments of an Expression.  

    INTERNAL function, not guaranteed to remain backward compatible.

    :param args: list of Expressions arguments (list[Any] | tuple[Any, ...])
    :param supported: a set of names of supported global constraints and global functions (will not be decomposed).
    :param supported_reified: a set of names of supported reified global constraints (those with Boolean return type only).
    :param csemap: a dictionary of 'expr: expr' mappings, for Common Subexpression Elimination

    Returns:  
        tuple[bool, list[Any], list[Expression]]: (changed, newargs, toplevel)  
        - ``changed`` is True if a decomposition was done (or a recursive call changed something).
        - ``newargs`` is the decomposed sequence (same length as ``args``).
        - ``toplevel`` is the list of auxiliary constraints to post at top level.
    """
    toplevel: list[Expression] = []
    newargs: list[Any] = []
    changed = False
    for arg in args:
        if isinstance(arg, Expression):
            unsupported_globalcons = False
            unsupported_globalfunc = False
            if isinstance(arg, GlobalConstraint):
                unsupported_globalcons = arg.name not in supported_reified  # as an argumented = nested
            elif isinstance(arg, GlobalFunction):
                unsupported_globalfunc = arg.name not in supported

            # csemap needs to check & store the expression before recursing over the arguments
            arg_orig: Optional[Expression] = None
            if csemap is not None and (unsupported_globalcons or unsupported_globalfunc):
                # shortcut, already computed
                decomp = csemap.get_decomposition(arg)
                if decomp is not None:
                    changed = True
                    newargs.append(decomp)
                    continue
                arg_orig = arg

            # if it has subexprs, decompose its arguments first
            if arg.has_subexpr():
                rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    changed = True
                    arg = copy.copy(arg)
                    arg.update_args(rec_newargs)
                    if len(rec_toplevel) > 0:
                        toplevel.extend(rec_toplevel)

            if unsupported_globalcons:
                changed = True
                if decompose_custom is not None and arg.name in decompose_custom:
                    exprs, toplevel_exprs = cast(tuple[list[Expression], list[Expression]], decompose_custom[arg.name](arg))
                else:
                    exprs, toplevel_exprs = cast(GlobalConstraint, arg).decompose()
                if len(toplevel_exprs) > 0:
                    toplevel.extend(toplevel_exprs)

                # the decomp may itself contain globals
                rec_changed, rec_exprs, rec_toplevel = _decompose_in_tree_args(exprs, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    exprs = cast(list[Expression], rec_exprs)
                    if len(rec_toplevel) > 0:
                        toplevel.extend(rec_toplevel)

                if len(exprs) == 1:
                    arg = exprs[0]
                else:
                    # replace arg by conjunction of decompose
                    arg = Operator("and", exprs)
                if arg_orig is not None and csemap is not None:
                    csemap.save_decomposition(arg_orig, arg)
            
            elif unsupported_globalfunc:
                changed = True
                # this is a bit awkward, but the decompose can return a new GlobFunc to decompose...
                while isinstance(arg, GlobalFunction) and arg.name not in supported:
                    if decompose_custom is not None and arg.name in decompose_custom:
                        newarg, toplevel_exprs = cast(tuple[Expression, list[Expression]], decompose_custom[arg.name](arg))
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
                        break
                if isinstance(arg, int):
                    if arg_orig is not None and csemap is not None:
                        csemap.save_decomposition(arg_orig, arg)
                    newargs.append(arg)
                    continue

                # the decomp may itself contain globals
                rec_changed, rec_arg, rec_toplevel = _decompose_in_tree_args((arg,), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    assert len(rec_arg) == 1, "decompose_in_tree_args: expected a single expression as decomposed global function but got {rec_arg}"
                    arg = rec_arg[0]
                    if len(rec_toplevel) > 0:
                        toplevel.extend(rec_toplevel)
                if arg_orig is not None and csemap is not None:
                    csemap.save_decomposition(arg_orig, arg)

            newargs.append(arg)
            continue

        elif isinstance(arg, np.ndarray) and arg.dtype == object:
            if isinstance(arg, NDVarArray):
                if arg.has_subexpr():
                    rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(tuple(arg.flat), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                    if rec_changed:
                        changed = True
                        newargs.append(cpm_array(rec_newargs).reshape(arg.shape))
                        if len(rec_toplevel) > 0:
                            toplevel.extend(toplevel_exprs)
                        continue
            else:  # regular np.array
                rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(tuple(arg.flat), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    changed = True
                    newargs.append(np.array(rec_newargs).reshape(arg.shape))
                    if len(rec_toplevel) > 0:
                        toplevel.extend(toplevel_exprs)
                    continue
        
        elif isinstance(arg, (list, tuple)):
            rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg, supported=supported,
                                                                             supported_reified=supported_reified,
                                                                             csemap=csemap,
                                                                             decompose_custom=decompose_custom)
            if rec_changed:
                changed = True
                newargs.append(rec_newargs)
                if len(rec_toplevel) > 0:
                    toplevel.extend(toplevel_exprs)
                continue
        
        # all the rest: not allowed to contain expressions
        newargs.append(arg)

    return (changed, newargs, toplevel)
