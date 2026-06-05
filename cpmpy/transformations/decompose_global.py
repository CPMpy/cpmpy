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
from typing import AbstractSet, Optional, Dict, Any, Callable, Protocol, cast, overload
import numpy as np


from .cse import CSEMap
from ..expressions.core import Expression, BoolVal, Operator
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import NDVarArray, cpm_array
from ..expressions.python_builtins import all as cpm_all

class CustomDecomp(Protocol):
    @overload
    def __call__(self, expr: GlobalConstraint, /) -> tuple[list[Expression], list[Expression]]: ...
    @overload
    def __call__(self, expr: GlobalFunction, /) -> tuple[Expression, list[Expression]]: ...

def decompose_in_tree(lst_of_expr: list[Expression],
                      supported: Optional[AbstractSet[str]] = None,
                      supported_reified: Optional[AbstractSet[str]] = None,
                      _toplevel=None, nested=False,
                      csemap: Optional[CSEMap] = None,
                      decompose_custom: Optional[Dict[str, CustomDecomp]] = None,
                      decompose_custom_positive: Optional[Dict[str, CustomDecomp]] = None) -> list[Expression]:
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
        if isinstance(expr, GlobalConstraint) and expr.name not in supported:
            # toplevel/positive global constraint, decompose
            changed = True
            if csemap is not None:
                decomp = csemap.get_decomposition(expr)
                if decomp is not None:
                    assert decomp.name == "and", "decompose_in_tree: expected a conjunction but got {decomp}"
                    newlist.extend(decomp.args)
                    continue

            if decompose_custom_positive is not None and expr.name in decompose_custom_positive:
                exprs, toplevel_exprs = decompose_custom_positive[expr.name](expr)
            elif decompose_custom is not None and expr.name in decompose_custom:
                exprs, toplevel_exprs = decompose_custom[expr.name](expr)
            else:
                exprs, toplevel_exprs = expr.decompose_positive()
            # we merge the list toplevel rather than create an 'and'
            # we add them to todolist because both might contain globals
            if len(toplevel_exprs) > 0:
                todolist.extend(toplevel_exprs)
            if len(exprs) > 0:
                todolist.extend(exprs)
                if csemap is not None:
                    csemap.save_decomposition(expr, Operator("and", exprs))
        elif isinstance(expr, (bool, np.bool_)):
            # TODO: violates type!!! from `.decompose()` functions that are not cleaned yet
            changed = True
            newlist.append(BoolVal(expr))
        elif expr.has_subexpr():
            # special case for positive reified
            decomposed_positive = False
            if expr.name == "->" and isinstance(expr.args[1], GlobalConstraint) and expr.args[1].name not in supported_reified:
                changed = True
                exprs, toplevel_exprs = expr.args[1].decompose_positive()
                if len(toplevel_exprs) > 0:
                    todolist.extend(toplevel_exprs)
                expr = Operator("->", [expr.args[0], cpm_all(exprs)])   
                decomposed_positive = True

            # decompose its arguments
            arg_changed, arg_newargs, arg_toplevel = _decompose_in_tree_args(expr.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
            if arg_changed:
                changed = True
                if len(arg_toplevel) > 0:
                    todolist.extend(arg_toplevel)
                # if decompose_positive: we know 'expr' is a fresh expression
                if not decomposed_positive:
                    expr = copy.copy(expr)
                expr.update_args(arg_newargs)

            newlist.append(expr)
        else:
            newlist.append(expr)

    # recurse on any newly generated toplevel expressions
    if len(todolist) > 0:
        return newlist + decompose_in_tree(todolist, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom, decompose_custom_positive=decompose_custom_positive)
    elif changed:
        return newlist
    else:  # not changed
        return lst_of_expr


def decompose_objective(expr: Expression,
                        supported: Optional[AbstractSet[str]] = None,
                        supported_reified: Optional[AbstractSet[str]] = None,
                        csemap: Optional[CSEMap] = None,
                        decompose_custom: Optional[Dict[str, CustomDecomp]]=None) -> tuple[Expression, list[Expression]]:
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
                            decompose_custom:Optional[Dict[str, CustomDecomp]]=None) -> tuple[bool, list[Any]|tuple[Any], list[Expression]]:
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
            # a nested expression (its inside an args)
            if isinstance(arg, GlobalConstraint) and arg.name not in supported_reified:
                changed = True
                if csemap is not None:
                    decomp = csemap.get_decomposition(arg)
                    if decomp is not None:
                        newargs.append(decomp)
                        continue
                arg_orig = arg

                # new global constraint, decompose
                if decompose_custom is not None and arg.name in decompose_custom:
                    exprs, toplevel_exprs = decompose_custom[arg.name](arg)
                else:
                    exprs, toplevel_exprs = arg.decompose()
                if len(toplevel_exprs) > 0:
                    toplevel.extend(toplevel_exprs)
            
                # a decomposition may contain globals, recurse into the exprs
                # we have to do this here anyway, hence we do not recurse into the args upfront
                # (the csemap catches duplicate effort anyway)
                rec_changed, rec_newexprs, rec_toplevel = _decompose_in_tree_args(exprs, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    exprs = cast(list[Expression], rec_newexprs)
                    if len(rec_toplevel) > 0:
                        toplevel.extend(rec_toplevel)

                if len(exprs) == 1:
                    arg = exprs[0]
                else:
                    # replace arg by conjunction of decompose
                    arg = Operator("and", exprs)

                if csemap is not None:
                    csemap.save_decomposition(arg_orig, arg)
                newargs.append(arg)
                continue
            
            elif isinstance(arg, GlobalFunction) and arg.name not in supported:
                changed = True
                if csemap is not None:
                    decomp = csemap.get_decomposition(arg)
                    if decomp is not None:
                        newargs.append(decomp)
                        continue
                arg_orig2 = arg

                # a decomposition may consist of a new GlobFunc to decompose...
                while isinstance(arg, GlobalFunction) and arg.name not in supported:
                    if decompose_custom is not None and arg.name in decompose_custom:
                        newarg, toplevel_exprs = decompose_custom[arg.name](arg)
                    else:
                        newarg, toplevel_exprs = arg.decompose()
                    arg = newarg
                    if len(toplevel_exprs) > 0:
                        toplevel.extend(toplevel_exprs)
                            
                # TODO: violates type!!!
                # apparently in #630 we decided that decompose may return an int (e.g. for element)...
                # we should change that (Element constructor requires variable index; the []/__get__ override can still take anything)
                if isinstance(arg, int):
                    # can't store ints in csemap
                    newargs.append(arg)
                    continue
            
                # if the new decomposed arg has subexprs, decompose its arguments too
                if arg.has_subexpr():
                    rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                    if rec_changed:
                        changed = True
                        # XXX exception! here we know 'arg' is a new expression, so we don't need to copy
                        arg.update_args(rec_newargs)
                        if len(rec_toplevel) > 0:
                            toplevel.extend(rec_toplevel)

                if csemap is not None:
                    csemap.save_decomposition(arg_orig2, arg)
                newargs.append(arg)
                continue

            else:  # any other expression
                # if it has subexprs, decompose its arguments
                if arg.has_subexpr():
                    rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(arg.args, supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                    if rec_changed:
                        changed = True
                        arg = copy.copy(arg)
                        arg.update_args(rec_newargs)
                        if len(rec_toplevel) > 0:
                            toplevel.extend(rec_toplevel)
                    newargs.append(arg)
                    continue
        
        elif isinstance(arg, np.ndarray) and arg.dtype == object:
            if isinstance(arg, NDVarArray):
                if arg.has_subexpr():
                    rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(tuple(arg.flat), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                    if rec_changed:
                        changed = True
                        # we reconstruct it as a cpm_array here
                        newargs.append(cpm_array(rec_newargs).reshape(arg.shape))
                        if len(rec_toplevel) > 0:
                            toplevel.extend(rec_toplevel)
                        continue
            else:  # regular np.array
                rec_changed, rec_newargs, rec_toplevel = _decompose_in_tree_args(tuple(arg.flat), supported=supported, supported_reified=supported_reified, csemap=csemap, decompose_custom=decompose_custom)
                if rec_changed:
                    changed = True
                    # we reconstruct it as a np.array here
                    newargs.append(np.array(rec_newargs).reshape(arg.shape))
                    if len(rec_toplevel) > 0:
                        toplevel.extend(rec_toplevel)
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
                    toplevel.extend(rec_toplevel)
                continue
        
        # all the rest: not allowed to contain expressions so just append
        newargs.append(arg)

    return (changed, newargs, toplevel)
