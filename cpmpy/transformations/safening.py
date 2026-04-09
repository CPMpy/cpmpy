"""
    Transforms partial functions into total functions.
"""

from copy import copy
import numpy as np

from ..expressions.variables import boolvar, intvar, NDVarArray, _BoolVarImpl, _NumVarImpl
from ..expressions.core import Expression, BoolVal, ListLike, ExprLike
from ..expressions.utils import get_bounds, is_any_list
from ..expressions.python_builtins import all as cpm_all
from typing import Optional, AbstractSet, Any, Sequence


def no_partial_functions(lst_of_expr: list[Expression],
                        _toplevel: Optional[Any]=None, 
                        _nbc: Optional[Any] = None, 
                        safen_toplevel: Optional[AbstractSet[str]] = None) -> list[Expression]:
    """
    A partial function is a function whose output is not defined for all possible inputs.

    In CPMpy, this is the case for the following 3 numeric functions:

    - (Integer) division ``x // y``: undefined when y=0
    - Modulo ``x mod y``: undefined when y=0
    - Element ``Arr[idx]``: undefined when idx is not in the range of Arr

    A toplevel constraint must always be true, so constraint solvers simply propagate the 'unsafe'
    value(s) away. However, CPMpy allows arbitrary nesting and reification of constraints, so an
    expression like ``b <-> (Arr[idx] == 5)`` is allowed, even when variable `idx` goes outside the bounds of `Arr`.
    Should `idx` be restricted to be in-bounds? and what value should 'b' take if it is out-of-bounds?

    This transformation will transform a partial function (e.g. Arr[idx]) into a total function
    following the **relational semantics** as discussed in:
        Frisch, Alan M., and Peter J. Stuckey. "The proper treatment of undefinedness in
        constraint languages." International Conference on Principles and Practice of Constraint
        Programming. Berlin, Heidelberg: Springer Berlin Heidelberg, 2009.

    Under the relational semantic, an 'undefined' output for a (numerical) expression should
    propagate to `False` in the nearest Boolean parent expression. In the above example: `idx` should
    be allowed to take a value outside the bounds of `Arr`, and `b` should be False in that case.

    To enable this, we want to rewrite an expression like ``b <-> (partial == 5)`` to something like
    ``b <-> (is_defined & (total == 5))``. The key idea is to create a copy of the potentially unsafe
    argument, that can only take 'safe' values. Using this new argument in the original expression results
    in a total function. We now have the original argument variable, which is decoupled from the expression,
    and a new 'safe' argument variable which is used in the expression. The `is_defined` flag serves two
    purposes: it represents whether the original argument has a safe value; and if it is true the new
    argument must equal the original argument so the two are coupled again. If `is_defined` is false, the new
    argument remains decoupled (can take any value, as will the function's output).

    .. warning::
        Under the relational semantics, ``b <-> ~(partial==5)`` and ``b <-> (partial!=5)`` mean
        different things! The second is ``b <-> (is_defined & (total!=5))`` the first is
        ``b <-> (~is_defined | (total!=5))``.


    A clever observation of the implementation below is that for the above 3 expressions, the 'safe'
    domain of a potentially unsafe argument (y or idx) is either one 'trimmed' continuous domain
    (for idx and in case y = [0..n] or [-n..0]), or two 'trimmed' continuous domains (for y=[-n..m]).
    Furthermore, the reformulation for these two cases can be done generically, without needing
    to know the specifics of the partial function being made total.
    
    Arguments:
        lst_of_expr (list[Expression]): list of CPMpy expressions
        _toplevel (None): DEPRECATED
        _nbc (None): DEPRECATED
        safen_toplevel (set[str]): list of expression types that need to be safened, even when toplevel. Used when
                                    a solver does not support unsafe values in it's API (e.g., OR-Tools for `div`), 
                                    or when the solver does not support the global function, and it needs to be decomposed.
    """

    assert _toplevel is None, "no_partial_functions:  argument '_toplevel' is deprecated, do not use/modify it"
    assert _nbc is None, "no_partial_functions:  argument '_nbc' is deprecated, do not use/modify it"

    if safen_toplevel is None:
        safen_toplevel = frozenset()

    changed, new_lst, todo_toplevel, nbc = _no_partial_functions(lst_of_expr, is_toplevel=True, safen_toplevel=safen_toplevel)
    if not changed:
        return lst_of_expr  # return original list

    # we are the highest 'nearest Boolean context', so add the nbc constraints to the toplevel
    if len(nbc) > 0:
        todo_toplevel = todo_toplevel + nbc

    # new toplevel constraints may need to be safened too
    while len(todo_toplevel):
        changed, decomp, next_toplevel, next_nbc = _no_partial_functions(todo_toplevel, is_toplevel=True, safen_toplevel=safen_toplevel)
        if not changed:
            new_lst.extend(todo_toplevel)
            break

        # changed, loop again
        new_lst.extend(decomp)
        todo_toplevel = next_toplevel + next_nbc  # toplevel constraints may have introduced nested lists or ands

    return new_lst
    

def _no_partial_functions(lst_of_expr: ListLike[Any], is_toplevel: bool, safen_toplevel: AbstractSet[str]) -> tuple[bool, list[Expression], list[Expression], list[Expression]]:
    """
    Safen a list of expressions by replacing partial functions with total functions.

    INTERNAL function, not guaranteed to remain backward compatible.

    Arguments:
        lst_of_expr (list[Expression]): list of CPMpy expressions
        is_toplevel (bool): whether ``lst_of_expr`` is the toplevel list of constraints.
        safen_toplevel (set[str]): list of expression types that need to be safened, even when toplevel. Used when
                                        a solver does not support unsafe values in its API (e.g., OR-Tools for `div`).

    Returns:
        tuple[bool, list[Expression], list[Expression], list[Expression]]
        changed (bool): True if a partial function was safened was done (or a recursive call changed something).
        new_lst (list[Expression]): the safened list of expressions (same length as ``lst_of_expr``).
        toplevel (list[Expression]): the list of auxiliary constraints to post at top level.
        nbc (list[Expression]): the list of expressions to put in nearest Boolean context.
    """

    toplevel: list[Expression] = []
    nbc_for_each_expr : list[list[Expression]] = [[] for _ in range(len(lst_of_expr))]
    
    changed = False
    new_lst: list[Any] = [] # TODO: because of is_any_list, can be many things...
    for i, cpm_expr in enumerate(lst_of_expr):

        if is_any_list(cpm_expr):
            assert not is_toplevel, "Lists in lists is only allowed for arguments (e.g. of global constraints)." \
                                    "Make sure to run func:`cpmpy.transformations.normalize.toplevel_list` first."

            if isinstance(cpm_expr, NDVarArray) and not cpm_expr.has_subexpr():
                pass  # no subexpressions, nothing to do
            elif isinstance(cpm_expr, np.ndarray) and cpm_expr.dtype != object:
                pass  # only constants, nothing to do
            else:
                rec_changed, rec_expr, rec_toplevel, rec_nbc = _no_partial_functions(cpm_expr, is_toplevel=False, safen_toplevel=safen_toplevel)
                if rec_changed:
                    cpm_expr = rec_expr
                    toplevel.extend(rec_toplevel)
                    nbc_for_each_expr[i].extend(rec_nbc)
                    changed = True
            new_lst.append(cpm_expr)
            continue

        if isinstance(cpm_expr, Expression):

            # safen its arguments first
            if cpm_expr.has_subexpr():
                rec_changed, newargs, rec_toplevel, rec_nbc= _no_partial_functions(cpm_expr.args, is_toplevel=False, safen_toplevel=safen_toplevel)
                if rec_changed:
                    cpm_expr = copy(cpm_expr)
                    cpm_expr.update_args(newargs)
                    toplevel.extend(rec_toplevel)
                    nbc_for_each_expr[i].extend(rec_nbc)
                    changed = True

            if cpm_expr.is_bool() and not is_toplevel and len(nbc_for_each_expr[i]) > 0: # filled nbc in recursive call
                # add guards to this Boolean expression
                cpm_expr = cpm_all(nbc_for_each_expr[i]) & cpm_expr
                nbc_for_each_expr[i] = [] # handled, no need to bring to previous rec level

            # else, global function, which may need to be safened
            elif cpm_expr.name == "element":

                if is_toplevel and cpm_expr.name not in safen_toplevel: # no need to safen
                    new_lst.append(cpm_expr)
                    continue

                arr, idx = cpm_expr.args
                lb, ub = get_bounds(idx)
                guard: Optional[_BoolVarImpl | BoolVal] = None # for mypy

                if lb < 0 or ub >= len(arr): # index can be out of bounds
                    guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(0, len(arr)-1), idx_to_safen=1)

                    nbc_for_each_expr[i].append(guard)  # guard must be added to nearest Boolean context
                    toplevel += extra_cons  # any additional constraint that must be true
                    cpm_expr = output_expr  # replace partial function by this (total) new output expression
                    changed = True

            elif cpm_expr.name == "multidim_element":

                if is_toplevel and cpm_expr.name not in safen_toplevel: # no need to safen
                    new_lst.append(cpm_expr)
                    continue

                arr = cpm_expr.args[0]
                for dim_idx, (idx, dim) in enumerate(zip(cpm_expr.args[1:], arr.shape)):
                    lb, ub = get_bounds(idx)
                    if lb < 0 or ub >= dim: # index can be out of bounds
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(0, dim-1), idx_to_safen=1+dim_idx)

                        nbc_for_each_expr[i].append(guard)  # guard must be added to nearest Boolean context
                        toplevel += extra_cons  # any additional constraint that must be true
                        cpm_expr = output_expr  # replace partial function by this (total) new output expression
                        changed = True

            elif cpm_expr.name == "div" or cpm_expr.name == "mod":
                if is_toplevel and cpm_expr.name not in safen_toplevel: # no need to safen
                    new_lst.append(cpm_expr)
                    continue

                idx_to_safen = 1
                lb, ub = get_bounds(cpm_expr.args[idx_to_safen])

                if lb <= 0 <= ub:
                    if lb == ub == 0:
                        # (unlikely) edge case, nearest Boolean context should propagate to False.
                        # introduce dummy numerical integer expression
                        guard, output_expr, extra_cons = BoolVal(False), intvar(0, 1), []
                    elif lb == 0:
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(1, ub), idx_to_safen=idx_to_safen)
                    elif ub == 0:
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(lb, -1), idx_to_safen=idx_to_safen)
                    else:  # proper hole
                        guard, output_expr, extra_cons = _safen_hole(cpm_expr, exclude=0, idx_to_safen=idx_to_safen)

                    nbc_for_each_expr[i].append(guard)  # guard must be added to nearest Boolean context
                    toplevel += extra_cons  # any additional constraint that must be true
                    cpm_expr = output_expr  # replace partial function by this (total) new output expression
                    changed = True

            # add the (potentially new) cpm_expr to the list
            new_lst.append(cpm_expr)
            continue
        
        # constants, variables, other expressions are left as is
        new_lst.append(cpm_expr)

    assert len(new_lst) == len(lst_of_expr), f"Nested safening should not change the number of expressions\n{lst_of_expr}\n{new_lst}"
    nbc = [expr for lst in nbc_for_each_expr for expr in lst] # merge remaining nbc expressions
    return changed, new_lst, toplevel, nbc

def _safen_range(partial_expr:Expression, safe_range:tuple[int,int], idx_to_safen:int) -> tuple[_BoolVarImpl, Expression, list[Expression]]:
    """
    Replace partial function `cpm_expr` that has potentially unsafe argument at `idx_to_safen`,
    by a total function using a safe argument with domain `safe_range`. Also returns
    a Boolean flag indicating whether the original argument's value is in the safe range
    (for use in the nearest Boolean context), and a list of toplevel constraints that help
    define the new total function.

    An example is `Element` where the index should be within the array's range.

    Arguments:
        partial_expr (Expression): The partial function expression to make total
        safe_range (tuple[int,int]): The range of safe argument values
        idx_to_safen (int): The index of the potentially unsafe argument in the expression

    Returns:
        is_defined (bool): The guard indicating whether the original argument's value is in the safe range
        total_expr (Expression): The new total function expression
        toplevel (list[Expression]): The list of auxiliary constraints to post at top level.
    """
    safe_lb, safe_ub = safe_range

    orig_arg = partial_expr.args[idx_to_safen]
    new_arg = intvar(safe_lb, safe_ub)  # values for which the partial function is defined

    total_expr = copy(partial_expr)  # the new total function, with the new arg
    total_expr.update_args([new_arg if i == idx_to_safen else a for i,a in enumerate(partial_expr.args)])

    is_defined = boolvar()
    toplevel = [is_defined == ((safe_lb <= orig_arg) & (orig_arg <= safe_ub)),
                is_defined.implies(new_arg == orig_arg),
                # Extra: avoid additional solutions when new var is unconstrained
                # (should always work in reified context because total_expr is newly defined?)
                (~is_defined).implies(new_arg == safe_lb)]

    return is_defined, total_expr, toplevel


def _safen_hole(cpm_expr: Expression, exclude: int, idx_to_safen: int) -> tuple[_BoolVarImpl, _NumVarImpl, list[Expression]]:
    """
    Safen expression where a single value of an argument can cause undefinedness.
    Examples include `div` where 0 has to be removed from the denominator

    Constructs an expression for each interval of safe values, and introduces a new `output_var` variable.

    Arguments:
        cpm_expr (Expression): The numerical expression to safen
        exclude (int): The domain value to exclude
        idx_to_safen (int): The index of unsafe argument in the expression

    Returns:
        is_defined (bool): The guard indicating whether the original argument's value is safe
        output_var (Expression): The new total function expression
        toplevel (list[Expression]): The list of auxiliary constraints to post at top level.
    """
    orig_arg = cpm_expr.args[idx_to_safen]
    orig_lb, orig_ub = get_bounds(orig_arg)

    # expr when arg in [orig_lb..exclude-1]
    new_arg_lower = intvar(orig_lb, exclude-1)
    total_expr_lower = copy(cpm_expr)
    total_expr_lower.update_args([new_arg_lower if i == idx_to_safen else a for i,a in enumerate(cpm_expr.args)])
    # expr when arg in [exclude+1..orig_ub]
    new_arg_upper = intvar(exclude+1, orig_ub)
    total_expr_upper = copy(cpm_expr)
    total_expr_upper.update_args([new_arg_upper if i == idx_to_safen else a for i, a in enumerate(cpm_expr.args)])

    is_defined = boolvar()
    is_defined_lower = boolvar()
    is_defined_upper = boolvar()
    output_var = intvar(*get_bounds(cpm_expr))

    toplevel = [
        is_defined_lower == (orig_arg < exclude),
        is_defined_upper == (orig_arg > exclude),
        is_defined == (is_defined_lower | is_defined_upper),

        is_defined_lower.implies(orig_arg == new_arg_lower),
        is_defined_lower.implies(output_var == total_expr_lower),
        is_defined_upper.implies(orig_arg == new_arg_upper),
        is_defined_upper.implies(output_var == total_expr_upper),

        # Extra: avoid additional solutions when new vars are unconstrained
        # (should always work in reified context because total_expr's are newly defined?)
        (~is_defined_lower).implies(new_arg_lower == exclude-1),
        (~is_defined_upper).implies(new_arg_upper == exclude+1),
        (~is_defined).implies(output_var == output_var.lb)
    ]

    return is_defined, output_var, toplevel


def safen_objective(expr: Expression) -> tuple[ExprLike, list[Expression]]:
    """
    Safen any partial functions in the objective function expression.

    Arguments:
        expr (Expression): objective expression (e.g. ``x // y``, ``arr[x] + arr[y]``).

    Returns:
        tuple[Expression, list[Expression]]
        safe_expr (Expression): the safened objective expression
        toplevel (list[Expression]): the list of auxiliary constraints to post at top level.
    """
    if is_any_list(expr):
        raise ValueError(f"Expected numerical expression as objective but got a list {expr}")

    changed, safe_expr, toplevel, nbc = _no_partial_functions((expr,), is_toplevel=False, safen_toplevel=frozenset())
    if changed:
        assert len(safe_expr) == 1, f"Safening should not alter the number of expressions"
        return safe_expr[0], toplevel + nbc
    else:
        return expr, []

