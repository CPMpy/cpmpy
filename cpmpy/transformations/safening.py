"""
    Transforms partial functions into total functions.
"""

from copy import copy
import numpy as np

from ..expressions.variables import boolvar, intvar, NDVarArray
from ..expressions.core import Expression, BoolVal, ListLike, ExprLike
from ..expressions.utils import get_bounds, is_any_list
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.python_builtins import all as cpm_all
from typing import Optional, cast, AbstractSet, Any

def no_partial_functions(lst_of_expr:list[ExprLike], 
                         _toplevel: Optional[list[ExprLike]]=None, 
                         _nbc: Optional[list[ExprLike]] = None, 
                         safen_toplevel: Optional[AbstractSet[str]]=None) -> list[ExprLike]:
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


        :param list_of_expr: list of CPMpy expressions
        :param _toplevel: list of new expressions to put toplevel (used internally)
        :param _nbc: list of new expressions to put in nearest Boolean context (used internally)
        :param safen toplevel: list of expression types that need to be safened, even when toplevel. Used when
                                 a solver does not support unsafe values in it's API (e.g., OR-Tools for `div`).
    """

    assert _toplevel is None, "no_partial_functions:  argument '_toplevel' is deprecated, do not use/modify it"
    assert _nbc is None, "no_partial_functions:  argument '_nbc' is deprecated, do not use/modify it"

    changed, new_lst, todo_toplevel = _no_partial_functions(lst_of_expr, safen_toplevel=safen_toplevel)
    if not changed:
        return list(lst_of_expr) # return original list

    # new toplevel constraints may need to be safened too
    while len(todo_toplevel):
        changed, decomp, next_toplevel = _no_partial_functions(todo_toplevel, safen_toplevel=safen_toplevel)
        if not changed:
            new_lst.extend(todo_toplevel)
            break

        # changed, loop again
        new_lst.extend(decomp)
        todo_toplevel = next_toplevel # toplevel constraints may have introduced nested lists or ands

    return new_lst
    

def _no_partial_functions(lst_of_expr:ListLike[ExprLike], 
                          toplevel: Optional[list[ExprLike]]=None, 
                          nbc: Optional[list[ExprLike]] = None, 
                          safen_toplevel: Optional[AbstractSet[str]] = None) -> tuple[bool,list[ExprLike], list[ExprLike]]:
    """
        Safen a list of expressions by replacing partial functions with total functions.

        INTERNAL function, not guaranteed to remain backward compatible.

        :param lst_of_expr: list of CPMpy expressions
        :param _toplevel: list of new expressions to put toplevel (used internally)
        :param _nbc: list of new expressions to put in nearest Boolean context (used internally)
        :param safen toplevel: list of expression types that need to be safened, even when toplevel. Used when
                                 a solver does not support unsafe values in it's API (e.g., OR-Tools for `div`).

        :returns: tuple of (list of safened expressions, list of toplevel constraints)
    """

    if toplevel is None:
        is_toplevel = True
        toplevel = []
        assert nbc is None, "nbc must be None when is_toplevel is True"
        nbc = toplevel # we start at toplevel, with the nearest Boolean context being toplevel
    else:
        is_toplevel = False

    assert nbc is not None # for mypy
        
    if safen_toplevel is None:
        safen_toplevel = frozenset()
    
    changed = False
    new_lst: list[Any] = [] # TODO: because of is_any_list, can be many things...
    for cpm_expr in lst_of_expr:

        if is_any_list(cpm_expr):
            assert not is_toplevel, "Lists in lists is only allowed for arguments (e.g. of global constrainst)." \
                                    "Make sure to run func:`cpmpy.transformations.normalize.toplevel_list` first."

            cpm_expr = cast(ListLike[Expression], cpm_expr)  # TODO: avoid is_any_list()
            if isinstance(cpm_expr, NDVarArray) and not cpm_expr.has_subexpr():
                pass  # no subexpressions, nothing to do
            elif isinstance(cpm_expr, np.ndarray) and cpm_expr.dtype != object:
                pass  # only constants, nothing to do
            else:
                rec_changed, rec_expr, rec_toplevel = _no_partial_functions(list(cpm_expr), toplevel=toplevel, nbc=nbc, safen_toplevel=safen_toplevel)
                if rec_changed:
                    cpm_expr = rec_expr
                    toplevel.extend(rec_toplevel)
                    changed = True
            new_lst.append(cpm_expr)
            continue

        # if an expression, safen its arguments first
        if isinstance(cpm_expr, Expression) and cpm_expr.has_subexpr():
            if cpm_expr.is_bool() and not is_toplevel: 
                # current Boolean expression will serve as the nearest Boolean context
                nbc = []

            rec_changed, newargs, rec_toplevel = _no_partial_functions(cpm_expr.args, toplevel=toplevel, nbc=nbc, safen_toplevel=safen_toplevel)
            if rec_changed:
                cpm_expr = copy(cpm_expr)
                cpm_expr.update_args(newargs)
                toplevel.extend(rec_toplevel)
                changed = True

        if hasattr(cpm_expr, "args"): # expression with some arguments (not a variable or a constant)

            assert isinstance(cpm_expr, Expression)

            if cpm_expr.is_bool() and len(nbc) != 0:
                # add guards to this Boolean expression
                cpm_expr = cpm_all(nbc) & cpm_expr
            
            # else, global function, which may need to be safened
            elif isinstance(cpm_expr, GlobalFunction) and cpm_expr.name == "element":
                if is_toplevel and cpm_expr.name not in safen_toplevel: # no need to safen
                    new_lst.append(cpm_expr)
                    continue

                arr, idx = cpm_expr.args
                lb, ub = get_bounds(idx)

                if lb < 0 or ub >= len(arr): # index can be out of bounds
                    guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(0, len(arr)-1), idx_to_safen=1)

                    nbc.append(guard)  # guard must be added to nearest Boolean context
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
                        # (unlikely) edge case, neirest Boolean context should propagate to False.
                        # introduce dummy numerical integer expression
                        guard, output_expr, extra_cons = BoolVal(False), intvar(0, 1), []
                    elif lb == 0:
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(1, ub), idx_to_safen=idx_to_safen)
                    elif ub == 0:
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(lb, -1), idx_to_safen=idx_to_safen)
                    else:  # proper hole
                        guard, output_expr, extra_cons = _safen_hole(cpm_expr, exclude=0, idx_to_safen=idx_to_safen)

                    nbc.append(guard)  # guard must be added to nearest Boolean context
                    toplevel += extra_cons  # any additional constraint that must be true
                    cpm_expr = output_expr  # replace partial function by this (total) new output expression
                    changed = True

            # add the (potentially new) cpm_expr to the list
            new_lst.append(cpm_expr)
            continue
        
        # constants, variables, other expressions are left as is
        new_lst.append(cpm_expr)

    assert is_toplevel or len(new_lst) == len(lst_of_expr), f"Nested safening should not change the number of expressions\n{lst_of_expr}\n{new_lst}"
    return changed, new_lst, toplevel

def _safen_range(partial_expr, safe_range, idx_to_safen):
    """
        Replace partial function `cpm_expr` that has potentially unsafe argument at `idx_to_safen`,
        by a total function using a safe argument with domain `safe_range`. Also returns
        a Boolean flag indicating whether the original argument's value is in the safe range
        (for use in the nearest Boolean context), and a list of toplevel constraints that help
        define the new total function.

        An example is `Element` where the index should be within the array's range.

        :param partial_expr: The partial function expression to make total
        :param safe_range: The range of safe argument values
        :param idx_to_safen: The index of the potentially unsafe argument in the expression

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


def _safen_hole(cpm_expr, exclude, idx_to_safen):
    """
        Safen expression where a single value of an argument can cause undefinedness.
        Examples include `div` where 0 has to be removed from the denominator

        Constructs an expression for each interval of safe values, and
        introduces a new `output_var` variable

        :param cpm_expr: The numerical expression to safen
        :param exclude: The domain value to exclude
        :param idx_to_safen: The index of unsafe argument in the expression
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


def safen_objective(expr):
    if is_any_list(expr):
        raise ValueError(f"Expected numerical expression as objective but got a list {expr}")

    toplevel, nbc = [],[]
    safe_expr = no_partial_functions([expr], _toplevel=toplevel, _nbc=nbc)
    assert len(safe_expr) == 1
    return safe_expr[0], toplevel + nbc

