import copy
import warnings  # for deprecation warning
import numpy as np

from .normalize import toplevel_list
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, _NumVarImpl
from ..expressions.utils import is_any_list

def push_down_negation(lst_of_expr, toplevel=True):
    """
        Transformation that checks all elements from the list,
        and pushes down any negation it finds with the `recurse_negation()` function.

        Assumes the input is a list (typically from `toplevel_list()`) en ensures the output is
        a toplevel_list if the input was.
    """
    if isinstance(lst_of_expr, np.ndarray) and not (lst_of_expr.dtype == object):
        # shortcut for data array, return as is
        return lst_of_expr

    newlist = []
    for expr in lst_of_expr:
        if is_any_list(expr):
            # can be a nested list with expressions?
            newlist.append(push_down_negation(expr, toplevel=toplevel))

        elif not isinstance(expr, Expression) or isinstance(expr, (_NumVarImpl,BoolVal)):
            # nothing to do
            newlist.append(expr)

        elif expr.name == "not":
            # the negative case, negate
            arg_neg = recurse_negation(expr.args[0])
            if toplevel:
                # make sure there is no toplevel 'and' (could do explicit check for and?)
                newlist.extend(toplevel_list(arg_neg))
            else:
                newlist.append(arg_neg)

        else:
            # an Expression, we remain in the positive case
            newexpr = copy.copy(expr)
            # TODO, check that an arg changed? otherwise no copy needed here...
            newexpr.args = push_down_negation(expr.args, toplevel=False)  # check if 'not' is present in arguments
            newlist.append(newexpr)

    return newlist

def recurse_negation(expr):
    """
        Negate 'expr' by pushing the negation down into it and its args

        Comparison: swap comparison sign
        Operator.is_bool(): apply DeMorgan
        Global: leave "NOT" operator before global constraint. Use `decompose_globals` for this (AFTER ISSUE #293)
    """

    if isinstance(expr, (_BoolVarImpl,BoolVal)):
        return ~expr

    elif isinstance(expr, Comparison):
        newexpr = copy.copy(expr)
        if   expr.name == '==': newexpr.name = '!='
        elif expr.name == '!=': newexpr.name = '=='
        elif expr.name == '<=': newexpr.name = '>'
        elif expr.name == '<':  newexpr.name = '>='
        elif expr.name == '>=': newexpr.name = '<'
        elif expr.name == '>':  newexpr.name = '<='
        else: raise ValueError(f"Unknown comparison to negate {expr}")
        # args are positive now, still check if no 'not' in its arguments
        newexpr.args = push_down_negation(expr.args, toplevel=False)
        return newexpr

    elif isinstance(expr, Operator):
        assert(expr.is_bool()), f"Can only negate boolean expressions but got {expr}"

        if expr.name == "not":
            # negation while in negative context = switch back to positive case
            neg_args = push_down_negation(expr.args, toplevel=False)
            return neg_args[0]  # not has only 1 argument

        elif expr.name == "->":
            # ~(x -> y) :: x & ~y
            # arg0 remains positive, but check its arguments
            # (must wrap awkwardly in a list, but can make no assumption about expr.args[0] has .args)
            newarg0_lst = push_down_negation([expr.args[0]], toplevel=False)
            return newarg0_lst[0] & recurse_negation(expr.args[1])

        else:
            newexpr = copy.copy(expr)
            if   expr.name == "and": newexpr.name = "or"
            elif expr.name == "or": newexpr.name = "and"
            else: raise ValueError(f"Unknown operator to negate {expr}")
            # continue negating the args
            newexpr.args = [recurse_negation(a) for a in expr.args]
            return newexpr

    # global constraints
    elif hasattr(expr, "decompose"):
        newexpr = copy.copy(expr)
        # args are positive as we will negate the global, still check if no 'not' in its arguments
        newexpr.args = push_down_negation(expr.args, toplevel=False)
        return ~newexpr

    # numvars or direct constraint
    else:
        raise ValueError(f"Unsupported expression to negate: {expr}")


def negated_normal(expr):
    warnings.warn("Deprecated, use `recurse_negation()` instead which will negate and push down all negations in the expression (or use `push_down_negation` on the full expression tree); will be removed in stable version", DeprecationWarning)
    return recurse_negation(expr)
