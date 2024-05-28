#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## utils.py
##
"""
Internal utilities for expression handling.

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        is_int
        is_num
        is_pure_list
        is_any_list
        flatlist
        all_pairs
        argval
        eval_comparison
"""

import numpy as np
import math
from collections.abc import Iterable  # for flatten
from itertools import combinations
from cpmpy.exceptions import IncompleteFunctionError


def is_bool(arg):
    """ is it a boolean (incl numpy variants)
    """
    from cpmpy import BoolVal
    return isinstance(arg, (bool, np.bool_, BoolVal))


def is_int(arg):
    """ can it be interpreted as an integer? (incl bool and numpy variants)
    """
    return is_bool(arg) or isinstance(arg, (int, np.integer))


def is_num(arg):
    """ is it an int or float? (incl numpy variants)
    """
    return is_int(arg) or isinstance(arg, (float, np.floating))


def is_false_cst(arg):
    """ is the argument the constant False (can be of type bool, np.bool and BoolVal)
    """
    from cpmpy import BoolVal
    if arg is False or arg is np.False_:
        return True
    elif isinstance(arg, BoolVal):
        return not arg.value()
    return False


def is_true_cst(arg):
    """ is the argument the constant True (can be of type bool, np.bool and BoolVal)
    """
    from cpmpy import BoolVal
    if arg is True or arg is np.True_:
        return True
    elif isinstance(arg, BoolVal):
        return arg.value()
    return False


def is_boolexpr(expr):
    """ is the argument a boolean expression or a boolean value
    """
    #boolexpr
    if hasattr(expr, 'is_bool'):
        return expr.is_bool()
    #boolean constant
    return is_bool(expr)


def is_pure_list(arg):
    """ is it a list or tuple?
    """
    return isinstance(arg, (list, tuple))


def is_any_list(arg):
    """ is it a list or tuple or numpy array?
    """
    return isinstance(arg, (list, tuple, np.ndarray))


def is_transition(arg):
    """ test if the argument is a transition, i.e. a 3-elements-tuple specifying a starting state,
    a transition value and an ending node"""
    return len(arg) == 3 and \
        isinstance(arg[0], (int, str)) and is_int(arg[1]) and isinstance(arg[2], (int, str))

def flatlist(args):
    """ recursively flatten arguments into one single list
    """
    return list(_flatten(args))


def _flatten(args):
    """ flattens the irregular nested list into an iterator

        from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    """
    for el in args:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el


def all_pairs(args):
    """ returns all pairwise combinations of elements in args
    """
    return list(combinations(args, 2))


def argval(a):
    """ returns .value() of Expression, otherwise the variable itself

        We check with hasattr instead of isinstance to avoid circular dependency
    """
    if hasattr(a, "value"):
        try:
            return a.value()
        except IncompleteFunctionError as e:
            if a.is_bool():
                return False
            else:
                raise e
    return a


def argvals(arr):
    if is_any_list(arr):
        return [argvals(arg) for arg in arr]
    return argval(arr)


def is_leaf(a):
    if hasattr(a, 'is_leaf'):
        return a.is_leaf()
    if is_any_list(a):
        return all([is_leaf(x) for x in a])
    else:
        return True


def has_nested(expr):
    if is_leaf(expr):
        return False
    if hasattr(expr, 'args'):
        return not all([is_leaf(x) for x in expr.args])
    return True


def eval_comparison(str_op, lhs, rhs):
    """
        Internal function: evaluates the textual `str_op` comparison operator
        lhs <str_op> rhs

        Valid str_op's:
        * '=='
        * '!='
        * '>'
        * '>='
        * '<'
        * '<='

        Especially useful in decomposition and transformation functions that already involve a comparison.
    """
    if str_op == '==':
        return lhs == rhs
    elif str_op == '!=':
        return lhs != rhs
    elif str_op == '>':
        return lhs > rhs
    elif str_op == '>=':
        return lhs >= rhs
    elif str_op == '<':
        return lhs < rhs
    elif str_op == '<=':
        return lhs <= rhs
    else:
        raise Exception("Not a known comparison:", str_op)


def get_bounds(expr):
    """ return the bounds of the expression
    returns appropriately rounded integers
    """

    # import here to avoid circular import
    from cpmpy.expressions.core import Expression
    from cpmpy.expressions.variables import cpm_array

    if isinstance(expr, Expression):
        return expr.get_bounds()
    elif is_any_list(expr):
        lbs, ubs = zip(*[get_bounds(e) for e in expr])
        return list(lbs), list(ubs) # return list as NDVarArray is covered above
    else:
        assert is_num(expr), f"All Expressions should have a get_bounds function, `{expr}`"
        if is_bool(expr):
            return int(expr), int(expr)
        return math.floor(expr), math.ceil(expr)
