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
from collections.abc import Iterable # for _flatten
from itertools import chain, combinations
from cpmpy.exceptions import IncompleteFunctionError


def is_bool(arg):
    """ is it a boolean (incl numpy variants)
    """
    return isinstance(arg, (bool, np.bool_))


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
    try:
        return a.value() if hasattr(a, "value") else a
    except IncompleteFunctionError as e:
        if a.is_bool(): return False
        raise e


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

    from cpmpy.expressions.core import Expression
    if isinstance(expr, Expression):
        return expr.get_bounds()
    else:
        assert is_num(expr), f"All Expressions should have a get_bounds function, `{expr}`"
        if is_bool(expr):
            return int(expr), int(expr)
        return math.floor(expr), math.ceil(expr)
