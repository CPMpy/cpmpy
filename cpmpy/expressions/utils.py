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

        is_bool
        is_int
        is_num
        is_false_cst
        is_true_cst
        is_boolexpr
        is_pure_list
        is_any_list
        is_transition
        flatlist
        all_pairs
        argval
        argvals
        eval_comparison
        get_bounds     
"""

import cpmpy as cp
import numpy as np
import math
from collections.abc import Iterable  # for flatten
from itertools import combinations
from cpmpy.exceptions import IncompleteFunctionError


def is_bool(arg):
    """ is it a boolean (incl numpy variants)
    """
    return isinstance(arg, (bool, np.bool_, cp.BoolVal))


def is_int(arg):
    """ can it be interpreted as an integer? (incl bool and numpy variants)
    """
    return isinstance(arg, (bool, np.bool_, cp.BoolVal, int, np.integer))


def is_num(arg):
    """ is it an int or float? (incl numpy variants)
    """
    return isinstance(arg, (bool, np.bool_, cp.BoolVal, int, np.integer, float, np.floating))


def is_false_cst(arg):
    """ is the argument the constant False (can be of type bool, np.bool and BoolVal)
    """
    if arg is False or arg is np.False_:
        return True
    elif isinstance(arg, cp.BoolVal):
        return not arg.value()
    return False


def is_true_cst(arg):
    """ is the argument the constant True (can be of type bool, np.bool and BoolVal)
    """
    if arg is True or arg is np.True_:
        return True
    elif isinstance(arg, cp.BoolVal):
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
    if hasattr(a, "value"):
        try:
            val = a.value()
        except IncompleteFunctionError as e:
            if a.is_bool():
                return False
            else:
                raise e
    else:
        val = a

    if isinstance(val, np.generic):
        return val.item() # ensure it is a Python native value
    return val


def argvals(arr):
    if is_any_list(arr):
        return [argvals(arg) for arg in arr]
    return argval(arr)


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
    if isinstance(lhs, (np.integer, np.bool_)):
        lhs = int(lhs)
    if isinstance(rhs, (np.integer, np.bool_)):
        rhs = int(rhs)

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
    # from cpmpy.expressions.core import Expression
    # from cpmpy.expressions.variables import cpm_array

    if isinstance(expr, cp.expressions.core.Expression):
        return expr.get_bounds()
    elif is_any_list(expr):
        lbs, ubs = zip(*[get_bounds(e) for e in expr])
        return list(lbs), list(ubs) # return list as NDVarArray is covered above
    else:
        assert is_num(expr), f"All Expressions should have a get_bounds function, `{expr}`"
        if is_bool(expr):
            return int(expr), int(expr)
        return math.floor(expr), math.ceil(expr)

def implies(expr, other):
    """ like :func:`~cpmpy.expressions.core.Expression.implies`, but also safe to use for non-expressions """
    if isinstance(expr, cp.expressions.core.Expression):
        return expr.implies(other)
    elif is_true_cst(expr):
        return other
    elif is_false_cst(expr):
        return cp.BoolVal(True)
    else:
        return expr.implies(other)

# Specific stuff for ShortTabel global (should this be in globalconstraints.py instead?)
STAR = "*" # define constant here
def is_star(arg):
    """
        Check if arg is star as used in the ShortTable global constraint
    """
    return isinstance(arg, type(STAR)) and arg == STAR
