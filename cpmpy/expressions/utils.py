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
from collections.abc import Iterable # for _flatten
from itertools import chain, combinations

def is_int(arg):
    """ can it be interpreted as an integer? (incl bool and numpy variants)
    """
    return isinstance(arg, (bool, np.bool_, int, np.integer))
def is_num(arg):
    """ is it an int or float? (incl numpy variants)
    """
    return isinstance(arg, (bool, np.bool_, int, np.integer, float, np.floating))
def is_bool(arg):
    """ is it a boolean (incl numpy variants)
    """
    return isinstance(arg, (bool, np.bool_))
def is_boolexpr(expr):
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
    # from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    # returns an iterator, not a list
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
    return a.value() if hasattr(a, "value") else a

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
    # can return floats, use floor and ceil when creating an intvar!
    from cpmpy.expressions.core import Expression
    if isinstance(expr,Expression):
        return expr.get_bounds()
    else:
        assert is_num(expr), f"All Expressions should have a get_bounds function, `{expr}`"
        if is_bool(expr):
            return 0, 1
        return expr, expr
