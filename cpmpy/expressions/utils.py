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
from __future__ import annotations
import functools
import numpy as np
import math
from collections.abc import Iterable  # for flatten
from itertools import combinations
from cpmpy.exceptions import IncompleteFunctionError

from functools import singledispatch


# @singledispatch
def is_bool(arg):
    """ is it a boolean (incl numpy variants)
    """
    return type(arg) in (np.bool_, bool, "BoolVal") 
    
    return isinstance(arg, (np.bool_, bool)) or  type(arg) == "BoolVal"

# @is_bool.register("BoolVal")
# @is_bool.register(np.bool_)
# @is_bool.register(bool)
# def _(arg):
#     return True

# @singledispatch
def is_int(arg):
    """ can it be interpreted as an integer? (incl bool and numpy variants)
    """
    return type(arg) in (np.bool_, bool, "BoolVal", int, np.integer) 
    return is_bool(arg)  or isinstance(arg, (int, np.integer))

# @is_int.register(bool)
# @is_int.register(np.bool_)
# @is_int.register(int)
# @is_int.register(np.integer)
# def _(arg):
#     return True


# @singledispatch
def is_num(arg):
    """ is it an int or float? (incl numpy variants)
    """
    return type(arg) in (np.bool_, bool, "BoolVal", int, np.integer, float, np.floating) 
    return is_int(arg) or isinstance(arg, (float, np.floating))

# @is_num.register(bool)
# @is_num.register(np.bool_)
# @is_num.register(int)
# @is_num.register(np.integer)
# @is_num.register(float)
# @is_num.register(np.floating)
# def _(arg):
#     return True


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

@singledispatch
def is_pure_list(arg):
    """ is it a list or tuple?
    """
    # return type(arg) in (list, tuple)
    return False

@is_pure_list.register(list)
def _(args:list):
    return True

@is_pure_list.register(tuple)
def _(args:tuple):
    return True

# @singledispatch
def is_any_list(arg):
    """ is it a list or tuple or numpy array?
    """
    # return False
    return type(arg) in (list, tuple, np.ndarray)

# @is_any_list.register(list)
# def _(args:list):
#     return True

# @is_any_list.register(tuple)
# def _(args:tuple):
#     return True

# @is_any_list.register(np.ndarray)
# def _(args:np.ndarray):
#     return True



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

def is_leaf_vectorized(a) -> bool: # 0.61 -> 0.57
    # return all(map(is_leaf_helper, a.flat))
    vectorized = np.vectorize(is_leaf_helper, otypes=[bool])
    b = vectorized(a)
    # return np.all(vectorized(a))
    c = b[np.argmin(b)] # short-circuits
    if isinstance(c, bool):
        return c
    else:
        return b.flat[0]


def is_leaf(a) -> bool:        
    if hasattr(a, 'is_leaf'):
        return a.is_leaf()
    if is_any_list(a):
        return all(map(is_leaf, a))
    else:
        return True
    
def is_leaf_helper(a) -> bool:
    if hasattr(a, 'is_leaf'):
        return a.is_leaf()
    else:
        return True



def has_nested(expr) -> bool:
    if hasattr(expr, 'args'):

        if expr.name == "table":
            
            a, b = expr.args
            # print("a", a)
            # print("b", b)
            a = np.array(a)
            if not is_leaf_vectorized(a):
                # print("a not leaf")
                return True
            c = np.array(b)
            if not is_leaf_vectorized(c):
                # print("b not leaf")
                return True
            # print("leaf")
            return False
            
        return not all(map(is_leaf, expr.args))
    if is_leaf(expr):
        return False
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
