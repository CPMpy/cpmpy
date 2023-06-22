#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## python_builtins.py
##
"""
    Overwrites a number of python built-ins, so that they work over variables as expected.

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        all
        any
        max
        min
        sum
"""
import numpy as np
import builtins  # to use the original Python-builtins

from .utils import is_false_cst, is_true_cst
from .variables import NDVarArray
from .core import Expression, Operator
from .globalfunctions import Minimum, Maximum

# Overwriting all/any python built-ins
# all: listwise 'and'
def all(iterable):
    """
        all() overwrites python built-in,
        if iterable contains an `Expression`, then returns an Operator("and", iterable)
        otherwise returns whether all of the arguments is true
    """
    if isinstance(iterable, NDVarArray): iterable=iterable.flat # 1D iterator
    collect = [] # logical expressions
    for elem in iterable:
        if is_false_cst(elem):
            return False  # no need to create constraint
        elif is_true_cst(elem):
            pass
        elif isinstance(elem, Expression) and elem.is_bool():
            collect.append( elem )
        else:
            raise Exception("Non-Boolean argument '{}' to 'all'".format(elem))
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("and", collect)
    return True

# any: listwise 'or'
def any(iterable):
    """
        any() overwrites python built-in,
        if iterable contains an `Expression`, then returns an Operator("or", iterable)
        otherwise returns whether any of the arguments is true
    """
    if isinstance(iterable, NDVarArray): iterable=iterable.flat # 1D iterator
    collect = [] # logical expressions
    for elem in iterable:
        if is_true_cst(elem):
            return True # no need to create constraint
        elif is_false_cst(elem):
            pass
        elif isinstance(elem, Expression) and elem.is_bool():
            collect.append( elem )
        else:
            raise Exception("Non-Boolean argument '{}' to 'all'".format(elem))
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("or", collect)
    return False

def max(iterable):
    """
        max() overwrites python built-in,
        checks if all constants and computes np.max() in that case
    """
    if not builtins.any(isinstance(elem, Expression) for elem in iterable):
        return np.max(iterable)
    return Maximum(iterable)

def min(iterable):
    """
        min() overwrites python built-in,
        checks if all constants and computes np.min() in that case
    """
    if not builtins.any(isinstance(elem, Expression) for elem in iterable):
        return np.min(iterable)
    return Minimum(iterable)

def sum(iterable):
    """
        sum() overwrites python built-in,
        checks if all constants and computes np.sum() in that case
        otherwise, makes a sum Operator directly on `iterable`
    """
    iterable = list(iterable) # Fix generator polling
    if not builtins.any(isinstance(elem, Expression) for elem in iterable):
        return np.sum(iterable)
    return Operator("sum", iterable)
