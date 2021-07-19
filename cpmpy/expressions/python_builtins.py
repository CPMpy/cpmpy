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
"""
import numpy as np
from .core import Expression, Operator
from .globalconstraints import Minimum, Maximum

# Overwriting all/any python built-ins
# all: listwise 'and'
def all(iterable):
    """
        all() overwrites python built-in,
        if iterable contains an `Expression`, then returns an Operator("and", iterable)
        otherwise returns whether all of the arguments is true
    """
    collect = [] # logical expressions
    for elem in iterable:
        if elem is False:
            return False # no need to create constraint
        elif elem is True:
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
    collect = [] # logical expressions
    for elem in iterable:
        if elem is True:
            return True # no need to create constraint
        elif elem is False:
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
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.max(iterable)
    return Maximum(iterable)

def min(iterable):
    """
        min() overwrites python built-in,
        checks if all constants and computes np.min() in that case
    """
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.min(iterable)
    return Minimum(iterable)


