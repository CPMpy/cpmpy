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
from fishhook import *
from .variables import NDVarArray, _IntVarImpl, _BoolVarImpl
from .core import Expression, Operator
from .globalconstraints import Minimum, Maximum
from ..exceptions import CPMpyException

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
    if isinstance(iterable, NDVarArray): iterable=iterable.flat # 1D iterator
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

def sum(iterable):
    """
        sum() overwrites python built-in,
        checks if all constants and computes np.sum() in that case
        otherwise, makes a sum Operator directly on `iterable`
    """
    iterable = list(iterable) # Fix generator polling
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.sum(iterable)
    return Operator("sum", iterable)

@hook(int)
def implies(self, other):
    print("am i here?")
    if self is True:
        return other
    if self is False:
        return True
    # or alternatively
    return Operator("->", [self, other])

@hook(bool, int)
def __or__(self, other):
    # or alternatively
    if not (isinstance(other,_IntVarImpl) or isinstance(other, _BoolVarImpl) ):
        return orig(self, other)
    else:
        if not isinstance(other,_BoolVarImpl):
            raise CPMpyException(f" logical conjunction involving an IntVar ({other}) is not allowed")
        return Operator("or", [self, other])

@hook(_IntVarImpl)
def __or__(self, other):
    if not isinstance(self, _BoolVarImpl):
        raise CPMpyException(f" logical conjunction involving an IntVar ({other}) is not allowed")
    else:
        return Operator("or", [self, other])

@hook(_IntVarImpl)
def __and__(self, other):
    if not isinstance(self, _BoolVarImpl):
        raise CPMpyException(f" logical conjunction involving an IntVar ({other}) is not allowed")
    else:
        return Operator("and", [self, other])

@hook(_IntVarImpl)
def implies(self, other):
    if not isinstance(self, _BoolVarImpl):
        raise CPMpyException(f" logical conjunction involving an IntVar ({other}) is not allowed")
    else:
        return Operator("->", [self, other])


@hook(int)
def is_bool(self):
    if self > 1 or self < 0:
        return False
    else:
        return True