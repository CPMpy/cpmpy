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
        abs
"""
import builtins  # to use the original Python-builtins

from .utils import is_false_cst, is_true_cst, is_any_list
from .variables import NDVarArray, cpm_array
from .core import Expression, Operator
from .globalfunctions import Minimum, Maximum, Abs
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
        if is_false_cst(elem):
            return False  # no need to create constraint
        elif is_true_cst(elem):
            pass
        elif isinstance(elem, Expression) and elem.is_bool():
            collect.append(elem)
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
            collect.append(elem)
        else:
            raise Exception("Non-Boolean argument '{}' to 'any'".format(elem))
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("or", collect)
    return False


def max(*iterable, **kwargs):
    """
        max() overwrites the python built-in to support decision variables.

        if iterable does not contain CPMpy expressions, the built-in is called
        else a Maximum functional global constraint is constructed; no keyword
          arguments are supported in that case
    """
    if len(iterable) == 1:
        iterable = tuple(iterable[0])
    if not builtins.any(isinstance(elem, Expression) for elem in iterable):
        return builtins.max(*iterable, **kwargs)

    assert len(kwargs)==0, "max over decision variables does not support keyword arguments"
    return Maximum(iterable)


def min(*iterable, **kwargs):
    """
        min() overwrites the python built-in to support decision variables.

        if iterable does not contain CPMpy expressions, the built-in is called
        else a Minimum functional global constraint is constructed; no keyword
          arguments are supported in that case
    """
    if len(iterable) == 1:
        iterable = tuple(iterable[0])
    if not builtins.any(isinstance(elem, Expression) for elem in iterable):
        return builtins.min(*iterable, **kwargs)

    assert len(kwargs)==0, "min over decision variables does not support keyword arguments"
    return Minimum(iterable)


def sum(*iterable, **kwargs):
    """
        sum() overwrites the python built-in to support decision variables.

        if iterable does not contain CPMpy expressions, the built-in is called
        checks if all constants and uses built-in sum() in that case
    """
    if len(iterable) == 1:
        iterable = tuple(iterable[0]) # Fix generator polling
    if not builtins.any(isinstance(elem, Expression) for elem in iterable):
        return builtins.sum(iterable, **kwargs)

    assert len(kwargs)==0, "sum over decision variables does not support keyword arguments"
    return Operator("sum", iterable)


def abs(element):
    """
        abs() overwrites the python built-in to support decision variables.

        if the element given is not a CPMpy expression, the built-in is called
        else an Absolute functional global constraint is constructed.
    """
    if is_any_list(element):  # compat: not allowed by builtins.abs(), but allowed by numpy.abs()
        return cpm_array([abs(elem) for elem in element])

    if isinstance(element, Expression):
        # create global
        return Abs(element)
    
    return builtins.abs(element)

    
