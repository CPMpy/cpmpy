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
"""

import numpy as np
from collections.abc import Iterable # for _flatten
from itertools import chain, combinations

def is_int(arg):
    """ is it an integer? (incl numpy variants)
    """
    return isinstance(arg, (int, np.integer))
def is_num(arg):
    """ is it an int or float? (incl numpy variants)
    """
    return isinstance(arg, (int, np.integer, float, np.float64))
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
