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
"""

import numpy as np
from collections.abc import Iterable # for _flatten

# Helpers for type checking
def is_int(arg):
    return isinstance(arg, (int, np.integer))
def is_num(arg):
    return isinstance(arg, (int, np.integer, float, np.float64))
def is_pure_list(arg):
    return isinstance(arg, (list, tuple))
def is_any_list(arg):
    return isinstance(arg, (list, tuple, np.ndarray))

def flatlist(args):
    return list(_flatten(args))
def _flatten(args):
    # from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    # returns an iterator, not a list
    for el in args:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
