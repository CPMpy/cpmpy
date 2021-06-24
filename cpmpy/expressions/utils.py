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

# Helpers for type checking
def is_int(arg):
    return isinstance(arg, (int, np.integer))
def is_num(arg):
    return isinstance(arg, (int, np.integer, float, np.float64))
def is_pure_list(arg):
    return isinstance(arg, (list, tuple))
def is_any_list(arg):
    return isinstance(arg, (list, tuple, np.ndarray))
