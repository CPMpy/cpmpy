#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## utils.py
##
"""
    Utilities for handling solvers

    Contains a static variable `builtin_solvers` that lists
    CPMpy solvers (first one is the default solver by default)

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        get_supported_solvers
"""

#from .minizinc import CPMpyMiniZinc # closed for maintenance
from .ortools import CPM_ortools
from .pysat import CPM_pysat

def get_supported_solvers():
    """
        Returns a list of solvers supported on this machine.

    :return: a list of SolverInterface sub-classes :list[SolverInterface]:
    """
    return [sv for sv in builtin_solvers if sv.supported()]

# Order matters! first is default, then tries second, etc...
builtin_solvers=[CPM_ortools,CPM_pysat]
