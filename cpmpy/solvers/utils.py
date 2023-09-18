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

        param_combinations
"""

import warnings # for deprecation warning

from .gurobi import CPM_gurobi
from .ortools import CPM_ortools
from .minizinc import CPM_minizinc
from .pysat import CPM_pysat
from .z3 import CPM_z3
from .pysdd import CPM_pysdd
from .exact import CPM_exact
from .choco import CPM_choco

def param_combinations(all_params, remaining_keys=None, cur_params=None):
    """
        Recursively yield all combinations of param values

        For example usage, see `examples/advanced/hyperparameter_search.py`

        - all_params is a dict of {key: list} items, e.g.:
            {'val': [1,2], 'opt': [True,False]}

        - output is an generator over all {key:value} combinations
          of the keys and values. For the example above:
          generator([{'val':1,'opt':True},{'val':1,'opt':False},{'val':2,'opt':True},{'val':2,'opt':False}])
    """
    if remaining_keys is None or cur_params is None:
        # init
        remaining_keys = list(all_params.keys())
        cur_params = dict()

    cur_key = remaining_keys[0]
    myresults = [] # (runtime, cur_params)
    for cur_value in all_params[cur_key]:
        cur_params[cur_key] = cur_value
        if len(remaining_keys) == 1:
            # terminal, return copy
            yield dict(cur_params)
        else:
            # recursive call
            yield from param_combinations(all_params, 
                            remaining_keys=remaining_keys[1:],
                            cur_params=cur_params)

class SolverLookup():
    @staticmethod
    def base_solvers():
        """
            Return ordered list of (name, class) of base CPMpy
            solvers

            First one is default
        """
        return [("ortools", CPM_ortools),
                ("z3", CPM_z3),
                ("minizinc", CPM_minizinc),
                ("gurobi", CPM_gurobi),
                ("pysat", CPM_pysat),
                ("pysdd", CPM_pysdd),
                ("exact", CPM_exact),
                ("choco", CPM_choco),
               ]

    @staticmethod
    def solvernames():
        names = []
        for (basename, CPM_slv) in SolverLookup.base_solvers():
            if CPM_slv.supported():
                names.append(basename)
                if hasattr(CPM_slv, "solvernames"):
                    subnames = CPM_slv.solvernames()
                    for subn in subnames:
                        names.append(basename+":"+subn)
        return names

    @staticmethod
    def get(name=None, model=None):
        """
            get a specific solver (by name), with 'model' passed to its constructor

            This is the preferred way to initialise a solver from its name
        """
        cls = SolverLookup.lookup(name=name)

        # check for a 'solver:subsolver' name
        subname = None
        if name is not None and ':' in name:
            _,subname = name.split(':',maxsplit=1)
        return cls(model, subsolver=subname)

    @staticmethod
    def lookup(name=None):
        """
            lookup a solver _class_ by its name

            warning: returns a 'class', not an object!
            see get() for normal uses
        """
        if name is None:
            # first solver class
            return SolverLookup.base_solvers()[0][1]

        # split name if relevant
        solvername = name
        subname = None
        if ':' in solvername:
            solvername,_ = solvername.split(':',maxsplit=1)


        for (basename, CPM_slv) in SolverLookup.base_solvers():
            if basename == solvername:
                # found the right solver
                return CPM_slv

        return None


# using `builtin_solvers` is DEPRECATED, use `SolverLookup` object instead
# Order matters! first is default, then tries second, etc...
builtin_solvers = [CPM_ortools, CPM_gurobi, CPM_minizinc, CPM_pysat, CPM_exact, CPM_choco]
def get_supported_solvers():
    """
        Returns a list of solvers supported on this machine.

    :return: a list of SolverInterface sub-classes :list[SolverInterface]:
    """
    warnings.warn("Deprecated, use Model.solvernames() instead, will be removed in stable version", DeprecationWarning)
    return [sv for sv in builtin_solvers if sv.supported()]
