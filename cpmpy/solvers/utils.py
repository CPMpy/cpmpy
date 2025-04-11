#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## utils.py
##
"""
    Utilities for handling solvers

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
from .gcs import CPM_gcs
from .pysdd import CPM_pysdd
from .exact import CPM_exact
from .choco import CPM_choco
from .cpo   import CPM_cpo

def param_combinations(all_params, remaining_keys=None, cur_params=None):
    """
        Recursively yield all combinations of param values

        For example usage, see `examples/advanced/hyperparameter_search.py`
        https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/hyperparameter_search.py

        - all_params is a dict of `{key: list}` items, e.g.:
          ``{'val': [1,2], 'opt': [True,False]}``

        - output is an generator over all `{key:value}` combinations
          of the keys and values. For the example above:
          ``generator([{'val':1,'opt':True},{'val':1,'opt':False},{'val':2,'opt':True},{'val':2,'opt':False}])``
    """
    if remaining_keys is None or cur_params is None:
        # init
        remaining_keys = list(all_params.keys())
        cur_params = dict()

    cur_key = remaining_keys[0]
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
    @classmethod
    def base_solvers(cls):
        """
            Return ordered list of (name, class) of base CPMpy
            solvers

            First one is default
        """
        return [("ortools", CPM_ortools),
                ("z3", CPM_z3),
                ("minizinc", CPM_minizinc),
                ("gcs", CPM_gcs),
                ("gurobi", CPM_gurobi),
                ("pysat", CPM_pysat),
                ("pysdd", CPM_pysdd),
                ("exact", CPM_exact),
                ("choco", CPM_choco),
                ("cpo", CPM_cpo),
               ]

    @classmethod
    def solvernames(cls):
        names = []
        for (basename, CPM_slv) in cls.base_solvers():
            if CPM_slv.supported():
                names.append(basename)
                if hasattr(CPM_slv, "solvernames"):
                    subnames = CPM_slv.solvernames()
                    for subn in subnames:
                        names.append(basename+":"+subn)
        return names

    @classmethod
    def get(cls, name=None, model=None):
        """
            get a specific solver (by name), with 'model' passed to its constructor

            This is the preferred way to initialise a solver from its name
        """
        solver_cls = cls.lookup(name=name)

        # check for a 'solver:subsolver' name
        subname = None
        if name is not None and ':' in name:
            _,subname = name.split(':',maxsplit=1)
        return solver_cls(model, subsolver=subname)

    @classmethod
    def lookup(cls, name=None):
        """
            lookup a solver _class_ by its name

            warning: returns a 'class', not an object!
            see get() for normal uses
        """
        if name is None:
            # first solver class
            return cls.base_solvers()[0][1]

        # split name if relevant
        solvername = name
        subname = None
        if ':' in solvername:
            solvername,_ = solvername.split(':',maxsplit=1)

        for (basename, CPM_slv) in cls.base_solvers():
            if basename == solvername:
                # found the right solver
                return CPM_slv
        raise ValueError(f"Unknown solver '{name}', chose from {cls.solvernames()}")
    

    @classmethod
    def status(cls):
        result = []
        for (basename, CPM_slv) in cls.base_solvers():
            installed = CPM_slv.supported()
            version = CPM_slv.version() if installed and hasattr(CPM_slv, 'version') else None
            
            # Collect main solver status
            result.append((basename, installed, version))
            
            # TODO: Can add subsolver status report once pull request #623 has been resolved
            # # Handle subsolvers if applicable
            # if installed and hasattr(CPM_slv, 'solvernames'):
            #     subnames = CPM_slv.solvernames()
            #     installed_subnames = CPM_slv.solvernames(installed=True)
            #     for subn in subnames:
            #         is_installed = subn in installed_subnames
            #         subsolver_status = (basename + ":" + subn, is_installed, None)  # No version information for subsolvers in this example
            #         result.append(subsolver_status)  # Append subsolver status
        return result


    @classmethod
    def print_status(cls):
        """
        Prints a tabulated status report of the different solvers,
        i.e. whether they are installed on the system and if so which version.
        """
        
        # Get the status information using the status() method
        solver_status = cls.status()

        # Print the header
        print(f"{'Solver':<25} {'Installed':<10} {'Version':<15}")
        print("-" * 50)

        # Iterate over the solver status
        for basename, installed, version in solver_status:

            # TODO: Can add subsolver status report once pull request #623 has been resolved
            # # If this is a subsolver (indicated by a ':' in the name), indent the output
            # if ':' in basename:
            #     print(f" ↪ {basename.split(":")[-1]:<22} {'Yes' if installed else 'No':<10} {' ':<15}")  # Subsolver with indentation
            # else:

            # For main solvers, show version if available
            version = version if version else "Not found" if installed else "-"
            print(f"{basename:<25} {'Yes' if installed else 'No':<10} {version:<15}")


# using `builtin_solvers` is DEPRECATED, use `SolverLookup` object instead
# Order matters! first is default, then tries second, etc...
builtin_solvers = [CPM_ortools, CPM_gurobi, CPM_minizinc, CPM_pysat, CPM_exact, CPM_choco]
def get_supported_solvers():
    """
        Returns a list of solvers supported on this machine.
       
        .. deprecated:: 0.9.4
            Please use :class:`SolverLookup` object instead.

        :return: a list of SolverInterface sub-classes :list[SolverInterface]:
    """
    warnings.warn("Deprecated, use Model.solvernames() instead, will be removed in stable version", DeprecationWarning)
    return [sv for sv in builtin_solvers if sv.supported()]
