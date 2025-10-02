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
from .pumpkin import CPM_pumpkin
from .cpo   import CPM_cpo
from .cplex import CPM_cplex
from .pindakaas import CPM_pindakaas
from .hexaly import CPM_hexaly

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
        return [
                ("ortools", CPM_ortools),
                ("z3", CPM_z3),
                ("minizinc", CPM_minizinc),
                ("gcs", CPM_gcs),
                ("gurobi", CPM_gurobi),
                ("pysat", CPM_pysat),
                ("pysdd", CPM_pysdd),
                ("exact", CPM_exact),
                ("choco", CPM_choco),
                ("pumpkin", CPM_pumpkin),
                ("cpo", CPM_cpo),
                ("cplex", CPM_cplex),
                ("pindakaas", CPM_pindakaas),
                ("hexaly", CPM_hexaly)
               ]

    @classmethod
    def print_status(cls):
        """
            Print all CPMpy solvers and their installation status on this system.
        """
        for (basename, CPM_slv) in cls.base_solvers():
            if CPM_slv.supported():
                print(f"{basename}: Supported, ready to use.")
            else:
                print(f"{basename}: Not supported (missing Python package, binary or license).")

    @classmethod
    def supported(cls):
        """
            Return the list of names of all solvers (and subsolvers) supported on this system.

            If a solver name is returned, it means that the solver's `.supported()` function returns True
            and it is hence ready for immediate use
            (e.g. any separate binaries are also installed if necessary, and licenses are active if needed).

            Typical use case is to use these names in `SolverLookup.get(name)`.
        """
        names = []
        for (basename, CPM_slv) in cls.base_solvers():
            if CPM_slv.supported():
                names.append(basename)
                if hasattr(CPM_slv, "solvernames"):
                    subnames = CPM_slv.solvernames(installed=True)
                    for subn in subnames:
                        names.append(basename+":"+subn)
        return names

    @classmethod
    def solvernames(cls):
        # The older (more indirectly named) way to get the list of names of *supported* solvers.
        # Will be deprecated at some point.
        return cls.supported()

    @classmethod
    def get(cls, name=None, model=None, **init_kwargs):
        """
            get a specific solver (by name), with 'model' passed to its constructor

            This is the preferred way to initialise a solver from its name

            :param name: name of the solver to use
            :param model: model to pass to the solver constructor
            :param init_kwargs: additional keyword arguments to pass to the solver constructor
        """
        solver_cls = cls.lookup(name=name)

        # check for a 'solver:subsolver' name
        subname = None
        if name is not None and ':' in name:
            _,subname = name.split(':',maxsplit=1)
        return solver_cls(model, subsolver=subname, **init_kwargs)

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
        raise ValueError(f"Unknown solver '{name}', choose from {cls.solvernames()}")
 

    @classmethod
    def version(cls):
        """
        Returns an overview of all solvers supported by CPMpy as a list of dicts.

        Each dict consists of:

        - "name": <base_solver> or <base_solver>:<subsolver>
        - "installed": install status (True/False)
        - "version": version of solver's Python library (or one of its subsolvers if applicable)
        """
        result = []
        for (basename, CPM_slv) in cls.base_solvers():
            installed = CPM_slv.supported()
            version = CPM_slv.version() if installed and hasattr(CPM_slv, 'version') else None
            
            # Collect main solver status
            result.append({
                    "name": basename,
                    "installed": installed, 
                    "version": version,
                })
            
            # Handle subsolvers if applicable
            if installed and hasattr(CPM_slv, 'solvernames'):
                subnames = CPM_slv.solvernames()
                installed_subnames = CPM_slv.solvernames(installed=True)
                for subn in subnames:
                    is_installed = subn in installed_subnames
                    subsolver_status = {
                        "name": basename + ":" + subn, 
                        "installed": is_installed, 
                        "version": CPM_slv.solverversion(subn) if installed else None,
                    }
                    result.append(subsolver_status)  # Append subsolver status
        return result


    @classmethod
    def print_version(cls):
        """
        Prints a tabulated report on the different solvers supported by CPMpy,
        i.e. whether they are installed on the current system and if so which version.
        """
        
        # Get the solver information using the version() method
        solver_versions = cls.version()

        # Print the header
        print(f"{'Solver':<25} {'Installed':<10} {'Version':<15}")
        print("-" * 50)

        # Iterate over the solvers
        for solver_version in solver_versions:
            basename, installed, version = solver_version["name"], solver_version["installed"], solver_version["version"]

            # If this is a subsolver (indicated by a ':' in the name), indent the output
            if ':' in basename:
                print(f" ↪ {basename.split(':')[-1]:<22} {'Yes' if installed else 'No':<10} {(version if version else ' '):<15}")  # Subsolver with indentation
            else:
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
