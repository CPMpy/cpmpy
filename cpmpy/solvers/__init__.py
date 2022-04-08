"""
    CPMpy interfaces to (the Python API interface of) solvers

    Solvers typically use some of the generic transformations in
    `transformations` as well as specific reformulations to map the
    CPMpy expression to the solver's Python API

    ==================
    List of submodules
    ==================
    .. autosummary::
        :nosignatures:

        ortools
        pysat
        minizinc
        gurobi
        utils

    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        CPM_ortools
        CPM_pysat
        CPM_minizinc
        CPM_gurobi

        SolverLookup

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        get_supported_solvers
        builtin_solvers
        param_combinations
      
"""

from .utils import builtin_solvers, get_supported_solvers, param_combinations
from .ortools import CPM_ortools
from .pysat import CPM_pysat
from .minizinc import CPM_minizinc
from .gurobi import  CPM_gurobi
