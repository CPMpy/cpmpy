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
        gurobi
        pysdd
        z3
        exact
        utils

    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        CPM_ortools
        CPM_pysat
        CPM_gurobi
        CPM_pysdd
        CPM_z3
        CPM_exact

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        param_combinations
"""

from .utils import builtin_solvers, get_supported_solvers, param_combinations
from .ortools import CPM_ortools
from .pysat import CPM_pysat
from .minizinc import CPM_minizinc
from .gurobi import  CPM_gurobi
from .pysdd import CPM_pysdd
from .z3 import CPM_z3
from .exact import CPM_exact
