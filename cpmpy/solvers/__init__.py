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
        pysat_rc2
        z3
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
        CPM_RC2
        CPM_z3

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        param_combinations
"""

from .utils import builtin_solvers, get_supported_solvers, param_combinations
from .ortools import CPM_ortools
from .minizinc import CPM_minizinc
from .gurobi import  CPM_gurobi
from .pysdd import CPM_pysdd
from .pysat import CPM_pysat
from .pysat_rc2 import CPM_RC2
from .z3 import CPM_z3
