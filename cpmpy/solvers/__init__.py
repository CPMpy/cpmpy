"""
    CPMpy interfaces to (the Python API interface of) solvers

    Solvers typically use some of the generic transformations in
    :mod:`cpmpy.transformations` as well as specific reformulations to map the
    CPMpy expression to the solver's Python API.

    =========================
    List of helper submodules
    =========================
    .. autosummary::
        :nosignatures:

        solver_interface
        utils

    =========================
    List of solver submodules
    =========================
    .. autosummary::
        :nosignatures:

        ortools
        minizinc
        pysat
        gurobi
        pysdd
        z3
        exact
        choco
        gcs
        cpo

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
from .choco import CPM_choco
from .gcs import CPM_gcs
from .cpo import CPM_cpo

