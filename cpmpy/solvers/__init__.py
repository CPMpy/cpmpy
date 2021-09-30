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
        utils

    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        CPM_ortools
        CPM_pysat

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
