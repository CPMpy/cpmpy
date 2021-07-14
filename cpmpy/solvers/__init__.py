"""
    CPMpy interfaces to (the Python API interface of) solvers

    Solvers typically use some of the generic transformations in
    `transformations` as well as specific reformulations to map the
    CPMpy expression to the solver's Python API
"""

from .utils import builtin_solvers, get_supported_solvers
from .ortools import CPM_ortools
from .pysat import CPM_pysat
# from minizinc import CPMpyMiniZinc # closed for maintenance
