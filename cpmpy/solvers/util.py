from .minizinc import CPMpyMiniZinc
from .ortools import CPMpyORTools

def get_supported_solvers():
    """
        Returns a list of currently supported solvers.

    :return: a list of SolverInterface sub-classes :list[SolverInterface]:
    """
    return [sv for sv in builtin_solvers if sv.supported()]

# Order matters! first is default, then tries second, etc...
builtin_solvers=[CPMpyORTools,CPMpyMiniZinc]
