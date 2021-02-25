from .minizinc_python import MiniZincPython
from .minizinc_text import MiniZincText
from .ortools_python import ORToolsPython

def get_supported_solvers():
    """
        Returns a list of currently supported solvers.

    :return: a list of SolverInterface sub-classes :list[SolverInterface]:
    """
    return [sv for sv in builtin_solvers if sv.supported()]

builtin_solvers=[MiniZincPython(),MiniZincText(),ORToolsPython()]