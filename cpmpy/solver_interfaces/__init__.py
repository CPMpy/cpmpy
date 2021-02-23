# solver interfaces, including status/stats
from enum import Enum

class SolverInterface:
    def __init__(self):
        self.name = "dummy"

    def supported(self):
        return True

    def solve(self, model):
        return SolverStats()

def get_supported_solvers():
    return [sv for sv in builtin_solvers if sv.supported()]

class ExitStatus(Enum):
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5

class SolverStats(object):
    def __init__(self):
        self.status = ExitStatus.NOT_RUN
        self.runtime = None
    
    def __repr__(self):
        return "{} ({} seconds)".format(self.status, self.runtime)

# builtin solvers implementing SolverInterface
from .minizinc_text import *
from .minizinc_python import *
from .ortools_python import *

# the order matters: default will be first supported one
builtin_solvers=[MiniZincPython(),MiniZincText(),ORToolsPython()]

