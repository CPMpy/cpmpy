# solver interfaces, including status/stats
from enum import Enum

class SolverInterface:
    def __init__(self):
        self.name = "dummy"

    def supported(self):
        return True

    def solve(self, model):
        return SolverStats()


# builtin solvers implementing SolverInterface
from .minizinc_text import *

builtin_solvers=[MiniZincText()]

def get_supported_solvers():
    solverdict = dict()
    for solver in builtin_solvers:
        if solver.supported():
            solverdict[solver.name] = solver
    return solverdict


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
