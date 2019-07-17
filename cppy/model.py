from enum import Enum
from .expressions import *
import numpy as np

class ExitStatus(Enum):
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5


class Model(object):
    def __init__(self, *args):
        self.constraints = []
        self.objective = None
        
        # flatten and group all constraints and objective(s?)
        for arg in flatten(args):
            if isinstance(arg, Objective):
                # an objective function
                if self.objective == None:
                    self.objective = arg
                elif isinstance(self.objective, list):
                    self.objective.append(arg)
                else:
                    self.objective = [self.objective, arg]
            else:
                # a constraint
                self.constraints.append(arg)

    def __repr__(self):
        return "Constraints: {}\nObjective: {}".format(self.constraints, self.objective)
    
    def solve(self):
        return SolverStats()


class SolverStats(object):
    def __init__(self):
        self.status = ExitStatus.NOT_RUN
        self.runtime = None
    
    def __repr__(self):
        return "{} ({} seconds)".format(self.status, self.runtime)

# http://jugad2.blogspot.com/2014/10/flattening-arbitrarily-nested-list-in.html
def flatten(iterable):
    for elem in iterable:
        if isinstance(elem, (list,tuple)):
            yield from flatten(elem)
        else:
            yield elem
