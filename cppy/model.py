from .expressions import *
from .solver_interfaces import *
import numpy as np


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
    
    # solver: name of supported solver or any SolverInterface object
    def solve(self, solver=None):
        # default solver?
        if solver == None:
            solver = SolverInterface()
        elif not isinstance(solver, SolverInterface):
            solverdict = get_supported_solvers()
            if not solver in solverdict:
                raise "'{}' is not in the list of supported solvers and not a SolverInterface object".format(solver)
            solver = solverdict[solver]
                
        return solver.solve(self)


# http://jugad2.blogspot.com/2014/10/flattening-arbitrarily-nested-list-in.html
def flatten(iterable):
    for elem in iterable:
        if isinstance(elem, (list,tuple)):
            yield from flatten(elem)
        else:
            yield elem
