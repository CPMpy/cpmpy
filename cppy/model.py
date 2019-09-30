from .expressions import *
from .solver_interfaces import *
import numpy as np


class Model(object):
    def __init__(self, *args, minimize=None, maximize=None):
        assert ((minimize is None) or (maximize is None)), "can not set both minimize and maximize"
        # list of constraints
        self.constraints = [self.make_and_from_list(c) for c in args]
        # an expresion or None
        self.objective = None
        self.objective_max = None

        if not maximize is None:
            self.objective = maximize
            self.objective_max = True
        if not minimize is None:
            self.objective = minimize
            self.objective_max = False

    def make_and_from_list(self, args):
        lst = list(args) # make mutable copy of type list
        # do recursive where needed, with overwrite
        for (i, expr) in enumerate(lst):
            if isinstance(expr, list):
                lst[i] = self.make_and_from_list(expr)
        if len(lst) == 1:
            return lst[0]
        return Operator("and", lst)
        

    def __repr__(self):
        cons_str = ""
        for c in self.constraints:
            cons_str += "    {}\n".format(c)

        obj_str = ""
        if not self.objective is None:
            if self.objective_max:
                obj_str = "maximize "
            else:
                obj_str = "minimize "
        obj_str += str(self.objective)
            
        return "Constraints:\n{}Objective: {}".format(cons_str, obj_str)
    
    # solver: name of supported solver or any SolverInterface object
    def solve(self, solver=None):
        # default solver?
        if solver is None:
            solver = SolverInterface()
        elif not isinstance(solver, SolverInterface):
            solverdict = get_supported_solvers()
            if not solver in solverdict:
                raise "'{}' is not in the list of supported solvers and not a SolverInterface object".format(solver)
            solver = solverdict[solver]
                
        return solver.solve(self)

