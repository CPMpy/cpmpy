from .expressions import *
from .solver_interfaces import *
import numpy as np


class Model(object):
    def __init__(self, *args):
        self.constraints = []
        self.objective = None
        
        # filter out the objective instances
        cons = self.filter_objectives(args)
        # turn lists into 'AND' constraints
        self.constraints = self.make_and_from_list(cons)

    # take args, put objectives in self
    # return same args but without objectives (to keep structure)
    def filter_objectives(self, args):
        lst = list(args) # make mutable copy of type list
        for (i, expr) in enumerate(lst):
            if isinstance(expr, Objective):
                self.add_objective(expr)
                del lst[i] # filter out from list
            elif isinstance(expr, (list,tuple,np.ndarray)):
                lst[i] = self.filter_objectives(expr) # recursive
        return lst

    def add_objective(self, arg):
        # an objective function
        if self.objective is None:
            self.objective = arg
        elif isinstance(self.objective, list):
            self.objective.append(arg)
        else:
            self.objective = [self.objective, arg]

    def make_and_from_list(self, lst):
        # do recursive where needed, with overwrite
        for (i, expr) in enumerate(lst):
            if isinstance(expr, list):
                lst[i] = self.make_and_from_list(expr)
        if len(lst) == 1:
            return lst[0]
        return BoolOperator("and", lst)
        

    def __repr__(self):
        cons_str = ""
        # pretty-printing of first-level grouping (if present):
        if isinstance(self.constraints, BoolOperator) and self.constraints.name == "and":
            cons_str += "\n"
            for elem in self.constraints.elems:
                cons_str += "    {}\n".format(elem)
        else: # top level constraint is not an 'and'
            cons_str += self.constraints.__repr__()
            
        return "Constraints: {}\nObjective: {}".format(cons_str, self.objective)
    
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

