from .expressions import *
from .solver_interfaces import *
import numpy as np


class Model(object):
    """
    CpMPy Model object, contains the constraint and objective expression trees

    Arguments of constructor:
    *args: Expression object(s) or list(s) of Expression objects
    minimize: Expression object representing the objective to minimize
    maximize: Expression object representing the objective to maximize

    At most one of minimize/maximize can be set, if none are set, it is assumed to be a satisfaction problem
    """
    def __init__(self, *args, minimize=None, maximize=None):
        assert ((minimize is None) or (maximize is None)), "can not set both minimize and maximize"
        # list of constraints (arguments of top-level conjunction)
        root_constr = self.make_and_from_list(args)
        if root_constr.name == 'and':
            # unpack top-level conjuction
            self.constraints = root_constr.args
        else:
            # wrap in list
            self.constraints = [root_constr]

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
        """ recursively reads a list of Expression and returns the 'And' conjunctive of the elements in the list """
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
        """ Send the model to a solver and get the result

        'solver': None (default) or in [s.name in get_supported_solvers()] or a SolverInterface object
        verifies that the solver is supported on the current system
        """
        # get supported solvers
        supsolvers = get_supported_solvers()
        if solver is None: # default is first
            solver = supsolvers[0]
        elif not isinstance(solver, SolverInterface):
            solvername = solver
            for s in supsolvers:
                if s.name == solvername:
                    solver = s
                    break # break and hence 'solver' is correct object

            if not isinstance(solver, SolverInterface) or not solver.supported():
                raise Exception("'{}' is not in the list of supported solvers and not a SolverInterface object".format(solver))
                
        return solver.solve(self)

