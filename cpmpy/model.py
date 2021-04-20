#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## model.py
##
"""
    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        Model

    ==================
    Module description
    ==================

    Class that contains the constraints and the objective function,
    and that provides an easy solve() abstraction which will call a solver.

    See the examples for basic usage, which involves:

    - creation, e.g. m = Model(cons, minimize=obj)
    - solving, e.g. m.solve()
    - optionally, checking status/runtime, e.g. m.status()

    ==============
    Module details
    ==============
"""
import numpy as np
from .expressions import Operator
from .solver_interfaces.util import get_supported_solvers
from .solver_interfaces.solver_interface import SolverInterface, SolverStatus, ExitStatus


class Model(object):
    """
    CPMpy Model object, contains the constraint and objective expression trees

    Arguments of constructor:

    - `*args`: Expression object(s) or list(s) of Expression objects
    - `minimize`: Expression object representing the objective to minimize
    - `maximize`: Expression object representing the objective to maximize

    At most one of minimize/maximize can be set, if none are set, it is assumed to be a satisfaction problem
    """
    def __init__(self, *args, minimize=None, maximize=None):
        assert ((minimize is None) or (maximize is None)), "can not set both minimize and maximize"
        self.cpm_status = SolverStatus("Model") # status of solving this model, will be replaced

        # list of constraints (arguments of top-level conjunction)
        if len(args) == 0 or (len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 0): # None or empty list
            self.constraints = []
        else:
            root_constr = self._make_and_from_list(args)
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
        
    def __add__(self, cons):
        """
            Add one or more constraints to the model

            m = Model()
            m += [x > 0]
        """
        self.constraints += cons
        return self

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
    def solve(self, solver=None, time_limit=None):
        """ Send the model to a solver and get the result

        :param solver: solver to use (verifies that the solver is supported on the current system)
        :type solver: None (default) or in [s.name in get_supported_solvers()] or a SolverInterface class (Class, not object! e.g. CPMpyOrTools, not CPMpyOrTools()!)

        :param time_limit: optional, time limit in seconds
        :type time_limit: int or float

        :return: the computed output:
            - True      if it is a satisfaction problem and it is satisfiable
            - False     if it is a satisfaction problem and not satisfiable
            - [int]     if it is an optimisation problem
        """
        solver_class = None
        # get supported solvers
        supsolvers = get_supported_solvers()
        if solver is None: # default is first
            solver_class = supsolvers[0]
        elif not isinstance(solver, SolverInterface):
            solvername = solver
            for s in supsolvers:
                if s.name == solvername:
                    solver_class = s
                    break # break and hence 'solver' is correct object
        elif isinstance(solver, SolverInterface) and solver.supported():
            solver_class = solver
        else:
            raise Exception("'{}' is not in the list of supported solvers and not a SolverInterface class".format(solver))
        assert(solver_class is not None)
                
        # instatiate solver with this model
        s = solver_class(self)
        # call solver
        ret = s.solve(time_limit=time_limit)
        # store CPMpy status (s object has no further use)
        self.cpm_status = s.status()
        return ret

    def status(self):
        """
            Returns the status of the latest solver run on this model

            Status information includes exit status (optimality) and runtime.

        :return: an object of :class:`SolverStatus`
        """
        return self.cpm_status

    def _make_and_from_list(self, args):
        """ recursively reads a list of Expression and returns the 'And' conjunctive of the elements in the list """
        lst = list(args) # make mutable copy of type list
        # do recursive where needed, with overwrite
        for (i, expr) in enumerate(lst):
            if isinstance(expr, list):
                lst[i] = self._make_and_from_list(expr)
        if len(lst) == 1:
            return lst[0]
        return Operator("and", lst)
