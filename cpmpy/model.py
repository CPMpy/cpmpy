#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## model.py
##
"""
    The `Model` class is a lazy container for constraints and an objective function.

    It is lazy in that it only stores the constraints and objective that are added
    to it. Processing only starts when solve() is called, and this does not modify
    the constraints or objective stored in the model.

    A model can be solved multiple times, and constraints can be added to it inbetween
    solve calls.

    See the examples for basic usage, which involves:

    - creation, e.g. m = Model(cons, minimize=obj)
    - solving, e.g. m.solve()
    - optionally, checking status/runtime, e.g. m.status()

    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        Model
"""
import numpy as np
from .expressions.core import Operator
from .expressions.utils import is_any_list
from .solvers.utils import SolverLookup
from .solvers.solver_interface import SolverInterface, SolverStatus, ExitStatus

class Model(object):
    """
    CPMpy Model object, contains the constraint and objective expressions
    """

    def __init__(self, *args, minimize=None, maximize=None):
        """
            Arguments of constructor:

            - `*args`: Expression object(s) or list(s) of Expression objects
            - `minimize`: Expression object representing the objective to minimize
            - `maximize`: Expression object representing the objective to maximize

            At most one of minimize/maximize can be set, if none are set, it is assumed to be a satisfaction problem
        """
        assert ((minimize is None) or (maximize is None)), "can not set both minimize and maximize"
        self.cpm_status = SolverStatus("Model") # status of solving this model, will be replaced

        # list of constraints
        if len(args) == 0:
            self.constraints = []
        elif len(args) == 1 and is_any_list(args[0]):
            # top level list of constraints
            self.constraints = args[0]
        else:
            self.constraints = list(args) # instead of tuple

        # objective: an expresion or None
        self.objective = None
        self.objective_max = None
        if maximize is not None:
            self.maximize(maximize)
        if minimize is not None:
            self.minimize(minimize)
        
    def __add__(self, con):
        """
            Add one or more constraints to the model

            m = Model()
            m += [x > 0]
        """
        if is_any_list(con) and len(con) == 1 and is_any_list(con[0]):
            # top level list of constraints
            con = con[0]
        self.constraints.append(con)
        return self

    def minimize(self, expr):
        """
            Minimize the given objective function

            `minimize()` can be called multiple times, only the last one is stored
        """
        self.objective = expr
        self.objective_max = False

    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is stored
        """
        self.objective = expr
        self.objective_max = True

    
    # solver: name of supported solver or any SolverInterface object
    def solve(self, solver=None, time_limit=None):
        """ Send the model to a solver and get the result

        :param solver: name of a solver to use. Run SolverLookup.solvernames() to find out the valid solver names on your system. (default: None = first available solver)
        :type string: None (default) or in SolverLookup.solvernames() or a SolverInterface class (Class, not object! e.g. CPMpyOrTools, not CPMpyOrTools()!)

        :param time_limit: optional, time limit in seconds
        :type time_limit: int or float

        :return: the computed output:
            - True      if it is a satisfaction problem and it is satisfiable
            - False     if it is a satisfaction problem and not satisfiable
            - [int]     if it is an optimisation problem
        """
        if isinstance(solver, SolverInterface):
            solver_class = solver
        else:
            solver_class = SolverLookup.lookup(solver)
        assert(solver_class is not None)
                
        # instatiate solver with this model
        if isinstance(solver, str) and ':' in solver:
            # solver is a name that contains a subsolver
            s = solver_class(self, solver=solver)
        else:
            # no subsolver
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
