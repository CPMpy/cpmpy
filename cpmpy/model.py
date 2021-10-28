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
from cpmpy.expressions.variables import _IntVarImpl, NDVarArray
import numpy as np
from .transformations.get_variables import get_variables_model
from .transformations.to_bool import to_bool_constraint, intvar_to_boolvar
from .expressions.core import Operator
from .expressions.utils import is_any_list
from .solvers.utils import SolverLookup
from .solvers.solver_interface import SolverInterface, SolverStatus, ExitStatus
# from .transformations.get_variables import get_variables, get_variables_model
# from .transformations.to_bool import intvar_to_boolvar, reified_intvar_to_boolvar, reify_translate_constraint, translate_constraint


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
            self.constraints = list(args[0])
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

    def int2bool_onehot(self):

        user_vars = get_variables_model(self)

        ivarmap, bool_cons = intvar_to_boolvar(user_vars)

        bool_model = Model(bool_cons)

        for constraint in self.constraints:
            bool_model += to_bool_constraint(constraint, ivarmap)

        return (ivarmap, bool_model)


# class BoolModel(Model):
#     """
#         CPMpy Model object, contains the constraint and objective expressions
#     """
#     def __init__(self, *args, int_model=None, reify=False):
#         self.constraints = []
#         self.reification_vars = []
#         # self.constraints = []
#         self.reify = reify
#         # int var to boolvar mapping
#         # boolvar to intvar-value mapping
#         self.int_var_mapping, self.bool_var_mapping = {}, {}

#         if len(args) == 0:
#             self.constraints = []
#         elif len(args) == 1 and is_any_list(args[0]):
#             # top level list of constraints
#             self.constraints = []
#             for con in list(args[0]):
#                 self.__add__(con=con)
#         else:
#             for con in args:
#                 print("con=", con)
#                 self.__add__(con=con)

#     def from_int_model(self, int_model, reify=False):
#         if reify:
#             int_model_variables = get_variables_model(int_model)

#             iv_mapping, bv_mapping, constraints = intvar_to_boolvar(int_model_variables)

#             self.constraints += constraints

#             self.int_var_mapping.update(iv_mapping)
#             self.bool_var_mapping.update(bv_mapping)

#             for constraint in int_model.constraints:
#                 new_constraint = translate_constraint(constraint, self.int_var_mapping)
#                 self.constraints += new_constraint
#         else:

#             int_model_variables = get_variables_model(int_model)

#             iv_mapping, bv_mapping, constraints, reification_vars = reified_intvar_to_boolvar(int_model_variables)

#             self.reification_vars += reification_vars
#             self.constraints += constraints
#             self.int_var_mapping.update(iv_mapping)
#             self.bool_var_mapping.update(bv_mapping)

#             for constraint in int_model.constraints:
#                 new_constraint, reification_vars = reify_translate_constraint(constraint, self.int_var_mapping)
#                 self.constraints += new_constraint
#                 self.reification_vars += reification_vars

#     def __add__(self, con):
#         if is_any_list(con) and len(con) == 1 and is_any_list(con[0]):
#             # top level list of constraints
#             con = con[0]

#         constraint_intvars = [ivar for ivar in get_variables(con)if ivar not in self.int_var_mapping]
#         print(constraint_intvars)

#         if self.reify:
#             iv_mapping, bv_mapping, constraints, reification_vars = reified_intvar_to_boolvar(constraint_intvars)

#             self.reification_vars += reification_vars
#             self.constraints += constraints
#             self.int_var_mapping.update(iv_mapping)
#             self.bool_var_mapping.update(bv_mapping)

#             new_constraint, reification_vars = reify_translate_constraint(con, self.int_var_mapping)
#             self.reification_vars += reification_vars
#             self.constraints += new_constraint + constraints
#         else:
#             iv_mapping, bv_mapping, constraints = intvar_to_boolvar(constraint_intvars)

#             self.int_var_mapping.update(iv_mapping)
#             self.bool_var_mapping.update(bv_mapping)

#             new_constraint = translate_constraint(con, self.int_var_mapping)
#             self.constraints += new_constraint + constraints



#         return self

#     def __repr__(self):
#         cons_str = ""
#         for c in self.constraints:
#             cons_str += "\t{}\n".format(c)

#         obj_str = ""

#         if not self.objective is None:
#             if self.objective_max:
#                 obj_str = "maximize "
#             else:
#                 obj_str = "minimize "
#         obj_str += str(self.objective)

#         return "Constraints:\n{}\n\nReification Vars:\n\n\t{}\n\nObjective: {}".format(cons_str, self.reification_vars ,obj_str)

#     def get_assignment(self):
#         all_bool_vars = get_variables_model(self)

#         for var in all_bool_vars:
#             if var.value():
#                 intvar, value = self.bool_var_mapping[var]
#                 intvar._value = value
