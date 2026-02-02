#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## model.py
##
"""
    The :class:`Model <cpmpy.model.Model>` class is a lazy container for constraints and an objective function.
    Constraints and objectives are CPMpy :mod:`expressions <cpmpy.expressions>`.

    It is lazy in that it only stores the constraints and objective that are added
    to it. Processing only starts when :meth:`solve() <cpmpy.model.Model.solve>` is called, and this does not modify
    the constraints or objective stored in the model.

    A model can be solved multiple times, and constraints can be added inbetween solve calls.
    Note that constraints are added using :meth:`.add(...) <cpmpy.model.Model.add>` or using the ``+=`` operator (implemented by :meth:`__add__()`).

    See the full list of functions below.

    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:
        :toctree:

        Model
"""
import copy
import warnings
from typing import Optional

import numpy as np

from .exceptions import NotSupportedError
from .expressions.core import Expression
from .expressions.variables import NDVarArray
from .expressions.utils import is_any_list
from .solvers.utils import SolverLookup
from .solvers.solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback

import pickle

class Model(object):
    """
    CPMpy Model object, contains the constraint and objective expressions
    """

    def __init__(self, *args, minimize=None, maximize=None):
        """
            Arguments of constructor:

            Arguments:
                *args (Expression or list[Expression]): The constraints of the model
                minimize (Expression): The objective to minimize
                maximize (Expression): The objective to maximize

            At most one of minimize/maximize can be set, if none are set, it is assumed to be a satisfaction problem
        """
        assert ((minimize is None) or (maximize is None)), "can not set both minimize and maximize"
        self.cpm_status = SolverStatus("Model") # status of solving this model, will be replaced

        # init list of constraints and objective
        self.constraints = []
        self.objective_ = None
        self.objective_is_min = None

        if len(args) == 1 and is_any_list(args):
            args = args[0]  # historical shortcut, treat as *args
        # use `__add__()` for typecheck
        if is_any_list(args):
            # add (and type-check) one by one
            for a in args:
                self += a
        else:
            self += args

        # store objective if present
        if maximize is not None:
            self.maximize(maximize)
        if minimize is not None:
            self.minimize(minimize)

        
    def add(self, con):
        """
        Add one or more constraints to the model.

        Arguments:
            con (Expression or list[Expression]): Expression object(s) or list(s) of Expression objects representing constraints

        Returns:
            Model: Returns ``self`` to allow for method chaining

        Example:
            .. code-block:: python

                m = Model()
                m.add([x > 0])
        """
        if is_any_list(con):
            # catch some beginner mistakes: check that top-level Expressions in the list have Boolean return type
            for elem in con:
                if isinstance(elem, Expression) and not elem.is_bool() and not isinstance(elem, NDVarArray):
                    raise Exception(f"Model error: constraints must be expressions that return a Boolean value, `{elem}` does not.")

            if len(con) == 0:
                # ignore empty list
                return self
            elif len(con) == 1:
                # unpack size 1 list
                con = con[0]

        elif isinstance(con, Expression) and not con.is_bool():
            # catch some beginner mistakes: ensure that a top-level Expression has Boolean return type
            raise Exception(f"Model error: constraints must be expressions that return a Boolean value, `{con}` does not.")

        self.constraints.append(con)
        return self
    __add__ = add  # Make __add__() (for the += operation) be the same as add() 


    def minimize(self, expr):
        """
            Minimize the given objective function

            ``minimize()`` can be called multiple times, only the last one is stored
        """
        self.objective(expr, minimize=True)

    def maximize(self, expr):
        """
            Maximize the given objective function

            ``maximize()`` can be called multiple times, only the last one is stored
        """
        self.objective(expr, minimize=False)

    def objective(self, expr, minimize):
        """
            Users will typically use :meth:`minimize() <cpmpy.model.Model.minimize>` or :meth:`maximize() <cpmpy.model.Model.maximize>` to set the objective function,
            this is the generic implementation for both.

            Arguments:
                expr (Expression):      the CPMpy expression that represents the objective function
                minimize (bool):        whether it is a minimization problem (``True``) or maximization problem (``False``)

            ``objective()`` can be called multiple times, only the last one is stored
        """
        self.objective_ = expr
        self.objective_is_min = minimize

    def has_objective(self):
        """
            Check if the model has an objective function

            Returns:
                bool: ``True`` if the model has an objective function, ``False`` otherwise
        """
        return self.objective_ is not None

    def objective_value(self):
        """
            Returns the value of the objective function of the last solver run on this model

            Returns:
                int, optional:  The objective value as an integer or ``None`` if it is not run or is a satisfaction problem
        """
        return self.objective_.value()

    def solve(self, solver:Optional[str]=None, time_limit:Optional[int|float]=None, **kwargs):
        """ 
        Send the model to a solver and get the result.

        Run :func:`SolverLookup.solvernames() <cpmpy.solvers.utils.SolverLookup.solvernames>` to find out the valid solver names on your system. (default: None = first available solver)

        Arguments:
            solver (string or a name in SolverLookup.solvernames() or a SolverInterface class (Class, not object!), optional): 
                name of a solver to use.
            time_limit (int or float, optional): time limit in seconds

            
        Returns:
            bool: the computed output:

            - ``True``      if a solution is found (not necessarily optimal, e.g. could be after timeout)
            - ``False``     if no solution is found
        """
        if kwargs and solver is None:
            raise NotSupportedError("Specify the solver when using kwargs, since they are solver-specific!")

        if isinstance(solver, SolverInterface):
            # for advanced use, call its constructor with this model
            s = solver(self)
        else:
            s = SolverLookup.get(solver, self)

        # call solver
        ret = s.solve(time_limit=time_limit, **kwargs)
        # store CPMpy status (s object has no further use)
        self.cpm_status = s.status()
        return ret

    def solveAll(self, solver:Optional[str]=None, display:Optional[Callback]=None, time_limit:Optional[int|float]=None, solution_limit:Optional[int]=None, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            If no solution is found, the solver status will be 'Unsatisfiable'.
            If at least one solution was found and the solver exhausted all possible solutions, the solver status will be 'Optimal', otherwise 'Feasible'.

            Arguments:
                display:                            either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                                                    default/None: nothing displayed
                solution_limit (int, optional):     stop after this many solutions (default: None)

            Returns:
                int: number of solutions found (within the time and solution limit)
        """
        if kwargs and solver is None:
            raise NotSupportedError("Specify the solver when using kwargs, since they are solver-specific!")

        if isinstance(solver, SolverInterface):
            # for advanced use, call its constructor with this model
            s = solver(self)
        else:
            s = SolverLookup.get(solver, self)

        # call solver
        ret = s.solveAll(display=display,time_limit=time_limit,solution_limit=solution_limit, call_from_model=True, **kwargs)
        # store CPMpy status (s object has no further use)
        self.cpm_status = s.status()
        return ret

    def status(self):
        """
            Returns the status of the latest solver run on this model

            Status information includes exit status (optimality) and runtime.

            Returns:
                an object of :class:`SolverStatus`
        """
        return self.cpm_status

    def __repr__(self):
        """
            Returns a string representation of the model

            Returns:
                str: A string representation of the model
        """
        cons_str = ""
        for c in self.constraints:
            cons_str += "    {}\n".format(c)

        obj_str = ""
        if not self.objective_ is None:
            if self.objective_is_min:
                obj_str = "minimize "
            else:
                obj_str = "maximize "
        obj_str += str(self.objective_)
            
        return "Constraints:\n{}Objective: {}".format(cons_str, obj_str)


    def to_file(self, fname):
        """
            Serializes this model to a `.pickle` format

            Arguments:
                fname (FileDescriptorOrPath): Filename of the resulting serialized model
        """
        with open(fname,"wb") as f:
            pickle.dump(self, file=f)


    @staticmethod
    def from_file(fname):
        """
            Reads a Model instance from a binary pickled file

            Returns:
                an object of :class:`Model`
        """
        with open(fname, "rb") as f:
            m = pickle.load(f)
            # bug 158, we should increase the boolvar/intvar counters to avoid duplicate names
            from cpmpy.transformations.get_variables import get_variables_model  # avoid circular import
            from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl, _BV_PREFIX, _IV_PREFIX # avoid circular import
            vs = get_variables_model(m)
            bv_counter = 0
            iv_counter = 0
            for v in vs:
                if v.name.startswith(_BV_PREFIX):
                    try:
                        bv_counter = max(bv_counter, int(v.name[2:])+1)
                    except:
                        pass
                elif v.name.startswith(_IV_PREFIX):
                    try:
                        iv_counter = max(iv_counter, int(v.name[2:])+1)
                    except:
                        pass

            if (_BoolVarImpl.counter > 0 and bv_counter > 0) or \
                    (_IntVarImpl.counter > 0 and iv_counter > 0):
                warnings.warn(f"from_file '{fname}': contains auxiliary {_IV_PREFIX}*/{_BV_PREFIX}* variables with the same name as already created. Only add expressions created AFTER loadig this model to avoid issues with duplicate variables.")
            _BoolVarImpl.counter = max(_BoolVarImpl.counter, bv_counter)
            _IntVarImpl.counter = max(_IntVarImpl.counter, iv_counter)
            return m

    def copy(self):
        """
            Makes a shallow copy of the model.
            Constraints and variables are shared among the original and copied model (references to the same Expression objects). The /list/ of constraints itself is different, so adding or removing constraints from one model does not affect the other.
        """
        if self.objective_is_min:
            return Model(self.constraints, minimize=self.objective_)
        else:
            return Model(self.constraints, maximize=self.objective_)



    # keep for backwards compatibility
    def deepcopy(self, memodict={}):
        warnings.warn("Deprecated, use copy.deepcopy() instead, will be removed in stable version", DeprecationWarning)
        return copy.deepcopy(self, memodict)
