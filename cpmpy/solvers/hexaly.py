#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## hexaly.py
##
"""
    Interface to Hexaly's API


    Hexaly is a local search solver with support for  global constraints.

    Always use :func:`cp.SolverLookup.get("hexaly") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'hexaly' python package is installed:

    .. code-block:: console

        $ pip install hexaly -i https://pip.hexaly.com                
    
    The Hexaly local solver requires an active licence (for example a free academic license)
    You can read more about available licences at https://www.hexaly.com/

    See detailed installation instructions at:
    https://www.hexaly.com/docs/last/installation/pythonsetup.html

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_hexaly
"""

from typing import Optional

from importlib.metadata import version, PackageNotFoundError

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, GlobalFunction, DirectConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, eval_comparison, flatlist
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree

class CPM_hexaly(SolverInterface):
    """
    Interface to Hexaly's API

    Creates the following attributes (see parent constructor for more):

    - hex_model: object, Hexaly's model object
    - hex_solver: object, Hexaly's solver object (to solve hex_model)

    Documentation of the solver's own Python API:
    https://www.hexaly.com/docs/last/pythonapi/index.html
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import hexaly as hex
            return True
        except ModuleNotFoundError: # if solver's Python package is not installed
            return False
        except Exception as e:
            raise e

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        try:
            return version('hexaly')
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_hexaly: Install the python package 'hexaly' to use this solver interface.")

        from hexaly.optimizer import HexalyOptimizer

        assert subsolver is None # hexaly does not have subsolvers

        # initialise the native solver object
        self.hex_solver = HexalyOptimizer()
        self.hex_solver.param.verbosity = 0
        self.hex_model = self.hex_solver.model
        self.is_satisfaction = True

        # initialise everything else and post the constraints/objective
        super().__init__(name="hexaly", cpm_model=cpm_model)

    @property
    def native_model(self):
        return self.hex_model

    def solve(self, time_limit=None, **kwargs):
        """
            Call the Hexaly solver

            Arguments:
                time_limit:  maximum solve time in seconds (float, optional)
                kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:

            - nb_threads: number of threads used to parallelize the search.
            - iteration_limit: max number of iterations
            - verbosity: verbosity level

            full list of parameters availble at:
            https://www.hexaly.com/docs/last/pythonapi/optimizer/hxparam.html
        """
        from hexaly.optimizer import HxObjectiveDirection

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        if time_limit is not None:
            if time_limit < 0:
                raise ValueError(f"Time limit must be positive but was {time_limit}")
            self.hex_solver.param.time_limit = int(time_limit)  # hexaly does not support float time limit

        # set solver parameters
        for arg, val in kwargs.items():
            setattr(self.hex_solver, arg, val)

        if self.is_satisfaction: # set dummy objective for satisfaction problems
            self.hex_model.add_objective(0, HxObjectiveDirection.MINIMIZE)

        # new status, translate runtime
        self.hex_model.close() # model must be closed
        self.hex_solver.solve()
        self.hex_sol = self.hex_solver.get_solution()
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.hex_solver.statistics.running_time # wallclock time in (float) seconds

        # Translate solver exit status to CPMpy exit status
        # CSP:                         COP:
        # ├─ sat -> FEASIBLE           ├─ optimal -> OPTIMAL
        # ├─ unsat -> UNSATISFIABLE    ├─ sub-optimal -> FEASIBLE
        # └─ timeout -> UNKNOWN        ├─ unsat -> UNSATISFIABLE
        #                              └─ timeout -> UNKNOWN

        from hexaly.optimizer import HxSolutionStatus
        if self.hex_sol.status == HxSolutionStatus.INCONSISTENT:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.hex_sol.status == HxSolutionStatus.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif self.hex_sol.status == HxSolutionStatus.FEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif self.hex_sol.status == HxSolutionStatus.OPTIMAL:
            if self.is_satisfaction:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        else:  # another?
            raise NotImplementedError(self.hex_sol.status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if cpm_var.is_bool():
                    cpm_var._value = bool(self.hex_sol.get_value(sol_var))
                else:
                    cpm_var._value = int(self.hex_sol.get_value(sol_var))

            # translate objective, for optimisation problems only
            if not self.is_satisfaction:
                self.objective_value_ = self.hex_sol.get_objective_bound(0)

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        # now open model again, we might want to add new constraints after
        self.hex_model.open()

        if self.is_satisfaction:
            self.hex_model.remove_objective(0) # reset to not have any objectives

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return ~self.solver_var(cpm_var._bv)

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.hex_model.bool()
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.hex_model.int(cpm_var.lb, cpm_var.ub)
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            # set name of variable
            revar.set_name(str(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective

            are permanently posted to the solver)
        """
        from hexaly.optimizer import HxObjectiveDirection
        # make objective function or variable and post
        while self.has_objective(): # remove prev objective(s)
            self.hex_model.remove_objective(0)
        self.is_satisfaction = False
        hex_obj = self._hex_expr(expr)
        if minimize:
            self.hex_model.add_objective(hex_obj,HxObjectiveDirection.MINIMIZE)
        else:
            self.hex_model.add_objective(hex_obj,HxObjectiveDirection.MAXIMIZE)

    def has_objective(self):
        return self.hex_model.nb_objectives > 0

    # `add()` first calls `transform()`
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations
        cpm_cons = toplevel_list(cpm_expr)
        # no flattening, so also no safening required
        cpm_cons = decompose_in_tree(cpm_cons, supported={"min", "max", "abs", "element"})
        return cpm_cons

    def add(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """
        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):
            hex_expr = self._hex_expr(cpm_expr)
            self.hex_model.add_constraint(hex_expr)

        return self
    __add__ = add  # avoid redirect in superclass

    def _hex_expr(self, cpm_expr):

        # get transformed constraint

        if is_any_list(cpm_expr):
            return [self._hex_expr(expr) for expr in cpm_expr]

        # constants
        if isinstance(cpm_expr, BoolVal):
            return bool(cpm_expr)
        if is_num(cpm_expr):
            return cpm_expr
        # variables
        if isinstance(cpm_expr, _NumVarImpl):
            return self.solver_var(cpm_expr)

        if isinstance(cpm_expr, Operator):
            if cpm_expr.name == "and":
                return self.hex_model.and_(self._hex_expr(cpm_expr.args))
            if cpm_expr.name == "or":
                return self.hex_model.or_(self._hex_expr(cpm_expr.args))
            if cpm_expr.name == "not":
                return ~self._hex_expr(cpm_expr.args[0])
            if cpm_expr.name == "->":
                cond, subexpr = cpm_expr.args
                return self._hex_expr(~cond | subexpr) # post as disjunction
            if cpm_expr.name == "sum":
                return self.hex_model.sum(self._hex_expr(cpm_expr.args))
            if cpm_expr.name == "wsum":
                weights, args = cpm_expr.args
                return self.hex_model.sum([w * a for w,a in zip(weights, self._hex_expr(args))])
            if cpm_expr.name == "sub":
                a,b = self._hex_expr(cpm_expr.args)
                return a - b
            if cpm_expr.name == "-":
                return -self._hex_expr(cpm_expr.args[0])
            if cpm_expr.name == "mul":
                a,b = self._hex_expr(cpm_expr.args)
                return a * b
            if cpm_expr.name == "div":
                a, b = self._hex_expr(cpm_expr.args)
                # ensure we are rounding towards zero
                return self.hex_model.iif((a >= 0) & (b >= 0), self.hex_model.floor(a / b), # result is positive
                       self.hex_model.iif((a <= 0) & (b <= 0), self.hex_model.floor(a / b), # result is positive
                       self.hex_model.iif((a >= 0) & (b <= 0), self.hex_model.ceil(a / b), # result is negative
                       self.hex_model.iif((a <= 0) & (b >= 0), self.hex_model.ceil(a / b), 0)))) # result is negative

            if cpm_expr.name == "mod":
                a, b = self._hex_expr(cpm_expr.args)
                return a % b
            if cpm_expr.name == "pow":
                a, b = self._hex_expr(cpm_expr.args)
                return a ** b
            raise ValueError(f"Unknown operator {cpm_expr}")

        elif isinstance(cpm_expr, Comparison):
            x,y = self._hex_expr(cpm_expr.args)
            return eval_comparison(cpm_expr.name, x,y)

        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name == "alldifferent":
                hex_arr = self.hex_model.array(self._hex_expr(cpm_expr.args))
                return self.hex_model.distinct(hex_arr)
            raise ValueError(f"Global constraint {cpm_expr} is not supported by hexaly")

        elif isinstance(cpm_expr, GlobalFunction):
            if cpm_expr.name == "nvalues":
                return self.hex_model.distinct(self._hex_expr(cpm_expr.args))
            if cpm_expr.name == "element":
                hex_arr = self.hex_model.array(self._hex_expr(cpm_expr.args[0]))
                idx = self._hex_expr(cpm_expr.args[1])
                return self.hex_model.at(hex_arr,idx)
            if cpm_expr.name == "abs":
                return self.hex_model.abs(self._hex_expr(cpm_expr.args[0]))
            if cpm_expr.name == "min":
                return self.hex_model.min(*self._hex_expr(cpm_expr.args))
            if cpm_expr.name == "max":
                return self.hex_model.max(*self._hex_expr(cpm_expr.args))
            raise ValueError(f"Global function {cpm_expr} is not supported by hexaly")

        elif isinstance(cpm_expr, DirectConstraint):
            return cpm_expr.callSolver(self, self.hex_model)

        raise NotImplementedError(f"Unexpected expression {cpm_expr}")

    
    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all solutions, map them to CPMpy and optionally display the solutions.

            Arguments:
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                solution_limit: stop after this many solutions (default: None)
                time_limit (float):   maximum solve time in seconds

            Returns: 
                number of solutions found

            .. note::
                Hexaly does not support exhaustive search to find all solutions. Set `time_limit` to do a limited search.
        """
        if time_limit is None:
            raise ValueError("Hexaly does not support exhaustive search to find all solutions. "
                             "Set time limit to do a limited search")

        return super(CPM_hexaly, self).solveAll(display, time_limit, solution_limit, call_from_model, **kwargs)


    def solution_hint(self, cpm_vars, vals):
        from hexaly.optimizer import HxObjectiveDirection
        if self.is_satisfaction: # set dummy objective, otherwise cannot close model
            self.hex_model.add_objective(0, HxObjectiveDirection.MINIMIZE)

        cpm_vars = flatlist(cpm_vars)
        vals = flatlist(vals)

        self.hex_model.close() # must be closed before we can set a solution hint
        for hex_var, val in zip(self.solver_vars(cpm_vars), vals):
            hex_var.value = val
        self.hex_model.open() # re-open

    def __del__(self):
        # release lock on licence file
        self.hex_solver.delete()

