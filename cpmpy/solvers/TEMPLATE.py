#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## TEMPLATE.py
##
"""
    Interface to TEMPLATE's API

    .. note::
        [GUIDELINE] Replace <TEMPLATE> by the solver's name, and implement the missing pieces
        The functions are ordered in a way that could be convenient to 
        start from the top and continue in that order.

    .. note::
        After you are done filling in the template, remove all comments starting with [GUIDELINE]

    .. warning::
        [GUIDELINE] do not include the python package at the top of the file,
        as CPMpy should also work without this solver installed.
        To ensure that, include it inside supported() and other functions that need it...

    <some information on the solver>

    Always use :func:`cp.SolverLookup.get("TEMPLATE") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'TEMPLATEpy' python package is installed:

    .. code-block:: console
    
        $ pip install TEMPLATEpy

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_template
"""

from typing import Optional
import warnings
import pkg_resources
from pkg_resources import VersionConflict

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_boolexpr
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies

class CPM_template(SolverInterface):
    """
    Interface to TEMPLATE's API

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, TEMPLATE's model object

    Documentation of the solver's own Python API:
    <URL to docs or source code>
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import TEMPLATEpy as gp
            # optionally enforce a specific version
            pkg_resources.require("TEMPLATEpy>=2.1.0")
            return True
        except ModuleNotFoundError: # if solver's Python package is not installed
            return False
        except VersionConflict: # unsupported version of TEMPLATEpy (optional)
            warnings.warn(f"CPMpy uses features only available from TEMPLATEpy version 0.2.1, "
                          f"but you have version {pkg_resources.get_distribution('TEMPLATEpy').version}.")
            return False
        except Exception as e:
            raise e

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        try:
            return pkg_resources.get_distribution('TEMPLATEpy').version
        except pkg_resources.DistributionNotFound:
            return None
        
    # [GUIDELINE] If your solver supports different subsolvers, implement below method to return a list of subsolver names
    @staticmethod
    def solvernames(installed:bool=True):
        """
            Returns solvers supported by TEMPLATE (on your system).

            Arguments:
                installed (boolean): whether to filter the solvernames to those installed on your system (default True)
               
            Returns:
                list of solver names
        """
        if CPM_template.supported():
            # Collect solver names
            if installed:
                return # [ ... list of the installed subsolver names ... ]
            else:
                return # [ ... list of all subsolver names ... ]
        else:
            warnings.warn("TEMPLATE is not installed or not supported on this system.")
            return []

    # [GUIDELINE] If your solver supports different subsolvers, implement below method to return their respective versions
    @classmethod
    def solverversion(cls, subsolver:str) -> Optional[str]:
        """
        Returns the version of the requested subsolver.

        Arguments:
            subsolver (str): name of the subsolver

        Returns:
            Version number of the subsolver if installed, else None 
        """
        # return version of requested subsolver (if installed)
        # if requested subsolver does not exist, raise ValueError
        pass

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_TEMPLATE: Install the python package 'TEMPLATEpy' to use this solver interface.")

        import TEMPLATEpy

        assert subsolver is None # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        # [GUIDELINE] we commonly use 3-letter abbrivations to refer to native objects:
        #           OR-tools uses ort_solver, Gurobi grb_solver, Exact xct_solver...
        self.TPL_solver = TEMPLATEpy.Solver("cpmpy") 
        self.TPL_model = TEMPLATEpy.model("cpmpy")

        # initialise everything else and post the constraints/objective
        # [GUIDELINE] this superclass call should happen AFTER all solver-native objects are created.
        #           internally, the constructor relies on `add()` which uses the above solver native object(s)
        super().__init__(name="TEMPLATE", cpm_model=cpm_model)


    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.TPL_model

    def solve(self, time_limit=None, **kwargs):
        """
            Call the TEMPLATE solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        if time_limit is not None:
            self.TPL_solver.set_timelimit_seconds(time_limit)

        # [GUIDELINE] if your solver supports solving under assumptions, add `assumptions` as argument in header
        #       e.g., def solve(self, time_limit=None, assumptions=None, **kwargs):
        #       then translate assumptions here; assumptions are a list of Boolean variables or NegBoolViews

        # call the solver, with parameters
        my_status = self.TPL_solver.solve(**kwargs)
        # [GUIDELINE] consider saving the status as self.TPL_status so that advanced CPMpy users can access the status object.
        #       This is mainly useful when more elaborate information about the solve-call is saved into the status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.TPL_solver.time() # wallclock time in (float) seconds

        # Translate solver exit status to CPMpy exit status
        # CSP:                         COP:
        # ├─ sat -> FEASIBLE           ├─ optimal -> OPTIMAL
        # ├─ unsat -> UNSATISFIABLE    ├─ sub-optimal -> FEASIBLE
        # └─ timeout -> UNKNOWN        ├─ unsat -> UNSATISFIABLE
        #                              └─ timeout -> UNKNOWN
        if my_status is True:
            # COP
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = self.TPL_solver.value(sol_var)
                raise NotImplementedError("TEMPLATE: back-translating the solution values")

            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.TPL_solver.ObjectiveValue()

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # [GUIDELINE] some solver interfaces explicitely create variables on a solver object
        #       then use self.TPL_solver.NewBoolVar(...) instead of TEMPLATEpy.NewBoolVar(...)

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return TEMPLATEpy.negate(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = TEMPLATEpy.NewBoolVar(str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = TEMPLATEpy.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    # [GUIDELINE] if TEMPLATE does not support objective functions, you can delete this function definition
    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective

            are permanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr, csemap=self._csemap)
        self += flat_cons # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj)) # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        # [GUIDELINE] if the solver interface does not provide a solver native "numeric expression" object,
        #         _make_numexpr may be removed and an objective can be posted as:
        #           self.TPL_solver.MinimizeWeightedSum(obj.args[0], self.solver_vars(obj.args[1]) or similar

        if minimize:
            self.TPL_solver.Minimize(obj)
        else:
            self.TPL_solver.Maximize(obj)

    def has_objective(self):
        return self.TPL_solver.hasObjective()

    def _make_numexpr(self, cpm_expr):
        """
            Converts a numeric CPMpy 'flat' expression into a solver-specific numeric expression

            Primarily used for setting objective functions, and optionally in constraint posting
        """

        # [GUIDELINE] not all solver interfaces have a native "numerical expression" object.
        #       in that case, this function may be removed and a case-by-case analysis of the numerical expression
        #           used in the constraint at hand is required in `add()`
        #       For an example of such solver interface, check out solvers/choco.py or solvers/exact.py

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # any solver-native numerical expression
        if isinstance(cpm_expr, Operator):
           if cpm_expr.name == 'sum':
               return self.TPL_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.TPL_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.TPL_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("TEMPLATE: Not a known supported numexpr {}".format(cpm_expr))


    # `add()` first calls `transform()`
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations
        # XXX chose the transformations your solver needs, see cpmpy/transformations/
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported={"alldifferent"})
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        # ...
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

            if isinstance(cpm_expr, _BoolVarImpl):
                # base case, just var or ~var
                self.TPL_solver.add_clause([ self.solver_var(cpm_expr) ])

            elif isinstance(cpm_expr, Operator):
                if cpm_expr.name == "or":
                    self.TPL_solver.add_clause(self.solver_vars(cpm_expr.args))
                elif cpm_expr.name == "->": # half-reification
                    bv, subexpr = cpm_expr.args
                    # [GUIDELINE] example code for a half-reified sum/wsum comparison e.g. BV -> sum(IVs) >= 5
                    if isinstance(subexpr, Comparison):
                        lhs, rhs = subexpr.args
                        if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                            TPL_lhs = self._make_numexpr(lhs)
                            self.TPL_solver.add_half_reified_comparison(self.solver_var(bv),
                                                                        TPL_lhs, subexpr.name, self.solver_var(rhs))
                        else:
                            raise NotImplementedError("TEMPLATE: no support for half-reified comparison:", subexpr)
                    else:
                        raise NotImplementedError("TEMPLATE: no support for half-reified constraint:", subexpr)

            elif isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args

                # [GUIDELINE] == is used for both double reification and numerical comparisons
                #       need case by case analysis here. Note that if your solver does not support full-reification,
                #       you can rely on the transformation only_implies to convert all reifications to half-reification
                #       for more information, please reach out on github!
                if cpm_expr.name == "==" and is_boolexpr(lhs) and is_boolexpr(rhs): # reification
                    bv, subexpr = lhs, rhs
                    assert isinstance(lhs, _BoolVarImpl), "lhs of reification should be var because of only_bv_reifies"

                    if isinstance(subexpr, Comparison):
                        lhs, rhs = subexpr.args
                        if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                            TPL_lhs = self._make_numexpr(lhs)
                            self.TPL_solver.add_reified_comparison(self.solver_var(bv),
                                                                   TPL_lhs, subexpr.name, self.solver_var(rhs))
                        else:
                            raise NotImplementedError("TEMPLATE: no support for reified comparison:", subexpr)
                    else:
                        raise NotImplementedError("TEMPLATE: no support for reified constraint:", subexpr)

                # otherwise, numerical comparisons
                if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                    TPL_lhs = self._make_numexpr(lhs)
                    self.TPL_solver.add_comparison(TPL_lhs, cpm_expr.name, self.solver_var(rhs))
                # global functions
                elif cpm_expr.name == "==":
                    TPL_rhs = self.solver_var(rhs)
                    if lhs.name == "max":
                        self.TPL_solver.add_max_constraint(self.solver_vars(lhs), TPL_rhs)
                    elif lhs.name == "element":
                        TPL_arr, TPL_idx = self.solver_vars(lhs.args)
                        self.TPL_solver.add_element_constraint(TPL_arr, TPL_idx, TPL_rhs)
                    # elif...
                    else:
                        raise NotImplementedError("TEMPLATE: unknown equality constraint:", cpm_expr)
                else:
                    raise NotImplementedError("TEMPLATE: unknown comparison constraint", cpm_expr)

            # global constraints
            elif cpm_expr.name == "alldifferent":
                self.TPL_solver.add_alldifferent(self.solver_vars(cpm_expr.args))
            else:
                raise NotImplementedError("TEMPLATE: constraint not (yet) supported", cpm_expr)

        return self
    __add__ = add  # avoid redirect in superclass

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            If the problem is an optimization problem, returns only optimal solutions.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

        # check if objective function (optional if solver doesn't support finding all solutions for COP)
        if self.has_objective():
            raise NotSupportedError("TEMPLATE does not support finding all optimal solutions")

        # A. Example code if solver supports callbacks
        if is_any_list(display):
            callback = lambda : print([var.value() for var in display])
        else:
            callback = display

        my_status = self.solve(time_limit, callback=callback, enumerate_all_solutions=True, **kwargs)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.TPL_solver.time() # wallclock time in (float) seconds (of entire solveAll call)

        # Translate solver exit status to CPMpy exit status
        # CSP & COP:
        # ├─ all solutions found -> OPTIMAL
        # ├─ at least 1 found (timeout / solution limit reached) -> FEASIBLE
        # ├─ 0 solutions found due to timeout -> UNKNOWN
        # └─ unsat -> UNSATISFIABLE

        # self.cpm_status.exitstatus = ...

        # clear user vars if no solution found
        if self.TPL_solver.SolutionCount() == 0:
            for var in self.user_vars:
                var.clear()
        return self.TPL_solver.SolutionCount()

        # B. Example code if solver does not support callbacks
        self.solve(time_limit, enumerate_all_solutions=True, **kwargs)
        solution_count = 0
        for solution in self.TPL_solver.GetAllSolutions():
            solution_count += 1
            # Translate solution to variables
            for cpm_var in self.user_vars:
                cpm_var._value = solution.value(solver_var)

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

        # clear user vars if no solution found
        if solution_count == 0:
            for var in self.user_vars:
                var.clear()

        return solution_count
