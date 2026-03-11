#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## exact.py
##
"""
    Interface to Exact's Python API

    Exact solves decision and optimization problems formulated as integer linear programs. 
    Under the hood, it converts integer variables to binary (0-1) variables and applies highly efficient 
    propagation routines and strong cutting-planes / pseudo-Boolean conflict analysis.

    The solver's git repository:
    https://gitlab.com/nonfiction-software/exact

    Always use :func:`cp.SolverLookup.get("exact") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'exact' python package is installed:

    .. code-block:: console
    
        $ pip install exact

    .. warning::
        Exact requires Python 3.10 or higher and the pip install only works on Linux and Windows.
        On MacOS, you have to install the package from source.

    See https://pypi.org/project/exact for more information.

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_exact

    ==============
    Module details
    ==============
"""
import sys  # for stdout checking
import time
from typing import Optional, List

from packaging.version import Version

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..expressions.core import *
from ..expressions.globalfunctions import Multiplication
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv, only_positive_bv_wsum, decompose_linear, decompose_linear_objective, linearize_constraint, linearize_reified_variables
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.normalize import toplevel_list
from ..transformations.safening import no_partial_functions, safen_objective
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import flatlist, argvals, argval
from ..exceptions import NotSupportedError

import numpy as np
import numbers

class CPM_exact(SolverInterface):
    """
    Interface to Exact's Python API

    Creates the following attributes (see parent constructor for more):

    - ``xct_solver`` : the Exact instance used in solve() and solveAll()
    - ``assumption_dict`` : maps Exact variables to (Exact value, CPM assumption expression)

    Documentation of the solver's own Python API is sparse, but example usage can be found at:
    https://gitlab.com/nonfiction-software/exact/-/tree/main/python_examples
    """

    supported_global_constraints = frozenset({"mul"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            # check if exact is installed
            import exact
            xct_version = CPM_exact.version()
            if Version(xct_version) < Version("2.1.0"):
                warnings.warn(f"CPMpy requires Exact version >=2.1.0 is required but you have version "
                              f"{xct_version}, beware exact>=2.1.0 requires Python 3.10 or higher.")
                return False
            return True
        except ModuleNotFoundError: # exact is not installed
            return False
        except Exception as e:
            raise e
        
    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('exact')
        except PackageNotFoundError:
            return None


    def __init__(self, cpm_model=None, subsolver=None, **kwargs):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        Exact solver object xct_solver.

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None

        Exact takes options at initialization instead of solving.
        The Exact solver parameters are defined by https://gitlab.com/nonfiction-software/exact/-/blob/main/src/Options.hpp
        Note some parameters use "-" which cannot be used in a keyword argument.
        A workaround is to use dict-unpacking: `CPM_Exact(**{parameter-with-hyphen: 42})`
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_exact: Install the python package 'cpmpy[exact]' to use this solver interface.")
        
        assert subsolver is None, "Exact does not allow subsolvers."

        from exact import Exact as xct
        # initialise the native solver object
        options = list(kwargs.items()) # options is a list of string-pairs, e.g. [("verbosity","1")]
        options = [(opt[0], str(opt[1])) for opt in options] # Ensure values are also strings
        self.xct_solver = xct(options)

        # can override encoding of variables
        self.encoding = None

        # for solving with assumption variables,
        self.assumption_dict = None
        self.objective_ = None
        self.objective_is_min_ = True

        # initialise everything else and post the constraints/objective
        super().__init__(name="exact", cpm_model=cpm_model)
        
    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.xct_solver

    def _fillVars(self):
        if not self.xct_solver.hasSolution():
            self.objective_value_ = None
            for cpm_var in self.user_vars:
                cpm_var._value = None
            return

        # fill in variable values
        lst_vars = list(self.user_vars)
        exact_vals = self.xct_solver.getLastSolutionFor(self.solver_vars(lst_vars))
        for cpm_var, val in zip(lst_vars,exact_vals):
            cpm_var._value = bool(val) if isinstance(cpm_var, _BoolVarImpl) else val # xct value is always an int

    def solve(self, time_limit:Optional[float]=None, assumptions:Optional[List[_BoolVarImpl]]=None, **kwargs):
        """
            Call Exact

            Overwrites ``self.cpm_status``

            :param assumptions: CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with :func:`s.get_core() <get_core()>`: if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.
            :type assumptions: list of CPMpy Boolean variables

            :param time_limit: optional, time limit in seconds
            :type time_limit: int or float

            :return: Bool:
                - True      if a solution is found (not necessarily optimal, e.g. could be after timeout)
                - False     if no solution is found
        """
        from exact import Exact as xct
 
        # set additional keyword arguments
        if(len(kwargs.items())>0):
            wrn_txt = f"Exact only supports options at initialization. Ignoring additional options {kwargs.items()}\n"
            wrn_txt += "Use cp.SolverLookup.lookup('exact', **{parameter-with-hyphen: 42}) to set Exact parameters"
            warnings.warn(wrn_txt)

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        self.xct_solver.clearAssumptions()

        # set assumptions
        if assumptions is not None:
            assert all(v.is_bool() for v in assumptions), "Non-Boolean assumptions given to Exact: " + str([v for v in assumptions if not v.is_bool()])
            assump_vals = [int(not isinstance(v, NegBoolView)) for v in assumptions]
            assump_vars = [self.solver_var(v._bv if isinstance(v, NegBoolView) else v) for v in assumptions]
            self.assumption_dict = {xct_var: (xct_val,cpm_assump) for (xct_var, xct_val, cpm_assump) in zip(assump_vars,assump_vals,assumptions)}
            self.xct_solver.setAssumptions(list(zip(assump_vars,assump_vals)))

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            timeout = time_limit
        else:
            timeout = 0
            
        # call the solver, with parameters
        start = time.time()
        my_status, obj_val = self.xct_solver.toOptimum(timeout=timeout)
        end = time.time()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        self.objective_value_ = None
        # translate exit status
        #   see 'toOptimum' documentation:
        #   https://gitlab.com/nonfiction-software/exact/-/blob/main/src/interface/IntProg.cpp#L877
        if my_status == "UNSAT": # found unsatisfiability
            if self.has_objective() and self.xct_solver.hasSolution(): # optimisation problem -> unsat = no better solution found
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == "SAT": # the optimal value has been found
            assert self.xct_solver.hasSolution()
            # COP
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == "INCONSISTENT": # found inconsistency over assumptions
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == "TIMEOUT": # found timeout
            if self.has_objective() and self.xct_solver.hasSolution(): # found a (sub-)optimal solution
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else: # no solution found
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github
        
        self._fillVars()
        if self.has_objective():
            if self.objective_is_min_:
                self.objective_value_ = obj_val
            else: # maximize, so actually negative value
                self.objective_value_ = -obj_val
        
        # True/False depending on self.cpm_status
        return self._solve_return(self.cpm_status)

    def _update_time(self, timelim, start, end):
        """
            Internal helper function to keep track of remaining time.
        """
        if timelim != 0:
            timelim -= (end - start)
            if timelim == 0: timelim = -1
        return timelim

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally, display the solutions.

            Arguments:
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                time_limit: stop after this many seconds (default: None)
                solution_limit: stop after this many solutions (default: None)
                call_from_model: whether the method is called from a CPMpy Model instance or not
                any other keyword argument

            Returns: number of solutions found
        """
        # set additional keyword arguments
        if len(kwargs.items()) > 0:
            warnings.warn(f"Exact only supports options at initialization: {kwargs.items()}")

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        self.xct_solver.clearAssumptions()

        timelim = time_limit if time_limit is not None else 0

        total_start = time.time()

        if self.has_objective():
            start = time.time()
            if not call_from_model:
                warnings.warn("Adding constraints to solver object to find all solutions, solver state will be invalid after this call!")

            (my_status, objval) = self.xct_solver.toOptimum(timelim) # fix the solution to the optimal objective
            if my_status == "UNSAT": # found unsatisfiability
                total_end = time.time()
                self._fillVars() # erases the solution
                # update exit status
                self.cpm_status = SolverStatus(self.name)
                self.cpm_status.runtime = total_end - total_start
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
                # early exit
                return 0
            elif my_status == "INCONSISTENT": # found inconsistency
                raise ValueError("Error: inconsistency during solveAll should not happen, please warn the developers of this bug")
            elif my_status == "TIMEOUT": # found timeout
                total_end = time.time()
                # update exit status
                self.cpm_status = SolverStatus(self.name)
                self.cpm_status.runtime = total_end - total_start
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
                # early exit
                return 0
            else:
                assert my_status == "SAT", "Unexpected status from Exact"
            self += self.objective_ == objval # fix obj val
            end = time.time()
            timelim = self._update_time(timelim, start, end) # update remaining time


        solsfound = 0
        while timelim >= 0 and (solution_limit is None or solsfound < solution_limit):
            # call the solver, with parameters
            start = time.time()
            my_status = self.xct_solver.runFull(optimize=False, timeout=timelim)
            end = time.time()
            timelim = self._update_time(timelim, start, end) # update remaining time

            assert my_status in ["UNSAT","SAT","INCONSISTENT","TIMEOUT"], "Unexpected status code for Exact: " + my_status
            if my_status == "UNSAT": # found unsatisfiability (no more solutions to be found)
                self._fillVars() # erases the solution
                break
            elif my_status == "SAT": # found solution, but not optimality proven
                assert self.xct_solver.hasSolution()
                solsfound += 1
                self.xct_solver.invalidateLastSol() # TODO: pass user vars to this function
                if display is not None:
                    self._fillVars()
                    if isinstance(display, Expression):
                        print(argval(display))
                    elif isinstance(display, list):
                        print(argvals(display))
                    else:
                        display()  # callback
            elif my_status == "INCONSISTENT": # found inconsistency
                raise ValueError("Error: inconsistency during solveAll should not happen, please warn the developers of this bug")
            elif my_status == "TIMEOUT": # found timeout
                break
        total_end = time.time()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = total_end - total_start

        if solsfound: # found some solutions
            if solsfound == solution_limit: # matched solution limit
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif my_status == "UNSAT": # found all solutions (before limits were reached)
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else: # timeout
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            if my_status == "UNSAT": # unsat problem
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            elif my_status == "TIMEOUT": # timeout
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                raise NotImplementedError(my_status) # please report on GitHub

        return solsfound

    def has_objective(self):
        """
            Returns whether the solver has an objective function or not.
        """
        return self.objective_ is not None


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view. Should be eliminated in linearize
        if isinstance(cpm_var, NegBoolView):
            raise NotSupportedError("Negative literals should not be left as part of any equation. Please report.")

        # return it if it already exists
        if cpm_var in self._varmap:
            return self._varmap[cpm_var]

        # create if it does not exist
        revar = str(cpm_var)
        if isinstance(cpm_var, _BoolVarImpl):
            self.xct_solver.addVariable(revar)
        elif isinstance(cpm_var, _IntVarImpl):
            lb, ub = cpm_var.get_bounds()
            if self.encoding is None:
                encoding = "order" if ub-lb < 8 else "log" # heuristic bound
            else:
                encoding = self.encoding # can also force it
            self.xct_solver.addVariable(revar, lb, ub, encoding)
        else:
            raise NotImplementedError("Not a known var {}".format(cpm_var))
        self._varmap[cpm_var] = revar
        return revar


    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize.

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

        """
        self.objective_ = expr
        self.objective_is_min_ = minimize

        # save user variables
        get_variables(expr, self.user_vars)

        # transform objective
        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_linear_objective(obj,
                                                      supported=self.supported_global_constraints,
                                                      supported_reified=self.supported_reified_global_constraints,
                                                      csemap=self._csemap)
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        obj = only_positive_bv_wsum(obj)  # remove negboolviews

        self.add(safe_cons + decomp_cons + flat_cons)

        # make objective function or variable and post
        xct_cfvars,xct_rhs = self._make_numexpr(obj,0)

        # TODO: make this a custom transformation?
        newcfvrs = []
        for c,v in xct_cfvars:
            if is_num(v):
                xct_rhs += c*v
            else:
                newcfvrs.append((c,v))

        self.xct_solver.setObjective(newcfvrs,minimize,xct_rhs)

    @staticmethod
    def fix(o):
        x = o.item() if isinstance(o, np.generic) else o
        if not isinstance(x, numbers.Integral):
            raise NotImplementedError("Exact requires all values to be integral")
        return x

    def _make_numexpr(self, lhs, rhs):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression
        """

        xcfvars = []
        xrhs = 0
        
        assert is_num(rhs), "RHS of inequality should be numeric after transformations: {}".format(rhs)
        xrhs += rhs

        if is_num(lhs):
            xrhs -= lhs
        elif isinstance(lhs, _NumVarImpl):
            xcfvars = [(1,self.solver_var(lhs))]
        elif lhs.name == "sum":
            xcfvars = [(1,x) for x in self.solver_vars(lhs.args)]
        elif lhs.name == "wsum":
            xcfvars = list(zip([self.fix(c) for c in lhs.args[0]],self.solver_vars(lhs.args[1])))
        else:
            raise NotImplementedError("Exact: Unexpected lhs {} for expression {}".format(lhs.name,lhs))

        return xcfvars, self.fix(xrhs)

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

        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"}) # linearize and decompose expects safe exprs
        cpm_cons = decompose_linear(cpm_cons,
                                    supported=self.supported_global_constraints,
                                    supported_reified = self.supported_reified_global_constraints,
                                    csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']), csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)  # supports >, <, !=
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum","wsum","->","mul"}), csemap=self._csemap)  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearisation, rewrite ~bv into 1-bv

        return cpm_cons

        # NOTE: the transformations that are still done specifically for Exact are two-fold:
        # transform '==' and '<=' to '>='
        #
        # this seems quite general and is a candidate to function as an independent transformation.

    def _add_xct_constr(self, xct_cfvars, uselb, lb, useub, ub):
        self.xct_solver.addConstraint(xct_cfvars, uselb, self.fix(lb), useub, self.fix(ub))

    def _add_xct_reif_right(self, head, sign, xct_cfvars, xct_rhs):
        self.xct_solver.addRightReification(head, sign, xct_cfvars, self.fix(xct_rhs))

    @staticmethod
    def is_multiplication(cpm_expr):  # helper function (Multiplication is GlobalFunction name 'mul')
        return isinstance(cpm_expr, Multiplication)

    def add(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr_orig: CPMpy expression, or list thereof
        :type cpm_expr_orig: Expression or list of Expression

        :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):
            # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
            # numexpr `comp` bvar|const
            if isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                if cpm_expr.name == "==":
                    # lhs can be Operator (sum, wsum) or Multiplication (GlobalFunction name 'mul')
                    if lhs.name == "mul":
                        if is_num(rhs): # make dummy var
                            rhs = cp.intvar(rhs, rhs)
                        xct_rhs = self.solver_var(rhs)
                        assert all(isinstance(v, _IntVarImpl) for v in lhs.args), "constant * var should be " \
                                                                                  "rewritten by linearize"
                        self.xct_solver.addMultiplication(self.solver_vars(lhs.args), True, xct_rhs, True, xct_rhs)

                    else:
                        assert isinstance(lhs, Operator) and (lhs.name == "sum" or lhs.name == "wsum")
                        xct_cfvars, xct_rhs = self._make_numexpr(lhs, rhs)
                        self._add_xct_constr(xct_cfvars, True, xct_rhs, True, xct_rhs)

                elif cpm_expr.name == "<=":
                    xct_cfvars, xct_rhs = self._make_numexpr(lhs, rhs)
                    self._add_xct_constr(xct_cfvars, False, 0, True, xct_rhs)

                elif cpm_expr.name == ">=":
                    xct_cfvars, xct_rhs = self._make_numexpr(lhs, rhs)
                    self._add_xct_constr(xct_cfvars, True, xct_rhs, False, 0)

                else:
                    raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

            elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
                # Indicator constraints
                # Take form bvar -> sum(x,y,z) >= rvar
                cond, sub_expr = cpm_expr.args
                assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
                assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

                if sub_expr.name not in ["==", ">=", "<="]:
                    raise NotImplementedError("Constraint not supported by Exact '{}' {}".format(lhs.name, cpm_expr))

                lhs, rhs = sub_expr.args
                xct_cfvars, xct_rhs = self._make_numexpr(lhs,rhs)

                if isinstance(cond, NegBoolView):
                    cond, bool_val = self.solver_var(cond._bv), False
                else:
                    cond, bool_val = self.solver_var(cond), True

                if sub_expr.name in ["==", ">="]:
                    # a -> (b >=c)
                    self._add_xct_reif_right(cond, bool_val, xct_cfvars, xct_rhs)
                if sub_expr.name in ["==", "<="]:
                    # a -> (b =< c), rewrite to a -> (-b >= -c)
                    self._add_xct_reif_right(cond, bool_val, [(-x,y) for x,y in xct_cfvars], -xct_rhs)

            # True or False
            elif isinstance(cpm_expr, BoolVal):
                self._add_xct_constr([], True, 0 if cpm_expr.args[0] else 1, False, 0)

            # a direct constraint, pass to solver
            elif isinstance(cpm_expr, DirectConstraint):
                cpm_expr.callSolver(self, self.xct_solver)
                return self

            else:
                raise NotImplementedError(cpm_expr)  # if you reach this... please report on github
            
        return self
    __add__ = add  # avoid redirect in superclass

    def get_core(self):
        from exact import Exact as xct
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """
        assert self.assumption_dict is not None,  "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"

        # return cpm_variables corresponding to Exact core
        return [self.assumption_dict[i][1] for i in self.xct_solver.getLastCore()]


    def solution_hint(self, cpm_vars:List[_NumVarImpl], vals:List[int|bool]):
        """
        Exact supports warmstarting the solver with a partial feasible assignment.

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        # clear previous solution hints
        self.xct_solver.clearSolutionHints(self.solver_vars(list(self.user_vars)))

        cpm_vars = flatlist(cpm_vars)
        vals = flatlist(vals)
        assert (len(cpm_vars) == len(vals)), "Variables and values must have the same size for hinting"
        self.xct_solver.setSolutionHints(list(zip(self.solver_vars(cpm_vars), vals)))
